import json
import os
import copy
import numpy as np
import gensim
import torch.nn as nn
from torch.distributions import Bernoulli
from torch.utils.data import TensorDataset, DataLoader
import torch

from models.representation import get_doc2vec_representation


# Token used to identify the end of each post
END_OF_POST_TOKEN = '$END_OF_POST$'


class BaselineNetwork(nn.Module):
    """Baseline network.

    A network which predicts the average reward observed
    during a markov decision-making process.
    Weights are updated w.r.t. the mean squared error between
    its prediction and the observed reward.

    Parameters
    ----------
    input_size : int
        Size of the input layer.
    output_size : int
        Size of the output layer.
    """

    def __init__(self, input_size, output_size):
        super(BaselineNetwork, self).__init__()

        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        b = self.fc(x.detach())
        return b


class Controller(nn.Module):
    """Controller network.

    A network that chooses whether or not enough information
    has been seen to predict a label.

    Parameters
    ----------
    input_size : int
        Size of the input layer.
    output_size : int
        Size of the output layer.
    """

    def __init__(self, input_size, output_size):
        super(Controller, self).__init__()

        self.fc = nn.Linear(input_size, output_size)  # Optimized w.r.t. reward
        self.register_buffer('_epsilon', torch.randn(1))

    def forward(self, x):
        stopping_probability = torch.sigmoid(self.fc(x))
        stopping_probability = stopping_probability.squeeze()
        aux = self._epsilon.new([0.05])
        # Explore/exploit depending on the value of epsilon
        stopping_probability = (1 - self._epsilon) * stopping_probability + self._epsilon * aux

        m = Bernoulli(probs=stopping_probability)
        action = m.sample()  # sample an action
        log_pi = m.log_prob(action)  # compute log probability of sampled action
        return action, log_pi, -torch.log(stopping_probability)


class EARLIEST(nn.Module):
    """Early and Adaptive Recurrent Label ESTimator (EARLIEST).

    Code adapted from https://github.com/Thartvigsen/EARLIEST
    to work with text data.

    Parameters
    ----------
    n_inputs : int
        Number of features in the input data.
    n_classes : int, default=1
        Number of classes in the input labels.
    n_hidden : int, default=50
        Number of dimensions in the RNN's hidden states.
    n_layers : int, default=1
        Number of layers in the RNN.
    lam : float, default=0.0001
        Earliness weight, i.e., emphasis on earliness.
    num_epochs : int, default=100
        Number of epochs use during training.
    drop_p : float, default=0
        Dropout probability. If non-zero, introduces a dropout layer on
        the outputs of each LSTM layer except the last layer.
    weights : torch.Tensor
        Weights for each class.
    device : str, default='cpu'
        Device to run the model from.
    max_sequence_length : int, default=200
        Maximum number of posts to represent per user.
    is_competition : bool, default=True
        Flag to indicate if run in the challenge.
    representation : gensim.models.doc2vec.Doc2Vec, default=None
        Trained representation to use for each post.
    representation_path : str, default=None
        Path to the trained representation.
    """

    def __init__(self, n_inputs, n_classes=1, n_hidden=50, n_layers=1, lam=0.0001, num_epochs=100,
                 drop_p=0, weights=None, device='cpu', max_sequence_length=200, is_competition=True,
                 representation=None, representation_path=None):
        super(EARLIEST, self).__init__()

        # --- Hyper-parameters ---
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.drop_p = drop_p
        self.device = device

        self.is_competition = is_competition

        if representation is None and representation_path is None:
            raise Exception("You should provide the trained representation or the path to it.")
        self.representation = representation
        self.representation_path = representation_path
        self.last_docs_rep = None
        self.last_num_posts_processed = None
        self.num_post_processed = 0
        self.max_sequence_length = max_sequence_length
        self.last_idx_non_stopped_doc = []

        self.predictions = None
        self.probabilities = None
        self.delays = None
        self.already_finished = None

        self.register_buffer('lam', torch.tensor([lam]))
        self.register_buffer('time', torch.tensor([1.], dtype=torch.float))
        self.register_buffer('_exponentials', self.exponential_decay(num_epochs))
        self.register_buffer('weights', weights)

        # --- Sub-networks ---
        self.Controller = Controller(n_hidden + 1, 1)
        self.BaselineNetwork = BaselineNetwork(n_hidden + 1, 1)
        self.RNN = nn.LSTM(input_size=n_inputs, hidden_size=n_hidden, num_layers=n_layers, dropout=drop_p)
        self.out = nn.Linear(n_hidden, n_classes)

    @staticmethod
    def exponential_decay(n):
        """Calculate samples from the exponential decay.

        Parameters
        ----------
        n : int
            Number of samples to take.

        Returns
        -------
        y : torch.Tensor
            Tensor with `n` samples of the exponential decay.
        """
        tau = 1
        tmax = 7
        t = np.linspace(0, tmax, n)
        y = torch.tensor(np.exp(-t / tau), dtype=torch.float)
        return y

    def save(self, path_json):
        """Save the EARLIEST's state and learnable parameters to disk.

        Also save the trained representation in the same folder.

        Parameters
        ----------
        path_json : str
            Path to save the state of the model.

        Notes
        -----
        The learnable parameters are stored in a file with the same name
        as `path_json` but with extension `.pt`.
        """
        base_path = os.path.dirname(os.path.abspath(path_json))
        file_name = os.path.basename(os.path.abspath(path_json)).split('.')[0]

        if self.representation_path is None:
            self.representation_path = os.path.join(base_path, file_name + '_trained_representation.pkl')
        self.representation.save(self.representation_path)

        state_dict_path = os.path.join(base_path, file_name + '.pt')
        model_params = {
            'n_inputs': self.n_inputs,
            'n_classes': self.n_classes,
            'n_hidden': self.n_hidden,
            'n_layers': self.n_layers,
            'drop_p': self.drop_p,
            'device': self.device,
            'representation_path': self.representation_path,
            'max_sequence_length': self.max_sequence_length,
            'is_competition': self.is_competition,
        }
        model_information = {
            'model_params': model_params,
            'state_dict_path': state_dict_path,
        }
        # To be able to resume processing the input during the competition
        # after an interruption, we save the internal state of the model.
        if self.is_competition:
            model_information.update({
                # If loading a pre-trained model for the first time, these
                # models should be cleared before using the model. For
                # that use the method `clear_model_state()`.
                'last_docs_rep': self.last_docs_rep.tolist(),
                'last_num_posts_processed': self.last_num_posts_processed,
                'num_post_processed': self.num_post_processed,
                'last_idx_non_stopped_doc': self.last_idx_non_stopped_doc,
                'predictions': self.predictions.tolist(),
                'probabilities': self.probabilities.tolist(),
                'delays': self.delays.tolist(),
                'already_finished': self.already_finished.tolist(),
            })
        with open(path_json, "w") as fp:
            json.dump(fp=fp, obj=model_information, indent='\t')

        if not self.is_competition:
            # Since this version of PyTorch does not implement yet the
            # `persistent = False` parameter for the register buffers,
            # before saving the model we must remove these from the
            # stated_dict.
            del self.Controller._epsilon
            del self.lam
            del self.time
            del self._exponentials
            del self.weights
            torch.save(self.state_dict(), state_dict_path)
        else:
            # During the competition, we are interested in keeping these register buffers.
            # Thus, we make a copy of the model from which we remove them, before saving.
            new_model = copy.deepcopy(self)
            del new_model.Controller._epsilon
            del new_model.lam
            del new_model.time
            del new_model._exponentials
            del new_model.weights

            torch.save(new_model.state_dict(), state_dict_path)

            del new_model

    @staticmethod
    def load(path_json, for_competition=False):
        """Load EARLIEST model.

        Parameters
        ----------
        path_json : str
            Path to the file containing the state of the EARLIEST model.
        for_competition : bool
            Flag to indicate if it is for competition or not.

        Returns
        --------
        earliest_model : EARLIEST
            The loaded EARLIEST model.
        """
        with open(path_json, "r") as fp:
            model_information = json.load(fp=fp)
        model_params = model_information['model_params']
        earliest_model = EARLIEST(**model_params)
        state_dict_path = model_information['state_dict_path']
        earliest_model.load_state_dict(torch.load(state_dict_path, map_location=earliest_model.device), strict=False)
        if earliest_model.device == 'cuda':
            earliest_model.to(earliest_model.device)

        if earliest_model.is_competition:
            earliest_model.num_post_processed = model_information['num_post_processed']
            earliest_model.last_num_posts_processed = model_information['last_num_posts_processed']
            earliest_model.last_idx_non_stopped_doc = model_information['last_idx_non_stopped_doc']
            earliest_model.last_docs_rep = np.array(model_information['last_docs_rep'])
            earliest_model.delays = np.array(model_information['delays'])
            earliest_model.already_finished = np.array(model_information['already_finished'])
            earliest_model.predictions = np.array(model_information['predictions'])
            earliest_model.probabilities = np.array(model_information['probabilities'])
        if for_competition:
            earliest_model.is_competition = True

        return earliest_model

    def clear_model_state(self):
        """Clear the internal state of the model.

        Use this function if loading a pre-trained EARLIEST model for the
        first time.
        """
        self.last_docs_rep = None
        self.last_num_posts_processed = None
        self.num_post_processed = 0
        self.last_idx_non_stopped_doc = []
        self.predictions = None
        self.probabilities = None
        self.delays = None
        self.already_finished = None

    def predict_probability(self, loader):
        """Predict probability for partial documents.

        Parameters
        ----------
        loader : DataLoader
            DataLoader with all the partial documents.

        Returns
        -------
        torch.Tensor
            Predictions for each document.
        torch.Tensor
            Probabilities for each document.
        torch.Tensor
            Halting point for each document.
        """
        all_predictions = []
        all_probabilities = []
        all_halting_points = []
        self.eval()
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0]
                inputs = inputs.to(self.device)
                inputs = torch.transpose(inputs, 0, 1)

                logits, halting_points, halting_points_mean = self(inputs, test=True)

                if self.n_classes > 1:
                    probabilities, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)
                else:
                    probabilities = torch.sigmoid(logits)
                    predictions = torch.round(probabilities).int()

                all_predictions.append(predictions.view(-1))
                all_probabilities.append(probabilities.view(-1))
                all_halting_points.append(halting_points.view(-1))
        return torch.cat(all_predictions), torch.cat(all_probabilities), torch.cat(all_halting_points)

    def get_representation(self, documents, idx_to_remove):
        """Get representation of the documents.

        Parameters
        ----------
        documents : list of str
            Raw documents to get the representation of.
        idx_to_remove : list of int
            Indexes of documents already finished.

        Returns
        -------
        last_docs_rep : numpy.ndarray
            Representation of the documents.
        """
        if self.representation is None:
            self.representation = gensim.models.doc2vec.Doc2Vec.load(self.representation_path)
        if self.last_docs_rep is None:
            max_num_post = max([len(posts.split(END_OF_POST_TOKEN)) for posts in documents])
            self.last_docs_rep = get_doc2vec_representation(documents=documents, doc2vec_model=self.representation,
                                                            sequential=True, is_competition=True)
            self.last_num_posts_processed = max_num_post
        else:
            replace_old_rep = False
            max_num_post = max([len(posts.split(END_OF_POST_TOKEN)) for posts in documents])
            if max_num_post - self.last_num_posts_processed > self.max_sequence_length:
                initial_subset_docs_idx = max_num_post - self.max_sequence_length
                replace_old_rep = True
            else:
                initial_subset_docs_idx = self.last_num_posts_processed
            current_num_posts = max_num_post - initial_subset_docs_idx
            current_documents = [END_OF_POST_TOKEN.join(
                posts.split(END_OF_POST_TOKEN)[initial_subset_docs_idx:max_num_post])
                for posts in documents]
            current_rep = get_doc2vec_representation(documents=current_documents, doc2vec_model=self.representation,
                                                     sequential=True, is_competition=True)
            old_seq_length = self.last_docs_rep.shape[1]
            current_seq_length = current_rep.shape[1]
            assert current_num_posts == current_seq_length

            if replace_old_rep:
                self.last_docs_rep = current_rep
            else:
                if idx_to_remove:
                    current_num_doc = self.last_docs_rep.shape[0]
                    mask = np.ones(current_num_doc, dtype=bool)
                    remap_idx_to_remove = [i for i, v in enumerate(self.last_idx_non_stopped_doc)
                                           if v in idx_to_remove]
                    mask[remap_idx_to_remove] = False
                    self.last_docs_rep = self.last_docs_rep[mask, :, :]
                overflow_num = old_seq_length + current_seq_length - self.max_sequence_length
                if overflow_num > 0:
                    self.last_docs_rep = self.last_docs_rep[:, overflow_num:, :]
                self.last_docs_rep = np.concatenate((self.last_docs_rep, current_rep), axis=1)
            self.last_num_posts_processed = max_num_post
        assert self.last_docs_rep.shape[1] <= self.max_sequence_length
        return self.last_docs_rep

    def predict(self, documents_test, delay):
        """Predict the class for the current users' posts.

        Parameters
        ----------
        documents_test : list of str
            List of users' posts.
        delay : int
            Current delay, i.e., post number being processed.

        Returns
        -------
        decisions : numpy.ndarray
            List of predicted class for each user. The value 1 indicates
            that the user is at-risk, while 0 indicates the user is not
            at-risk.
        scores : numpy.ndarray
            List of scores for each user. The score represents the
            estimated level of risk of a user.
        """
        # The first time initialize auxiliary variables.
        if self.predictions is None:
            self.predictions = np.array([-1] * len(documents_test))
            self.probabilities = -np.ones_like(self.predictions, dtype=float)
            self.delays = -np.ones_like(self.predictions)
            self.already_finished = np.zeros_like(self.predictions)

        # Flag documents that have already ended.
        cant_posts_docs = [len(doc.split(END_OF_POST_TOKEN)) for doc in documents_test]
        for j, num_posts in enumerate(cant_posts_docs):
            if num_posts < delay:
                self.already_finished[j] = 1
                if self.delays[j] == -1:
                    self.delays[j] = delay - 1

        # For the challenge, we keep processing the posts to notify their scores.
        idx_non_stopped_doc = [j for j, has_finished in enumerate(self.already_finished) if not has_finished]
        if self.last_idx_non_stopped_doc:
            idx_newly_stopped_docs = set(self.last_idx_non_stopped_doc) - set(idx_non_stopped_doc)
            idx_newly_stopped_docs = list(idx_newly_stopped_docs)
        else:
            idx_newly_stopped_docs = []

        if len(idx_non_stopped_doc) > 0:
            documents_not_finished = [documents_test[j] for j in idx_non_stopped_doc]

            x_test = self.get_representation(documents_not_finished, idx_newly_stopped_docs)
            self.last_idx_non_stopped_doc = idx_non_stopped_doc

            test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float))
            test_loader = DataLoader(test_data, batch_size=20, shuffle=False)

            y_predicted, probabilities, halting_points = self.predict_probability(loader=test_loader)

            y_predicted = y_predicted.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
            halting_points = halting_points.cpu().numpy()

            self.predictions[idx_non_stopped_doc] = y_predicted
            self.probabilities[idx_non_stopped_doc] = probabilities

            for j, idx in enumerate(idx_non_stopped_doc):
                if halting_points[j] != -1 and self.delays[idx] == -1:
                    self.delays[idx] = delay
            self.num_post_processed += 1

        decisions = [self.predictions[j] if self.delays[j] != -1 else 0 for j in range(len(self.delays))]
        scores = self.probabilities
        return decisions, scores

    def forward(self, x, epoch=0, test=False):
        """Compute halting points and predictions."""
        if test:
            # No random decisions during testing.
            self.Controller._epsilon = x.new_zeros(1).float()
        else:
            self.Controller._epsilon = self._exponentials[epoch]  # Explore/exploit

        T, B, V = x.shape

        baselines = []  # Predicted baselines
        actions = []  # Which classes to halt at each step
        log_pi = []  # Log probability of chosen actions
        halt_probs = []
        hidden = self.init_hidden(B)
        halt_points = -x.new_ones(B).float()
        predictions = x.new_zeros((B, self.n_classes), requires_grad=True).float()

        logits = 0.

        # --- for each time-step, select a set of actions ---
        for t in range(T):
            # run Base RNN on new data at step t
            rnn_input = x[t].unsqueeze(0)

            output, hidden = self.RNN(rnn_input, hidden)

            # predict logits for all elements in the batch
            logits = self.out(output)

            # compute halting probability and sample an action
            self.time = self.time.new([t]).view(1, 1, 1).repeat(1, B, 1)

            c_in = torch.cat((output, self.time), dim=2).detach()
            a_t, p_t, w_t = self.Controller(c_in)

            a_t_new_dimension = a_t.unsqueeze(1)

            # If a_t == 1 and this class hasn't been halted, save its logits
            predictions = torch.where((a_t_new_dimension == 1) & (predictions == 0), logits, predictions)

            # If a_t == 1 and this class hasn't been halted, save the time
            halt_points = torch.where((halt_points == -1) & (a_t == 1), self.time.squeeze(), halt_points)

            # compute baseline
            b_t = self.BaselineNetwork(torch.cat((output, self.time), dim=2).detach())

            actions.append(a_t)
            baselines.append(b_t.squeeze())
            log_pi.append(p_t)
            halt_probs.append(w_t)
            if (halt_points == -1).sum() == 0:  # If no negative values, every input has been halted
                break

        # If one element in the batch has not been halting, use its final prediction
        logits = torch.where(predictions == 0.0, logits, predictions).squeeze(0)
        if not self.is_competition:
            halt_points = torch.where(halt_points == -1, self.time.squeeze(), halt_points)

        self.baselines = torch.stack(baselines)
        self.log_pi = torch.stack(log_pi)
        self.halt_probs = torch.stack(halt_probs)
        self.actions = torch.stack(actions)

        # --- Compute mask for where actions are updated ---
        # this lets us batch the algorithm and just set the rewards to 0
        # when the method has already halted one instances but not another.
        self.grad_mask = torch.zeros_like(self.actions)
        for b in range(B):
            self.grad_mask[:(1 + halt_points[b]).long(), b] = 1
        return logits, halt_points + 1, (1 + halt_points).mean() / (T + 1)

    def init_hidden(self, bsz):
        """Initialize hidden states."""
        h = (self.lam.new_zeros(self.n_layers,
                                bsz,
                                self.n_hidden),
             self.lam.new_zeros(self.n_layers,
                                bsz,
                                self.n_hidden))
        return h

    def compute_loss(self, logits, y):
        """Compute loss."""
        # --- compute reward ---
        if self.n_classes > 1:
            _, y_hat = torch.max(torch.softmax(logits, dim=1), dim=1)
        else:
            y_hat = torch.round(torch.sigmoid(logits))
            y = y.float()
        self.r = (2 * (y_hat.float().round() == y.float()).float() - 1).detach()
        self.grad_mask = self.grad_mask.squeeze(1)
        self.R = self.r * self.grad_mask

        # --- rescale reward with baseline ---
        b = self.grad_mask * self.baselines
        self.adjusted_reward = self.R - b.detach()

        # --- compute losses ---
        mse_loss_function = nn.MSELoss()
        if self.n_classes == 1:
            ce_loss_function = nn.BCEWithLogitsLoss(pos_weight=self.weights)
        else:
            ce_loss_function = nn.CrossEntropyLoss(weight=self.weights)
        self.loss_b = mse_loss_function(b, self.R)  # Baseline should approximate mean reward
        self.loss_r = (-self.log_pi * self.adjusted_reward).sum() / self.log_pi.shape[0]  # RL loss
        self.loss_c = ce_loss_function(logits, y)  # Classification loss
        self.wait_penalty = self.halt_probs.sum(1).mean()  # Penalize late predictions

        loss = self.loss_r + self.loss_b + self.loss_c + self.lam * self.wait_penalty
        return loss


def train_earliest_model(earliest_model, earliest_optimizer, earliest_scheduler, loader, current_epoch, device,
                         num_epochs):
    """Train EARLIEST model.

    Parameters
    ----------
    earliest_model : EARLIEST
        The EARLIEST model to train.
    earliest_optimizer : torch.optim.Optimizer
        The optimizer to use.
    earliest_scheduler : torch.optim._LRScheduler
        The scheduler for the learning rate to use.
    loader : torch.utils.data.DataLoader
        The DataLoader with the training data.
    current_epoch : int
        The current epoch of training.
    device : str
        The device to run the model from.
    num_epochs : int
        The total number of epoch to train.

    Returns
    -------
    torch.Tensor
        The training delays
    float
        The sum of the loss.
    """
    training_delays = []
    _loss_sum = 0.0

    earliest_model.train()
    for i, batch in enumerate(loader):
        y, x = batch
        x, y = x.to(device), y.to(device)

        x = torch.transpose(x, 0, 1)
        # --- Forward pass ---
        logits, halting_points, halting_points_mean = earliest_model(x, current_epoch)

        training_delays.append(halting_points)

        # --- Compute gradients and update weights ---
        earliest_optimizer.zero_grad()
        loss = earliest_model.compute_loss(logits.squeeze(1), y)
        loss.backward()

        _loss_sum += loss.item()

        earliest_optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(current_epoch + 1, num_epochs, i + 1,
                                                                      len(loader),
                                                                      loss.item()))
    earliest_scheduler.step()

    return torch.cat(training_delays), _loss_sum


def validate_earliest_model(earliest_model, loader, device):
    """Validate EARLIEST model.

        Parameters
        ----------
        earliest_model : EARLIEST
            The EARLIEST model to train.
        loader : torch.utils.data.DataLoader
            The DataLoader with the validation data.
        device : str
            The device to run the model from.

        Returns
        -------
        float
            The sum of the loss.
        """
    _validation_epoch_loss = 0.0

    earliest_model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            y_val, x_val = batch
            x_val, y_val = x_val.to(device), y_val.to(device)

            x_val = torch.transpose(x_val, 0, 1)
            logits_val, halting_points_val, halting_points_val_mean = earliest_model(x_val, test=True)

            loss_val = earliest_model.compute_loss(logits_val.squeeze(1), y_val)
            _validation_epoch_loss += loss_val.item()

    return _validation_epoch_loss
