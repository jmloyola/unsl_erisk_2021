import json
import pickle
import numpy as np
import gensim
import os

from models.representation import get_doc2vec_representation, get_bow_representation


PICKLE_PROTOCOL = 4

# Token used to identify the end of each post
END_OF_POST_TOKEN = '$END_OF_POST$'


class SimpleStopCriterion:
    """Simple decision policy to determine when to send an alarm.

    Parameters
    ----------
    threshold : float
        The probability threshold to consider a user as risky.
    min_delay : int, default=None
        The minimum delay, that is, the minimum number of posts necessary
        to start considering if a user is at-risk or not.
    """
    def __init__(self, threshold, min_delay=None):
        self.threshold = threshold
        self.min_delay = min_delay

    def __repr__(self):
        str_representation = "SimpleStopCriterion:\n" + \
                             f"Threshold = {self.threshold}\n" + \
                             f"Min delay = {self.min_delay}"
        return str_representation

    def __eq__(self, other):
        are_equal = self.__class__ == other.__class__
        if are_equal:
            are_equal = are_equal and self.threshold == other.threshold
            are_equal = are_equal and self.min_delay == other.min_delay
        return are_equal

    def get_parameters(self):
        return {'threshold': self.threshold,
                'min_delay': self.min_delay}

    def decide(self, probs, delay):
        """Decide to issue an alarm or not for each user.

        Parameters
        ----------
        probs : numpy.ndarray
            The probability of belonging to the positive class as
            estimated by the model.
        delay : int
            The current number of post being processed.

        Returns
        -------
        numpy.ndarray
            The decision to issue an alarm or not for every user.
        """
        should_stop_threshold = probs >= self.threshold
        should_stop_min_delay = np.ones_like(should_stop_threshold)
        if self.min_delay is not None and delay <= self.min_delay:
            should_stop_min_delay = np.zeros_like(should_stop_min_delay)
        return should_stop_min_delay & should_stop_threshold


class EarlyModel:
    """Simple model for early classification.

    This model is composed of two parts:
        - a classifier able to categorize partial documents;
        - a rule capable of determining if an alarm should be raised.

    Parameters
    ----------
    representation_type : {'bow', 'doc2vec'}
        The representation type to use.
    trained_representation : {tuple of CountVectorizer, TfidfTransformer, gensim.Doc2Vec}
        The trained representation.
    trained_classifier : sklearn.BaseEstimator
        The trained classifier.
    stop_criterion : SimpleStopCriterion
        The decision policy to determine when to send an alarm.
    """
    def __init__(self, representation_type, trained_representation, trained_classifier, stop_criterion):
        if representation_type not in ['bow', 'doc2vec']:
            raise Exception('Representation not implemented.')
        self.representation_type = representation_type
        self.representation = trained_representation
        self.classifier = trained_classifier

        self.stop_criterion = stop_criterion
        self.predictions = None
        self.probabilities = None
        self.delays = None
        self.already_finished = None
        self.num_post_processed = 0

    def __repr__(self):
        str_representation = "EarlyModel:" + \
                             "\n--- Representation ---\n" + \
                             self.representation.__str__() + \
                             "\n--- Classifier ---\n" + \
                             self.classifier.__repr__() + \
                             "\n--- Decision policy ----\n" + \
                             self.stop_criterion.__repr__()
        return str_representation

    def get_representation(self, documents):
        """Get representation of the documents.

        Parameters
        ----------
        documents : list of str
            Raw documents to get the representation of.

        Returns
        -------
        numpy.ndarray
            Representation of the documents.
        """
        if self.representation_type == 'bow':
            count_vect, tfidf_transformer = self.representation
            return get_bow_representation(documents=documents, count_vect=count_vect,
                                          tfidf_transformer=tfidf_transformer)
        else:
            return get_doc2vec_representation(documents=documents, doc2vec_model=self.representation, sequential=False)

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
        # If this is the first time classifying, initialize internal variables.
        if self.predictions is None:
            self.predictions = np.array([-1] * len(documents_test))
            self.probabilities = -np.ones_like(self.predictions, dtype=float)
            self.delays = -np.ones_like(self.predictions)
            self.already_finished = np.zeros_like(self.predictions)

        # Do not predict documents already finished.
        cant_posts_docs = [len(doc.split(END_OF_POST_TOKEN)) for doc in documents_test]
        for j, num_posts in enumerate(cant_posts_docs):
            if num_posts < delay:
                self.already_finished[j] = 1
                if self.delays[j] == -1:
                    self.delays[j] = delay - 1

        # For the challenge, we keep processing the posts to notify their scores.
        idx_non_stopped_doc = [j for j, has_finished in enumerate(self.already_finished) if not has_finished]

        if len(idx_non_stopped_doc) > 0:
            documents_not_finished = [documents_test[j] for j in idx_non_stopped_doc]

            x_test = self.get_representation(documents_not_finished)

            y_predicted = self.classifier.predict(x_test)
            # Since we are only interested in the positive class probability, we only retrieve the second element
            # of the second dimension.
            probabilities = self.classifier.predict_proba(x_test)[:, 1]

            self.predictions[idx_non_stopped_doc] = y_predicted
            self.probabilities[idx_non_stopped_doc] = probabilities
            stop_reading = self.stop_criterion.decide(probs=probabilities, delay=delay)
            for j, idx in enumerate(idx_non_stopped_doc):
                if stop_reading[j] and self.delays[idx] == -1:
                    self.delays[idx] = delay
            self.num_post_processed += 1

        decisions = [self.predictions[j] if self.delays[j] != -1 else 0 for j in range(len(self.delays))]
        scores = self.probabilities
        return decisions, scores

    def save(self, path_json):
        """Save the information and state of the EarlyModel.

        Also save the trained representation and classifier in the same
        folder.

        Parameters
        ----------
        path_json : str
            Path to save the information and state of the model.
        """
        base_path = os.path.dirname(os.path.abspath(path_json))
        file_name = os.path.basename(os.path.abspath(path_json)).split('.')[0]

        trained_representation_path = os.path.join(base_path, file_name + '_trained_representation.pkl')
        if self.representation_type == 'bow':
            with open(trained_representation_path, 'wb') as fp:
                pickle.dump(self.representation, fp, protocol=PICKLE_PROTOCOL)
        else:
            self.representation.save(trained_representation_path)

        trained_classifier_path = os.path.join(base_path, file_name + '_trained_classifier.pkl')
        with open(trained_classifier_path, 'wb') as fp:
            pickle.dump(self.classifier, fp, protocol=PICKLE_PROTOCOL)

        model_information = {
            'representation_type': self.representation_type,
            'trained_representation_path': trained_representation_path,
            'trained_classifier_path': trained_classifier_path,
            'criterion_params': self.stop_criterion.get_parameters(),
            'num_post_processed': self.num_post_processed,
            'delays': self.delays.tolist(),
            'already_finished': self.already_finished.tolist(),
            'predictions': self.predictions.tolist(),
            'probabilities': self.probabilities.tolist()
        }
        with open(path_json, "w") as fp:
            json.dump(fp=fp, obj=model_information, indent='\t')

    @staticmethod
    def load(path_json):
        """Load EarlyModel model.

        Parameters
        ----------
        path_json : str
            Path to the file containing the state of the EarlyModel.

        Returns
        --------
        early_model : EarlyModel
            The loaded EarlyModel model.
        """
        with open(path_json, "r") as fp:
            model_information = json.load(fp=fp)
        representation_type = model_information['representation_type']
        trained_representation_path = model_information['trained_representation_path']
        trained_classifier_path = model_information['trained_classifier_path']
        # Load trained representation.
        if representation_type == 'bow':
            with open(trained_representation_path, 'rb') as fp:
                trained_representation = pickle.load(fp)
        elif representation_type == 'doc2vec':
            trained_representation = gensim.models.doc2vec.Doc2Vec.load(trained_representation_path)
        else:
            trained_representation = None

        # Load trained classifier.
        with open(trained_classifier_path, 'rb') as fp:
            trained_classifier = pickle.load(fp)

        criterion_params = model_information['criterion_params']
        stop_criterion = SimpleStopCriterion(**criterion_params)
        early_model = EarlyModel(representation_type=representation_type,
                                 trained_representation=trained_representation,
                                 trained_classifier=trained_classifier,
                                 stop_criterion=stop_criterion)
        early_model.num_post_processed = model_information['num_post_processed']
        early_model.delays = np.array(model_information['delays'])
        early_model.already_finished = np.array(model_information['already_finished'])
        early_model.predictions = np.array(model_information['predictions'])
        early_model.probabilities = np.array(model_information['probabilities'])

        return early_model
