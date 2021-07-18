import argparse
import os
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gensim
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.representation import get_doc2vec_representation
from models.earliest import EARLIEST, train_earliest_model, validate_earliest_model
from performace_metrics.erde import erde
from performace_metrics.f_latency import f_latency


# Token used to identify the end of each post
END_OF_POST_TOKEN = '$END_OF_POST$'

# Paths to the datasets
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_CORPUS = os.path.join(CURRENT_PATH, '../datasets/interim')


def get_data(users_documents, delay):
    """Get users' documents up until certain delay.

    Parameters
    ----------
    users_documents : list of str
        The complete users' documents.
    delay : int
        The upper limit of posts for each user.

    Returns
    -------
    list of str
        The list with the users' documents up until certain delay.
    """
    return [END_OF_POST_TOKEN.join(_user_posts.split(END_OF_POST_TOKEN)[:delay]) for _user_posts in users_documents]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script for the model EARLIEST.")
    parser.add_argument("corpus", help="eRisk task corpus name", choices=['t1', 't2'])
    parser.add_argument("device", help="Device to use", choices=['cpu', 'cuda'])
    parser.add_argument("-n", "--num_post_limit", help="Number of post to process. In case of 0, the maximum number of "
                                                       "post in the test dataset will be used.", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA drivers not installed. Device fallback to cpu")
        device = 'cpu'

    # Set the random seeds
    random_seed = 30
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if args.num_post_limit < 0:
        raise Exception('The number of post to process should be greater or equal to 0.\n'
                        'In case of 0, the maximum number of post in the test dataset will'
                        ' be used.')

    train_path = os.path.join(PATH_CORPUS, args.corpus, f'{args.corpus}-train-clean.txt')
    test_path = os.path.join(PATH_CORPUS, args.corpus, f'{args.corpus}-test-clean.txt')

    if os.path.exists(train_path):
        print(f'Loading training dataset from {train_path}')
        labels_train = []
        documents_train = []
        with open(train_path, 'r') as f:
            for line in f:
                label, posts = line.split(maxsplit=1)
                label = 1 if label == 'positive' else 0
                labels_train.append(label)
                documents_train.append(posts)
        y_train = np.array(labels_train, dtype=np.float32)

        print(f'Loading testing dataset from {test_path}')
        labels_test = []
        documents_test = []
        max_number_posts = 0
        with open(test_path, 'r') as f:
            for line in f:
                label, posts = line.split(maxsplit=1)
                label = 1 if label == 'positive' else 0
                labels_test.append(label)
                documents_test.append(posts)
                number_posts = len(posts.split(END_OF_POST_TOKEN))
                if number_posts > max_number_posts:
                    max_number_posts = number_posts
        y_test = np.array(labels_test, dtype=np.float32)

        doc2vec_documents = []
        i = 0
        for user_posts in documents_train:
            for post in user_posts.split(END_OF_POST_TOKEN):
                doc2vec_documents.append(gensim.models.doc2vec.TaggedDocument(post.split(), [i]))
                i = i + 1

        vector_size = 200
        doc2vec_params = {
            'dm': 1,
            'vector_size': vector_size,
            'window': 5,
            'seed': 30,
            'workers': 1,
            'epochs': 50,
            'min_count': 10,
        }
        print(f'Training representation')
        doc2vec_model = gensim.models.doc2vec.Doc2Vec(documents=doc2vec_documents, **doc2vec_params)
        print(doc2vec_model.__str__())

        max_sequence_length = 200
        batch_size = 25
        x_train = get_doc2vec_representation(documents=documents_train,
                                             doc2vec_model=doc2vec_model,
                                             sequential=True,
                                             max_sequence_length=max_sequence_length)
        x_test = get_doc2vec_representation(documents=documents_test,
                                            doc2vec_model=doc2vec_model,
                                            sequential=True,
                                            max_sequence_length=max_sequence_length)

        train_data = TensorDataset(torch.tensor(y_train, dtype=torch.long), torch.tensor(x_train, dtype=torch.float))
        test_data = TensorDataset(torch.tensor(y_test, dtype=torch.long), torch.tensor(x_test, dtype=torch.float))

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        num_epochs = 10
        n_classes = 1
        num_positives = sum(labels_train)
        num_negatives = len(labels_train) - num_positives
        w = torch.tensor([num_negatives, num_positives], dtype=torch.float32)
        w = w / w.sum()
        w = 1.0 / w

        weights = w.clone()
        if n_classes == 1:
            weights = weights[1]
        weights.to(device)

        print('Creating EARLIEST model')
        earliest = EARLIEST(n_inputs=vector_size,
                            n_classes=n_classes,
                            n_hidden=256,
                            n_layers=1,
                            lam=0.00001,
                            num_epochs=num_epochs,
                            representation=doc2vec_model,
                            device=device,
                            weights=weights,
                            )

        earliest = earliest.to(device)
        optimizer = torch.optim.Adam(earliest.parameters())
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        model_save_epoch = num_epochs
        training_loss = []
        validation_loss = []
        best_validation_loss = float('inf')
        for epoch in range(num_epochs):
            training_delays, loss_sum = train_earliest_model(earliest_model=earliest,
                                                             earliest_optimizer=optimizer,
                                                             earliest_scheduler=scheduler,
                                                             loader=train_loader,
                                                             current_epoch=epoch,
                                                             device=device,
                                                             num_epochs=num_epochs)
            training_loss.append(np.round(loss_sum / len(train_loader), 3))

            validation_epoch_loss = validate_earliest_model(earliest_model=earliest,
                                                            loader=test_loader,
                                                            device=device)
            validation_loss.append(np.round(validation_epoch_loss / len(test_loader), 3))

            if epoch > 5 and validation_epoch_loss < best_validation_loss:
                best_validation_loss = validation_epoch_loss
                torch.save(earliest.state_dict(), 'earliest_checkpoint.pt')
                model_save_epoch = epoch

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        ax.plot(training_loss, label='Training Loss')
        ax.plot(validation_loss, label='Validation Loss')
        ax.axvline(x=model_save_epoch, color='r', linestyle='--', ymin=0, ymax=3, label='Epoch of saved model')
        ax.legend()
        plt.show()

        # We load the best model
        earliest.load_state_dict(torch.load('earliest_checkpoint.pt'))

        delay_limit = max_number_posts if (max_number_posts < args.num_post_limit) or (args.num_post_limit == 0) \
            else args.num_post_limit

        print(f'Evaluating EARLIEST on the testing dataset')
        for current_delay in range(1, delay_limit):
            print(f'Current delay: {current_delay}')
            current_posts = get_data(users_documents=documents_test, delay=current_delay)
            decisions, scores = earliest.predict(current_posts, current_delay)

        # We fill the delays of documents that have not ended yet.
        delays = np.where(earliest.delays == -1, delay_limit, earliest.delays)

        c_fp = num_positives / (num_positives + num_negatives)
        erde_performance = erde(labels_list=decisions,
                                true_labels_list=labels_test,
                                delay_list=delays,
                                c_fp=c_fp,
                                o=50
                                )
        penalty = 0.0078
        f_latency_performance = f_latency(labels=decisions,
                                          true_labels=labels_test,
                                          delays=delays,
                                          penalty=penalty)
        classification_report_text = classification_report(labels_test, decisions)
        confusion_matrix_text = confusion_matrix(labels_test, decisions)
        accuracy = np.round(accuracy_score(labels_test, decisions), 3)
        print('-' * 75)
        print(f"Accuracy: {accuracy}")
        print(classification_report_text)
        print(confusion_matrix_text)
        print('-' * 75)
        print(f'ERDE 50: {erde_performance}')
        print(f'F Latency: {f_latency_performance}')
        print('-' * 75)
        print('Saving EARLIEST')
        earliest.save('earliest.json')

    else:
        raise Exception('You need to generate the datasets before. Run the scrips `make_reddit_corpus` '
                        'and `clean_corpus`.')
