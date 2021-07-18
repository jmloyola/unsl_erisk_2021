import argparse
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pyss3
import numpy as np

from models.ss3 import SS3
from performace_metrics.erde import erde
from performace_metrics.f_latency import f_latency


# Token used to identify the end of each post
END_OF_POST_TOKEN = '$END_OF_POST$'

# Paths to the datasets
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_CORPUS = os.path.join(CURRENT_PATH, '../datasets/interim')


def get_data(users_documents, delay):
    """Get documents of the users from certain delay.

    In case the user does not have a post in that delay, the empty post
    is returned.

    Parameters
    ----------
    users_documents : list of str
        The complete users' documents.
    delay : int
        The delay of the documents to fetch.

    Returns
    -------
    list of str
        The list with the users' documents from certain delay.
    """
    documents_current_delay = []
    for user_posts in users_documents:
        splitted_users_posts = user_posts.split(END_OF_POST_TOKEN)
        if len(splitted_users_posts) < delay:
            current_post = ''
        else:
            current_post = splitted_users_posts[delay - 1]
        documents_current_delay.append(current_post)
    return documents_current_delay


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script for the model SS3.")
    parser.add_argument("corpus", help="eRisk task corpus name", choices=['t1', 't2'])
    parser.add_argument("-n", "--num_post_limit", help="Number of post to process. In case of 0, the maximum number of "
                                                       "post in the test dataset will be used.", type=int, default=0)
    args = parser.parse_args()

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
                label, document = line.split(maxsplit=1)
                labels_train.append(label)
                posts = ' '.join(document.split(END_OF_POST_TOKEN))
                documents_train.append(posts)

        print(f'Loading testing dataset from {test_path}')
        labels_test = []
        documents_test = []
        max_number_posts = 0
        with open(test_path, 'r') as f:
            for line in f:
                label, posts = line.split(maxsplit=1)
                labels_test.append(label)
                documents_test.append(posts)
                number_posts = len(document.split(END_OF_POST_TOKEN))
                if number_posts > max_number_posts:
                    max_number_posts = number_posts

        base_ss3_classifier = pyss3.SS3()

        print('Training base SS3 classifier')
        print(f'Base SS3 hyperparameters: {base_ss3_classifier.get_hyperparameters()}')
        base_ss3_classifier.fit(documents_train, labels_train)

        ss3 = SS3(ss3_model=base_ss3_classifier, policy_value=2)

        delay_limit = max_number_posts if (max_number_posts < args.num_post_limit) or (args.num_post_limit == 0) \
            else args.num_post_limit
        print(f'Evaluating SS3 on the testing dataset for {delay_limit} posts')
        for current_delay in range(1, delay_limit):
            print(f'Current delay: {current_delay}')
            current_posts = get_data(users_documents=documents_test, delay=current_delay)
            decisions, scores = ss3.predict(current_posts, current_delay)

        # We fill the delays of documents that have not ended yet.
        delays = np.where(ss3.delays == -1, delay_limit, ss3.delays)

        labels_test = [1 if label == 'positive' else 0 for label in labels_test]
        num_positives = sum(labels_test)
        num_negatives = len(labels_test) - num_positives
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
        print('Saving SS3 model')
        ss3.save('ss3.json')

    else:
        raise Exception('You need to generate the datasets before. Run the scrips `make_reddit_corpus` '
                        'and `clean_corpus`.')
