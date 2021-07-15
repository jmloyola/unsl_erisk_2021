import argparse
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
import numpy as np

from models.earlymodel import EarlyModel, SimpleStopCriterion
from performace_metrics.erde import erde
from performace_metrics.f_latency import f_latency


# Token used to identify the end of each post
END_OF_POST_TOKEN = '$END_OF_POST$'

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
PATH_CORPUS = os.path.join(CURRENT_PATH, '../datasets/interim')


def get_data(users_documents, delay):
    return [END_OF_POST_TOKEN.join(user_posts.split(END_OF_POST_TOKEN)[:delay]) for user_posts in users_documents]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script for the model EarlyModel.")
    parser.add_argument("corpus", help="eRisk task corpus name", choices=['t1', 't2'])
    args = parser.parse_args()

    train_path = os.path.join(PATH_CORPUS, args.corpus, f'{args.corpus}-train-clean.txt')
    test_path = os.path.join(PATH_CORPUS, args.corpus, f'{args.corpus}-test-clean.txt')

    if os.path.exists(train_path):
        print(f'Loading training dataset from {train_path}')
        labels_train = []
        documents_train = []
        with open(train_path, 'r') as f:
            for line in f:
                label, document = line.split(maxsplit=1)
                label = 1 if label == 'positive' else 0
                labels_train.append(label)
                posts = ' '.join(document.split(END_OF_POST_TOKEN))
                documents_train.append(posts)

        countvectorizer_params = {
            'analyzer': 'word',
            'ngram_range': (2, 2),
            'max_df': 1.0,
            'min_df': 1,
        }
        tfidftransformer_params = {
            'norm': 'l2',
            'use_idf': True,
        }
        count_vect = CountVectorizer(**countvectorizer_params)
        tfidf_transformer = TfidfTransformer(**tfidftransformer_params)

        print(f'Training representation')
        print(count_vect.__str__())
        print(tfidf_transformer.__str__())
        x_train_counts = count_vect.fit_transform(documents_train)
        x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
        y_train = np.array(labels_train, dtype=np.float32)

        classifier_params = {
            'C': 2**3,
            'gamma': 'scale',
            'probability': True,
            'class_weight': 'balanced',
        }
        classifier = SVC(**classifier_params)

        print('Training classifier')
        print(classifier.__str__())
        classifier.fit(x_train_tfidf, y_train)

        print('Creating EarlyModel')
        stop_criterion = SimpleStopCriterion(threshold=0.75, min_delay=10)
        earlymodel = EarlyModel(representation_type='bow',
                                trained_representation=(count_vect, tfidf_transformer),
                                trained_classifier=classifier,
                                stop_criterion=stop_criterion)

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
                number_posts = len(document.split(END_OF_POST_TOKEN))
                if number_posts > max_number_posts:
                    max_number_posts = number_posts
        print('Evaluating EarlyModel on the testing dataset')
        for current_delay in range(1, max_number_posts):
            print(f'Current delay: {current_delay}')
            current_posts = get_data(users_documents=documents_test, delay=current_delay)
            decisions, scores = earlymodel.predict(current_posts, current_delay)
        num_positives = sum(labels_test)
        num_negatives = len(labels_test) - num_positives
        c_fp = num_positives / (num_positives + num_negatives)
        erde(labels_list=decisions,
             true_labels_list=labels_test,
             delay_list=earlymodel.delays,
             c_fp=c_fp,
             o=50
         )

    else:
        raise Exception('You need to generate the datasets before. Run the scrips `make_reddit_corpus` '
                        'and `clean_corpus`.')
