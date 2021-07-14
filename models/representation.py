import numpy as np


# Token used to identify the end of each post
END_OF_POST_TOKEN = '$END_OF_POST$'


def get_bow_representation(documents, count_vect, tfidf_transformer):
    """Get BoW representation for the users' documents.

    Parameters
    ----------
    documents : list of str
        List of users' posts. Every element of the list correspond to a
        different user. All the posts and comments from a user are
        contained in a string, separated with `END_OF_POST_TOKEN`.
    count_vect : sklearn.feature_extraction.text.CountVectorizer
        The trained scikit-learn CountVectorizer to use.
    tfidf_transformer : sklearn.feature_extraction.text.TfidfTransformer
        The trained scikit-learn TfidfTransformer to use.

    Returns
    -------
    x_tfidf : sparse matrix
        The tf or tf-idf representation of the users' posts.
    """
    concat_posts = [user_posts.replace(END_OF_POST_TOKEN, ' ') for user_posts in documents]
    x_counts = count_vect.transform(concat_posts)
    x_tfidf = tfidf_transformer.transform(x_counts)
    return x_tfidf


def get_doc2vec_representation(documents, doc2vec_model, sequential=False, max_sequence_length=None,
                               is_competition=False):
    """Get doc2vec representation of documents.

    Parameters
    ----------
    documents : list of str
        List of users' posts. Every element of the list correspond to a
        different user. All the posts and comments from a user are
        contained in a string, separated with `END_OF_POST_TOKEN`.
    doc2vec_model : gensim.models.doc2vec.Doc2Vec
        The trained doc2vec model used to infer the vector representation
        of each post.
    sequential : bool, default=False
        A flag to indicate if the input should be represented as a sequence
        of posts or as one big post for each user. If set to True a
        document embedding will be inferred for each post of a user.
    max_sequence_length : int, default=None
        The maximum sequence length, i.e., the maximum number of posts
        allowed for each user. User to limit the size of the
        representation in memory.
        Used only when `sequential=True` and `is_competition=False`, that
        is, during training for the competition.
    is_competition : bool, default=False
        A flag to indicate if the current representation is to be used
        during the competition or not.

    Returns
    -------
    x_doc2vec : numpy.ndarray
        The doc2vec representation of the users' posts.
    """
    if sequential:
        if is_competition:
            max_num_post = max([len(posts.split(END_OF_POST_TOKEN)) for posts in documents])
            users_posts_truncated = [user_posts.split(END_OF_POST_TOKEN) for user_posts in documents]
            x_doc2vec = np.zeros((len(documents), max_num_post, doc2vec_model.vector_size), dtype=np.float32)
            for j, posts in enumerate(users_posts_truncated):
                for k, current_post in enumerate(posts):
                    if current_post == '':
                        continue
                    x_doc2vec[j, k, :] = doc2vec_model.infer_vector(current_post.split())
            return x_doc2vec
        else:
            assert max_sequence_length is not None
            max_num_post = max([len(posts.split(END_OF_POST_TOKEN)) for posts in documents])
            seq_lim = max_sequence_length if max_sequence_length < max_num_post else max_num_post
            users_posts_truncated = [user_posts.split(END_OF_POST_TOKEN)[:seq_lim] for user_posts in documents]
            x_doc2vec = np.zeros((len(documents), seq_lim, doc2vec_model.vector_size), dtype=np.float32)
            for j, posts in enumerate(users_posts_truncated):
                for k, current_post in enumerate(posts):
                    x_doc2vec[j, k, :] = doc2vec_model.infer_vector(current_post.split())
            return x_doc2vec
    else:
        concat_posts = [user_posts.replace(END_OF_POST_TOKEN, ' ').split() for user_posts in documents]
        x_doc2vec = np.zeros((len(documents), doc2vec_model.vector_size), dtype=np.float32)
        for j, post in enumerate(concat_posts):
            x_doc2vec[j, :] = doc2vec_model.infer_vector(post)
        return x_doc2vec
