import pyss3
import json


class SS3:
    """SS3 based model for early classification.

    Parameters
    ----------
    ss3_model : pyss3.SS3
        Trained SS3 model.
    policy_value : float
        Policy value (gamma) that affects the final decision for each
        user. Note that, if the final score of a user is greater than
        median(scores) + policy_value · MAD(scores) then the user is
        flag as positive, otherwise is flag as negative.

    References
    ----------
    .. [1] `Burdisso, S. G., Errecalde, M., & Montes-y-Gómez, M. (2019).
        A text classification framework for simple and effective early
        depression detection over social media streams. Expert Systems
        with Applications, 133, 182-197.
        <https://arxiv.org/abs/1905.08772>`_

    .. [2] `pySS3: A Python package implementing a new model for text
        classification with visualization tools for Explainable AI
        <https://github.com/sergioburdisso/pyss3>`_
    """
    __model__ = None
    __policy_value__ = 2

    # state
    __acc_cv__ = None  # []
    delays = None

    def __init__(self, ss3_model, policy_value):
        self.__model__ = ss3_model
        self.__policy_value__ = policy_value

    def __repr__(self):
        smoothness, significance, sanction, alpha = self.__model__.get_hyperparameters()
        str_representation = "SS3:\n" + \
                             f"smoothness = {smoothness} (σ),\n" + \
                             f"significance = {significance} (λ),\n" + \
                             f"sanction = {sanction} (ρ),\n" + \
                             f"alpha = {alpha} (α)\n" + \
                             "- " * 13 + "\n" + \
                             f"policy value = {self.__policy_value__}\n"
        return str_representation

    @staticmethod
    def mad_median(users_scores):
        """Median and median absolute deviation of the scores.

        Parameters
        ----------
        users_scores : list of float
            Users scores.

        Returns
        -------
        m : float
            Median of the users scores.
        sd : float
            Median absolute deviation of the scores.
        """
        users_scores = sorted(users_scores)[::-1]
        n = len(users_scores)
        if n == 2:
            return users_scores[0], users_scores[0]

        values_m = n // 2 if n % 2 else n // 2 - 1
        m = users_scores[values_m]  # Median
        diffs = sorted([abs(m - v) for v in users_scores])
        sd = diffs[len(diffs) // 2]  # sd Mean
        return m, sd

    @staticmethod
    def load(state_path, model_folder_path, model_name, policy_value):
        """Load SS3 model.

        Parameters
        ----------
        state_path : str
            Path to the file containing the state of the SS3 model.
        model_folder_path : str
            Path to load the model from. Note that, by default, pyss3
            assumes the model checkpoint is placed in a folder named
            "ss3_models". Thus, `model_folder_path` should point to the
            parent folder of the "ss3_models" directory.
        model_name : str
            Name of the model to load.
        policy_value : float
            Policy value (gamma) that affects the final decision for each
            user. Note that, if the final score of a user is greater than
            median(scores) + policy_value · MAD(scores) then the user is
            flag as positive, otherwise is flag as negative.

        Returns
        -------
        model : pyss3.SS3
            SS3 loaded model.
        """
        # load model
        clf = pyss3.SS3(name=model_name)
        clf.load_model(model_folder_path)
        model = SS3(clf, policy_value)

        # load state
        with open(state_path, "r") as fp:
            state = json.load(fp=fp)
        model.__acc_cv__ = state['acc_cv']
        model.delays = state['delays']

        return model

    def save(self, state_path):
        """Save SS3 model's state to disk.

        Parameters
        ----------
        state_path : str
            Path to save the state of the model.
        """
        state = {'acc_cv': self.__acc_cv__,
                 'delays': self.delays
                 }
        with open(state_path, "w") as fp:
            json.dump(fp=fp, obj=state, indent='\t')

    def predict(self, users_post, delay):
        """Predict the class for the current users' posts.

        Parameters
        ----------
        users_post : list of str
            List of users' posts.
        delay : int
            Current delay, i.e., post number being processed.

        Returns
        -------
        decisions : list of int
            List of predicted class for each user. The value 1 indicates
            that the user is at-risk, while 0 indicates the user is not
            at-risk.
        scores : list of float
            List of scores for each user. The score represents the
            estimated level of risk of a user.
        """
        n_users = len(users_post)

        # if these are the first posts, then
        # initialize the internal state
        if delay - 1 == 0:
            self.__acc_cv__ = [None] * n_users
            self.delays = [-1] * n_users

        clf = self.__model__
        for i, post in enumerate(users_post):
            if not post:
                continue

            cv = clf.classify(post, sort=False)
            accumulated_cv = self.__acc_cv__[i]

            if accumulated_cv is None:
                self.__acc_cv__[i] = list(cv)
            else:
                self.__acc_cv__[i] = list(map(float.__add__,
                                              map(float, accumulated_cv),
                                              map(float, cv)))

        pos_i = clf.get_category_index('positive')
        neg_i = clf.get_category_index('negative')

        acc_cv = self.__acc_cv__
        scores = [
            acc_cv[i][pos_i] - acc_cv[i][neg_i]
            for i in range(len(acc_cv))
        ]

        # get median and mad from the scores (ranking)
        m, mad = SS3.mad_median(scores)

        decisions = [0] * len(scores)
        mad_threshold = self.__policy_value__
        for i in range(len(scores)):
            user_at_risk = (scores[i] > m + mad_threshold * mad) and delay > 2
            if user_at_risk:
                decisions[i] = 1
                self.delays[i] = delay

        return decisions, scores

    def clear_model_state(self):
        """Clear the internal state of the model.

        Use this function if loading a pre-trained SS3 model for the
        first time.
        """
        self.__acc_cv__ = None
        self.delays = None
