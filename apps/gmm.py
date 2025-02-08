import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve

################################################################################
# GMMKeystrokeModel
################################################################################
class GMMKeystrokeModel:
    """
    A snippet-inspired GMM model: 
      - One GMM per (user, digraph).
      - For each user, we store train/valid/test sets split by ratio.
      - We do a threshold search on the validation set, then evaluate on test.

    By default, we interpret 'digraph' very loosely. In the Buffalo data, 
    we have columns: subject, key, H, UD, DD. It's up to you how to define 
    a "digraph string" or "feature dimension" for each row. 
    For the snippet logic, we just need:
      - For each user,
      - For each digraph,
      - A 1D array of timing values to fit a GMM on (like H or DD, etc.).
    """

    def __init__(self, M=3, delta=1.0, s_thresh=0.3, 
                 train_ratio=0.7, valid_ratio=0.3):
        """
        :param M: Number of components in each GMM.
        :param delta: Similarity tolerance parameter in [mean - delta*stdev, mean + delta*stdev].
        :param s_thresh: Initial threshold for accept/reject (will be tuned).
        :param train_ratio: fraction of data for training (per user)
        :param valid_ratio: fraction of data for validation (per user)
        :param test_ratio: fraction of data for testing (per user)
        """
        self.M = M
        self.delta = delta
        self.s_thresh = s_thresh
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

        # We will discover users from CSV automatically
        self.users = []

        # For each user, these dictionaries map { digraph_string -> np.array of times }
        self.user_digraphs_train = {}
        self.user_digraphs_valid = {}
        self.user_digraphs_test  = {}

        # For each user, store { digraph_string -> (means, covars, weights) }
        self.user_gmm_params = {}

    def load_csv(self, csv_path):
        """
        Reads the CSV and splits each user's data into train/valid/test 
        sets by ratio. This mimics the snippet's approach of having 
        separate train/valid/test.

        Expects columns like: [subject, key, H, UD, DD], etc.

        Modify `_make_digraph()` below if you want to define actual two-key combos.
        """
        df = pd.read_csv(csv_path)
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=["H", "UD", "DD"])

        # Identify all users
        self.users = sorted(df["subject"].unique().tolist())

        # For each user, isolate their rows and do a random shuffle + ratio split
        for user_id in self.users:
            df_user = df[df["subject"] == user_id].copy()
            df_user = df_user.sample(frac=1.0, random_state=42).reset_index(drop=True)

            n_total = len(df_user)
            n_train = int(self.train_ratio * n_total)
            n_valid = int(self.valid_ratio * n_total)
            # remainder goes to test

            df_train = df_user.iloc[:n_train]
            df_valid = df_user.iloc[n_train : n_train + n_valid]

            # Convert each subset to { digraph -> array of times }
            self.user_digraphs_train[user_id] = self._subset_to_digraphs(df_train)
            self.user_digraphs_valid[user_id] = self._subset_to_digraphs(df_valid)

    def _subset_to_digraphs(self, df_subset):
        """
        Convert a subset of user data into a dict:
          { digraph_string -> np.array of times }

        Here, we must decide which "time" dimension to use and how to define 
        the "digraph string." The snippet code uses a single dimension for the GMM 
        (like 'delay times'). We'll show an example using the "H" (hold time) only.

        If you truly want a bigram, you'll need to store (key1->key2) for each row. 
        But your CSV doesn't do that by default, so we might just use "key" alone. 
        """

        # For demonstration, let's choose "H" as the timing dimension
        # and define "digraph = row['key']" or something similar.
        # In real usage, you'd want actual 2-key combos.
        digraph_dict = {}

        for idx, row in df_subset.iterrows():
            digraph_str = self._make_digraph(row)
            hold_time   = row["H"]  # pick one dimension, or combine them if you prefer

            if digraph_str not in digraph_dict:
                digraph_dict[digraph_str] = []
            digraph_dict[digraph_str].append(hold_time)

        # Convert lists to numpy arrays
        for dg in digraph_dict:
            digraph_dict[dg] = np.array(digraph_dict[dg], dtype=float)

        return digraph_dict

    def _make_digraph(self, row):
        """
        Generate a 'digraph string' from a single row of the dataset.
        If you want actual bigrams, you'd need your CSV to store (prev_key, current_key).
        Right now, your CSV has columns: subject, key, H, UD, DD.

        We'll just treat the single 'key' as the 'digraph' for demonstration.
        """
        return str(row["key"])

    def fit(self):
        """
        Fits a separate GMM for each (user, digraph) on the training subset.
        """
        for user_id in self.users:
            self.user_gmm_params[user_id] = {}
            train_dict = self.user_digraphs_train[user_id]

            for digraph_str, times_array in train_dict.items():
                # Filter out very small sets 
                if len(times_array) < self.M:
                    continue
                if len(set(times_array)) < self.M:
                    continue

                # Fit a GMM on shape (N, 1)
                X = times_array.reshape(-1, 1)
                gmm = GaussianMixture(n_components=self.M, covariance_type="full", random_state=42)
                gmm.fit(X)

                self.user_gmm_params[user_id][digraph_str] = (
                    gmm.means_,
                    gmm.covariances_,
                    gmm.weights_,
                )

    def calculate_scores(self, split="valid"):
        """
        Calculate the snippet-based similarity scores for every combination of
        query_user vs claimed_user using the specified data split ("valid" or "test").

        Returns:
          genuine_scores   -> list of similarity scores for genuine attempts
          imposter_scores  -> list of similarity scores for impostor attempts
        """
        genuine_scores = []
        imposter_scores = []

        for query_user_id in self.users:
            for claimed_user_id in self.users:
                S = self._compute_similarity(query_user_id, claimed_user_id, split=split)

                if query_user_id == claimed_user_id:
                    genuine_scores.append(S)
                else:
                    imposter_scores.append(S)

        return genuine_scores, imposter_scores

    def _compute_similarity(self, query_user_id, claimed_user_id, split="valid"):
        """
        For the snippet approach (Algorithm 1), we:
          1) Gather all the digraphs and times from the query_user for the specified split.
          2) For each digraph/time, see if claimed_user has a GMM. If not, skip.
          3) For each sample time t, check if it falls in [mean_i - delta*stdev_i, mean_i + delta*stdev_i]
             for any mixture component i. If it does, accumulate that component's weight_i.
          4) Sum over all samples and digraphs, then divide by total sample count to get S.
        """
        # Get the dictionary of query times
        if split == "valid":
            query_dict = self.user_digraphs_valid[query_user_id]

        # Get claimed_user's GMM param dictionary
        claimed_gmms = self.user_gmm_params[claimed_user_id]

        total_count = 0
        similarity_sum = 0.0

        for digraph_str, times_array in query_dict.items():
            if digraph_str not in claimed_gmms:
                # If claimed user doesn't have a GMM for that digraph, skip
                continue

            means, covars, weights = claimed_gmms[digraph_str]
            means   = means.flatten()
            covars  = covars.flatten()
            weights = weights.flatten()

            for t in times_array:
                total_count += 1

                # Check each component
                for i in range(self.M):
                    mean_i = means[i]
                    covar_i = covars[i]  # variance
                    weight_i = weights[i]

                    stdev_i = np.sqrt(covar_i)
                    lower = mean_i - self.delta * stdev_i
                    upper = mean_i + self.delta * stdev_i

                    if lower <= t <= upper:
                        similarity_sum += weight_i
        
        if total_count == 0:
            return 0.0
        return similarity_sum / float(total_count)

    def compute_FAR(self, imposter_scores, threshold):
        """
        Fraction of imposter scores that exceed the threshold (false acceptance).
        """
        if len(imposter_scores) == 0:
            return 0.0
        fa_count = sum(score >= threshold for score in imposter_scores)
        return fa_count / float(len(imposter_scores))

    def compute_FRR(self, genuine_scores, threshold):
        """
        Fraction of genuine scores that fall below the threshold (false rejection).
        """
        if len(genuine_scores) == 0:
            return 0.0
        fr_count = sum(score < threshold for score in genuine_scores)
        return fr_count / float(len(genuine_scores))

################################################################################
# Main convenience function for your route: train_gmm_model
################################################################################
def train_gmm_model(
    csv_path,
    M=3,
    delta=1.0,
    train_ratio=0.7,
    valid_ratio=0.3
):
    """
    Overhauled version of 'train_gmm_model' that uses the snippet-based approach:

    1) Loads all data from 'csv_path' and splits each user's data by (train_ratio, valid_ratio, test_ratio).
    2) Fits a separate GMM for each (user, digraph).
    3) Computes scores on the validation set for threshold tuning.
    4) Builds a full array of (threshold -> FAR, FRR) so we can produce an ROC.

    Returns:
        fpr: array of false positive rates (same as FAR) for candidate thresholds
        tpr: array of true positive rates (1 - FRR) for candidate thresholds
        thresholds: array of threshold values used
    """

    # 1) Instantiate the snippet-based model
    model = GMMKeystrokeModel(
        M=M,
        delta=delta,
        s_thresh=0.3,  # initial guess
        train_ratio=train_ratio,
        valid_ratio=valid_ratio
    )

    # 2) Load CSV into model
    model.load_csv(csv_path=csv_path)

    # 3) Fit GMMs
    model.fit()

    # 4) Calculate genuine & imposter scores on validation
    genuine_scores, imposter_scores = model.calculate_scores(split="valid")

    # 5) Sweep thresholds from 0.0 to 1.0 (since snippet-based S ~ [0..1])
    thresholds = np.arange(0.0, 1.01, 0.01)
    fpr_list = []
    tpr_list = []

    print("Threshold | FAR  | FRR")
    print("----------------------")

    for thr in thresholds:
        FAR = model.compute_FAR(imposter_scores, thr)  # false accept rate
        FRR = model.compute_FRR(genuine_scores, thr)   # false reject rate

        fpr_list.append(FAR)
        tpr_list.append(1.0 - FRR)

        print(f"{thr:.2f}      | {FAR:.4f} | {FRR:.4f}")

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)

    # NOTE: If you want to finalize your threshold and test, you can do:
    #
    #   # pick best threshold to minimize FAR+FRR on valid
    #   best_thr = thresholds[np.argmin(fpr + (1 - tpr))]
    #   model.s_thresh = best_thr
    #
    #   # measure test scores
    #   genuine_scores_test, imposter_scores_test = model.calculate_scores(split="test")
    #   final_FAR = model.compute_FAR(imposter_scores_test, best_thr)
    #   final_FRR = model.compute_FRR(genuine_scores_test, best_thr)
    #
    # But the snippet approach typically returns FPR/TPR arrays so
    # you can do EER or AUC, just like your original code.

    return fpr, tpr, thresholds
