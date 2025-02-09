import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve

################################################################################
# MahalanobisKeystrokeModel
################################################################################
class MahalanobisDetector:
    """
    A snippet-inspired Mahalanobis distance model:
      - One single-component Gaussian distribution per (user, digraph).
      - For each user, we store train/valid sets split by ratio.
      - We do a threshold search on the validation set, then evaluate on test.

    In 1D, 'Mahalanobis distance' from a single Gaussian is essentially:
        d(t) = |t - mean| / std
    and you can interpret 'delta' as "how many standard deviations away" 
    we allow for acceptance. The resulting similarity score S is the fraction 
    of samples that lie within [mean - delta*stdev, mean + delta*stdev].
    """

    def __init__(
        self, 
        delta=1.0, 
        s_thresh=0.3,
        train_ratio=0.7, 
        valid_ratio=0.3
    ):
        """
        :param delta: Similarity tolerance (in stdev units). 
                      We'll accept a sample t if MahalanobisDistance(t) <= delta.
        :param s_thresh: Initial threshold for accept/reject (will be tuned).
        :param train_ratio: fraction of data for training (per user).
        :param valid_ratio: fraction of data for validation (per user).
                           The remainder is test ratio.
        """
        self.delta = delta
        self.s_thresh = s_thresh
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

        # Discovered users
        self.users = []

        # For each user: { digraph_string -> np.array of times }
        self.user_digraphs_train = {}
        self.user_digraphs_valid = {}
        self.user_digraphs_test  = {}

        # For each user, store { digraph_string -> (mean, var) }
        self.user_params = {}

    def load_csv(self, csv_path):
        """
        Reads the CSV and splits each user's data into train/valid/test 
        sets by ratio. Expects columns like: [subject, key, H, UD, DD], etc.
        By default, we'll only use 'H' as the 1D timing dimension.
        """
        df = pd.read_csv(csv_path)
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=["H", "UD", "DD"])

        # Identify all users
        self.users = sorted(df["subject"].unique().tolist())

        for user_id in self.users:
            df_user = df[df["subject"] == user_id].copy()
            # Shuffle
            df_user = df_user.sample(frac=1.0, random_state=42).reset_index(drop=True)

            n_total = len(df_user)
            n_train = int(self.train_ratio * n_total)
            n_valid = int(self.valid_ratio * n_total)
            # remainder goes to test

            df_train = df_user.iloc[:n_train]
            df_valid = df_user.iloc[n_train : n_train + n_valid]
            df_test  = df_user.iloc[n_train + n_valid :]

            self.user_digraphs_train[user_id] = self._subset_to_digraphs(df_train)
            self.user_digraphs_valid[user_id] = self._subset_to_digraphs(df_valid)
            self.user_digraphs_test[user_id]  = self._subset_to_digraphs(df_test)

    def _subset_to_digraphs(self, df_subset):
        """
        Convert a subset of user data into a dict:
          { digraph_string -> np.array of times }

        For simplicity, we'll treat each row's 'key' as a separate "digraph,"
        using only the 'H' (hold time) dimension.
        """
        digraph_dict = {}

        for idx, row in df_subset.iterrows():
            digraph_str = self._make_digraph(row)
            hold_time   = row["H"]  # 1D feature

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
        If you want actual bigrams, you need (prev_key, current_key).
        Here, we'll just use the single 'key' for demonstration.
        """
        return str(row["key"])

    def fit(self):
        """
        Compute mean and variance for each (user, digraph) from training data.
        """
        for user_id in self.users:
            self.user_params[user_id] = {}
            train_dict = self.user_digraphs_train[user_id]

            for digraph_str, times_array in train_dict.items():
                # ignore degenerate cases
                if len(times_array) < 2:
                    continue

                mu   = np.mean(times_array)
                var  = np.var(times_array, ddof=1)  # unbiased variance

                # if variance is 0, set something small to avoid div-by-zero
                if var < 1e-9:
                    var = 1e-9

                self.user_params[user_id][digraph_str] = (mu, var)

    def calculate_scores(self, split="valid"):
        """
        Compute similarity scores for all combinations (query_user, claimed_user)
        on the specified split. 
        Returns:
          genuine_scores   -> list of S for genuine attempts
          imposter_scores  -> list of S for impostor attempts
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
        'Snippet-inspired' approach in 1D:
          - For each sample t in query_user's data (on the chosen split),
            we check if t lies within [mean - delta*stdev, mean + delta*stdev]
            (i.e., its Mahalanobis distance <= delta).
          - If yes, we increment a counter.
          - The final score S = (accepted_count) / (total_count).
        """
        if split == "valid":
            query_dict = self.user_digraphs_valid[query_user_id]
        elif split == "test":
            query_dict = self.user_digraphs_test[query_user_id]
        else:
            query_dict = self.user_digraphs_train[query_user_id]

        claimed_params = self.user_params[claimed_user_id]

        total_count = 0
        accepted_count = 0

        for digraph_str, times_array in query_dict.items():
            if digraph_str not in claimed_params:
                # no distribution to compare against
                continue

            mu, var = claimed_params[digraph_str]
            std = np.sqrt(var)

            for t in times_array:
                total_count += 1
                # 1D "Mahalanobis distance" => |t - mu| / std
                dist = abs(t - mu) / std
                if dist <= self.delta:
                    accepted_count += 1

        if total_count == 0:
            return 0.0
        return accepted_count / float(total_count)

    def compute_FAR(self, imposter_scores, threshold):
        """
        Fraction of imposter scores >= threshold (false acceptance).
        """
        if len(imposter_scores) == 0:
            return 0.0
        fa_count = sum(score >= threshold for score in imposter_scores)
        return fa_count / float(len(imposter_scores))

    def compute_FRR(self, genuine_scores, threshold):
        """
        Fraction of genuine scores < threshold (false rejection).
        """
        if len(genuine_scores) == 0:
            return 0.0
        fr_count = sum(score < threshold for score in genuine_scores)
        return fr_count / float(len(genuine_scores))


################################################################################
# Main convenience function: train_mahalanobis_model
################################################################################
def train_mahalanobis_model(
    csv_path,
    delta=1.0,
    train_ratio=0.7,
    valid_ratio=0.3
):
    """
    1) Loads all data from 'csv_path' and splits each user's data by the given ratios.
    2) Fits a single Gaussian (mean,var) for each (user, digraph).
    3) Computes scores on the validation set for threshold tuning.
    4) Sweeps thresholds from 0.0 to 1.0 to produce (FPR, TPR) for ROC/EER analysis.

    Returns:
        fpr: array of false positive rates (FAR)
        tpr: array of true positive rates (1 - FRR)
        thresholds: array of threshold values
    """
    # 1) Instantiate the model
    model = MahalanobisDetector(
        delta=delta,
        s_thresh=0.3,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio
    )

    # 2) Load CSV
    model.load_csv(csv_path=csv_path)

    # 3) Fit distributions
    model.fit()

    # 4) Calculate genuine & imposter scores on validation set
    genuine_scores, imposter_scores = model.calculate_scores(split="valid")

    # 5) Sweep thresholds in [0..1] (scores are fractions of 'accepted' samples)
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

    # Example of how you'd pick an operating threshold:
    # best_thr = thresholds[np.argmin(fpr + (1 - tpr))]
    # model.s_thresh = best_thr
    # # Evaluate on test
    # genuine_scores_test, imposter_scores_test = model.calculate_scores(split="test")
    # final_FAR = model.compute_FAR(imposter_scores_test, best_thr)
    # final_FRR = model.compute_FRR(genuine_scores_test, best_thr)

    return fpr, tpr, thresholds
