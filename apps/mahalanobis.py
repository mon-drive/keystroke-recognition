import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

class MahalanobisDetector:
    """
    A full Mahalanobis distance-based keystroke authentication model.
    Uses multiple features (H, UD, DD) for measuring typing similarity.
    """

    def __init__(self, delta=1.0, s_thresh=0.3, train_ratio=0.7, valid_ratio=0.3):
        self.delta = delta  # Mahalanobis distance threshold
        self.s_thresh = s_thresh  # Similarity threshold
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

        self.users = []
        self.user_digraphs_train = {}
        self.user_digraphs_valid = {}
        self.user_digraphs_test = {}

        # Store mean vectors and covariance matrices
        self.user_params = {}  # { user -> { digraph -> (mean_vector, covariance_inv) } }

    def load_csv(self, csv_path):
        """
        Loads CSV and splits data into train/valid/test sets per user.
        Uses multiple features (H, UD, DD) per key/digraph.
        """
        df = pd.read_csv(csv_path).dropna(subset=["H", "UD", "DD"])

        self.users = sorted(df["subject"].unique().tolist())

        for user_id in self.users:
            df_user = df[df["subject"] == user_id].sample(frac=1.0, random_state=42).reset_index(drop=True)

            n_total = len(df_user)
            n_train = int(self.train_ratio * n_total)
            n_valid = int(self.valid_ratio * n_total)

            df_train = df_user.iloc[:n_train]
            df_valid = df_user.iloc[n_train: n_train + n_valid]
            df_test = df_user.iloc[n_train + n_valid:]

            self.user_digraphs_train[user_id] = self._subset_to_digraphs(df_train)
            self.user_digraphs_valid[user_id] = self._subset_to_digraphs(df_valid)
            self.user_digraphs_test[user_id] = self._subset_to_digraphs(df_test)

    def _subset_to_digraphs(self, df_subset):
        """
        Converts a subset of user data into a dictionary:
          { digraph_string -> np.array([[H, UD, DD], ...]) }
        """
        digraph_dict = {}

        for _, row in df_subset.iterrows():
            digraph_str = str(row["key"])
            feature_vector = np.array([row["H"], row["UD"], row["DD"]], dtype=float)

            if digraph_str not in digraph_dict:
                digraph_dict[digraph_str] = []
            digraph_dict[digraph_str].append(feature_vector)

        # Convert lists to numpy arrays
        for dg in digraph_dict:
            digraph_dict[dg] = np.array(digraph_dict[dg])

        return digraph_dict

    def fit(self):
        """
        Compute mean vector and inverse covariance matrix for each user and digraph.
        Handles singular matrices by adding regularization.
        """
        for user_id in self.users:
            self.user_params[user_id] = {}
            train_dict = self.user_digraphs_train[user_id]

            for digraph_str, times_matrix in train_dict.items():
                if times_matrix.shape[0] < 5:
                    continue  # Need at least 2 samples for covariance calculation

                mean_vector = np.mean(times_matrix, axis=0)
                covariance_matrix = np.cov(times_matrix, rowvar=False)

                # Regularization to ensure invertibility
                covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-5

                cov_inv = np.linalg.pinv(covariance_matrix)

                self.user_params[user_id][digraph_str] = (mean_vector, cov_inv)

    def _compute_similarity(self, query_user_id, claimed_user_id, split="valid"):
        """
        Computes the similarity score using full Mahalanobis distance.
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

        for digraph_str, times_matrix in query_dict.items():
            if digraph_str not in claimed_params:
                continue  # No distribution to compare against

            mean_vector, cov_inv = claimed_params[digraph_str]

            for feature_vector in times_matrix:
                total_count += 1
                # Compute Mahalanobis distance
                dist = mahalanobis(feature_vector, mean_vector, cov_inv)

                if dist <= self.delta:
                    accepted_count += 1

        epsilon = 1e-5  # Small smoothing factor to avoid overconfidence
        return (accepted_count + epsilon) / (total_count + epsilon)

    def calculate_scores(self, split="valid"):
        """
        Computes similarity scores for all user pairs.
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

    def compute_FAR(self, imposter_scores, threshold):
        return sum(score >= threshold for score in imposter_scores) / float(len(imposter_scores)) if imposter_scores else 0.0

    def compute_FRR(self, genuine_scores, threshold):
        return sum(score < threshold for score in genuine_scores) / float(len(genuine_scores)) if genuine_scores else 0.0


################################################################################
# **Training with Multiple Features**
################################################################################

def train_mahalanobis_model(csv_path, delta=1.0, train_ratio=0.7, valid_ratio=0.3):
    """
    Trains a Mahalanobis-based keystroke authentication model using multiple features.
    Returns ROC curve values (FPR, TPR, thresholds).
    """
    model = MahalanobisDetector(delta=delta, train_ratio=train_ratio, valid_ratio=valid_ratio)
    model.load_csv(csv_path)
    model.fit()

    genuine_scores, imposter_scores = model.calculate_scores(split="valid")

    thresholds = np.arange(0.0, 1.01, 0.01)
    fpr_list, tpr_list = [], []

    print("Threshold | FAR  | FRR")
    print("----------------------")

    for thr in thresholds:
        FAR = model.compute_FAR(imposter_scores, thr)
        FRR = model.compute_FRR(genuine_scores, thr)

        fpr_list.append(FAR)
        tpr_list.append(1.0 - FRR)

        print(f"{thr:.2f}      | {FAR:.4f} | {FRR:.4f}")

    return np.array(fpr_list), np.array(tpr_list), thresholds
