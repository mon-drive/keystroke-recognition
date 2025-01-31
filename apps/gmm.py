import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

class GmmDetector:
    def __init__(self, data, n_components=2):
        """
        :param data: Pandas DataFrame containing user data with columns: 
                     [subject, H, DD, UD, ...].
        :param n_components: number of Gaussian mixture components (tunable hyperparam).
        """
        self.data = data
        self.n_components = n_components
        self.user_models = {}  # subject -> trained GMM
        self.user_scores = []
        self.imposter_scores = []

    def train_user_model(self, subject, X_train):
        """
        Train a GMM for a specific user given the training features X_train.
        """
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=42)
        gmm.fit(X_train)
        return gmm

    def evaluate_set(self):
        """
        - For each subject:
          1) Split that subject's data into train/test.
          2) Train the GMM on the train subset (user_model).
          3) Compute log-likelihood for the user's own test data (genuine scores).
          4) Compute log-likelihood for an impostor subset of data from other users.
        - Aggregate all genuine and impostor scores.
        - Compute ROC curve (FPR, TPR, thresholds).
        """
        subjects = self.data['subject'].unique()

        # We'll store all subject-level FPR, TPR for an overall curve
        all_genuine_scores = []
        all_imposter_scores = []

        for subject in subjects:
            # Filter data for this subject
            df_subject = self.data[self.data['subject'] == subject]
            # Filter data for others
            df_impostors = self.data[self.data['subject'] != subject]

            # Convert to numeric feature matrix
            # Example columns: H, DD, UD
            # Adjust as needed if you have more columns
            X_subject = df_subject[["H", "DD", "UD"]].values

            # Train/test split
            # For simplicity, let’s do a quick hand-split:
            # - train on first 70%, test on remaining 30%
            split_index = int(0.7 * len(X_subject))
            X_train = X_subject[:split_index]
            X_test_genuine = X_subject[split_index:]

            # Train GMM for this user
            user_model = self.train_user_model(subject, X_train)

            # Score on genuine test
            if len(X_test_genuine) > 0:
                genuine_scores = user_model.score_samples(X_test_genuine)
                # By default, GMM score_samples is the log-likelihood
                all_genuine_scores.extend(genuine_scores)

            # Score on a small subset of impostors (for speed, maybe 5 from each impostor subject)
            X_impostors = df_impostors.groupby("subject").head(5)[["H","DD","UD"]].values
            if len(X_impostors) > 0:
                imposter_scores = user_model.score_samples(X_impostors)
                all_imposter_scores.extend(imposter_scores)

        # Now compute ROC curve: Label genuine=1, impostor=0
        y_true = np.array([1]*len(all_genuine_scores) + [0]*len(all_imposter_scores))
        y_scores = np.concatenate((all_genuine_scores, all_imposter_scores))

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        return fpr, tpr, thresholds

def train_gmm_model(csv_path):
    """
    Convenience function: read CSV, train GMM on all data, 
    return fpr, tpr, thresholds for plotting or analysis.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # Drop rows with NaN or invalid values
    df = df.dropna(subset=["H","DD","UD"])

    detector = GmmDetector(df, n_components=2)  # or tune n_components
    fpr, tpr, thresholds = detector.evaluate_set()
    return fpr, tpr, thresholds