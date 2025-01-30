import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

def extract_keystroke_features(df):
    """Computes digraph latencies & hold times."""
    df = df.sort_values(["user", "timestamp"])
    
    df["latency"] = df.groupby("user")["timestamp"].diff()  # Press-Press latency
    df["hold_time"] = df.groupby(["user", "key"])["timestamp"].diff()  # Hold time
    df.dropna(inplace=True)

    return df[["user", "latency", "hold_time"]]

def train_gmm_model(df, users):
    """Trains a Gaussian Mixture Model for each user."""
    print("start")
    user_models = {}
    df = extract_keystroke_features(df)

    for user in users:
        user_data = df[df["user"] == user][["latency", "hold_time"]].dropna().values
        if len(user_data) > 10:  
            gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
            gmm.fit(user_data)
            user_models[user] = gmm

    return user_models

def authenticate_keystrokes(df, user_models):
    """Authenticates test users and computes prediction scores."""
    df = extract_keystroke_features(df)
    
    y_true, y_scores = [], []
    for _, row in df.iterrows():
        user, sample = row["user"], row[["latency", "hold_time"]].values.reshape(1, -1)

        # Compute likelihood scores from GMM
        if user in user_models:
            scores = {u: model.score_samples(sample)[0] for u, model in user_models.items()}
            best_match = max(scores, key=scores.get)
            y_true.append(1 if best_match == user else 0)
            y_scores.append(scores[user] if user in scores else 0)

    return np.array(y_true), np.array(y_scores)
