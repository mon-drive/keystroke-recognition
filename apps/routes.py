from flask import Blueprint, render_template, request, jsonify
from apps.utils import execute_experimentGP, process_keystrokes_with_repetitionsManhattan,extract_from_buffalo
from apps.manhattan import ManhattanDetector
from apps.gmm import train_gmm_model,authenticate_keystrokes
import os
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc

plt.style.use('ggplot')

main = Blueprint('main', __name__)

data_folder = "dataset"

# Route to serve the HTML page
@main.route("/")
def home():
    return render_template("index.html")

# Route to handle keystroke data
@main.route("/keystrokes", methods=["POST"])
def keystrokes():
    data = request.json
    print("Keystroke data received:", data)  # For debugging/logging
    # Here you can save the data to a database or file for further analysis

    #features = extract_features(data)

    print("  ")

    #print("Extracted features:", features["key_keydown"])  # For debugging/logging

    return jsonify({"status": "success"})

@main.route("/experimentGP", methods=["POST"])
def TestBuffaloGP():
    # original data
    execute_experimentGP()

    #open file
    auth_file = "dataset/original_authentification_data.csv"
    df = pd.read_csv(auth_file)

    # Define legitimate users (1) and imposters (0)
    y_true = (df["FalseRejectAttempts"] > 0).astype(int)  # 1 for legitimate users
    y_true[df["FalseAcceptError"] > 0] = 0  # 0 for imposters

    # Use FalseReject1 as the decision score
    y_scores = df["FalseReject1"].values  

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal reference line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return jsonify({"status": "success"})

@main.route("/experimentManhattan", methods=["POST"])
def TestBuffaloManhattan():
    # original data
    input_path = "dataset"
    output_csv = "dataset/output_Manhattan.csv" 
    process_keystrokes_with_repetitionsManhattan(input_path, output_csv)

    data1 = pd.read_csv(output_csv)
    subjects1 = data1["subject"].unique()
    print("Subjects: ")
    fpr1_1, tpr1_1, thresholds1_1 = ManhattanDetector(subjects1, data1).evaluateSet1()
    eer1_1 = brentq(lambda x : 1. - x - interp1d(fpr1_1, tpr1_1)(x), 0., 1.)
    print("EER1_1: ", eer1_1)

    plt.figure(figsize = (10,5))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr1_1, tpr1_1, label='AUC = {:.3f}, EER = {:.3f} Set-1-Manhattan'.format(auc(fpr1_1, tpr1_1), eer1_1))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


    return jsonify({"status": "success"})

# Train & Evaluate GMM on Buffalo Dataset
@main.route("/experimentGMM", methods=["POST"])
def TestBuffaloGMM():
    """Processes Buffalo fixed-text data and trains GMM authentication models."""

    # Step 1: Extract data from Buffalo dataset (Fixed Text)
    input_path = "./dataset"
    output_csv = "./dataset/keystroke_fixed_text_gmm.csv"
    process_keystrokes_for_gmm(input_path, output_csv)  # Uses the new function

    # Step 2: Load the extracted CSV
    df = pd.read_csv(output_csv)
    users = df["subject"].unique()  # Users are identified by 'subject'

    # Step 3: Train GMM models
    user_models = train_gmm_model(df, users)

    # Step 4: Authenticate users & compute ROC curve
    y_true, y_scores = authenticate_keystrokes(df, user_models)

    # Compute ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    # Step 5: Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {roc_auc:.2f}, EER = {eer:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve - GMM Authentication (Fixed Text)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    return jsonify({"status": "success", "AUC": roc_auc, "EER": eer})
