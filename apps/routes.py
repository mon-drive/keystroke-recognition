from flask import Blueprint, render_template, request, jsonify, send_from_directory
from apps.utils import execute_experimentGP, process_buffalo_keystrokes, convert_xlsx_to_csv, processAalto
from apps.gmm import train_gmm_model
from apps.mahalanobis import MahalanobisDetector
import os
import uuid
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, roc_curve, auc
import numpy as np

import matplotlib
matplotlib.use('Agg') # Non-GUI backend

import matplotlib.pyplot as plt
plt.style.use('ggplot')

main = Blueprint('main', __name__)

data_folder = "dataset"

# Directory to store temporary images
STATIC_IMAGE_FOLDER = os.path.join("apps","static","temp_images")
os.makedirs(STATIC_IMAGE_FOLDER, exist_ok=True)

# Route to serve the HTML page
@main.route("/")
def home():
    return render_template("index.html")

# Route to handle dataset selection
@main.route("/dataset_selection", methods=["POST"])
def dataset_selection():
    dataset = request.form.get("dropdown_dataset")
    print(f"Dataset selected: {dataset}")  # Debugging

    return render_template("index.html", selected_value=dataset)


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

@main.route("/get_plot/<filename>")
def get_plot(filename):
    return send_from_directory(STATIC_IMAGE_FOLDER, filename)

@main.route("/experimentGP", methods=["POST"])
def TestBuffaloGP():
    # original data

    dataset = request.form.get("selected_dataset")

    execute_experimentGP(dataset)

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

    # Save the plot as an image
    image_filename = f"{uuid.uuid4()}.png"
    image_path = os.path.join(STATIC_IMAGE_FOLDER, image_filename)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color="red", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal reference line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve - GP ")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(image_path)  # Save the image
    plt.close()  # Close plot to free memory

    return jsonify({"status": "success", "image_url": f"/static/temp_images/{image_filename}"})

# Train & Evaluate GMM on Buffalo Dataset
@main.route("/experimentGMM", methods=["POST"])
def TestGMM():
    """Processes fixed-text and free-text data and trains GMM authentication models."""
    
    input_path = "dataset"   # your base folder
    output_csv = "dataset/output_gmm.csv"

    dataset = request.form.get("selected_dataset")

    if(dataset == "Buffalo Fixed Text"):
        process_buffalo_keystrokes(input_path, output_csv, 0)
    elif(dataset == "Buffalo Free Text"):
        process_buffalo_keystrokes(input_path, output_csv, 1)
    elif(dataset == "Aalto"):
        processAalto("dataset/Aalto/files", output_csv, 10000, 16000)
    elif(dataset == "Nanglae-Bhattarakosol"):
        xls1 = "dataset/fullname_userInformation.xlsx"
        xls2 = "dataset/email_userInformation.xlsx"
        xls3 = "dataset/phone_userInformation.xlsx"
        #convert_xlsx_to_csv([xls1,xls2,xls3], output_csv)

    # 2. Train GMM and evaluate
    fpr, tpr, thresholds = train_gmm_model(output_csv, M=3,delta=1.0)

    # 3. Compute EER
    # The EER is where FPR == 1 - TPR. We can approximate with brentq:
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # 4. Plot and display AUC, EER
    roc_auc = auc(fpr, tpr)

    # Compute FRR and FAR
    frr = 1 - tpr  # False Rejection Rate
    far = fpr      # False Acceptance Rate

    # Unique filenames for images
    roc_image_filename = f"{uuid.uuid4()}.png"
    roc_image_path = os.path.join(STATIC_IMAGE_FOLDER, roc_image_filename)

    frr_image_filename = f"{uuid.uuid4()}.png"
    frr_image_path = os.path.join(STATIC_IMAGE_FOLDER, frr_image_filename)

    far_image_filename = f"{uuid.uuid4()}.png"
    far_image_path = os.path.join(STATIC_IMAGE_FOLDER, far_image_filename)

    # === 1. ROC Curve with EER ===
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"AUC: {roc_auc:.3f}, EER: {eer:.3f}", color="blue")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve - GMM - {dataset}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(roc_image_path)
    plt.close()

    # === 2. FRR vs Threshold ===
    plt.figure(figsize=(8, 8))
    plt.plot(thresholds, frr, color="green", label="FRR (False Rejection Rate)")
    plt.xlabel("Threshold")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title(f"FRR vs. Threshold - GMM - {dataset}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(frr_image_path)
    plt.close()

    # === 3. FAR vs Threshold ===
    plt.figure(figsize=(8, 8))
    plt.plot(thresholds, far, color="red", label="FAR (False Acceptance Rate)")
    plt.xlabel("Threshold")
    plt.ylabel("False Acceptance Rate (FAR)")
    plt.title(f"FAR vs. Threshold - GMM - {dataset}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(far_image_path)
    plt.close()

    return jsonify({
    "status": "success",
    "roc_image_url": f"/static/temp_images/{roc_image_filename}",
    "frr_image_url": f"/static/temp_images/{frr_image_filename}",
    "far_image_url": f"/static/temp_images/{far_image_filename}",
    })

@main.route("/experimentMahalanobis", methods=["POST"])
def TestMahalanobis():
    # original data
    input_path = "dataset"
    output_csv = "dataset/output_Mahalanobis.csv" 

    dataset = request.form.get("selected_dataset")
    
    if(dataset == "Buffalo Fixed Text"):
        process_buffalo_keystrokes(input_path, output_csv, 0)
    elif(dataset == "Buffalo Free Text"):
        process_buffalo_keystrokes(input_path, output_csv, 1)
    elif(dataset == "Aalto"):
        processAalto("dataset/Aalto/files", output_csv, 10000, 16000)
    elif(dataset == "Nanglae-Bhattarakosol"):
        xls1 = "dataset/fullname_userInformation.xlsx"
        xls2 = "dataset/email_userInformation.xlsx"
        xls3 = "dataset/phone_userInformation.xlsx"
        convert_xlsx_to_csv([xls1, xls2, xls3], output_csv)

    data1 = pd.read_csv(output_csv)
    subjects1 = data1["subject"].unique()
    print("Subjects: ")
    fpr1_1, tpr1_1, thresholds1_1 = MahalanobisDetector(subjects1, data1).evaluateSet1()
    eer1_1 = brentq(lambda x : 1. - x - interp1d(fpr1_1, tpr1_1)(x), 0., 1.)
    print("EER1_1: ", eer1_1)

    # Compute FRR and FAR
    frr = 1 - tpr1_1  # False Rejection Rate
    far = fpr1_1      # False Acceptance Rate

    # Generate unique filenames for images
    roc_image_filename = f"{uuid.uuid4()}.png"
    roc_image_path = os.path.join(STATIC_IMAGE_FOLDER, roc_image_filename)

    frr_image_filename = f"{uuid.uuid4()}.png"
    frr_image_path = os.path.join(STATIC_IMAGE_FOLDER, frr_image_filename)

    far_image_filename = f"{uuid.uuid4()}.png"
    far_image_path = os.path.join(STATIC_IMAGE_FOLDER, far_image_filename)

    # === 1. ROC Curve with EER ===
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.plot(fpr1_1, tpr1_1, color="blue", label=f"AUC: {auc(fpr1_1, tpr1_1):.3f}, EER: {eer1_1:.3f}")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve - Mahalanobis - {dataset}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(roc_image_path)
    plt.close()

    # === 2. FRR vs Threshold ===
    plt.figure(figsize=(8, 8))
    plt.plot(thresholds1_1, frr, color="green", label="FRR (False Rejection Rate)")
    plt.xlabel("Threshold")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title(f"FRR vs. Threshold - Mahalanobis - {dataset}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(frr_image_path)
    plt.close()

    # === 3. FAR vs Threshold ===
    plt.figure(figsize=(8, 8))
    plt.plot(thresholds1_1, far, color="red", label="FAR (False Acceptance Rate)")
    plt.xlabel("Threshold")
    plt.ylabel("False Acceptance Rate (FAR)")
    plt.title(f"FAR vs. Threshold - Mahalanobis - {dataset}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(far_image_path)
    plt.close()


    return jsonify({
    "status": "success",
    "roc_image_url": f"/static/temp_images/{roc_image_filename}",
    "frr_image_url": f"/static/temp_images/{frr_image_filename}",
    "far_image_url": f"/static/temp_images/{far_image_filename}",
    })
