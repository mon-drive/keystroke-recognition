from flask import Blueprint, render_template, request, jsonify, send_from_directory
from apps.utils import execute_experimentGP, process_buffalo_keystrokes, convert_xlsx_to_csv, processAalto
from apps.gmm import train_gmm_model
from apps.mahalanobis import train_mahalanobis_model
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
    distance_measure = request.form.get("selected_distance")

    print(distance_measure)

    y_true,y_scores = execute_experimentGP(dataset,distance_measure)
    
    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    far = fpr
    frr = 1 - tpr

    # Find Equal Error Rate (EER) point
    eer_index = np.nanargmin(np.abs(far - frr))  # Find the index where FAR and FRR are closest
    eer_threshold = thresholds[eer_index]  # Get the corresponding threshold
    eer_far = far[eer_index]  # Single value for FAR at EER
    eer_frr = frr[eer_index]  # Single value for FRR at EER

    print(f"FAR: {eer_far}")
    print(f"FRR: {eer_frr}")

    return jsonify({
        "status": "GP",
        "far": eer_far,
        "frr": eer_frr,
    })

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
        processAalto("dataset/Aalto/files", output_csv, 10000, 13000)
    elif(dataset == "Nanglae-Bhattarakosol"):
        xls1 = "dataset/fullname_userInformation.xlsx"
        xls2 = "dataset/email_userInformation.xlsx"
        xls3 = "dataset/phone_userInformation.xlsx"
        convert_xlsx_to_csv([xls1,xls2,xls3], output_csv)

    # 2. Train GMM and evaluate
    fpr, tpr, thresholds = train_gmm_model(output_csv, M=3,delta=1.0)

    # 3. Compute EER
    # The EER is where FPR == 1 - TPR. We can approximate with brentq:
    interp_func = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))

    eer = brentq(lambda x: 1. - x - interp_func(x), 0., 1.)

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

    # 2. Train GMM and evaluate
    fpr, tpr, thresholds = train_mahalanobis_model(output_csv, delta=1.0)

    # 3. Compute EER
    # The EER is where FPR == 1 - TPR. We can approximate with brentq:# Ensure the interpolation function starts and ends at valid points
    eer = brentq(
        lambda x: 1.0 - x - interp1d(fpr, tpr, fill_value="extrapolate")(x), 
        min(fpr) + 1e-6, max(fpr) - 1e-6
    )


    # Compute FRR and FAR
    frr = 1 - tpr  # False Rejection Rate
    far = fpr      # False Acceptance Rate

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
    plt.plot(fpr, tpr, color="blue", label=f"AUC: {auc(fpr, tpr):.3f}, EER: {eer:.3f}")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve - Mahalanobis - {dataset}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(roc_image_path)
    plt.close()

    # === 2. FRR vs Threshold ===
    plt.figure(figsize=(8, 8))
    plt.plot(thresholds, frr, color="green", label="FRR (False Rejection Rate)")
    plt.xlabel("Threshold")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title(f"FRR vs. Threshold - Mahalanobis - {dataset}")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(frr_image_path)
    plt.close()

    # === 3. FAR vs Threshold ===
    plt.figure(figsize=(8, 8))
    plt.plot(thresholds, far, color="red", label="FAR (False Acceptance Rate)")
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
