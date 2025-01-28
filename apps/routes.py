from flask import Blueprint, render_template, request, jsonify
from apps.utils import execute_experimentGP, process_keystrokes_with_repetitionsManhattan
from apps.manhattan import ManhattanDetector
import os
import pandas
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
    return jsonify({"status": "success"})

@main.route("/experimentManhattan", methods=["POST"])
def TestBuffaloManhattan():
    # original data
    input_path = "dataset"
    output_csv = "dataset/output_Manhattan.csv" 
    process_keystrokes_with_repetitionsManhattan(input_path, output_csv)

    data1 = pandas.read_csv(output_csv)
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