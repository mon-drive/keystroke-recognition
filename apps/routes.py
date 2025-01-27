from flask import Blueprint, render_template, request, jsonify

from apps.utils import extract_from_buffalo
from apps.GunettiPicardi import create_user_profiles, experiment
import os

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
def execute_experimentGP():
    # original data
    extract_from_buffalo()
    original_set = "./dataset/keystroke_baseline_task1.csv"
    original_data_profiles = f"./{data_folder}/original_data_profiles"

    print("Original data profiles: ", original_data_profiles)

    if not os.path.isfile(original_data_profiles):
        create_user_profiles(original_set, original_data_profiles)

    experiment(original_data_profiles, original_data_profiles, "original", filter)
    return jsonify({"status": "success"})