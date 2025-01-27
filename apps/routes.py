from flask import Blueprint, render_template, request, jsonify
from apps.utils import execute_experimentGP
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
def TestBuffaloGP():
    # original data
    execute_experimentGP()
    return jsonify({"status": "success"})