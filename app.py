from flask import Flask, render_template, request, jsonify
import time

app = Flask(__name__)

# Route to serve the HTML page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle keystroke data
@app.route("/keystrokes", methods=["POST"])
def keystrokes():
    data = request.json
    print("Keystroke data received:", data)  # For debugging/logging
    # Here you can save the data to a database or file for further analysis
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)
