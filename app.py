from flask import Flask, render_template, request, jsonify
import time
import json

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

    features = extract_features(data)

    print("  ")

    print("Extracted features:", features)  # For debugging/logging

    return jsonify({"status": "success"})


def extract_features(keystrokes):
    """
    Extracts features from the raw keystroke data for recognition.
    Features include key press durations, flight times, and typing speed.
    """
    key_durations = {}  # Dictionary to store key press durations
    flight_times = []   # List to store flight times between key events
    total_keys = 0      # Counter for total keys typed
    valid_keys = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ") # whitelist of valid keys

    # Iterate through keystroke events to calculate durations and flight times
    last_keyup_time = None
    start_time = keystrokes[0]['timestamp'] if keystrokes else 0
    end_time = keystrokes[-1]['timestamp'] if keystrokes else 0

    for event in keystrokes:
        if event['type'] == 'keydown':
            # Record the time of keydown for duration calculation
            key_durations[event['key']] = key_durations.get(event['key'], [])
            key_durations[event['key']].append(event['timestamp'])
        elif event['type'] == 'keyup':
            # Calculate key press duration
            if event['key'] in key_durations and key_durations[event['key']]:
                keydown_time = key_durations[event['key']].pop(0)
                key_durations[event['key']].append(event['timestamp'] - keydown_time)
            
            # Count valid keys
            if event['key'] in valid_keys:
                total_keys += 1
            
            # Calculate flight time if a previous keyup exists
            if last_keyup_time is not None:
                flight_times.append(event['timestamp'] - last_keyup_time)
            last_keyup_time = event['timestamp']

    # Average durations and flight times
    avg_durations = {key: sum(durations) / len(durations) for key, durations in key_durations.items() if durations}
    avg_flight_time = sum(flight_times) / len(flight_times) if flight_times else 0

    # Calculate typing speed (WPM)
    total_time_seconds = (end_time - start_time) / 1000  # Convert milliseconds to seconds
    total_time_minutes = total_time_seconds / 60        # Convert seconds to minutes
    wpm = (total_keys / 5) / total_time_minutes if total_time_minutes > 0 else 0

    return {
        "average_key_durations": avg_durations,
        "average_flight_time": avg_flight_time,
        "typing_speed_wpm": round(wpm, 2),
        "total_time_seconds": total_time_seconds,
        "total_keys": total_keys
    }


if __name__ == "__main__":
    app.run(debug=True)