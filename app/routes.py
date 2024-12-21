from flask import Blueprint, render_template, request, jsonify

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/identify', methods=['POST'])
def identify_user():
    data = request.json
    keystroke_features = data['features']  # Features sent from the frontend
    user_prediction = model.predict([keystroke_features])
    return jsonify({'user': user_prediction[0]})
