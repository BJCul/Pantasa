from flask import request, jsonify
from . import create_app

app = create_app()

@app.route('/')
def index():
    return "Welcome to the Pantasa Grammar Checker API!"

# Define other routes for uploading text, processing, etc.
@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()
    # Call necessary preprocessing and correction functions
    return jsonify({"processed_text": "Result goes here"})
