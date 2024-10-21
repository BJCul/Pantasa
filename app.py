from flask import Flask, render_template, request, jsonify
from app.pantasa_checker import pantasa_checker
import logging
import os
from threading import Lock

# Specify the correct paths for templates and static folders
app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Lock and global variable to track logging activity
logging_lock = Lock()
is_logging = False

# Route to handle grammar checking POST request
@app.route('/get_text', methods=['POST'])
def get_text():
    try:
        # Get JSON data from the POST request
        global is_logging
        data = request.get_json()
        text_input = data.get('text_input', '')

        if text_input:
            jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
            model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
            rule_path = 'data/processed/detailed_ngrams_merged.csv'
            directory_path = 'data/raw/dictionary.csv'
            pos_path = 'data/processed/pos_dic'

            # Track when processing starts
            with logging_lock:
                is_logging = True
            
            # Call the Pantasa function to process the sentence and get the suggestions and misspelled words
            corrected_sentence, spelling_suggestions, incorrect_words = pantasa_checker(text_input, jar_path, model_path, rule_path, directory_path, pos_path)
            logging.info(f"Detected Incorrect Words: {spelling_suggestions}")
            logging.info(f"Suggestion for Incorrect Words: {incorrect_words}")


            # Return the grammar-checking result as JSON
            result = {
                "input_text": text_input,
                "corrected_text": corrected_sentence,
                "incorrect_words": incorrect_words,
                "spelling_suggestions": spelling_suggestions,
            }

            # Track when processing ends
            with logging_lock:
                is_logging = False

            return jsonify(result)
        else:
            return jsonify({"error": "No text_input provided"}), 400
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Invalid request"}), 400

# Route to serve the main HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Route to check logging status
@app.route('/status', methods=['GET'])
def check_status():
    global is_logging
    with logging_lock:
        return jsonify({'logging': is_logging})

# Handle CORS preflight OPTIONS request
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS, HEAD'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Run the Flask app
if __name__ == "__main__":
    app.run(port=8000, debug=True)