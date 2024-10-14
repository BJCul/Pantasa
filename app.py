from flask import Flask, render_template, request, jsonify
from app.pantasa_checker import pantasa_checker
from app.utils import load_hybrid_ngram_patterns
import logging
import os

# Specify the correct paths for templates and static folders
app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Route to handle grammar checking POST request
@app.route('/get_text', methods=['POST'])
def get_text():
    try:
        # Get JSON data from the POST request
        data = request.get_json()
        text_input = data.get('text_input', '')

        if text_input:
            jar_path = 'rules/Libraries/FSPOST/stanford-postagger.jar'
            model_path = 'rules/Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
            hybrid_ngram_patterns = load_hybrid_ngram_patterns('data/processed/hngrams.csv')
            
            # Call the Pantasa function to process the sentence and get the suggestions and misspelled words
            suggestions, misspelled_words, rule_corrected_text = pantasa_checker(text_input, jar_path, model_path, hybrid_ngram_patterns)
            
            highlighted_text = text_input
            for word, suggestion in misspelled_words:
                highlighted_text = highlighted_text.replace(
                    word, f'<span class="error" data-suggestions="{"".join(suggestion)}">{word}</span>'
                )

                            
            # Return the grammar-checking result as JSON
            result = {
                "input_text": text_input,
                "highlighted_text": highlighted_text,
                "corrected_text": rule_corrected_text,
                "suggestions": suggestions,
                "misspelled_words": misspelled_words
            }
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
