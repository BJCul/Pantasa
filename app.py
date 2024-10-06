from flask import Flask, render_template, request, jsonify
import logging
from preprocess import preprocess_text, lemmatize_text
from error_detection import detect_grammar_errors
from error_correction import correct_grammar_errors

# Set up the tokenizer and model during Flask app initialization
tokenizer, model = set_up()

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
            # Step 1: Preprocess the input (tokenize, lemmatize, etc.)
            preprocessed_text = preprocess_text(text_input)
            
            # Step 2: Detect grammar errors using the hybrid n-gram rules and POS tags
            grammar_errors = detect_grammar_errors(preprocessed_text)

            # Step 3: Correct the grammar errors using the model and rules
            corrected_text = correct_grammar_errors(grammar_errors, model, tokenizer)

            # Step 4: Return the grammar-checking result
            result = {
                "input": text_input,
                "preprocessed": preprocessed_text,
                "grammar_errors": grammar_errors,
                "corrected_text": corrected_text
            }

            # Log the input and result
            logging.info(f"Received Input: {text_input}")
            logging.info(f"Grammar Errors: {grammar_errors}")
            logging.info(f"Corrected Text: {corrected_text}")

            # Return the grammar-checking result as JSON
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
