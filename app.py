from flask import Flask, jsonify, render_template, request, abort
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up tokenizer and model
# tokenizer, model = set_up()

# Route to handle the grammar-checking POST request
# @app.route('/get_text', methods=['POST'])
# def get_text():
#     try:
#         # Get JSON data from the POST request
#         data = request.get_json()
#         text_input = data.get('text_input', '')

#         if text_input:
#             result = figcheck(text_input, tokenizer, model)

#             # Log the input and result
#             logging.info(f"Received Input: {text_input}")
#             logging.info(f"Grammar Predictions: {result['grammar_predictions']}")

#             # Return the grammar-checking result as JSON
#             return jsonify(result)
#         else:
#             return jsonify({"error": "No text_input provided"}), 400
#     except Exception as e:
#         logging.error(f"Error processing request: {str(e)}")
#         return jsonify({"error": "Invalid request"}), 400

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
