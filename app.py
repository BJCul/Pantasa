from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Route for home page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Route for about page (about.html)
@app.route('/about')
def about():
    return render_template('about.html')

# Route for contact page (contact.html)
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Route for main page (main.html)
@app.route('/main')
def main():
    return render_template('main.html')

# Route for handling grammar checking (POST request from JavaScript)
@app.route('/get_text', methods=['POST'])
def get_text():
    data = request.get_json()  # Receive the input text from the request
    text_input = data.get('text_input', '')

    # Example processing: Here you can add grammar checking logic
    if text_input:
        # Return a dummy response for now
        response = {
            "highlighted_text": f"Received: {text_input}",
            "grammar_predictions": [0]  # Example response for grammatical correctness
        }
    else:
        response = {
            "highlighted_text": "No input provided.",
            "grammar_predictions": []
        }

    return jsonify(response)  # Send JSON response back to the client

if __name__ == '__main__':
    app.run(debug=True)
