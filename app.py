from flask import Flask, render_template

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

if __name__ == '__main__':
    app.run(debug=True)
