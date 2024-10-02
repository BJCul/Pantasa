# Configuration settings for the Flask app
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'default_secret_key'
    DEBUG = True
    # Add more configurations as needed (e.g., DB connection)
