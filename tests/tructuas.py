import os

# Get absolute path
log_file_path = os.path.abspath('logs/pantasa.log')

# Ensure the directory exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Open the file
with open(log_file_path, 'w') as log_file:
    log_file.truncate()
