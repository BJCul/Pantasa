# Deep learning-based error correction
from model.transformer_model import correct_sentence

def correct_text(text):
    # Use a pre-trained model to correct grammar
    return correct_sentence(text)
