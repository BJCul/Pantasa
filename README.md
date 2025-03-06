# **Pantasa: A Rule-Based and Deep Learning Approach to Tagalog Grammar Correction**

## **1. Abstract**
Pantasa is a grammar correction tool designed to address the scarcity of high-quality Tagalog grammar checkers by integrating rule-based and deep learning approaches. The study aimed to improve the accuracy of Tagalog grammar correction by combining a hybrid n-gram model for error detection with an iterative tagging sequence for correction. The system was developed using Python and various NLP tools, including Stanford POS Tagger and Morphinas Lemmatizer, and was trained on a dataset consisting of parallel corpora, editorial articles, and synthetic error-filled sentences. The evaluation of Pantasa was conducted using precision, recall, F1 measure, and accuracy as performance metrics. The grammar error detection module achieved a precision of 60.51%, a recall of 69.20%, an F1 measure of 62.05%, and an accuracy of 60.12%. The system was benchmarked against the existing Balarila model using a One-Mean T-Test, demonstrating competitive results in iterative corrections.
The findings highlight the effectiveness of integrating rule-based detection with deep learning correction in improving Tagalog grammar checking. However, challenges such as false positives, false negatives, and limitations due to the low-resource nature of the Tagalog language were observed. Future improvements include expanding the dataset with real-world sentences, refining predefined grammar rules, and enhancing model training with authentic errors instead of synthetic augmentations. Pantasa’s development contributes to the advancement of natural language processing for Tagalog, enabling more accurate and efficient grammar correction. The system has potential applications in education, professional writing, and automated language assessment, providing a reliable tool for improving written Tagalog text quality. Further research is recommended to refine the model’s performance.


## **2. Purpose of the Project**
The primary objective of Pantasa is to enhance the quality of written Tagalog by addressing common grammatical errors. This project aims to:
- Provide an automated tool for **Tagalog grammar correction** using both linguistic rules and AI-driven corrections.
- Develop an **iterative refinement model** that enhances the accuracy of grammatical structures in sentences.
- Compare and evaluate the effectiveness of **rule-based** and **deep learning** approaches in grammar correction.
- Contribute to the advancement of **natural language processing (NLP)** for Tagalog.

## **3. System Features**
- **Hybrid Rule-Based Error Detection**: Uses n-grams   and predefined grammar rules to detect inconsistencies in sentence structures.
- **Deep Learning-Based Correction**: Employs a fine-tuned neural network model to iteratively refine and improve sentences.
- **Iterative Refinement**: The system corrects errors in multiple passes until the sentence meets an acceptable grammatical standard.
- **Evaluation Metrics**: Performance is assessed using precision, recall, F1-score, and accuracy against a test dataset.

## **4. Authors and Contributors**
**Developers:** Carlo Agas,
                James Alcantara, 
                Rachell Ann Tapia, 
                Jarlson Figueroa
**Project Advisors:** [Advisor Names]  
**Affiliation:** Polytechnic University of the Philippines
**Year:** 2025  

## **5. System Requirements**
To run Pantasa, ensure that your system meets the following requirements:
- **Operating System:** Windows, Linux, or macOS
- **Programming Language:** Python 3.8+
- **Dependencies:**
  - TensorFlow / PyTorch
  - NLTK
  - NumPy
  - Pandas
  - Flask (if using the web-based API)
  - Jupyter Notebook (for testing and development)
- **Hardware:**
  - Minimum: 8GB RAM, Dual-Core Processor
  - Recommended: 16GB RAM, GPU support for deep learning

## **7. Evaluation and Benchmarking**
Pantasa is evaluated based on standard NLP performance metrics:
- **Precision**: Measures the accuracy of corrections made.
- **Recall**: Evaluates the ability to detect grammar errors.
- **F1-Score**: Balances precision and recall for overall performance.
- **Accuracy**: Determines the percentage of correctly corrected sentences.

## **8. Future Enhancements**
- Expansion of grammar rules and deep learning training data.
- Support for colloquial and informal Tagalog grammar correction.
- Integration with educational tools for automated feedback.
- Deployment as a browser extension or mobile application.

## **9. Citation**
For inquiries or collaborations, contact jamesalcantara185@gmail.com, rachellann344@gmail.com, jarlson.figueroa16@gmail.com, carlo.agas341@gmail.com.


