document.addEventListener('DOMContentLoaded', function() {
    // Select the checkbox and the 'Try Now' link
    const checkBox = document.getElementById('check');
    const tryNowLink = document.querySelector('.nav ul li a[href="#GrammarChecker"]');

    // Add click event listener to the 'Try Now' link
    tryNowLink.addEventListener('click', function(event) {
        if (checkBox && checkBox.checked) {
            checkBox.checked = false; // Uncheck the checkbox to hide the menu
        }
    });
});

// Function to hide the menu after a link is clicked
document.querySelectorAll('.nav ul li a').forEach(link => {
    link.addEventListener('click', function () {
        const checkBox = document.getElementById('check');
        if (checkBox) {
            checkBox.checked = false; // Uncheck the checkbox to hide the menu
        }
        const homeContent = document.querySelector('.home-content');
        if (homeContent) {
            homeContent.style.display = 'block'; // Show the home-content again
        }
    });
});

function showSection(sectionId) {
    // Hide all sections
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        section.classList.add('hidden');
    });

    // Show the target section
    const targetSection = document.getElementById(sectionId);
    targetSection.classList.remove('hidden');
}

let timeout = null;

document.getElementById('inputText').addEventListener('input', function () {
    clearTimeout(timeout);
    const textInput = this.value;

    // Check if the first letter is capitalized
    const isFirstLetterCapitalized = textInput.charAt(0) === textInput.charAt(0).toUpperCase();

    if (!isFirstLetterCapitalized && textInput.length > 0) {
        alert('Please start your sentence with a capitalized letter.');
        return;
    }

    // Hide previous predictions and suggestions
    const predictionsContent = document.getElementById('predictionsContent');
    const suggestionsContent = document.getElementById('suggestionsContent');
    const suggestionsHeader = document.getElementById('suggestionsHeader');

    if (predictionsContent) predictionsContent.innerHTML = '';
    if (suggestionsContent) suggestionsContent.innerHTML = '';
    if (suggestionsHeader) suggestionsHeader.classList.add('hidden');

    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
        loadingElement.style.display = 'flex';
        loadingElement.querySelector('p').textContent = 'Loading...';
    }

    console.log("Input Text:", textInput);

    timeout = setTimeout(async () => {
        try {
            const response = await fetch('/get_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text_input: textInput })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            console.log("Received Data:", data);

            const hasSpellingErrors = data.highlighted_text && data.highlighted_text.includes('error');

            if (hasSpellingErrors) {
                if (predictionsContent) predictionsContent.innerHTML = data.highlighted_text || 'No errors detected.';
            } else {
                if (data.grammar_predictions && data.grammar_predictions.length) {
                    data.grammar_predictions.forEach((predictionArray) => {
                        const prediction = Array.isArray(predictionArray) ? predictionArray[0] : predictionArray;
                        const predictionText = prediction === 0
                            ? `Grammatically correct.<br>`
                            : `Grammatical error detected.<br>`;
                        if (predictionsContent) predictionsContent.innerHTML += predictionText;
                    });
                } else if (predictionsContent) {
                    predictionsContent.innerHTML = 'No grammatical predictions available.';
                }
            }

            document.querySelectorAll('.error').forEach(element => {
                element.addEventListener('click', function () {
                    const suggestions = this.getAttribute('data-suggestions').split(',<br>');
                    showSuggestions(suggestions, this);
                });
            });

        } catch (error) {
            console.error('Error:', error);
            if (predictionsContent) predictionsContent.innerHTML = 'Error retrieving data. Maybe you forgot a period?.';
        } finally {
            if (loadingElement) {
                loadingElement.querySelector('p').textContent = 'Complete';
                setTimeout(() => {
                    loadingElement.style.display = 'none';
                }, 500);
            }
        }
    }, 1000);
});

function processText() {
    const inputText = document.getElementById('inputText').value;
    document.getElementById('outputText').value = inputText;
}

function showSuggestions(suggestions, errorElement) {
    const suggestionBox = document.createElement('div');
    suggestionBox.className = 'suggestion-box';
    suggestionBox.innerHTML = `<strong>Suggestions:</strong><br>${suggestions.join('<br>')}`;

    const rect = errorElement.getBoundingClientRect();
    suggestionBox.style.position = 'absolute';
    suggestionBox.style.left = `${rect.left}px`;
    suggestionBox.style.top = `${rect.bottom}px`;

    document.body.appendChild(suggestionBox);
    suggestionBox.addEventListener('click', () => {
        document.body.removeChild(suggestionBox);
    });
}
