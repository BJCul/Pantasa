document.addEventListener('DOMContentLoaded', function () {
    // Select the checkbox and the 'Try Now' link
    const checkBox = document.getElementById('check');
    const tryNowLink = document.querySelector('.nav ul li a[href="#GrammarChecker"]');

    // Add click event listener to the 'Try Now' link
    tryNowLink.addEventListener('click', function (event) {
        if (checkBox.checked) {
            checkBox.checked = false; // Uncheck the checkbox to hide the menu
        }
    });
});

// Function to hide the menu after a link is clicked
document.querySelectorAll('.nav ul li a').forEach(link => {
    link.addEventListener('click', function () {
        document.getElementById('check').checked = false; // Uncheck the checkbox to hide the menu
        document.querySelector('.home-content').style.display = 'block'; // Show the home-content again
    });
});


document.addEventListener('DOMContentLoaded', function () {
    // Select all sections to be observed
    const sections = document.querySelectorAll('section');

    // Function to handle the visibility changes
    function handleIntersect(entries, observer) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                console.log('Visible section:', entry.target.id);  // Logs the visible section
                // You can apply logic here to change the navigation bar or other elements
                updateNavbar(entry.target.id); // Example: Update navbar based on the section ID
            }
        });
    }

    // Create an IntersectionObserver with a callback
    const observer = new IntersectionObserver(handleIntersect, {
        threshold: 0.5  // 50% of the section must be visible to trigger
    });

    // Observe each section
    sections.forEach(section => {
        observer.observe(section);
    });

    // Example function to update navbar or style based on section
    function updateNavbar(sectionId) {
        const navbar = document.querySelector('.nav');

        if (sectionId === 'Home') {
            navbar.classList.add('white');
            navbar.classList.remove('black');
        } else {
            navbar.classList.add('black');
            navbar.classList.remove('white');
        }
    }
});

function showSection(sectionId) {
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        section.classList.add('hidden');
    });

    const targetSection = document.getElementById(sectionId);
    targetSection.classList.remove('hidden');
}


document.getElementById('grammarTextarea').addEventListener('input', function (event) { //for character limit

    const textarea = document.getElementById('grammarTextarea');
    const charCount = document.getElementById('charCount');
    const maxLength = 150;
    let currentLength = textarea.textContent.length;

    // Prevent further input if character limit is reached
    if (currentLength > maxLength) {
        // Prevent the additional input beyond the limit
        event.preventDefault();
        textarea.textContent = textarea.textContent.substring(0, maxLength); // Trim excess characters
        currentLength = maxLength;
    }

    const remainingChars = maxLength - currentLength;
    charCount.textContent = `${currentLength}/${maxLength}`;

    // Change color based on remaining characters
    if (remainingChars <= 25) {
        charCount.style.color = '#B9291C';  // Red when limit is reached
    } else if (remainingChars <= 50) {
        charCount.style.color = '#DB7F15';  // Orange when few characters left
    } else if (remainingChars <= 75) {
        charCount.style.color = '#EEBA2B';  // Yellow
    } else {
        charCount.style.color = '#7c7573';  // Default color
    }
});

document.getElementById('grammarTextarea').addEventListener('keydown', function (event) {
    const textarea = document.getElementById('grammarTextarea');
    const maxLength = 150;
    
    // Prevent typing if character limit is reached
    if (textarea.textContent.length >= maxLength && event.key !== "Backspace" && event.key !== "Delete") {
        event.preventDefault();  // Prevent further input
    }
});



let timeout = null;

document.getElementById('grammarTextarea').addEventListener('input', function () {
    clearTimeout(timeout);
    const grammarTextarea = document.getElementById('grammarTextarea');
    const textInput = grammarTextarea.innerHTML;  // Get input text

    // Hide previous predictions and suggestions
    const predictionsContent = document.getElementById('predictionsContent');
    const suggestionsContent = document.getElementById('suggestionsContent');
    const suggestionsHeader = document.getElementById('suggestionsHeader');
    
    predictionsContent.innerHTML = '';
    suggestionsContent.innerHTML = '';
    suggestionsHeader.classList.add('hidden'); // Hide suggestions header initially
    
    // Show loading icon and set text to "Loading..."
    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
        predictionsContent.style.display= 'none';
        loadingElement.style.display = 'block';
        loadingElement.querySelector('p').textContent = 'grammar checking is processing...';
    }    

    // Log the input text before sending it
    console.log("Input Text:", textInput);

    // Clear previous corrections
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
            console.log('Response data:', data);  // Debugging

            // Check if there are any spelling errors
            const hasSpellingErrors = data.highlighted_text && data.highlighted_text.includes('error');

            if (hasSpellingErrors) {
                // Display only spell suggestions if there are errors
                predictionsContent.innerHTML = data.highlighted_text || 'No errors detected.'; 
            } else {
                // Display grammatical predictions if no spelling errors
                if (data.grammar_predictions && data.grammar_predictions.length) {
                    data.grammar_predictions.forEach((predictionArray) => {
                        const prediction = Array.isArray(predictionArray) ? predictionArray[0] : predictionArray;
                        const predictionText = prediction == 0
                            ? `Grammatically correct.<br>`
                            : `Grammatical error detected.<br>`;
                        predictionsContent.innerHTML += predictionText;
                    });
                } else {
                    predictionsContent.innerHTML = 'No grammatical errors detected';
                }
            }

            // Add mouseenter and mouseleave event listeners for highlighted errors (if any)
            document.querySelectorAll('.error').forEach(element => {
                element.addEventListener('mouseenter', function () {
                    const suggestions = this.getAttribute('data-suggestions').split(',<br>');
                    showSuggestions(suggestions, this);
                });

                element.addEventListener('mouseleave', function () {
                    hideSuggestions();
                });
            });


        } catch (error) {
            console.error('Error retrieving data:', error);  // Log the actual error
            document.getElementById('correctedText').textContent = 'Error retrieving data.';
        } finally {
            // Change loading text to "Complete"
            loadingElement.querySelector('p').textContent = 'Complete';

            // Hide loading icon after a short delay
            setTimeout(() => {
                predictionsContent.style.display= 'block';
                loadingElement.style.display = 'none';
            }, 500);
        }
    }, 1000);  // Adjust delay if needed
});

// Function to show suggestions in a suggestion box
function showSuggestions(suggestions, errorElement) {
    // Remove any existing suggestion box
    hideSuggestions();

    // Create a new suggestion box
    const suggestionBox = document.createElement('div');
    suggestionBox.className = 'suggestion-box';
    suggestionBox.innerHTML = `<strong>Suggestions:</strong><br>${suggestions.join('<br>')}`;

    // Position the suggestion box near the highlighted word
    const rect = errorElement.getBoundingClientRect();
    suggestionBox.style.position = 'absolute';
    suggestionBox.style.left = `${rect.left}px`;
    suggestionBox.style.top = `${rect.bottom + window.scrollY}px`; // Adjust for scrolling

    // Add the suggestion box to the body
    document.body.appendChild(suggestionBox);
}

// Function to hide the suggestion box
function hideSuggestions() {
    const existingSuggestionBox = document.querySelector('.suggestion-box');
    if (existingSuggestionBox) {
        existingSuggestionBox.remove();
    }
}

