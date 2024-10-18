document.addEventListener('DOMContentLoaded', function () {
    const checkBox = document.getElementById('check');
    const tryNowLink = document.querySelector('.nav ul li a[href="#GrammarChecker"]');

    tryNowLink.addEventListener('click', function (event) {
        if (checkBox.checked) {
            checkBox.checked = false;
        }
    });

    let selectedWordElement = null;

    // Highlighted text click handler
    document.addEventListener('click', function (event) {
        if (event.target.classList.contains('highlight')) {
            selectedWordElement = event.target;  // Store the clicked word element
            const clickedWord = event.target.textContent;  // Get the clicked word
            const suggestionsList = document.getElementById('suggestionsList');

            // Clear previous suggestions
            suggestionsList.innerHTML = '';

            // Fetch the spelling suggestions for the clicked word
            const suggestions = window.spellingSuggestions[clickedWord] || ['No suggestions available'];

            // Display each suggestion as a list item
            suggestions.forEach(suggestion => {
                const suggestionItem = document.createElement('li');
                suggestionItem.textContent = suggestion;

                // Add click listener for each suggestion
                suggestionItem.addEventListener('click', function () {
                    replaceHighlightedWord(clickedWord, suggestion);
                });

                suggestionsList.appendChild(suggestionItem);
            });
        }
    });

    

    // Function to replace the highlighted word with the clicked suggestion
    function replaceHighlightedWord(incorrectWord, newWord) {
        const grammarTextarea = document.getElementById('grammarTextarea');

        // Replace the incorrect word with the new word in the grammarTextarea
        if (selectedWordElement) {
            selectedWordElement.textContent = newWord;  // Update the displayed word
            selectedWordElement.classList.remove('highlight');  // Remove highlight after correction
        }

        // Trigger the grammar check again
        triggerGrammarCheck();
    }

    function triggerGrammarCheck() {
        clearTimeout(timeout);
        const grammarTextarea = document.getElementById('grammarTextarea');
        const textInput = grammarTextarea.innerHTML;  // Get input text

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

                if (response.status >= 400 && response.status < 600) {
                    throw new Error('Server returned an error');
                }

                const data = await response.json();
                console.log('Response data:', data);  // Debugging

                if (data.corrected_text && data.incorrect_words) {
                    let highlightedText = textInput;

                    // Highlight incorrect words
                    data.incorrect_words.forEach(word => {
                        const regex = new RegExp(`\\b${word}\\b`, 'gi');
                        highlightedText = highlightedText.replace(regex, `<span class="highlight">${word}</span>`);
                    });

                    // Store spelling suggestions in the global object
                    window.spellingSuggestions = data.spelling_suggestions;

                    // Display highlighted text in the textarea
                    grammarTextarea.innerHTML = highlightedText;
                    document.getElementById('correctedText').textContent = data.corrected_text.replace(/<[^>]+>/g, "");
                    // Store corrected sentence globally
                    window.correctedSentence = data.corrected_text;

                } else {
                    grammarTextarea.innerHTML = textInput;
                    document.getElementById('correctedText').textContent = "No corrections needed.";
                }

            } catch (error) {
                console.error('Error retrieving data:', error);
                document.getElementById('correctedText').textContent = 'Error retrieving data.';
            } finally {
                hideLoadingSpinner();  // Always hide spinner
            }
        }, 1000);  // Adjust delay if needed
    }

});

document.getElementById('grammarTextarea').addEventListener('input', function (event) {
    const textarea = document.getElementById('grammarTextarea');
    const charCount = document.getElementById('charCount');
    const maxLength = 150;
    let currentLength = textarea.textContent.length;

    // Prevent further input if character limit is reached
    if (currentLength > maxLength) {
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
        event.preventDefault();
    }
});

function checkFlaskStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            const isLogging = data.logging;
            const spinner = document.getElementById('loading');

            if (isLogging) {
                spinner.style.display = 'block'; // Show the spinner when Flask is logging
            } else {
                spinner.style.display = 'none';  // Hide the spinner when Flask is idle
            }
        })
        .catch(error => {
            console.error('Error checking Flask status:', error);
        });
}

// Set an interval to check the status every 2 seconds
setInterval(checkFlaskStatus, 4000);

// Function to hide the menu after a link is clicked
document.querySelectorAll('.nav ul li a').forEach(link => {
    link.addEventListener('click', function () {
        document.getElementById('check').checked = false;
        document.querySelector('.home-content').style.display = 'block';
    });
});

function showSection(sectionId) {
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        section.classList.add('hidden');
    });

    const targetSection = document.getElementById(sectionId);
    targetSection.classList.remove('hidden');
}

let timeout = null;

document.getElementById('grammarTextarea').addEventListener('input', function () {
    clearTimeout(timeout);
    const grammarTextarea = document.getElementById('grammarTextarea');
    const textInput = grammarTextarea.innerHTML;  // Get input text

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

            if (response.status >= 400 && response.status < 600) {
                throw new Error('Server returned an error');
            }

            const data = await response.json();
            console.log('Response data:', data);  // Debugging

            if (data.corrected_text && data.incorrect_words) {
                let highlightedText = textInput;

                // Highlight incorrect words
                data.incorrect_words.forEach(word => {
                    const regex = new RegExp(`\\b${word}\\b`, 'gi');
                    highlightedText = highlightedText.replace(regex, `<span class="highlight">${word}</span>`);
                });

                // Store spelling suggestions in the global object
                window.spellingSuggestions = data.spelling_suggestions;

                // Display highlighted text in the textarea
                grammarTextarea.innerHTML = highlightedText;
                document.getElementById('correctedText').innerHTML = data.corrected_text;


                // Store corrected sentence globally
                window.correctedSentence = data.corrected_text;

            } else {
                grammarTextarea.innerHTML = textInput;
                document.getElementById('correctedText').textContent = "No corrections needed.";
            }

        } catch (error) {
            console.error('Error retrieving data:', error);
            document.getElementById('correctedText').textContent = 'Error retrieving data.';
        } finally {
            hideLoadingSpinner();  // Always hide spinner
        }
    }, 1000);  // Adjust delay if needed
});

document.addEventListener('DOMContentLoaded', function () {
    // Select all sections to be observed
    const sections = document.querySelectorAll('section');

    // Function to handle the visibility changes
    function handleIntersect(entries, observer) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                console.log('Visible section:', entry.target.id);  // Logs the visible section
                updateNavbar(entry.target.id);  // Example: Update navbar based on the section ID
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
