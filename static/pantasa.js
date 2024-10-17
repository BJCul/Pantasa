document.addEventListener('DOMContentLoaded', function () {
    const checkBox = document.getElementById('check');
    const tryNowLink = document.querySelector('.nav ul li a[href="#GrammarChecker"]');
    const characterLimit = 150;  // Set your character limit here

    // Character count display element
    const characterCount = document.createElement('p');
    characterCount.style.color = 'gray';
    characterCount.textContent = `0 / ${characterLimit}`;

    const grammarTextarea = document.getElementById('grammarTextarea');
    grammarTextarea.parentNode.appendChild(characterCount);  // Add the character counter below the text area

    grammarTextarea.addEventListener('input', function () {
        const textInput = grammarTextarea.innerText.trim();
        const textLength = textInput.length;
        const remainingChars = maxLength - textLength;

        // Update live character count
        characterCount.textContent = `${textLength} / ${characterLimit}`;

        // Change color if the character limit is reached
        if (remainingChars == 0) {
            characterCount.style.color = 'red';  // Make the count red
            grammarTextarea.innerText = textInput.substring(0, characterLimit);
        } else if (remainingChars <=25) {
            charCount.style.color = 'orange';  // Between none and 25 characters remaining
        } else if (remainingChars <=50) {
            charCount.style.color = 'yellow';  // Between 25 and 50 characters remaining
        } else {
            charCount.style.color = 'gray';  // Above 50 characters remaining
        }

    });

    tryNowLink.addEventListener('click', function (event) {
        if (checkBox.checked) {
            checkBox.checked = false;
        }
    });

    // Highlighted text click handler
    document.addEventListener('click', function (event) {
        if (event.target.classList.contains('highlight')) {
            const suggestionsList = document.getElementById('suggestionsList');
            const suggestionItem = document.createElement('li');
            suggestionItem.textContent = 'Clicked';
            suggestionsList.innerHTML = '';  // Clear previous suggestions
            suggestionsList.appendChild(suggestionItem);
        }
    });
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
    const textInput = grammarTextarea.innerText;  // Get the raw text input from the div

    // Toggle the empty class if no content
    if (textInput === '') {
        grammarTextarea.classList.add('empty');
    } else {
        grammarTextarea.classList.remove('empty');
    }

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

            if (data.corrected_text && data.incorrect_words) {
                let highlightedText = textInput;  // Keep the original input

                // Highlight incorrect words
                data.incorrect_words.forEach(word => {
                    const regex = new RegExp(`\\b${word}\\b`, 'gi');
                    highlightedText = highlightedText.replace(regex, `<span class="highlight">${word}</span>`);
                });

                // Display highlighted text in the textarea
                grammarTextarea.innerHTML = highlightedText;

                // Display corrected sentence
                document.getElementById('correctedText').textContent = data.corrected_text;

                hideLoadingSpinner();

            } else {
                // If no corrections, keep the original input
                grammarTextarea.innerHTML = textInput;
                document.getElementById('correctedText').textContent = "No corrections needed.";
                hideLoadingSpinner();
            }

        } catch (error) {
            console.error('Error:', error);
            document.getElementById('correctedText').textContent = 'Error retrieving data.';
            hideLoadingSpinner();
        }
    }, 1000);
});

// Function to check if Flask is logging
function checkFlaskStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            const isLogging = data.logging;
            const spinner = document.getElementById('loading');

            if (isLogging) {
                // Show the spinner when Flask is logging
                spinner.style.display = 'block';
                correctedText.style.display = 'none';
            } else {
                // Hide the spinner when Flask is not logging
                spinner.style.display = 'none';
                correctedText.style.display = 'block';
            }
        })
        .catch(error => {
            console.error('Error checking Flask status:', error);
        });
}

// Set an interval to check the status every 2 seconds
setInterval(checkFlaskStatus, 2000);
