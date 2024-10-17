document.addEventListener('DOMContentLoaded', function () { //for clickable highlighted corrections
    const checkBox = document.getElementById('check'); 
    const tryNowLink = document.querySelector('.nav ul li a[href="#GrammarChecker"]');

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
    const textInput = grammarTextarea.innerHTML;  // Use textContent instead of innerText for contenteditable div

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
    }, 10000);
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

// Set an interval to check the status every 21seconds
setInterval(checkFlaskStatus, 1000);


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