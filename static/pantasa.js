// Define showSection function in the global scope
function showSection(sectionId) {
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        section.classList.add('hidden');
    });

    const targetSection = document.getElementById(sectionId);
    targetSection.classList.remove('hidden');
}

// Define triggerGrammarCheck in the global scope
function triggerGrammarCheck() {
    const grammarTextarea = document.getElementById('grammarTextarea');
    const textInput = grammarTextarea.textContent.trim();
    const correctedText = document.getElementById('correctedText')

    if (!textInput) {
        console.log("No text to check.");
        return;
    }

    correctedText.innerHTML = ''
    
    // Show loading spinner
    document.getElementById('loading').style.display = 'block';

    fetch('/get_text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text_input: textInput })
    })
    .then(response => {
        if (!response.ok) throw new Error('Server returned an error');
        return response.json();
    })
    .then(data => {
        let highlightedText = textInput;

        // Store the spelling suggestions globally for access later
        window.spellingSuggestions = data.spelling_suggestions || {};

        if (data.incorrect_words) {
            data.incorrect_words.forEach(word => {
                const regex = new RegExp(`\\b${word}\\b`, 'gi');
                highlightedText = highlightedText.replace(regex, `<span class="highlight">${word}</span>`);
            });
        }

        // Display highlighted text
        grammarTextarea.innerHTML = highlightedText;
        document.getElementById('correctedText').textContent = data.corrected_text || "No corrections needed.";
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('correctedText').textContent = 'Error retrieving data.';
    })
    .finally(() => {
        document.getElementById('loading').style.display = 'none';
    });
}

// Define the updateCharacterCount function in the global scope
function updateCharacterCount() {
    const textarea = document.getElementById('grammarTextarea');
    const charCount = document.getElementById('charCount');
    const maxLength = 150;
    let currentLength = textarea.textContent.length;

    // Prevent further input if character limit is reached
    if (currentLength > maxLength) {
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
}

let selectedWordElement = null;

// Highlighted text click handler
document.addEventListener('click', function (event) {
    if (event.target.classList.contains('highlight')) {
        selectedWordElement = event.target;  // Store the clicked word element
        const clickedWord = event.target.textContent;  // Get the clicked word
        const suggestionsList = document.getElementById('suggestionsList');

        // Clear previous suggestions
        suggestionsList.innerHTML = '';

        // Fetch the spelling suggestions for the clicked word from the globally stored object
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
        console.log('Replacing word:', incorrectWord, 'with:', newWord);  // Debugging log

        if (selectedWordElement) {
            selectedWordElement.textContent = newWord;  // Update the displayed word
            selectedWordElement.classList.remove('highlight');  // Remove highlight after correction
        }

        // Clear the suggestions list after a suggestion is clicked
        const suggestionsList = document.getElementById('suggestionsList');
        suggestionsList.innerHTML = '';  // Clear the suggestions

        // Trigger the grammar check again
        triggerGrammarCheck();
    }

    grammarTextarea.addEventListener('keydown', function (event) {
        const maxLength = 150;

        // Prevent typing if character limit is reached
        if (grammarTextarea.textContent.length >= maxLength && event.key !== "Backspace" && event.key !== "Delete") {
            event.preventDefault();
        }
    });

document.addEventListener('DOMContentLoaded', function () {
    const checkBox = document.getElementById('check');
    const tryNowLink = document.querySelector('.nav ul li a[href="#GrammarChecker"]');
    

    // Hide the checkbox when the "Try It Now" link is clicked
    tryNowLink.addEventListener('click', function (event) {
        if (checkBox.checked) {
            checkBox.checked = false;   
        }
    });

    // Function to hide the menu after a link is clicked
    document.querySelectorAll('.nav ul li a').forEach(link => {
        link.addEventListener('click', function () {
            document.getElementById('check').checked = false;
            document.querySelector('.home-content').style.display = 'block';
        });
    });

    // Intersection Observer for sections
    const sections = document.querySelectorAll('section');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                console.log('Visible section:', entry.target.id);  // Logs the visible section
                updateNavbar(entry.target.id);  // Example: Update navbar based on the section ID
            }
        });
    }, { threshold: 0.5 });

    sections.forEach(section => {
        observer.observe(section);
    });

    // Function to update navbar style based on the section
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
