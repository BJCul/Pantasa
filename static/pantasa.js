document.addEventListener('DOMContentLoaded', function () {
    const checkBox = document.getElementById('check');
    const tryNowLink = document.querySelector('.nav ul li a[href="#GrammarChecker"]');

    tryNowLink.addEventListener('click', function (event) {
        if (checkBox.checked) {
            checkBox.checked = false;
        }
    });
});

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
    const textInput = this.value;

    // Clear previous corrections
    const correctedTextElement = document.getElementById('correctedText');
    correctedTextElement.innerHTML = '';

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

            // Display the corrected sentence
            if (data.corrected_text) {
                correctedTextElement.innerHTML = `<strong>Corrected Sentence:</strong><br>${data.corrected_text}`;
            } else {
                correctedTextElement.innerHTML = 'No corrections made.';
            }

        } catch (error) {
            console.error('Error:', error);
            correctedTextElement.innerHTML = 'Error retrieving data.';
        }
    }, 1000);
});
