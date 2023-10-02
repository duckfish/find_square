function generateSessionId() {
    return Math.random().toString(36).substring(2, 10);
}

const sessionId = generateSessionId();

// const squareSizeInput = htmx.find('#square-size-input')

// Get a reference to the range input and the span element to display the value
const squareSizeInput = document.getElementById('square-size');
const squareSizeIndicator = document.getElementById('square-size-indicator');
squareSizeIndicator.textContent = squareSizeInput.value;

// Add an event listener to the range input for the 'input' event
squareSizeInput.addEventListener('input', function() {
    // Update the text content of the span element with the current value of the range input
    squareSizeIndicator.textContent = squareSizeInput.value;
});

// Get a reference to the range input and the span element to display the value
const linesNumberInput = document.getElementById('lines-number');
const linesNumberIndicator = document.getElementById('lines-number-indicator');
linesNumberIndicator.textContent = linesNumberInput.value;

// Add an event listener to the range input for the 'input' event
linesNumberInput.addEventListener('input', function() {
    // Update the text content of the span element with the current value of the range input
    linesNumberIndicator.textContent = linesNumberInput.value;
});

// Get a reference to the range input and the span element to display the value
const linesWidthInput = document.getElementById('line-width');
const linesWidthIndicator = document.getElementById('line-width-indicator');
linesWidthIndicator.textContent = linesWidthInput.value;

// Add an event listener to the range input for the 'input' event
linesWidthInput.addEventListener('input', function() {
    // Update the text content of the span element with the current value of the range input
    linesWidthIndicator.textContent = linesWidthInput.value;
});


const generateButton = document.getElementById("generate-img-button");

generateButton.addEventListener("click", async () => {

    const response = await fetch(`/generate-image?session_id=${sessionId}`, {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded",
        },
        body: `square_size=${squareSizeInput.value}&lines_number=${linesNumberInput.value}&line_width=${linesWidthInput.value}`,
    });

    if (response.ok) {
        const responseData = await response.json();
        htmx.find('#image').src = responseData.img;
        // Display the generated image in your UI using responseData.image
    } else {
        // Handle errors
    }
});

const findButton = document.getElementById("find-square-button");

findButton.addEventListener("click", async () => {

    const response = await fetch(`/find-square?session_id=${sessionId}`, {
        method: "GET",
        headers: {
            "Content-Type": "application/json", // Updated content type
        },
    });

    if (response.ok) {
        const responseData = await response.json();
        htmx.find('#image').src = responseData.img;
        // Display the generated image in your UI using responseData.image
    } else {
        // Handle errors
    }
});