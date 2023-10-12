function generateSessionId() {
    return Math.random().toString(36).substring(2, 10);
}

const sessionId = generateSessionId();
let timestamp;

const zoomableImage = document.getElementById('image');
panzoom(zoomableImage, { 
    bounds: true,
    maxZoom: 1.55,
    minZoom: 1,
    zoomDoubleClickSpeed: 1,
  });
// panzoom(zoomableImage);

// const squareSizeInput = htmx.find('#square-size-input')

// Get a reference to the range input and the span element to display the value
const squareSizeInput = document.getElementById('square-size');
const squareSizeIndicator = document.getElementById('square-size-indicator');
squareSizeIndicator.textContent = squareSizeInput.value;

// Add an event listener to the range input for the 'input' event
squareSizeInput.addEventListener('input', function () {
    // Update the text content of the span element with the current value of the range input
    squareSizeIndicator.textContent = squareSizeInput.value;
});

// Get a reference to the range input and the span element to display the value
const linesNumberInput = document.getElementById('lines-number');
const linesNumberIndicator = document.getElementById('lines-number-indicator');
linesNumberIndicator.textContent = linesNumberInput.value;

// Add an event listener to the range input for the 'input' event
linesNumberInput.addEventListener('input', function () {
    // Update the text content of the span element with the current value of the range input
    linesNumberIndicator.textContent = linesNumberInput.value;
});

// Get a reference to the range input and the span element to display the value
const linesWidthInput = document.getElementById('line-width');
const linesWidthIndicator = document.getElementById('line-width-indicator');
linesWidthIndicator.textContent = linesWidthInput.value;

// Add an event listener to the range input for the 'input' event
linesWidthInput.addEventListener('input', function () {
    // Update the text content of the span element with the current value of the range input
    linesWidthIndicator.textContent = linesWidthInput.value;
});

// Get a reference to the range input and the span element to display the value
const ransacInput = document.getElementById('ransac');
const ransacIndicator = document.getElementById('ransac-indicator');
ransacIndicator.textContent = ransacInput.value;

// Add an event listener to the range input for the 'input' event
ransacInput.addEventListener('input', function () {
    // Update the text content of the span element with the current value of the range input
    ransacIndicator.textContent = ransacInput.value;
});


const generateButton = document.getElementById("generate-img-button");

generateButton.addEventListener("click", async () => {
    fail.style.display = 'none';
    timestamp = new Date().getTime();

    const data = {
            _id: timestamp,
            session_id: sessionId,
            square_size: squareSizeInput.value,
            lines_numb: linesNumberInput.value,
            line_thickness: linesWidthInput.value
    }

    const response = await fetch('/generate-image', {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
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

// findButton.addEventListener("click", async () => {

//     const data = {
//             _id: timestamp,
//             ransac_iterations: ransacInput.value
//         };

//     const response = await fetch('/find-square', {
//         method: "POST",
//         headers: {
//             "Content-Type": "application/json",
//         },
//         body: JSON.stringify(data),
//     });

//     if (response.ok) {
//         const responseData = await response.json();
//         htmx.find('#image').src = responseData.img;
//         htmx.find('#elapsed-time-indicator').textContent = `${responseData.elapsed_time} ms`;
//     };
// });

const fail = htmx.find('#fail');

findButton.addEventListener("click", async () => {
    
    fail.style.display = 'none';

    const data = {
        _id: timestamp,
        ransac_iterations: ransacInput.value
    };

    // Display a loading indicator with multiple dots
    const loadingIndicator = htmx.find('#elapsed-time-indicator');
    let dots = 0;
    const loadingInterval = setInterval(() => {
        loadingIndicator.textContent = `${'. '.repeat(dots)}`;
        dots = (dots + 1) % 10; // Cycle through 0, 1, 2, 3
    }, 300); // Adjust the interval duration as needed

    const response = await fetch('/find-square', {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
    });

    clearInterval(loadingInterval); // Clear the loading indicator

    if (response.ok) {
        const responseData = await response.json();
        const img = responseData.img;
        const success = responseData.success;

        htmx.find('#image').src = responseData.img;
        if (!success) {
            fail.style.display = "block";
        };
        
        htmx.find('#elapsed-time-indicator').textContent = `${responseData.elapsed_time} ms`;
    }
});
