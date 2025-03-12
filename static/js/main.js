const squareSizeInput = document.getElementById("square-size");
const squareSizeIndicator = document.getElementById("square-size-indicator");
squareSizeIndicator.textContent = squareSizeInput.value;

squareSizeInput.addEventListener("input", function () {
  squareSizeIndicator.textContent = squareSizeInput.value;
});

const linesNumberInput = document.getElementById("lines-number");
const linesNumberIndicator = document.getElementById("lines-number-indicator");
linesNumberIndicator.textContent = linesNumberInput.value;

linesNumberInput.addEventListener("input", function () {
  linesNumberIndicator.textContent = linesNumberInput.value;
});

const lineThicknessInput = document.getElementById("line-thickness");
const lineThicknessIndicator = document.getElementById("line-thickness-indicator");
lineThicknessIndicator.textContent = lineThicknessInput.value;

lineThicknessInput.addEventListener("input", function () {
  lineThicknessIndicator.textContent = lineThicknessInput.value;
});

const ransacInput = document.getElementById("ransac");
const ransacIndicator = document.getElementById("ransac-indicator");
ransacIndicator.textContent = ransacInput.value;

ransacInput.addEventListener("input", function () {
  ransacIndicator.textContent = ransacInput.value;
});

const generateButton = document.getElementById("generate-img-button");

generateButton.addEventListener("click", async () => {
  fail.style.display = "none";

  const data = {
    square_size: squareSizeInput.value,
    lines_qty: linesNumberInput.value,
    lines_thickness: lineThicknessInput.value,
  };

  const response = await fetch("image/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (response.ok) {
    const responseData = await response.json();
    htmx.find("#image").src = responseData.img;
  }
});

const findButton = document.getElementById("find-square-button");

const fail = htmx.find("#fail");

findButton.addEventListener("click", async () => {
  fail.style.display = "none";
  const squareNetRadio = htmx.find("#SquareNet-radio");
  let detector = "RANSAC";

  if (squareNetRadio.checked) {
    detector = "SquareNet";
  }

  const data = {
    ransac_iterations: ransacInput.value,
    detector: detector,
  };

  const loadingIndicator = htmx.find("#elapsed-time-indicator");
  let dots = 0;
  const loadingInterval = setInterval(() => {
    loadingIndicator.textContent = `${". ".repeat(dots)}`;
    dots = (dots + 1) % 10;
  }, 300);

  const response = await fetch("image/find-square", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  clearInterval(loadingInterval);

  if (response.ok) {
    const responseData = await response.json();
    const img = responseData.img;
    const success = responseData.success;

    htmx.find("#image").src = responseData.img;
    if (!success) {
      fail.style.display = "block";
    }

    htmx.find("#elapsed-time-indicator").textContent = `${responseData.elapsed_time} ms`;
  }
});
