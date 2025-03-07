import base64
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import keras.backend
import keras.models
import numpy as np
from config import config
from cv.math_processor import MathProcessor

from .models import Line, Point


@keras.saving.register_keras_serializable()
def iou_metric(y_true, y_pred):  # noqa: D103
    # Extract coordinates from tensors
    x1, y1, x2, y2 = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    x1_pred, y1_pred, x2_pred, y2_pred = (
        y_pred[:, 0],
        y_pred[:, 1],
        y_pred[:, 2],
        y_pred[:, 3],
    )

    # Calculate intersection coordinates
    x1_intersection = keras.backend.maximum(x1, x1_pred)
    y1_intersection = keras.backend.maximum(y1, y1_pred)
    x2_intersection = keras.backend.minimum(x2, x2_pred)
    y2_intersection = keras.backend.minimum(y2, y2_pred)

    # Calculate intersection area
    intersection_area = keras.backend.maximum(
        0.0, x2_intersection - x1_intersection
    ) * keras.backend.maximum(0.0, y2_intersection - y1_intersection)

    # Calculate union area
    area_true = (x2 - x1) * (y2 - y1)
    area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    union_area = area_true + area_pred - intersection_area

    # Calculate IoU
    iou = intersection_area / (
        union_area + keras.backend.epsilon()
    )  # Adding epsilon to avoid division by zero

    return keras.backend.mean(iou)


class BaseImageProcessor:
    def get_img_base64(self, img: np.ndarray) -> str:
        """Convert the input image array into a base64 encoded JPEG image string.

        Args:
            img (np.ndarray): Input image array.

        Returns:
            str: Base64 encoded image string with the appropriate data URI prefix.
        """
        _, img_encoded = cv2.imencode(".jpeg", img)
        img_base64 = base64.b64encode(img_encoded)  # type: ignore
        img_base64 = "data:img/jpeg;base64," + img_base64.decode("utf-8")
        return img_base64


class ImageGenerator(BaseImageProcessor):
    """Generating noisy images with random straight lines and filled square.

    Attributes:
        img_height (int): The height of the generated image.
        img_width (int): The width of the generated image.

    Methods:
        generate_img(
            self, square_size: int, lines_numb: int, line_thickness: int
        ) -> np.ndarray:
            Generates a noisy image with random straight lines and a filled square.
    """

    ELEMENTS_COLOR = (0, 0, 0)

    def __init__(self, img_size: tuple[int, int]) -> None:
        self.img_width = img_size[0]
        self.img_height = img_size[1]

    def _generate_random_lines(self, lines_numb: int) -> list[Line]:
        """Generate random lines with specified parameters within the image boundaries.

        Args:
            lines_numb (int): The number of random lines to generate.

        Returns:
            List[Tuple[Tuple[int, int]]]: A list of tuples, each containing two tuples
            representing the coordinates of the start and end points of
            a randomly generated line segment.
        """
        lines = []
        for _ in range(lines_numb):
            # Randomly generate line parameters (slope and y-intercept)
            k = random.uniform(-2, 2)
            b = random.uniform(0, self.img_height)

            # Calculate points where the line crosses the image boundaries
            x1 = -100
            y1 = int(k * x1 + b)
            x2 = self.img_width + 100
            y2 = int(k * x2 + b)

            lines.append(Line(x1, y1, x2, y2))
        return lines

    def _generate_square_points(self, square_size: int) -> tuple[Point, Point]:
        """Generate random coordinates for a square within the image canvas.

        Args:
            square_size (int): The size of the square's sides.

        Returns:
            Tuple[Tuple[int, int]]: A tuple containing two points representing
            the top-left and bottom-right corners of the randomly positioned square.
        """
        shift = int(square_size * 1.2)  # A square to be within image canvas
        x_square = random.randint(0, self.img_width - shift)
        y_square = random.randint(0, self.img_height - shift)
        top_left = Point(x_square, y_square)
        bottom_right = Point(x_square + square_size, y_square + square_size)
        return top_left, bottom_right

    def _add_salt_and_pepper_noise(
        self, img: np.ndarray, salt_prob: float = 0.1, pepper_prob: float = 0.06
    ) -> np.ndarray:
        """Add salt and pepper noise to an img.

        Args:
            img (np.ndarray): The input img.
            salt_prob (float): Probability of adding salt noise. Default is 0.1
            pepper_prob (float): Probability of adding pepper noise. Default is 0.06.

        Returns:
            np.ndarray: The img with added salt and pepper noise.
        """
        total_pixels = img.size

        # Add salt noise
        num_salt = int(total_pixels * salt_prob)
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
        img[salt_coords[0], salt_coords[1]] = 255

        # Add pepper noise
        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
        img[pepper_coords[0], pepper_coords[1]] = 0

        return img

    def generate_img(
        self, square_size: int, lines_numb: int, line_thickness: int
    ) -> tuple[np.ndarray, str]:
        """Generate a noisy image with random straight lines and a filled square.

        Args:
            square_size (int): The size of the square to be drawn.
            lines_numb (int): The number of random straight lines to draw.
            line_thickness (int): The thickness of the lines.

        Returns:
            np.ndarray: The generated image as a NumPy array.
        """
        # Create a blank white image
        img = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 255

        lines = self._generate_random_lines(lines_numb)
        for line in lines:
            cv2.line(
                img,
                (line.x1, line.y1),
                (line.x2, line.y2),
                self.ELEMENTS_COLOR,
                line_thickness,
            )

        square_points = self._generate_square_points(square_size)
        cv2.rectangle(
            img,
            square_points[0],
            square_points[1],
            self.ELEMENTS_COLOR,
            -1,
        )

        img = self._add_salt_and_pepper_noise(img)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = Path(f"data/images/{timestamp}.png")
        img_path.parent.mkdir(exist_ok=True, parents=True)

        cv2.imwrite(filename=str(img_path), img=img)
        return img, str(img_path)


class SquareDetector(BaseImageProcessor):
    """A class for detecting and marking a square in an image.

    Methods:
        find_square(img: np.ndarray) -> np.ndarray:
            Detects a square in the input image and returns the result as
            a np.ndarray image.

    """

    COLOR_RESULT = (0, 92, 250)

    def __init__(self, img_size: int, model_path: str) -> None:
        super().__init__()
        self.img_size = img_size
        self.math_processor = MathProcessor()
        self.model = keras.models.load_model(model_path)

    def _remove_noise(
        self, img: np.ndarray, median_kernel_size: int = 3, morph_kernel_size: int = 5
    ) -> np.ndarray:
        """Remove salt and pepper noise from an image.

        Args:
            img (np.ndarray): The noisy input image.
            median_kernel_size (int): The size of the median filter kernel.
            Default is 3.
            morph_kernel_size (int): The size of the kernel for morphological
            operations. Default is 5.

        Returns:
            np.ndarray: The cleaned image.
        """
        img_blurred = cv2.medianBlur(img, median_kernel_size)
        img_blurred = cv2.medianBlur(img_blurred, median_kernel_size)

        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        img_eroded = cv2.erode(img_blurred, kernel, iterations=1)
        img_dilated = cv2.dilate(img_eroded, kernel, iterations=1)

        return img_dilated

    def _get_lines_intersections(self, img: np.ndarray) -> list[Point]:
        """Find intersections of lines detected in the input image.

        Args:
            img (np.ndarray): Input image as a NumPy array.

        Returns:
            List[Point]: A list of tuples containing (x, y) coordinates
            of intersections.
        """
        img_canny = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            img_canny, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=2
        )

        height, width = img.shape
        intersections = self.math_processor.find_intersections(lines, width, height)

        return intersections

    def _get_square_vertices(
        self, img: np.ndarray, ransac_iterations: int
    ) -> list[Point] | None:
        """Gets the vertices of the black square using RANSAC approach.

        Args:
            img: The input grayscale image.
            ransac_iterations: The number of RANSAC iterations.

        Returns:
            A list of four vertices of the black square or None if not found.
        """
        intersections = self._get_lines_intersections(img)
        square_vertices = self.math_processor.get_vertices_ransac(
            img,
            intersections=intersections,
            ransac_iterations=ransac_iterations,
        )

        return square_vertices

    def _preprocess_img_m(self, img: np.ndarray) -> np.ndarray:
        """Preprocess the input image for the SquareNet model.

        Args:
            img (np.ndarray): Input image as a NumPy array.

        Returns:
            np.ndarray: Preprocessed image ready to be fed into the SquareNet model.
        """
        target_img_size = self.model.layers[0].input_shape[1:3]
        img = cv2.resize(img, target_img_size, cv2.INTER_AREA)  # type: ignore
        img = np.expand_dims(
            img, axis=-1
        )  # Add an extra dimension for grayscale channel
        img = np.array([img], dtype="float32") / 255.0
        return img

    def _draw_result(
        self, img: np.ndarray, vertices: list[Point] | np.ndarray
    ) -> np.ndarray:
        """Draw the result on the input image by marking the detected vertices.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            vertices (Sequence[Tuple[int, int]]): A sequence of (x, y) coordinates
            of vertices.
            detector (str): The detector used to identify the shape.
                Options: "RANSAC" for RANSAC-based detection or "SquareNet" for a neural
                network-based detector.

        Returns:
            np.ndarray: Image result.
        """
        img_res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if isinstance(vertices, list):
            vertices = np.array(vertices, np.int32)
            vertices = vertices.reshape((-1, 1, 2))
            cv2.polylines(img_res, [vertices], True, self.COLOR_RESULT, 2)  # type: ignore
        elif isinstance(vertices, np.ndarray):
            vertices = map(round, vertices * self.img_size)  # type: ignore
            x1, y1, x2, y2 = vertices
            cv2.rectangle(img_res, (x1, y1), (x2, y2), self.COLOR_RESULT, 2)  # type: ignore
        return img_res

    def find_square(
        self, img: np.ndarray, ransac_iterations: int, detector: str
    ) -> tuple[np.ndarray | None, int]:
        """Find a square in the input image and return a new image with the square outlined.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            ransac_iterations (int): The number of iterations to perform for the RANSAC
                algorithm or the corresponding parameter for the selected detector.
            detector (str): The detector to be used to identify the square.
                Options: "RANSAC" for RANSAC-based detection or "SquareNet" for
                a neural network-based detector.

        Returns:
            Tuple[Optional[np.ndarray], int]: A tuple containing the resulting image
            with the square outlined and the elapsed time in milliseconds.
        """
        t_start = perf_counter()
        if detector == "RANSAC":
            img_cleaned = self._remove_noise(img)
            _, img_thr = cv2.threshold(img_cleaned, 128, 255, cv2.THRESH_BINARY)
            vertices = self._get_square_vertices(img_thr, ransac_iterations)
        elif detector == "SquareNet":
            img_cleaned = self._remove_noise(img)
            img_preprocessed = self._preprocess_img_m(img_cleaned)
            vertices = self.model.predict(img_preprocessed)[0]
        t_stop = perf_counter()
        elapsed_time = int((t_stop - t_start) * 1000)  # ms

        if vertices is None:
            return None, elapsed_time

        img_res = self._draw_result(img, vertices)
        return img_res, elapsed_time


image_generator = ImageGenerator(img_size=(config.IMG_SIZE, config.IMG_SIZE))
square_detector = SquareDetector(img_size=config.IMG_SIZE, model_path=config.MODEL_PATH)
