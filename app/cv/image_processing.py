import base64
import random
from time import perf_counter
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from config import config
from cv.math_processor import MathProcessor


class BaseImageProcessor:
    def get_img_base64(self, img: np.ndarray):
        _, img_encoded = cv2.imencode(".jpeg", img)
        img_base64 = base64.b64encode(img_encoded)
        img_base64 = "data:img/jpeg;base64," + img_base64.decode("utf-8")
        return img_base64


class ImageGenerator(BaseImageProcessor):
    ELEMENTS_COLOR = (0, 0, 0)

    def __init__(self):
        self.img_width = config.IMG_SIZE
        self.img_height = config.IMG_SIZE

    def _generate_random_lines(self, lines_numb: int) -> List[Tuple[Tuple[int, int]]]:
        """
        Generate random lines with specified parameters within the image boundaries.

        Args:
            lines_numb (int): The number of random lines to generate.

        Returns:
            List[Tuple[Tuple[int, int]]]: A list of tuples, each containing two tuples representing
            the coordinates of the start and end points of a randomly generated line segment.
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

            lines.append(((x1, y1), (x2, y2)))
        return lines

    def _generate_square_points(self, square_size: int) -> Tuple[Tuple[int, int]]:
        """
        Generate random coordinates for a square within the image canvas.

        Args:
            square_size (int): The size of the square's sides.

        Returns:
            Tuple[Tuple[int, int]]: A tuple containing two tuples, each containing (x, y) coordinates
            representing the top-left and bottom-right corners of the randomly positioned square.
        """
        shift = int(square_size * 1.2)  # A square to be within image canvas
        x_square = random.randint(0, self.img_width - shift)
        y_square = random.randint(0, self.img_height - shift)
        point1 = (x_square, y_square)
        point2 = (x_square + square_size, y_square + square_size)
        return point1, point2

    def _add_salt_and_pepper_noise(
        self, img: np.ndarray, salt_prob: float = 0.1, pepper_prob: float = 0.06
    ) -> np.ndarray:
        """
        Add salt and pepper noise to an img.

        Args:
            img (np.ndarray): The input img.
            salt_prob (float): Probability of adding salt noise.
            pepper_prob (float): Probability of adding pepper noise.

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
    ) -> np.ndarray:
        """
        Generate a noisy image with random straight lines and a filled square.

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
            cv2.line(img, line[0], line[1], self.ELEMENTS_COLOR, line_thickness)

        square_points = self._generate_square_points(square_size)
        cv2.rectangle(
            img,
            square_points[0],
            square_points[1],
            self.ELEMENTS_COLOR,
            -1,
        )
        img = self._add_salt_and_pepper_noise(img)

        return img


class SquareDetector(BaseImageProcessor):
    """
    A class for detecting and marking a square in an image.

    Attributes:
        COLOR_RESULT (tuple): The color for marking the detected square in RGBA format.

    Methods:
        find_square(img: np.ndarray) -> str:
            Detects a square in the input image and returns the result as a base64-encoded image.

    """

    COLOR_RESULT = (255, 162, 38, 128)
    COLOR_RESULT = (0, 92, 250, 255)

    def __init__(self) -> None:
        super().__init__()
        self.math_processor = MathProcessor()

    def _remove_noise(self, img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Remove salt and pepper noise from an image using median filtering.

        Args:
            image (np.ndarray): The noisy input image.
            kernel_size (int): The size of the median filter kernel. Default is 3.

        Returns:
            np.ndarray: The cleaned image.
        """
        cleaned_image = cv2.medianBlur(img, kernel_size)
        cleaned_image = cv2.medianBlur(cleaned_image, kernel_size)

        kernel = np.ones((5, 5), np.uint8)
        eroded_image = cv2.erode(cleaned_image, kernel, iterations=1)
        dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

        return dilated_image

    def _get_lines_intersections(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find intersections of lines detected in the input image using the Hough Line Transform.

        Args:
            img (np.ndarray): Input image as a NumPy array.

        Returns:
            List[Tuple[int, int]]: A list of tuples containing (x, y) coordinates of intersections.
        """
        img_canny = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            img_canny, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=2
        )

        height, width = img.shape
        intersections = self.math_processor.find_intersections(lines, width, height)

        # img_res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(img_res, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # for point in intersections:
        #     cv2.circle(img_res, point, 2, (0, 255, 0), 1)
        # cv2.imshow("test", img_res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return intersections

    def _get_square_vertices(
        self, img: np.ndarray, ransac_iterations: int
    ) -> List[Tuple[int, int]]:
        """
        Get the vertices of the square region of interest (ROI) in the image.

        Args:
            img (np.ndarray): The input grayscale image.
            img_res (np.ndarray): The result image where vertices will be marked.

        Returns:
            tuple: A tuple containing two elements:
                1. list: List of four vertices of the square ROI.
                2. np.ndarray: The result image with vertices marked.
        """
        intersections = self._get_lines_intersections(img)
        square_vertices = self.math_processor.get_vertices_ransac(
            img,
            intersections=intersections,
            ransac_iterations=ransac_iterations,
        )

        return square_vertices

    def _draw_result(
        self, img: np.ndarray, verticies: Sequence[Tuple[int, int]]
    ) -> str:
        """
        Draw the result on the input image by marking the detected vertices with circles.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            verticies (Sequence[Tuple[int, int]]): A sequence of (x, y) coordinates of vertices.

        Returns:
            str: Base64-encoded image result as a string.
        """
        img_res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        for point in verticies:
            x, y = point
            cv2.circle(img_res, (x, y), 5, self.COLOR_RESULT, -1)
        verticies = np.array(verticies, np.int32)
        verticies = verticies.reshape((-1, 1, 2))
        cv2.polylines(img_res, [verticies], True, self.COLOR_RESULT, 2)
        return img_res

    def find_square(
        self, img: np.ndarray, ransac_iterations: int
    ) -> Tuple[np.ndarray, int]:
        """
        Find a square in the input image and return a new image with the square outlined.

        Args:
            img (np.ndarray): Input image as a NumPy array.

        Returns:
            str: Base64-encoded image result as a string.

        This method processes the input image to detect a square, removes noise, thresholds
        the image, identifies the square's vertices, and outlines the square in the result image.
        The result image is encoded as a base64 string and returned.
        """
        t_start = perf_counter()
        img_cleaned = self._remove_noise(img)
        _, img_thr = cv2.threshold(img_cleaned, 128, 255, cv2.THRESH_BINARY)
        verticies = self._get_square_vertices(img_thr, ransac_iterations)
        t_stop = perf_counter()
        elapsed_time = int((t_stop - t_start) * 1000)  # ms

        if not verticies:
            return None, elapsed_time

        img_res = self._draw_result(img, verticies)
        return img_res, elapsed_time


image_generator = ImageGenerator()
square_detector = SquareDetector()
