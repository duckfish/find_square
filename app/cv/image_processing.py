import base64
import random
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from math_processor import MathProcessor


class BaseImageProcessor:
    def get_img_base64(self, img: np.ndarray):
        _, img_encoded = cv2.imencode(".jpeg", img)
        img_base64 = base64.b64encode(img_encoded)
        img_base64 = "data:img/jpeg;base64," + img_base64.decode("utf-8")
        return img_base64


class ImageGenerator(BaseImageProcessor):
    ELEMENTS_COLOR = (0, 0, 0)

    def __init__(self):
        self.img_width = 1000
        self.img_height = 1000

    def _generate_random_lines(self, lines_numb: int) -> List[tuple]:
        lines = []
        for _ in range(lines_numb):
            # Randomly generate line parameters (slope and y-intercept)
            k = random.uniform(-2, 2)  # Adjust the range as needed
            b = random.uniform(0, self.img_height)

            # Calculate points where the line crosses the image boundaries
            x1 = -100
            y1 = int(k * x1 + b)
            x2 = self.img_width + 100
            y2 = int(k * x2 + b)

            lines.append(((x1, y1), (x2, y2)))
        return lines

    def _generate_square_points(self, square_size: int) -> Tuple[tuple]:
        # Randomly generate square parameters (position and size)
        shift = int(square_size * 1.2)  # A square to be within image canvas
        x_square = random.randint(0, self.img_width - shift)
        y_square = random.randint(0, self.img_height - shift)
        point1 = (x_square, y_square)
        point2 = (x_square + square_size, y_square + square_size)
        return point1, point2

    def _add_gaussian_noise(self, img: np.ndarray, mean=0, std=25):
        """
        Add Gaussian noise to an image.

        Args:
            image (numpy.ndarray): The input image.
            mean (int): Mean of the Gaussian noise. Default is 0.
            std (int): Standard deviation of the Gaussian noise. Default is 25.

        Returns:
            numpy.ndarray: The image with added Gaussian noise.
        """
        noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
        img_noisy = cv2.add(img, noise)
        return img_noisy

    def _add_salt_and_pepper_noise(self, img, salt_prob=0.1, pepper_prob=0.06):
        """
        Add salt and pepper noise to an img.

        Args:
            img (numpy.ndarray): The input img.
            salt_prob (float): Probability of adding salt noise.
            pepper_prob (float): Probability of adding pepper noise.

        Returns:
            numpy.ndarray: The img with added salt and pepper noise.
        """
        noisy_img = np.copy(img)
        total_pixels = img.size

        # Add salt noise
        num_salt = int(total_pixels * salt_prob)
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
        noisy_img[salt_coords[0], salt_coords[1]] = 255

        # Add pepper noise
        num_pepper = int(total_pixels * pepper_prob)
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
        noisy_img[pepper_coords[0], pepper_coords[1]] = 0

        return noisy_img

    def generate_img(self, square_size: int, lines_numb: int, line_thickness: int):
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
    COLOR_RESULT = (255, 0, 0)
    NEUTRAL_4 = (255, 162, 38, 128)

    def __init__(self) -> None:
        super().__init__()
        self.math_processor = MathProcessor()

    def _remove_noise(self, img, kernel_size=3):
        """
        Remove salt and pepper noise from an image using median filtering.

        Args:
            image (numpy.ndarray): The noisy input image.
            kernel_size (int): The size of the median filter kernel. Default is 3.

        Returns:
            numpy.ndarray: The cleaned image.
        """
        cleaned_image = cv2.medianBlur(img, kernel_size)

        return cleaned_image

    def _get_lines_intersections(self, img: np.ndarray) -> dict:
        img_canny = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            img_canny, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10
        )
        height, width = img.shape
        intersections = self.math_processor.find_intersections(lines, width, height)
        # intersections = self.math_processor.split_points_by_quadrant(
        #     intersections, width, height
        # )
        # img_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # for _, line in enumerate(lines):
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(img_lines, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

        # for point in intersections:
        #     cv2.circle(img_lines, point, 2, (255, 0, 0), -1)

        # cv2.imshow("lines", img_lines)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return intersections

    def _get_square_vertices(
        self,
        img: np.ndarray,
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
        self.intersections = self._get_lines_intersections(img)
        square_vertices, _ = self.math_processor.get_vertices_ransac(
            img,
            intersections=self.intersections,
            ransac_iterations=100000,
        )

        return square_vertices

    def _draw_result(self, img: np.ndarray, verticies: Sequence[Tuple[int, int]]):
        img_res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        for point in verticies:
            x, y = point
            cv2.circle(img_res, (x, y), 5, self.NEUTRAL_4, -1)
        for point in verticies:
            x, y = point
            cv2.circle(img_res, (x, y), 5, self.NEUTRAL_4, -1)
        # cv2.imshow("img", img_res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # img_res = cv2.cvtColor(img_res, cv2.COLOR_BGRA2RGBA)
        # img_res = self.get_img_base64(img_res)
        return img_res

    def find_square(self, img):
        img_cleaned = self._remove_noise(img)
        _, img_thr = cv2.threshold(img_cleaned, 128, 255, cv2.THRESH_BINARY)
        verticies = self._get_square_vertices(img_thr)

        img_res = self._draw_result(img, verticies)
        return img_res


image_generator = ImageGenerator()
square_detector = SquareDetector()
