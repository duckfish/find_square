import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class MathProcessor:
    def get_random_quad(
        self, intersections: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Generate a random quadrilateral (quad) by selecting one point from each set of points.

        Args:
            intersections (List[Tuple[int, int]]): A list of intersections.

        Returns:
            List[Tuple[int, int]]: A list of four tuples, each containing (x, y) coordinates of a randomly selected points.
        """

        quad = random.sample(intersections, 4)
        return quad

    def _perpendicular_lines(
        self, line1: Sequence[float], line2: Sequence[float], tolerance: float = 0.5
    ) -> bool:
        """
        Check if two line segments are approximately perpendicular.

        Args:
            line1 (Sequence[float]): A sequence of four float values representing the coordinates
                of the endpoints of the first line segment in the format (x1, y1, x2, y2).
            line2 (Sequence[float]): A sequence of four float values representing the coordinates
                of the endpoints of the second line segment in the format (x3, y3, x4, y4).
            tolerance (float, optional): The angular tolerance (in degrees) to consider lines as perpendicular.
                Defaults to 0.5 degrees.

        Returns:
            bool: True if the lines are approximately perpendicular, False otherwise.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        angle1 = np.arctan2(y2 - y1, x2 - x1)
        angle2 = np.arctan2(y4 - y3, x4 - x3)

        angle_diff = np.abs(angle1 - angle2)

        if np.abs(90 - np.degrees(angle_diff)) < tolerance:
            return True
        else:
            return False

    def calculate_intersection(
        self, line1: Sequence[float], line2: Sequence[float]
    ) -> Optional[Tuple[int, int]]:
        """
        Calculate the intersection point of two line segments.

        Args:
            line1 (Sequence[float]): A sequence of four float values representing the coordinates
                of the endpoints of the first line segment in the format (x1, y1, x2, y2).
            line2 (Sequence[float]): A sequence of four float values representing the coordinates
                of the endpoints of the second line segment in the format (x3, y3, x4, y4).

        Returns:
            Optional[Tuple[int, int]]: A tuple containing the (x, y) coordinates of the intersection point
            if the line segments intersect, or None if they are parallel or coincident.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if det == 0:
            return None  # Lines are parallel or coincident

        if not self._perpendicular_lines(line1, line2):
            return None

        intersection_x = int(
            ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        )
        intersection_y = int(
            ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
        )

        return intersection_x, intersection_y

    def find_intersections(
        self, lines: np.ndarray, width: int, height: int
    ) -> List[Tuple[int, int]]:
        """
        Find intersections between line segments and filter points within the image canvas.

        Args:
            lines (np.ndarray): An array of line segments represented as numpy arrays,
                where each row contains four float values (x1, y1, x2, y2) representing
                the coordinates of the endpoints of a line segment.
            width (int): The width of the image canvas.
            height (int): The height of the image canvas.

        Returns:
            List[Tuple[int, int]]: A list of (x, y) coordinates representing the intersection points
            between line segments, filtered to include only points within the image canvas boundaries.
        """
        intersections = []

        for i, line1 in enumerate(lines):
            for line2 in lines[i + 1 :]:
                intersection = self.calculate_intersection(line1[0], line2[0])
                if intersection is not None:
                    x, y = intersection
                    # filtering points inside image canvas
                    if 0 <= x < width and 0 <= y < height:
                        intersections.append(intersection)

        return intersections

    def calculate_distance(
        self, point1: Sequence[float], point2: Sequence[float]
    ) -> float:
        """
        Calculate the Euclidean distance between two points.

        Args:
            point1 (Sequence[float]): A sequence of two float values representing the (x, y) coordinates
                of the first point.
            point2 (Sequence[float]): A sequence of two float values representing the (x, y) coordinates
                of the second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        x1, y1 = point1
        x2, y2 = point2
        distance = math.hypot(x2 - x1, y2 - y1)
        return distance

    def _is_square(self, quad: List[Tuple[int, int]]) -> Tuple[bool, float]:
        """
        Check if the given quadrilateral is a square and calculate the maximum error.

        Args:
            quad (List[Tuple[int, int]]): A list of four tuples, each containing (x, y) coordinates
                representing the vertices of the quadrilateral.

        Returns:
            Tuple[bool, float]: A tuple containing:
                - A boolean indicating whether the quadrilateral is a square.
                - The maximum error in the distances between vertices.
        """
        distances = []
        for i in range(len(quad)):
            for j in range(i + 1, len(quad)):
                distance = self.calculate_distance(quad[i], quad[j])
                distances.append(distance)

        diagonal1 = max(distances)
        distances.remove(diagonal1)
        diagonal2 = max(distances)
        distances.remove(diagonal2)

        error_max = 0
        if abs(diagonal1 - diagonal2) > diagonal1 * 0.05:
            return False, error_max

        for i in range(len(distances)):
            for j in range(i + 1, len(distances)):
                error = abs(distances[i] - distances[j])
                if error > error_max:
                    error_max = error
        if not all(error_max < x * 0.05 for x in distances):
            return False, error_max
        return True, error_max

    def _sort_quad(self, quad: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Sort the vertices of a quadrilateral in a clockwise (CW) order based on their angles to the centroid.

        Args:
            quad (List[Tuple[int, int]]): A list of four tuples, each containing (x, y) coordinates
                representing the vertices of the quadrilateral.

        Returns:
            List[Tuple[int, int]]: The sorted vertices of the quadrilateral in clockwise order.
        """
        centroid = np.mean(quad, axis=0)
        quad_sorted = sorted(
            quad, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0])
        )
        return quad_sorted

    def _mask_image(self, img: np.ndarray, quad: List[Tuple[int, int]]) -> np.ndarray:
        """
        Apply a mask to an image, masking out a specified quadrilateral region.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            quad (List[Tuple[int, int]]): A list of four tuples, each containing (x, y) coordinates
                representing the vertices of the quadrilateral.

        Returns:
            np.ndarray: The masked image with the specified quadrilateral region filled with black (0) pixels.
        """
        mask = np.ones_like(img) * 255
        cv2.fillPoly(mask, [np.array(quad, dtype=np.int32)], color=(0, 0, 0))
        img_masked = cv2.bitwise_or(img, mask)
        return img_masked

    def _count_black_pixels(self, quad: List[Tuple[int, int]], img: np.ndarray) -> int:
        """
        Count the number of black pixels (pixel values equal to 0) within a quadrilateral region of an image.

        Args:
            quad (List[Tuple[int, int]]): A list of four tuples, each containing (x, y) coordinates
                representing the vertices of the quadrilateral.
            img (np.ndarray): Input image as a NumPy array.

        Returns:
            int: The count of black pixels within the specified quadrilateral region of the image.
        """
        quad_sorted = self._sort_quad(quad)

        side_length = self.calculate_distance(quad_sorted[0], quad_sorted[1])
        footage = side_length**2

        img_masked = self._mask_image(img, quad_sorted)
        black_pixels = np.count_nonzero(img_masked == 0)

        fake_square = False
        if black_pixels < 0.9 * footage:
            fake_square = True

        return fake_square, black_pixels

    def get_vertices_ransac(
        self,
        img: np.ndarray,
        intersections: List[Tuple[int, int]],
        ransac_iterations: int = 1000,
    ):
        """
        Use the RANSAC algorithm to estimate the vertices of a square in the image.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            intersections (List[Tuple[int, int]]): A list of intersections.
            ransac_iterations (int, optional): The number of RANSAC iterations to perform. Default is 1000.

        Returns:
            Optional[List[Tuple[int, int]]]: The estimated square vertices as a list of (x, y) coordinates,
            or None if no square is found.
        """
        best_square_vertices = None
        black_pixels_max = 0

        for _ in range(ransac_iterations):
            # Randomly sample four intersections
            quad = self.get_random_quad(intersections)
            if quad:
                is_square, _ = self._is_square(quad)
                if is_square:
                    # Calculate the black pixels
                    fake_square, black_pixels = self._count_black_pixels(quad, img)
                    if not fake_square and black_pixels > black_pixels_max:
                        best_square_vertices = quad
                        black_pixels_max = black_pixels

        return best_square_vertices
