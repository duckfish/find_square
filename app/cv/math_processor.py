import math
import random

import cv2
import numpy as np

from .models import Point


class MathProcessor:
    """A utility class that provides various mathematical operations for image processing tasks."""

    def _get_random_quad(self, intersections: list[Point]) -> list[Point] | None:
        """Generate a random quadrilateral (quad) by selecting random points.

        Args:
            intersections (list[Point]): A list of lines intersections.

        Returns:
            list[Point] | None: A list of four points, each containing
            (x, y) coordinates of a randomly selected point from the input list,
            or None if there are not enough points for random sampling.
        """
        try:
            quad = random.sample(intersections, 4)
        except ValueError:
            return None
        return quad

    def _perpendicular_lines(
        self, line1: np.ndarray, line2: np.ndarray, tolerance: float = 0.5
    ) -> bool:
        """Check if two line segments are approximately perpendicular.

        Args:
            line1: Endpoints of first line segment as (x1, y1, x2, y2)
            line2: Endpoints of second line segment as (x1, y1, x2, y2)
            tolerance: Maximum angular deviation from 90° to consider perpendicular.
                Defaults to 0.5 degrees.

        Returns:
            True if angle difference is within tolerance of 90°, False otherwise.
        """
        x1, y1, x2, y2 = line1
        angle1 = math.atan2(y2 - y1, x2 - x1)
        x1, y1, x2, y2 = line2
        angle2 = math.atan2(y2 - y1, x2 - x1)

        angle_diff = abs(angle1 - angle2)

        return abs(90 - math.degrees(angle_diff)) < tolerance

    def _calculate_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Point | None:
        """Calculate the intersection point of two lines.

        Args:
            line1: Endpoints of first line segment as (x1, y1, x2, y2)
            line2: Endpoints of second line segment as (x1, y1, x2, y2)

        Returns:
            Point | None: A tuple containing the (x, y) coordinates of
                the intersection point if the line segments intersect, or None if
                they are parallel or coincident.
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

        return Point(intersection_x, intersection_y)

    def find_intersections(self, lines: np.ndarray, width: int, height: int) -> list[Point]:
        """Find intersections between lines and filter points within the image canvas.

        Args:
            lines (np.ndarray): An array of line segments represented as numpy arrays,
                where each row contains four values (x1, y1, x2, y2) representing
                the coordinates of the endpoints of a line segment.
            width (int): The width of the image canvas.
            height (int): The height of the image canvas.

        Returns:
            list[Point]: A list of Point objects representing the intersection points
            between line segments, filtered to include only points within the
            image canvas boundaries.
        """
        lines = lines.squeeze(axis=1)
        intersections = []
        for i, line1 in enumerate(lines):
            for line2 in lines[i + 1 :]:
                intersection = self._calculate_intersection(line1, line2)
                if (
                    intersection is not None
                    and 0 <= intersection.x < width
                    and 0 <= intersection.y < height
                ):
                    intersections.append(intersection)

        return intersections

    def _equal_sides(self, sides: list[float], tolerance: float = 0.005) -> bool:
        """Check whether all sides are approximately equal within a certain tolerance.

        Args:
            sides (list[float]): A list of side lengths.
            tolerance (float): Tolerance value for checking the equality of sides.
            Defaults to 0.005.

        Returns:
            bool: True if all sides are approximately equal within the specified
            tolerance, False otherwise.
        """
        reference_distance = np.mean(sides)
        return all(math.isclose(reference_distance, dist, rel_tol=tolerance) for dist in sides)

    def _is_square(self, quad: list[Point]) -> tuple[bool, float]:
        """Check if the given quadrilateral is a square and calculate the maximum error.

        Args:
            quad (list[Point]): A list of four tuples, each containing (x, y)
                coordinates representing the vertices of the quadrilateral.

        Returns:
            tuple[bool, float]: A tuple containing:
                - A boolean indicating whether the quadrilateral is a square.
                - The maximum error in the distances between vertices.
        """
        distances = []
        for i in range(len(quad)):
            for j in range(i + 1, len(quad)):
                point1, point2 = quad[i], quad[j]
                distance = math.hypot(point1.x - point2.x, point1.y - point2.y)
                distances.append(distance)

        diagonal1 = max(distances)
        distances.remove(diagonal1)
        diagonal2 = max(distances)
        distances.remove(diagonal2)

        error_max = 0
        if abs(diagonal1 - diagonal2) > diagonal1 * 0.01:
            return False, error_max

        return self._equal_sides(distances), error_max

    def _sort_quad(self, quad: list[Point]) -> list[Point]:
        """Sort the vertices of a quadrilateral in a clockwise (CW) order.

        Args:
            quad (list[Point]): A list of four tuples, each containing (x, y)
                coordinates representing the vertices of the quadrilateral.

        Returns:
            list[Point]: The sorted vertices of the quadrilateral
                in clockwise order.
        """
        centroid = np.mean(quad, axis=0)
        quad_sorted = sorted(quad, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))
        return quad_sorted

    def _mask_image(self, img: np.ndarray, quad: list[Point]) -> np.ndarray:
        """Apply a mask to an image, masking out a specified quadrilateral region.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            quad (list[Point]): A list of four tuples, each containing (x, y)
                coordinates representing the vertices of the quadrilateral.

        Returns:
            np.ndarray: The masked image with the specified quadrilateral region
                filled with black (0) pixels.
        """
        mask = np.ones_like(img) * 255
        cv2.fillPoly(mask, [np.array(quad, dtype=np.int32)], color=(0, 0, 0))
        img_masked = cv2.bitwise_or(img, mask)
        return img_masked

    def _count_black_pixels(self, quad: list[Point], img: np.ndarray) -> tuple[bool, int]:
        """Count the number of black pixels within a quadrilateral region of an image.

        Args:
            quad (list[Point]): A list of four tuples, each containing
                (x, y) coordinates representing the vertices of the quadrilateral.
            img (np.ndarray): Input image as a NumPy array.

        Returns:
            tuple[bool, int]: A tuple with a boolean value indicating whether
                the square is fake, and the count of black pixels within the specified
                quadrilateral region of the image.
        """
        quad_sorted = self._sort_quad(quad)

        point1, point2 = quad_sorted[0], quad_sorted[1]
        side_length = math.hypot(point1.x - point2.x, point1.y - point2.y)
        footage = side_length**2

        img_masked = self._mask_image(img, quad_sorted)
        black_pixels = np.count_nonzero(img_masked == 0)

        fake_square = False
        if black_pixels < 0.98 * footage:
            fake_square = True

        return fake_square, black_pixels

    def get_vertices_ransac(
        self,
        img: np.ndarray,
        intersections: list[Point],
        ransac_iterations: int = 1000,
    ) -> list[Point] | None:
        """Use the RANSAC algorithm to estimate the vertices of a square in the image.

        Args:
            img (np.ndarray): Input image as a NumPy array.
            intersections (list[Point]): A list of intersections.
            ransac_iterations (int, optional): The number of RANSAC iterations
                to perform. Default is 1000.

        Returns:
            list[Point] | None: The estimated square vertices as a list
                of (x, y) coordinates, or None if no square is found.
        """
        best_square_vertices = None
        black_pixels_max = 0

        for _ in range(ransac_iterations):
            # Randomly sample four intersections
            quad = self._get_random_quad(intersections)
            if quad:
                is_square, _ = self._is_square(quad)
                if is_square:
                    # Calculate the black pixels
                    fake_square, black_pixels = self._count_black_pixels(quad, img)
                    if not fake_square and black_pixels > black_pixels_max:
                        best_square_vertices = quad
                        black_pixels_max = black_pixels

        if best_square_vertices:
            return self._sort_quad(best_square_vertices)
