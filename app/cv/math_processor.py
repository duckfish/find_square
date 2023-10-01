import itertools
import logging
import math
import random
from typing import Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("pet")


class MathProcessor:
    def __init__(self):
        self.sampled_points = set()

    # def _calculate_quadrant(
    #     self, point: Tuple[int, int], width: int, height: int
    # ) -> int:
    #     """
    #     Calculate the quadrant of a point relative to the center of a coordinate plane.

    #     Args:
    #         point (Tuple[int, int]): A tuple containing the (x, y) coordinates of the point to be categorized.
    #         width (int): The width of the coordinate plane.
    #         height (int): The height of the coordinate plane.

    #     Returns:
    #         int: An integer representing the quadrant in which the point is located.
    #             - Quadrant 1: Top-left
    #             - Quadrant 2: Top-right
    #             - Quadrant 3: Bottom-right
    #             - Quadrant 4: Bottom-left
    #     """
    #     x, y = point
    #     if x <= (width / 2) and y <= (height / 2):
    #         return 1
    #     elif x > (width / 2) and y <= (height / 2):
    #         return 2
    #     elif x > (width / 2) and y > (height / 2):
    #         return 3
    #     else:
    #         return 4

    # def split_points_by_quadrant(
    #     self, points: List[Tuple[int, int]], width: int, height: int
    # ) -> Dict[int, List[Tuple[int, int]]]:
    #     """
    #     Split a list of points into different quadrants of a coordinate plane.

    #     Args:
    #         points (List[Tuple[int, int]]): A list of tuples, where each tuple contains (x, y) coordinates of a point.
    #         width (int): The width of the coordinate plane.
    #         height (int): The height of the coordinate plane.

    #     Returns:
    #         Dict[int, List[Tuple[int, int]]]: A dictionary where each key represents a quadrant (1, 2, 3, or 4),
    #         and the corresponding value is a list of points belonging to that quadrant.

    #     Quadrant Definitions:
    #         - Quadrant 1: Top-left
    #         - Quadrant 2: Top-right
    #         - Quadrant 3: Bottom-right
    #         - Quadrant 4: Bottom-left
    #     """
    #     quadrants = {1: [], 2: [], 3: [], 4: []}

    #     for point in points:
    #         quadrant = self._calculate_quadrant(point, width, height)
    #         quadrants[quadrant].append(point)

    #     return quadrants

    def get_random_quad(
        self, intersections: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Generate a random quadrilateral (quad) by selecting one point from each set of points.

        Args:
            intersections (List[Tuple[int, int]]): A list of intersections.

        Returns:
            List[Tuple[int, int]]: A list of four tuples, each containing (x, y) coordinates of a randomly selected point from each quadrant.
        """

        # max_iterations = math.comb(len(intersections), 4)
        # print(max_iterations)
        # while True:
        quad = random.sample(intersections, 4)
            # quad = self._sort_quad(quad)
            # Convert the quad to a tuple for hashing
            # quad_tuple = tuple(quad)

            # if quad_tuple not in self.sampled_points:
            #     self.sampled_points.add(quad_tuple)
            #     print(len(self.sampled_points))
                
        return quad
        # return None

    def _perpendicular_lines(
        self, line1: Sequence[float], line2: Sequence[float], tolerance: float = 0.5
    ):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        # Calculate the angles of the lines
        angle1 = np.arctan2(y2 - y1, x2 - x1)
        angle2 = np.arctan2(y4 - y3, x4 - x3)

        # Calculate the absolute difference in angles
        angle_diff = np.abs(angle1 - angle2)

        # Check if the angle difference is within the specified tolerance
        if np.abs(90 - np.degrees(angle_diff)) < tolerance:
            return True
        else:
            return False

    def calculate_intersection(
        self, line1: Sequence[float], line2: Sequence[float]
    ) -> Union[Tuple[int, int], None]:
        # Calculate the intersection point of two line segments
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

        return (intersection_x, intersection_y)

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

    def _is_square(self, quad: List[Tuple[int, int]]):
        # Find all distances between the given four points
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

        # distance_avrg = sum(distances) / len(distances)
        # errors = []

        # for side_length in distances:
        #     error = abs(side_length - distance_avrg)
        #     if error > distance_avrg * 0.01:
        #         return False, error
        #     errors.append(error)

        # if len(errors) != 0:
        #     error = sum(errors) / len(errors)

        for i in range(len(distances)):
            for j in range(i + 1, len(distances)):
                error = abs(distances[i] - distances[j])
                if error > error_max:
                    error_max = error
        if not all(error_max < x * 0.05 for x in distances):
            return False, error_max
        return True, error_max

    def _sort_quad(self, quad):
        # Calculate the centroid of the points
        centroid = np.mean(quad, axis=0)

        # Sort the points based on their angle with respect to the centroid
        quad_sorted = sorted(
            quad, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0])
        )
        return quad_sorted

    def _mask_image(self, img: np.ndarray, quad: List[Tuple[int, int]]):
        # Create a mask
        mask = np.ones_like(img) * 255

        # Fill the polygon on the mask
        cv2.fillPoly(mask, [np.array(quad, dtype=np.int32)], color=(0, 0, 0))

        # Use the mask to combine the white background with the original image
        img_masked = cv2.bitwise_or(img, mask)

        # # Display the result
        # cv2.imshow("Result", img_masked)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img_masked

    def _count_black_pixels(self, quad: List[Tuple[int, int]], img: np.ndarray):
        quad_sorted = self._sort_quad(quad)
        img_masked = self._mask_image(img, quad_sorted)
        black_pixels = np.count_nonzero(img_masked == 0)
        return black_pixels

    def get_vertices_ransac(
        self,
        img,
        intersections: dict,
        ransac_iterations: int = 1000,
    ):
        best_square_vertices = None
        black_pixels_max = 0

        for _ in range(ransac_iterations):
            # Randomly sample four intersections
            quad = self.get_random_quad(intersections)
            if quad:
                is_square, _ = self._is_square(quad)
                if is_square:
                    # Calculate the black pixels
                    black_pixels = self._count_black_pixels(quad, img)
                    if black_pixels > black_pixels_max:
                        best_square_vertices = quad
                        black_pixels_max = black_pixels

        return best_square_vertices
