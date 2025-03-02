import numpy as np
from shapely.geometry import Polygon, Point
import random
import matplotlib.pyplot as plt
from polygon_geofence import PolygonGeofence
from waypoint_manager import WaypointManager

def generate_points_in_polygon(polygon: Polygon, num_points: int) -> list[tuple[float, float, float]]:
    """
    Generates a list of 3D points where (x, y) values are within a given polygon, and z is always 0.

    Args:
        polygon: A Shapely Polygon object defining the valid area.
        num_points: Number of points to generate.

    Returns:
        A list of (x, y, z) tuples where (x, y) are inside the polygon and z = 0.
    """
    min_x, min_y, max_x, max_y = polygon.bounds  # Get bounding box
    points = []

    while len(points) < num_points:
        x, y = np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)
        if polygon.contains(Point(x, y)):  # Check if point is inside polygon
            points.append((x, y, 0.0))

    return points

if __name__ == "__main__":
    vertices = [(0, 0), (4, 0), (4, 4), (3.3, 3.3), (3.2, 2.7), (3.1, 3.1), (2, 2), (0, 4)]  # Example geofence with an inward cut
    geofence = PolygonGeofence(vertices)

    # Waypoints inside the polygon
    waypoints = generate_points_in_polygon(geofence.polygon, 10)
    manager = WaypointManager(waypoints)

    fig, ax = geofence.visualize()
    spline, wp_indicies = manager.generate_spline(0.1)
    better = manager.get_better_path(spline, geofence, 0.15)
    manager.visualize(better, ax)
    plt.show()
