from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

class PolygonGeofence:
    def __init__(self, vertices):
        """
        Initialize the geofence with a list of (x, y) tuples.
        """
        self.polygon = Polygon(vertices)

    def visualize(self) -> tuple:
        """
        Visualize the polygon in 2D. Returns subplot.
        """
        fig, ax = plt.subplots()
        x, y = self.polygon.exterior.xy
        ax.plot(x, y, 'b-', label='Geofence', color='orange')
        ax.fill(x, y, 'orange', alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Local map')
        ax.grid()
        return (fig, ax)
    
class WaypointManager:
    def __init__(self, waypoints: tuple[float, float, float]):
        self.waypoints = waypoints
        self.longest_distance = self._get_longest_distance(self.waypoints)

    def generate_spline(self, waypoints, segment_distance=10.0) -> list[tuple[float, float, float]]:
        """
        Generate a cubic spline trajectory from the first waypoint to the last.
        """
        waypoints = np.array(waypoints)
        t = np.linspace(0, 1, len(waypoints))
        cs_x = CubicSpline(t, waypoints[:, 0])
        cs_y = CubicSpline(t, waypoints[:, 1])
        cs_z = CubicSpline(t, waypoints[:, 2])
        length = self.get_spline_distance(cs_x, cs_y, cs_z)
        t_vals = np.linspace(0, 1, int(length / segment_distance))
        x_vals, y_vals, z_vals = cs_x(t_vals), cs_y(t_vals), cs_z(t_vals)
        return [(x, y, z) for x, y, z in zip(x_vals, y_vals, z_vals)]

    def _get_longest_distance(self, waypoints):
        """
        Gets longest distance between points.
        """
        longest = 0.0
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i+1]
            dist = (wp1[0] - wp2[0]) ** 2 + (wp1[1] - wp2[1]) ** 2 + (wp1[2] - wp2[2]) ** 2
            if dist > longest:
                longest = dist
        return longest ** 0.5

    def get_spline_distance(self, spline_x: CubicSpline, spline_y: CubicSpline, spline_z: CubicSpline):
        dx = spline_x.derivative()
        dy = spline_y.derivative()
        dz = spline_z.derivative()

        def integrand(x):
            return np.sqrt(dx(x) ** 2 + dy(x) ** 2 + dz(x) ** 2)

        length, _ = quad(integrand, spline_x.x[0], spline_x.x[-1])
        return length
    
    def visualize(self, points: list[tuple[float, float, float]], subplot_ax):
        points = np.array(points)
        x_points, y_points = points[:, 0], points[:, 1]
        subplot_ax.plot(x_points, y_points, "g-", color='blue', label='Planned trajectory')

        waypoints = np.array(self.waypoints)
        x_waypoints, y_waypoints = waypoints[:, 0], waypoints[:, 1]
        subplot_ax.scatter(x_waypoints, y_waypoints, color='green', label="Waypoints")

        subplot_ax.legend()

vertices = [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)]  # Example geofence with an inward cut
geofence = PolygonGeofence(vertices)

# Waypoints inside the polygon
waypoints = [(0.5, 2.5, 0), (3, 2.5, 0), (3.5, 3, 0)]
manager = WaypointManager(waypoints)

fig, ax = geofence.visualize()
manager.visualize(manager.generate_spline(manager.waypoints, 0.1), ax)
plt.show()