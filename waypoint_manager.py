from shapely.geometry import Point, Polygon, LineString
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from polygon_geofence import PolygonGeofence

class WaypointManager:
    def __init__(self, waypoints: list[tuple[float, float, float]]):
        self.waypoints = np.array(waypoints)

    def generate_spline(self, segment_distance=1.0):
        """
        Generate a cubic spline ensuring equal arc-length spacing.
        Returns both the spline points and the indices of points corresponding to original waypoints.
        """
        waypoints = self.waypoints
        t_original = np.linspace(0, 1, len(waypoints))
        
        # Create cubic splines for each coordinate
        cs_x = CubicSpline(t_original, waypoints[:, 0])
        cs_y = CubicSpline(t_original, waypoints[:, 1])
        cs_z = CubicSpline(t_original, waypoints[:, 2])

        # Compute total arc length of the spline
        total_length = self.get_spline_distance(cs_x, cs_y, cs_z)
        num_points = max(int(total_length / segment_distance), 2)

        # Generate points with equal arc length spacing
        spline_points = self.get_equally_spaced_points(cs_x, cs_y, cs_z, num_points)

        # Find indices of generated points closest to original waypoints
        waypoint_indices = self.find_closest_indices(waypoints, spline_points)

        return spline_points, waypoint_indices

    def get_spline_distance(self, spline_x, spline_y, spline_z):
        """
        Computes the total arc length of the spline.
        """
        dx = spline_x.derivative()
        dy = spline_y.derivative()
        dz = spline_z.derivative()

        def integrand(t):
            return np.sqrt(dx(t) ** 2 + dy(t) ** 2 + dz(t) ** 2)

        length, _ = quad(integrand, 0, 1)
        return length

    def get_equally_spaced_points(self, cs_x, cs_y, cs_z, num_points):
        """
        Generate points along the spline with equal arc-length spacing.
        """
        arc_lengths = np.linspace(0, self.get_spline_distance(cs_x, cs_y, cs_z), num_points)
        t_vals = [0]  # Start at t=0

        # Find t-values that correspond to evenly spaced arc lengths
        for i in range(1, num_points):
            target_length = arc_lengths[i]

            def objective(t):
                return self.arc_length(cs_x, cs_y, cs_z, 0, t) - target_length

            # root_scalar finds the value of t for objective(t) = 0
            sol = root_scalar(objective, bracket=[t_vals[-1], 1], method="brentq")
            t_vals.append(sol.root)

        # Compute the actual coordinates
        x_vals, y_vals, z_vals = cs_x(t_vals), cs_y(t_vals), cs_z(t_vals)
        return [(x, y, z) for x, y, z in zip(x_vals, y_vals, z_vals)]

    def arc_length(self, spline_x, spline_y, spline_z, t0, t1):
        """
        Compute arc length of the spline between two t-values.
        """
        dx, dy, dz = spline_x.derivative(), spline_y.derivative(), spline_z.derivative()

        def integrand(t):
            return np.sqrt(dx(t) ** 2 + dy(t) ** 2 + dz(t) ** 2)

        length, _ = quad(integrand, t0, t1)
        return length

    def find_closest_indices(self, waypoints, spline_points):
        """
        Find indices in spline_points that correspond to the original waypoints.
        """
        waypoint_indices = []
        for wp in waypoints:
            best_idx = min(range(len(spline_points)), key=lambda i: np.linalg.norm(np.array(spline_points[i]) - wp))
            waypoint_indices.append(best_idx)
        return waypoint_indices

    
    def visualize(self, points: list[tuple[float, float, float]], subplot_ax):
        """
        Visualizes the trajectory with alternating colors for better segment visualization.
        """
        points = np.array(points)
        x_points, y_points = points[:, 0], points[:, 1]

        # Define a set of alternating colors
        colors = ['blue', 'red']

        # Plot each segment separately to apply different colors
        for i in range(len(points) - 1):
            subplot_ax.plot(
                [x_points[i], x_points[i+1]], 
                [y_points[i], y_points[i+1]], 
                color=colors[i % len(colors)], 
                linewidth=2
            )

        # Plot waypoints
        waypoints = self.waypoints
        x_waypoints, y_waypoints = waypoints[:, 0], waypoints[:, 1]
        subplot_ax.scatter(x_waypoints, y_waypoints, color='green', marker='o', label="Waypoints")

        subplot_ax.legend()

    def get_collision_pairs(self, path: list[tuple[float, float, float]], fence: Polygon) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        """
        Returns a list of tuple pairs containing the indicies of a medium point whose lines
        intersect the passed in geofence.
        """
        pairs = []
        for i in range(len(path) - 1):
            line = LineString([(float(path[i][0]), float(path[i][1])), (float(path[i+1][0]), float(path[i+1][1]))])
            vertices = fence.exterior.coords[:]
            for j in range(len(vertices) - 1):
                line2 = LineString([vertices[j], vertices[j + 1]])
                p = line.intersects(line2)
                if p:
                    pt = line.intersection(line2)
                    pt = (pt.x, pt.y)
                    pairs.append((path[i], path[i+1], pt))
                
        return pairs
    
    def get_better_path(self, path: list[tuple[float, float, float]], fence: PolygonGeofence):
        pairs = self.get_collision_pairs(path, fence.polygon)
        
        fence_fixes = fence.get_fence_path(fence.polygon, pairs)
        print(fence_fixes)
        better_path = []
        replace = False
        for pt in path:
            if len(pairs) > 1 and pt is pairs[0][0]:
                fix = fence_fixes.pop(0)
                better_path = better_path + [(_pt[0], _pt[1], pt[2]) for _pt in fix]
                replace = True
            elif len(pairs) > 1 and pt is pairs[1][1]:
                
                replace = False
                pairs.pop(0)
                pairs.pop(0)
            elif not replace:
                better_path.append(pt)
        return better_path
    
if __name__ == "__main__":
    vertices = [(0, 0), (4, 0), (4, 4), (3.3, 3.3), (3.2, 2.7), (3.1, 3.1), (2, 2), (0, 4)]  # Example geofence with an inward cut
    geofence = PolygonGeofence(vertices)

    # Waypoints inside the polygon
    waypoints = [(0.5, 2.5, 0), (3, 2.5, 0), (3.5, 3.0, 0), (3.0, 2.7, 0.0)]
    manager = WaypointManager(waypoints)

    fig, ax = geofence.visualize()
    spline, wp_indicies = manager.generate_spline(0.1)
    better = manager.get_better_path(spline, geofence)
    manager.visualize(better, ax)
    plt.show()