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
        self.verticies = vertices

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
    
    def get_verticies(self, fence: Polygon):
        return fence.exterior.coords[:]
    
    def get_wedged_index(self, fence: Polygon, point: tuple[float, float]) -> int:
        p = Polygon([(0.01 + point[0], point[1]), (point[0], 0.01 + point[1]), (-0.01 + point[0], point[1]), (point[0], -0.01 + point[1])])
        vertices = self.get_verticies(fence)
        for i in range(len(vertices) - 1):
            line = LineString([vertices[i], vertices[i+1]])
            if p.intersects(line):
                return i + 1
            
    def get_fence_path_direction(self, fence: list[tuple[float, float]], start:int, end:int, dir=0) -> tuple[float, list[tuple[float, float]]]:
        if dir:
            step = 1
        else:
            step = -1

        idx = start
        next = idx
        dist = 0.0
        path = []
        while next != end or dist == 0.0:
            idx = next
            next = idx + step
            if not(0 <= next < len(fence)):
                next = 0 if step == 1 else len(fence) - 1
            
            dist += ((fence[idx][0] - fence[next][0]) ** 2 + (fence[idx][1] - fence[next][1]) ** 2) ** 0.5
            path.append(fence[idx])
        path.append(fence[next])
        return (dist, path)


    def get_fence_path(self, fence: Polygon, path: list[tuple[int, int, tuple[float, float]]]) -> list[list[tuple[float, float]]]:
        output = []
        for i in range(0, len(path), 2):
            idx1 = self.get_wedged_index(fence, path[i][2])
            idx2 = self.get_wedged_index(fence, path[i + 1][2])
            verticies = self.get_verticies(fence)[:-1]
            
            if idx1 > idx2:
                verticies.insert(idx1, path[i][2])
                verticies.insert(idx2, path[i + 1][2])
                idx1 += 1
            else:
                verticies.insert(idx2, path[i + 1][2])
                verticies.insert(idx1, path[i][2])
                idx2 += 1
            
            op1 = self.get_fence_path_direction(verticies, idx1, idx2)
            op2 = self.get_fence_path_direction(verticies, idx1, idx2, 1)

            output.append(op1[1] if op1[0] < op2[0] else op2[1])
        return output
            
    
class WaypointManager:
    def __init__(self, waypoints: tuple[float, float, float]):
        self.waypoints = waypoints
        self.longest_distance = self._get_longest_distance(self.waypoints)

    def generate_spline(self, waypoints, segment_distance=1.0) -> tuple[list[tuple[float, float, float]], list[int]]:
        """
        Generate a cubic spline trajectory from the first waypoint to the last.
        Returns both the spline points and the indices of points corresponding to original waypoints.
        """
        waypoints = np.array(waypoints)
        
        # Create the spline parameterized from 0 to 1
        t_original = np.linspace(0, 1, len(waypoints))
        cs_x = CubicSpline(t_original, waypoints[:, 0])
        cs_y = CubicSpline(t_original, waypoints[:, 1])
        cs_z = CubicSpline(t_original, waypoints[:, 2])
        
        # Calculate the total length to determine number of points
        length = self.get_spline_distance(cs_x, cs_y, cs_z)
        
        # Generate evenly spaced points along the spline
        t_vals = np.linspace(0, 1, int(length / segment_distance))
        x_vals, y_vals, z_vals = cs_x(t_vals), cs_y(t_vals), cs_z(t_vals)
        spline_points = [(x, y, z) for x, y, z in zip(x_vals, y_vals, z_vals)]
        
        # Find indices of generated points that best correspond to original waypoints
        waypoint_indices = []
        for wp in waypoints:
            best_idx = 0
            best = (spline_points[0][0] - wp[0]) ** 2 + (spline_points[0][1] - wp[1]) ** 2 + (spline_points[0][1] - wp[1]) ** 2
            for idx, point in enumerate(spline_points):
                dist = (point[0] - wp[0]) ** 2 + (point[1] - wp[1]) ** 2 + (point[1] - wp[1]) ** 2
                if dist < best:
                    best = dist
                    best_idx = idx
            waypoint_indices.append(best_idx)

        return spline_points, waypoint_indices

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

    def lineLineIntersection(self, A, B, C, D):
        # Line AB represented as a1x + b1y = c1
        a1 = B[1] - A[1]
        b1 = A[0] - B[0]
        c1 = a1*(A[0]) + b1*(A[1])
    
        # Line CD represented as a2x + b2y = c2
        a2 = D[1] - C[1]
        b2 = C[0] - D[0]
        c2 = a2*(C[0]) + b2*(C[1])
    
        determinant = a1*b2 - a2*b1
    
        if (determinant == 0):
            # The lines are parallel. This is simplified
            # by returning a pair of FLT_MAX
            return (10**9, 10**9)
        else:
            x = (b2*c1 - b1*c2)/determinant
            y = (a1*c2 - a2*c1)/determinant
            return (x, y)

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
                    p = self.lineLineIntersection((path[i][0], path[i][1]), (path[i+1][0], path[i+1][1]), vertices[j], vertices[j + 1])
                    pairs.append((path[i], path[i+1], p))
                
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


vertices = [(0, 0), (4, 0), (4, 4), (3.3, 3.3), (3.2, 2.7), (3.1, 3.1), (2, 2), (0, 4)]  # Example geofence with an inward cut
geofence = PolygonGeofence(vertices)

# Waypoints inside the polygon
waypoints = [(0.5, 2.5, 0), (3, 2.5, 0), (3.5, 3.0, 0), (3.0, 2.7, 0.0)]
manager = WaypointManager(waypoints)

fig, ax = geofence.visualize()
spline, wp_indicies = manager.generate_spline(manager.waypoints, 0.01)
better = manager.get_better_path(spline, geofence)
manager.visualize(better, ax)
plt.show()
