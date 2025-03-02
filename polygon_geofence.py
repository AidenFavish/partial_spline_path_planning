from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt

class PolygonGeofence:
    def __init__(self, vertices):
        """
        Initialize the geofence with a list of (x, y) tuples.
        """
        self.polygon = Polygon(vertices)
        self.verticies = vertices

    def visualize(self) -> tuple:
        """
        Visualize the polygon in 2D. Returns (figure, axis) subplot.
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
    
    def get_verticies(self, fence: Polygon=None):
        if fence is None:
            fence = self.polygon
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