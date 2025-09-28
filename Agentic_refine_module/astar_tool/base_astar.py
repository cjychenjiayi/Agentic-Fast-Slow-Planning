import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.patches import Circle, Polygon
import sys
import os
import matplotlib.patheffects as pe


class BASE_ASTAR:
    def __init__(self, points, grid_size):
        self.P1, self.P2, self.P3, self.P4 = points
        self.grid_size = grid_size
        self.origin = np.array(self.P1, dtype=float)
        self.x_dir = np.array(self.P3, dtype=float) - np.array(self.P1, dtype=float)
        self.y_dir = np.array(self.P2, dtype=float) - np.array(self.P1, dtype=float)
        self.x_dir /= np.linalg.norm(self.x_dir)
        self.y_dir /= np.linalg.norm(self.y_dir)

        self.x_length = np.linalg.norm(np.array(self.P3) - np.array(self.P1))
        self.y_length = np.linalg.norm(np.array(self.P2) - np.array(self.P1))

        self.x_steps = int(self.x_length / grid_size) + 1
        self.y_steps = int(self.y_length / grid_size) + 1

        self.grid = np.zeros((self.x_steps, self.y_steps), dtype=int)
        self.obstacles = []
        self.obstacles_vertex = []

    def coord_to_grid(self, point):
        rel = np.array(point, dtype=float) - self.origin
        i_f = np.dot(rel, self.x_dir) / self.grid_size
        j_f = np.dot(rel, self.y_dir) / self.grid_size
        i = int(np.floor(i_f + 1e-6))
        j = int(np.floor(j_f + 1e-6))
        i = max(0, min(self.x_steps - 1, i))
        j = max(0, min(self.y_steps - 1, j))
        return i, j

    def grid_to_coord(self, grid_point):
        i, j = grid_point
        coord = self.origin + i * self.grid_size * self.x_dir + j * self.grid_size * self.y_dir
        return float(coord[0]), float(coord[1])

    def add_obstacle(self, center, radius):
        cx, cy = center
        self.obstacles.append((center, radius))
        r2 = radius ** 2
        for i in range(self.x_steps):
            for j in range(self.y_steps):
                x, y = self.grid_to_coord((i, j))
                if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                    self.grid[i, j] += 1

    def add_obstacle_vertex(self, vertex):
        vertices = np.array([np.array(v).flatten() for v in vertex])
        self.obstacles_vertex.append(vertices)
        for i in range(self.x_steps):
            for j in range(self.y_steps):
                x, y = self.grid_to_coord((i, j))
                if self.is_point_in_polygon((x, y), vertices):
                    self.grid[i, j] += 1
    def is_point_in_polygon(self, point, vertices, eps=1e-9):
        if self.point_to_polygon_distance(point, vertices) <= eps:
            return True
        x, y = point
        inside = False
        px, py = vertices[-1, 0], vertices[-1, 1]
        for cx, cy in vertices:
            if (cy > y) != (py > y):
                ix = (py - cy)
                if abs(ix) > eps:
                    inter_x = (px - cx) * (y - cy) / ix + cx
                    if x < inter_x:
                        inside = not inside
            px, py = cx, cy
        return inside

    def point_to_polygon_distance(self, point, vertices):
        p = np.array(point, dtype=float)
        min_distance = float('inf')
        n = len(vertices)
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            a = np.array([x1, y1], dtype=float)
            b = np.array([x2, y2], dtype=float)
            ab = b - a
            ab_len2 = np.dot(ab, ab)
            if ab_len2 == 0:
                dist = np.linalg.norm(p - a)
            else:
                t = np.clip(np.dot(p - a, ab) / ab_len2, 0.0, 1.0)
                nearest = a + t * ab
                dist = np.linalg.norm(p - nearest)
            min_distance = min(min_distance, dist)
        return float(min_distance)

    def remove_obstacle(self, center, radius):
        if (center, radius) not in self.obstacles:
            return
        self.obstacles.remove((center, radius))
        cx, cy = center
        r2 = radius ** 2
        for i in range(self.x_steps):
            for j in range(self.y_steps):
                x, y = self.grid_to_coord((i, j))
                if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                    self.grid[i, j] -= 1
    def re_initialize(self):
        self.grid = np.zeros((self.x_steps, self.y_steps), dtype=int)
        self.obstacles = []
        self.obstacles_vertex = []
    
    def list_obstacles(self, obstacle_list):
        self.re_initialize()
        for obstacle in obstacle_list:
            self.add_obstacle_vertex(obstacle)

    def is_valid(self, grid_point):
        i, j = grid_point
        return 0 <= i < self.x_steps and 0 <= j < self.y_steps and self.grid[i, j] == 0

    def get_neighbors(self, node):
        i, j = node
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        return [(i + di, j + dj) for di, dj in deltas if self.is_valid((i + di, j + dj))]

    def snap_to_grid(self, point):
        i, j = self.coord_to_grid(point)
        candidates = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
        valid_candidates = [
            candidate for candidate in candidates if self.is_valid(candidate)
        ]
        return valid_candidates[0] if valid_candidates else (i, j)

    def calculate_obstacle_penalty(self, real_coord, obstacle_cache):
        if real_coord in obstacle_cache:
            return obstacle_cache[real_coord]
        penalty = 0
        for (cx, cy), radius in self.obstacles:
            dist = np.hypot(real_coord[0] - cx, real_coord[1] - cy)
            if dist < radius:
                penalty += 1e6
            else:
                penalty += 3 / (dist - radius + 1e-5)
        for vertex in self.obstacles_vertex:
            if self.is_point_in_polygon(real_coord, vertex):
                penalty += 1e6
            else:
                dist = self.point_to_polygon_distance(real_coord, vertex)
                penalty += 3 / (dist + 1e-5)
        obstacle_cache[real_coord] = penalty
        return penalty
        
    def a_star(self, start, goal):
        start_idx = self.coord_to_grid(start)
        goal_idx = self.coord_to_grid(goal)

        x1, y1 = self.grid_to_coord(start_idx)
        x2, y2 = self.grid_to_coord(goal_idx)
        A, B, C = y2 - y1, x1 - x2, x2 * y1 - y2 * x1
        obstacle_cache = {}

        def heuristic(node1, node2):
            dist = np.hypot(node1[0] - node2[0], node1[1] - node2[1])
            x, y = self.grid_to_coord(node1)
            line_dist = abs(A * x + B * y + C) / (np.hypot(A, B) + 1e-12)
            obstacle_penalty = self.calculate_obstacle_penalty((x, y), obstacle_cache)
            i, j = node1
            edge_d = min(i, self.x_steps - 1 - i, j, self.y_steps - 1 - j)
            boundary_penalty = 10.0 / (edge_d + 1e-5)
            return 0.8 * dist + obstacle_penalty + 1.5 * line_dist + boundary_penalty
        def calculate_cost(current, next_state):
            x, y = current
            nx, ny = next_state
            base_cost = 1 if abs(nx - x) + abs(ny - y) == 1 else np.sqrt(2)
            return base_cost

        def forward_neighbors(pos):
            i, j = pos
            cand = [(i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
            return [p for p in cand if self.is_valid(p)]

        start_state = (start_idx[0], start_idx[1])
        open_set = []
        heapq.heappush(open_set, (0, start_state))
        came_from = {}
        g_score = {start_state: 0}
        f_score = {start_state: heuristic(start_idx, goal_idx)}

        while open_set:
            _, current_state = heapq.heappop(open_set)
            cur_pos    = (current_state[0], current_state[1])
            if cur_pos == goal_idx:
                path = []
                s = current_state
                while s in came_from:
                    path.append(self.grid_to_coord((s[0], s[1])))
                    s = came_from[s]
                path.append(start)
                return path[::-1]

            for nxt in forward_neighbors(cur_pos):
                step_cost  = calculate_cost(cur_pos, nxt)
                tentative_g_score = g_score[current_state] + step_cost
                child_state = (nxt[0], nxt[1])
                if child_state not in g_score or tentative_g_score < g_score[child_state]:
                    came_from[child_state] = current_state
                    g_score[child_state] = tentative_g_score
                    f = tentative_g_score + heuristic(nxt, goal_idx)
                    f_score[child_state] = f
                    heapq.heappush(open_set, (f, child_state))
        return None

    def dijkstra(self, start, goal):
        start_idx = self.coord_to_grid(start)
        goal_idx = self.coord_to_grid(goal)
        
        open_set = []
        heapq.heappush(open_set, (0, start_idx))
        came_from = {}
        cost = {start_idx: 0}

        while open_set:
            current_cost, current = heapq.heappop(open_set)
            if current == goal_idx:
                path = []
                while current in came_from:
                    path.append(self.grid_to_coord(current))
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                step_cost = 1 if abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) == 1 else np.sqrt(2)
                tentative_cost = current_cost + step_cost
                if neighbor not in cost or tentative_cost < cost[neighbor]:
                    came_from[neighbor] = current
                    cost[neighbor] = tentative_cost
                    heapq.heappush(open_set, (tentative_cost, neighbor))
        return None

    def smooth_path(self, path, fine_grid_size):
        if not path or len(path) < 2:
            return path

        x = [p[0] for p in path]
        y = [p[1] for p in path]

        t = [0]
        for i in range(1, len(path)):
            t.append(t[-1] + np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))

        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)

        t_fine = np.linspace(0, t[-1], int(t[-1] / fine_grid_size))
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)

        return list(zip(x_fine, y_fine))

    def plot(self, path, mode="arrows", tier="mid", output_path=None, action_index = None):
        fig, ax = plt.subplots(dpi=300, facecolor="white")
        # ax.set_aspect("auto")
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        ax.axis("off")

        tier_color = {"up":"#22C55E","mid":"#3B82F6","down":"#F59E0B"}.get(tier,"#3B82F6")
        grid_c = "#6B7280"
        circ_fill, circ_edge = "#FCA5A5", "#B91C1C"
        poly_fill, poly_edge = "#93C5FD", "#1D4ED8"

        
        for i in range(self.x_steps + 1):
            x0, y0 = self.grid_to_coord((i, 0))
            x1, y1 = self.grid_to_coord((i, self.y_steps))
            ax.plot([x0, x0], [y0, y1], color=grid_c, linewidth=1.1, zorder=0)
        for j in range(self.y_steps + 1):
            x0, y0 = self.grid_to_coord((0, j))
            x1, y1 = self.grid_to_coord((self.x_steps, j))
            ax.plot([x0, x1], [y0, y0], color=grid_c, linewidth=1.1, zorder=0)

        for (cx, cy), r in getattr(self, "obstacles", []):
            ax.add_patch(Circle((cx, cy), r, facecolor=circ_fill, edgecolor=circ_edge, linewidth=1.6, alpha=0.65))
        for verts in getattr(self, "obstacles_vertex", []):
            ax.add_patch(Polygon(verts, closed=True, facecolor=poly_fill, edgecolor=poly_edge, linewidth=1.6, alpha=0.65))

        px, py = np.array([p[0] for p in path]), np.array([p[1] for p in path])

        if mode == "line":
            ax.plot(px, py, color=tier_color, linewidth=2.6, alpha=0.98)
        else:
            for i in range(len(path)-1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                color = "red" if action_index is not None and i in action_index else "#1E3A8A"
                ax.annotate("",
                            xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="->",
                                            lw=1.24,
                                            color=color,  # 深蓝色
                                            shrinkA=0, shrinkB=0,
                                            mutation_scale=12),
                            zorder=5)
        ax.plot(path[0][0], path[0][1],
                marker="o", markersize=9,
                color="#FACC15", markeredgecolor="black", zorder=6)
        ax.text(
            path[0][0], path[0][1] - 1.1,
            "Start", ha="center", va="top",
            fontsize=7, color="black", zorder=6,
            path_effects=[pe.withStroke(linewidth=1, foreground=poly_edge)]
        )

        
        ax.plot(path[-1][0], path[-1][1],
                marker="*", markersize=13,
                color="#DC2626", markeredgecolor="black", zorder=6)
        ax.text(
            path[-1][0], path[-1][1] - 1.1,
            "Goal", ha="center", va="top",
            fontsize=7, color="black", zorder=6,
            path_effects=[pe.withStroke(linewidth=1, foreground=poly_edge)]
        )


        xs = [self.grid_to_coord((i, 0))[0] for i in range(self.x_steps + 1)]
        ys = [self.grid_to_coord((0, j))[1] for j in range(self.y_steps + 1)]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dx = abs(self.grid_to_coord((1,0))[0] - self.grid_to_coord((0,0))[0]) if self.x_steps>0 else 1.0
        dy = abs(self.grid_to_coord((0,1))[1] - self.grid_to_coord((0,0))[1]) if self.y_steps>0 else 1.0
        margin = max(dx, dy)
        w, h = max_x - min_x, max_y - min_y
        side = max(w, h) + 2*margin
        cx, cy = (max_x + min_x)/2, (max_y + min_y)/2
        ax.set_xlim(cx - side/2, cx + side/2)
        ax.set_ylim(cy - side/2, cy + side/2)
        fig.tight_layout(pad=0)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)



def load_vertices_txt(filename):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(BASE_DIR, filename)
    obstacles, current = [], []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current: 
                    obstacles.append(np.array(current, dtype=float))
                    current = []
            else:
                current.append([float(x) for x in line.split()])
        if current:
            obstacles.append(np.array(current, dtype=float))
    return obstacles

def base_scena():
    map_grid = BASE_ASTAR([[104, -152], [104, -133], [220, -152], [220, -133]], 3)
    vertexes = load_vertices_txt("obstacle.txt")
    for vertex in vertexes:
        map_grid.add_obstacle_vertex(vertex)
    
    start = (105, -143)
    goal = (226, -143)

    path_a_star = map_grid.a_star(start, goal)
    map_grid.plot(path=path_a_star, output_path="base_astar.png")
    path_dijkstra = map_grid.dijkstra(start, goal)
    map_grid.plot(path=path_dijkstra, output_path="base_dijkstra.png")

if __name__ == "__main__":
    base_scena()
    