import numpy as np
import heapq
try:
    from astar_tool.base_astar import BASE_ASTAR, load_vertices_txt
except:
    from base_astar import BASE_ASTAR, load_vertices_txt
import sys
import os
sys.path.append(os.getcwd())

class LLMAID_ASTAR(BASE_ASTAR):
    def __init__(self, points, grid_size):
        super().__init__(points, grid_size)

    def a_star(self, start, goal, llm_guide=None, hyper_params = [-1, 0.1, 5, 0.8]):
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
            return 0.8 * dist + obstacle_penalty + 1.5 * line_dist + boundary_penalty, line_dist

        def calculate_cost(current, next_state):
            x, y = current
            nx, ny = next_state
            base_cost = 1 if abs(nx - x) + abs(ny - y) == 1 else np.sqrt(2)
            return base_cost

        def forward_neighbors(pos):
            i, j = pos
            cand = [(i + 1, j - 1), (i + 1, j), (i + 1, j + 1)]
            return [p for p in cand if self.is_valid(p)]

        def move_label(u, v):
            dj = v[1] - u[1]
            if dj == 1:
                return 'FL'
            if dj == 0:
                return 'F'
            if dj == -1:
                return 'FR'
            return 'UNK'

        def current_action(action_idx):
            return llm_guide[action_idx] if action_idx < len(llm_guide) else None

        def is_action_completed(prev_move, cur_pos, next_pos, action):
            if not action:
                return False
            a = action.lower()
            dj = next_pos[1] - cur_pos[1]
            if a in ('keep') and dj == 0:
                return True
            
            cur_dir = 1 if dj > 0 else -1 if dj < 0 else 0
            prev_dir = 1 if prev_move == 'FL' else -1 if prev_move == 'FR' else 0
            if a in ('right', 'r'):
                return prev_dir == -1 and cur_dir != -1
            if a in ('left', 'l'):
                return prev_dir == 1 and cur_dir != 1
            return False

        def semantic_step_cost(prev_move, cur_pos, next_pos, action):
            C_CORR, C_DELAY, C_WRONG, C_OVER = hyper_params

            def d_prev(m):
                if m == 'FL':
                    return 1
                if m == 'FR':
                    return -1
                return 0

            def d_step(u, v):
                dj = v[1] - u[1]
                return 1 if dj > 0 else -1 if dj < 0 else 0

            def d_act(a):
                if not a:
                    return None
                a = a.lower()
                if a in ('right', 'r'):
                    return -1
                if a in ('left', 'l'):
                    return 1
                if a in ('keep'):
                    return 0
                return None

            prev_dir = d_prev(prev_move)
            cur_dir = d_step(cur_pos, next_pos)
            tgt_dir = d_act(action)

            if tgt_dir is None:
                return 0, None

            if tgt_dir != 0:
                if cur_dir == tgt_dir and prev_dir != tgt_dir:
                    return C_CORR, "correct"
                if cur_dir == tgt_dir and prev_dir == tgt_dir:
                    return C_OVER, "over"
                if cur_dir == 0 and prev_dir == 0:
                    return C_DELAY, "delay"
                if cur_dir == -tgt_dir:
                    return C_WRONG, "wrong"
                return C_DELAY, "delay"

            if cur_dir == 0:
                return  C_CORR, "delay"
            return C_WRONG, "wrong"

        start_state = (start_idx[0], start_idx[1], 'START', 0)

        open_set = []
        heapq.heappush(open_set, (0, start_state))

        came_from = {}
        g_score = {start_state: 0}
        cost, _ = heuristic(start_idx, goal_idx)
        f_score = {start_state: cost}

        def current_action(idx):
            if isinstance(llm_guide, (list, tuple)):
                return llm_guide[idx] if 0 <= idx < len(llm_guide) else None
            return llm_guide
        state_punish_dict = {}
        while open_set:
            _, current_state = heapq.heappop(open_set)
            cur_pos = (current_state[0], current_state[1])
            prev_move = current_state[2]
            cur_aidx = current_state[3]

            if cur_pos == goal_idx:
                need = (len(llm_guide) if isinstance(llm_guide, (list, tuple)) else (0 if llm_guide in (None, '') else 1))
                if cur_aidx < need:
                    continue
                path = []
                move = []
                llm_match_position = []
                cost_name = []
                s = current_state
                lst = len(llm_guide)
                while s in came_from:
                    if lst != s[3]:
                        llm_match_position.append(1)
                        # print(self.grid_to_coord((s[0], s[1])))
                    else:
                        llm_match_position.append(0)
                    cost_name.append(state_punish_dict[s])
                    lst = s[3]
                    path.append(self.grid_to_coord((s[0], s[1])))
                    move.append(s[2])
                    s = came_from[s]
                path.append(start)
                def index_action_pairs(match_seq, actions):
                    result = []
                    action_idx = 0 
                    for i, flag in enumerate(match_seq):
                        if flag == 1:
                            result.append((i, actions[action_idx]))
                            action_idx += 1
                    return result
                convert_dict = {"F":"keep", "FL":"left", "FR":"right"}
                move = [convert_dict[m] if m in convert_dict else "unknown" for m in move]
                action_index= index_action_pairs(llm_match_position[::-1], llm_guide)
                path = path[::-1]
                action_loc = [[path[index[0]], index[1]] for index in action_index]
                return path,action_loc,move[::-1],  cost_name[::-1], action_index

            for nxt in forward_neighbors(cur_pos):
                curr_move = move_label(cur_pos, nxt)
                llm_action = current_action(cur_aidx)
                step_cost = calculate_cost(cur_pos, nxt)
                sem_cost, cost_name = semantic_step_cost(prev_move, cur_pos, nxt, llm_action)
                
                tentative_g_score = g_score[current_state] + step_cost + sem_cost

                completed = is_action_completed(prev_move, cur_pos, nxt, llm_action)
                new_aidx = cur_aidx + 1 if completed else cur_aidx

                child_state = (nxt[0], nxt[1], curr_move, new_aidx)
                if child_state not in g_score or tentative_g_score < g_score[child_state]:
                    came_from[child_state] = current_state
                    heuristic_cost, line_cost = heuristic(nxt, goal_idx)
                    g_score[child_state] = tentative_g_score + line_cost
                    state_punish_dict[child_state] = cost_name
                    f = tentative_g_score + heuristic_cost
                    f_score[child_state] = f
                    heapq.heappush(open_set, (f, child_state))
        print("FAIIIIL")
        return None


def llm_scena_1():
    map_grid = LLMAID_ASTAR([[104, -152], [104, -133], [220, -152], [220, -133]], 3)
    map_grid = LLMAID_ASTAR([[137, -150], [137, -133], [223, -150], [223, -133]], 3)
    # map_grid = LLMAID_ASTAR([[137, -152], [137, -133], [223, -152], [223, -133]], 3)
    vertexes = load_vertices_txt("obstacle_v1.txt")
    for vertex in vertexes:
        map_grid.add_obstacle_vertex(vertex)

    start = (141, -143)
    goal = (226, -143)

    path_a_star, path_info, llm_action = map_grid.a_star(start, goal, llm_guide=["left", "right"])
    map_grid.plot(path=path_a_star, output_path="llm_left_v0.png")
    with open("llm_left_v0.txt", "w") as f:
        for point in path_a_star:
            f.write(f"{point[0]} {point[1]}\n")
    path_a_star, path_info, llm_action = map_grid.a_star(start, goal, llm_guide=["right", "left"])
    map_grid.plot(path=path_a_star, output_path="llm_right_v0.png")
    with open("llm_right_v0.txt", "w") as f:
        for point in path_a_star:
            f.write(f"{point[0]} {point[1]}\n")

DEFAULT_MAP_1 = [[104, -152], [104, -133], [220, -152], [220, -133]]
DEFAULT_MAP_2 = [[137, -152], [137, -133], [223, -152], [223, -133]]
hyper_params = [-5, 0.1, 10, 0.8]
def llm_path_generate( grid_map = DEFAULT_MAP_2, grid_size = 3, vertexes_path = "obstacle_v1.txt", 
                      start = (141, -143), goal = (226, -143), llm_guide = ["right", "keep",  "left", "right"],
                      hyper_params = hyper_params, output_path = None):
    
    map_grid = LLMAID_ASTAR(grid_map, grid_size)
    vertexes = load_vertices_txt(vertexes_path)
    for vertex in vertexes:
        map_grid.add_obstacle_vertex(vertex)
    path_a_star, action_loc, llm_action, cost_name, action_index = map_grid.a_star(start, goal, llm_guide=llm_guide, hyper_params=hyper_params)
    # print(path_info)
    print(action_index)
    action_index = [a[0] for a in action_index]
    print(action_index)
    if output_path is not None:
        map_grid.plot(path=path_a_star, output_path=output_path,action_index=action_index)
    return path_a_star, action_loc, llm_action, cost_name
    

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    hyper_params1 = [-5, 0.1, 10, 0.8]
    hyper_params2 = [-5, 0.1, 5, 0.8]
    hyper_params3 = [-5, 0.3, 5, 0.8]
    hyper_params4 = [-10, 0.1, 10, 0.8]
    hyper_params5 = [-5, 1, 5, 0.8]
    hyper_params = [hyper_params1, hyper_params2, hyper_params3, hyper_params4, hyper_params5]
    for index, param in enumerate(hyper_params):
        with open(f"output/param_{index}.txt", "w") as f:
            f.write(str(param))
        path_astar, _, _, _ = llm_path_generate(hyper_params=param, output_path=f"output/llm_agent_{index}.png")
        with open(f"output/path_{index}.txt", "a") as f:
            for point in path_astar:
                f.write(f"{point[0]} {point[1]}\n") 