import os, json, math
try:
    from memory_topo_scena.compare_rag import match_and_score
except:
    from compare_rag import match_and_score
import re
import random
def parse_scene(text: str):

    results = []
    items = re.split("], ", text)
    for item in items:
        item = item.replace("[", "").replace("]", "").replace('"', "").replace("'", "").strip()
        parts = [p.strip() for p in item.split(',')]
        category = parts[0]
        num1 = float(parts[1])
        num2 = float(parts[2])
        results.append([category, num1, num2])
    
    return results

def parse_hyperparam(text: str):
    content = text.strip()[1:-1]
    parts = [p.strip() for p in content.split(',')]
    nums = [float(p) for p in parts]
    return nums


class RagStore:
    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.file = os.path.join(path, "database.json")
        if not os.path.exists(self.file):
            with open(self.file, "w") as f: json.dump([], f)
        with open(self.file) as f: self.data = json.load(f)

    def add(self, scene, hyperparams):
        self.data.append({"scene": scene, "hyperparams": hyperparams})
        with open(self.file, "w") as f: json.dump(self.data, f)
        
    def add_llm(self, scene, hyperparams):
        
        scene = parse_scene(scene)
        hyperparams = parse_hyperparam(hyperparams)
        self.data.append({"scene": scene, "hyperparams": hyperparams})
        with open(self.file, "w") as f: json.dump(self.data, f, indent=4)

    def match(self, scene, k=3):
        scene = parse_scene(scene)
        best, bestd = None, 1e9
        scored = []
        for item in self.data:
            d = match_and_score(scene, item["scene"])
            scored.append((d["similarity"], item))

        scored.sort(key=lambda x: x[0])
        topk = scored[:k]
        return random.choice(topk)[1]
    
file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)

scene_store = RagStore(os.path.join(dir_path, "database"))
if __name__ == "__main__":
    # store = RagStore("database")
    # scene_to_add = [['streetbarrier',  9.7, -0.5], ['trafficcone',  13.7, -15.5], ['vehicle', 17.1, 10.5], ['vehicle', 54.7, 0.0]]
    # hyper_param = [-5, 0.1, 5, 0.8]
    # store.add(scene_to_add, hyper_param)
    parse_s = '[["streetbarrier", 9.7, -0.5], ["trafficcone", 13.7, -15.5], ["vehicle", 17.1, 10.5], ["vehicle", 54.7, 0.0]]'
    # parse_s = parse_scene(parse_s)
    print(scene_store.data)
    print(scene_store.match(parse_s))
    # scene_to_add = []
    # hyper_param = [-6, 0.05, 7, 1]
    # store.add(scene_to_add, hyper_param)