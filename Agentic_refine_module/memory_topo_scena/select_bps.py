import random
import time
random.seed(time.time())
with open("blueprint_static.txt", "r") as f:
    bps = f.readlines()

random.shuffle(bps)

with open("blueprint_static_cur.txt", "w") as f:
    for bp in bps[:20]:
        f.write(bp.strip()+"\n")