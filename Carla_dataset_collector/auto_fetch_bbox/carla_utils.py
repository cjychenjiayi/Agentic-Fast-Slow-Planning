import carla
import random
import json
import time
random.seed(time.time())
def point_in_canvas(pos, img_h, img_w):
    return pos[0] >= 0 and pos[0] < img_w and pos[1] >= 0 and pos[1] < img_h

def set_sync(world):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

def get_save_bp(world):
    blueprints = world.get_blueprint_library().filter("vehicle")
    blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
    blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
    blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
    blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
    blueprints = [x for x in blueprints if not x.id.endswith('t2')]
    return blueprints

def point_in_canvas(pos, img_h, img_w):
    return pos[0] >= 0 and pos[0] < img_w and pos[1] >= 0 and pos[1] < img_h

predefined_weathers = {
    name: getattr(carla.WeatherParameters, name)
    for name in dir(carla.WeatherParameters)
    if isinstance(getattr(carla.WeatherParameters, name), carla.WeatherParameters)
}
del predefined_weathers["Default"]
to_delete = []
for key, value in predefined_weathers.items():
    if "night" in key.lower():
        to_delete.append(key)
for key in to_delete:
    del predefined_weathers[key]
def random_set_weather(world, special_weather_ratio):
    if random.random() < special_weather_ratio:
        weather = random.choice(list(predefined_weathers.keys()))
        world.set_weather(predefined_weathers[weather])
        
def filter_static_blueprints(world, size_threshold=0.3, major_axis_threshold=0.5):
    blueprint_library = world.get_blueprint_library()
    ret = []
    with open("blueprint_static_cur.txt", "r") as f:
        bps = f.readlines()
        bps = [bp for bp in bps if len(bp) > 3]
    ret = [blueprint_library.find(bp.strip()) for bp in bps]  
    return ret

map_list = ['Town10HD_Opt',  'Town01_Opt', 'Town05_Opt', 'Town03_Opt', 'Town02_Opt', 'Town04_Opt', 'Town06_Opt', 'Town07_Opt']

def get_all_waypoints(world, distance=2.5):
    map = world.get_map()
    waypoints = map.generate_waypoints(distance)
    return waypoints

with open("bbox.json", "r") as f:
    bbox_json = json.load(f)
    
def get_bbox_bp(bp):
    bp_bbox = bbox_json[bp]
    bp_vec = bp_bbox["ext"]
    bp_loc = bp_bbox["loc"]
    bbox_loc_bp = carla.Location(x=bp_loc[0], y=bp_loc[1], z=bp_loc[2])
    bbox_vec_bp = carla.Vector3D(x=bp_vec[0], y=bp_vec[1], z=bp_vec[2])
    return carla.BoundingBox(bbox_loc_bp, bbox_vec_bp)

def get_z(bp):
    bp_bbox = bbox_json[bp]
    bp_loc = bp_bbox["loc"]
    return 0.5

def spawn_random_static_objects(world, num=10):
    static_objects = []
    blueprints = filter_static_blueprints(world)
    waypoints = get_all_waypoints(world)
    used_waypoints = []
    for _ in range(num):
        try:
            blueprint = random.choice(blueprints)
            waypoint = random.choice(waypoints)
            cnt = 0
            while waypoint in used_waypoints and cnt < 10:
                waypoint = random.choice(waypoints)
                cnt += 1
            if cnt >= 10:
                continue
            used_waypoints.append(waypoint)
            transform = waypoint.transform
            transform.location.z = get_z(blueprint.id)
            static_object = world.try_spawn_actor(blueprint, transform)

            if static_object is not None:
                static_objects.append(static_object)
                # print(f"Spawned {static_object.type_id} at {transform.location}")
        except Exception as e:
            print(f"Error while spawning static object: {e}")

    return static_objects

