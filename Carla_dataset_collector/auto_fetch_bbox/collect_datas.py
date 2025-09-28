import carla
import random
import queue
import numpy as np
import cv2
from camera_utils import *
from carla_utils import *
import random
from datetime import datetime
import os
import argparse

import numpy as np
import time
random.seed(time.time())
def calculate_relative_angle(self_location, other_location, self_yaw):
    self_location = np.array(self_location)
    other_location = np.array(other_location)
    relative_vector = other_location - self_location
    relative_vector_2d = relative_vector[:2]
    self_yaw_rad = np.radians(self_yaw)
    forward_vector = np.array([np.cos(self_yaw_rad), np.sin(self_yaw_rad)])
    dot_product = np.dot(relative_vector_2d, forward_vector)
    magnitude_product = np.linalg.norm(relative_vector_2d) * np.linalg.norm(forward_vector)
    angle_rad = np.arccos(dot_product / magnitude_product)  # 弧度制
    angle_deg = np.degrees(angle_rad)
    cross_product = relative_vector_2d[0] * forward_vector[1] - relative_vector_2d[1] * forward_vector[0]
    if cross_product < 0:
        angle_deg = -angle_deg

    return angle_deg


def main(output_dir, map_name):
    def generate_timestamp_string():
        return datetime.now().strftime("%m_%d_%H_%M")

    timestamp_string = generate_timestamp_string()
    output_dir = os.path.join(output_dir,timestamp_string)
    seg_dir = os.path.join(output_dir, "seg")
    rgb_dir = os.path.join(output_dir, "rgb")
    data_dir = os.path.join(output_dir, "data")

    

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.load_world(f"{map_name}_Opt")
    world.unload_map_layer(carla.MapLayer.Props)
    world.unload_map_layer(carla.MapLayer.ParkedVehicles)
    random_set_weather(world, special_weather_ratio=0.7)

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
    vehicle.set_autopilot(True)
    set_sync(world)

    image_queue_sem_seg = queue.Queue()
    image_queue_rgb = queue.Queue()

    def process_image_sem_seg(image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        image_queue_sem_seg.put(image)

    def process_image_rgb(image):
        image_queue_rgb.put(image)

    camera_sem_seg, K_sem_seg = gene_cam(world, vehicle, bp_lib, 'sensor.camera.semantic_segmentation', offset=2)
    camera_rgb, K_rgb = gene_cam(world, vehicle, bp_lib, 'sensor.camera.rgb', offset=2.05)
    camera_sem_seg.listen(process_image_sem_seg)
    camera_rgb.listen(process_image_rgb)

    all_npc = []
    bp_lib_vehicle = get_save_bp(world)
    small_map_list = ["Town02", "Town05"]
    num_static = 15 if map_name in small_map_list else 25
    num_vehicle = 8 if map_name in small_map_list else 13
    large_map_list = ["Town06", "Town10HD"]
    if map_name in large_map_list:
        num_static = 30
        num_vehicle = 20
    for i in range(num_vehicle):
        vehicle_bp = random.choice(bp_lib_vehicle)
        try:
            npc = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
            npc.set_autopilot(True)
            all_npc.append(npc)
        except Exception as e:
            print(e)
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    static_objects = spawn_random_static_objects(world, num=num_static)
    all_npc.extend(static_objects)
    world.tick()
    try:
        cnt = 0
        start_time = time.time()
        while True:
            if not image_queue_sem_seg.empty() and not image_queue_rgb.empty():
                image_sem_seg = image_queue_sem_seg.get()
                image_rgb = image_queue_rgb.get()
                if cnt % 4 != 0:
                    cnt += 1
                    world.tick()
                    continue
                img_height = image_rgb.height
                img_width = image_rgb.width
                img_sem_seg = np.reshape(np.copy(image_sem_seg.raw_data), (image_sem_seg.height, image_sem_seg.width, 4))
                img_rgb = np.reshape(np.copy(image_rgb.raw_data), (image_rgb.height, image_rgb.width, 4))

                world_2_camera = np.array(camera_rgb.get_transform().get_inverse_matrix())
                vehicle_trans = vehicle.get_transform()
                objects = []
                yaw = vehicle_trans.rotation.yaw
                metadata = [{
                        "vehicle_location": {
                                "x": vehicle_trans.location.x,
                                "y": vehicle_trans.location.y,
                                "z": vehicle_trans.location.z,
                        },
                        "vehicle_rotation":yaw
                        }
                    ]
                vehicle_loc = [vehicle_trans.location.x, vehicle_trans.location.y, vehicle_trans.location.z]
                for npc in all_npc:
                    bb = npc.bounding_box
                    npc_transform = npc.get_transform()
                    dist = npc_transform.location.distance(vehicle_trans.location)

                    if dist < 60:
                        vertices = [v for v in bb.get_world_vertices(npc_transform)]
                        x_max, x_min = -float('inf'), float('inf')
                        y_max, y_min = -float('inf'), float('inf')

                        for vert in vertices:
                            p = get_image_point(vert, K_rgb, world_2_camera)
                            x_max = max(p[0], x_max)
                            x_min = min(p[0], x_min)
                            y_max = max(p[1], y_max)
                            y_min = min(p[1], y_min)
                        if abs(x_max) > img_width * 1.2 or abs(x_min) > img_width * 1.2 or abs(y_max) > img_height * 1.2 or abs(y_min) > img_height * 1.2:
                            continue
                        obj_loc = [npc_transform.location.x, npc_transform.location.y, npc_transform.location.z]
                        real_oriented = calculate_relative_angle(vehicle_loc, obj_loc, yaw)
                        objects.append({
                            "id": npc.type_id,
                            "location": {
                                "x": npc_transform.location.x,
                                "y": npc_transform.location.y,
                                "z": npc_transform.location.z,
                            },
                            "bbox": {
                                "x_min": x_min,
                                "x_max": x_max,
                                "y_min": y_min,
                                "y_max": y_max
                            },
                            "related_orient":real_oriented
                        })

                metadata.append(objects)
                seg_path = os.path.join(seg_dir, f"{cnt}.png")
                cv2.imwrite(seg_path, img_sem_seg)
                rgb_path = os.path.join(rgb_dir, f"{cnt}.png")
                cv2.imwrite(rgb_path, img_rgb)
                data_path = os.path.join(data_dir, f"{cnt}.json")
                with open(data_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                cnt += 1
                world.tick()
            if cnt % 60 == 0 and time.time() - start_time > 120:
                break

    except Exception as e:
        print(e)
    finally:
        # for npc in all_npc:
        #     try:
        #         npc.destroy()
        #     except:
        #         pass
        cv2.destroyAllWindows()
        try:
            camera_sem_seg.destroy()
            camera_rgb.destroy()
            vehicle.destroy()
        except:
            pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--map_name', type=str, default="Town10HD")
    args = parser.parse_args()
    main(args.output_dir, args.map_name)
    