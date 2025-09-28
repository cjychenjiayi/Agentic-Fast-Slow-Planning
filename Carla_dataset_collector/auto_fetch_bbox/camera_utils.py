import numpy as np
import carla
import queue

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

def gene_cam(world, vehicle, bp_lib, camera_type, offset):
    camera_bp = bp_lib.find(camera_type)
    camera_bp.set_attribute("image_size_x", "1080")
    camera_bp.set_attribute("image_size_y", "720")
    camera_bp.set_attribute("fov", "90")
    camera_init_trans = carla.Transform(carla.Location(x = 2.0, z=offset))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()
    K = build_projection_matrix(image_w, image_h, fov)
    return camera, K

def setup_camera_listener(camera, queue):
    def process_image(image):
        if camera.type_id == 'sensor.camera.semantic_segmentation':
            image.convert(carla.ColorConverter.CityScapesPalette)
        queue.put(image)
    camera.listen(process_image)