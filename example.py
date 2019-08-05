import cv2
import numpy as np
import pyrealsense2 as rs
from mtcnn_pytorch import Predictor, get_biggest_face

RESOLUTION_WIDTH = 1280
RESOLUTION_HEIGHT = 720
FPS = 30

def project_color_pix_to_depth(color_intrin, depth_intrin, color_to_depth_extrin, depth_scale, color_pix):

    color_point = rs.rs2_deproject_pixel_to_point(color_intrin, color_pix, 1.0 / depth_scale)
    depth_point = rs.rs2_transform_point_to_point(color_to_depth_extrin, color_point)
    depth_pix = rs.rs2_project_point_to_pixel(depth_intrin, depth_point)

    depth_pix = [int(x) for x in depth_pix]

    return depth_pix


def draw_same_box_test():

    face_detector = Predictor()

    realsense = rs.pipeline()
    rs_config = rs.config()

    rs_config.enable_stream(rs.stream.color, RESOLUTION_WIDTH, RESOLUTION_HEIGHT,
                            rs.format.any, FPS)
    rs_config.enable_stream(rs.stream.depth, RESOLUTION_WIDTH, RESOLUTION_HEIGHT,
                            rs.format.any, FPS)

    profile = realsense.start(rs_config)

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0)

    frames = realsense.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    color_to_depth_extrin = color_frame.profile.get_extrinsics_to(depth_frame.profile)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    while True:
        frames = realsense.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)

        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        boxes = face_detector.predict_bounding_boxes(color_image)

        if len(boxes) > 0:
            src_rect = get_biggest_face(boxes)

            x1, y1, x2, y2 = src_rect[:4]

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

            p1 = project_color_pix_to_depth(color_intrin, depth_intrin,
                                            color_to_depth_extrin, depth_scale, [x1, y1])

            p2 = project_color_pix_to_depth(color_intrin, depth_intrin,
                                            color_to_depth_extrin, depth_scale, [x2, y2])

            cv2.rectangle(depth_image, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 0), thickness=3)

        image_top = np.concatenate([color_image, depth_image], axis=1)
        h, w, _ = image_top.shape
        image_top = cv2.resize(image_top, (int(w / 1.5), int(h / 1.5)))

        cv2.imshow('press q to exit', image_top)
        key = cv2.waitKey(1)

        if key == ord("q"):
            exit(0)


if __name__ == "__main__":
    draw_same_box_test()
