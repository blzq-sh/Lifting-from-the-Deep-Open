#!/usr/bin/env python
# -*- coding: utf-8 -*-

import __init__

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from tf_pose import common
from tf_pose.common import MPIIPart, CocoPart

from lifting.utils import Prob3dPose
from lifting.utils import plot_pose
from lifting.utils import config

import cv2
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, realpath, join
import time

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

FROM_FILE = True
VIDEO_FILE_PATH = PROJECT_PATH + '/data/images/input.mp4'

DO_3D = True


def main():
    if not FROM_FILE:
        device = cv2.CAP_OPENNI2
    else:
        device = VIDEO_FILE_PATH
    capture = cv2.VideoCapture(device)
    if not capture.isOpened():
        print("Capture device not opened")
        capture.release()
        return 1
    fig = plt.figure(figsize=(16, 8))

    # create pose estimator
    retval = capture.grab()
    _, color_im = capture.retrieve(0, cv2.CAP_OPENNI_BGR_IMAGE)
    image_size = color_im.shape
    pose_estimator2D = TfPoseEstimator(get_graph_path('cmu'),
                                       target_size=(image_size[1],
                                                    image_size[0]))
    pose_lifter3D = Prob3dPose(PROB_MODEL_PATH)

    while True:
        retval = capture.grab()
        retval, color_im = capture.retrieve(0, cv2.CAP_OPENNI_BGR_IMAGE)
        color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)

        # estimation
        start_time_2D = time.perf_counter()
        estimated_pose_2d = pose_estimator2D.inference(
            color_im, resize_to_default=True, upsample_size=2.0)
        end_time_2D = time.perf_counter()

        if len(estimated_pose_2d) == 0:
            plt.subplot(1, 1, 1)
            plt.imshow(color_im)
            plt.pause(0.01)
            continue

        if DO_3D:
            pose_2d_mpii, visibility = to_mpii_pose_2d(estimated_pose_2d)
            start_time_3D = time.perf_counter()
            transformed_pose_2d, weights = pose_lifter3D.transform_joints(
                np.array(pose_2d_mpii), visibility)
            pose_3d = pose_lifter3D.compute_3d(transformed_pose_2d, weights)
            end_time_3D = time.perf_counter()
        else:
            pose_3d = []
            start_time_3D, end_time_3D = 0, 0

        # Show 2D and 3D poses
        display_results(color_im, estimated_pose_2d, pose_3d)

        print("OP - 2D: {}, 3D: {}".format(
            end_time_2D - start_time_2D, end_time_3D - start_time_3D))
        plt.pause(0.01)

    return retval


def to_mpii_pose_2d(humans):
    visibility = []
    pose_2d_mpii = []

    for human in humans:
        one_pose_mpii, one_visible = from_coco(human)
        pose_2d_mpii.append(one_pose_mpii)
        visibility.append(one_visible)
    return np.array(pose_2d_mpii), np.array(visibility)


def from_coco(human):  # Reversed y and x from tf_openpose
    t = [
        (MPIIPart.Head, CocoPart.Nose),
        (MPIIPart.Neck, CocoPart.Neck),
        (MPIIPart.RShoulder, CocoPart.RShoulder),
        (MPIIPart.RElbow, CocoPart.RElbow),
        (MPIIPart.RWrist, CocoPart.RWrist),
        (MPIIPart.LShoulder, CocoPart.LShoulder),
        (MPIIPart.LElbow, CocoPart.LElbow),
        (MPIIPart.LWrist, CocoPart.LWrist),
        (MPIIPart.RHip, CocoPart.RHip),
        (MPIIPart.RKnee, CocoPart.RKnee),
        (MPIIPart.RAnkle, CocoPart.RAnkle),
        (MPIIPart.LHip, CocoPart.LHip),
        (MPIIPart.LKnee, CocoPart.LKnee),
        (MPIIPart.LAnkle, CocoPart.LAnkle),
    ]

    pose_2d_mpii = []
    visibility = []
    for mpi, coco in t:
        if coco.value not in human.body_parts.keys():
            pose_2d_mpii.append((0, 0))
            visibility.append(False)
            continue
        pose_2d_mpii.append((human.body_parts[coco.value].y,
                             human.body_parts[coco.value].x))
        visibility.append(True)
    return pose_2d_mpii, visibility


def display_results(in_image, data_2d, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    n_poses = len(data_3d)
    color_im = TfPoseEstimator.draw_humans(in_image, data_2d,
                                           imgcopy=False)
    plt.subplot(1, n_poses+1, 1)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for idx, single_3D in enumerate(data_3d):
        plot_pose(Prob3dPose.centre_all(single_3D), n_poses+1, idx+2)

if __name__ == '__main__':
    import sys
    sys.exit(main())
