#!/usr/bin/env python
# -*- coding: utf-8 -*-

import __init__

from tf_pose.estimator import TfPoseEstimator as OpPoseEstimator
from tf_pose.networks import get_graph_path
from tf_pose import common
from tf_pose.common import MPIIPart, CocoPart

from lifting import PoseEstimator as LftdPoseEstimator
from lifting.utils import Prob3dPose, plot_pose, config, draw_limbs

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import dirname, realpath, join
import time
import argparse

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'


def main(args):
    if args.input is None:
        device = cv2.CAP_OPENNI2
    else:
        device = os.path.abspath(args.input)
    print(device)
    capture = cv2.VideoCapture(device)
    if args.input is None and not capture.isOpened():
        print("Capture device not opened")
        capture.release()
        return 1
    elif args.input is not None and not capture.isOpened():
        print("File not found")
        capture.release()
        return 1
    fig = plt.figure(figsize=(12, 6))

    # create pose estimator
    retval = capture.grab()
    _, color_im = capture.retrieve(0, cv2.CAP_OPENNI_BGR_IMAGE)
    image_size = color_im.shape

    if args.mode == 'openpose':
        pose_estimator2D = OpPoseEstimator(get_graph_path('cmu'),
                                           target_size=(image_size[1],
                                                        image_size[0]))
        pose_lifter3D = Prob3dPose(PROB_MODEL_PATH)
    else:
        pose_estimator = LftdPoseEstimator(
            image_size, SESSION_PATH, PROB_MODEL_PATH)
        pose_estimator.initialise()

    n = 0
    while True:
        n += 1
        retval = capture.grab()
        if not retval:
            break
        retval, color_im = capture.retrieve(0, cv2.CAP_OPENNI_BGR_IMAGE)
        color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)

        if args.mode == 'openpose':
            # estimation by OpenPose
            start_time_2D = time.perf_counter()
            estimated_pose_2d = pose_estimator2D.inference(
                    color_im, resize_to_default=True, upsample_size=2.0)
            end_time_2D = time.perf_counter()
        else:
            # estimation by LFTD
            estimated_pose_2d, visibility, pose_3d = \
                pose_estimator.estimate(color_im)

        if len(estimated_pose_2d) == 0:
            plt.subplot(1, 1, 1)
            plt.imshow(color_im)
            plt.pause(0.01)
            continue

        if not args.do_3d:
            pose_3d = []
            start_time_3D, end_time_3D = 0, 0
        elif args.mode == 'openpose':
            pose_2d_mpii, visibility = to_mpii_pose_2d(estimated_pose_2d)
            start_time_3D = time.perf_counter()
            transformed_pose_2d, weights = pose_lifter3D.transform_joints(
                np.array(pose_2d_mpii), visibility)
            pose_3d = pose_lifter3D.compute_3d(transformed_pose_2d, weights)
            end_time_3D = time.perf_counter()
        if args.track_one:
            if 'prev' not in locals():
                prev = np.zeros(2)
            if args.mode == 'openpose':
                means = np.mean(transformed_pose_2d, axis=1)
            else:
                means = np.mean(estimated_pose_2d, axis=1)
            argmin = np.argmin(np.linalg.norm(means - prev, ord=1, axis=1))
            pose_3d = np.expand_dims(pose_3d[argmin], axis=0)
            prev = means[argmin]

        # Show 2D and 3D poses
        display_results(args.mode,
                        color_im, estimated_pose_2d, visibility, pose_3d)
        if args.output:
            fig.savefig(os.path.join(os.path.abspath(args.output),
                                     'out{:09d}.png'.format(n)))

        if args.mode == 'openpose':
            print("OP - 2D: {}, 3D: {}".format(
                end_time_2D - start_time_2D,
                end_time_3D - start_time_3D))
        plt.pause(0.01)
        fig.clf()

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


def visibility_to_3d(single_visibility):
    _H36M_ORDER = [8, 9, 10, 11, 12, 13, 1, 0, 5, 6, 7, 2, 3, 4]
    _W_POS = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]

    visibility_3d = np.ones(config.H36M_NUM_JOINTS)
    ordered_visibility = single_visibility[_H36M_ORDER]
    visibility_3d[_W_POS] = ordered_visibility
    return visibility_3d


def display_results(mode, in_image, data_2d, visibilities, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    n_poses = len(data_3d)
    if mode == 'openpose':
        color_im = OpPoseEstimator.draw_humans(in_image, data_2d,
                                               imgcopy=False)
    else:
        draw_limbs(in_image, data_2d, visibilities)

    plt.subplot(1, n_poses+1, 1)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for idx, single_3D in enumerate(data_3d):
        plot_pose(Prob3dPose.centre_all(single_3D),
                  visibility_to_3d(visibilities[idx]),
                  n_poses+1, idx+2)

if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', default=None,
                        help='Input video file path '
                             '(leave empty to use camera)')
    parser.add_argument('-o', '--output', dest='output', default=None,
                        help='Output images directory path '
                             '(leave empty to not save images)')
    parser.add_argument('-m', '--mode', dest='mode', default='openpose',
                        choices=['openpose', 'lftd'],
                        help='Engine for 2D pose computation')
    parser.add_argument('--2d', dest='do_3d', action='store_false',
                        default=True,
                        help='Whether to only compute 2D pose')
    parser.add_argument('--track_one', dest='track_one', action='store_true',
                        default=False,
                        help='Whether to only track one human')
    args = parser.parse_args()
    print(args)

    sys.exit(main(args))
