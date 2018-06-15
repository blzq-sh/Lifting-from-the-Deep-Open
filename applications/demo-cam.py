#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""

import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import cv2
import matplotlib.pyplot as plt
from os.path import dirname, realpath

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
# IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/test_image.png'
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'


def main():
    device = cv2.CAP_OPENNI2
    capture = cv2.VideoCapture(device)
    if not capture.isOpened():
        print("Capture device not opened")
        capture.release()
        return 1
    plt.figure()

    # create pose estimator
    retval = capture.grab()
    _, color_im = capture.retrieve(0, cv2.CAP_OPENNI_BGR_IMAGE)
    image_size = color_im.shape
    pose_estimator = PoseEstimator(
        image_size, SESSION_PATH, PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()
    
    try:
        while True:
            retval = capture.grab()
            retval, color_im = capture.retrieve(0, cv2.CAP_OPENNI_BGR_IMAGE)
            color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)

            # estimation
            pose_2d, visibility, pose_3d = pose_estimator.estimate(color_im)

            if pose_2d.size == 0:
                plt.subplot(1, 1, 1)
                plt.imshow(color_im)
                plt.pause(0.01)
                continue

            # Show 2D and 3D poses
            display_results(color_im, pose_2d, visibility, pose_3d)
            plt.pause(0.01)

    finally:
        # close model
        pose_estimator.close()

    return retval


def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    n_poses = len(data_3d)
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.subplot(1, n_poses+1, 1)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for idx, single_3D in enumerate(data_3d):
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D, n_poses+1, 2+idx)

if __name__ == '__main__':
    import sys
    sys.exit(main())
