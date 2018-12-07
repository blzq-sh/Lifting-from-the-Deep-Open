from lifting import PoseEstimatorInterface
from lifting.utils import Prob3dPose

from tf_pose.estimator import TfPoseEstimator as OpPoseEstimator
from tf_pose.networks import get_graph_path
from tf_pose import common
from tf_pose.common import MPIIPart, CocoPart

import numpy as np


__all__ = [
    'OpPoseLFTDEstimator',
    'PlainOpPoseEstimator'
]


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


def to_mpii_pose_2d(humans):
    visibility = []
    pose_2d_mpii = []

    for human in humans:
        one_pose_mpii, one_visible = from_coco(human)
        pose_2d_mpii.append(one_pose_mpii)
        visibility.append(one_visible)
    return np.array(pose_2d_mpii), np.array(visibility)


def restructure_2d_pose(pose, width, height):
    pose_2d_mpii, visibility = to_mpii_pose_2d(pose)

    pose_2d_mpii[0][:, 0] = pose_2d_mpii[0][:, 0] * height
    pose_2d_mpii[0][:, 1] = pose_2d_mpii[0][:, 1] * width

    pose_2d_mpii = pose_2d_mpii.astype(int)

    return pose_2d_mpii


class HybridPoseEstimator(PoseEstimatorInterface):
    def __init__(self, estimator_2d=None, estimator_3d=None):
        self._estimator_2d = estimator_2d
        self._estimator_3d = estimator_3d

    def estimate(self, image):
        pose_2d = []
        pose_3d = []
        visibility = []

        if self._estimator_2d:
            pose_2d = self._estimator_2d.estimate(image)
            if self._estimator_3d:
                pose_2d, visibility, pose_3d = self.\
                    _estimator_3d.estimate(pose_2d)
            else:
                pose_2d = restructure_2d_pose(pose_2d,
                                              self._estimator_2d.image_width,
                                              self._estimator_2d.image_height)
                visibility = [[not np.array_equal(joint, np.array([0, 0]))
                              for joint in pose] for pose in pose_2d]
                pose_3d = None
        return pose_2d, visibility, pose_3d

    def initialise(self):
        pass

    def close(self):
        pass


class OpPoseEstimatorDecorator(PoseEstimatorInterface):
    def __init__(self, estimator, size, resize, upsample):
        self._estimator = estimator
        self._resize = resize
        self._upsample = upsample
        self._image_height = size[0]
        self._image_width = size[1]

    @property
    def image_width(self):
        return self._image_width

    @property
    def image_height(self):
        return self._image_height

    def estimate(self, image):
        return self._estimator.inference(image, self._resize, self._upsample)

    def initialise(self):
        pass

    def close(self):
        pass


class OpPoseBasedLFTDLifter(PoseEstimatorInterface):
    def __init__(self, model_path):
        self._lifter_3d = Prob3dPose(model_path)

    def estimate(self, pose_2d):
        pose_2d_mpii, visibility = to_mpii_pose_2d(pose_2d)
        transformed_pose_2d, weights = self._lifter_3d.transform_joints(
            np.array(pose_2d_mpii), visibility)
        pose_3d = self._lifter_3d.compute_3d(transformed_pose_2d, weights)

        pose_2d_mpii[0][:, 0] = pose_2d_mpii[0][:, 0] * 480
        pose_2d_mpii[0][:, 1] = pose_2d_mpii[0][:, 1] * 640

        """
        aux_y = transformed_pose_2d[0][:,1]*480
        aux_x = transformed_pose_2d[0][:,0]*640
        transformed_pose_2d[0][:,0] = aux_y
        transformed_pose_2d[0][:,1] = aux_x
        """

        pose_2d_mpii = pose_2d_mpii.astype(int)
        #transformed_pose_2d = transformed_pose_2d.astype(int)

        return pose_2d_mpii, visibility, pose_3d

    def initialise(self):
        pass

    def close(self):
        pass


def OpPoseLFTDEstimator(image_size, lifter_model_path, resize=True,
                        upsample=2.0):
    open_pose_estimator = OpPoseEstimator(get_graph_path('cmu'),
                                          target_size=(image_size[1],
                                                       image_size[0]))

    estimator_2d = OpPoseEstimatorDecorator(open_pose_estimator, image_size,
                                            resize, upsample)
    estimator_3d = OpPoseBasedLFTDLifter(lifter_model_path)

    est = HybridPoseEstimator(estimator_2d, estimator_3d)

    return est


def PlainOpPoseEstimator(image_size, lifter_model_path, resize=True,
                         upsample=2.0):
    open_pose_estimator = OpPoseEstimator(get_graph_path('cmu'),
                                          target_size=(image_size[1],
                                                       image_size[0]))

    estimator_2d = OpPoseEstimatorDecorator(open_pose_estimator, image_size,
                                            resize, upsample)
    estimator_3d = None

    est = HybridPoseEstimator(estimator_2d, estimator_3d)

    return est
