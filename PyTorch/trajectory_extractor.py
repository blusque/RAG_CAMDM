import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.bvh_motion import Motion
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pickle as pkl

import os
from os.path import join as pjoin

BASE_PATH = os.path.abspath(os.curdir)

def align_trajectory_with_given_direction(trajectory: np.ndarray, rotations: np.ndarray, target_direction, frame_idx=0, mod=True):
    if mod:
        frame_idx = frame_idx % trajectory.shape[0]
    trajectory = trajectory.copy()
    target_direction = target_direction / np.linalg.norm(target_direction)
    origin_rot = rotations[frame_idx]
    rot = np.arctan2(target_direction[1], target_direction[0]) - origin_rot

    rot_matrix = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
    trajectory = np.dot(trajectory, rot_matrix.T)

    rotations += rot
    
    return trajectory, rotations


def align_trajectory_with_given_position(trajectory: np.ndarray, target_position, frame_idx=0, mod=True):
    if mod:
        frame_idx = frame_idx % trajectory.shape[0]
    trajectory = trajectory.copy()
    origin_position = trajectory[frame_idx]
    offset = target_position - origin_position
    trajectory += offset
    return trajectory


def align_motion_with_given_direction(motion: Motion, target_direction: np.ndarray, frame_idx=0, mod=True):
    '''
    Align the motion with the target orientation
    Parameters:
        - motion: the motion
        - target_direction: the target orientation in xz plane, shape (2,)
    Return: the aligned motion
    '''
    if mod:
        frame_idx = frame_idx % motion.local_joint_rotations.shape[0]
    res = motion.copy()
    target_direction = target_direction / np.linalg.norm(target_direction)
    trajectory = res.local_joint_positions[:, 0, [0, 2]]
    origin_direction = R.from_quat(res.local_joint_rotations[frame_idx, 0]).apply([0, 0, 1])[[0, 2]]
    rot = np.arctan2(target_direction[0], target_direction[1]) - np.arctan2(origin_direction[0], origin_direction[1])
    
    rot_R = R.from_euler('y', rot)
    res.local_joint_rotations[:, 0] = (rot_R * R.from_quat(res.local_joint_rotations[:, 0])).as_quat()

    rot_matrix = np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
    trajectory = np.dot(trajectory, rot_matrix.T)
    res.local_joint_positions[:, 0, [0, 2]] = trajectory

    return res

    
def align_motion_with_given_position(motion: Motion, target_position: np.ndarray, frame_idx=0, mod=True):
    '''
    Align the motion with the target position
    Parameters:
        - motion: the motion
        - target_position: the target position, shape (2,)
    Return: the aligned motion
    '''
    if mod:
        frame_idx = frame_idx % motion.local_joint_rotations.shape[0]
    res = motion.raw_copy()
    origin_position = res.local_joint_positions[frame_idx, 0, [0, 2]]
    offset = target_position - origin_position
    res.local_joint_positions[:, 0, [0, 2]] += offset
    return res


def get_trajectory(motion: Motion, start=0, end=-1):
    if end == -1:
        return motion.get_joint_positions()[start:, 0, [0, 2]]
    return motion.get_joint_positions()[start:end + 1, 0, [0, 2]]


def plot_trajectory(traj: np.ndarray, traj_rot: np.ndarray, color='Blues', marker='.'):
    # plt.xlim((-3, 3))
    # plt.ylim((-3, 3))
    colors = np.linspace(1, 0.5, len(traj))
    plt.scatter(traj[:, 0], traj[:, 1], c=colors, cmap=color, marker=marker)

def load_data(path):
    data = pkl.load(open(path, 'rb'))
    traj = {}
    traj_rot = {}
    text = []
    for motion in data['motions']:
        text.append(motion['filepath'])
        content = motion['filepath'].split('.')[0].split('_')[-1]
        if content not in traj.keys():
            traj[content] = []
            traj_rot[content] = []
        traj[content] += [motion['traj'][0]]
        traj_rot[content] += [motion['traj_angles'][0]]
    return text, traj, traj_rot


if __name__ == '__main__':
    text, trajs, traj_rots = load_data(pjoin(BASE_PATH, './data/pkls/100style.pkl'))
    # print(text)
    colors = ['Blues', 'Greys', 'Oranges']
    markers = ['.', '+', 'x']
    window_size = 55

    for k in trajs.keys():
        traj_list = trajs[k]
        traj_rot_list = traj_rots[k]

        fig = plt.figure()
        plt.title(k)
        traj_cut = []
        for i in range(len(traj_list)):
            traj = traj_list[i]
            traj_rot = traj_rot_list[i]

        # plot_trajectory(traj, traj_rot, color=colors[0])

        # plt.show()

            # traj_rot = traj_rot[:, np.newaxis]
            # traj = np.concatenate([traj, traj_rot], axis=-1)

            for i in range(0, len(traj), window_size):
                if i + window_size >= len(traj):
                    break

                cut = traj[i : i + window_size]
                cut_rot = traj_rot[i : i + window_size]

                cut, cut_rot = align_trajectory_with_given_direction(cut, cut_rot, np.array([0, 1]))
                cut = align_trajectory_with_given_position(cut, np.array([0, 0]))

                cut_rot = cut_rot[:, np.newaxis]
                cut = np.concatenate([cut, cut_rot], axis=-1)

                traj_cut += [cut.reshape((1, -1))]

        traj_cut = np.concatenate(traj_cut, axis=0)
        traj_rand = np.random.permutation(traj_cut)
        for i in range(3):
            t = traj_rand[i].reshape((window_size, -1))[:, [0, 1]]
            plot_trajectory(t, None, color=colors[i], marker=markers[i])
        # print(traj_cut.shape)

        # tsne = TSNE(n_components=2, random_state=42)
        # traj_cut_proj = tsne.fit_transform(traj_cut)
        # # print(traj_cut_proj.shape)
        # plt.scatter(traj_cut_proj[:, 0], traj_cut_proj[:, 1], c='r', s=1)
        plt.show()

        # print(traj.shape, traj_rot.shape)

        # break


