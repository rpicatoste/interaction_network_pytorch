import os
import numpy as np
from math import sin, cos, radians, pi
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


G = 10**4

MASS = 0
POS_X = 1
POS_Y = 2
V_X = 3
V_Y = 4

POS = [POS_X, POS_Y]
VEL = [V_X, V_Y]

MASS_POS = [MASS] + POS
MECHANICS = POS + VEL
STATE = [MASS] + MECHANICS

N_FEATURES = len(STATE)
BODY_SHAPE = (N_FEATURES,)


def generate_data_with_simulator(n_objects, orbit, time_steps, dt):
    print(f'Generating data with {time_steps} time steps for {n_objects} objects.')
    data = get_init_state(time_steps, n_objects, orbit)

    for ii in range(1, time_steps):
        data[ii] = calculate_next_state(data[ii - 1], n_objects, dt)

    return data


def get_init_state(time_steps, n_body, orbit):
    data = np.zeros((time_steps, n_body, N_FEATURES), dtype=float)

    if orbit:
        data[0, 0] = get_central_mass()

    for body_i in range(1 if orbit else 0, n_body):
        data[0, body_i] = get_random_body(orbit=orbit, central_mass=data[0, 0])

    return data


def get_central_mass():
    body_data = np.zeros(BODY_SHAPE, dtype=np.float)
    body_data[MASS] = 10000.0
    body_data[MECHANICS] = 0.0

    return body_data


def get_random_body(orbit, central_mass=None):
    body_data = np.zeros(BODY_SHAPE, dtype=np.float)
    
    body_data[0] = np.random.rand() * 8.98 + 0.02
    distance = np.random.rand() * 90.0 + 10.0
    theta = np.random.rand() * 360
    theta_rad = pi / 2 - radians(theta)

    body_data[POS_X] = distance * cos(theta_rad)
    body_data[POS_Y] = distance * sin(theta_rad)

    if orbit:
        # https://www.physicsclassroom.com/class/circles/Lesson-4/Mathematics-of-Satellite-Motion
        v = np.sqrt(G * central_mass[MASS] / distance)

        relative_pos = body_data[POS] - central_mass[POS]
        pos_direction = relative_pos / np.linalg.norm(relative_pos)
        v_direction = np.array([pos_direction[1], -pos_direction[0]])
        assert (np.linalg.norm(v_direction) - 1.0) < 0.01

        body_data[V_X] = v*v_direction[0]
        body_data[V_Y] = v*v_direction[1]
    else:
        mean_speed = 20.0
        body_data[V_X] = np.random.randn() * mean_speed
        body_data[V_Y] = np.random.randn() * mean_speed

    return body_data


def get_gravitational_force(receiver, sender):
    relative_pos = sender[POS] - receiver[POS]
    distance = np.linalg.norm(relative_pos)
    if distance < 1.0:
        distance = 1

    force = G * receiver[MASS] * sender[MASS] / (distance ** 3) * relative_pos

    return force


def calculate_next_state(current_state, n_bodies, dt):
    next_state = np.zeros((n_bodies, N_FEATURES), dtype=float)

    forces_matrix = np.zeros((n_bodies, n_bodies, 2), dtype=float)
    forces_sum = np.zeros((n_bodies, 2), dtype=float)
    acc = np.zeros((n_bodies, 2), dtype=float)

    for body_i in range(n_bodies):

        for body_j in range(body_i + 1, n_bodies):
            assert body_j != body_i
            f = get_gravitational_force(current_state[body_i], current_state[body_j])
            forces_matrix[body_i, body_j] += f
            forces_matrix[body_j, body_i] -= f

        forces_sum[body_i] = np.sum(forces_matrix[body_i], axis=0)
        acc[body_i] = forces_sum[body_i] / current_state[body_i][MASS]
        next_state[body_i][MASS] = current_state[body_i][MASS]
        next_state[body_i][VEL] = current_state[body_i][VEL] + acc[body_i] * dt
        next_state[body_i][POS] = current_state[body_i][POS] + next_state[body_i][VEL] * dt

    return next_state


def make_video(data, filename):
    time_steps = len(data)
    n_bodies = len(data[0])

    print(f'Generating videos for data with shape {data.shape}.')

    x_min = np.min(data[:, :, POS_X])
    x_max = np.max(data[:, :, POS_X])
    y_min = np.min(data[:, :, POS_Y])
    y_max = np.max(data[:, :, POS_Y])
    x_delta = x_max - x_min
    x_min -= x_delta * 0.05
    x_max += x_delta * 0.05
    y_delta = y_max - y_min
    y_min -= y_delta * 0.05
    y_max += y_delta * 0.05

    os.system("rm -rf pics/*")
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=30, metadata=metadata)

    color = ['ro','bo','go','ko','yo','mo','co']
    marker_sizes = [5] + [3]*6

    fig = plt.figure()
    plt.ylim(x_min, x_max)
    plt.xlim(y_min, y_max)

    with writer.saving(fig, filename, time_steps):
        for frame in range(time_steps):
            if frame%(time_steps/5)==0:
                print(f'Frame {frame} ...')

            for body_i in range(n_bodies):
                modifier_idx = body_i % len(color)
                plt.plot(data[frame, body_i, POS_Y],
                         data[frame, body_i, POS_X],
                         color[modifier_idx],
                         markersize=marker_sizes[modifier_idx])
            writer.grab_frame()

    print(f'Video done.')

if __name__ == '__main__':
    data = generate_data_with_simulator(n_objects=5, orbit=True, time_steps=30 * 2, dt=0.001)
    make_video(data, "test.mp4")
