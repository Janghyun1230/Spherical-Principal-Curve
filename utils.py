import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rad2euclid(x):
    # change data representation from radian to euclidian
    if x.ndim == 2:
        t1 = x[0, :]
        t2 = x[1, :]
        output = np.vstack((np.sin(t1) * np.cos(t2), np.sin(t1) * np.sin(t2), np.cos(t1)))
    elif x.ndim == 1:
        t1 = x[0]
        t2 = x[1]
        output = np.array([np.sin(t1) * np.cos(t2), np.sin(t1) * np.sin(t2), np.cos(t1)])
    else:
        raise AssertionError("input dim should be 1 or 2")
    return output


def euclid2rad(x):
    # change data representation from radian to euclidian
    if x.ndim == 2:
        x[2, :][x[2, :] > 1] = 1.
        output = np.vstack((np.arccos(x[2, :]), np.arctan2(x[1, :], x[0, :])))
    elif x.ndim == 1:
        if x[2] > 1:
            x[2] = 1.
        output = np.array([np.arccos(x[2]), np.arctan2(x[1], x[0])])
    else:
        raise AssertionError("input dim should be 1 or 2")
    return output


def rotate(p, x):
    # rotation from p -> (0,0,1)
    # x -> output
    if (p[0] == 0) & (p[1] == 0):
        c = 1
        s = 0
    else:
        c = p[0] / (np.sqrt(p[0] ** 2 + p[1] ** 2))
        s = p[1] / (np.sqrt(p[0] ** 2 + p[1] ** 2))

    R1 = np.array([[c, s, 0],
                   [-s, c, 0],
                   [0, 0, 1]])
    # R! makes p to [*, 0, *]

    xy = np.sqrt(p[0] ** 2 + p[1] ** 2)
    z = p[2]
    c2 = z / np.sqrt(z ** 2 + xy ** 2)
    s2 = xy / np.sqrt(z ** 2 + xy ** 2)

    R2 = np.array([[c2, 0, -s2],
                   [0, 1, 0],
                   [s2, 0, c2]])
    # R2, R1 makes p to [0,0,1]

    output = np.dot(R2, np.dot(R1, x))

    return output


def inv_rotate(p, x):
    # rotation from (0,0,1) -> p
    # x -> output
    if (p[0] == 0) & (p[1] == 0):
        c = 1
        s = 0
    else:
        c = p[0] / (np.sqrt(p[0] ** 2 + p[1] ** 2))
        s = p[1] / (np.sqrt(p[0] ** 2 + p[1] ** 2))

    R1 = np.array([[c, s, 0],
                   [-s, c, 0],
                   [0, 0, 1]])

    xy = np.sqrt(p[0] ** 2 + p[1] ** 2)
    z = p[2]
    c2 = z / np.sqrt(z ** 2 + xy ** 2)
    s2 = xy / np.sqrt(z ** 2 + xy ** 2)

    R2 = np.array([[c2, 0, -s2],
                   [0, 1, 0],
                   [s2, 0, c2]])

    output = np.dot(np.linalg.inv(np.dot(R2, R1)), x)

    return output


def exp_map(p, x):
    # tangent space to sphere at p
    eps = 1e-5
    if x.ndim == 1:
        mag = np.sqrt(np.sum(x ** 2)) + eps
        exp_x = np.array([x[0] * np.sin(mag) / mag, x[1] * np.sin(mag) / mag, np.cos(mag)])
        output = inv_rotate(p, exp_x)

    elif x.ndim == 2:
        mag = np.sqrt(np.sum(x ** 2, axis=0)) + eps
        exp_x = np.vstack((x[0, :] * np.sin(mag) / mag, x[1, :] * np.sin(mag) / mag, np.cos(mag)))
        output = inv_rotate(p, exp_x)
    else:
        raise AttributeError("wrong dimension of x")

    return output


def log_map(p, x):
    # sphere to tangent space at p
    rx = np.clip(rotate(p,x), a_min= -1, a_max= 1)
    eps = 1e-9
    if rx.ndim == 2:
        theta = np.arccos(rx[2, :]) + eps
        output = np.vstack((rx[0, :] * theta / np.sin(theta), rx[1, :] * theta / np.sin(theta)))
    elif rx.ndim == 1:
        theta = np.arccos(rx[2]) + eps
        output = np.array((rx[0] * theta / np.sin(theta), rx[1] * theta / np.sin(theta)))
    else:
        raise AttributeError("wrong dimension of x")

    return output


def plot_sphere(x, title, extra=None, line=False, extra2=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    # plot sphere
    u1, v1 = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x_ = np.cos(u1) * np.sin(v1)
    y_ = np.sin(u1) * np.sin(v1)
    z_ = np.cos(v1)
    ax.plot_wireframe(x_, y_, z_, color="g", linewidth=0.5)

    # plot data
    ax.scatter(x[0,], x[1,], x[2,], c='b', s=4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    # plot extra data or line
    if extra is not None:
        if extra.ndim == 1:
            ax.scatter(extra[0], extra[1], extra[2], c='r', s=20)
        elif extra.ndim == 2:
            if line is False:
                ax.scatter(extra[0, :], extra[1, :], extra[2, :], c='r', s=4)
            else:
                ax.plot(extra[0, :], extra[1, :], extra[2, :], c='r', linewidth=1)

    if extra2 is not None:
        if extra2.ndim == 1:
            ax.scatter(extra2[0], extra2[1], extra2[2], c='m', s=20)
        elif extra2.ndim == 2:
            if line is False:
                ax.scatter(extra2[0, :], extra2[1, :], extra2[2, :], c='m', s=4)
            else:
                ax.plot(extra2[0, :], extra2[1, :], extra2[2, :], c='m', linewidth=1)
    plt.draw()

