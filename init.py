import numpy as np
import math as m
import torch
from torch.autograd import Variable
from utils import *


def distance(p, x):
    # geodesic distance from point p to data x
    if x.ndim == 2:
        data1 = x[0, :]
        data2 = x[1, :]
    elif x.ndim == 1:
        data1 = x[0]
        data2 = x[1]
    else:
        raise AssertionError("input dim should be 1 or 2")
    d = np.cos(data1) * np.cos(p[0]) + np.sin(data1) * np.sin(p[0]) * np.cos(data2 - p[1])
    if x.ndim == 1:
        if d > 1:
            d = 1.
    output = np.arccos(d)
    return output


def circle_distance(p1, p2, p3, x):
    # x : x[0,:] lattitude, x[1,:] longitude

    data1 = x[0, :]
    data2 = x[1, :]
    p1_ = p1.expand_as(data1)
    p2_ = p2.expand_as(data2)

    output = torch.acos(torch.cos(data1) * torch.cos(p1_) + torch.sin(data1) * torch.sin(p1_) * torch.cos(data2 - p2_))
    p3_ = p3.expand_as(output)

    return torch.mean((m.pi / 2 * p3_ - output) ** 2)


def intrinsic_mean(points, weights=None, is_print=False, intrinsic=True):
    # calculate intrinsic mean by gradient descent
    N = points.shape[1]
    if weights is None:
        weights = 1. / N
    else:
        weights = weights / np.sum(weights)
        weights = np.expand_dims(weights, axis=0)

    if intrinsic is True:
        u = np.mean(weights * points, axis=1)
        u = u / np.sqrt(np.sum(u ** 2))

        for i in range(100):
            delta_u = np.sum(weights * log_map(u, points), axis=1)
            u = exp_map(u, delta_u)

            if sum(delta_u ** 2) < 1e-10:
                if is_print is True:
                    print("optimization is ended (iter : %d) \n" % (i + 1))
                    print("intrinsic mean : ", u, '\n')
                break
    else:
        u = np.mean(weights * points, axis=1)
        u = u / np.sqrt(np.sum(u ** 2))

    return u


def pga_circle(x_theta, is_print=False):
    # optimize distance between data and circle represented by 3 parameters (center, latitude)
    x_theta_torch = Variable(torch.DoubleTensor(x_theta), requires_grad=True)
    p = Variable(torch.DoubleTensor([0, 0, 1]), requires_grad=True)
    lr = 1

    if is_print is True:
        print("iteration start")
        print("initial value")
        print("p : ", p.data.numpy())

    for i in range(100):
        loss = circle_distance(p[0], p[1], p[2], x_theta_torch)
        loss.backward()
        if is_print is True:
            print("\niteration : %d" % (i + 1))
            print("loss : %f" % loss.data.numpy())
            print("p : ", p.data.numpy())
            print(p.grad)

        p.data[:2] -= lr * p.grad.data[:2]
        p.data[2] -= 0.05 * p.grad.data[2]

        if np.sum(np.abs(p.grad.data.numpy())) < 1e-4:
            if is_print is True:
                print("\noptimization is succeeded, iteration : %d" % i)
                print("p : ", p.data.numpy())
            break

        p.grad.data.zero_()

    # p3 is longitude of circle
    p3 = (p.data.numpy()[2]) * m.pi / 2

    # p_ is center of circle
    p_ = np.array([np.sin(p.data.numpy()[0]) * np.cos(p.data.numpy()[1]),
                   np.sin(p.data.numpy()[0]) * np.sin(p.data.numpy()[1]), np.cos(p.data.numpy()[0])])

    return p_, p3