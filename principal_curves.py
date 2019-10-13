import numpy as np
import matplotlib.pyplot as plt
import math as m
from mpl_toolkits.mplot3d import Axes3D
from utils import *
from init import *


# projection on geodesic segment
def project_seg(x, a, b, exact=True):
    if exact == True:
        # projection of x on a-b
        rot_a = rotate(a, np.stack([x, a, b], axis=1))
        center = rad2euclid(np.array([m.pi / 2., euclid2rad(rot_a)[1, 2] + m.pi / 2.]))
        rot_c = euclid2rad(rotate(center, rot_a))
        x_ = rot_c[:, 0]
        a_ = rot_c[:, 1]
        b_ = rot_c[:, 2]
        x_proj = np.copy(x_)
        x_proj[0] = m.pi / 2.
        dist = np.abs(x_[0] - m.pi / 2.)

        # the longitude angle is from -pi ~ pi
        if (np.maximum(a_[1], b_[1]) - np.minimum(a_[1], b_[1])) > m.pi:
            if (x_proj[1] <= np.minimum(a_[1], b_[1])) or (x_proj[1] > np.maximum(a_[1], b_[1])):
                pass
            else:
                if (x_proj[1] - np.minimum(a_[1], b_[1])) <= (np.maximum(a_[1], b_[1]) - x_proj[1]):
                    x_proj[1] = np.minimum(a_[1], b_[1])
                else:
                    x_proj[1] = np.maximum(a_[1], b_[1])
        else:
            if (x_proj[1] > np.minimum(a_[1], b_[1])) and (x_proj[1] <= np.maximum(a_[1], b_[1])):
                pass
            else:
                if x_proj[1] >= 0:
                    if (x_proj[1] - np.maximum(a_[1], b_[1])) > (2 * m.pi - (x_proj[1] - np.minimum(a_[1], b_[1]))):
                        x_proj[1] = np.minimum(a_[1], b_[1])
                    else:
                        x_proj[1] = np.maximum(a_[1], b_[1])
                else:
                    if (2 * m.pi - (np.maximum(a_[1], b_[1]) - x_proj[1])) > (np.minimum(a_[1], b_[1]) - x_proj[1]):
                        x_proj[1] = np.minimum(a_[1], b_[1])
                    else:
                        x_proj[1] = np.maximum(a_[1], b_[1])

        dist = distance(x_, x_proj)
        x_proj = inv_rotate(a, inv_rotate(center, np.expand_dims(rad2euclid(x_proj), axis=1)))
    else:
        x_ = euclid2rad(x)
        a_ = euclid2rad(a)
        b_ = euclid2rad(b)
        if distance(x_, a_) <= distance(x_, b_):
            dist = distance(x_, a_)
            x_proj = a
        else:
            dist = distance(x_, b_)
            x_proj = b
    return dist, x_proj


# project data on curve
# ouput order is same as data order
def project_curve(data, points, exact=True):
    n_data = data.shape[1]
    n_points= points.shape[1]
    dist_arr = []
    projs = []
    lambda_arr = []
    geodesic_arr = [0]
    
    points_rad = euclid2rad(points)
    # curve의 points들의 geodesic거리 list
    for i in range(n_points-1):            
        geodesic_arr.append(geodesic_arr[i] + distance(points_rad[:,i], points_rad[:,i+1]))
    curve_len = geodesic_arr[-1] + distance(points_rad[:,-1], points_rad[:,0])

    for i in range(n_data):
        d_min=10
        proj_min = []
        seg_num= 0
        for j in range(n_points-1):
            d, proj = project_seg(data[:,i], points[:,j], points[:,j+1], exact = exact)
            if d <= d_min:
                d_min = d
                proj_min = np.squeeze(proj).tolist()
                seg_num = j

        dist_arr.append(d_min)
        projs.append(proj_min)
        lambda_arr.append(geodesic_arr[seg_num]+ distance(euclid2rad(points[:,seg_num]), euclid2rad(np.array(proj_min))))
        
    return np.array(dist_arr), np.transpose(np.array(projs)), np.array(lambda_arr), curve_len, geodesic_arr


# updata curve
def expectation(data, points, lambda_arr, geodesic_arr, curve_len, fraction, intrinsic, is_print):
    n_points= points.shape[1]
    n_data= data.shape[1]
    
    thres = fraction * curve_len
    updated_points = np.copy(points)
    if is_print:
        print("curve length: {:.1f}".format(curve_len))
        
    positive = 0
    weight = np.zeros(shape=(n_points, n_data))
    for i in range(n_points):
        for j in range(n_data):
            dist = min(np.abs(geodesic_arr[i] - lambda_arr[j]),
                       curve_len - np.abs(geodesic_arr[i] - lambda_arr[j]))
            weight[i][j] = max(1 - (dist / thres)**2, 0)**2

        if np.sum(weight[i]>0) > 0:
            updated_points[:,i]= intrinsic_mean(data[:,weight[i]>0], weight[i][weight[i]>0], is_print= False, intrinsic=intrinsic)
            positive += np.sum(weight[i]>0)
        else:
            continue
    
    if is_print:
        print("positive data: ", positive//n_points)
        
    return updated_points


def plot_curve(x, points, lambda_arr, d, iter, neigh_ratio, exact=True):
    points = points[:, np.argsort(lambda_arr)]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # draw sphere
    u1, v1 = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x_ = np.cos(u1) * np.sin(v1)
    y_ = np.sin(u1) * np.sin(v1)
    z_ = np.cos(v1)
    ax.plot_wireframe(x_, y_, z_, color="g", linewidth=0.5)

    ax.scatter(x[0,], x[1,], x[2,], c='b', s=4)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    #     if exact ==  True:
    #         ax.set_title("%0.2f fraction ratio"%(neigh_ratio))
    #     else:
    #         ax.set_title("%0.2f fraction ratio/ discrete version"%(neigh_ratio))
    points_add = np.hstack([points, points[:, 0:1]])
    ax.plot(points_add[0, :], points_add[1, :], points_add[2, :], c='m', linewidth=1)
    ax.scatter(points[0, :], points[1, :], points[2, :], c='r', s=3)

    plt.draw()


def train_principal_curve(data, init_project, iter, neigh_ratio=0.35, exact=True, is_print=True, intrinsic=True):
    points= init_project
    d, projected, lambda_arr, curve_len, geodesic_arr = project_curve(data, points, exact = exact)
            
    if is_print==True:
        plot_curve(data, points, geodesic_arr, d, 0, neigh_ratio=neigh_ratio, exact=exact)
        plot_sphere(projected, "projected")
        
    total_dist= sum(d**2)

    for i in range(iter):
        points = expectation(data, points, lambda_arr, geodesic_arr, curve_len, neigh_ratio, intrinsic, is_print)        
        d, projected, lambda_arr, curve_len, geodesic_arr= project_curve(data, points, exact = exact)
                
        if abs((total_dist- sum(d**2))/total_dist) < 1e-2 :
            break
        else :
            total_dist = sum(d**2)
            
        if is_print==True:
            plot_curve(data, points, geodesic_arr, d, i+1, neigh_ratio=neigh_ratio,exact = exact)
            plot_sphere(projected, "projected")
            print("error: {:5.3f}".format(total_dist))
            print("distinct projection : {} / {} ".format(len(np.unique(lambda_arr)), len(lambda_arr)))

    total_dist = sum(d**2)
    print("iteration ended at %d"%i)
    print("total distance is %5.3f"%total_dist)
    print("distinct projection : {} / {} ".format(len(np.unique(lambda_arr)), len(lambda_arr)))
    points = points[:, np.argsort(geodesic_arr)]   
    
    return points, total_dist, lambda_arr


def cv_train_principal_curve(data, init_project, iter, neigh_ratio=0.35, exact=True, is_print=True, cv=5,
                             intrinsic=True):
    # init
    data_len = data.shape[1]
    random_index = np.arange(data_len)
    np.random.shuffle(random_index)
    test_len = int(data_len / cv)
    cv_error = 0
    for k in range(cv):
        test_index = random_index[k * test_len:(k + 1) * test_len]
        test_data = data[:, test_index]
        train_index = [i for i in range(data.shape[1]) if i not in test_index]
        train_data = data[:, train_index]

        d, points, lambda_arr, curve_len = project_curve(train_data, init_project, exact=exact)
        if is_print == True:
            plot_curve(train_data, points, lambda_arr, d, 0)
        total_dist = sum(d ** 2)

        for i in range(iter):
            points = expectation(train_data, points, lambda_arr, curve_len, neigh_ratio, intrinsic)
            d, points, lambda_arr, curve_len = project_curve(train_data, points, exact=exact)
            if is_print == True:
                plot_curve(train_data, points, lambda_arr, d, i + 1)
            if abs((total_dist - sum(d ** 2)) / total_dist) < 1e-2:
                print(k)
                total_dist = sum(d ** 2)
                print("train total distance is %5.3f" % total_dist)
                d, points, lambda_arr, curve_len = project_curve(test_data, points, exact=exact)
                test_dist = sum(d ** 2)
                cv_error += test_dist
                print("test total distance is %5.3f" % test_dist)
                break
            else:
                total_dist = sum(d ** 2)

    print("cv error : %5.3f" % (cv_error / cv))
    return points, cv_error / cv