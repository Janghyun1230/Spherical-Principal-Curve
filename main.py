import pandas as pd
from principal_curves import *
import time
import matplotlib.pyplot as plt
import os
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train(data_path, save=True, q=0.3, n_point=100, intrinsic=False, exact=True):
    # load dataset
    data= pd.read_csv(data_path)

    # select latitude and longitude column
    THETA= np.array(np.vstack((data['latitude'].values, data['longitude'].values))/360.*m.pi*2)
    t1= m.pi/2- THETA[0]
    t2= THETA[1] + 5*m.pi/8

    x= np.vstack((np.sin(t1)*np.cos(t2), np.sin(t1)*np.sin(t2) ,np.cos(t1)))
    x_theta= np.vstack((t1,t2))
    data_len= x.shape[1]

    print(THETA.shape)
    plot_sphere(x,"earthquake data")

    # initialization (principal circle)
    p_, p3 = pga_circle(x_theta, False)

    # initial projection
    init_project = np.zeros(shape=[2, n_point])
    init_project[0, :] = p3
    init_project[1, :] = np.arange(0, 2*m.pi, 2*m.pi/n_point)
    init_project = inv_rotate(p_, rad2euclid(init_project))

    # iteration
    start_time= time.time()
    points, _, _ = train_principal_curve(x, init_project, iter=100, neigh_ratio=q, intrinsic=intrinsic, exact=exact)
    end_time= time.time()
    print("training time %.3f s"%(end_time-start_time))

    if save:
        np.save("results/{}_fitted.npy".format(os.path.splitext(os.path.basename(data_path))[0]),points)
        print("fitted curve is saved as results/{}_fitted.npy".format(os.path.splitext(os.path.basename(data_path))[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', type=float, default=0.05, help='neighborhood ratio')
    parser.add_argument('-n', type=int, default=77, help='number of curve points')
    parser.add_argument('-i', '--intrinsic', type=str2bool, default=False, help='intrinsic or extrinsic')
    parser.add_argument('-e', '--exact', type=str2bool, default=True, help='exact projection or not')

    args = parser.parse_args()

    train(data_path="data/earthquake_new.csv", q=args.q, n_point=args.n, intrinsic=args.intrinsic, exact=args.exact)
    plt.show()