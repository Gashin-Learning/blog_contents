import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_toy_data(theta, x_dim, y_dim, z_dim, plot_option=True, my_seed=0):

    np.random.seed(my_seed)
    # set parameter of bernouli distributions
    theta=theta

    R_raw = np.concatenate([np.concatenate([np.concatenate([np.random.binomial(1, p_ijk, size=(x_dim,y_dim,z_dim)) for p_ijk in p_ij], axis=2) for p_ij in p_i], axis=1) for p_i in theta])

    # shuffle x y z
    R_X_index = np.random.permutation(np.arange(R_raw.shape[0]))
    R_Y_index = np.random.permutation(np.arange(R_raw.shape[1]))
    R_Z_index = np.random.permutation(np.arange(R_raw.shape[2]))
    R = R_raw[R_X_index,:,:]
    R = R[:, R_Y_index, :]
    R = R[:, :, R_Z_index]

    if plot_option=='scatter':
        fig = plt.figure(figsize=(14,5))
        ax0 = fig.add_subplot(121, projection='3d')
        for x in range(R_raw.shape[0]):

            R_raw_per_x = R_raw[x,:,:]
            x1=np.arange(R_raw_per_x.shape[1])
            x2=np.arange(R_raw_per_x.shape[0])
            X1, X2 = np.meshgrid(x1, x2)
            X1_on = np.ravel(X1)[np.ravel(R_raw_per_x)==1]
            X2_on = np.ravel(X2)[np.ravel(R_raw_per_x)==1]
            ax0.scatter3D(np.ravel(X1_on), np.ravel(X2_on), x, marker="o", linestyle='None', color='purple')

        ax1 = fig.add_subplot(122, projection='3d')
        for x in range(R.shape[0]):
            R_per_x = R[x,:,:]
            x1=np.arange(R_per_x.shape[1])
            x2=np.arange(R_per_x.shape[0])
            X1, X2 = np.meshgrid(x1, x2)
            X1_on = np.ravel(X1)[np.ravel(R_per_x)==1]
            X2_on = np.ravel(X2)[np.ravel(R_per_x)==1]

            ax1.scatter3D(np.ravel(X1_on), np.ravel(X2_on), x, marker="o",linestyle='None', color='purple')

        ax0.set_title("R_raw", fontsize=15)
        ax1.set_title("Shuffled as Original data", fontsize=15);

    else:
        fig = plt.figure(figsize=(16,6))

        colors = np.where(R_raw==1,'#00bfffC0', '#00000000')
        voxels1=R_raw==1
        voxels2=R_raw==0
        ax0 = fig.add_subplot(121, projection='3d')
        ax0 = fig.gca(projection='3d')
        ax0.voxels(voxels1, facecolors=colors, edgecolor='#00008b73')
        ax0.voxels(voxels2, facecolors=colors, edgecolor='#ff8c0073')
        ax0.set_title("Raw data", fontsize=20)

        colors = np.where(R==1,'#00bfffC0', '#00000000')
        voxels1=R==1
        voxels2=R==0
        ax1 = fig.add_subplot(122, projection='3d')
        ax1 = fig.gca(projection='3d')
        ax1.voxels(voxels1, facecolors=colors, edgecolor='#00008b73')
        ax1.voxels(voxels2, facecolors=colors, edgecolor='#ff8c0073')
        ax1.set_title("Shuffled", fontsize=20);

    return R
