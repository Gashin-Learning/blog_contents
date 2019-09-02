import numpy as np
import math
from collections import Counter
from scipy import stats as st
from scipy.special import beta
from scipy.special import betaln
from functools import reduce
from operator import mul

# chinese restaurant process
def CRP(alpha, sample_num, my_seed=0):
    np.random.seed(my_seed)
    s = [0]
    for _ in range(sample_num-1):
        n = len(s)
        p_exist = np.bincount(s, minlength=np.max(s))
        p_new = alpha
        p = np.append(p_exist, p_new)/(n+alpha)
        sample_s = np.argmax(np.random.multinomial(n=1, pvals=p, size=1))
        s.append(sample_s)
    return np.array(s)

# Ewens formila for calculating p(s)
def Ewens_sampling_formula(s, alpha):
    n = len(s)
    c = len(np.unique(s))
    count_c = np.bincount(s)
    AF = reduce(mul, [alpha+i for i in range(n)])
    n_1_factorials = np.array(list(map(math.factorial, np.bincount(s)-1)))
    p_s = (alpha**c) * np.prod(n_1_factorials)/ AF
    return p_s

# if category_num is decreased, fill empty category number
def reset_s_number(s):
    count_category = np.bincount(s, minlength=np.max(s))
    empty_num = np.where(count_category==0)[0]
    if len(empty_num)>=1:
        s[s >= empty_num[0]]-=1
    return s

# count data num per clusters
def count_n_ijk(s1, s2, s3):
    num_n_ijk_matrix = np.zeros((max(s1)+1, max(s2)+1, max(s3)+1))
    count_n_ijk = Counter([(i, j, k) for i in s1 for j in s2 for k in s3])
    for (i, j, k), num_ijk in count_n_ijk.items():
        num_n_ijk_matrix[i, j, k] = num_ijk
    return num_n_ijk_matrix

# count num of 1 and 0 per cluster categorys in R_2Dmatrix conditioned one axis
def count_one_zero_2D(R_per_sorted, sorted_s_first, sorted_s_second):

    switch_s_first_id = np.hstack([np.array([0]), np.diff(sorted_s_first)])
    switch_s_second_id = np.hstack([np.array([0]), np.diff(sorted_s_second)])
    switch_s_first_idx = np.hstack([np.array([0]), np.where(switch_s_first_id==1)[0]])
    switch_s_second_idx = np.hstack([np.array([0]), np.where(switch_s_second_id==1)[0]])

    sum_one_across_an_axis = np.add.reduceat(R_per_sorted, switch_s_first_idx, axis=0)
    R_ijk_1 = np.add.reduceat(sum_one_across_an_axis, switch_s_second_idx, axis=1)
    sum_all_across_an_axis = np.add.reduceat(np.ones(shape=R_per_sorted.shape), switch_s_first_idx, axis=0)
    R_ijk_all = np.add.reduceat(sum_all_across_an_axis, switch_s_second_idx, axis=1)

    R_ijk_0 = R_ijk_all - R_ijk_1
    return R_ijk_1, R_ijk_0

# count num of 1 and 0 per cluster categorys in R_3Dmatrix
def count_one_zero_3D(R_per_sorted, sorted_s_first, sorted_s_second, sorted_s_third):

    switch_s_first_id = np.hstack([np.array([0]), np.diff(sorted_s_first)])
    switch_s_second_id = np.hstack([np.array([0]), np.diff(sorted_s_second)])
    switch_s_third_id = np.hstack([np.array([0]), np.diff(sorted_s_third)])
    switch_s_first_idx = np.hstack([np.array([0]), np.where(switch_s_first_id==1)[0]])
    switch_s_second_idx = np.hstack([np.array([0]), np.where(switch_s_second_id==1)[0]])
    switch_s_third_idx = np.hstack([np.array([0]), np.where(switch_s_third_id==1)[0]])

    sum_one_across_an_axis = np.add.reduceat(R_per_sorted, switch_s_first_idx, axis=0)
    sum_one_across_two_axes = np.add.reduceat(sum_one_across_an_axis, switch_s_second_idx, axis=1)
    R_ijk_1 = np.add.reduceat(sum_one_across_two_axes, switch_s_third_idx, axis=2)
    sum_all_across_an_axis = np.add.reduceat(np.ones(shape=R_per_sorted.shape), switch_s_first_idx, axis=0)
    sum_all_across_two_axes = np.add.reduceat(sum_all_across_an_axis, switch_s_second_idx, axis=1)
    R_ijk_all = np.add.reduceat(sum_all_across_two_axes, switch_s_third_idx, axis=2)

    R_ijk_0 = R_ijk_all - R_ijk_1
    return R_ijk_1, R_ijk_0

# calculate bernouli parameter as the mean of beta posterior distribution
# return mean
def posterier_theta(s1, s2, s3,  R, a, b):

    sorted_s1_index = s1.argsort()
    sorted_s2_index = s2.argsort()
    sorted_s3_index = s3.argsort()
    sorted_s1 = s1[sorted_s1_index]
    sorted_s2 = s2[sorted_s2_index]
    sorted_s3 = s3[sorted_s3_index]
    R_sorted = R[sorted_s1_index,:,:][:,sorted_s2_index,:][:,:,sorted_s3_index]

    R_ijk_1, R_ijk_0 = count_one_zero_3D(R_sorted, sorted_s1, sorted_s2, sorted_s3)

    a_update = a + R_ijk_1
    b_update = b + R_ijk_0
    theta=a_update/(a_update + b_update)
    return theta

# log P(R|sx, sy, sz) marginalize parameter
def log_R_probability(s1, s2, s3,  R, a, b):

    sorted_s1_index = s1.argsort()
    sorted_s2_index = s2.argsort()
    sorted_s3_index = s3.argsort()
    sorted_s1 = s1[sorted_s1_index]
    sorted_s2 = s2[sorted_s2_index]
    sorted_s3 = s3[sorted_s3_index]
    R_sorted = R[sorted_s1_index,:,:][:,sorted_s2_index,:][:,:,sorted_s3_index]

    R_ijk_1, R_ijk_0 = count_one_zero_3D(R_sorted, sorted_s1, sorted_s2, sorted_s3)
    a_update = a + R_ijk_1
    b_update = b + R_ijk_0
    return np.sum(betaln(a_update, b_update)-betaln(a,b))

# log P(R|theta, sx, sy, sz)
def log_R_theta_probability(s1, s2, s3,  R, theta):

    sorted_s1_index = s1.argsort()
    sorted_s2_index = s2.argsort()
    sorted_s3_index = s3.argsort()
    sorted_s1 = s1[sorted_s1_index]
    sorted_s2 = s2[sorted_s2_index]
    sorted_s3 = s3[sorted_s3_index]
    R_sorted = R[sorted_s1_index,:,:][:,sorted_s2_index,:][:,:,sorted_s3_index]

    R_ijk_1, R_ijk_0 = count_one_zero_3D(R_sorted, sorted_s1, sorted_s2, sorted_s3)

    return np.sum(R_ijk_1 * np.log(theta))+ np.sum(R_ijk_0 * np.log(1-theta))


# update s
def s_update(s1, s2, s3, theta, R, a, b, alpha, axis):

    if axis==1:
        theta = theta.transpose((1,2,0))
    elif axis==2:
        theta = theta.transpose((2,0,1))
    # sort orderby s2,s3 for easy calculation
    sorted_s2_index = s2.argsort()
    sorted_s3_index = s3.argsort()
    sorted_s2 = s2[sorted_s2_index]
    sorted_s3 = s3[sorted_s3_index]
    R_sorted = R[:,sorted_s2_index,:][:,:,sorted_s3_index]

    for idx in range(len(s1)):
        # remove s1_x for gibbs sampling
        s1_delete = s1[idx]
        s1_left = np.delete(s1, idx)
        theta_left = theta[np.unique(s1_left),:, :]

        # if category_num is decreased, fill empty category number
        s1_left = reset_s_number(s1_left)

        # count n_ij
        num_n_ijk_left = count_n_ijk(s1_left, s2, s3)

        # log_p(s1_k | s1_left) by Dirichlet Process
        n_i = np.add.reduce(num_n_ijk_left, axis=(1,2))
        n_i = np.append(n_i, alpha)
        ln_p_s1_idx_s1_left = np.log(n_i/(np.add.reduce(num_n_ijk_left,axis=(0,1,2)) + alpha))

        # log_p(R| s1_left, s2, s3)
        R_idx_sorted =R_sorted[idx,:,:]
        R_ijk_1, R_ijk_0 = count_one_zero_2D(R_idx_sorted, sorted_s2, sorted_s3)
        ln_p_R_xyz_new = np.sum(betaln(R_ijk_1+a, R_ijk_0 +b) - betaln(a,b))
        ln_p_R_xyz_exist= np.sum(R_ijk_1 * np.log(theta_left), axis=(1,2))+ np.sum(R_ijk_0 * np.log(1-theta_left),axis=(1,2))

        # ratio for choosing new s1_x '+100' is for preventing underflow
        p_s1_idx = np.exp(ln_p_s1_idx_s1_left + np.append(ln_p_R_xyz_exist, ln_p_R_xyz_new)+100)
        p_s1_idx/=np.sum(p_s1_idx)
        s_new = np.argmax(np.random.multinomial(n=1, pvals=p_s1_idx))

        # new s1 updated
        s1 = np.insert(s1_left, idx, s_new)

        # update theta
        theta = posterier_theta(s1, s2, s3, R, a,b)

    if axis==1:
        theta = theta.transpose((2,0,1))
    elif axis==2:
        theta = theta.transpose((1,2,0))

    return s1, theta


# gibbs sampling
def predict_S(R, alpha,a,b, iter_num=500, reset_iter_num=100, my_seed=0):

    np.random.seed(my_seed)

    X, Y, Z = R.shape

    # set first values  ##########################
    sx = CRP(alpha=alpha, sample_num=X, my_seed)
    sy = CRP(alpha=alpha, sample_num=Y, my_seed)
    sz = CRP(alpha=alpha, sample_num=Z, my_seed)
    theta = posterier_theta(sx, sy, sz, R, a, b)
    ##############################################

    max_v = -np.inf
    # to recycle 'def s_update'
    R_transpose_y = R.transpose((1,2,0))
    R_transpose_z = R.transpose((2,0,1))

    # gibbs sampling
    for t in range(iter_num):
        print("\r calculating... t={}".format(t), end="")
        sx, theta = s_update(sx, sy, sz, theta, R, a, b, alpha, axis=0)
        sy, theta = s_update(sy, sz, sx, theta, R_transpose_y, a, b, alpha, axis=1)
        sz, theta = s_update(sz, sx, sy, theta, R_transpose_z, a, b, alpha, axis=2)

        log_p_sx = np.log(Ewens_sampling_formula(sx, alpha))
        log_p_sy = np.log(Ewens_sampling_formula(sy, alpha))
        log_p_sz = np.log(Ewens_sampling_formula(sz, alpha))
        log_p_theta = np.sum(st.beta.logpdf(theta, a,b))

        log_p_R_theta = log_R_theta_probability(sx, sy, sz,  R, theta)
        #log_p_R_ijk = log_R_probability(sx, sy, sz,  R, a, b)

        # logP(sx, sy, sz, theta| R)
        v = log_p_sx + log_p_sy + log_p_sz + log_p_theta + log_p_R_theta

        # update if over max
        if v > max_v:
            max_v = v
            max_sx = sx
            max_sy = sy
            max_sz = sz
            max_theta = theta
            print("  update S and theta : logP(sx, sy, sz, theta| R) = ", v)

        # to prevent getting stuck local minima, reset S and theta
        if t%reset_iter_num==0:
            sx = CRP(alpha=alpha, sample_num=X, my_seed)
            sy = CRP(alpha=alpha, sample_num=Y, my_seed)
            sz = CRP(alpha=alpha, sample_num=Z, my_seed)
            theta = posterier_theta(sx, sy, sz, R, a, b)

    return max_sx, max_sy, max_sz, max_theta
