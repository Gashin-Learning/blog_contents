import numpy as np
import math

from scipy import special as sp
from scipy import stats as st


def predict_s(x, K=3, iter_num=20, my_seed=0):

    #set seed
    np.random.seed(seed=my_seed)

    # sample num
    N=len(x)

    ### prior parameters
    # first value gamma distribution
    a=200
    b=5

    # first value parameter of Dirichlet distribution
    alpha = np.array([30,20,10])

    # parameter of gammma distribution
    a_update = np.array([200,200,200])
    b_update = np.array([5,5,5])

    # parameter of Dirichlet distribution for \pi
    alpha_update = np.array([30,20,10])

    # set s first value
    s_mean=[]
    for n in range(N):
        s_mean.append([0.4,0.3,0.3])
    s_mean = np.array(s_mean)


    for i in range(iter_num):
        #print("\r Iteration：{}".format(i))

        #####################################################
        # expectation of λ、lnλ、π

        lam_mean=np.zeros(K)
        ln_lam_mean=np.zeros(K)
        ln_pi_mean=np.zeros(K)

        lam_mean = a_update/b_update
        ln_lam_mean=sp.digamma(a_update)-np.log(b_update)
        ln_pi_mean=sp.digamma(alpha_update)-sp.digamma(np.sum(alpha_update))

        #####################################################
         # q(sn)
        s_mean = np.exp(x.reshape(len(x), 1)*ln_lam_mean-lam_mean + ln_pi_mean)
        s_mean /=np.sum(s_mean, axis=1).reshape(N, 1)

        ###########################################
        # update a, b
        a_update = np.sum(x.reshape(len(x),1) * s_mean, axis=0) + a
        b_update = np.sum(s_mean, axis=0) + b

        # update α
        alpha_update = np.sum(s_mean, axis=0) + alpha
        #####################################################


    # determine group number by order of λ
    s_order = st.gamma(a=a_update, scale=1/b_update).mean().argsort()
    s_mean_ordered = s_mean[:,s_order]

    return s_mean_ordered
