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

  # first value Dirichlet distribution for \pi
  alpha = np.array([30,20,10])
  # first value Dirichlet distribution for \A
  beta = np.array([[20,10,10],[10,20,10],[10,10,20]])

  # gammma distribution
  a_update = np.array([200,200,200])
  b_update = np.array([5,5,5])

  # Dirichlet distribution
  alpha_update = np.array([30,20,10])
  beta_update = np.array([[20,10,10],[10,20,10],[10,10,20]])

  # set s first value
  s_mean=[]
  for n in range(N):
      s_mean.append([0.4,0.3,0.3])
  s_mean = np.array(s_mean)

  for i in range(iter_num):
      #print("\r Iteration：{}".format(i))

      #####################################################
      # expectation of λ、π、A

      lam_mean=np.zeros(K)
      ln_lam_mean=np.zeros(K)
      ln_pi_mean=np.zeros(K)
      ln_A_mean=(np.zeros((K,K))).tolist()

      lam_mean = a_update/b_update
      ln_lam_mean=sp.digamma(a_update)-np.log(b_update)
      ln_pi_mean=sp.digamma(alpha_update)-sp.digamma(np.sum(alpha_update))
      ln_A_mean = sp.digamma(beta_update) - sp.digamma(np.sum(beta_update, axis=0))

      #####################################################
      # q(Sn)、q(Sn, Sn-1)

      p_xn_sn_array =np.zeros((N, 3))

      # forward algorithm
      f_pass = np.zeros((N, 3))
      for n in range(N):

          ln_xn_sn = x[n]*ln_lam_mean-lam_mean
          p_xn_sn = np.exp(ln_xn_sn)
          p_xn_sn/=np.sum(p_xn_sn)
          p_xn_sn_array[n] = p_xn_sn

          p_s1 =  np.exp(ln_pi_mean)
          p_s1/=np.sum(p_s1)

          p_sn_sn_1 = np.exp(ln_A_mean)
          p_sn_sn_1 /=np.sum(p_sn_sn_1, axis=0)

          if n==0:
              f_pass[n] = p_xn_sn * p_s1
          else:
              f_pass[n] = p_xn_sn * np.sum(f_pass[n-1] * p_sn_sn_1, axis=1)

          f_pass[n] /=sum(f_pass[n])

      # backward algorithm
      b_pass = np.zeros((N, 3))
      for n in reversed(range(N)):

          if n==N-1:
              b_pass[n] = np.ones(3)

          else:
              ln_xn_sn = x[n+1]*ln_lam_mean-lam_mean
              p_xn_sn = np.exp(ln_xn_sn)
              p_xn_sn/=np.sum(p_xn_sn)

              p_sn_sn_1 = np.exp(ln_A_mean)
              p_sn_sn_1 /=np.sum(p_sn_sn_1, axis=0)

              b_pass[n] = np.sum(b_pass[n+1].reshape(3,1) * p_sn_sn_1 * p_xn_sn.reshape(3,1), axis=0)

          b_pass[n] /=sum(b_pass[n])


      s_mean = f_pass * b_pass
      s_mean /=np.sum(s_mean, axis=1).reshape(N, 1)

      f_sn_1_b_sn = f_pass[1:].reshape(len(f_pass[1:]), 1, 3) * b_pass[:-1].reshape(len(b_pass[:-1]), 3, 1)
      s_s_1_mean = f_sn_1_b_sn * p_sn_sn_1 * p_xn_sn_array[1:].reshape(len(p_xn_sn_array[1:]), 3,1)
      s_s_1_mean = s_s_1_mean/np.sum(s_s_1_mean, axis=1).reshape(N-1,3,1)

      ###########################################
      # update a, b
      a_update = np.sum(x.reshape(len(x),1) * s_mean, axis=0) + a
      b_update = np.sum(s_mean, axis=0) + b

      # update α
      alpha_update = s_mean[0] + alpha

      # update β
      beta_update = np.sum(s_s_1_mean + beta, axis=0)

      #####################################################


  # determine group number by order of λ
  s_order = st.gamma(a=a_update, scale=1/b_update).mean().argsort()
  s_mean_ordered = s_mean[:,s_order]

  return s_mean_ordered
