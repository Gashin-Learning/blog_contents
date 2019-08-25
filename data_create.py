import numpy as np
import random
from scipy import stats as st


def make_toy_data(expriment_mode=False, lambdas_input=[1,1,1], print_option=False, my_seed=11,N=200,\
                    alpha=[50,30,10], beta1=[50,5,5], beta2=[5,50,5], beta3=[5,5,50],\
                    a=[50, 450, 1000], b=[4, 15, 20]):

    # sampling from Dirichlet distribution for \pi
    pi = np.random.dirichlet(alpha, size=1)[0]

    # sampling from Dirichlet distributions for A[:,i]
    A = np.hstack([np.random.dirichlet(beta1, size=1).transpose()
               ,np.random.dirichlet(beta2, size=1).transpose(),
               np.random.dirichlet(beta3, size=1).transpose()])

    # sampling from Categorical distribution for state_series
    s = []
    s.append(np.argmax(np.random.multinomial(n=1, pvals=pi, size=1)))

    for i in range(1,N):
        s.append(np.argmax(np.random.multinomial(n=1, pvals=A[:,s[i-1]], size=1)))

    # sampling from gamma distributions for \lambda
    lamdas=[]
    for i in range(len(a)):
        lamda = st.gamma(a=a[i], scale=1/b[i]).rvs()
        lamdas.append(lamda)
    # for experiment not sampling lambdas but set
    if expriment_mode:
        lamdas = lambdas_input


    # generation x by sampling from Poisson distributions switching by state categories
    x = []
    for i in range(len(s)):
        x_per = np.random.poisson(lam=lamdas[s[i]])
        x.append(x_per)

    if print_option:
        print('sampling π')
        print(pi,'\n')

        print('sampling A[:,i]')
        print(A,'\n')

        print('sampling λ')
        print(lamdas, '\n')


    return np.array(x), np.array(s)
