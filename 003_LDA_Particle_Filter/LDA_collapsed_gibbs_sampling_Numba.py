import numpy as np
from numba import jit

# calc n_dk :topic_num in the document
@jit('f8[:, :](i8[:], i8[:], i8, i8)', nopython=True)
def count_n_dk_numba(row_index, col_index, docu_num, topic_num):
    n_array = np.zeros((docu_num, topic_num))
    for row,col in zip(row_index, col_index):
        n_array[row,col] += 1.0
    return n_array

# calc n_kv :word_num assigned to the topic
@jit('f8[:, :](i8[:], i8[:], i8, i8)', nopython=True)
def count_n_kv_numba(row_index, col_index, word_num, topic_num):
    n_array = np.zeros((word_num, topic_num))
    for row,col in zip(row_index, col_index):
        n_array[row,col] += 1.0
    return n_array


@jit('Tuple((i8[:], f8[:,:], f8[:,:]))(i8[:], i8, i8, f8[:,:], f8[:,:], i8[:], i8[:], f8[:], f8[:], f8, f8[:])', nopython=True)
def collapsed_gibbs_sampling_numba(z, len_z, iter_num, n_dk, n_kv, docu_one_index, word_one_index, alpha, beta, sum_beta, n_kv_sum_axis0):

    for s in range(iter_num):
        for idx in range(len_z):
            docu_id = docu_one_index[idx]
            word_id = word_one_index[idx]
            z_id = z[idx]

            # exclude z_id for gibbs_sampling
            n_dk[docu_id, z_id] -=1.0
            n_kv[word_id, z_id] -=1.0
            n_kv_sum_axis0[z_id] -=1.0

            ## p(z_k| z_\k)
            n_dk_alpha = n_dk[docu_id, :] + alpha
            n_d_alpha = np.sum(n_dk_alpha)
            n_dk_alpha /=n_d_alpha

            ## p(w_k| z, w_\k)
            n_kv_beta = n_kv[word_id, :] + beta[word_id]
            n_kv_beta /=n_kv_sum_axis0

            # p(z|w)
            sample_p = n_dk_alpha * n_kv_beta
            sample_p /=np.sum(sample_p)

            # sampling from multinomial distribution
            z_new_array = np.random.multinomial(n=1, pvals=sample_p)
            z_new = np.nonzero(z_new_array)[0][0]
            z[idx]=z_new

            # update parameters
            n_dk[docu_id, z_new] +=1.0
            n_kv[word_id, z_new] +=1.0
            n_kv_sum_axis0[z_new]+=1.0

    return z, n_dk, n_kv
