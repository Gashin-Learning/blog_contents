import numpy as np
from numba import jit

# sampling Numba
@jit('i8[:](i8[:], f8[:], f8[:,:], i8)', nopython=True)
def multinomial_particles_numba(z_new, random_sampling, sample_p_cumsum, particle_num):
    for i in range(particle_num):
        random_sampling_per = random_sampling[i]
        sample_p_cumsum_per = sample_p_cumsum[:,i]
        z_new[i] = np.min(np.where(sample_p_cumsum_per > random_sampling_per)[0])
    return z_new


def Partilcle_Filter(particle_num, topic_num, PF_len, docu_PF_index, word_PF_index, n_vk_beta_particles, alpha, beta):

    #list for evaluation
    n_dk_alpha_list = []
    w_list = []
    pre_docu_id=0

    # array-box for topic_particles
    topic_particles = np.zeros((PF_len, particle_num), dtype='int')
    topic_new = np.zeros(particle_num, dtype=int)

    # weight initialization
    w = np.ones(particle_num)

    # n_dk (document, topic) initialization
    n_dk_alpha_cond_d_particles = np.zeros((topic_num, particle_num))+ alpha

    for idx in range(PF_len):

        # select one word in a document sequentially
        docu_id = docu_PF_index[idx]
        word_id = word_PF_index[idx]
        v_num = n_vk_beta_particles.shape[0]

        # when document changes, reset N_dk
        if docu_id > pre_docu_id:

            #######################################################
            ## no need for algorithm, but store for evaluation. ##
            n_dk_alpha_list.append(n_dk_alpha_cond_d_particles)
            w_list.append(w)
            #######################################################

            pre_docu_id = docu_id
            n_dk_alpha_cond_d_particles = np.zeros((topic_num, particle_num))+ alpha

        #######################################################
        #### if new word appears, this process is needed ######
        #if word_id>=v_num:
        #    new_v_num = word_id +1 - v_num
        #    n_vk_particles = np.concatenate((n_vk_particles, np.zeros((new_v_num, topic_num,  particle_num))))
        #######################################################

        # Sequential Importance Sampling

        ## p(z_k| z_{1:k-1})
        n_d_alpha_cond_d_particles = np.sum(n_dk_alpha_cond_d_particles, axis=0)
        p_z = n_dk_alpha_cond_d_particles /n_d_alpha_cond_d_particles

        ## p(w_k| z_{1:k}, w_{1:k-1})
        n_vk_beta_cond_k_particles = n_vk_beta_particles[word_id, :, :]
        n_k_beta_cond_k_particles = np.sum(n_vk_beta_cond_k_particles, axis=0)
        p_w = n_vk_beta_cond_k_particles /n_k_beta_cond_k_particles

        ## p(z_k| z_{1:k-1})p(w_k| z_{1:k}, w_{1:k-1})
        sample_p = p_z * p_w
        sample_p_sum = np.sum(sample_p, axis=0)
        sample_p /=sample_p_sum

        # sampling from multinomial distribution
        random_sampling = np.random.uniform(0,1, size=particle_num)
        sample_p_cumsum = np.cumsum(sample_p, axis=0)
        topic_new = multinomial_particles_numba(topic_new, random_sampling, sample_p_cumsum, particle_num)
        topic_particles[idx] = topic_new

        # update parameters
        n_dk_alpha_cond_d_particles[topic_new, np.arange(100)] += 1.0
        n_vk_beta_particles[word_id, topic_new, np.arange(100)] += 1.0

        # update weight
        w*= sample_p_sum
        w /= np.sum(w)

        # ESS
        ESS = 1/np.sum(w**2)
        if ESS < 10:

            # Residual Resampling ##################################
            K = particle_num * w
            K_int = np.trunc(K)
            redisual_p = K - K_int
            redisual_p/=np.sum(redisual_p)

            K_int = K.astype('int') # main resample
            M = particle_num - np.sum(K_int)
            residual_resampling_array = np.random.multinomial(n=M, pvals=redisual_p) # residual resample

            resampling_idx = K_int + residual_resampling_array
            particle_idx = np.repeat(np.arange(particle_num), resampling_idx)
            topic_particles = topic_particles[:,particle_idx]
            n_vk_beta_particles = n_vk_beta_particles[:,:,particle_idx]
            n_dk_alpha_cond_d_particles = n_dk_alpha_cond_d_particles[:,particle_idx]

            w = np.ones(particle_num)/particle_num
            #########################################################

    n_dk_alpha_list.append(n_dk_alpha_cond_d_particles)
    w_list.append(w)

    return n_dk_alpha_list, n_vk_beta_particles, w_list, w
