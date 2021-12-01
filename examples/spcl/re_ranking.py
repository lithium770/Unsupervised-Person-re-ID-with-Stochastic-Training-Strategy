#!/usr/bin/env python2/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22.
- This version accepts distance matrix instead of raw features.
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

Modified by Zhedong Zheng, 2018-1-12.
- replace sort with topK, which save about 30s.
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""


import numpy as np
import torch
import torch.nn.functional as F
import time

def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(target_features, cam_labels, cams, k1=20, k2=6):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    end = time.time()

    N = target_features.size(0)
    #original_dist = target_features.mm(target_features.t()).cpu()
    cam_dist = torch.zeros([cams, cams])

    for cam1 in range(cams):
        for cam2 in range(cam1 + 1):
            cam1_mask = cam_labels.eq(cam1)
            cam2_mask = cam_labels.eq(cam2)
            #cam_mask = (cam1_mask.unsqueeze(0).expand(N, N)) * (cam2_mask.unsqueeze(1).expand(N, N))
            #cam_mask = cam_mask + cam_mask.t()
            #offset = original_dist[cam_mask].mean()
            offset = (target_features[cam1_mask].mm(target_features[cam2_mask].t())).mean()
            cam_dist[cam1][cam2] = offset
            cam_dist[cam2][cam1] = offset
            #original_dist[cam_mask] = original_dist[cam_mask] - offset / 2
    cam_offset = torch.zeros(cams, N).cuda()
    for i in range(cams):
        cam_offset[i] = cam_dist[i][cam_labels]
    del cam_dist

    #original_dist = 2. - 2 * original_dist  # change the cosine similarity metric to euclidean similarity metric
    #ori_dist = original_dist.numpy()

    #original_dist = np.power(original_dist, 2).astype(np.float32)
    #original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.zeros(shape=(N, N))
    for i in range(N):
        initial_rank[i] = np.argpartition(2-2*((target_features[i].unsqueeze(0).mm(target_features.t()))-cam_offset[cam_labels[i]].unsqueeze(0)).cpu(), range(1, k1 + 1))
    initial_rank = initial_rank.astype(np.int32)
    #initial_rank = np.argpartition( ori_dist, range(1,k1+1) )
    #del ori_dist

    #N = target_features.size(0)
    mat_type = np.float32
    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        #dist = original_dist[i][k_reciprocal_expansion_index].unsqueeze(0)
        dist = 2 - 2 * (torch.mm(target_features[i].unsqueeze(0).contiguous(),
                                target_features[k_reciprocal_expansion_index].t())-cam_offset[cam_labels[i]][k_reciprocal_expansion_index].unsqueeze(0))

        V[i, k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    print("Jaccard distance computing time cost: {}".format(time.time() - end))

    return jaccard_dist
