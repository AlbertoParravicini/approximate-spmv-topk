# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 19:22:56 2020

@author: albyr
"""

import numpy as np
import csv 
import pandas as pd
import time
import math
from decimal import *
from fractions import Fraction

# K = [8, 16, 32, 50, 75, 100]
# PARTITIONS = [16]
# NUM_TESTS = 100
# MATRIX_SIZE = 1000000
# PARTITION_K = 8

K = [8, 16, 32, 50, 75, 100, 300]
PARTITIONS = [8, 16, 32]
NUM_TESTS = 10 # 100
MATRIX_SIZE = 100000 # 1000000
PARTITION_K = 8


def closed_form_approx(n, b, k, partition_k):
    if k <= partition_k:
        return 1
    if partition_k * b < k:
        return 0
    denom = math.comb(n, k)
    delta = 0
    for i in range(partition_k + 1, min(n // b, k)):
        delta += math.comb(n // b, i)
    return 1 - Fraction(b * delta, denom)


def closed_form_precision_estimation(n, b, k, partition_k):
    return np.mean([closed_form_approx(n, b, k_i, partition_k) for k_i in range(1, k + 1)])

getcontext().prec = 200
res = []
start = time.time()
for n in range(NUM_TESTS):
    for p in PARTITIONS: 
        start_indices=[i * (MATRIX_SIZE // p) + min(i, MATRIX_SIZE % p) for i in range(p)]
        res_vec = np.random.uniform(low=0.0, high=1.0, size=(MATRIX_SIZE,))
        sorted_res = np.argsort(res_vec)
        splitted_res = np.array_split(res_vec, p)
        sorted_splits = [np.argsort(i) for i in splitted_res]
        for k in K:
            print(n, p, k, f"elapsed={time.time() - start:.2f} sec")
        
            # CALCOLA VERA TOP-K
            top_k_res_indexes = sorted_res[-k:][::-1]
            # print(res_vec)
            # print("top-k=", top_k_res_indexes)
            
            # CALCOLA TOP-K DI OGNI PARTIZIONE
            top_k_splitted_res_indexes = [split[-(PARTITION_K):][::-1] for split in sorted_splits]

            # AGGIUSTA INDICI IN BASE ALLA PARTIZIONE
            top_k_splitted_res_initial_indexes = np.concatenate([i + start_indices[index] for index, i in enumerate(top_k_splitted_res_indexes)])
            # print("top_k_splitted_res_initial_indexes :")
            # print(top_k_splitted_res_initial_indexes)
            
            # CONSIDERA SOLO LA TOP-K E SEGA IL RESTO
            sw = np.argsort(res_vec[top_k_splitted_res_initial_indexes])[-k:]
            sw_splitted_res = top_k_splitted_res_initial_indexes[sw]
            
            # CHECK FOR ERRORS
            true_set_res_indexes = set(top_k_res_indexes)
            partitioned_set_res_indexes = set(sw_splitted_res)
            diff = true_set_res_indexes.difference(partitioned_set_res_indexes)
            errors = len(true_set_res_indexes.difference(partitioned_set_res_indexes))
            
            # CLOSED FORM RESULT
            closed_form_res = closed_form_precision_estimation(MATRIX_SIZE, p, k, PARTITION_K)
            
            res += [[MATRIX_SIZE, p, k, errors, 1 - errors / len(true_set_res_indexes), float(closed_form_res)]]
            
data = pd.DataFrame(res, columns=["N", "partitions", "K", "errors", "precision", "closed_form_precision"])
data_agg = data.groupby(["N", "partitions", "K"]).mean()
groups = data.groupby(["N", "partitions", "K"]).mean()
print(groups)
