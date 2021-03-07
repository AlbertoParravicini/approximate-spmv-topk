from scipy.sparse import rand
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import scipy.io
from random import sample
import random
import gc
import sys
import time
import os
from datetime import datetime
import argparse
if sys.version_info[0] >= 3:
    from sparse_dot_topn import sparse_dot_topn as ct
    from sparse_dot_topn import sparse_dot_topn_threaded as ct_thread
else:
    import sparse_dot_topn as ct
    import sparse_dot_topn_threaded as ct_thread

from sparse_dot_topn import awesome_cossim_topn
import pandas as pd

################################################################
# Script used for CPU testing, no need to change anything here #
################################################################

INPUT_PATH = "/path/to/default/matrix/matrix_100000_512_20_uniform.mtx"
NUM_TESTS = 30
OUTPUT_PATH = "data/results/cpu"

K = 100
THRESHOLD = 0.0

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="run top-k spmv on cpu")
    
    parser.add_argument("-d", "--debug", action='store_true',
                        help="If present, print debug messages")
    parser.add_argument("-t", "--num_tests", metavar="N", type=int, default=NUM_TESTS,
                        help="Number of times each test is executed")
    parser.add_argument("-k", "--K", metavar="N", type=int, default=K,
                        help="Number of top-k to retrieve")
    parser.add_argument("-z", "--zero_index", action='store_true',
                        help="If true, the MTX matrix is zero-indexed")
    parser.add_argument("-i", "--input", type=str, default=INPUT_PATH,
                        help="Relative path to the input MTX matrix")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_PATH,
                        help="Relative path to the output CSV")
    
    args = parser.parse_args()
    
    DEBUG = args.debug
    NUM_TESTS = args.num_tests
    K = args.K
    ZERO_INDEX = args.zero_index
    INPUT_PATH = args.input
    output_path = args.output
    
    x = None
    y = None
    val = None
    with open(INPUT_PATH) as f:
        lines = f.readlines()
        size = int(lines[2].split(" ")[2])
        rows = int(lines[2].split(" ")[0])
        cols = int(lines[2].split(" ")[1])
        x = np.zeros(size, dtype=int)
        y = np.zeros(size, dtype=int)
        val = np.zeros(size)
        if ZERO_INDEX:
            for i, l in enumerate(lines[3:]):
                x_c, y_c, v_c = l.split(" ")
                x_c = int(x_c) 
                y_c = int(y_c)
                v_c = float(v_c)
                x[i] = x_c
                y[i] = y_c
                val[i] = v_c
        else:
            for i, l in enumerate(lines[3:]):
                x_c, y_c, v_c = l.split(" ")
                x_c = int(x_c) 
                y_c = int(y_c)
                v_c = float(v_c)
                x[i] = x_c - 1
                y[i] = y_c - 1
                val[i] = v_c
                
    # Turn the arrays into a CSR;
    csr = csr_matrix((val, (x, y)))
    
    print(f"loaded matrix of size {rows}x{cols}, {size} nnz")
    
    results = []
    
    for t in range(NUM_TESTS):
        # Create a random vector;
        vec_np = np.random.uniform(low=0.0, high=1.0, size=(cols,))
        vec_np /= np.linalg.norm(vec_np) 
        vec = csr_matrix(vec_np)
        
        start = time.time()
        res = awesome_cossim_topn(csr, vec.transpose(), K, THRESHOLD, use_threads=True, n_jobs=40)
        end = (time.time() - start) * 1000
        results +=  [[t, rows, cols, size, K, end]]
        if DEBUG:
            print(f"finished iteration {t + 1}/{NUM_TESTS}, time={end} ms, res=\n{res[:8]}")
        else:
            print(results[-1])
    
    results_df = pd.DataFrame(results, columns = ["iter", "rows", "cols", "nnz", "K", "exec_time_ms"])
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    input_matrix = os.path.splitext(os.path.basename(INPUT_PATH))[0]
    
    if output_path:
        out_path = output_path
    else:
        out_dir = os.path.join(OUTPUT_PATH, now)
        out_path = os.path.join(out_dir, input_matrix + ".csv")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    results_df.to_csv(out_path, index=False)