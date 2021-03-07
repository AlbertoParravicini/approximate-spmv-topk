# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:45:26 2020

@author: albyr
"""

import numpy as np
import os
import argparse
import time
from datetime import datetime
from numba import jit
import ray
import psutil
import itertools


DEFAULT_DEBUG = True
DEFAULT_L2_NORM = True
DEFAULT_STORE_RESULTS = True
DEFAULT_PARALLEL = False
NUM_CORES = 40  
NUM_ROWS = [10000]
MAX_COLS = [512, 1024]
AVERAGE_DEGREE = [20, 40]
DISTRIBUTION = ["uniform", "gamma"]
DEFAULT_OUTPUT = "../../../data/matrices_for_testing"
DEFAULT_PRECISION = 10

GAMMA_K = 3  # The lower (it must be > 0), the higher the left skew; Gamma "scale" is given by mean / K;  

MTX_HEADER = "%%MatrixMarket matrix coordinate real general\n%\n{} {} {}\n"

#############################
#############################

def debug_print(message: str) -> None:
    currtime = datetime.now().strftime("[%Y/%m/%d - %H:%M:%S]")
    print(currtime, message)
    
    
@jit(nopython=True)
def create_row(x: np.array, y: np.array, val: np.array, curr_nnz: int, curr_start: int, max_cols: int, x_i: int, l2_norm: bool) -> None:
    curr_y = sorted(np.random.randint(0, max_cols, curr_nnz))
    x[curr_start:(curr_start + curr_nnz)] = x_i  
    y[curr_start:(curr_start + curr_nnz)] = curr_y
    if l2_norm:
        val[curr_start:(curr_start + curr_nnz)] /= np.linalg.norm(val[curr_start:(curr_start + curr_nnz)])


@jit(nopython=True)    
def create_row_fast(max_cols: int, curr_nnz: int, val: np.array) -> tuple:
    return sorted(np.random.randint(0, max_cols, curr_nnz)), val / np.linalg.norm(val)


def create_sparse_matrix(
        num_rows: int,
        max_cols: int,
        average_degree: int,
        distribution: str,
        output_file: str,
        l2_norm: bool = True,
        debug: bool = False,
        store_results: bool = False,
        precision: int = DEFAULT_PRECISION
        ) -> tuple:
    """
    :param num_rows: number of rows in the matrix
    :param max_cols: number of columns in the matrix
    :param average_degree: average number of non-zero entries in each row
    :param distribution: what distribution should be used to generate the number of non-zero entries per row
    :param output_file: full path to where the matrix file is stored
    :param l2_norm: if True, normalize in L2 norm the values in each row
    :param debug: if True, print debug information
    :param store_results: if True, return the arrays that represent the generated matrix, instead of just writing them to file
    :param precision: number of decimal digits stored in floating point numbers
    """
    
    start = time.time()
    
    # Create in advance the degree of each row, as it's faster;
    if distribution == "uniform":
        min_degree = average_degree // 2
        max_degree = int(average_degree * 1.5)
        nnz = np.random.randint(min_degree, max_degree + 1, num_rows)
        total_nnz = np.sum(nnz)
        if debug:
            debug_print(f"  uniform distribution, min_degree={min_degree}, max_degree={max_degree}, average_degree={np.mean(nnz)}, nnz={total_nnz}")
    elif distribution == "gamma":
        nnz = np.maximum(np.random.gamma(GAMMA_K, average_degree / GAMMA_K, num_rows).astype(int), 1)
        total_nnz = np.sum(nnz)
    else:
        raise ValueError(f"unknown distribution {distribution}")
        
    # First, create the file and put the MTX header in it;
    with open(output_file, "w") as f:
        
        f.write(MTX_HEADER.format(num_rows, max_cols, total_nnz))
        
        if store_results:
            x = np.zeros(total_nnz, dtype=int)
            y = np.zeros(total_nnz, dtype=int)
        val = np.random.rand(total_nnz)  # Create all random values at the same time, it's faster;
                 
        # Create one row at a time;
        curr_start = 0
        for x_i in range(num_rows):
            
            if debug and x_i % 100000 == 0 and x_i > 0:
                debug_print(f"    current row={x_i}/{num_rows}, current time={time.time() - start:.2f} sec")
            
            curr_nnz = nnz[x_i]
            
            # If storing results in Python variables, take a slower path;
            if store_results:
                create_row(x, y, val, curr_nnz, curr_start, max_cols, x_i, l2_norm)
                # Write current values;
                for t in zip(x[curr_start:(curr_start + curr_nnz)], y[curr_start:(curr_start + curr_nnz)], val[curr_start:(curr_start + curr_nnz)]):
                    f.write(f"{t[0] + 1} {t[1] + 1} {t[2]:.{precision}}\n")
            else:
                # Else, write results directly to file;
                curr_y, curr_val = create_row_fast(max_cols, curr_nnz, val[curr_start:(curr_start + curr_nnz)])
                f.writelines([f"{x_i + 1} {curr_y[i] + 1} {curr_val[i]:.{precision}}\n" for i in range(curr_nnz)])
            curr_start += curr_nnz
        
        if store_results:
            return x, y, val
        else:
            return ([], [], [])
  
@ray.remote
def create_sparse_matrix_parallel_inner(thread_id, val, nnz, s, e, nnz_s, nnz_e, max_cols) -> dict:
    
    x_out = np.zeros(nnz_e - nnz_s, dtype=int)
    y_out = np.zeros(nnz_e - nnz_s, dtype=int)
    val_out = val[nnz_s:nnz_e]
    val_out_out = []
    
    curr_start = 0
    for x_i in range(s, e):
        curr_nnz = nnz[x_i] 
        x_out[curr_start:(curr_start + curr_nnz)] = x_i  
        y_out[curr_start:(curr_start + curr_nnz)] = sorted(np.random.randint(0, max_cols, curr_nnz))
        if l2_norm:
            val_out_out += (val_out[curr_start:(curr_start + curr_nnz)] / np.linalg.norm(val_out[curr_start:(curr_start + curr_nnz)])).tolist()
        curr_start += curr_nnz
    return {"id": thread_id, "x": x_out, "y": y_out, "val": val_out_out}
    

def create_sparse_matrix_parallel(
        num_rows: int,
        max_cols: int,
        average_degree: int,
        distribution: str,
        output_file: str,
        l2_norm: bool = True,
        debug: bool = False,
        store_results: bool = False,
        precision: int = DEFAULT_PRECISION
        ) -> tuple:
    
    starts = [p[0] for p in np.array_split(np.arange(num_rows), NUM_CORES)] + [num_rows]
    
    # Create in advance the degree of each row, as it's faster;
    if distribution == "uniform":
        min_degree = average_degree // 2
        max_degree = int(average_degree * 1.5)
        nnz = np.random.randint(min_degree, max_degree + 1, num_rows)
        total_nnz = np.sum(nnz)
        if debug:
            debug_print(f"  uniform distribution, min_degree={min_degree}, max_degree={max_degree}, average_degree={np.mean(nnz)}, nnz={total_nnz}")
    elif distribution == "gamma":
        nnz = np.maximum(np.random.gamma(GAMMA_K, average_degree / GAMMA_K, num_rows).astype(int), 1)
        total_nnz = np.sum(nnz)
    else:
        raise ValueError(f"unknown distribution {distribution}")
        
    nnz_starts = [np.sum(nnz[:p]) for p in starts]
                
    val = np.random.rand(total_nnz)
    
    val_r = ray.put(val)
    nnz_r = ray.put(nnz)
    
    threads = []
    for i in range(NUM_CORES):
        threads += [create_sparse_matrix_parallel_inner.remote(i, val_r, nnz_r, starts[i], starts[i + 1], nnz_starts[i], nnz_starts[i + 1], max_cols)]
    results = ray.get(threads) 
    
    results = sorted(results, key=lambda r: r["id"])
    
    x = list(itertools.chain(*[r["x"] for r in results]))
    y = list(itertools.chain(*[r["y"] for r in results]))
    val_final = list(itertools.chain(*[r["val"] for r in results]))
    
    with open(output_file, "w") as f: 
        f.write(MTX_HEADER.format(num_rows, max_cols, total_nnz))
        for t in zip(x, y, val_final):
            f.write(f"{t[0] + 1} {t[1] + 1} {t[2]:.{precision}}\n")
                    
    return x, y, val_final
        

#############################
#############################

if __name__ == "__main__":
    
    num_cpus = psutil.cpu_count(logical=False)
    ray.shutdown()
    ray.init(num_cpus=num_cpus)
    
    folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    parser = argparse.ArgumentParser(description="Generate a random sparse matrix in MTX format")
    
    parser.add_argument("-d", "--debug", action="store_true", default=DEFAULT_DEBUG, help="If present, print additional debug information")
    parser.add_argument("-r", "--rows", type=int, nargs="+", help="Number of rows in the matrix", default=NUM_ROWS)   
    parser.add_argument("-c", "--max_cols", type=int, nargs="+", help="Maximum number of columns in the matrix", default=MAX_COLS)   
    parser.add_argument("--degree", type=int, nargs="+", help="Average number of non-zero entries per row", default=AVERAGE_DEGREE)   
    parser.add_argument("--distribution", type=str, nargs="+", help="Distribution that determines the number of non-zero entries per row", choices=DISTRIBUTION, default=DISTRIBUTION)  
    parser.add_argument("--l2_norm", type=bool, help="If true, perform L2 normalization on the values of each row", default=DEFAULT_L2_NORM)  
    parser.add_argument("-o", "--output_path", type=str, help="Folder where matrices are stored", default=os.path.join(DEFAULT_OUTPUT, folder_name))  
    parser.add_argument("--parallel", action="store_true", default=DEFAULT_PARALLEL, help="If present, use a multithreaded implementation")
    parser.add_argument("--precision", type=int, default=DEFAULT_PRECISION, help="Number of decimal digits to add in floating point numbers")
    args = parser.parse_args()
    
    debug = args.debug
    parallel = args.parallel
    rows = [r for r in args.rows if r > 0]
    cols = [c for c in args.max_cols if c > 0]
    degrees = [d for d in args.degree if d > 0]
    distributions = [d for d in args.distribution]
    l2_norm = args.l2_norm
    precision = args.precision
    output_folder = args.output_path
    
    total_matrices = len(rows) * len(cols) * len(degrees) * len(distributions)
    
    if debug:
        debug_print("Starting matrices generation...")
        debug_print(f"  rows={rows}")    
        debug_print(f"  max_cols={cols}")    
        debug_print(f"  degrees={degrees}")    
        debug_print(f"  distributions={distributions}") 
        debug_print(f"  l2 normalization={l2_norm}") 
        debug_print(f"  output folder={output_folder}") 
        debug_print(f"  use parallel implementation={parallel}") 
        debug_print(f"  decimal precision digits={precision}") 
        debug_print(f"creating a total of {total_matrices} matrices") 
        
    if not os.path.exists(output_folder):
        if debug:
            debug_print(f"creating output folder={output_folder}") 
        os.makedirs(output_folder)
    
    if debug:
        debug_print("-" * 30)
        
    start_time_tot = time.time()
    curr_matrix_index = 1
    for r in rows:
        for c in cols:
            for degree in degrees:
                for dist in distributions:
                    start_time = time.time()
                    filename = f"matrix_{r}_{c}_{degree}_{dist}.mtx"
                    if debug:
                        debug_print(f"creating matrix {curr_matrix_index}/{total_matrices}, start_time={start_time - start_time_tot:.2f} sec")
                        debug_print(f"  rows={r}, max_cols={c}, average_degree={degree}, distribution={dist}, filename={filename}")
                    # Create new matrix;
                    if parallel:
                        x, y, val = create_sparse_matrix_parallel(r, c, degree, dist, os.path.join(output_folder, filename), l2_norm, debug, DEFAULT_STORE_RESULTS, precision)
                    else:
                        x, y, val = create_sparse_matrix(r, c, degree, dist, os.path.join(output_folder, filename), l2_norm, debug, DEFAULT_STORE_RESULTS, precision)
                    if debug:
                        debug_print(f"finished matrix {curr_matrix_index}/{total_matrices}, creation_time={time.time() - start_time:.2f} sec, total_time={time.time() - start_time_tot:.2f}")
                        debug_print("-" * 30)
                    curr_matrix_index += 1   

