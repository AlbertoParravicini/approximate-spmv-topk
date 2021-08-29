#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:52:11 2020

@author: aparravi
"""

import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import os
import matplotlib.lines as lines
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.ticker as ticker
from matplotlib import transforms
from plot_utils import *
from matplotlib.patches import Patch

# Old results (DATE)
# MAIN_RESULT_FOLDER = "../../../../data/results/2020_09_21_2"
# CPU_RESULT_FOLDER = "../../../../data/results/2020_09_11_cpu"
# DATE = "2020_09_21_2"

# New results (DAC)
CPU_RESULT_FOLDER = "../../../../data/results/cpu/2020_11_22_xeon_6248"
GPU_RESULT_FOLDER = "../../../../data/results/gpu/2020_11_22"
FPGA_RESULT_FOLDER = "../../../../data/results/fpga/2020_11_23"

# Newest results (Thesis)
FPGA_RESULT_FOLDER = "../../../../data/results/fpga/2021_08_14_19_20_18"

DATE = "2021_08_16"

CPU_CORES = 80  # Double Xeon 6248
GPU_CORES = 56  # P100 SM cores;
GPU_IMPL_DICT = {0: "CSR", 1: "CSR LightSpMV", 2: "COO"}
GPU_LABELS = {"gpu": "GPU\nF32", "gpu_half": "GPU\nF16", "gpu_CSR": "GPU", "gpu_CSR_half": "GPU\nF16", "gpu_CSR LightSpMV": "GPU\nLS", "gpu_CSR LightSpMV_half": "GPU\nLS F16"}

def read_datasets():
    result_list = []
    
    for matrix_name in os.listdir(MAIN_RESULT_FOLDER):
        res_file = os.path.join(MAIN_RESULT_FOLDER, matrix_name)
        if res_file.endswith(".csv"):
            with open(res_file, "r") as f:
                # Read results, but skip the header;
                result = f.readlines()[1:]
                
                # Parse the file name;
                rows, n_bit, n_cores, k, n_iter = os.path.splitext(matrix_name)[0].split("_")
                
                try:
                    n_bit = int(n_bit[:-3]) if n_bit != "float" else "F32"
                except ValueError:
                    pass
                
                for r in result:
                    iteration, error_idx, error_val, sw_full_time_ms, sw_topk_time_ms, \
                    hw_setup_time_ms, hw_exec_time_ms, readback_time_ms, k, sw_res_idx, \
                    sw_res_val, hw_res_idx, hw_res_val = r.split(",")

                    # Add the result line to the list;
                    new_res_line = [int(rows), str(n_bit), int(n_cores), int(iteration), int(n_iter), int(error_idx), int(error_val), \
                                    float(sw_full_time_ms), float(sw_topk_time_ms), float(hw_setup_time_ms),
                                    float(hw_exec_time_ms), float(readback_time_ms), int(k),
                                    sw_res_idx, sw_res_val, hw_res_idx, hw_res_val]
                    if float(hw_exec_time_ms) <= (100 if n_bit != "F32" else 300):
                        result_list += [new_res_line]
                        
    # Create a dataframe;
    result_df = pd.DataFrame(result_list,
                             columns=["rows", "n_bit", "n_cores", "n_iter",  "max_iter", "error_idx", "error_val",
                                      "sw_full_time_ms", "sw_topk_time_ms", "hw_setup_time_ms",
                                      "hw_exec_time_ms", "readback_time_ms", "k",
                                      "sw_res_idx", "sw_res_val", "hw_res_idx", "hw_res_val"])
    
    # Remove outliers;
    # result_df = result_df[result_df["hw_exec_time_ms"] <= 100].reset_index(drop=True)
    result_df = remove_outliers_df_grouped(result_df, "hw_exec_time_ms", ["rows", "n_bit", "n_cores"], reset_index=True, drop_index=True, sigmas=2)
            
    # Same stuff, for CPU results;
    cpu_result_list = []
    
    for res_name in os.listdir(CPU_RESULT_FOLDER):
        if (res_name.endswith(".txt")):
            res_path = os.path.join(CPU_RESULT_FOLDER, res_name)
            with open(res_path, "r") as f:
                num_rows = res_name.split("_")[-2]
                # Skip header;
                cpu_result_list += [[int(num_rows), "cpu", 40, i, 30, 0, 0, float(l) * 1000] for i, l in enumerate(f.readlines())]                    
                    
    # Create a dataframe;
    cpu_result_df = pd.DataFrame(cpu_result_list, 
                                  columns=["rows", "n_bit", "n_cores", "n_iter", "max_iter",
                                           "error_idx", "error_val", "exec_time_ms"])    
    
    # Create a dataframe for execution times;
    result_df_small = result_df[["rows", "n_bit", "n_cores", "n_iter",  "max_iter", "error_idx", "error_val", "hw_exec_time_ms"]].rename(columns={"hw_exec_time_ms": "exec_time_ms"})

    # Skip small matrix;
    result_df_small = result_df_small[result_df_small["rows"] > 10**4]
    cpu_result_df = cpu_result_df[cpu_result_df["rows"] > 10**4]
    
    # Join the datasets;
    res = pd.concat([result_df_small, cpu_result_df]).reset_index(drop=True)
    
    # Compute speedups;
    
    # Initialize speedup values;
    res["speedup"] = 1
    res["baseline_time_ms"] = 0
    
    grouped_data = res.groupby(["rows", "n_bit", "n_cores"], as_index=False)
    for group_key, group in grouped_data:
        # Compute the median baseline computation time;
        median_baseline = np.median(res.loc[(res["n_bit"] == "cpu") & (res["rows"] == group_key[0]), "exec_time_ms"])
        # Compute the speedup for this group;
        group.loc[:, "speedup"] = median_baseline / group["exec_time_ms"]
        group.loc[:, "baseline_time_ms"] = median_baseline
        res.loc[group.index, :] = group
        # Guarantee that the geometric mean of speedup referred to the baseline is 1, and adjust speedups accordingly;
        gmean_speedup = gmean(res.loc[(res["n_bit"] == "cpu") & (res["rows"] == group_key[0]), "speedup"])
        group.loc[:, "speedup"] /= gmean_speedup
        res.loc[group.index, :] = group
            
    return result_df, cpu_result_df, res, res.groupby(["rows", "n_bit", "n_cores"]).aggregate(gmean)


def read_results_cpu(folder):
    
    result_list = []
    
    for f in os.listdir(folder):
        res_file = os.path.join(folder, f)
        if res_file.endswith(".csv"):
            result = pd.read_csv(res_file)

            # Parse the file name;
            hardware, rows, max_cols, distribution, nnz_per_row, k, max_iter = os.path.splitext(f)[0].split("_")
            result["hardware"] = hardware
            result["n_cores"] = CPU_CORES
            result["distribution"] = distribution
            result["nnz_per_row"] = int(nnz_per_row)
            result["max_iter"] = int(max_iter)
            result["error_idx"] = 0
            result["error_val"] = 0
            result_list += [result]
    results = pd.concat(result_list).reset_index(drop=True)
    results = results[["rows", "cols", "nnz_per_row", "distribution", "hardware",
                       "iter", "nnz", "n_cores", "max_iter", "error_idx", "error_val",
                       "exec_time_ms"]].rename(columns={"iter": "iteration", "cols": "max_cols"})
   
    # Skip first iteration
    results = results[results["iteration"] > 0]
    
    results_grouped = results.groupby(["rows", "max_cols", "nnz_per_row", "distribution", "hardware", "nnz"])[["error_idx", "error_val", "exec_time_ms"]].agg(np.mean)

    return results, results_grouped


def read_results_gpu(folder):
    
    result_list = []
    
    for f in os.listdir(folder):
        res_file = os.path.join(folder, f)
        if res_file.endswith(".csv"):
            result = pd.read_csv(res_file)

            # Parse the file name;
            hardware, rows, max_cols, distribution, nnz_per_row, impl, half_precision, k, max_iter = os.path.splitext(f)[0].split("_")
            result["hardware"] = hardware
            result["impl"] = GPU_IMPL_DICT[int(impl)]
            result["half_precision"] = half_precision == "True"
            result["n_cores"] = GPU_CORES
            result["distribution"] = distribution
            result["nnz_per_row"] = int(nnz_per_row)
            result["max_iter"] = int(max_iter)
            result["rows"] = int(rows)
            result["max_cols"] = int(max_cols)
            result["nnz_per_row"] = int(nnz_per_row)
            result_list += [result]
    results = pd.concat(result_list).reset_index(drop=True)
    results = results[["rows", "max_cols", "nnz_per_row", "distribution", 
                               "hardware", "impl", "half_precision", "iteration", "n_cores", "max_iter",
                               "error_idx", "error_val", "hw_spmv_only_time_ms", 
                               "hw_exec_time_ms"]].rename(columns={"hw_spmv_only_time_ms": "spmv_exec_time_ms", "hw_exec_time_ms": "exec_time_ms"})
   
    # Skip first iteration
    results = results[results["iteration"] > 0]
    # results = remove_outliers_df_grouped(results, "exec_time_ms", ["rows", "max_cols", "nnz_per_row", "distribution", "hardware", "impl", "half_precision"], reset_index=True, drop_index=True, sigmas=1)

    results_grouped = results.groupby(["rows", "max_cols", "nnz_per_row", "distribution", "hardware", "impl", "half_precision"])[["error_idx", "error_val", "spmv_exec_time_ms", "exec_time_ms"]].agg(np.mean)

    return results, results_grouped


def read_results_fpga(folder):
    
    result_list = []
    
    for f in os.listdir(folder):
        res_file = os.path.join(folder, f)
        if res_file.endswith(".csv"):
            result = pd.read_csv(res_file)

            # Parse the file name;
            hardware, rows, max_cols, distribution, nnz_per_row, bits, cores, mhz, k, max_iter = os.path.splitext(f)[0].split("_")
            result["hardware"] = hardware
            result["bits"] = bits.replace("bit", "")
            result["cores"] = int(cores.replace("core", ""))
            result["mhz"] = int(mhz.replace("mhz", ""))
            result["n_cores"] = GPU_CORES
            result["distribution"] = distribution
            result["nnz_per_row"] = int(nnz_per_row)
            result["max_iter"] = int(max_iter)
            result["rows"] = int(rows)
            result["max_cols"] = int(max_cols)
            result["nnz_per_row"] = int(nnz_per_row)
            result_list += [result]
    results = pd.concat(result_list).reset_index(drop=True)
    results = results[["rows", "max_cols", "nnz_per_row", "distribution", 
                               "hardware", "bits", "cores", "mhz", "iteration", "n_cores", "max_iter",
                               "error_idx", "error_val", 
                               "hw_exec_time_ms", "hw_full_exec_time_ms"]].rename(columns={"hw_exec_time_ms": "exec_time_ms", "hw_full_exec_time_ms": "full_exec_time_ms"})
    
    results = results[results["bits"] != "21"]
    results = results[~((results["bits"] == "26") & (results["cores"] == 24))]
    
    results.loc[results["bits"] == "20", "exec_time_ms"] *= 10
    # results.loc[results["bits"] == "float", "exec_time_ms"] *= 10
   
    # Skip first iteration
    results = results[results["iteration"] > 0]
    # results = remove_outliers_df_grouped(results, "exec_time_ms", ["rows", "max_cols", "nnz_per_row", "distribution", "hardware", "bits", "cores", "mhz"], reset_index=True, drop_index=True, sigmas=1)
    
    results_grouped = results.groupby(["rows", "max_cols", "nnz_per_row", "distribution", "hardware", "bits", "cores", "mhz"])[["error_idx", "error_val", "exec_time_ms", "full_exec_time_ms"]].agg(np.mean)

    return results, results_grouped


def join_datasets(res_cpu, res_gpu, res_fpga, add_spmv_time_to_cpu_and_fpga=True):
    
    # Fix nnz-per-row in GloVe;
    # glove_nnz = res_cpu[res_cpu["distribution"] == "glove"]["nnz_per_row"].unique()[0]
        
    # res_gpu.loc[res_gpu["distribution"] == "glove", "nnz_per_row"] = glove_nnz
    # res_fpga.loc[res_fpga["distribution"] == "glove", "nnz_per_row"] = glove_nnz
    
    # Add SpMV to CPU;
    if add_spmv_time_to_cpu_and_fpga:
        res_cpu["spmv_exec_time_ms"] = res_cpu["exec_time_ms"]
        res_fpga["spmv_exec_time_ms"] = res_fpga["exec_time_ms"]
    
    # Keep only CSR CPU;
    res_gpu = res_gpu[res_gpu["impl"] == "CSR"].copy()
    # res_gpu = res_gpu[res_gpu["impl"] != "COO"].copy()
    
    if len(set(res_gpu["impl"])) > 1:
        res_gpu["hardware"] += "_" + res_gpu["impl"]
    
    res_gpu["hardware"] += np.where(res_gpu["half_precision"], "_half", "")
    res_fpga["hardware"] += "_" + res_fpga["bits"] + "_" + res_fpga["cores"].astype(str) + "_" + res_fpga["mhz"].astype(str)
    
    # Add missing nnz information;
    res_gpu["nnz"] = 0
    res_fpga["nnz"] = 0
    
    # Filter rows;
    res_gpu = res_gpu[res_cpu.columns]
    res_fpga = res_fpga[res_cpu.columns]
    res = pd.concat([res_cpu, res_gpu, res_fpga]).reset_index(drop=True)
    
    # Fill missing nnz information;
    for i, g in res.groupby(["rows", "max_cols", "nnz_per_row", "distribution"]):
        nnz = g[g["hardware"] == "cpu"]["nnz"]
        if len(set(nnz)) != 1:
            print("error, multiple nnz=", set(nnz), "for", g["rows"].unique())
        assert(len(set(nnz)) == 1 or set(g["distribution"]) == set("glove"))
        nnz = nnz.iloc[0]
        res.loc[g.index, "nnz"] = nnz
        
    # Add speedup metric;
    
    # Initialize speedup values;
    res["speedup"] = 1
    res["spmv_speedup"] = 1
    res["baseline_time_ms"] = 0
    res["spmv_baseline_time_ms"] = 0
        
    grouped_data = res.groupby(["rows", "max_cols", "nnz_per_row", "distribution", "hardware"], as_index=False)
    for group_key, group in grouped_data:
        # Compute the median baseline computation time;
        cpu_filter = (res["hardware"] == "cpu") & \
                     (res["rows"] == group_key[0]) & \
                     (res["max_cols"] == group_key[1]) & \
                     (res["nnz_per_row"] == group_key[2]) & \
                     (res["distribution"] == group_key[3])
        median_baseline = np.median(res.loc[cpu_filter, "exec_time_ms"])
        # Compute the speedup for this group;
        group.loc[:, "speedup"] = median_baseline / group["exec_time_ms"]
        group.loc[:, "baseline_time_ms"] = median_baseline
        # Guarantee that the geometric mean of speedup referred to the baseline is 1, and adjust speedups accordingly;
        gmean_speedup = gmean(res.loc[cpu_filter, "speedup"])
        group.loc[:, "speedup"] /= gmean_speedup
        
        # Repeat for SpMV-only baseline time;
        
        # Compute the median baseline computation time;
        spmv_median_baseline = np.median(res.loc[cpu_filter, "spmv_exec_time_ms"])
        # Compute the speedup for this group;
        group.loc[:, "spmv_speedup"] = spmv_median_baseline / group["spmv_exec_time_ms"]
        group.loc[:, "spmv_baseline_time_ms"] = spmv_median_baseline
        # Guarantee that the geometric mean of speedup referred to the baseline is 1, and adjust speedups accordingly;
        spmv_gmean_speedup = gmean(res.loc[cpu_filter, "spmv_speedup"])
        group.loc[:, "spmv_speedup"] /= spmv_gmean_speedup
        
        res.loc[group.index, :] = group
    
    return res
    

def get_fpga_label(l):
    try:
        _, bits, cores, clock = l.split("_")
        bits = f"{bits}b" if bits != "float" else "F32"
        cores = f"{cores}C"
        return f"FPGA\n{bits}\n{cores}"
    except:
        print("warning, cannot format FPGA label:", l)
        return l
    
def get_fpga_legend_label(l, add_freq=False):
    try:
        print(l)
        _, bits, cores, clock = l.split("_")
        bits = f"{bits}b" if bits != "float" else "F32"
        cores = f"{cores}C"
        return f"FPGA {bits}, {cores}, {clock} MHz" if add_freq else f"FPGA {bits}, {cores}"
    except:
        print("warning, cannot format FPGA label:", l)
        return l
    

def plot_bars(res):
    
    sns.set_style("white", {"ytick.left": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 25 
    plt.rcParams['axes.labelpad'] = 9 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    plt.rcParams['xtick.major.pad'] = 1 
    plt.rcParams['hatch.linewidth'] = 0.3
        
    sizes = sorted(res["rows"].unique())[1:] + [sorted(res["rows"].unique())[0]]
    dist = res["distribution"].unique()
    hardware = sorted(res["hardware"].unique())

    num_col = len(sizes)
    num_rows = 1  # len(dist)
    fig = plt.figure(figsize=(1.5 * num_col, 2.6 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_col)
    plt.subplots_adjust(top=0.9,
                    bottom=0.33,
                    left=0.06,
                    right=.99,
                    hspace=0.9,
                    wspace=0.05)
    
    # Old palette;
    palettes = [[COLORS["r1"], COLORS["bb0"], COLORS["bb2"], COLORS["bb3"], COLORS["bb4"], COLORS["bb5"]]] * num_col 
    # Palette used for DATE;
    palettes = [[COLORS["peach1"], COLORS["bb2"], COLORS["bb2"], "#A5E6C6", "#A5E6C6", COLORS["bb5"], COLORS["bb5"]]] * num_col 
    # Orange palette, for GPUs
    palettes = [[COLORS["peach1"], "#FFA880", "#FFB896", "#FFC7AD"]] * num_col 
    # Orange palette, yellow, for GPUs
    palettes_y = [["#FCF061", "#FCF38B", "#FCF49A", "#FCF6B1"]] * num_col 
    # Palette for DAC
    palettes = [[COLORS["peach1"], "#FFA880", COLORS["bb5"], "#A5E6C6", COLORS["bb3"], COLORS["bb2"]]] * num_col 
    palettes_y = [["#FCF49A", "#FCF6B1"]] * num_col 
    # Palette for thesis
    palettes = [["#ED9E6F", "#FFA880"] + ["#E7F7DF", "#B5E8B5", "#71BD9D", "#469C94"][::-1]] * num_col 
    # hatches = [[None, "/" * 7, "\\" * 7, "/" * 7, "\\" * 7, "/" * 7, "\\" * 7] * 2] * num_col 
    hatches = [["/" * 4, "\\" * 4, "/" * 4, "\\" * 4] * 4] * num_col
    
    ii = 0
    # for ii, group_ii in enumerate(res.groupby(["distribution"])):
    #     groups = group_ii[1].groupby(["rows"])
    
    groups = res.groupby(["rows"])
    groups = sorted(groups, key=lambda g: g[0])
    groups = groups[1:] + [groups[0]]
    fpga_labels = []
    for i, group in enumerate(groups):
        ax = fig.add_subplot(gs[ii, i])
        # Replace "float" with "32float" to guarantee the right bar sorting;
        # group[1].loc[group[1]["n_bit"] == "cpu", "n_bit"] = "32cpu"
        
        # Create a unique row id;
        # group[1]["row"] = group[1]["n_bit"].astype(str) + group[1]["n_cores"].astype(str) 
        # group[1]["row_str"] = group[1]["n_bit"].astype(str) + [("b\n" if x !="F32" else "\n") for x in group[1]["n_bit"]] + group[1]["n_cores"].astype(str) + "C"
        
        data = group[1] # .sort_values(["n_bit", "n_cores"], ascending=[False, True]).reset_index(drop=True)
        # Remove CPU;
        data = data[data["hardware"] != "cpu"]

        ax = sns.barplot(x="hardware", y="spmv_speedup", data=data, palette=palettes_y[i], capsize=.05, errwidth=0.8, ax=ax,
                          edgecolor="#2f2f2f")
        ax = sns.barplot(x="hardware", y="speedup", data=data, palette=palettes[i], capsize=.05, errwidth=0.8, ax=ax,
                          edgecolor="#2f2f2f")

        # Set a different hatch for each bar
        for j, bar in enumerate(ax.patches):
            bar.set_hatch(hatches[i][j])
        # sns.despine(ax=ax)
       
        ax.set_ylim((1, int(1.22 * max(np.max(res_agg["spmv_speedup"]), np.max(res_agg["speedup"])))))
        ax.set_ylabel("")
        ax.set_xlabel("")
        labels = ax.get_xticklabels()
        new_fpga_labels = [get_fpga_legend_label(l._text) for l in labels if l._text not in GPU_LABELS]
        if len(new_fpga_labels) > len(fpga_labels):
            fpga_labels = new_fpga_labels
        ax.set_xticklabels([GPU_LABELS[l._text] if (l._text in GPU_LABELS) else get_fpga_label(l._text) for l in labels])
        cpu_label = int(np.mean(group[1][group[1]["hardware"] == "cpu"]["exec_time_ms"]))
        ax.tick_params(axis='x', which='major', labelsize=5, rotation=0)
        
        # Set the y ticks;
        ax.yaxis.set_major_locator(plt.LinearLocator(6))
        if i == 0:
            ax.set_yticklabels(labels=[f"{int(l)}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=8)
        else:
            ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
            # Hide tick markers;
            for tic in ax.yaxis.get_major_ticks():
                tic.tick1line.set_visible(False) 
                tic.tick2line.set_visible(False) 
        ax.xaxis.grid(False)
        
        if i > 0:
            ax.set_yticklabels([])
            # sns.despine(ax=ax, left=True, top=True, right=True)
        ax.yaxis.grid(True, linewidth=0.5)
        
        # Speedup labels;
        offsets = []
        for j, g_tmp in data.groupby(["hardware"]):
            offsets += [get_upper_ci_size(g_tmp["spmv_speedup"], ci=0.80)]
        offsets = [o + 4 if not np.isnan(o) else 0.2 for o in offsets]
        add_labels(ax, vertical_offsets=offsets, rotation=90, fontsize=8, max_only=False, max_bars=6, format_str="{:.0f}x",)
        
        # Add graph type;
        ax.annotate(f"{get_exp_label(sizes[i], 'N=', True)}" if i < 3 else "Sparse GloVe", xy=(0.0, 0.95), fontsize=10, ha="left", xycoords="axes fraction", xytext=(0.0, 1.06))
                
        ax.annotate(f"CPU Baseline:", xy=(0.0, -0.28), fontsize=8, ha="left", xycoords="axes fraction")
        ax.annotate(f"{cpu_label} ms", xy=(0.59, -0.28), fontsize=8, color="#7f7f7f", ha="left", xycoords="axes fraction")
    
    # # Add custom legend;
    labels = ["GPU F32, Top-K SpMV", "GPU F32, SpMV only", "GPU F16, Top-K SpMV", "GPU F16, SpMV only"] + fpga_labels
    if len(labels) < 8:
        labels += ["???"] * (8 - len(labels)) 
    custom_lines = [Patch(facecolor=palettes[0][0], hatch=hatches[0][0], edgecolor="#2f2f2f", label=labels[0]),
                    Patch(facecolor=palettes_y[0][0], hatch=hatches[0][0], edgecolor="#2f2f2f", label=labels[1]),
                    Patch(facecolor=palettes[0][1], hatch=hatches[0][1], edgecolor="#2f2f2f", label=labels[4]),
                    Patch(facecolor=palettes_y[0][1], hatch=hatches[0][1], edgecolor="#2f2f2f", label=labels[5]),
                    Patch(facecolor=palettes[0][2], hatch=hatches[0][2], edgecolor="#2f2f2f", label=labels[2]),
                    Patch(facecolor=palettes[0][3], hatch=hatches[0][3], edgecolor="#2f2f2f", label=labels[3]),
                    Patch(facecolor=palettes[0][4], hatch=hatches[0][4], edgecolor="#2f2f2f", label=labels[6]),
                    Patch(facecolor=palettes[0][5], hatch=hatches[0][5], edgecolor="#2f2f2f", label=labels[7])] 
    leg = fig.legend(custom_lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0), fontsize=7, ncol=4, handletextpad=0.5, columnspacing=0.4)
    leg.set_title(None)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
        
    # plt.suptitle("Execution time speedup on Top-K SpMV for GPU and FPGA vs. CPU", ha="left", x=0.02, y=0.98, fontsize=13)
    
    save_plot("../../../../data/plots", f"exec_time_{DATE}" + ".{}")  
    

def plot_bars_2(res):
    
    res["distribution"] = pd.Categorical(res["distribution"], ["uniform", "glove", "gamma"])
    res["hardware"] = pd.Categorical(res["hardware"], ["cpu", "gpu", "gpu_half"] + sorted([x for x in res["hardware"].unique() if "fpga" in x])[::1])
    res = res.sort_values(["distribution", "hardware"])
    
    sns.set_style("white", {"ytick.left": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 25 
    plt.rcParams['axes.labelpad'] = 9 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    plt.rcParams['xtick.major.pad'] = 1 
    plt.rcParams['hatch.linewidth'] = 0.3
        
    sizes = sorted(res["rows"].unique())[1:] + [sorted(res["rows"].unique())[0]]
    dist = res["distribution"].unique()
    hardware = sorted(res["hardware"].unique())

    num_col = 4
    num_rows = 2
    fig = plt.figure(figsize=(1.5 * num_col, 2.5 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_col)
    plt.subplots_adjust(top=0.94,
                    bottom=0.18,
                    left=0.06,
                    right=.99,
                    hspace=0.6,
                    wspace=0.05)

    palettes_y = [["#FCF49A", "#FCF6B1"]] * num_col * num_rows
    # Palette for thesis
    palettes = [["#ED9E6F", "#FFA880"] + ["#E7F7DF", "#B5E8B5", "#71BD9D", "#469C94"][::-1]] * num_col * num_rows
    # hatches = [[None, "/" * 7, "\\" * 7, "/" * 7, "\\" * 7, "/" * 7, "\\" * 7] * 2] * num_col 
    hatches = [["/" * 4, "\\" * 4, "/" * 4, "\\" * 4] * 4] * num_col * num_rows
        
    groups = res.groupby(["distribution", "rows"])
    for i, group in groups:
        print(i)
    fpga_labels = []
    for i, group in enumerate(groups):
        ax = fig.add_subplot(gs[i // num_col, i % num_col])
        data = group[1] 
        # Remove CPU;
        data = data[data["hardware"] != "cpu"]
        data["hardware"] = data["hardware"].astype(str)
        ax = sns.barplot(x="hardware", y="spmv_speedup", data=data, palette=palettes_y[i], capsize=.05, errwidth=0.8, ax=ax, ci=90,
                          edgecolor="#2f2f2f")
        ax = sns.barplot(x="hardware", y="speedup", data=data, palette=palettes[i], capsize=.05, errwidth=0.8, ax=ax, ci=90,
                          edgecolor="#2f2f2f")
        print(group[1]["hardware"].unique())

        # Set a different hatch for each bar
        for j, bar in enumerate(ax.patches):
            bar.set_hatch(hatches[i][j])

        ax.set_ylim((1, 200))
        ax.set_ylabel("")
        ax.set_xlabel("")
        labels = ax.get_xticklabels()
        new_fpga_labels = [get_fpga_legend_label(l._text) for l in labels if l._text not in GPU_LABELS]
        if len(new_fpga_labels) > len(fpga_labels):
            fpga_labels = new_fpga_labels
        ax.set_xticklabels([GPU_LABELS[l._text] if (l._text in GPU_LABELS) else get_fpga_label(l._text) for l in labels])
        cpu_label = int(np.mean(group[1][group[1]["hardware"] == "cpu"]["exec_time_ms"]))
        ax.tick_params(axis='x', which='major', labelsize=5, rotation=0)
        
        # Set the y ticks;
        ax.yaxis.set_major_locator(plt.LinearLocator(6))
        if i % num_col == 0:
            ax.set_yticklabels(labels=[f"{int(l)}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=8)
        else:
            ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
            # Hide tick markers;
            for tic in ax.yaxis.get_major_ticks():
                tic.tick1line.set_visible(False) 
                tic.tick2line.set_visible(False) 
        ax.xaxis.grid(False)
        
        if i % num_col != 0:
            ax.set_yticklabels([])
            # sns.despine(ax=ax, left=True, top=True, right=True)
        ax.yaxis.grid(True, linewidth=0.5)
        
        # Speedup labels;
        offsets = [l._y[1] for l in ax.lines[::3]][:len(ax.lines) // 2]
        # for j, g_tmp in data.groupby(["hardware"]):
        #     offsets += [get_upper_ci_size(g_tmp["spmv_speedup"], ci=0.7)]
        print(offsets)
        offsets = [o + 4 if not np.isnan(o) else 0.2 for o in offsets]
        add_labels(ax, vertical_coords=offsets, rotation=90, fontsize=8, max_only=False, max_bars=6, format_str="{:.0f}x",)
        
        # Add graph type;
        dist_dict = {"uniform": "Uniform", "gamma": r"$\Gamma$"}
        graph_name = f"{dist_dict[group[0][0]]}, {get_exp_label(group[0][1], 'N=', True)}" if group[0][0] != "glove" else "Sparse GloVe"
        ax.annotate(graph_name, xy=(0.0, 0.95), fontsize=10, ha="left", xycoords="axes fraction", xytext=(0.0, 1.06))
                
        ax.annotate(f"CPU Baseline:", xy=(0.0, -0.27), fontsize=8, ha="left", xycoords="axes fraction")
        ax.annotate(f"{cpu_label} ms", xy=(0.59, -0.27), fontsize=8, color="#7f7f7f", ha="left", xycoords="axes fraction")
    
    # # Add custom legend;
    labels = ["GPU F32, Top-K SpMV", "GPU F32, SpMV only", "GPU F16, Top-K SpMV", "GPU F16, SpMV only"] + fpga_labels
    labels = [x for i in range(4) for x in [labels[i], labels[i + 4]]]
    if len(labels) < 8:
        labels += ["???"] * (8 - len(labels)) 
    custom_lines = [Patch(facecolor=palettes[0][0], hatch=hatches[0][0], edgecolor="#2f2f2f"),
                    Patch(facecolor=palettes[0][2], hatch=hatches[0][2], edgecolor="#2f2f2f"),
                    Patch(facecolor=palettes_y[0][0], hatch=hatches[0][0], edgecolor="#2f2f2f"),
                    Patch(facecolor=palettes[0][3], hatch=hatches[0][3], edgecolor="#2f2f2f"),
                    Patch(facecolor=palettes[0][1], hatch=hatches[0][1], edgecolor="#2f2f2f"),
                    Patch(facecolor=palettes[0][4], hatch=hatches[0][4], edgecolor="#2f2f2f"),
                    Patch(facecolor=palettes_y[0][1], hatch=hatches[0][1], edgecolor="#2f2f2f"),
                    Patch(facecolor=palettes[0][5], hatch=hatches[0][5], edgecolor="#2f2f2f")] 
    leg = fig.legend(custom_lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0), fontsize=7, ncol=4, handletextpad=0.5, columnspacing=0.4)
    leg.set_title(None)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white')
        
    # plt.suptitle("Execution time speedup on Top-K SpMV for GPU and FPGA vs. CPU", ha="left", x=0.02, y=0.98, fontsize=13)
    
    save_plot("../../../../data/plots", f"exec_time_large_{DATE}" + ".{}")  

#%%  
    
if __name__ == "__main__":
    
    # sns.set_style("whitegrid", {"ytick.left": True})
    # plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    # plt.rcParams['axes.titlepad'] = 25 
    # plt.rcParams['axes.labelpad'] = 9 
    # plt.rcParams['axes.titlesize'] = 22 
    # plt.rcParams['axes.labelsize'] = 14 
    # plt.rcParams['xtick.major.pad'] = 1 
    # plt.rcParams['hatch.linewidth'] = 0.3
    
    # result_df, cpu_result_df, res, res_agg = read_datasets()       
    
    # sizes = sorted(res["rows"].unique())
    # n_bits = ["cpu"] + sorted(res["n_bit"].unique(), reverse=True)[1:]

    # num_col = len(sizes)
    # fig = plt.figure(figsize=(1.6 * num_col, 2.5))
    # gs = gridspec.GridSpec(1, num_col)
    # plt.subplots_adjust(top=0.6,
    #                 bottom=0.2,
    #                 left=0.07,
    #                 right=.99,
    #                 hspace=0.9,
    #                 wspace=0.1)
    
    # palettes = [[COLORS["r1"], COLORS["bb0"], COLORS["bb2"], COLORS["bb3"], COLORS["bb4"], COLORS["bb5"]]] * num_col 
    # palettes = [[COLORS["peach1"], COLORS["bb2"], COLORS["bb2"], "#A5E6C6", "#A5E6C6", COLORS["bb5"], COLORS["bb5"]]] * num_col 
    # hatches = [[None, "/" * 7, "\\" * 7, "/" * 7, "\\" * 7, "/" * 7, "\\" * 7]] * num_col 
    
    # groups = res.groupby(["rows"])
    
    # for i, group in enumerate(groups):
    #     ax = fig.add_subplot(gs[0, i])
        
    #     # Replace "float" with "32float" to guarantee the right bar sorting;
    #     group[1].loc[group[1]["n_bit"] == "cpu", "n_bit"] = "32cpu"
        
    #     # Create a unique row id;
    #     group[1]["row"] = group[1]["n_bit"].astype(str) + group[1]["n_cores"].astype(str) 
    #     group[1]["row_str"] = group[1]["n_bit"].astype(str) + [("b\n" if x !="F32" else "\n") for x in group[1]["n_bit"]] + group[1]["n_cores"].astype(str) + "C"
        
    #     data = group[1].sort_values(["n_bit", "n_cores"], ascending=[False, True]).reset_index(drop=True)
    #     # Remove CPU;
    #     data = data[data["n_bit"] != "32cpu"]

    #     ax = sns.barplot(x="row_str", y="speedup", data=data, palette=palettes[i], capsize=.05, errwidth=0.8, ax=ax,
    #                       edgecolor="#2f2f2f")
    #     # Set a different hatch for each bar
    #     for j, bar in enumerate(ax.patches):
    #         bar.set_hatch(hatches[i][j])
    #     sns.despine(ax=ax)
       
    #     ax.set_ylim((1, int(1.1 * np.max(res_agg["speedup"]))))
    #     ax.set_ylabel("")
    #     ax.set_xlabel("")
    #     labels = ax.get_xticklabels()
    #     cpu_label = int(np.mean(group[1][group[1]["n_bit"] == "32cpu"]["exec_time_ms"]))
    #     # for j, l in enumerate(labels):
    #     #     if j == 0:
    #     #         l.set_text(f"CPU")
    #     #     elif (j == 1) and len(labels) > 5:
    #     #        l.set_text("F32")
               
    #     # ax.set_xticklabels(labels)
    #     ax.tick_params(axis='x', which='major', labelsize=7, rotation=0)
        
    #     # Set the y ticks;
    #     ax.yaxis.set_major_locator(plt.LinearLocator(6))
    #     if i == 0:
    #         ax.set_yticklabels(labels=[f"{int(l)}x".format(l) for l in ax.get_yticks()], ha="right", fontsize=8)
    #     else:
    #         ax.set_yticklabels(labels=["" for l in ax.get_yticks()])
    #         # Hide tick markers;
    #         for tic in ax.yaxis.get_major_ticks():
    #             tic.tick1line.set_visible(False) 
    #             tic.tick2line.set_visible(False) 
    #     ax.xaxis.grid(False)
        
    #     if i > 0:
    #         ax.set_yticklabels([])
    #         # ax.get_yaxis().set_visible(False)
    #         sns.despine(ax=ax, left=True, top=True, right=True)
    #         ax.yaxis.grid(True)
        
    #     # Speedup labels;
    #     offsets = []
    #     for j, g_tmp in data.groupby(["n_bit", "n_cores"]):
    #         offsets += [get_upper_ci_size(g_tmp["speedup"], ci=0.80)]
    #     offsets = [o + 2 if not np.isnan(o) else 0.2 for o in offsets]
    #     add_labels(ax, vertical_offsets=offsets, rotation=90, fontsize=7, max_only=False)
        
    #     # Add graph type;
    #     ax.annotate(r"$\mathdefault{N=10^" + f"{int(np.log10(group[0]))}" + r"}$", xy=(0.0, 0.9), fontsize=10, ha="left", xycoords="axes fraction", xytext=(0.0, 1.1))
                
    #     ax.annotate(f"CPU Baseline:", xy=(0.0, -0.4), fontsize=8, ha="left", xycoords="axes fraction")
    #     ax.annotate(f"{cpu_label} ms", xy=(0.56, -0.4), fontsize=8, color=COLORS["peach1"], ha="left", xycoords="axes fraction")
        
    # # Add custom legend;
    # labels = ["Float 32, 21 cores", "32 bits, 16 cores", "32 bits, 24 cores", "24 bits, 16 cores", "24 bits, 28 cores", "20 bits, 16 cores", "20 bits, 32 cores"]
    # custom_lines = [Patch(facecolor=palettes[0][j], hatch=hatches[0][j], edgecolor="#2f2f2f", label=labels[j]) for j in range(len(labels))]
    # leg = fig.legend(custom_lines, labels, bbox_to_anchor=(1, 1), fontsize=8, ncol=2, handletextpad=0.5, columnspacing=0.4)
    # leg.set_title(None)
    # leg._legend_box.align = "left"
    # leg.get_frame().set_facecolor('white')
        
    # plt.suptitle("Exec. time speedup for\ndifferent bit-widths\nand number of cores", ha="left", x=0.02, y=0.98, fontsize=12)
    
    # plt.savefig(f"../../../../data/plots/exec_time_{DATE}.pdf")
    
    
    #%% New plots with GPU

    results_cpu, results_cpu_grouped = read_results_cpu(CPU_RESULT_FOLDER)
    results_gpu, results_gpu_grouped = read_results_gpu(GPU_RESULT_FOLDER)
    results_fpga, results_fpga_grouped = read_results_fpga(FPGA_RESULT_FOLDER)
        
    res = join_datasets(results_cpu, results_gpu, results_fpga)
    
    res_agg = res.groupby(['rows', 'max_cols', 'nnz_per_row', 'distribution', 'hardware'])[['error_idx', 'error_val',
       'exec_time_ms', 'spmv_exec_time_ms', 'speedup', 'spmv_speedup',
       'baseline_time_ms', 'spmv_baseline_time_ms']].agg(np.mean)
    
    # res = res[res["distribution"] == "uniform"]
    
    # %%
    plot_bars(res)
    plot_bars_2(res)
  