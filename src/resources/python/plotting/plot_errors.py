#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:30:36 2020

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
from matplotlib.patches import Patch, Rectangle
from plot_exec_time import get_exp_label, get_upper_ci_size
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.lines import Line2D
from plot_utils import *

bt1 = "#55819E"
bt2 = "#538F6F"

FPGA_RESULT_FOLDER = "../../../../data/results/fpga/2020_11_21_15_07_03"
GPU_RESULT_FOLDER = "../../../../data/results/gpu/2020_11_19_15_39_53"
# FPGA_RESULT_FOLDER = "../../../../data/results/fpga/2020_11_22"
# GPU_RESULT_FOLDER = "../../../../data/results/gpu/2020_11_22"
DATE = "2021_03_07"

THRESHOLDS = [8, 16, 32, 50, 75, 100]

def read_data_fpga():
    result_list = []
    
    for f in os.listdir(FPGA_RESULT_FOLDER):
        res_file = os.path.join(FPGA_RESULT_FOLDER, f)
        if res_file.endswith(".csv"):
            with open(res_file) as file:
                result = file.readlines()[1:]
    
                # Parse the file name;
                hardware, rows, max_cols, distribution, nnz_per_row, n_bit, n_cores, mhz, k, n_iter = os.path.splitext(f)[0].split("_")
                n_cores = int(n_cores.replace("core", ""))
                # Parse the file name;
                try:
                    n_bit = int(n_bit[:-3]) if n_bit != "float" else "F32"
                except ValueError:
                    pass
                
                for r in result:
                    try:
                        iteration, error_idx, error_val, sw_full_time_ms, sw_topk_time_ms, \
                        hw_setup_time_ms, hw_exec_time_ms, full_hw_exec_time_ms, readback_time_ms, k, sw_res_idx, \
                        sw_res_val, hw_res_idx, hw_res_val = r.split(",")
                    except ValueError:
                        iteration, error_idx, error_val, sw_full_time_ms, sw_topk_time_ms, \
                        hw_setup_time_ms, hw_exec_time_ms, readback_time_ms, k, sw_res_idx, \
                        sw_res_val, hw_res_idx, hw_res_val = r.split(",")
                    k = int(k)
                    
                    # Process results;
                    sw_res_idx = [int(x) for x in sw_res_idx.split(";")]
                    sw_res_val = [float(x) for x in sw_res_val.split(";")]
                    hw_res_idx = [int(x) for x in hw_res_idx.split(";")][:k]
                    hw_res_val = [float(x) for x in hw_res_val.split(";")][:k]
                    assert(len(sw_res_idx) == k)
                    assert(len(sw_res_val) == k)
                    assert(len(hw_res_idx) == k)
                    assert(len(hw_res_val) == k)
                    
                    prec = []
                    kendall = []
                    ndcg_vals = []
                    for t in THRESHOLDS:
                        set_cpu = set(sw_res_idx[:t])
                        set_fpga = set(hw_res_idx[:t])
                        prec += [len(set_cpu.intersection(set_fpga)) / t]                   
                        kendall += [kendall_tau(sw_res_idx[:t], hw_res_idx[:t])]
                        ndcg_vals += [ndcg(sw_res_idx[:t], sw_res_val[:t], hw_res_idx[:t], hw_res_val[:t])[0]]
    
                    # Add the result line to the list;
                    new_res_line = [hardware, int(rows), int(max_cols), distribution, int(nnz_per_row), str(n_bit), int(n_cores), int(iteration), int(n_iter), int(error_idx), int(error_val), \
                                    float(sw_full_time_ms), float(sw_topk_time_ms), float(hw_setup_time_ms),
                                    float(hw_exec_time_ms), float(readback_time_ms), int(k)] + prec + kendall + ndcg_vals
                    if float(hw_exec_time_ms) <= (100 if n_bit != "F32" else 300):
                        result_list += [new_res_line]
                        
    # Create a dataframe;
    result_df = pd.DataFrame(result_list,
                             columns=["hardware", "rows", "max_cols", "distribution", "nnz_per_row", "n_bit", "n_cores", "n_iter",  "max_iter", "error_idx", "error_val",
                                      "sw_full_time_ms", "sw_topk_time_ms", "hw_setup_time_ms",
                                      "hw_exec_time_ms", "readback_time_ms", "k"]
                             + [f"prec_{x}" for x in THRESHOLDS]
                             + [f"kendall_{x}" for x in THRESHOLDS]
                             + [f"ndcg_{x}" for x in THRESHOLDS])
    
    # Remove outliers;
    res = remove_outliers_df_grouped(result_df, "hw_exec_time_ms", ["hardware", "rows", "max_cols", "distribution", "nnz_per_row", "n_bit", "n_cores"], reset_index=True, drop_index=True, sigmas=2)
     
    return res, res.groupby(["hardware", "rows", "max_cols", "distribution", "nnz_per_row", "n_bit", "n_cores"]).aggregate(np.mean).reset_index()


def read_data_gpu():
    result_list = []
    
    for f in os.listdir(GPU_RESULT_FOLDER):
        res_file = os.path.join(GPU_RESULT_FOLDER, f)
        if res_file.endswith(".csv"):
            with open(res_file) as file:
                result = file.readlines()[1:]
    
                # Parse the file name;
                hardware, rows, max_cols, distribution, nnz_per_row, impl, half_precision, k, n_iter = os.path.splitext(f)[0].split("_")
                n_cores = 56
                # Parse the file name;
                try:
                    n_bit = "F16" if half_precision == "True" else "F32"
                except ValueError:
                    pass
                
                for r in result:
                    iteration, error_idx, error_val, sw_full_time_ms, sw_topk_time_ms, \
                    hw_setup_time_ms, hw_spmv_only_time_ms, hw_exec_time_ms, readback_time_ms, k, sw_res_idx, \
                    sw_res_val, hw_res_idx, hw_res_val = r.split(",")
                    k = int(k)
                    
                    # Process results;
                    sw_res_idx = [int(x) for x in sw_res_idx.split(";")]
                    sw_res_val = [float(x) for x in sw_res_val.split(";")]
                    hw_res_idx = [int(x) for x in hw_res_idx.split(";")][:k]
                    hw_res_val = [float(x) for x in hw_res_val.split(";")][:k]
                    assert(len(sw_res_idx) == k)
                    assert(len(sw_res_val) == k)
                    assert(len(hw_res_idx) == k)
                    assert(len(hw_res_val) == k)
                    
                    prec = []
                    kendall = []
                    ndcg_vals = []
                    for t in THRESHOLDS:
                        set_cpu = set(sw_res_idx[:t])
                        set_fpga = set(hw_res_idx[:t])
                        prec += [len(set_cpu.intersection(set_fpga)) / t]                   
                        kendall += [kendall_tau(sw_res_idx[:t], hw_res_idx[:t])]
                        ndcg_vals += [ndcg(sw_res_idx[:t], sw_res_val[:t], hw_res_idx[:t], hw_res_val[:t])[0]]
    
                    # Add the result line to the list;
                    new_res_line = [hardware, int(rows), int(max_cols), distribution, int(nnz_per_row), str(n_bit), int(n_cores), impl, int(iteration), int(n_iter), int(error_idx), int(error_val), \
                                    float(sw_full_time_ms), float(sw_topk_time_ms), float(hw_setup_time_ms), float(hw_spmv_only_time_ms),
                                    float(hw_exec_time_ms), float(readback_time_ms), int(k)] + prec + kendall + ndcg_vals
                    result_list += [new_res_line]
                        
    # Create a dataframe;
    result_df = pd.DataFrame(result_list,
                             columns=["hardware", "rows", "max_cols", "distribution", "nnz_per_row", "n_bit", "n_cores", "impl", "n_iter",  "max_iter", "error_idx", "error_val",
                                      "sw_full_time_ms", "sw_topk_time_ms", "hw_setup_time_ms", "hw_spmv_only_time_ms",
                                      "hw_exec_time_ms", "readback_time_ms", "k"]
                             + [f"prec_{x}" for x in THRESHOLDS]
                             + [f"kendall_{x}" for x in THRESHOLDS]
                             + [f"ndcg_{x}" for x in THRESHOLDS])
    
    # Remove outliers;
    res = remove_outliers_df_grouped(result_df, "hw_exec_time_ms", ["hardware", "rows", "max_cols", "distribution", "nnz_per_row", "n_bit", "n_cores", "impl"], reset_index=True, drop_index=True, sigmas=2)
     
    return res, res.groupby(["hardware", "rows", "max_cols", "distribution", "nnz_per_row", "n_bit", "n_cores", "impl"]).aggregate(np.mean).reset_index()


def kendall_tau(reference_rank, predicted_rank):
    
    # Items with correct relative rank;
    c_plus = 0
    # Items without correct relative rank;
    c_minus = 0
    # Items for which a ranking exists in the predicted rank;
    c_s = 0
    # Items for which a ranking exists in the reference rank;
    c_u = 0
    
    item_set = set(reference_rank + predicted_rank)
    reference_rank_dict = {item: pos for pos, item in enumerate(reference_rank)}
    predicted_rank_dict = {item: pos for pos, item in enumerate(predicted_rank)}
    
    for i, item_1 in enumerate(item_set):
        for j, item_2 in enumerate(item_set):
            # Consider each pair exactly once;
            if i >= j:
                continue
            else:
                ref_found = False
                pred_found = False
                if item_1 in reference_rank_dict and item_2 in reference_rank_dict:
                    ref_found = True
                    c_u += 1
                if item_1 in predicted_rank_dict and item_2 in predicted_rank_dict:
                    pred_found = True
                    c_s += 1
                if ref_found and pred_found:
                    if (reference_rank_dict[item_1] - reference_rank_dict[item_2]) * (predicted_rank_dict[item_1] - predicted_rank_dict[item_2]) > 0:
                        c_plus += 1
                    else:
                        c_minus += 1
                        
    return (c_plus - c_minus) / (np.sqrt(c_u) * np.sqrt(c_s))


def ndcg(sw_res_idx, sw_res_val, hw_res_idx, hw_res_val): 
    
    sw_res = {k: v for (k, v) in zip(sw_res_idx, sw_res_val)}
    dcg = 0
    idcg = 0
    for i, (idx, res) in enumerate(zip(hw_res_idx, hw_res_val)):
        relevance = sw_res[idx] if idx in sw_res else 0
        dcg += relevance / np.log2(i + 1 + 1)
    for i, (idx, res) in enumerate(zip(sw_res_idx, sw_res_val)):
        relevance = res
        idcg += relevance / np.log2(i + 1 + 1)
    return dcg / idcg, dcg, idcg


if __name__ == "__main__":
    
    sns.set_style("white", {"ytick.left": True})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 40 
    plt.rcParams['axes.labelpad'] = 4
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    plt.rcParams['xtick.major.pad'] = 10

    res_fpga, agg_fpga = read_data_fpga()
    res_gpu, agg_gpu = read_data_gpu()
    
    #%%
    
     # Filter wrong output data;
    agg_fpga = agg_fpga[agg_fpga["prec_100"] > 0.2]
    agg_fpga = agg_fpga[agg_fpga["n_bit"].isin(["20", "32", "F32"])]
    
    agg_gpu = agg_gpu[agg_gpu["n_bit"] == "F16"]
    agg_gpu = agg_gpu[agg_gpu["impl"] == "0"]
     
    agg = pd.concat([agg_fpga, agg_gpu], ignore_index=True).reset_index(drop=True)
    
    agg = agg[agg["max_cols"] != 512]
        
    old_size = len(agg)
    agg = agg[agg["prec_100"] > 0.9]
    if len(agg) < old_size:
        print(f"removed {len(agg)- old_size} rows with low precision")
    z_added = False
    
    #%%
    
    # Setup plot;
    plt.rcParams['mathtext.fontset'] = "cm" 
    error_metrics = ["Precision", "Kendall's " + r"$\mathbf{\tau}$", "NDCG"]
    error_metrics_raw = ["prec", "kendall", "ndcg"]
    error_max = [1, 1, 1]
    error_min = [0.96, 0.95, 0.96]
    
    sizes = sorted(agg["rows"].unique())
    sizes = sizes[1:] + [sizes[0]]
    num_col = len(sizes)
    num_rows = len(error_metrics) 
    fig = plt.figure(figsize=(1.6 * num_col, 1.5 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_col)
    plt.subplots_adjust(top=0.9,
                    bottom=0.22,
                    left=0.13,
                    right=0.97,
                    hspace=0.5,
                    wspace=0.5)

    markers = ["o", "X", "D", "P", "P"]
    palette = [COLORS["peach1"], COLORS["bb2"], "#A5E6C6", COLORS["bb5"], COLORS["bb5"]]           
    palette = [COLORS["bb2"], "#A5E6C6", COLORS["bb5"], COLORS["peach1"]]   
    palette_dict = {"z_F16": COLORS["peach1"], "F32": COLORS["bb5"], "32":"#A5E6C6" , "20": C<OLORS["bb2"]}
    markers_dict = {"z_F16": "P", "F32": "D", "32": "X", "20": "o"}

    if not z_added:
        z_added = True
        agg.loc[agg["hardware"] == "gpu", "n_bit"] = "z_" + agg.loc[agg["hardware"] == "gpu", "n_bit"] 

    # One row per graph;
    for i, size in enumerate(sizes):
        
        data = agg[agg["rows"] == size]
        data = data.melt(id_vars=["n_bit"], value_vars=[e + "_" + str(d) for e in error_metrics_raw for d in THRESHOLDS])
        data["error_type"] = [s.split("_")[0] for s in data["variable"]]
        data["error_size"] = [int(s.split("_")[1]) for s in data["variable"]]
        
        # One column per error metric;
        for j, e in enumerate(error_metrics_raw):
            
            curr_data = data[data["error_type"] == e]
            curr_data = data[data["error_size"] >= error_min[j]]
            curr_data["error_size"] = curr_data["error_size"].astype(str)
            # data = group[1].sort_values(["n_bit"], ascending=False).reset_index(drop=True)
            order = sorted(data["n_bit"].unique(), reverse=False)
            
            ax = fig.add_subplot(gs[j, i])
            colors = len(curr_data["n_bit"].unique())
            ax = sns.lineplot(x="error_size", y="value", hue="n_bit", data=curr_data, ax=ax, sort=False, palette=palette_dict,
                  err_style="bars", linewidth=2, legend=False, zorder=2, ci=None, hue_order=order, clip_on=False)
            data_averaged = curr_data.groupby(["n_bit", "error_size"], as_index=False).mean()
            ax = sns.scatterplot(x="error_size", y="value", hue="n_bit", data=data_averaged, ax=ax, edgecolor="#0f0f0f", palette=palette_dict,
                  size_norm=30, legend=False, zorder=3, ci=None, markers=markers_dict, style="n_bit", linewidth=0.05, hue_order=order, style_order=order, clip_on=False)
            ax.set_ylim([error_min[j], error_max[j]])
            # ax.set_xlim([min(curr_data["n_bit"]), max(curr_data["n_bit"])])
            ax.set_xlabel(None)
            if i == 0:
                ax.set_ylabel(f"{error_metrics[j]}", fontsize=12)
            else:
                ax.set_ylabel(None)
            # Matrix name;
            if j == 0:
                ax.annotate(f"{get_exp_label(sizes[i], 'N=', True)}" if i < 3 else "Sparse GloVe", xy=(0.5 if i < 3 else 0.4, 1), xycoords="axes fraction", fontsize=12, textcoords="offset points", xytext=(0, 15),
                            horizontalalignment="center", verticalalignment="center")
            ax.yaxis.set_major_locator(plt.LinearLocator(5))

            # sns.despine(ax=ax)
            ax.xaxis.grid(False)  
            # if i > 0:                
                # sns.despine(ax=ax, left=False, top=True, right=True)
            ax.yaxis.grid(True)
               
            if j == 0:
                ax.set_yticklabels(labels=[f"{int(l * 100)}%" for l in ax.get_yticks()], ha="right")
            else:
                ax.set_yticklabels(labels=[f"{l:.3f}" for l in ax.get_yticks()], ha="right")
            
            # sns.despine(ax=ax)              
            ax.tick_params(labelcolor="black", labelsize=9, pad=2)  
            ax.tick_params(axis='x', which='major', rotation=0, labelcolor="black", labelsize=9, pad=2)
            for tic in ax.xaxis.get_major_ticks():
                tic.tick1line.set_visible(True) 
            
    plt.annotate("Top-K  (from 8 to 100)", fontsize=12, xy=(0.5, 0.125), xycoords="figure fraction", ha="center")
           
    # fig.suptitle("Top-K SpMV accuracy for\ndifferent architectures",
                 # fontsize=16, ha="left", x=0.03)
    # plt.annotate("(higher is better)", fontsize=14, xy=(0.03, 0.86), xycoords="figure fraction", ha="left")
    
    # Legend;
    labels = ["FPGA 20b", "FPGA 32b", "FPGA F32", "GPU F16"]
    custom_lines = [
        Line2D([], [], color="white", marker=markers[0],
               markersize=10, label=labels[0], markerfacecolor=palette[0], markeredgecolor="#2f2f2f"),
        Line2D([], [], color="white", marker=markers[1],
               markersize=10, label=labels[1], markerfacecolor=palette[1], markeredgecolor="#2f2f2f"),
        Line2D([], [], color="white", marker=markers[2],
               markersize=10, label=labels[2], markerfacecolor=palette[2], markeredgecolor="#2f2f2f"),
        Line2D([], [], color="white", marker=markers[3],
                markersize=10, label=labels[3], markerfacecolor=palette[3], markeredgecolor="#2f2f2f"),
        ]
    
    leg = fig.legend(custom_lines,labels,
                             bbox_to_anchor=(0.5, 0), fontsize=12, ncol=4, handletextpad=0.3, loc="lower center", columnspacing=0.4)
    leg.set_title(None)
    leg._legend_box.align = "left"
            
    plt.savefig(f"../../../../data/plots/errors_{DATE}.pdf")
    

    #%%
    
    # Setup plot;
    # plt.rcParams['mathtext.fontset'] = "cm" 
    # error_metrics = ["Precision", "Kendall's " + r"$\mathbf{\tau}$", "NDCG"]
    # error_metrics_raw = ["prec", "kendall", "ndcg"]
    # error_max = [1, 1, 1]
    # error_min = [0.4, 0.4, 0.8]
    
    # sizes = sorted(agg["rows"].unique())[-3:]
    # num_col = len(sizes) * 2
    # num_rows = len(error_metrics) 
    # fig = plt.figure(figsize=(1.1 * num_col, 1.8 * num_rows))
    # gs = gridspec.GridSpec(num_rows, num_col)
    # plt.subplots_adjust(top=0.72,
    #                 bottom=0.12,
    #                 left=0.2,
    #                 right=0.95,
    #                 hspace=0.5,
    #                 wspace=0.1)

    # markers = [["o", "X", "D", "P"], ["X", "D", "P"]]
    # palette = [[COLORS["peach1"], COLORS["bb2"], "#A5E6C6", COLORS["bb5"]], [COLORS["bb2"], "#A5E6C6", COLORS["bb5"]]]     
    
    # agg["group"] = [1 if (x[0] == 16) else 0 for x in zip(agg["n_cores"], agg["n_bit"])]  
    
    # # One row per graph;
    # for i in range(num_col):
    #     g = i % 2
    #     size = sizes[i // 2]
        
    #     data = agg[agg["group"] == g]
        
    #     data = data[data["rows"] == size]
    #     data = data.melt(id_vars=["n_bit", "n_cores"], value_vars=[e + "_" + str(d) for e in error_metrics_raw for d in THRESHOLDS])
    #     data["error_type"] = [s.split("_")[0] for s in data["variable"]]
    #     data["error_size"] = [int(s.split("_")[1]) for s in data["variable"]]
        
    #     # One column per error metric;
    #     for j, e in enumerate(error_metrics_raw):
            
    #         curr_data = data[data["error_type"] == e]
    #         curr_data["error_size"] = curr_data["error_size"].astype(str)
    #         # data = group[1].sort_values(["n_bit"], ascending=False).reset_index(drop=True)
    #         order = sorted(data["n_bit"].unique(), reverse=True)
            
    #         ax = fig.add_subplot(gs[j, i])
    #         ax = sns.lineplot(x="error_size", y="value", hue="n_bit", data=curr_data, ax=ax, sort=False, palette=palette[g],
    #               err_style="bars", linewidth=2, legend=False, zorder=2, ci=None, hue_order=order)
    #         data_averaged = curr_data.groupby(["n_bit", "error_size"], as_index=False).mean()
    #         ax = sns.scatterplot(x="error_size", y="value", hue="n_bit", data=data_averaged, ax=ax, edgecolor="#0f0f0f", palette=palette[g],
    #               size_norm=30, legend=False, zorder=3, ci=None, markers=markers[g], style="n_bit", linewidth=0.05, hue_order=order, style_order=order)
    #         ax.set_ylim([error_min[j], error_max[j]])
    #         # ax.set_xlim([min(curr_data["n_bit"]), max(curr_data["n_bit"])])
    #         ax.set_xlabel(None)
    #         if i == 0:
    #             ax.set_ylabel(f"{error_metrics[j]}", fontsize=12)
    #         else:
    #             ax.set_ylabel(None)
    #         # Matrix name;
    #         if j == 0:
    #             ax.annotate(r"$\mathdefault{N=10^" + f"{int(np.log10(sizes[i // 2]))}" + r"}$",
    #                         xy=(0.5, 1), xycoords="axes fraction", fontsize=14, textcoords="offset points", xytext=(0, 15),
    #                         horizontalalignment="center", verticalalignment="center")
    #         ax.yaxis.set_major_locator(plt.LinearLocator(5))

    #         sns.despine(ax=ax)
    #         ax.xaxis.grid(False)  
    #         if i > 0:                # Hide tick markers;
    #             for tic in ax.yaxis.get_major_ticks():
    #                 tic.tick1line.set_visible(False) 
    #                 tic.tick2line.set_visible(False) 
    #             ax.set_yticklabels([])
    #             # ax.get_yaxis().set_visible(False)
    #             sns.despine(ax=ax, left=True, top=True, right=True)
    #             ax.yaxis.grid(True)
               
    #         # if j == 2:
    #         #     ax.set_yticklabels(labels=[f"{int(l * 100)}%" for l in ax.get_yticks()], ha="right")
                
    #         # Turn off tick lines;
            
    #         # sns.despine(ax=ax)              
    #         ax.tick_params(labelcolor="black", labelsize=10, pad=6)  
    #         ax.tick_params(axis='x', which='major', labelsize=10, rotation=0)
            
    # plt.annotate("Top-K Value", fontsize=14, xy=(0.5, 0.015), xycoords="figure fraction", ha="center")
           
    # fig.suptitle("Top-K SpMV accuracy for\ndifferent architectures",
    #              fontsize=16, ha="left", x=0.03)
    # plt.annotate("(higher is better)", fontsize=14, xy=(0.03, 0.86), xycoords="figure fraction", ha="left")
    
    # # Legend;
    # # labels = ["Float 32, 16 cores", "32 bits, 16 cores", "24 bits, 28 cores", "20 bits, 32 cores", ]
    # # custom_lines = [
    # #     Line2D([], [], color="white", marker=markers[0],
    # #            markersize=10, label=labels[0], markerfacecolor=palette[0], markeredgecolor="#2f2f2f"),
    # #     Line2D([], [], color="white", marker=markers[1],
    # #            markersize=10, label=labels[1], markerfacecolor=palette[1], markeredgecolor="#2f2f2f"),
    # #     Line2D([], [], color="white", marker=markers[2],
    # #            markersize=10, label=labels[2], markerfacecolor=palette[2], markeredgecolor="#2f2f2f"),
    # #     Line2D([], [], color="white", marker=markers[3],
    # #             markersize=10, label=labels[3], markerfacecolor=palette[3], markeredgecolor="#2f2f2f"),
    # #     ]
    
    # # leg = fig.legend(custom_lines,labels,
    # #                          bbox_to_anchor=(0.98, 1), fontsize=12, ncol=1)
    # # leg.set_title(None)
    # # leg._legend_box.align = "left"
            
    # plt.savefig(f"../../../../data/plots/errors_2_{DATE}.pdf")
    

    
    