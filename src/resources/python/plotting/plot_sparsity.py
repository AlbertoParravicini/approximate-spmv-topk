#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 12:49:37 2020

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
from plot_errors import kendall_tau


# Define some colors;
c1 = "#b1494a"
c2 = "#256482"
c3 = "#2f9c5a"
c4 = "#28464f"
c5 = "#FFEA70"

r4 = "#CE1922"
r3 = "#F41922"
r2 = "#FA3A51"
r1 = "#FA4D4A"
r5 = "#F07B71"
r6 = "#F0A694"

b1 = "#97E6DB"
b2 = "#C6E6DB"
b3 = "#CEF0E4"
b4 = "#9CCFC4"

b5 = "#AEDBF2"
b6 = "#B0E6DB"
b7 = "#B6FCDA"
b8 = "#7bd490"

bt1 = "#55819E"
bt2 = "#538F6F"

bt1 = "#55819E"
bt2 = "#538F6F"

bb0 = "#FFA685"
bb1 = "#75B0A2"
bb2 = b3
bb3 = b7
bb4 = "#7ED7B8"
bb5 = "#7BD490"

MAIN_RESULT_FOLDER = "../../../../data/results/raw_results/2020_05_16"
DATE = "2020_07_16"

def read_data():
    
    result_list = []
    
    for graph_name in os.listdir(MAIN_RESULT_FOLDER):
        dir_name = os.path.join(MAIN_RESULT_FOLDER, graph_name)
        if os.path.isdir(dir_name):
            for res_path in os.listdir(dir_name):
                res_file = os.path.join(MAIN_RESULT_FOLDER, graph_name, res_path)
                if res_file.endswith(".csv"):
                    with open(res_file, "r") as f:
                        # Read results, but skip the header;
                        try:
                            result = f.readlines()[1]
                        except IndexError:
                            print(f"skip empty file {res_file}")
                            continue
                        
                        # Parse the file name;
                        try:
                            _, _, n_bit, n_ppr, max_iter, _, _, _, _, _, n_iter = res_path[:-4].split("-")
                        except ValueError:
                            _, _, n_bit, n_ppr, max_iter, _, _, _, _, n_iter = res_path[:-4].split("-")
                        max_iter = max_iter.replace("it", "")
                        # Parse the result line;
                        skip_extra = False
                        try:
                            _, _, V, E, exec_time_ms, transfer_time_ms, errors, ndcg, edit_dist, convergence_error, mae, fpga_predictions, cpu_predictions = result.split(",")
                        except ValueError:
                            _, _, V, E, exec_time_ms, transfer_time_ms, errors, ndcg, edit_dist = result.split(",")
                            convergence_error = ""
                            mae = ""
                            fpga_predictions = ""
                            cpu_predictions = ""
                            skip_extra = True
                        
                        n_ppr = int(n_ppr)
                        if n_ppr == 8:
                        
                            # Obtain the single error metrics;
                            errors = errors.split(";")
                            ndcg = ndcg.split(";")
                            edit_dist = edit_dist.split(";")
                            mae = mae.split(";")
                            errors_10 = [int(e.split("|")[0]) for e in errors if e.strip()]
                            errors_20 = [int(e.split("|")[1]) for e in errors if e.strip()]
                            errors_50 = [int(e.split("|")[2]) for e in errors if e.strip()]
                            ndcg_10 = [float(e.split("|")[0]) for e in ndcg if e.strip()]
                            ndcg_20 = [float(e.split("|")[1]) for e in ndcg if e.strip()]
                            ndcg_50 = [float(e.split("|")[2]) for e in ndcg if e.strip()]
                            edit_dist_10 = [int(e.split("|")[0]) for e in edit_dist if e.strip()]
                            edit_dist_20 = [int(e.split("|")[1]) for e in edit_dist if e.strip()]
                            edit_dist_50 = [int(e.split("|")[2]) for e in edit_dist if e.strip()]
                            
                            if not skip_extra:
                                mae_10 = [float(e.split("|")[0]) for e in mae if e.strip()]
                                mae_20 = [float(e.split("|")[1]) for e in mae if e.strip()]
                                mae_50 = [float(e.split("|")[2]) for e in mae if e.strip()]
                                
                                cpu_predictions = cpu_predictions.split(";")
                                fpga_predictions = fpga_predictions.split(";")
                                
                                cpu_predictions_10 = [[int(x) for x in e.split("|")[:10]] for e in cpu_predictions if e.strip()]
                                cpu_predictions_20 = [[int(x) for x in e.split("|")[:20]] for e in cpu_predictions if e.strip()]
                                cpu_predictions_50 = [[int(x) for x in e.split("|")[:50]] for e in cpu_predictions if e.strip()]
                                fpga_predictions_10 = [[int(x) for x in e.split("|")[:10]] for e in fpga_predictions if e.strip()]
                                fpga_predictions_20 = [[int(x) for x in e.split("|")[:20]] for e in fpga_predictions if e.strip()]
                                fpga_predictions_50 = [[int(x) for x in e.split("|")[:50]] for e in fpga_predictions if e.strip()]
                                
                                prec_10 = [len(set(c).intersection(set(f))) / 10 for (c, f) in zip(cpu_predictions_10, fpga_predictions_10)]
                                prec_20 = [len(set(c).intersection(set(f))) / 20 for (c, f) in zip(cpu_predictions_20, fpga_predictions_20)]
                                prec_50 =[len(set(c).intersection(set(f))) / 50 for (c, f) in zip(cpu_predictions_50, fpga_predictions_50)]
                                
                                kendall_10 = [kendall_tau(c, f) for (c, f) in zip(cpu_predictions_10, fpga_predictions_10)]
                                kendall_20 = [kendall_tau(c, f) for (c, f) in zip(cpu_predictions_20, fpga_predictions_20)]
                                kendall_50 =[kendall_tau(c, f) for (c, f) in zip(cpu_predictions_50, fpga_predictions_50)]
                            else:
                                mae_10 = [0] * n_ppr
                                mae_20 = [0] * n_ppr
                                mae_50 = [0] * n_ppr
                                prec_10 = [0] * n_ppr
                                prec_20 = [0] * n_ppr
                                prec_50 = [0] * n_ppr
                                kendall_10 = [0] * n_ppr
                                kendall_20 = [0] * n_ppr
                                kendall_50 = [0] * n_ppr
                            
                            assert(n_ppr == len(errors_10))
                            assert(n_ppr == len(errors_20))
                            assert(n_ppr == len(errors_50))
                            assert(n_ppr == len(ndcg_10))
                            assert(n_ppr == len(ndcg_20))
                            assert(n_ppr == len(ndcg_50))
                            assert(n_ppr == len(edit_dist_10))
                            assert(n_ppr == len(edit_dist_20))
                            assert(n_ppr == len(edit_dist_50))
                            
                            # Add the result line to the list;
                            for i in range(n_ppr):  
                                new_res_line = [graph_name, int(V), int(E), n_bit, int(n_ppr),
                                                1, int(max_iter), float(exec_time_ms), float(transfer_time_ms),
                                                errors_10[i], errors_20[i], errors_50[i],
                                                ndcg_10[i], ndcg_20[i], ndcg_50[i],
                                                edit_dist_10[i], edit_dist_20[i], edit_dist_50[i],
                                                convergence_error,
                                                mae_10[i], mae_20[i], mae_50[i],
                                                prec_10[i], prec_20[i], prec_50[i],
                                                kendall_10[i], kendall_20[i], kendall_50[i]]
                                result_list += [new_res_line]
                        
    # Create a dataframe;
    result_df = pd.DataFrame(result_list,
                             columns=["graph_name", "V", "E", "n_bit", "n_ppr", "n_iter", "max_iter", 
                                      "exec_time_ms", "transfer_time_ms",
                                      "errors_10", "errors_20", "errors_50", 
                                      "ndcg_10", "ndcg_20", "ndcg_50",
                                      "edit-dist_10", "edit-dist_20", "edit-dist_50",
                                      "convergence_error",
                                      "mae_10", "mae_20", "mae_50",
                                      "prec_10", "prec_20", "prec_50",
                                      "kendall_10", "kendall_20", "kendall_50"])
    
    res_agg = result_df.groupby(["graph_name", "V", "E", "n_bit"], as_index=False).mean()
    return result_df, res_agg


# Round a number to the closest power of 10.
# E.g. 488 -> 100; 560 -> 1000
def fix_edge_number(val):
    exp = int(np.log10(val))
    rem = np.round(val / 10**(exp + 1))
    return 10**(exp + rem)


if __name__ == "__main__":
    
    sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 40 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    res, agg = read_data()
    
    #%%
    
    # Consider only graphs with 100k vertices;
    res = res[res["V"] == 10**5]
    
    # Fix the number of edges;
    # res["E_fixed"] = res["E"].apply(fix_edge_number)
    replace_edge_map = {
        499107: 5 * 10**5,
        1001339: 10**6,
        100000324: 10**8,
        119678: 10**5,
        5005110: 5 * 10**6,
        10002330: 10**7,
        501312: 5 * 10**5,
        857872: 10**6,
        3528924: 5 * 10**6,
        249660: 10**5,
        6853340: 10**7,
        65980874: 10**8,
        844974: 10**6,
        3533124: 5 * 10**6,
        182874: 10**5,
        492796: 5 * 10**5,
        6866454: 10**7
        }
    res["E_fixed"] = res["E"].replace(replace_edge_map, inplace=False)
    
    # Compute sparsity;
    res["sparsity"] = res["E_fixed"] / (res["V"] * res["V"])
   
    
    sparsities = [str(x) for x in sorted(res["sparsity"].unique())]
    
    
    #%%
    
    res2 = res[(res["graph_name"] != "pc") & (res["sparsity"] <= 0.0005) & (res["n_bit"] != "float")]
    res2["sparsity_str"] = res2["sparsity"].astype(str)
    
    plt.rcParams['mathtext.fontset'] = "cm" 
    
    # Setup plot;
    N = 50
    error_metrics = ["Precision\n(higher is better)"] # ["Num. Errors", "Edit Distance", "NDCG\n(higher is better)", "MAE", "Precision\n(higher is better)", "Kendall's " + r"$\mathbf{\tau}$" + "\n(higher is better)"]
    error_metrics_raw = [f"prec_{N}"] # [f"errors_{N}", f"edit-dist_{N}", f"ndcg_{N}", f"mae_{N}", f"prec_{N}", f"kendall_{N}"]
    error_max = [1] # [N, N, 1, 0.12, 1, 1]
    error_min = [0.3] # [0, 0, 0.6, 0, 0.3, 0.0]
    error_sizes = [10, 20, 50]
    
    num_iters = sorted(res["max_iter"].unique())
    num_rows = len(error_metrics)
    num_col = len(num_iters) 

    # N Rows;
    # fig = plt.figure(figsize=(2.0 * num_col, 3.1 * num_rows))
    # gs = gridspec.GridSpec(num_rows, num_col)
    # plt.subplots_adjust(top=0.85,
    #                 bottom=0.12,
    #                 left=0.12,
    #                 right=0.95,
    #                 hspace=0.8,
                    # wspace=0.6)
    # 1 Row;
    fig = plt.figure(figsize=(2.0 * num_col, 2.6 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_col)
    plt.subplots_adjust(top=0.6,
                    bottom=0.1,
                    left=0.2,
                    right=0.95,
                    hspace=0.8,
                    wspace=0.6)

    bitwidths_num = len(res2["n_bit"].unique())
    palette_list = [bb5, bb4, bb3, bb2, r1]
    palette = palette_list[:bitwidths_num]
    markers_list = ["o", "X", "^", "D", "D"]
    markers = markers_list[:bitwidths_num]
    
    for i, num_iter in enumerate(num_iters):
        # One row per error metric;
        for j, e in enumerate(error_metrics_raw):
            
            curr_data = res2[res2["max_iter"] == num_iter]
            curr_data = curr_data.sort_values(by=["sparsity", "n_bit"])
            
            ax = fig.add_subplot(gs[j, i])
            ax = sns.lineplot(x="sparsity_str", y=e, hue="n_bit", data=curr_data, palette=palette, ax=ax,
                  err_style="bars", linewidth=3, legend=None, zorder=2, ci=None, estimator="mean", sort=False)
            data_averaged = curr_data.groupby(["sparsity_str", "n_bit"], as_index=False).mean()
            ax = sns.scatterplot(x="sparsity_str", y=e, hue="n_bit", data=data_averaged, palette=palette, ax=ax, edgecolor="#0f0f0f",
                  size_norm=20, legend=False, zorder=3, ci=None, markers=markers, style="n_bit", linewidth=0.05, clip_on=False)
            ax.set_ylim([error_min[j], error_max[j]])
            # ax.set_xlim([min(curr_data["n_bit"]), max(curr_data["n_bit"])])
            
            if j == len(error_metrics_raw) - 1:
                # ax.set_xlabel(f"{num_iter} Iterations")
                ax.annotate(f"{num_iter} Iterations",
                        xy=(0.5, 1.2), xycoords="axes fraction", fontsize=12, # textcoords="offset points",
                        horizontalalignment="center", verticalalignment="center")
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            if i == 0:
                ax.set_ylabel(f"{error_metrics[j]}", fontsize=14)
            # ax.annotate(f"{error_metrics[j]}",
            #             xy=(-0.1, 0.5), xycoords="axes fraction", fontsize=12, textcoords="offset points", xytext=(0, 20),
            #             horizontalalignment="center", verticalalignment="center")
            # ax.set_title(f"{graph_names[i]}", fontsize=14, loc="center", xytext=(-40, 25))
            
            # Set the number of ticks on the y axis;
            # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            
            # Only for 1 row with precision;
            ax.set_yticks([(i + 1) / 5 for i in range(1, 5)])
            ax.set_yticklabels(labels=[f"{int(l * 100)}%" for l in ax.get_yticks()], ha="right", fontsize=12)
            
            ax.set_xticklabels(labels=[r"$\mathdefault{" +
                                       ((r"10^{" + str(int(np.log10(float(x))))) if (1 * float(x)) / 10**int(np.log10(float(x))) >= 1
                                        else (r"5\!·\!10^{" + str(int(np.log10(float(x))) - 1))) + r"}}$"
                                        for x in sparsities])
            if j == 2:
                ax.set_yticklabels(labels=[f"{int(l * 100)}%" for l in ax.get_yticks()], ha="right", fontsize=12)
                
            # Turn off tick lines;
            ax.xaxis.grid(False)  
            sns.despine(ax=ax)              
            ax.tick_params(axis="y", labelcolor="black", labelsize=12, pad=6)
            ax.tick_params(axis="x", labelcolor="black", labelsize=9, pad=4)
            
    # plt.annotate("Fixed-point Bitwidth", fontsize=16, xy=(0.5, 0.015), xycoords="figure fraction", ha="center")
           
    fig.suptitle(f"Top-50 Precision w.r.t sparsity,\nbit-width, and number of iterations",
                 fontsize=15, ha="left", x=0.05)
    
    # Legend;    
    bit_widths = ["20", "22", "24", "26", "Float"]
    custom_lines = [
        Line2D([], [], color="white", marker=markers_list[0],
               markersize=10, label=bit_widths[0], markerfacecolor=palette_list[0], markeredgecolor="#2f2f2f"),
        Line2D([], [], color="white", marker=markers_list[1],
               markersize=10, label=bit_widths[1], markerfacecolor=palette_list[1], markeredgecolor="#2f2f2f"),
        Line2D([], [], color="white", marker=markers_list[2],
               markersize=10, label=bit_widths[2], markerfacecolor=palette_list[2], markeredgecolor="#2f2f2f"),
        Line2D([], [], color="white", marker=markers_list[3],
               markersize=10, label=bit_widths[3], markerfacecolor=palette_list[3], markeredgecolor="#2f2f2f"),
        Line2D([], [], color="white", marker=markers_list[4],
               markersize=10, label=bit_widths[4], markerfacecolor=palette_list[4], markeredgecolor="#2f2f2f"),
        ][:bitwidths_num]
    
    leg = fig.legend(custom_lines, bit_widths[:bitwidths_num],
                             bbox_to_anchor=(0.98, 1), fontsize=12, ncol=2)
    leg.set_title(None)
    leg._legend_box.align = "left"
            
    plt.savefig(f"../../../../data/plots/sparsity_{DATE}_agg.pdf")           
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    