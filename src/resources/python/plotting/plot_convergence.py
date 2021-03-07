#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:59:57 2020

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


# Define some colors;
c1 = "#b1494a"
c2 = "#256482"
c3 = "#2f9c5a"
c4 = "#28464f"
c5 = "#FFEA70"
c6 = "#FFEF85"

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

bb0 = "#FFA685"
bb1 = "#75B0A2"
bb2 = b3
bb3 = b7
bb4 = "#7ED7B8"
bb5 = "#7BD490"

MAIN_RESULT_FOLDER = "../../../../data/results/raw_results/2020_05_142"
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
                        result = f.readlines()[1]
                        
                        # Parse the file name;
                        try:
                            _, _, n_bit, n_ppr, max_iter, _, _, _, _, _, n_iter = res_path[:-4].split("-")
                        except ValueError:
                            _, _, n_bit, n_ppr, max_iter, _, _, _, _, n_iter = res_path[:-4].split("-")
                        max_iter = max_iter.replace("it", "")
                        # Parse the result line;
                        _, _, V, E, exec_time_ms, transfer_time_ms, errors, ndcg, edit_dist, convergence_error, mae, fpga_predictions, cpu_predictions = result.split(",")
                       
                        n_ppr = int(n_ppr)
                        
                        if n_ppr == 8:
                        
                            # Obtain the single error metrics;
                            errors = errors.split(";")
                            ndcg = ndcg.split(";")
                            edit_dist = edit_dist.split(";")
                            convergence_error = convergence_error.split(";")
                            
                            errors_10 = [int(e.split("|")[0]) for e in errors if e.strip()]
                            errors_20 = [int(e.split("|")[1]) for e in errors if e.strip()]
                            errors_50 = [int(e.split("|")[2]) for e in errors if e.strip()]
                            ndcg_10 = [float(e.split("|")[0]) for e in ndcg if e.strip()]
                            ndcg_20 = [float(e.split("|")[1]) for e in ndcg if e.strip()]
                            ndcg_50 = [float(e.split("|")[2]) for e in ndcg if e.strip()]
                            edit_dist_10 = [int(e.split("|")[0]) for e in edit_dist if e.strip()]
                            edit_dist_20 = [int(e.split("|")[1]) for e in edit_dist if e.strip()]
                            edit_dist_50 = [int(e.split("|")[2]) for e in edit_dist if e.strip()]
                                                        
                            convergence = [[float(x) for x in e.split("|") if x.strip()] for e in convergence_error if e.strip()]
                            
                            assert(n_ppr == len(errors_10))
                            assert(n_ppr == len(errors_20))
                            assert(n_ppr == len(errors_50))
                            assert(n_ppr == len(ndcg_10))
                            assert(n_ppr == len(ndcg_20))
                            assert(n_ppr == len(ndcg_50))
                            assert(n_ppr == len(edit_dist_10))
                            assert(n_ppr == len(edit_dist_20))
                            assert(n_ppr == len(edit_dist_50))
                            assert(n_ppr == len(convergence))
                            
                            # Add the result line to the list;
                            for i in range(n_ppr):  
                                new_res_line = [graph_name, int(V), int(E), n_bit, int(n_ppr),
                                                1, int(max_iter), float(exec_time_ms), float(transfer_time_ms),
                                                errors_10[i], errors_20[i], errors_50[i],
                                                ndcg_10[i], ndcg_20[i], ndcg_50[i],
                                                edit_dist_10[i], edit_dist_20[i], edit_dist_50[i], convergence[i]]
                                result_list += [new_res_line]
                        
    # Create a dataframe;
    result_df = pd.DataFrame(result_list,
                             columns=["graph_name", "V", "E", "n_bit", "n_ppr", "n_iter", "max_iter", 
                                      "exec_time_ms", "transfer_time_ms",
                                      "errors_10", "errors_20", "errors_50", 
                                      "ndcg_10", "ndcg_20", "ndcg_50",
                                      "edit-dist_10", "edit-dist_20", "edit-dist_50", "convergence"])
    
    res_agg = result_df.groupby(["graph_name", "V", "E", "n_bit"], as_index=False).mean()
    return result_df, res_agg


if __name__ == "__main__":
    
    sns.set_style("whitegrid", {"xtick.bottom": True, "ytick.left": True, "xtick.color": ".8", "ytick.color": ".8"})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 0 
    plt.rcParams['axes.labelpad'] = 10 
    plt.rcParams['axes.titlesize'] = 22 
    plt.rcParams['axes.labelsize'] = 14 
    
    res_tmp, agg = read_data()
    
    res = res_tmp[~res_tmp["n_bit"].isin(["22", "24"])]
    
    # Get the number of iterations to have error below a threashold;
    threshold = 10e-6
    errors_df = []
    res_tmp["graph_id"] = res_tmp["graph_name"] + "_" + res_tmp["V"].astype(str)
    
    for i, g in res_tmp.groupby(["graph_id", "n_bit"]):
        print(i)
        convergence = []
        # For each row, see how long it takes for the error to go below the threshold;
        for j, errors in enumerate(g["convergence"]):
            
            # Skip vertices where the error starts at 0;
            if errors[0] < 1e-8 or errors[1] < 10e-8:
                continue
            
            conv_found = False
            for iteration, e in enumerate(np.sqrt(errors)): 
                if e <= threshold:
                    convergence += [iteration]
                    iteration = True
                    break
            # if not conv_found:
            #     convergence += [len(errors)]
        errors_df += [[i[0], i[1], np.mean(convergence)]]
    pd.DataFrame(errors_df).to_csv("../../../../data/results/convergence.csv", index=False)    
    #%%
    
    # Consider only graphs with 100k vertices;
    # res = res[(res["V"] == 10**5) | res["graph_name"].isin(["amazon", "twitter"])]
    # res = res[~res["V"].isin([10**5, 2 * 10**5])]
    
    # Setup plot;
    # graph_names = ["$\mathdefault{G_{n,p}}$", "Watts–Strogatz", "Holme and Kim", "Amazon", "Twitter"]
    graph_names = [f"|E|=~{get_exp_label(10**6)}", "Amazon", f"|E|=~{get_exp_label(2 * 10**6)}", "Twitter"]
   
    num_rows = 1
    num_col = len(res["V"].unique()) // num_rows
    fig = plt.figure(figsize=(1.2 * num_col, 2.2 * num_rows))
    gs = gridspec.GridSpec(num_rows, num_col)
    plt.subplots_adjust(top=0.70,
                    bottom=0.2,
                    left=0.13,
                    right=0.98,
                    hspace=0.6,
                    wspace=0.1)

    palette = [b8, b2, r1] # ["#87B7FF", bb5, bb0]
    markers = ["D"] * len(palette)
    
    sizes = [100000, 81306, 200000, 128000]
    
    MIN_ITER = 1
    MAX_ITER = 20
    max_iter_plot = [15, 15, 17, 19]
        
    # One row per graph;
    # for i, group in enumerate(res.groupby(["V"])):
    for i, size in enumerate(sizes):
        data = res[res["V"] == size]
        # data = group[1]
        # print("graph=", group[0])
        # Build a new dataframe with the convergence error;
        df_rows = []
        for j, row in data.iterrows():
            error = row["convergence"]
            
            # Skip vertices where the error starts at 0;
            if error[0] < 1e-8 or error[1] < 10e-8:
                continue
            
            for q, e in enumerate(error):

                # keep only iterations in the the selected range;
                if q > MAX_ITER:
                    break
                elif q >= MIN_ITER:
                    df_rows += [[q, e, row["n_bit"]]]
                
        df_temp = pd.DataFrame(df_rows, columns=["Iteration", "Error", "Bit-Width"])
        df_temp = df_temp.groupby(["Iteration", "Bit-Width"], as_index=False).median()
        df_temp["Error"] = np.log(np.sqrt(df_temp["Error"]))
                    
        ax = fig.add_subplot(gs[i // num_col, i % num_col])
        ax = sns.lineplot(x="Iteration", y="Error", hue="Bit-Width", data=df_temp, palette=palette, ax=ax,
              err_style="bars", linewidth=2, alpha=1, legend=False, zorder=2, ci=None)
        # ax = sns.scatterplot(x="Iteration", y="Error", hue="Bit-Width", data=df_temp, palette=palette, ax=ax, edgecolor="#0f0f0f",
        #       size_norm=30, legend=False, zorder=3, ci=None, markers=markers, style="Bit-Width", linewidth=0.05)

        ax.set_xlim([MIN_ITER, max_iter_plot[i]])
        ax.set_xlabel(None)
        ax.set_yticks([-i for i in range(1, 8)])
        ax.set_ylim([-7, -1])
        if i == 0:
            ax.set_yticklabels(labels=[r"$\mathdefault{" + r"{10}^{" + str(l) + r"}}$" for l in ax.get_yticks()])
        else:
            ax.set_yticklabels(labels=[], ha="right")
        
        if i == 0:
            ax.set_ylabel("Convergence Error", fontsize=10) 
        else:
            ax.set_ylabel(None) 

        # if i == 0:
        #      # Graph name;
        ax.annotate(f"{graph_names[i]}",
                    xy=(0.5, 1), xycoords="axes fraction", fontsize=10, textcoords="offset points", xytext=(0, 5),
                    horizontalalignment="center", verticalalignment="center")
            
        # Turn off tick lines;
        ax.xaxis.grid(False)  
        sns.despine(ax=ax)              
        ax.tick_params(labelcolor="black", labelsize=9, pad=4)
            
              
            
    plt.annotate("Iteration Number", fontsize=12, xy=(0.5, 0.015), xycoords="figure fraction", ha="center")
    # plt.annotate("Convergence Error", xy=(0.02, 0.3), fontsize=12, ha="center", va="center", rotation=90, xycoords="figure fraction")

    fig.suptitle(f"PPR Convergence Error\nw.r.t fixed-point bitwidth",
                  fontsize=12, ha="left", x=0.05)
    
    # Legend;
    custom_lines = [
        Patch(facecolor=palette[0], edgecolor="#2f2f2f", label="20"),
        # Patch(facecolor=palette[1], edgecolor="#2f2f2f", label="22"),
        # Patch(facecolor=palette[2], edgecolor="#2f2f2f", label="24"),
        Patch(facecolor=palette[1], edgecolor="#2f2f2f", label="26"),
        Patch(facecolor=palette[2], edgecolor="#2f2f2f", label="Float"),
        ]
    
    leg = fig.legend(custom_lines, ["20", "26", "Float"],
                              bbox_to_anchor=(0.97, 1), fontsize=10, ncol=3, handletextpad=0.3, columnspacing=0.6)
    leg.set_title(None)
    leg._legend_box.align = "left"
            
    plt.savefig(f"../../../../data/plots/convergence_{DATE}.pdf")
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    