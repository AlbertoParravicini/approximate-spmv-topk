# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:47:25 2020

@author: albyr
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

DATE = "2021_03_07"

SCATTER_SIZE = 14

if __name__ == "__main__":
    
    sns.set_style("white", {"ytick.left": True, "xticks.bottom": True, "grid.linewidth": 0.5, "axes.linewidth": 0.1})
    plt.rcParams["font.family"] = ["Latin Modern Roman Demi"]
    plt.rcParams['axes.titlepad'] = 40 
    plt.rcParams['axes.labelpad'] = 2
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.major.pad'] = 5

    #%%
    # single_core_time_ms = 32
    # n_cores_single_core = 8  # single core performance is derived from 8 cores;
    # multicore_times = [250, 32, 17, 10]
    # multicore_cores = [1, 8, 16, 32]
    # frequency = 250 * 10**6
    # max_cores = 32
    
    # bandwidth_single_core = 13.2 * 10**9
    # bandwidth_total = bandwidth_single_core * 32
    # bandwidths = [x * bandwidth_single_core for x in multicore_cores]
    
    markers = ["o", "X", "D", "P"]
    palette = [COLORS["peach1"], "#A5E6C6", COLORS["bb4"], COLORS["bb5"]]           
    
    # nnz = 249960781
    # op_intensity = 0.07812496249
    # op_intensity_multicore = 0.07812496249
    
    # best_time_single_core = 1000 * ((nnz / n_cores_single_core) / 5) / frequency  # Time to process nnz / 32
    # best_time_multi_core = 1000 * ((nnz / max_cores) / 5) / frequency  # Time to process nnz / 32
    # max_performance_single_core = 10**3 * (nnz / n_cores_single_core) / best_time_single_core  # NNZ per second
    # max_performance_multicore = 10**3 * nnz / best_time_multi_core
    
    # performance_single_core =  10**3 * (nnz / n_cores_single_core) / single_core_time_ms
    
    # plt.rcParams['mathtext.fontset'] = "cm" 
    
    # num_col = 2
    # num_row = 1
    # fig = plt.figure(figsize=(2.2 * num_col, 1.8 * num_row))
    # gs = gridspec.GridSpec(num_row, num_col)
    # plt.subplots_adjust(top=0.72,
    #                 bottom=0.2,
    #                 left=0.15,
    #                 right=0.8,
    #                 hspace=0.5,
    #                 wspace=0.5)
    
    # # Single core roofline;
    # ax = fig.add_subplot(gs[0, 0])
    # x_lim = 0.16
    # ax.set_xlim((0, x_lim))
    # ax.set_ylim((0, max_performance_single_core * 1.1))
    
    # line_cross = max_performance_single_core / bandwidth_single_core
    # line_cross_op_intensity = bandwidth_single_core * op_intensity
    # plt.plot([line_cross, x_lim], [max_performance_single_core, max_performance_single_core],
    #          color="#2f2f2f", linewidth=1, zorder=1)
    # plt.plot([0, line_cross], [0, max_performance_single_core],
    #          color="#2f2f2f", linewidth=1, zorder=1)
    # plt.plot([op_intensity, op_intensity], [0, line_cross_op_intensity],
    #          color="#757575", linewidth=0.5, linestyle="--", zorder=0)
    # plt.plot([0, x_lim], [max_performance_single_core, max_performance_single_core],
    #          color="#757575", linewidth=0.5, linestyle="--", zorder=0)
    
    # # Add performance point;
    # plt.scatter(op_intensity, performance_single_core, color=COLORS["peach1"], edgecolors="#2f2f2f", s=10, zorder=2, linewidth=0.3)
             
    # ax.xaxis.set_major_locator(plt.LinearLocator(5))
    # ax.yaxis.set_major_locator(plt.LinearLocator(5))
    # ax.xaxis.grid(False)
    # for tic in ax.xaxis.get_major_ticks():
    #     tic.tick1line.set_visible(True) 
    #     tic.tick2line.set_visible(False) 
    # ax.set_yticklabels(labels=[ r"$\mathdefault{" + f"{(l / 10**9):.2f}" + r"\!·\!{10}^9" + r"}$" for l in ax.get_yticks()], ha="right", fontsize=6)
    # ax.tick_params(labelcolor="black", labelsize=6, pad=4)  
    
    # ax.set_xlabel("Operational Intensity [NNZ/B]", fontsize=8)
    # ax.set_ylabel("Performance [NNZ/s]", fontsize=8)
    
    # plt.annotate("Single-core", fontsize=10, xy=(0.0, 1.1), xycoords="axes fraction", ha="left")
           
    # fig.suptitle("Roofline model for our Top-K SpMV architecture",
    #              fontsize=12, ha="left", x=0.03)
    
    # #%%
    # ########################
    # # Multicore
    # ########################
    
    # ax = fig.add_subplot(gs[0, 1])
    # plt.yscale("log")
    # plt.xscale("log")
    # x_lim = 1
    # ax.set_xlim((0.01, 1))
    # ax.set_ylim((0.1 * 10**9, max_performance_multicore * 1.1))
    
    # line_cross = max_performance_multicore / bandwidth_total
    # line_cross_op_intensity = bandwidth_total * op_intensity_multicore
    # plt.plot([line_cross, x_lim], [max_performance_multicore, max_performance_multicore],
    #          color="#2f2f2f", linewidth=1, zorder=2)
    
    # plt.plot([op_intensity_multicore, op_intensity_multicore], [0, line_cross_op_intensity],
    #          color="#757575", linewidth=0.5, linestyle="--", zorder=0)
    # plt.plot([0, x_lim], [max_performance_multicore, max_performance_multicore],
    #          color="#757575", linewidth=0.5, linestyle="--", zorder=0)
    
    # for i, c in enumerate(multicore_cores):
    #     perf = 10**3 * nnz / multicore_times[i]
    #     line_cross = max_performance_multicore / bandwidths[i]
    #     plt.plot([0, line_cross], [0, max_performance_multicore],
    #          color=palette[i], linewidth=1, zorder=1)
    
    # # Add performance point;
    # for i, t in enumerate(multicore_times):
    #     perf =  10**3 * nnz / t
    #     plt.scatter(op_intensity_multicore, perf, color=palette[i], edgecolors="#2f2f2f", marker=markers[i], s=10, zorder=2, linewidth=0.3)
             
    # ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    # ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    # ax.tick_params(axis="x", direction="out", which="both", bottom=True, top=False)
        
    # ax.xaxis.grid(False)
    # for tic in ax.xaxis.get_major_ticks():
    #     tic.tick1line.set_visible(True) 
    #     tic.tick2line.set_visible(False) 
    # for tic in ax.xaxis.get_minor_ticks():
    #     tic.tick1line.set_visible(True) 
    #     tic.tick2line.set_visible(False) 
    # ax.set_yticklabels(labels=[ r"$\mathdefault{" + f"{(l / 10**9):.2f}" + r"\!·\!{10}^9" + r"}$" for l in ax.get_yticks()], ha="right", fontsize=6)
    # ax.tick_params(labelcolor="black", labelsize=6, pad=4)  
    # ax.minorticks_on()
    # ax.set_xlabel("Operational Intensity [NNZ/B]", fontsize=8)
    # ax.set_ylabel(None)
    
    # # Add custom legend;
    # labels = ["1 core", "8 cores", "16 cores", "32 cores"]
    # custom_lines = [Line2D([], [], color="white", marker=markers[::-1][j],
    #            markersize=8, label=labels[::-1][j], markerfacecolor=palette[::-1][j], markeredgecolor="#2f2f2f") for j in range(len(labels))]
    # leg = fig.legend(custom_lines, labels[::-1], bbox_to_anchor=(1, 0.65), fontsize=8, ncol=1, handletextpad=0.5, columnspacing=0.4)
    # leg.set_title(None)
    # leg._legend_box.align = "left"
    # leg.get_frame().set_facecolor('white') 
    
    # plt.annotate("Multi-core architectures", fontsize=10, xy=(0.0, 1.1), xycoords="axes fraction", ha="left")
    
    # plt.savefig(f"../../../../data/plots/roofline_{DATE}.pdf")
    
    ########################
    # CPU-GPU
    ########################
    
    # CPU
    nnz_per_sec_cpu = 0.4 * 10**9
    max_bandwidth_cpu = 140.8 * 10**9
    peak_performance_cpu = 666 * 10**9
    operational_intensity_cpu = 1 / 12
    
    # GPU F32
    max_bandwidth_gpu = 549 * 10**9
    
    nnz_per_sec_gpu_f32 = 27 * 10**9
    peak_performance_gpu_f32 = 3100 * 10**9
    operational_intensity_gpu_f32 = 1 / 12
    
    nnz_per_sec_gpu_f16 = 30 * 10**9
    peak_performance_gpu_f16 = 6230 * 10**9
    operational_intensity_gpu_f16 = 1 / 8
    
    # FPGA
    bandwidth_single_core_fpga = 13.2 * 10**9
    bandwidth_total_fpga = bandwidth_single_core_fpga * 32
    max_cores_fpga = 64
    fpga_clock = 225 * 10**6
    packet_size_fpga = [11, 15]
    num_cores_fpga = [32, 32]
    num_bits_fpga = [32, 20]
    peak_performance_fpga = [p * fpga_clock * max_cores_fpga for p in packet_size_fpga]
    peak_bandwidth_fpga = [bandwidth_single_core_fpga * c for c in num_cores_fpga]
    operational_intensity_fpga = [p / (512 / 8) for p in packet_size_fpga]
    exec_times_fpga = [5, 3.5]  # Exec time in ms of 32b-24cores-219mhz and 21b-32cores-177mhz
    nnz_fpga = 20 * 10**7
    nnz_per_sec_fpga = [nnz_fpga / (p / 1000) for p in exec_times_fpga]
    
    # Plotting
    CPU_COLOR = "#6d6d6d"

    markers = ["o", "X", "D", "P"]
    palette = [COLORS["peach1"], "#A5E6C6", COLORS["bb4"], COLORS["bb5"]]  
    palette_y = ["#FCF49A", "#FCF6B1"]
    palette = [CPU_COLOR, COLORS["peach1"], "#E6D955", "#A5E6C6", COLORS["bb4"], COLORS["bb5"]]
    markers = ["o", "X", "D", "o", "P"]
    plt.rcParams['mathtext.fontset'] = "cm" 
    
    num_col = 2
    num_row = 1
    fig = plt.figure(figsize=(2.2 * num_col, 1.83 * num_row))
    gs = gridspec.GridSpec(num_row, num_col)
    plt.subplots_adjust(top=0.95,
                    bottom=0.265,
                    left=0.1,
                    right=0.89,
                    hspace=0,
                    wspace=0.5)
    
    ax = fig.add_subplot(gs[0, 1])
    
    # Change axis line width;
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
        
    plt.yscale("log")
    plt.xscale("log")
    x_lim = 10
    ax.set_xlim((0.02, x_lim))
    ax.set_ylim((0.1 * 10**9, np.max([peak_performance_cpu, peak_performance_gpu_f32, peak_performance_gpu_f16]) * 1.4))
    
    # CPU Plot;
    line_cross_cpu = peak_performance_cpu / max_bandwidth_cpu
    line_cross_op_intensity = max_bandwidth_cpu * operational_intensity_cpu
    plt.plot([line_cross_cpu, x_lim], [peak_performance_cpu, peak_performance_cpu],
              color=CPU_COLOR, linewidth=1, zorder=2)
    
    plt.plot([operational_intensity_cpu, operational_intensity_cpu], [0, line_cross_op_intensity],
              color="#757575", linewidth=0.5, linestyle="--", zorder=0)
    plt.plot([0, x_lim], [peak_performance_cpu, peak_performance_cpu],
              color="#757575", linewidth=0.5, linestyle="--", zorder=0)
    
    perf = nnz_per_sec_cpu
    line_cross = peak_performance_cpu / max_bandwidth_cpu
    plt.plot([0, line_cross], [0, peak_performance_cpu], color=CPU_COLOR, linewidth=1, zorder=1)
    plt.scatter(operational_intensity_cpu, perf, color=CPU_COLOR, edgecolors="#2f2f2f", marker=markers[0], s=SCATTER_SIZE, zorder=3, linewidth=0.3)
    
    # GPU Plot;
        
    line_cross_gpu_f16 = peak_performance_gpu_f16 / max_bandwidth_gpu
    line_cross_gpu_f32 = peak_performance_gpu_f32 / max_bandwidth_gpu
    line_cross_op_intensity_f32 = max_bandwidth_gpu * operational_intensity_gpu_f32
    line_cross_op_intensity_f16 = max_bandwidth_gpu * operational_intensity_gpu_f16
    plt.plot([line_cross_gpu_f32, x_lim], [peak_performance_gpu_f32, peak_performance_gpu_f32],
          color=palette[1], linewidth=1, zorder=2)
    plt.plot([line_cross_gpu_f16, x_lim], [peak_performance_gpu_f16, peak_performance_gpu_f16],
              color=palette[2], linewidth=1, zorder=2)
    
    plt.plot([operational_intensity_gpu_f32, operational_intensity_gpu_f32], [0, line_cross_op_intensity_f32],
              color="#757575", linewidth=0.5, linestyle="--", zorder=0)
    plt.plot([0, x_lim], [peak_performance_gpu_f32, peak_performance_gpu_f32],
              color="#757575", linewidth=0.5, linestyle="--", zorder=0)
    
    plt.plot([operational_intensity_gpu_f16, operational_intensity_gpu_f16], [0, line_cross_op_intensity_f16],
              color="#757575", linewidth=0.5, linestyle="--", zorder=0)
    plt.plot([0, x_lim], [peak_performance_gpu_f16, peak_performance_gpu_f16],
              color="#757575", linewidth=0.5, linestyle="--", zorder=0)
    
    perf = nnz_per_sec_gpu_f16
    line_cross = peak_performance_gpu_f16 / max_bandwidth_gpu
    plt.plot([0, line_cross], [0, peak_performance_gpu_f16], color=palette[2], linewidth=1, zorder=1)
    plt.scatter(operational_intensity_gpu_f16, perf, color=palette[2], edgecolors="#2f2f2f", marker=markers[2], s=SCATTER_SIZE, zorder=3, linewidth=0.3)
    
    perf = nnz_per_sec_gpu_f32
    line_cross = peak_performance_gpu_f32 / max_bandwidth_gpu
    plt.plot([0, line_cross], [0, peak_performance_gpu_f32], color=palette[1], linewidth=1, zorder=1)
    plt.scatter(operational_intensity_gpu_f32, perf, color=palette[1], edgecolors="#2f2f2f", marker=markers[1], s=SCATTER_SIZE, zorder=3, linewidth=0.3)
    
    
    # FPGA plot;
    
    for i in range(len(peak_performance_fpga))[::-1]: 
    
        line_cross = peak_performance_fpga[i] / peak_bandwidth_fpga[i]
        line_cross_op_intensity = peak_bandwidth_fpga[i] * operational_intensity_fpga[i]
        plt.plot([line_cross, x_lim], [peak_performance_fpga[i], peak_performance_fpga[i]],
              color=palette[i + 3], linewidth=1, zorder=2)
        
        plt.plot([operational_intensity_fpga[i], operational_intensity_fpga[i]], [0, line_cross_op_intensity],
                  color="#757575", linewidth=0.5, linestyle="--", zorder=0)
        plt.plot([0, x_lim], [peak_performance_fpga[i], peak_performance_fpga[i]],
                  color="#757575", linewidth=0.5, linestyle="--", zorder=0)
        
        perf = nnz_per_sec_fpga[i]
        line_cross = peak_performance_fpga[i] / peak_bandwidth_fpga[i]
        plt.plot([0, line_cross], [0, peak_performance_fpga[i]], color=palette[i + 3], linewidth=1, zorder=2)
        plt.scatter(operational_intensity_fpga[i], perf, color=palette[i + 3], edgecolors="#2f2f2f", marker=markers[i + 3], s=SCATTER_SIZE, zorder=3, linewidth=0.3)
            
    # Other;
    
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    ax.tick_params(axis="x", direction="out", which="both", bottom=True, top=False)
    
    ax.xaxis.grid(False)
    ax.yaxis.grid(linewidth=0.5)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(True) 
        tic.tick2line.set_visible(False) 
    for tic in ax.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(True) 
        tic.tick2line.set_visible(False) 
    ax.set_yticklabels(labels=[get_exp_label(l) for l in ax.get_yticks()], ha="right", fontsize=6)
    ax.tick_params(labelcolor="black", labelsize=6, pad=1)  
    ax.minorticks_on()
    ax.set_xlabel("Operational Intensity [NNZ/B]", fontsize=7)
    ax.set_ylabel(None)
         
    # fig.suptitle("Roofline model for our Top-K SpMV architecture",
                  # fontsize=11, ha="left", x=0.005)
    
    # plt.annotate("FPGA design vs. CPU and GPU", fontsize=8, xy=(0.0, 1.1), xycoords="axes fraction", ha="left")
    ax.annotate("(b)", fontsize=10, xy=(0.5, -0.34), ha="center", xycoords="axes fraction" )
    
    # Add custom legend;
    labels = ["CPU Top-K SpMV", "GPU SpMV, F32", "GPU SpMV, F16"] + [f"FPGA, {num_cores_fpga[i]}C {num_bits_fpga[i]}b" for i in range(len(num_bits_fpga))]
    custom_lines = [Line2D([], [], color="white", marker=markers[j], markeredgewidth=0.5, 
                   markersize=5, label=labels[j], markerfacecolor=palette[j], markeredgecolor="#2f2f2f") for j in range(len(labels))]
    leg = ax.legend(custom_lines, labels, bbox_to_anchor=(0.95, 0), fontsize=6, ncol=1, loc="lower center", handletextpad=0.3, columnspacing=0.4)
    leg.set_title(None)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white') 
    
    ########################
    # Multi-FPGA
    ########################
    
    packet_sizes = [5, 15]
    
    bandwidth_single_core_fpga = 13.2 * 10**9
    bandwidth_total_fpga = bandwidth_single_core_fpga * 32
    max_cores_fpga = 64
    fpga_clock = 225 * 10**6
    packet_size_fpga = [packet_sizes[1]] * 4
    num_cores_fpga = [1, 8, 16, 32]
    num_bits_fpga = [20, 20, 20, 20]
    peak_performance_fpga = [p * fpga_clock * max_cores_fpga for p in packet_size_fpga]
    peak_bandwidth_fpga = [bandwidth_single_core_fpga * c for c in num_cores_fpga]
    operational_intensity_fpga = [p / (512 / 8) for p in packet_size_fpga]
    exec_times_fpga = [27 * 4, 11.2, 5.6, 2.8]  # Exec time in ms of 20b-32cores-199mhz, for 4, 8, 16, 32 cores; 4 is scaled to 1 core;
    nnz_fpga = 20 * 10**7
    nnz_per_sec_fpga = [nnz_fpga / (p / 1000) for p in exec_times_fpga]
    
    old_nnz = 249960781
    old_multicore_times = [250, 32, 17, 10]
    old_multicore_cores = [1, 8, 16, 32]
    old_operational_intensity_fpga = [packet_sizes[0] / (512 / 8) for p in old_multicore_times]
    old_nnz_per_sec_fpga = [old_nnz / (t / 1000) for t in old_multicore_times]
    old_peak_performance_fpga = [5 * fpga_clock * max_cores_fpga for p in old_multicore_times]
    
    # Plotting
    
    markers = ["o", "X", "D", "P"]
    palette = [COLORS["peach1"], "#A5E6C6", COLORS["bb4"], COLORS["bb5"]]           
    
    ax = fig.add_subplot(gs[0, 0])
    
    # Change axis line width;
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)
    
    plt.yscale("log")
    plt.xscale("log")
    x_lim = 1
    ax.set_xlim((0.01, x_lim))
    ax.set_ylim((0.1 * 10**9, np.max(peak_performance_fpga) * 1.3))
    
    # New results;
    for i in range(len(peak_performance_fpga)): 
    
        line_cross = peak_performance_fpga[i] / peak_bandwidth_fpga[i]
        line_cross_op_intensity = peak_bandwidth_fpga[i] * operational_intensity_fpga[i]
        plt.plot([line_cross, x_lim], [peak_performance_fpga[i], peak_performance_fpga[i]],
              color="#2f2f2f", linewidth=1, zorder=3)
        
        plt.plot([operational_intensity_fpga[i], operational_intensity_fpga[i]], [0, line_cross_op_intensity],
                  color="#757575", linewidth=0.5, linestyle="--", zorder=0)
        plt.plot([0, x_lim], [peak_performance_fpga[i], peak_performance_fpga[i]],
                  color="#757575", linewidth=0.5, linestyle="--", zorder=0)
        
        perf = nnz_per_sec_fpga[i]
        line_cross = peak_performance_fpga[i] / peak_bandwidth_fpga[i]
        plt.plot([0, line_cross], [0, peak_performance_fpga[i]], color=palette[i], linewidth=1, zorder=2)
        plt.scatter(operational_intensity_fpga[i], perf, color=palette[i], edgecolors="#2f2f2f", marker=markers[i], s=SCATTER_SIZE, zorder=4, linewidth=0.3)
  
    # Old results;
    for i in range(len(old_peak_performance_fpga)):
        line_cross = old_peak_performance_fpga[i] / peak_bandwidth_fpga[i]
        line_cross_op_intensity = peak_bandwidth_fpga[i] * old_operational_intensity_fpga[i]
        # plt.plot([line_cross, x_lim], [old_peak_performance_fpga[i], old_peak_performance_fpga[i]],
        #       color="#2f2f2f", linewidth=1, zorder=1)
        
        plt.plot([old_operational_intensity_fpga[i], old_operational_intensity_fpga[i]], [0, line_cross_op_intensity],
                  color="#757575", linewidth=0.5, linestyle="--", zorder=0)
        # plt.plot([0, x_lim], [old_peak_performance_fpga[i], old_peak_performance_fpga[i]],
        #           color="#757575", linewidth=0.5, linestyle="--", zorder=0)
        
        perf = old_nnz_per_sec_fpga[i]
        line_cross = old_peak_performance_fpga[i] / peak_bandwidth_fpga[i]
        plt.plot([0, line_cross], [0, old_peak_performance_fpga[i]], color=palette[i], linewidth=1, zorder=1)
        plt.scatter(old_operational_intensity_fpga[i], perf, color=palette[i], edgecolors="#2f2f2f", marker=markers[i], s=SCATTER_SIZE, zorder=4, linewidth=0.3)     
                  
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
    ax.tick_params(axis="x", direction="out", which="both", bottom=True, top=False)
        
    ax.xaxis.grid(False)
    ax.yaxis.grid(linewidth=0.5)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(True) 
        tic.tick2line.set_visible(False) 
    for tic in ax.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(True) 
        tic.tick2line.set_visible(False) 
    ax.set_yticklabels(labels=[get_exp_label(l) for l in ax.get_yticks()], ha="right", fontsize=6)
    ax.tick_params(labelcolor="black", labelsize=6, pad=1)  
    ax.minorticks_on()
    ax.set_xlabel("Operational Intensity [NNZ/B]", fontsize=7)
    ax.set_ylabel("Performance [NNZ/s]", fontsize=7)
    
    # Add labels with bandwidth;
    for i in range(len(peak_bandwidth_fpga)): 
        op_int = 0.011
        label = f"{num_cores_fpga[i]} cores, {peak_bandwidth_fpga[i] / 10**9:.1f} GB/s"
        line_cross = peak_performance_fpga[i] / peak_bandwidth_fpga[i]
        tan = peak_performance_fpga[i] / line_cross
        angle = np.arctan(tan)
        angle = np.rad2deg(angle)
        trans_angle = ax.transData.transform_angles([angle], np.array([0, 0]).reshape((1, 2)))[0]
        ax.annotate(label, fontsize=4.5, xy=(op_int, peak_bandwidth_fpga[i] * op_int * 1.1), ha="left", rotation_mode='anchor', rotation=trans_angle, color="#2f2f2f")
    
    # Annotate B;
    for i in range(len(packet_sizes)): 
        op_int = packet_sizes[i] / (512 / 8)
        label = f"B={packet_sizes[i]}"
        ax.annotate(label, fontsize=6, xy=(op_int * 1.1, ax.get_ylim()[0] * 2), ha="left")
    ax.annotate("(a)", fontsize=10, xy=(0.5, -0.34), ha="center", xycoords="axes fraction" )
    
    # Add custom legend;
    labels = ["1 core", "8 cores", "16 cores", "32 cores"]
    custom_lines = [Line2D([], [], color="white", marker=markers[::-1][j], markeredgewidth=0.5, 
                markersize=5, label=labels[::-1][j], markerfacecolor=palette[::-1][j], markeredgecolor="#2f2f2f") for j in range(len(labels))]
    leg = ax.legend(custom_lines, labels[::-1], bbox_to_anchor=(1.1, 0), loc="lower center", fontsize=6, ncol=1, handletextpad=0.3, columnspacing=0.4)
    leg.set_title(None)
    leg._legend_box.align = "left"
    leg.get_frame().set_facecolor('white') 
    
    # plt.annotate("Operational intensity increase with BS-CSR", fontsize=8, xy=(-0.3, 1.1), xycoords="axes fraction", ha="left")
       
    plt.savefig(f"../../../../data/plots/roofline_cpu_gpu_{DATE}.pdf")