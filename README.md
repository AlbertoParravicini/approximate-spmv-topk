# Top-K SpMV for Approximate Embedding Similarity on FPGAs
Repository for the DAC 2021 paper **Scaling up HBM Efficiency of Top-K SpMV for Approximate Embedding Similarity on FPGAs**

## Abstract

Top-K SpMV is a key component of similarity-search on sparse embeddings. This sparse workload does not perform well on general-purpose NUMA systems that employ traditional caching strategies.
Instead, modern FPGA accelerator cards have a few tricks up their sleeve. We introduce a Top-K SpMV FPGA design that leverages reduced precision and a novel packet-wise CSR matrix compression, enabling custom data layouts and delivering bandwidth efficiency often unreachable even in architectures with higher peak bandwidth.
With HBM-based boards, we are 100x faster than a multi-threaded CPU implementation and 2x faster than a GPU with 20% higher bandwidth, with 14.2x higher power-efficiency.

In short, **blazing fast sparse embedding approximate lookup!!!**

### High-level drawing of our FPGA design

![High-level FPGA architecture](https://github.com/AlbertoParravicini/approximate-spmv-topk/blob/main/data/plots/architecture.png)

### Summary of the our performance results

![Speedup of our design](https://github.com/AlbertoParravicini/approximate-spmv-topk/blob/main/data/plots/exec_time_2021_03_07.png)

![Roofline model of our design](https://github.com/AlbertoParravicini/approximate-spmv-topk/blob/main/data/plots/roofline_cpu_gpu_2021_03_07.png)

If you liked the Roofline Model plot, check out how to [recreate](https://github.com/AlbertoParravicini/segretini-matplottini) it!

## Setup and installation

### Building the FPGA design

Our design was built and tested on a Xilinx Alveo U280, using platform `xilinx_u280_xdma_201920_3`, and compiled with Vitis 2019.2. It will likely work with more recent platforms or tools, however.
To build the FPGA design, just run `make build TARGET=hw`. Alternatively, if you just want to build the host, run `make host TARGET=hw`.
In the Makefile there are different settings you might want to change. The clock frequency is set to `450 MHz` by default, although the design will likely build to a lower frequency: if the build fails, try lowering this value first.
The Makefile will build a design with 8 cores, with 4 sub-cores each (for a total of 32 partitions): if you need a quicker build, or your FPGA has more (or fewer) than 32 HBM pseudo-channels, you might consider changing the number of cores.

File `src/common/types.hpp` allows specifying other configuration parameters for the design.
* `FIXED_WIDTH` is the fixed-point precision, assuming 1 bit for the integer part, as specified in `SCALE`
* `USE_FLOAT` must be set to `true` if building a floating-point design (leaving `FIXED_WIDTH` to 32)
* `SPMV_PARTITIONS` must match the total number of partitions as specified in the Makefile (by default, it is equal to 32, i.e. 8 * 4)
* `MAX_COLS` is the maximum embedding size, change it according to your application (by default, it is 1024)
* `K` is the number of Top-K embeddings computed in each partition. Lower values drastically reduce resource utilization and improve the clock frequency. The default value of 8 is a good compromise between accuracy and performance.
* `LIMITED_FINISHED_ROWS` is used as addtional approximation factor, as we assume that only `LIMITED_FINISHED_ROWS` can occur in each packet. It can drastically reduce resource utilization. You can use low values (e.g. 2 or 3) in matrices with low sparsity, as it's unlikely that many rows will finish in the same packet.

### Building the GPU baseline

In our performance comparisons we also use GPU code. If you want to build it, just do `cd src/gpu; make`.
Note that inside `src/gpu/Makefile` you might have to change `CC` to your local `nvcc` installation, and `-arch` (inside `FLAGS`) to reflect your current GPU architecture (e.g. `sm_60` is Pascal, `sm_70` is Volta, etc.).
See [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/) to find your architecture.

### Building the CPU baseline

For the CPU baseline, you need Python `>= 3.6` and the `sparse-dot-topn` package. Installation is straightforward using `pip`. 
See [here](https://pypi.org/project/sparse-dot-topn/) and [here](https://medium.com/wbaa/https-medium-com-ingwbaa-boosting-selection-of-the-most-similar-entities-in-large-scale-datasets-450b3242e618) to know more.

## Testing our FPGA design

To perform testing, you will need sparse matrices, stored as MTX files inside `data/matrices_for_testing`. Matrices should be 1-indexed, as it's by default in the MTX format (although there's support for 0-indexed matrices, but it requires recompilation of the host).
A few small matrices are provided as sample, but you can create bigger matrices through `src/resources/python/create_matrices.py`

Command-line options for the FPGA and GPU are contained inside `src/comon/utils/options.hpp`. Running tests automatically (on FPGA, GPU, CPU) is possible through the `test_spmv_topk.py` script.
Please refer to the comments at the start of the script to properly configure the testing setup (e.g. the path to your FPGA bitstream).

You can also just run the FPGA executable from the command line, as shown below

```
cd build/hw/xilinx_u280_xdma_201920_3
./spmv_coo_hbm_topk_multicore_mega_main -x spmv_bscsr_top_k_main.xclbin -d -m ../../../data/matrices_for_testing/matrix_10000_1024_20_gamma.mtx -k 100 -t 30`
```

## Credits and Contributors
Contributors: **Alberto Parravicini, Luca Giuseppe Cellamare, Marco Siracusa, Marco Domenico Santambrogio**

If you find this repository useful, please use the following citation(s):

```
@misc{parravicini2021scaling,
      title={Scaling up HBM Efficiency of Top-K SpMV for Approximate Embedding Similarity on FPGAs}, 
      author={Alberto Parravicini and Luca Giuseppe Cellamare and Marco Siracusa and Marco Domenico Santambrogio},
      year={2021},
      eprint={2103.04808},
      archivePrefix={arXiv},
      primaryClass={cs.AR}
}

@inproceedings{
    author = {Parravicini, Alberto and Cellamare, Luca Giuseppe and Siracusa, Marco and Santambrogio, Marco D},
    title = {Scaling up HBM Efficiency of Top-K SpMV for Approximate Embedding Similarity on FPGAs},
    booktitle = {To appear in Proceedings of the 58th Design Automation Conference (DAC)},
    year = {2021}
}
```