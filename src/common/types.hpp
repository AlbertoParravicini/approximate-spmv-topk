#pragma once

#include <vector>
#include <stdio.h>

//////////////////////////////
// OpenCL parameters /////////
//////////////////////////////

// Number of OpenCL out-of-order queues;
#define OPENCL_QUEUES 1

//////////////////////////////
// Generic ///////////////////
//////////////////////////////

typedef unsigned int int_type;

// Define fixed-point values;
#define FIXED_WIDTH 32
#define SCALE (FIXED_WIDTH - 1)
#define FIXED_INTEGER_PART (FIXED_WIDTH - SCALE)

// Define output fixed-point values;
#define FIXED_WIDTH_OUT 32
#define SCALE_OUT (FIXED_WIDTH_OUT - 1)
#define FIXED_INTEGER_PART_OUT (FIXED_WIDTH_OUT - SCALE_OUT)

#define USE_FLOAT false

//////////////////////////////
// COO SpMV //////////////////
//////////////////////////////

// Number of partions of SpMV, and total number of Compute Units on the FPGA;
#define SPMV_PARTITIONS 32
#define SUB_SPMV_PARTITIONS 4
#define SUPER_SPMV_PARTITIONS (SPMV_PARTITIONS / SUB_SPMV_PARTITIONS)
#define COO_PACKET_SIZE 5

// Define packets of 384 bits, 4 COO entries per packet;
#define COO_PACKET_SIZE_4 4

#define CACHE_SIZE 512

//////////////////////////////
// COO TopK-SpMV /////////////
//////////////////////////////

// Obtain at most the Top-8 ranked documents;
#define K 8
// Number of replicas of the result vectors used by Top-K SpMV: more replicas give a faster clock but produce more results and use more LUTs;
#define TOPK_RES_COPIES 1
// Maximum number of columns when storing vec in URAM/LUT
#define MAX_COLS 1024

//////////////////////////////
// Hybrid TopK-SpMV //////////
//////////////////////////////

#define AP_INT_VAL_BITWIDTH FIXED_WIDTH
#define AP_INT_COL_BITWIDTH 10
#define AP_INT_ROW_BITWIDTH 4

#define DIM_BOOL 1
#define PACKET_TRIPLET_WIDTH (AP_INT_VAL_BITWIDTH + AP_INT_COL_BITWIDTH + AP_INT_ROW_BITWIDTH)

// Define packets of COO entries;
//#define AP_INT_COL_BITWIDTH (int) log2(512)+1

#define BSCSR_PORT_BITWIDTH 512
#define BSCSR_PACKET_SIZE ((BSCSR_PORT_BITWIDTH - DIM_BOOL) / (PACKET_TRIPLET_WIDTH))
#define PADDING_SIZE (BSCSR_PORT_BITWIDTH - BSCSR_PACKET_SIZE * PACKET_TRIPLET_WIDTH + DIM_BOOL)

// As further approximation, assume that only LIMITED_FINISHED_ROWS can occur in each packet, saving a lot of resources;
//#define LIMITED_FINISHED_ROWS BSCSR_PACKET_SIZE
#define LIMITED_FINISHED_ROWS 4

#define VEC_REPLICAS ((BSCSR_PACKET_SIZE + 2 - 1) / 2)

//////////////////////////////
// CSR SpMV //////////////////
//////////////////////////////

#define CSR_PORT_BITWIDTH 512
#define CSR_PACKET_SIZE 8

//////////////////////////////
// Other/Old /////////////////
//////////////////////////////

#define MAX_LENGTH_DOC 43
#define NUM_DOCS 256
#define ALPHABET_SIZE 26

// Use bi-grams;
#define NGRAMS (ALPHABET_SIZE * ALPHABET_SIZE)
// Process multiple documents at once when possible;
#define DOC_BUFFER_SIZE 4

// Maximum number of rows when storing vec/result in URAM
#define MAX_ROWS (2 << 10) // 2^10

// Define packets of values;
#define BUFFER_SIZE 8 // Each ap_uint has at most 512 bits;
#define PACKET_ELEMENT_SIZE 32 // Single values have FIXED_WIDTH bits, but we have to 0-pad the packet to reach AP_UINT_BITWIDTH;
#define AP_UINT_BITWIDTH (PACKET_ELEMENT_SIZE * BUFFER_SIZE)

typedef unsigned int occ_count_type;
typedef int_type index_type;
typedef index_type doc_num_type;

//////////////////////////////
// HLS Pragmas ///////////////
//////////////////////////////

// Values required for HLS pragmas;
const int hls_num_docs = NUM_DOCS;
const int hls_buffer_size = BUFFER_SIZE;
const int hls_buffer_size_coo = COO_PACKET_SIZE;
const int hls_buffer_size_coo_4 = COO_PACKET_SIZE_4;
const int hls_ngrams = NGRAMS;
const int hls_doc_buffer_size = DOC_BUFFER_SIZE;

const int hls_num_rows = 1000;
const int hls_num_cols = MAX_COLS;
const int hls_degree = 25;
const int hls_k = K;
const int hls_num_nnz = hls_degree * hls_num_rows;
const int hls_iterations_rows = hls_num_rows / hls_buffer_size;
const int hls_iterations_cols = hls_num_cols / hls_buffer_size;
const int hls_iterations_nnz = hls_num_nnz / hls_buffer_size_coo;
const int hls_iterations_nnz_4 = hls_num_nnz / hls_buffer_size_coo_4;
const int hls_topk_res_copies = TOPK_RES_COPIES;
const int hls_topk_res_update_loop = (hls_iterations_nnz + TOPK_RES_COPIES - 1) / TOPK_RES_COPIES;
#if USE_FLOAT
const int hls_pipeline = 1;
#else
const int hls_pipeline = 1;
#endif
