#include "spmv_bscsr_top_k_multicore.hpp"

/////////////////////////////
/////////////////////////////

#define USE_URAM_FOR_VEC true

void spmv_bscsr_top_k_main(
		input_block *coo0,
		input_block *coo1,
		input_block *coo2,
		input_block *coo3,
		int_type num_rows0,
		int_type num_rows1,
		int_type num_rows2,
		int_type num_rows3,
		int_type num_cols0,
		int_type num_cols1,
		int_type num_cols2,
		int_type num_cols3,
		int_type nnz0,
		int_type nnz1,
		int_type nnz2,
		int_type nnz3,
		vec_real_inout_bscsr *vec,
		input_packet_int_bscsr *res_idx0,
		input_packet_int_bscsr *res_idx1,
		input_packet_int_bscsr *res_idx2,
		input_packet_int_bscsr *res_idx3,
		input_packet_real_inout_bscsr *res0,
		input_packet_real_inout_bscsr *res1,
		input_packet_real_inout_bscsr *res2,
		input_packet_real_inout_bscsr *res3
		) {
// Ports used to transfer data, using AXI master;
#pragma HLS INTERFACE m_axi port = coo0 offset = slave bundle = gmem0 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = coo1 offset = slave bundle = gmem1 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = coo2 offset = slave bundle = gmem2 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = coo3 offset = slave bundle = gmem3 // num_write_outstanding = 32 latency = 100

#pragma HLS INTERFACE m_axi port = vec offset = slave bundle = gmem0 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = res_idx0 offset = slave bundle = gmem0 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = res_idx1 offset = slave bundle = gmem1 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = res_idx2 offset = slave bundle = gmem2 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = res_idx3 offset = slave bundle = gmem3 // num_write_outstanding = 32 latency = 100

#pragma HLS INTERFACE m_axi port = res0 offset = slave bundle = gmem0 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = res1 offset = slave bundle = gmem1 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = res2 offset = slave bundle = gmem2 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = res3 offset = slave bundle = gmem3 // num_write_outstanding = 32 latency = 100

// Ports used for control signals, using AXI slave;
#pragma HLS INTERFACE s_axilite register port = num_rows0 bundle = control
#pragma HLS INTERFACE s_axilite register port = num_rows1 bundle = control
#pragma HLS INTERFACE s_axilite register port = num_rows2 bundle = control
#pragma HLS INTERFACE s_axilite register port = num_rows3 bundle = control
#pragma HLS INTERFACE s_axilite register port = num_cols0 bundle = control
#pragma HLS INTERFACE s_axilite register port = num_cols1 bundle = control
#pragma HLS INTERFACE s_axilite register port = num_cols2 bundle = control
#pragma HLS INTERFACE s_axilite register port = num_cols3 bundle = control

#pragma HLS INTERFACE s_axilite register port = nnz0 bundle = control
#pragma HLS INTERFACE s_axilite register port = nnz1 bundle = control
#pragma HLS INTERFACE s_axilite register port = nnz2 bundle = control
#pragma HLS INTERFACE s_axilite register port = nnz3 bundle = control

#pragma HLS INTERFACE s_axilite register port = return bundle = control

// Pragmas used for data-packing;
#pragma HLS data_pack variable=res_idx0 struct_level
#pragma HLS data_pack variable=res_idx1 struct_level
#pragma HLS data_pack variable=res_idx2 struct_level
#pragma HLS data_pack variable=res_idx3 struct_level

#pragma HLS data_pack variable=res0 struct_level
#pragma HLS data_pack variable=res1 struct_level
#pragma HLS data_pack variable=res2 struct_level
#pragma HLS data_pack variable=res3 struct_level

	static real_type res_local[SUB_SPMV_PARTITIONS][BSCSR_PACKET_SIZE][K];
#pragma HLS array_partition variable=res_local complete dim=0

	static int_type res_local_idx[SUB_SPMV_PARTITIONS][BSCSR_PACKET_SIZE][K];
#pragma HLS array_partition variable=res_local_idx complete dim=0

	// Dense vector allocated in URAM;
	static real_type vec_local[SUB_SPMV_PARTITIONS][VEC_REPLICAS][MAX_COLS];
#pragma HLS RESOURCE variable=vec_local core=XPM_MEMORY uram
#pragma HLS array_partition variable=vec_local complete dim=1
#pragma HLS array_partition variable=vec_local complete dim=2

	// Reset the local vector buffer;
	RESET_OUTPUT_0: for (int_type s = 0; s < SUB_SPMV_PARTITIONS; s++) {
#pragma HLS unroll
		RESET_OUTPUT_1: for (int_type i = 0; i < BSCSR_PACKET_SIZE; i++) {
#pragma HLS unroll
			RESET_OUTPUT_2: for (int_type j = 0; j < K; j++) {
#pragma HLS unroll
				res_local[s][i][j] = 0.0;
				res_local_idx[s][i][j] = 0;
			}
		}
	}

	static vec_real_inout_bscsr vec_loader[(MAX_COLS + BSCSR_PACKET_SIZE - 1) / BSCSR_PACKET_SIZE];
#pragma HLS RESOURCE variable=vec_loader core=XPM_MEMORY uram

	////////////////////////////////
	////////////////////////////////


	int_type num_blocks_cols0 = (num_cols0 + BSCSR_PACKET_SIZE - 1) / BSCSR_PACKET_SIZE;

	// Read the input "vec" and store it on a local buffer;
	vec_real_inout_bscsr vec_tmp;
	READ_INPUT: for (int_type i = 0; i < num_blocks_cols0; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=hls_iterations_cols max=hls_iterations_cols avg=hls_iterations_cols
		vec_loader[i] = vec[i];

	}

	// Copy the input "vec" on all the other "vec" copies;
	COPY_INPUT: for (int_type i = 0; i < num_cols0; i++) {
#pragma HLS loop_tripcount min=hls_num_cols max=hls_num_cols avg=hls_num_cols
#pragma HLS pipeline II=1
		vec_real_inout_bscsr vec_tmp = vec_loader[i / BSCSR_PACKET_SIZE];
		unsigned int lower_val_range = FIXED_WIDTH_OUT * (i % BSCSR_PACKET_SIZE);
		unsigned int upper_val_range = FIXED_WIDTH_OUT * ((i % BSCSR_PACKET_SIZE) + 1) - 1;
		unsigned int block_curr = vec_tmp.range(upper_val_range, lower_val_range);
		real_type_inout curr = *((real_type_inout *) &block_curr);

		for (int_type s = 0; s < SUB_SPMV_PARTITIONS; s++) {
#pragma HLS unroll
			for (int_type j = 0; j < VEC_REPLICAS; j++) {
#pragma HLS unroll
				vec_local[s][j][i] = (real_type) curr;
			}
		}
	}

	// Main SpMV computation;
	spmv_bscsr_top_k_multi_stream(coo0, num_rows0, num_cols0, nnz0, vec_local[0], res_local_idx[0], res_local[0]);
	spmv_bscsr_top_k_multi_stream(coo1, num_rows1, num_cols1, nnz1, vec_local[1], res_local_idx[1], res_local[1]);
	spmv_bscsr_top_k_multi_stream(coo2, num_rows2, num_cols2, nnz2, vec_local[2], res_local_idx[2], res_local[2]);
	spmv_bscsr_top_k_multi_stream(coo3, num_rows3, num_cols3, nnz3, vec_local[3], res_local_idx[3], res_local[3]);

	////////////////////////////////
	////////////////////////////////

	WRITE_OUTPUT: for (int_type i = 0; i < K; i++) {
#pragma HLS pipeline II=hls_pipeline
		input_packet_int_bscsr packet_idx0;
		input_packet_int_bscsr packet_idx1;
		input_packet_int_bscsr packet_idx2;
		input_packet_int_bscsr packet_idx3;
		input_packet_real_inout_bscsr packet0;
		input_packet_real_inout_bscsr packet1;
		input_packet_real_inout_bscsr packet2;
		input_packet_real_inout_bscsr packet3;

			input_packet_real_inout_bscsr packet;
		for (int_type j = 0; j < BSCSR_PACKET_SIZE; j++) {
#pragma HLS unroll
			packet_idx0[j] = res_local_idx[0][j][i];
			packet0[j] = (real_type_inout) res_local[0][j][i];

			packet_idx1[j] = res_local_idx[1][j][i];
			packet1[j] = (real_type_inout) res_local[1][j][i];

			packet_idx2[j] = res_local_idx[2][j][i];
			packet2[j] = (real_type_inout) res_local[2][j][i];

			packet_idx3[j] = res_local_idx[3][j][i];
			packet3[j] = (real_type_inout) res_local[3][j][i];
		}
		res_idx0[i] = packet_idx0;
		res0[i] = packet0;
		res_idx1[i] = packet_idx1;
		res1[i] = packet1;
		res_idx2[i] = packet_idx2;
		res2[i] = packet2;
		res_idx3[i] = packet_idx3;
		res3[i] = packet3;
	}
}
