#pragma once

#include "hls_stream.h"
#include "../fpga_utils.hpp"
#include "spmv_utils.hpp"
#include "../fpga_types.hpp"
#include <iostream>

/////////////////////////////
/////////////////////////////

// Used to handle partial results of each COO packet;
typedef struct {
	real_type aggregated_res[LIMITED_FINISHED_ROWS];
	int_type num_rows_in_packet;
	bool_type fin;
} reduction_result_topk;

// HLS cannot correctly partition arrays in structs, so we have to resort to "ap_uint" for some of the streams;
#define INT_SIZE 32
typedef ap_uint<LIMITED_FINISHED_ROWS + 1> finished_rows_block;
typedef ap_uint<BSCSR_PACKET_SIZE * AP_INT_ROW_BITWIDTH> x_block;
typedef ap_uint<(LIMITED_FINISHED_ROWS + 1) * FIXED_WIDTH> aggregated_res_block;

/////////////////////////////
/////////////////////////////

#define MIN(res, a, b) (((res[(a)]) < (res[(b)])) ? (a) : (b))

inline void argmin_1(real_type *res, x_bscsr *worst_idx, real_type *worst_res) {
#pragma HLS inline
	*worst_idx = 0;
	*worst_res = res[0];
}

inline void argmin_2(real_type *res, x_bscsr *worst_idx, real_type *worst_res) {
#pragma HLS inline
	*worst_idx = MIN(res, 0, 1);
	*worst_res = res[*worst_idx];
}

inline void argmin_4(real_type *res, x_bscsr *worst_idx, real_type *worst_res) {
#pragma HLS inline
	x_bscsr m0 = MIN(res, 0, 1);
	x_bscsr m1 = MIN(res, 2, 2);

	*worst_idx = MIN(res, m0, m1);
	*worst_res = res[*worst_idx];
}

inline void argmin_8(real_type *res, x_bscsr *worst_idx, real_type *worst_res) {
#pragma HLS inline
	x_bscsr m0 = MIN(res, 0, 1);
	x_bscsr m1 = MIN(res, 2, 3);
	x_bscsr m2 = MIN(res, 4, 5);
	x_bscsr m3 = MIN(res, 6, 7);

	x_bscsr m01 = MIN(res, m0, m1);
	x_bscsr m23 = MIN(res, m2, m3);

	*worst_idx = MIN(res, m01, m23);
	*worst_res = res[*worst_idx];
}

inline void argmin_16(real_type *res, x_bscsr *worst_idx, real_type *worst_res) {
#pragma HLS inline
	x_bscsr m0 = MIN(res, 0, 1);
	x_bscsr m1 = MIN(res, 2, 3);
	x_bscsr m2 = MIN(res, 4, 5);
	x_bscsr m3 = MIN(res, 6, 7);

	x_bscsr m4 = MIN(res, 8, 9);
	x_bscsr m5 = MIN(res, 10, 11);
	x_bscsr m6 = MIN(res, 12, 13);
	x_bscsr m7 = MIN(res, 14, 15);

	x_bscsr m01 = MIN(res, m0, m1);
	x_bscsr m23 = MIN(res, m2, m3);
	x_bscsr m45 = MIN(res, m4, m5);
	x_bscsr m67 = MIN(res, m6, m7);

	x_bscsr m0123 = MIN(res, m01, m23);
	x_bscsr m4567 = MIN(res, m45, m67);

	*worst_idx = MIN(res, m0123, m4567);
	*worst_res = res[*worst_idx];
}

template <uint k>
inline void argmin(real_type *res, x_bscsr *worst_idx, real_type *worst_res) {
#pragma HLS inline
	int_type curr_min = 0;
	for (uint i = 0; i < k; i++) {
#pragma HLS unroll
		curr_min = MIN(res, curr_min, i);
	}
	*worst_idx = curr_min;
	*worst_res = res[curr_min];
}

/////////////////////////////
/////////////////////////////

inline void inner_spmv_topk_product_stream(
		hls::stream<input_packet_int_x_bscsr> &x,
		hls::stream<input_packet_real_bscsr> &val,
		hls::stream<input_packet_real_bscsr> &vec,
		hls::stream<reduction_result_topk> &aggregated_res,
		hls::stream<bool_type> &xf,
		hls::stream<input_packet_int_x_bscsr> &x_out) {
#pragma HLS inline
	input_packet_int_x_bscsr x_local = x.read();
	input_packet_real_bscsr val_local = val.read();
	input_packet_real_bscsr vec_local = vec.read();
	x_block x_block_out;

	real_type pointwise_res_local[BSCSR_PACKET_SIZE];
#pragma HLS array_partition variable=pointwise_res_local complete dim=1

	// Point-wise multiplication of a chunk of "val" and "scattered_vec";
	POINTWISE: for (x_bscsr k = 0; k < BSCSR_PACKET_SIZE; k++) {
#pragma HLS unroll
		real_type val_float = val_local[k];
		real_type vec_float = vec_local[k];
		pointwise_res_local[k] = val_float * vec_float;
	}

	reduction_result_topk result;

	int_type num_rows_in_packet = 0;
	for (x_bscsr i = 0; i < LIMITED_FINISHED_ROWS; i++) {
#pragma HLS unroll
		real_type aggregator = 0;
		x_bscsr start = (i > 0) ? x_local[i - 1] : (x_bscsr) 0;
		x_bscsr end = x_local[i];
		num_rows_in_packet += (start != end);
		for (x_bscsr j = 0; j < BSCSR_PACKET_SIZE; j++) {
#pragma HLS unroll
			aggregator += (j >= start && j < end) ? pointwise_res_local[j] : (real_type) 0;
		}
		result.aggregated_res[i] = aggregator;
	}

	result.num_rows_in_packet = num_rows_in_packet;
	result.fin = xf.read();

	x_out << x_local;
	aggregated_res << result;
}

/////////////////////////////
/////////////////////////////

inline void reset_buffer(real_type res[LIMITED_FINISHED_ROWS][K], int_type res_idx[LIMITED_FINISHED_ROWS][K]) {
#pragma HLS inline
	WRITE_LOCAL_1: for (x_bscsr j = 0; j < LIMITED_FINISHED_ROWS; j++) {
#pragma HLS unroll
		WRITE_LOCAL_2: for (x_bscsr k = 0; k < K; k++) {
#pragma HLS unroll
			res[j][k] = (real_type) 0.0;
			res_idx[j][k] = 0;
		}
	}
}

/////////////////////////////
/////////////////////////////


inline void spmv_coo_loop_1(
		int_type num_packets_coo,
		input_block *coo,
		hls::stream<input_packet_int_x_bscsr> &x_stream,
		hls::stream<input_packet_real_bscsr> &vec_stream,
		hls::stream<input_packet_real_bscsr> &val_stream,
		hls::stream<bool_type> &x_f_stream,
		real_type vec[BSCSR_PACKET_SIZE][MAX_COLS]
		) {
#pragma HLS inline
	static x_bscsr x_local[BSCSR_PACKET_SIZE];
#pragma HLS array_partition variable=x_local complete
//#pragma HLS stable variable=x_local
	static y_bscsr y_local[BSCSR_PACKET_SIZE];
#pragma HLS array_partition variable=y_local complete
//#pragma HLS stable variable=y_local
	static real_type val_local[BSCSR_PACKET_SIZE];
#pragma HLS array_partition variable=val_local complete
//#pragma HLS stable variable=val_local



    // Process the values of the COO matrix in a stream-like fashion, block by block.
    // COO values are 0-padded to have length multiple of BSCSR_PACKET_SIZE, and it doesn't affect the results;
    LOOP_1: for (int_type i = 0; i < num_packets_coo; i++) {
#pragma HLS loop_tripcount min=hls_iterations_nnz max=hls_iterations_nnz avg=hls_iterations_nnz
#pragma HLS pipeline II=hls_pipeline


    	input_packet_int_x_bscsr x_local_out;
		input_packet_int_y_bscsr y_local_out;
		input_packet_real_bscsr val_local_out;
		input_packet_real_bscsr vec_local;

		bool_type x_f_local;
    	// Read chunks of "x", "y", "val", then scatter values of "vec";
		input_block coo_local = coo[i];
    	read_block_x(coo_local, x_local);
		read_block_y(coo_local, y_local);
		read_block_val(coo_local, val_local);
		read_block_xf(coo_local, &x_f_local);
		READ_COO: for (x_bscsr j = 0; j < BSCSR_PACKET_SIZE; j++) {
#pragma HLS unroll
			x_local_out[j] = x_local[j];
			y_local_out[j] = y_local[j];
			val_local_out[j] = val_local[j];
			//vec_local[j] = vec[j / 2][y_local[j]];
		}

		read_block_vec_uram_parallel_bscsr(y_local, vec_local, vec);
		val_stream << val_local_out;
		x_stream << x_local_out;
		vec_stream << vec_local;
		x_f_stream << x_f_local;
    }
}



/////////////////////////////
/////////////////////////////


inline void spmv_coo_loop_2(
		int_type num_packets_coo,
		hls::stream<input_packet_int_x_bscsr> &x_stream,
		hls::stream<input_packet_real_bscsr> &vec_stream,
		hls::stream<input_packet_real_bscsr> &val_stream,
		hls::stream<reduction_result_topk> &aggregated_res_stream,
		hls::stream<bool_type> &x_f_stream,
		hls::stream<input_packet_int_x_bscsr> &x_stream_out
		) {
#pragma HLS inline
    LOOP_2: for (int_type i = 0; i < num_packets_coo; i++) {
#pragma HLS loop_tripcount min=hls_iterations_nnz max=hls_iterations_nnz avg=hls_iterations_nnz
#pragma HLS pipeline II=hls_pipeline
		// Perform point-wise products;
		inner_spmv_topk_product_stream(x_stream, vec_stream, val_stream, aggregated_res_stream, x_f_stream, x_stream_out);
    }
}

/////////////////////////////
/////////////////////////////

inline void spmv_coo_loop_3(
		int_type num_packets_coo,
		hls::stream<reduction_result_topk> &aggregated_res_stream,
		hls::stream<input_packet_int_x_bscsr> &x_stream_out,
		hls::stream<finished_rows_block> &finished_rows_stream,
		hls::stream<int_type> &start_x_stream,
		hls::stream<aggregated_res_block> &aggregated_res_local_stream,
		real_type aggregated_res_local[LIMITED_FINISHED_ROWS + 1],
		bool finished_rows[LIMITED_FINISHED_ROWS + 1]
		) {
#pragma HLS inline
	// Support array used in the storage FSM.
    // Values written in it are the same for each column,
    //   but we need an array to have independent parallel R/W access;
	int_type last_row_of_packet = 0;
	real_type last_row_of_packet_output = (real_type) 0.0;

	LOOP_3: for (int_type i = 0; i < num_packets_coo; i++) {
#pragma HLS loop_tripcount min=hls_iterations_nnz max=hls_iterations_nnz avg=hls_iterations_nnz
#pragma HLS pipeline II=hls_pipeline

	    // Reset local storage buffers;
		RESET_BUFFERS: for (int_type j = 0; j < LIMITED_FINISHED_ROWS + 1; j++) {
#pragma HLS unroll
			aggregated_res_local[j] = 0;
			finished_rows[j] = false;
		}

		// Read the aggregated stream output;
		reduction_result_topk reduction_result_local = aggregated_res_stream.read();
		input_packet_int_x_bscsr x_packet_tmp = x_stream_out.read();

		int_type packet_starts_with_new_row = (i != 0) ? (int_type) reduction_result_local.fin : 0;

		int_type num_rows_in_packet = reduction_result_local.num_rows_in_packet;
		int_type finished_rows_num = num_rows_in_packet + packet_starts_with_new_row - 1;
		int_type start_row_of_packet = last_row_of_packet + packet_starts_with_new_row; // starting row of current packet
		last_row_of_packet += finished_rows_num; // last row of current packet

		READ_AGGREGATED_RES_PACKET: for (x_bscsr j = 0; j < LIMITED_FINISHED_ROWS; j++) {
#pragma HLS unroll
			aggregated_res_local[1 + j] = reduction_result_local.aggregated_res[j];  // Leave empty the first value;
		}

		// All rows with x_packet_tmp != 0 are finished, except the last non-zero entry (position "num_rows_in_packet - 1" in x_packet_tmp);
		FINISHED_ROW_CHECK: for (x_bscsr j = 1; j < LIMITED_FINISHED_ROWS; j++) {
#pragma HLS unroll
			finished_rows[j] = x_packet_tmp[j - 1] != ((j > 1) ? x_packet_tmp[j - 2] : (x_bscsr) 0);
		}
		finished_rows[num_rows_in_packet] = false;

		// If the last row in the previous packet was split between packets, update the first result in this packet;
		if (!packet_starts_with_new_row) {
			aggregated_res_local[1] += last_row_of_packet_output;
			aggregated_res_local[0] = 0;
			finished_rows[0] = false;
		} else {
			// If last packet row was finished at the end of the last packet, store its value at the start of this one,
			//   and process it in this iteration;
			aggregated_res_local[0] = last_row_of_packet_output;
			finished_rows[0] = true;
		}

		// Book-keeping at the end of processing a packet;
		last_row_of_packet_output = aggregated_res_local[num_rows_in_packet];

		// Prepare packets for the final stage of data-flow;
		finished_rows_block finished_rows_b;
		aggregated_res_block aggregated_res_b;
		for (int_type j = 0; j < LIMITED_FINISHED_ROWS + 1; j++) {
#pragma HLS unroll
			finished_rows_b.bit(j) = finished_rows[j];
			unsigned int lower_range = FIXED_WIDTH * j;
			unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
			real_type agg_tmp = aggregated_res_local[j];
			aggregated_res_b.range(upper_range, lower_range) = *((val_bscsr *) &agg_tmp);
		}

		finished_rows_stream << finished_rows_b;
		start_x_stream << start_row_of_packet;
		aggregated_res_local_stream << aggregated_res_b;
	}



}

/////////////////////////////
/////////////////////////////



inline void spmv_coo_loop_4(
	int_type num_packets_coo,
	hls::stream<int_type> &start_x_stream,
	hls::stream<finished_rows_block> &finished_rows_stream,
	hls::stream<aggregated_res_block> &aggregated_res_local_stream,
	int_type res_idx[BSCSR_PACKET_SIZE][K],
	real_type res[BSCSR_PACKET_SIZE][K],
	x_bscsr curr_worst_idx[LIMITED_FINISHED_ROWS],
	real_type curr_worst_val[LIMITED_FINISHED_ROWS],
	real_type aggregated_res_local_2[LIMITED_FINISHED_ROWS + 1],
	bool finished_rows_2[LIMITED_FINISHED_ROWS + 1],
	real_type res_local[LIMITED_FINISHED_ROWS][K],
	int_type res_idx_local[LIMITED_FINISHED_ROWS][K]

	){
#pragma HLS inline



	LOOP_4: for (int_type i = 0; i < num_packets_coo; i++) {
#pragma HLS loop_tripcount min=hls_iterations_nnz max=hls_iterations_nnz avg=hls_iterations_nnz
#pragma HLS pipeline II=hls_pipeline

		int_type start_x = start_x_stream.read();
		finished_rows_block finished_rows_b = finished_rows_stream.read();
		aggregated_res_block aggregated_res_b = aggregated_res_local_stream.read();

		// Move values to local arrays;
		for (int_type j = 0; j < LIMITED_FINISHED_ROWS + 1; j++) {
#pragma HLS unroll
			finished_rows_2[j] = (bool) finished_rows_b.bit(j);
			unsigned int lower_range = FIXED_WIDTH * j;
			unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
			unsigned int val_curr = aggregated_res_b.range(upper_range, lower_range);
			aggregated_res_local_2[j] = *((real_type *) &val_curr);
		}

		// Update the Top-K values;
		UPDATE_TOPK: for (x_bscsr j = 0; j < LIMITED_FINISHED_ROWS; j++) {
#pragma HLS unroll
			real_type curr_val = aggregated_res_local_2[j];
			// Replace the current worst with the new result;
			if (curr_val >= curr_worst_val[j] && finished_rows_2[j]) {
				// Replace the current worst with the new result;
				res_idx_local[j][curr_worst_idx[j]] = start_x + j - 1; // -1 as the first place in aggregated_res_local refers to the previous packet;
				res_local[j][curr_worst_idx[j]] = curr_val;
			}
			// Find the new worst;
#if (K == 1)
			argmin_1(res_local[j], &curr_worst_idx[j], &curr_worst_val[j]);
#elif (K == 2)
			argmin_2(res_local[j], &curr_worst_idx[j], &curr_worst_val[j]);
#elif (K == 4)
			argmin_4(res_local[j], &curr_worst_idx[j], &curr_worst_val[j]);
#elif (K == 8)
			argmin_8(res_local[j], &curr_worst_idx[j], &curr_worst_val[j]);
#elif (K == 16)
			argmin_16(res_local[j], &curr_worst_idx[j], &curr_worst_val[j]);
#else
			argmin<K>(res_local[j], &curr_worst_idx[j], &curr_worst_val[j]);
#endif
		}
    }

	// Handle the last packet outside of the data-flow region; remove it if the hw_emulation stalls. This loop prevents a dataflow synthesis, it seems;
//	LAST_ROW: for (int i = 0; i < BSCSR_PACKET_SIZE; i++) {
//#pragma HLS unroll
//		if (last_row_of_packet_output >= curr_worst_val[i]) {
//			res_idx_local[i][curr_worst_idx[i]] = last_row_of_packet;
//			res_local[i][curr_worst_idx[i]] = last_row_of_packet_output;
//		}
//	}

	WRITE_LOCAL_2: for (x_bscsr i = 0; i < K; i++) {
#pragma HLS unroll
		WRITE_LOCAL_RES_3: for (x_bscsr j = 0; j < K; j++) {
#pragma HLS unroll
			res[i][j] = (i < LIMITED_FINISHED_ROWS) ? res_local[i][j] : (real_type) 0;
			res_idx[i][j] = (i < LIMITED_FINISHED_ROWS) ? res_idx_local[i][j] : 0;
		}
	}




}



/////////////////////////////
/////////////////////////////



inline void spmv_bscsr_top_k_multi_stream_inner(input_block *coo, int_type rows, int_type cols, int_type nnz,
		real_type vec[BSCSR_PACKET_SIZE][MAX_COLS], int_type res_idx[BSCSR_PACKET_SIZE][K], real_type res[BSCSR_PACKET_SIZE][K],x_bscsr curr_worst_idx[LIMITED_FINISHED_ROWS],real_type curr_worst_val[LIMITED_FINISHED_ROWS]) {

#pragma HLS dataflow

#pragma HLS stable variable=res_idx
#pragma HLS stable variable=res
#pragma HLS stable variable=vec
#pragma HLS stable variable=curr_worst_idx
#pragma HLS stable variable=curr_worst_val

	int_type num_packets_coo = (nnz + BSCSR_PACKET_SIZE - 1) / BSCSR_PACKET_SIZE;



    // Define streams;
    hls::stream<input_packet_int_x_bscsr> x_stream_1("x_stream_1");
#pragma HLS STREAM variable=x_stream_1 depth=1
	hls::stream<input_packet_real_bscsr> val_stream_1("val_stream_1");
#pragma HLS STREAM variable=val_stream_1 depth=1
	hls::stream<input_packet_real_bscsr> vec_stream_1("vec_stream_1");
#pragma HLS STREAM variable=vec_stream_1 depth=1

    hls::stream<input_packet_real_coo> pointwise_stream;
#pragma HLS STREAM variable=pointwise_stream depth=1
    hls::stream<reduction_result_topk> aggregated_res_stream;
#pragma HLS STREAM variable=aggregated_res_stream depth=1

    hls::stream<bool_type> x_f_stream_1("x_f_stream_1");
#pragma HLS STREAM variable=x_f_stream_1 depth=1

    hls::stream<input_packet_int_x_bscsr> x_stream_out;
#pragma HLS STREAM variable=x_stream_out depth=1

    hls::stream<finished_rows_block> finished_rows_stream;
#pragma HLS STREAM variable=finished_rows_stream depth=1
    hls::stream<aggregated_res_block> aggregated_res_local_stream;
#pragma HLS STREAM variable=aggregated_res_local_stream depth=1
    hls::stream<int_type> start_x_stream;
#pragma HLS STREAM variable=start_x_stream depth=1

	real_type aggregated_res_local[LIMITED_FINISHED_ROWS + 1];
#pragma HLS array_partition variable=aggregated_res_local complete dim=0
//#pragma HLS stable variable=aggregated_res_local
	bool finished_rows[LIMITED_FINISHED_ROWS + 1];
#pragma HLS array_partition variable=finished_rows complete dim=0
//#pragma HLS stable variable=finished_rows


	real_type aggregated_res_local_2[LIMITED_FINISHED_ROWS + 1];
#pragma HLS array_partition variable=aggregated_res_local_2 complete dim=0
//#pragma HLS stable variable=aggregated_res_local_2
	bool finished_rows_2[LIMITED_FINISHED_ROWS + 1];
#pragma HLS array_partition variable=finished_rows_2 complete dim=0
//#pragma HLS stable variable=finished_rows_2

	// Store results;
	real_type res_local[LIMITED_FINISHED_ROWS][K];
#pragma HLS array_partition variable=res_local complete dim=0
//#pragma HLS stable variable=res_local

	int_type res_idx_local[LIMITED_FINISHED_ROWS][K];
#pragma HLS array_partition variable=res_idx_local complete dim=0
//#pragma HLS stable variable=res_idx_local

	reset_buffer(res_local, res_idx_local);


////////FROM HERE ONLY FUNCTION CALLS ARE ALLOWED///////////////////


	spmv_coo_loop_1(num_packets_coo, coo, x_stream_1, vec_stream_1, val_stream_1, x_f_stream_1, vec );

	spmv_coo_loop_2(num_packets_coo, x_stream_1, vec_stream_1, val_stream_1, aggregated_res_stream, x_f_stream_1, x_stream_out);

	spmv_coo_loop_3(num_packets_coo, aggregated_res_stream, x_stream_out, finished_rows_stream, start_x_stream, aggregated_res_local_stream,aggregated_res_local,finished_rows);

	spmv_coo_loop_4(num_packets_coo,start_x_stream,finished_rows_stream,aggregated_res_local_stream,res_idx,res,curr_worst_idx,curr_worst_val,aggregated_res_local_2,finished_rows_2,res_local,res_idx_local);



}

/////////////////////////////
/////////////////////////////

inline void spmv_bscsr_top_k_multi_stream(input_block *coo, int_type rows, int_type cols, int_type nnz,
		real_type vec[BSCSR_PACKET_SIZE][MAX_COLS], int_type res_idx[BSCSR_PACKET_SIZE][K], real_type res[BSCSR_PACKET_SIZE][K]) {
#pragma HLS inline

	x_bscsr curr_worst_idx[LIMITED_FINISHED_ROWS];
#pragma HLS array_partition variable=curr_worst_idx complete dim=0
//#pragma HLS stable variable=curr_worst_idx
	real_type curr_worst_val[LIMITED_FINISHED_ROWS];
#pragma HLS array_partition variable=curr_worst_val complete dim=0
//#pragma HLS stable variable=curr_worst_val



	for (x_bscsr i = 0; i < LIMITED_FINISHED_ROWS; i++) {
#pragma HLS unroll
		curr_worst_idx[i] = 0;
		curr_worst_val[i] = (real_type) 0.0;
	}
	// It's useful to wrap the data-flow function so that we can do some post-processing if required;
	spmv_bscsr_top_k_multi_stream_inner(coo, rows, cols, nnz, vec, res_idx, res,curr_worst_idx,curr_worst_val);
}

/////////////////////////////
/////////////////////////////

extern "C" void spmv_bscsr_top_k_main(
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
		);
