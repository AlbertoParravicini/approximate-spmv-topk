#pragma once

#include "fpga_types.hpp"
#ifndef uint
#define uint unsigned int
#endif

/////////////////////////////
////APPPROXIMATE_PAGERANK////
/////////////////////////////

// Read 16 values from a  bits block, and write them in a buffer of the specified type;
inline void read_block_index(input_block block, int_type buffer_out[BUFFER_SIZE]) {
#pragma HLS INLINE
	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int lower_range = sizeof(int_type) * j;
		unsigned int upper_range = sizeof(int_type) * (j + 1) - 1;
		unsigned int block_curr = block.range(upper_range, lower_range);
		buffer_out[j] = block_curr;
	}
}

//inline void read_packet_int(input_block_int block, int_type buffer_out[BUFFER_SIZE]) {
//#pragma HLS INLINE
//	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
//#pragma HLS UNROLL
//		buffer_out[j] = block[j];
//	}
//}

inline void write_block_index(input_block *block, int_type buffer_in[BUFFER_SIZE]) {
#pragma HLS INLINE
	WRITE_BLOCK: for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS unroll
		unsigned int lower = sizeof(int_type) * i;
		unsigned int upper = sizeof(int_type) * (i + 1) - 1;
		block->range(upper, lower) = buffer_in[i];
	}
}

//inline void read_block_dangling(input_block block, dangling_type buffer_out[AP_UINT_BITWIDTH]) {
//#pragma HLS INLINE
//	READ_BLOCK: for (int j = 0; j < AP_UINT_BITWIDTH; ++j) {
//#pragma HLS unroll
//		buffer_out[j] = block.bit(j);
//	}
//}

inline void read_block_float(input_block block, float_type buffer_out[BUFFER_SIZE]) {
#pragma HLS INLINE
	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		unsigned int block_curr = block.range(upper_range, lower_range);
		buffer_out[j] = *((float_type *) &block_curr);
	}
}





inline void read_packet_real_inout(real_type_inout block[BUFFER_SIZE], real_type buffer_out[BUFFER_SIZE]) {
#pragma HLS INLINE
	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		buffer_out[j] = (real_type) block[j];
	}
}

//inline void read_packet_real(input_block_real block, real_type buffer_out[BUFFER_SIZE]) {
//#pragma HLS INLINE
//	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
//#pragma HLS UNROLL
//		buffer_out[j] = block[j];
//	}
//}

//inline void write_packet_real_inout(input_block_real_inout& block, real_type buffer_in[BUFFER_SIZE]) {
//#pragma HLS INLINE
//	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
//#pragma HLS UNROLL
//		block[j] = (real_type_inout) buffer_in[j];
//	}
//}

//inline void write_packet_real(input_block_real& block, real_type buffer_in[BUFFER_SIZE]) {
//#pragma HLS INLINE
//	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
//#pragma HLS UNROLL
//		block[j] = buffer_in[j];
//	}
//}




// Write 16 values to a 512 bits block, taking them from a buffer of the specified type;
inline void write_block_float(input_block* block, float_type buffer_in[BUFFER_SIZE]) {
#pragma HLS INLINE
	WRITE_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		float_type curr_val = buffer_in[j];
		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		block->range(upper_range, lower_range) = *((unsigned int *) &curr_val);

	}
}

inline void memcpy_buf_to_buf(input_block *dest, input_block *src) {
#pragma HLS INLINE
	MEMCPY: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int begin = FIXED_WIDTH * j;
		unsigned int end = FIXED_WIDTH * (j + 1) - 1;
		unsigned int value = src->range(end, begin);
		dest->range(end, begin) = value;
	}
}

template<typename T>
inline void write_packed_array(
		T* array_in,
		input_block* array_packed_out,
		index_type array_size,
		index_type array_packed_size,
		uint buffer_size = BUFFER_SIZE,
		uint bitwidth = FIXED_WIDTH) {
	for (uint i = 0; i < array_packed_size; i++) {
		input_block new_block = 0;
		for (uint j = 0; j < buffer_size; j++) {
			T curr_val = buffer_size * i + j < array_size ? array_in[buffer_size * i + j] : (T) 0;
			unsigned int lower_range = bitwidth * j;
			unsigned int upper_range = bitwidth * (j + 1) - 1;
			unsigned int curr_val_in = *((unsigned int *) &curr_val);
			new_block.range(upper_range, lower_range) = curr_val_in;
		}
		array_packed_out[i] = new_block;
	}
}

template<typename I, typename T>
inline void write_packed_array(
		I* array_in,
		T* array_packed_out,
		int_type array_size,
		int_type array_packed_size,
		uint buffer_size = BUFFER_SIZE,
		uint bitwidth = FIXED_WIDTH) {
	for (uint i = 0; i < array_packed_size; i++) {
		T new_block;
		for (uint j = 0; j < buffer_size; j++) {
			new_block[j] =  buffer_size * i + j < array_size ? array_in[buffer_size * i + j] : (I) 0;
		}
		array_packed_out[i] = new_block;
	}
}


// Write a matrix to packets, padding to 0 each row.
// For example, the matrix composed of column vectors [[1,2,3], [4,5,6], [7,8,9]]
// will become [[1,2,3,0], [4,5,6,0], [7,8,9,0]];
template<typename T>
inline void write_packed_matrix(
		T* array_in,
		input_block* array_packed_out,
		index_type num_columns,
		index_type num_rows,
		index_type array_packed_size,
		uint buffer_size = BUFFER_SIZE,
		uint bitwidth = FIXED_WIDTH) {
	for (uint c = 0; c < num_columns; c++) {
		int curr_start = c * num_rows; // Start of the current column in the input array;
		for (uint i = 0; i < array_packed_size; i++) {
			input_block new_block = 0;
			for (uint j = 0; j < buffer_size; j++) {
				T curr_val = ((buffer_size * i + j) < num_rows) ? array_in[(buffer_size * i + j) + curr_start] : (T) 0;
				unsigned int lower_range = bitwidth * j;
				unsigned int upper_range = bitwidth * (j + 1) - 1;
				unsigned int curr_val_in = *((unsigned int *) &curr_val);
				new_block.range(upper_range, lower_range) = curr_val_in;
			}
			array_packed_out[i + c * array_packed_size] = new_block;
		}
	}
}

template<typename T>
inline void read_packed_array(
		T* array_out,
		input_block* array_packed_in,
		index_type array_size,
		index_type array_packed_size,
		uint buffer_size = BUFFER_SIZE,
		uint bitwidth = FIXED_WIDTH) {
	for (uint i = 0; i < array_packed_size; i++) {
		input_block curr_block = array_packed_in[i];
		for (uint j = 0; j < buffer_size; j++) {
			if (buffer_size * i + j < array_size) {
				unsigned int lower_range = bitwidth * j;
				unsigned int upper_range = bitwidth * (j + 1) - 1;
				unsigned int val_curr_block = curr_block.range(upper_range, lower_range);
				array_out[buffer_size * i + j] = *((T*) &val_curr_block);
			}
		}
	}
}

template<typename T>
inline void read_packed_matrix(
		T* array_out,
		input_block* array_packed_in,
		index_type num_columns,
		index_type num_rows,
		index_type array_packed_size,
		uint buffer_size = BUFFER_SIZE,
		uint bitwidth = FIXED_WIDTH) {
	for (uint c = 0; c < num_columns; c++) {
		int curr_start = c * num_rows; // Start of the current column in the input array;
		for (uint i = 0; i < array_packed_size; i++) {
			input_block curr_block = array_packed_in[i + c * array_packed_size];
			for (uint j = 0; j < buffer_size; j++) {
				if (buffer_size * i + j < num_rows) {
					unsigned int lower_range = bitwidth * j;
					unsigned int upper_range = bitwidth * (j + 1) - 1;
					unsigned int val_curr_block = curr_block.range(upper_range, lower_range);
					array_out[buffer_size * i + j + curr_start] = *((T*) &val_curr_block);
				}
			}
		}
	}
}


inline float_type reduction_16(float_type input[16]) {
#pragma HLS INLINE
#pragma HLS array_partition variable=input complete
	float_type acc = 0.0;
	for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS UNROLL
		acc += input[i];
	}
	return acc;
}

inline float_type reduction(float_type input[BUFFER_SIZE]) {
#pragma HLS INLINE
#pragma HLS array_partition variable=input complete
	float_type acc = 0.0;
	for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS UNROLL
		acc += input[i];
	}
	return acc;
}

//////////////////////////////////
// TOP-K SPMV ////////////////////
//////////////////////////////////

inline void read_block_x(input_block block, x_bscsr buffer_out[BSCSR_PACKET_SIZE]) {
#pragma HLS INLINE
	READ_BLOCK: for (int j = 0; j < BSCSR_PACKET_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int lower_x_range = AP_INT_ROW_BITWIDTH * j;
		unsigned int upper_x_range = AP_INT_ROW_BITWIDTH * (j + 1) -1;
		unsigned int block_curr = block.range(upper_x_range, lower_x_range);
		buffer_out[j] = *((x_bscsr *) &block_curr);
	}
}

inline void read_block_y(input_block block, y_bscsr buffer_out[BSCSR_PACKET_SIZE]) {
#pragma HLS INLINE
	READ_BLOCK: for (int j = 0; j < BSCSR_PACKET_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int lower_y_range = (AP_INT_ROW_BITWIDTH * BSCSR_PACKET_SIZE) + AP_INT_COL_BITWIDTH * j;
		unsigned int upper_y_range = (AP_INT_ROW_BITWIDTH * BSCSR_PACKET_SIZE) + AP_INT_COL_BITWIDTH * (j + 1) - 1;
		unsigned int block_curr = block.range(upper_y_range, lower_y_range);
		buffer_out[j] = *((y_bscsr *) &block_curr);
	}
}

inline void read_block_val(input_block block, real_type buffer_out[BSCSR_PACKET_SIZE]) {
#pragma HLS INLINE
	READ_BLOCK: for (int j = 0; j < BSCSR_PACKET_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int lower_val_range = ((AP_INT_ROW_BITWIDTH + AP_INT_COL_BITWIDTH) * (BSCSR_PACKET_SIZE)) + AP_INT_VAL_BITWIDTH * j;
		unsigned int upper_val_range = ((AP_INT_ROW_BITWIDTH + AP_INT_COL_BITWIDTH) * (BSCSR_PACKET_SIZE)) + AP_INT_VAL_BITWIDTH * (j + 1) -1;
		unsigned int block_curr = block.range(upper_val_range, lower_val_range);
		buffer_out[j] = *((real_type *) &block_curr);
	}
}

inline void read_block_xf(input_block block, bool_type buffer_out[1]) {
#pragma HLS INLINE
//	unsigned int lower_xf_range = 511;
//	unsigned int upper_xf_range = 511;
//	unsigned int block_curr = block.range(upper_xf_range, lower_xf_range);
	unsigned int bit_curr = block.bit(BSCSR_PORT_BITWIDTH - 1);
	buffer_out[0] = *((bool_type *) &bit_curr);
}

// Write BSCSR_PACKET_SIZE values to a 512 bits block, taking them from a buffer of the specified type;
inline void write_block_x(input_block* block, int_type buffer_in[BSCSR_PACKET_SIZE]) {
#pragma HLS INLINE
	WRITE_BLOCK: for (int j = 0; j < BSCSR_PACKET_SIZE; ++j) {
#pragma HLS UNROLL
		int_type curr_val = buffer_in[j];
		unsigned int lower_x_range = AP_INT_ROW_BITWIDTH * j;
		unsigned int upper_x_range = AP_INT_ROW_BITWIDTH * (j + 1) -1;
		block->range(upper_x_range, lower_x_range) = *((x_bscsr *) &curr_val);
	}
}

// Write BSCSR_PACKET_SIZE values to a 512 bits block, taking them from a buffer of the specified type;
inline void write_block_y(input_block* block, int_type buffer_in[BSCSR_PACKET_SIZE]) {
#pragma HLS INLINE
	WRITE_BLOCK: for (int j = 0; j < BSCSR_PACKET_SIZE; ++j) {
#pragma HLS UNROLL
		int_type curr_val = buffer_in[j];
		unsigned int lower_y_range = (AP_INT_ROW_BITWIDTH * BSCSR_PACKET_SIZE) + AP_INT_COL_BITWIDTH * j;
		unsigned int upper_y_range = (AP_INT_ROW_BITWIDTH * BSCSR_PACKET_SIZE) + AP_INT_COL_BITWIDTH * (j + 1) -1;
		block->range(upper_y_range, lower_y_range) = *((y_bscsr *) &curr_val);
	}
}

// Write BSCSR_PACKET_SIZE values to a 512 bits block, taking them from a buffer of the specified type;
inline void write_block_val(input_block* block, real_type_inout buffer_in[BSCSR_PACKET_SIZE]) {
#pragma HLS INLINE
	WRITE_BLOCK: for (int j = 0; j < BSCSR_PACKET_SIZE; ++j) {
#pragma HLS UNROLL
#if USE_FLOAT
		real_type curr_val = (real_type) buffer_in[j];
#else
		real_type curr_val = (real_type) buffer_in[j].to_float();
#endif
		unsigned int lower_val_range = ((AP_INT_ROW_BITWIDTH + AP_INT_COL_BITWIDTH) * (BSCSR_PACKET_SIZE)) + AP_INT_VAL_BITWIDTH * j;
		unsigned int upper_val_range = ((AP_INT_ROW_BITWIDTH + AP_INT_COL_BITWIDTH) * (BSCSR_PACKET_SIZE)) + AP_INT_VAL_BITWIDTH * (j + 1) -1;
		block->range(upper_val_range, lower_val_range) = *((val_bscsr *) &curr_val);
	}
}

inline void write_block_vec(vec_real_inout_bscsr* block, real_type_inout buffer_in[BSCSR_PACKET_SIZE]) {
#pragma HLS INLINE
	WRITE_BLOCK: for (int j = 0; j < BSCSR_PACKET_SIZE; ++j) {
#pragma HLS UNROLL
		real_type_inout curr_val = buffer_in[j];
		unsigned int lower_val_range = FIXED_WIDTH_OUT * j;
		unsigned int upper_val_range = FIXED_WIDTH_OUT * (j + 1) - 1;
		block->range(upper_val_range, lower_val_range) = *((unsigned int *) &curr_val);
	}
}


// Write BSCSR_PACKET_SIZE values to a 512 bits block, taking them from a buffer of the specified type;
inline void write_block_xf(input_block* block, bool_type buffer_in[1]) {
#pragma HLS INLINE
//	unsigned int lower_xf_range = 511;
//	unsigned int upper_xf_range = 511;
//	block->range(upper_xf_range, lower_xf_range) = buffer_in[0];
	block->bit(BSCSR_PORT_BITWIDTH - 1) = buffer_in[0];
}
