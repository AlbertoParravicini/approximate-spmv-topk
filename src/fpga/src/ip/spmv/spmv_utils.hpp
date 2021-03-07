#pragma once

#include "../fpga_types.hpp"
#include "../fpga_utils.hpp"
//#include <iostream>
extern "C" {

inline void read_block_vec(input_block block,
		float_type buffer_out[BUFFER_SIZE], float_type vec[MAX_ROWS]) {
#pragma HLS INLINE

	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		unsigned int block_curr = block.range(upper_range, lower_range);
		index_type temp_ind = *((index_type *) &block_curr);
		buffer_out[j] = vec[temp_ind];
	}
}

inline void read_block_vec_gmem(input_block block,
		float_type buffer_out[BUFFER_SIZE], input_block *vec) {
#pragma HLS INLINE

	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		unsigned int lower_range = FIXED_WIDTH * j;
		unsigned int upper_range = FIXED_WIDTH * (j + 1) - 1;
		unsigned int block_curr = block.range(upper_range, lower_range);
		index_type temp_ind = *((index_type *) &block_curr);

		// Compute the block of "vec" where the required index is;
		index_type block_ind = temp_ind / BUFFER_SIZE;
		// Compute the position in the block where the required index is;
		index_type position_in_block_ind = temp_ind % BUFFER_SIZE;

		// Load the required "vec" value from the input block;
		input_block vec_curr_block = vec[block_ind];
		lower_range = FIXED_WIDTH * position_in_block_ind;
		upper_range = FIXED_WIDTH * (position_in_block_ind + 1) - 1;
		unsigned int required_vec_val = vec_curr_block.range(upper_range,
				lower_range);
		buffer_out[j] = *((float_type *) &required_vec_val);
	}
}

inline void read_block_vec_gmem_parallel(int_type block[BUFFER_SIZE],
		float_type buffer_out[BUFFER_SIZE], input_block *vec0,
		input_block *vec1, input_block *vec2, input_block *vec3,
		input_block *vec4, input_block *vec5, input_block *vec6,
		input_block *vec7, input_block *vec8, input_block *vec9,
		input_block *vec10, input_block *vec11, input_block *vec12,
		input_block *vec13, input_block *vec14, input_block *vec15) {

#pragma HLS ARRAY_PARTITION variable=buffer_out complete dim=1
#pragma HLS INLINE

	READ_BLOCK: for (int j = 0; j < BUFFER_SIZE; ++j) {
#pragma HLS UNROLL
		index_type temp_ind = block[j];

		// Compute the block of "vec" where the required index is;
		index_type block_ind = temp_ind / BUFFER_SIZE;
		// Compute the position in the block where the required index is;
		index_type position_in_block_ind = temp_ind % BUFFER_SIZE;

		// Load the required "vec" value from the input block;
		///input_block vec_curr_block = vec[block_ind];

		input_block vec_curr_block;
		switch (j % 16) {
		case 0:
			vec_curr_block = vec0[block_ind];
			break;
		case 1:
			vec_curr_block = vec1[block_ind];
			break;
		case 2:
			vec_curr_block = vec2[block_ind];
			break;
		case 3:
			vec_curr_block = vec3[block_ind];
			break;
		case 4:
			vec_curr_block = vec4[block_ind];
			break;
		case 5:
			vec_curr_block = vec5[block_ind];
			break;
		case 6:
			vec_curr_block = vec6[block_ind];
			break;
		case 7:
			vec_curr_block = vec7[block_ind];
			break;
		case 8:
			vec_curr_block = vec8[block_ind];
			break;
		case 9:
			vec_curr_block = vec9[block_ind];
			break;
		case 10:
			vec_curr_block = vec10[block_ind];
			break;
		case 11:
			vec_curr_block = vec11[block_ind];
			break;
		case 12:
			vec_curr_block = vec12[block_ind];
			break;
		case 13:
			vec_curr_block = vec13[block_ind];
			break;
		case 14:
			vec_curr_block = vec14[block_ind];
			break;
		case 15:
			vec_curr_block = vec15[block_ind];
			break;
		}

		unsigned int lower_range = FIXED_WIDTH * position_in_block_ind;
		unsigned int upper_range = FIXED_WIDTH * (position_in_block_ind + 1)
				- 1;
		unsigned int required_vec_val = vec_curr_block.range(upper_range,
				lower_range);
		buffer_out[j] = *((float_type *) &required_vec_val);
	}
}

inline void read_block_vec_uram_parallel_bscsr(y_bscsr* block_in,
	input_packet_real_bscsr &buffer_out,real_type vec[BSCSR_PACKET_SIZE][MAX_COLS]) {

#pragma HLS INLINE

	READ_BLOCK: for (int j = 0; j < BSCSR_PACKET_SIZE; j++) {
#pragma HLS UNROLL
		int_type temp_ind = block_in[j];

		// Load the required "vec" value from the input block;
		buffer_out[j] = vec[j / 2][temp_ind];
	}
}





inline void read_block_vec_gmem_parallel_coo_16(
		input_packet_int_y_bscsr &block_in,
		input_packet_real_bscsr &buffer_out, real_type_inout *vec0,
		real_type_inout *vec1, real_type_inout *vec2, real_type_inout *vec3,
		real_type_inout *vec4, real_type_inout *vec5, real_type_inout *vec6,
		real_type_inout *vec7, real_type_inout *vec8, real_type_inout *vec9,
		real_type_inout *vec10, real_type_inout *vec11, real_type_inout *vec12,
		real_type_inout *vec13, real_type_inout *vec14) {

#pragma HLS INLINE

	READ_BLOCK: for (int j = 0; j < BSCSR_PACKET_SIZE; j++) {
#pragma HLS UNROLL
		int_type temp_ind = block_in[j];

		// Load the required "vec" value from the input block;
		real_type_inout vec_curr_value;
		switch (j % 15) {
		case 0:
			vec_curr_value = vec0[temp_ind];
			break;
		case 1:
			vec_curr_value = vec1[temp_ind];
			break;
		case 2:
			vec_curr_value = vec2[temp_ind];
			break;
		case 3:
			vec_curr_value = vec3[temp_ind];
			break;
		case 4:
			vec_curr_value = vec4[temp_ind];
			break;
		case 5:
			vec_curr_value = vec5[temp_ind];
			break;
		case 6:
			vec_curr_value = vec6[temp_ind];
			break;
		case 7:
			vec_curr_value = vec7[temp_ind];
			break;
		case 8:
			vec_curr_value = vec8[temp_ind];
			break;
		case 9:
			vec_curr_value = vec9[temp_ind];
			break;
		case 10:
			vec_curr_value = vec10[temp_ind];
			break;
		case 11:
			vec_curr_value = vec11[temp_ind];
			break;
		case 12:
			vec_curr_value = vec12[temp_ind];
			break;
		case 13:
			vec_curr_value = vec13[temp_ind];
			break;
		case 14:
			vec_curr_value = vec14[temp_ind];
			break;
		}
		buffer_out[j] = (real_type) vec_curr_value;
	}
}

inline void read_block_vec_gmem_parallel_coo(input_packet_int_coo &block_in,
		input_packet_real_coo &buffer_out, real_type_inout *vec0,
		real_type_inout *vec1, real_type_inout *vec2, real_type_inout *vec3,
		real_type_inout *vec4) {
#pragma HLS INLINE

	READ_BLOCK: for (int j = 0; j < COO_PACKET_SIZE; j++) {
#pragma HLS UNROLL
		int_type temp_ind = block_in[j];

		// Load the required "vec" value from the input block;
		real_type_inout vec_curr_value;
		switch (j % 5) {
		case 0:
			vec_curr_value = vec0[temp_ind];
			break;
		case 1:
			vec_curr_value = vec1[temp_ind];
			break;
		case 2:
			vec_curr_value = vec2[temp_ind];
			break;
		case 3:
			vec_curr_value = vec3[temp_ind];
			break;
		case 4:
			vec_curr_value = vec4[temp_ind];
			break;
		}
		buffer_out[j] = (real_type_inout) vec_curr_value;
	}
}

inline void read_block_vec_gmem_parallel_coo_4_cache(input_packet_int_coo_4 &block_in,
		input_packet_real_coo_4 &buffer_out, real_type_inout *vec0,
		real_type_inout *vec1, real_type_inout *vec2, real_type_inout *vec3, int_type cache_indices[COO_PACKET_SIZE_4][CACHE_SIZE], real_type cache_values[COO_PACKET_SIZE_4][CACHE_SIZE], int_type *cache_counter, int_type *tot_counter) {
#pragma HLS INLINE

	READ_BLOCK: for (int j = 0; j < COO_PACKET_SIZE_4; j++) {
#pragma HLS UNROLL
		int_type temp_ind = block_in[j];
		int_type cache_index = temp_ind % CACHE_SIZE;

		tot_counter[0]++;
		// Check if the value is in cache;
		int_type loaded_cache_index = cache_indices[j][cache_index];
		real_type vec_curr_value = cache_values[j][cache_index];
		if (loaded_cache_index == temp_ind) {
			cache_counter[0]++;
		} else {
			// Load the required "vec" value from the input block;
			switch (j % COO_PACKET_SIZE_4) {
			case 0:
				vec_curr_value = (real_type) vec0[temp_ind];
				break;
			case 1:
				vec_curr_value = (real_type) vec1[temp_ind];
				break;
			case 2:
				vec_curr_value = (real_type) vec2[temp_ind];
				break;
			case 3:
				vec_curr_value = (real_type) vec3[temp_ind];
				break;
			}
		}
		cache_indices[j][cache_index] = temp_ind;
		cache_values[j][cache_index] = vec_curr_value;
		buffer_out[j] = vec_curr_value;
	}
}

inline void read_block_vec_gmem_parallel_coo_4_inner_data(int_type index, input_packet_int_coo_4 &block_in,
		real_type buffer_out[COO_PACKET_SIZE_4], real_type_inout *vec0,
		real_type_inout *vec1, real_type_inout *vec2, real_type_inout *vec3) {
#pragma HLS INLINE

	int_type temp_ind = block_in[index];
	real_type vec_curr_value;
	// Load the required "vec" value from the input block;
	switch (index % COO_PACKET_SIZE_4) {
	case 0:
		vec_curr_value = (real_type) vec0[temp_ind];
		break;
	case 1:
		vec_curr_value = (real_type) vec1[temp_ind];
		break;
	case 2:
		vec_curr_value = (real_type) vec2[temp_ind];
		break;
	case 3:
		vec_curr_value = (real_type) vec3[temp_ind];
		break;
	}
	buffer_out[index] = vec_curr_value;
}

inline void read_block_vec_gmem_parallel_coo_4_inner(int_type index, input_packet_int_coo_4 &block_in,
		input_packet_real_coo_4 &buffer_out, real_type_inout *vec0,
		real_type_inout *vec1, real_type_inout *vec2, real_type_inout *vec3) {
#pragma HLS INLINE
	read_block_vec_gmem_parallel_coo_4_inner_data(index, block_in, buffer_out.data, vec0, vec1, vec2, vec3);
}

inline void read_block_vec_gmem_parallel_coo_4(input_packet_int_coo_4 &block_in,
		input_packet_real_coo_4 &buffer_out, real_type_inout *vec0,
		real_type_inout *vec1, real_type_inout *vec2, real_type_inout *vec3) {
#pragma HLS INLINE

	READ_BLOCK: for (int i = 0; i < COO_PACKET_SIZE_4; i++) {
#pragma HLS UNROLL
		read_block_vec_gmem_parallel_coo_4_inner(i, block_in, buffer_out, vec0, vec1, vec2, vec3);
	}
}

inline void compute_coalesced_read_indices(input_packet_int_coo_4 &block_in, input_packet_int_coo_4 &coalesced_indices) {
#pragma HLS INLINE

	// First, compare all block_in indices to find accesses to the same memory region.
	// If two positions block_in[i] and block_in[j] are identical, store the biggest values j (j is always > i).
	// For each identical memory access, we want to access memory just once.
	// The index i used to perform this access is the biggest among the ones that identify a set of identical memory accesses.
	// Memory accesses that have a duplicate are marked with the largest index that identifies a set of identical accesses;
	for (int_type i = 0; i < COO_PACKET_SIZE_4; i++) {
#pragma HLS unroll
		int_type copy_from_index = 0;
		for (int_type j = 0; j < COO_PACKET_SIZE_4; j++) {
#pragma HLS unroll
			copy_from_index = ((j > i) & (block_in[i] == block_in[j])) ? ((j > copy_from_index) ? j : copy_from_index) : 0;
		}
		coalesced_indices[i] = copy_from_index;
	}
}

inline void read_block_vec_gmem_parallel_coo_4_mshr(input_packet_int_coo_4 &block_in,
		input_packet_real_coo_4 &buffer_out, input_packet_int_coo_4 &coalesced_indices,
		real_type_inout *vec0, real_type_inout *vec1, real_type_inout *vec2, real_type_inout *vec3,
		real_type coalescing_buffer_temp[COO_PACKET_SIZE_4]) {
#pragma HLS INLINE

//	for (int_type i = 0; i < COO_PACKET_SIZE_4; i++) {
//		std::cout << "block[" << i << "]=" << block_in[i] << " - R=" << coalescing_result[i] << ", ";
//	}
//	std::cout << std::endl;

	// Load values from memory, for each memory access which is not marked as duplicate of another access;
	for (int_type i = 0; i < COO_PACKET_SIZE_4; i++) {
#pragma HLS unroll
		if (coalesced_indices[i] == 0) {
			// Handle vector copies with a switch here;
			read_block_vec_gmem_parallel_coo_4_inner_data(i, block_in, coalescing_buffer_temp, vec0, vec1, vec2, vec3);
		}
	}
//	for (int_type i = 0; i < COO_PACKET_SIZE_4; i++) {
//		std::cout << "    vec[" << block_in[i] << "]=" << buffer_out[i].to_float() << ", ";
//	}
//	std::cout << std::endl;
	// Propagate copies, by loading the value from the index stored in the coalescing vector;
	for (int i = 0; i < COO_PACKET_SIZE_4; i++) {
#pragma HLS unroll
		if (coalesced_indices[i] != 0) {
			buffer_out[i] = coalescing_buffer_temp[coalesced_indices[i]];
		} else {
			buffer_out[i] = coalescing_buffer_temp[i];
		}
	}
//	for (int_type i = 0; i < COO_PACKET_SIZE_4; i++) {
//		std::cout << "    vec[" << block_in[i] << "]=" << buffer_out[i].to_float() << ", ";
//	}
//	std::cout << std::endl;
}

inline void read_block_vec_gmem_parallel_csr(input_packet_int_csr &block_in,
		input_packet_real_csr &buffer_out, real_type_inout *vec0,
		real_type_inout *vec1, real_type_inout *vec2, real_type_inout *vec3,
		real_type_inout *vec4, real_type_inout *vec5, real_type_inout *vec6,
		real_type_inout *vec7) {
#pragma HLS INLINE

	READ_BLOCK: for (int j = 0; j < CSR_PACKET_SIZE; j++) {
#pragma HLS UNROLL
		int_type temp_ind = block_in[j];

		// Load the required "vec" value from the input block;
		real_type_inout vec_curr_value;
		switch (j % 8) {
		case 0:
			vec_curr_value = vec0[temp_ind];
			break;
		case 1:
			vec_curr_value = vec1[temp_ind];
			break;
		case 2:
			vec_curr_value = vec2[temp_ind];
			break;
		case 3:
			vec_curr_value = vec3[temp_ind];
			break;
		case 4:
			vec_curr_value = vec4[temp_ind];
			break;
		case 5:
			vec_curr_value = vec5[temp_ind];
			break;
		case 6:
			vec_curr_value = vec6[temp_ind];
			break;
		case 7:
			vec_curr_value = vec7[temp_ind];
			break;

		}
		buffer_out[j] = (real_type) vec_curr_value;
	}
}

inline void reset_buffer(float_type buf[BUFFER_SIZE]) {
#pragma HLS INLINE
	for (int i = 0; i < BUFFER_SIZE; ++i) {
#pragma HLS unroll
		buf[i] = 0.0;
	}
}

inline void reset_large_buffer(float_type buf[2 * BUFFER_SIZE]) {
#pragma HLS inline
	for (int i = 0; i < 2 * BUFFER_SIZE; ++i) {
#pragma HLS unroll
		buf[i] = 0.0;
	}
}

}
