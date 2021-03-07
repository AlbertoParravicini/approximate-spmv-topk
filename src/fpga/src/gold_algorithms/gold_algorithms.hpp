#pragma once

#include "../ip/coo_fpga.hpp"

template<typename I, typename V>
inline void spmv_gold(I *ptr, I *idx, V *val, I N, V *result, V *vec) {
	I begin = ptr[0];
	for (I i = 0; i < N; ++i) {
		I end = ptr[i + 1];
		V acc = 0.0;

		for (I j = begin; j < end; ++j) {
			acc += val[j] * vec[idx[j]];
		}
		result[i] = acc;
		begin = end;
	}
}

template<typename I, typename V>
inline void multi_spmv_gold(I *ptr, I *idx, V *val, I N, I M, V *result, V *vec) {
	I begin = ptr[0];
	for (I i = 0; i < N; ++i) {
		I end = ptr[i + 1];

		for (I k = 0; k < M; k++) {
			V acc = 0.0;
			for (I j = begin; j < end; ++j) {
				acc += val[j] * vec[idx[j] + k * N];
			}
			result[k * N + i] = acc;
		}
		begin = end;
	}
}


template<typename I, typename V>
inline void spmv_coo_gold(coo_fixed_fpga_t<I, V> &csc, V *result, V *vec) {
	// Scatter "vec" values across a vector of size E;
	std::vector<V> vec_scattered(csc.E);
	for (int i = 0; i < csc.E; i++) {
		vec_scattered[i] = vec[csc.end[i]];
	}

	// Main computation;
	for (int i = 0; i < csc.E; i++) {
		result[csc.start[i]] += csc.val[i] * vec_scattered[i];
		// Result is BRAM with cyclic 16 -> a single packet cant access the same address twice
	}
}

template<typename I, typename V>
inline void spmv_coo_gold(coo_t<I, V> &coo, V *result, V *vec) {
	// Scatter "vec" values across a vector of size E;
	std::vector<V> vec_scattered(coo.num_nnz);
	for (int i = 0; i < coo.num_nnz; i++) {
		vec_scattered[i] = vec[coo.end[i]];
	}

	// Main computation;
	for (int i = 0; i < coo.num_nnz; i++) {
		result[coo.start[i]] += coo.val[i] * vec_scattered[i];
	}
}

template<typename I, typename V>
inline void spmv_coo_gold2(coo_fixed_fpga_t<I, V> &csc, V *result, V *vec) {
	// Scatter "vec" values across a vector of size E;
	std::vector<V> vec_scattered(csc.E);
	for (int i = 0; i < csc.E; i++) {
		vec_scattered[i] = vec[csc.end[i]];
	}

	// Store temporary results;
	std::vector<V> temp_res(csc.E);

	// Main computation;
	for (int i = 0; i < csc.E; i++) {
		temp_res[i] = csc.val[i] * vec_scattered[i];
	}

	// Gather;
	for (int i = 0; i < csc.E; i++) {
		result[csc.start[i]] += temp_res[i];
	}
}

template<typename I, typename V>
inline void spmv_coo_gold3(coo_fixed_fpga_t<I, V> &csc, V *result, V *vec) {

	int B = 2;

	// Scatter "vec" values across a vector of size E;
	std::vector<V> vec_scattered(csc.E_fixed);
	for (int i = 0; i < csc.E_fixed; i++) {
		vec_scattered[i] = vec[csc.end[i]];
	}

	// Main computation;
	for (int i = 0; i < (csc.E_fixed + B - 1) / B; i++) {

		index_type local_start[B];
		V res_temp[B];
		V res_fin[B];
		// Reset buffers;
		for (int j = 0; j < B; j++) {
			local_start[j] = 0;
			res_temp[j] = 0;
			res_fin[j] = 0;
		}

		// Compute values for a packet;
		for (int j = 0; j < B; j++) {
			if (i * B + j < csc.E_fixed) {
				local_start[j] = csc.start[i * B + j];
				res_temp[j] = csc.val[i * B + j] * vec_scattered[i * B + j];
			}
		}

		// Use B reductions;
		index_type min_id = local_start[0];
		for (int j = 0; j < B; j++) {
			index_type curr = j + min_id; // Identify which vertex is considered in this reduction;
			// Reduction;
			for (int q = 0; q < B; q++) {
				res_fin[j] += res_temp[q] * (curr == local_start[q] ? 1 : 0);
			}
		}

		for (int j = 0; j < B; j++) {
			result[j + min_id] += res_fin[j];
		}
//		for (int j = 0; j < B; j++) {
//			result[local_start[j]] += res_temp[j];
//		}
	}
}

template<typename I, typename V>
inline void spmv_coo_gold4(coo_t<I, V> &coo, V *result, V *vec) {

	int B = 2;

	// Scatter "vec" values across a vector of size E;
	std::vector<V> vec_scattered(coo.num_nnz);
	for (int i = 0; i < coo.num_nnz; i++) {
		vec_scattered[i] = vec[coo.end[i]];
	}

	// Main computation;
	for (int i = 0; i < (coo.num_nnz + B - 1) / B; i++) {

		index_type local_start[B];
		V res_temp[B];
		V res_fin[B];
		// Reset buffers;
		for (int j = 0; j < B; j++) {
			local_start[j] = 0;
			res_temp[j] = 0;
			res_fin[j] = 0;
		}

		// Compute values for a packet;
		for (int j = 0; j < B; j++) {
			if (i * B + j < coo.num_nnz) {
				local_start[j] = coo.start[i * B + j];
				res_temp[j] = coo.val[i * B + j] * vec_scattered[i * B + j];
			}
		}

		// Use B reductions;
		index_type min_id = local_start[0];
		for (int j = 0; j < B; j++) {
			index_type curr = j + min_id; // Identify which vertex is considered in this reduction;
			// Reduction;
			for (int q = 0; q < B; q++) {
				res_fin[j] += res_temp[q] * (curr == local_start[q] ? 1 : 0);
			}
		}

		for (int j = 0; j < B; j++) {
			result[j + min_id] += res_fin[j];
		}
	}
}

template<typename I, typename V>
inline void spmv_coo_gold_top_k(coo_t<I, V> &coo, V *vec, int k, I* res_idx, V* res_val) {
	// Scatter "vec" values across a vector of size E;
	std::vector<V> vec_scattered(coo.num_nnz);
	for (int i = 0; i < coo.num_nnz; i++) {
		vec_scattered[i] = vec[coo.end[i]];
	}

	// Track the current row being processed;
	I curr_row = coo.start[0];
	V curr_row_output = (V) 0.0;
	// Track the worst result in the Top-K;
	I curr_worst_idx = 0;
	V curr_worst_val = (V) 0.0;

	// Initialize output arrays;
	for (int i = 0; i < k; i++) {
		res_idx[i] = 0;
		res_val[i] = (V) 0.0;
	}

	// Main computation;
	for (int i = 0; i < coo.num_nnz; i++) {
		I curr_row_tmp = coo.start[i];

		V pointwise_contribution = coo.val[i] * vec_scattered[i];

		if (curr_row_tmp == curr_row) {
			// Accumulate result;
			curr_row_output += pointwise_contribution;
		} else {
			// Check if the previous row is in the Top-K;
			if (curr_row_output >= curr_worst_val) {
				// Replace the current worst with the new result;
				res_idx[curr_worst_idx] = curr_row;
				res_val[curr_worst_idx] = curr_row_output;
				// Find the new worst (linear scan, could be done in constant time by keeping track of the 2nd worst);
				I worst_val_idx = 0;
				V worst_val_tmp = res_val[0];
				for (int j = 0; j < k; j++) {
					worst_val_idx = (res_val[j] < worst_val_tmp) ? j : worst_val_idx;
					worst_val_tmp = (res_val[j] < worst_val_tmp) ? res_val[j] : worst_val_tmp;
				}
				curr_worst_idx = worst_val_idx;
				curr_worst_val = worst_val_tmp;
			}

			// Update book-keeping variables;
			curr_row = curr_row_tmp;
			curr_row_output = pointwise_contribution;
		}
	}
	// Handle the final row;
	if (curr_row_output >= curr_worst_val) {
		// Replace the current worst with the new result;
		res_idx[curr_worst_idx] = curr_row;
		res_val[curr_worst_idx] = curr_row_output;
	}
}

template<typename V>
inline void update_top_k(
		index_type* res_idx,
		V* res_val,
		index_type &curr_worst_idx,
		V &curr_worst_val,
		index_type curr_row,
		V curr_row_output,
		int k) {
	// Check if the previous row is in the Top-K;
	if (curr_row_output >= curr_worst_val) {
		// Replace the current worst with the new result;
		res_idx[curr_worst_idx] = curr_row;
		res_val[curr_worst_idx] = curr_row_output;
		// Find the new worst (linear scan, could be done in constant time by keeping track of the 2nd worst);
		int worst_val_idx = 0;
		V worst_val_tmp = res_val[0];
		for (int j = 0; j < k; j++) {
			worst_val_idx = (res_val[j] < worst_val_tmp) ? j : worst_val_idx;
			worst_val_tmp = (res_val[j] < worst_val_tmp) ? res_val[j] : worst_val_tmp;
		}
		curr_worst_idx = worst_val_idx;
		curr_worst_val = worst_val_tmp;
	}
}

#define B 4

template<typename I, typename V>
inline void spmv_coo_gold_top_k_packet(coo_t<I, V> &coo, V *vec, int k, index_type* res_idx, V* res_val) {

//	index_type B = 4;

	// Track the worst result in the Top-K;
	index_type curr_worst_idx = 0;
	V curr_worst_val = (V) 0.0;

	// Initialize output arrays;
	for (index_type i = 0; i < k; i++) {
		res_idx[i] = 0;
		res_val[i] = (V) 0.0;
	}

	// Scatter "vec" values across a vector of size E;
	std::vector<V> vec_scattered(coo.num_nnz);
	for (index_type i = 0; i < coo.num_nnz; i++) {
		vec_scattered[i] = vec[coo.end[i]];
	}

	// Track the last row found in a packet, to check if the row is split between contiguous packets;
	index_type last_packet_row = coo.start[0];
	V last_packet_row_output = (V) 0.0;

	// Main computation;
	for (index_type i = 0; i < (coo.num_nnz + B - 1) / B; i++) {

		// Local buffers
		index_type local_start[B] = { 0 };
		V res_temp[B] = { (V) 0.0 };
		V res_fin[B] = { (V) 0.0 };
		bool finished_rows[B] = { false };

		// Compute values for a packet;
		for (index_type j = 0; j < B; j++) {
			if (i * B + j < coo.num_nnz) {
				local_start[j] = coo.start[i * B + j];
				res_temp[j] = coo.val[i * B + j] * vec_scattered[i * B + j];
			}
		}

		// Use B reductions to obtain the aggregate contributions of the current packet;
		index_type min_id = local_start[0];
		for (index_type j = 0; j < B; j++) {
			index_type curr = j + min_id; // Identify which vertex is considered in this reduction;
			// Reduction;
			for (index_type q = 0; q < B; q++) {
				res_fin[j] += res_temp[q] * (curr == local_start[q] ? 1 : 0);
			}
		}

		// Check which rows within the current packet we have finished processing;
		int finished_rows_num = 0;
		for (index_type j = 0; j < B - 1; j++) {
			if (local_start[j] != local_start[j + 1]) {
				finished_rows[finished_rows_num] = true;
				finished_rows_num++; // Handle the first packet row in a different way;
			}
		}

		// If the last row in the previous packet was split between packets, update the first result in this packet;
		if (last_packet_row == local_start[0]) {
			res_fin[0] += last_packet_row_output;
		} else {
			// Else (i.e. this packet starts with a new row), check if the previous row is in the Top-K;
			update_top_k(res_idx, res_val, curr_worst_idx, curr_worst_val, last_packet_row, last_packet_row_output, k);
		}

		// Update the Top-K values;
		for (index_type j = 0; j < B; j++) {
			if (finished_rows[j]) {
				update_top_k(res_idx, res_val, curr_worst_idx, curr_worst_val, local_start[0] + j, res_fin[j], k);
			}
		}

		// Book-keeping at the end of processing a packet;
		last_packet_row = local_start[B - 1];
		last_packet_row_output = res_fin[finished_rows_num];
	}
	// Handle the final row;
	if (last_packet_row_output >= curr_worst_val) {
		// Replace the current worst with the new result;
		res_idx[curr_worst_idx] = last_packet_row;
		res_val[curr_worst_idx] = last_packet_row_output;
	}
}

inline void euclidean_distance_gold(unsigned int N, float *result,
		float *a, float *b) {

	*result = 0;
	for (int i = 0; i < N; ++i) {
		float tmp = a[i] - b[i];
		tmp *= tmp;
		*result += tmp;
	}

}

inline void axpb_gold(unsigned int N, float *result, float *a,
		float *x, float *b) {

	float local_a = *a;
	float local_b = *b;

	for (int i = 0; i < N; ++i) {
		result[i] = local_a * x[i] + local_b;
	}
}

inline void dot_product_gold(unsigned int N, float *result, unsigned int *a,
		float *b) {

	*result = 0;

	for (int i = 0; i < N; ++i) {
		*result += a[i] * b[i];
	}
}

inline void pagerank_golden(unsigned int *ptr, unsigned int *idx, float *val,
		unsigned int *N, unsigned int *E, float *result, float *pr_vec,
		unsigned int *dangling_bitmap, float *tmp_pr, float *max_err,
		float *alpha, unsigned int *max_iter, unsigned int *iterations_to_convergence) {

	float ERR = *max_err;
	unsigned int ITER = *max_iter;
	unsigned int NUM_VERTICES = *N;
	float DANGLING_SCALE = *alpha/ NUM_VERTICES;
	float SHIFT_FACTOR = ((float) 1.0 - *alpha) / NUM_VERTICES;

	float err = 0.0;
	float dangling_contrib = 0.0;
	unsigned int iter = 0;
	bool converged = false;

	while(!converged && iter < ITER){

		spmv_gold(ptr, idx, val, NUM_VERTICES, tmp_pr, pr_vec);
		dot_product_gold(NUM_VERTICES, &dangling_contrib, dangling_bitmap, pr_vec);
		float shifting_factor = SHIFT_FACTOR + (DANGLING_SCALE * dangling_contrib);
		axpb_gold(NUM_VERTICES, tmp_pr, alpha, tmp_pr, &shifting_factor);

		euclidean_distance_gold(NUM_VERTICES, &err, tmp_pr, pr_vec);

		converged = err <= ERR;

		memcpy(pr_vec, tmp_pr, sizeof(float) * NUM_VERTICES);
		iter ++;

	}
	*iterations_to_convergence = iter;
	std::cout << "Pagerank golden converged after " << iter << " iterations." << std::endl;
	memcpy(result, pr_vec, sizeof(float) * NUM_VERTICES);

}

