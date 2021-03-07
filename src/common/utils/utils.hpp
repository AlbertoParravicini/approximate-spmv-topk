//
// Created by Francesco Sgherzi on 26/04/19.
//

#pragma once

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <random>
#include <time.h> /* time */
#include <set>
#include <vector>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include "mmio.hpp"
#include <chrono>
#include "../../fpga/src/aligned_allocator.h"
// #include "hls_math.h"
#include "../csc_matrix/csc_matrix.hpp"
#include "../types.hpp"

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

///////////////////////////////
///////////////////////////////

inline csc_t load_graph_csc(const std::string dir_path, bool debug = true);

///////////////////////////////
///////////////////////////////

typedef struct csv_results_t {

	size_t fixed_float_width;
	size_t fixed_float_scale;

	int n_edges;
	int n_vertices;

	double execution_time;
	double transfer_time;
	double accuracy;
	double normalized_dcg;
	double error_pct;

} csv_results_t;

template<typename T>
inline bool check_error(T *e, const T error, const unsigned DIMV) {
	for (int i = 0; i < DIMV; ++i) {
		if (e[i] > error)
			return false;
	}
	return true;
}

inline void random_ints(int *v, int n, int max = RAND_MAX) {
	for (int i = 0; i < n; i++) {
		v[i] = rand() % max;
	}
}

template<typename T>
inline std::string format_array(T *v, int n, uint max = 20) {
	int max_val = std::min(n, (int) max);
	std::string out = "[";
	for (int i = 0; i < max_val; i++) {
		out += std::to_string(v[i]);
		if (i < max_val - 1) {
			out += ", ";
		}
	}
	return out + "]";
}

template<typename T>
inline void print_array(T *v, int n, uint max = 20) {
	std::cout << format_array(v, n, max) << std::endl;
}

template<typename T>
inline std::string format_array(std::vector<T> &v, uint max = 20) {
	format_array(v.data(), v.size(), max);
}

template<typename T>
inline std::string format_array(std::vector<T, aligned_allocator<T>> &v, uint max = 20) {
	format_array(v.data(), v.size(), max);
}

template<typename T>
inline void print_array(std::vector<T> &v, uint max = 20) {
	print_array(v.data(), v.size(), max);
}

template<typename T>
inline void print_array_indexed(T *v, uint n, uint max = 20) {
	uint start1 = 0;
	uint start2 = 0;
	uint end1 = n;
	uint end2 = n;

	if (n > max) {
	    end1 = max / 2;
	    start2 = n - max / 2;
	}
	for (uint i = start1; i < end1; i++) {
		std::cout << i << ") " << v[i] << std::endl;
	}
	if (n > max) {
		std::cout << "[...]" << std::endl;
	    for (uint i = start2; i < end2; i++) {
	    	std::cout << i << ") " << v[i] << std::endl;
	    }
	}
	std::cout << std::endl;
}

template<typename T>
inline void print_array_indexed(std::vector<T> &v, uint max = 20) {
	print_array_indexed(v.data(), v.size(), max);
}

template<typename T>
inline void print_matrix_indexed(T *m, int dim_row, int dim_col, int max_c = 4, int max_r = 4) {
	// Print 2 decimal digits;
	std::cout << std::setprecision(4) << std::fixed;
	// Save the old flags to restore them later;
	std::ios::fmtflags old_settings = std::cout.flags();

	// dim-col=16, dim_row=5
	std::cout << "[" << std::endl;

	int R = std::min(max_r, dim_col);
	int C = std::min(max_c, dim_row);

	for (int r = 0; r < R; r++) {
		std::cout << r << " [";
		for (int c = 0; c < C; c++) {
			std::cout << m[dim_col * c + r] << (c < C - 1 ? ", " : "");
		}
		std::cout << "]" << (r < R - 1 ? "," : "") << std::endl;
	}

	std::cout << "]" << std::endl;
	// Reset printing format;
	std::cout.flags(old_settings);
}

template<typename T>
inline void print_matrix_indexed(std::vector<T> &m, int num_rows, int num_cols, int max_r = 4, int max_c = 4) {
	// Print 4 decimal digits;
	std::cout << std::setprecision(4) << std::fixed;
	// Save the old flags to restore them later;
	std::ios::fmtflags old_settings = std::cout.flags();

	int R = std::min(max_r, num_rows);
	int C = std::min(max_c, num_cols);

	for (int r = 0; r < R; r++) {
		std::cout << r << ") ";
		for (int c = 0; c < C; c++) {
			std::cout << m[c][r] << (c < C - 1 ? ", " : "");
		}
		std::cout << ";" << std::endl;
	}

	// Reset printing format;
	std::cout.flags(old_settings);
}

template<typename T>
inline void print_matrix_indexed(std::vector<std::vector<T>> &m, int max_r = 4, int max_c = 4) {
	// Print 4 decimal digits;
	std::cout << std::setprecision(4) << std::fixed;
	// Save the old flags to restore them later;
	std::ios::fmtflags old_settings = std::cout.flags();

	int num_cols = m.size();
	int num_rows = m[0].size();
	int R = std::min(max_r, num_rows);
	int C = std::min(max_c, num_cols);

	for (int r = 0; r < R; r++) {
		std::cout << r << ") ";
		for (int c = 0; c < C; c++) {
			std::cout << m[c][r] << (c < C - 1 ? ", " : "");
		}
		std::cout << ";" << std::endl;
	}

	// Reset printing format;
	std::cout.flags(old_settings);
}

#define ABS(X) (((X) > 0) ? (X) : (-1 * (X)))

template<typename T>
inline int check_array_equality(T *x, T *y, int n, float tol = 0.0000001f, bool debug = false, int max_print = 20) {
	int num_errors = 0;
	for (int i = 0; i < n; i++) {
		float diff = (float)((x[i] > y[i]) ? (x[i] - y[i]) : (y[i] - x[i]));
		if (diff > tol) {
			num_errors++;
			if (debug && num_errors < max_print) {
				std::cout << i << ") X: " << x[i] << ", Y: " << y[i] << ", diff: " << diff << std::endl;
			}
		}
	}
	return num_errors;
}

template<typename T>
inline bool check_equality(T x, T y, float tol = 0.0000001f, bool debug = false) {
	bool equal = true;

	float diff = std::abs(x - y);
	if (diff > tol) {
		equal = false;
		if (debug) {
			std::cout << "x: " << x << ", y: " << y << ", diff: " << diff << std::endl;
		}
	}
	return equal;
}


template<typename T>
inline void create_sample_vector(T *vector, int size, bool random = false, bool sum_to_one = true, bool norm_one = false, int seed = 0) {

	if (random) {
		std::random_device rd;
		std::mt19937 engine(seed == 0 ? rd(): seed);
		std::uniform_real_distribution<double> dist(0, 1);
		for (int i = 0; i < size; i++) {
			vector[i] = (T) dist(engine);
		}
	} else {
		for (int i = 0; i < size; i++) {
			vector[i] = 1;
		}
	}

	if (sum_to_one) {
		float sum = 0;
		for (int i = 0; i < size; i++) {
			sum += (float) vector[i];
		}
		for (int i = 0; i < size; i++) {
			vector[i] = (T) ((float) vector[i] / sum);
		}
	} else if (norm_one) {
		double sum = 0;
		for (int i = 0; i < size; i++) {
			sum += (float) vector[i] * (float) vector[i];
		}
		for (int i = 0; i < size; i++) {
			vector[i] = (T) ((float) vector[i] / sqrt(sum));
		}
	}
}

inline void normalize_vector(float *vector, int size) {
	float mean = 0;
	for (int i = 0; i < size; i++) {
		mean += vector[i];
	}
	mean /= size;
	for (int i = 0; i < size; i++) {
		vector[i] -= mean;
	}
}

/////////////////////////////
/////////////////////////////

inline void create_random_graph(std::vector<int> &ptr, std::vector<int> &idx, int max_degree = 10,
		bool avoid_self_edges = true) {
	srand(time(NULL));
	int N = ptr.size() - 1;
	// For each vertex, generate a random number of edges, with a given max degree;
	for (uint v = 1; v < ptr.size(); v++) {
		int num_edges = rand() % std::min(N, max_degree);
		// Generate edges;
		std::set<int> edge_set;
		for (int e = 0; e < num_edges; e++) {
			edge_set.insert(rand() % N);
		}
		// Avoid self-edges;
		if (avoid_self_edges) {
			edge_set.erase(v - 1);
		}
		for (auto e : edge_set) {
			idx.push_back(e);
		}
		ptr[v] = edge_set.size() + ptr[v - 1];
	}
}

template<typename T>
inline void print_graph(std::vector<T> &ptr, std::vector<T> &idx, int max_N = 20, int max_E = 20) {
	std::cout << "-) degree: " << ptr[0] << std::endl;
	for (int v = 1; v < std::min((int) ptr.size(), max_N); v++) {
		std::cout << v - 1 << ") degree: " << ptr[v] - ptr[v - 1] << ", edges: ";
		for (int e = 0; e < ptr[v] - ptr[v - 1]; e++) {
			if (e < max_E) {
				std::cout << idx[ptr[v - 1] + e] << ", ";
			}
		}
		std::cout << std::endl;
	}
}

///////////////////////////////
///////////////////////////////

// Utility functions adapted from Graphblast, used to read MTX files;

template<typename I, typename T>
inline bool compare(const std::tuple<I, I, T, I> &lhs, const std::tuple<I, I, T, I> &rhs) {
	I a = std::get < 0 > (lhs);
	I b = std::get < 0 > (rhs);
	I c = std::get < 1 > (lhs);
	I d = std::get < 1 > (rhs);
	if (a == b)
		return c < d;
	else
		return a < b;
}

template<typename I>
inline bool compare(const std::tuple<I, I, I> &lhs, const std::tuple<I, I, I> &rhs) {
	I a = std::get < 0 > (lhs);
	I b = std::get < 0 > (rhs);
	I c = std::get < 1 > (lhs);
	I d = std::get < 1 > (rhs);
	if (a == b)
		return c < d;
	else
		return a < b;
}


template<typename I, typename T>
inline void customSort(std::vector<I> *row_indices, std::vector<I> *col_indices, std::vector<T> *values) {
	I nvals = row_indices->size();
	std::vector<std::tuple<I, I, T, I>> my_tuple;

	for (I i = 0; i < nvals; ++i)
		my_tuple.push_back(std::make_tuple((*row_indices)[i], (*col_indices)[i], (*values)[i], i));

	std::sort(my_tuple.begin(), my_tuple.end(), compare<I, T>);

	std::vector<I> v1 = *row_indices;
	std::vector<I> v2 = *col_indices;
	std::vector<T> v3 = *values;

	for (I i = 0; i < nvals; ++i) {
		I index = std::get < 3 > (my_tuple[i]);
		(*row_indices)[i] = v1[index];
		(*col_indices)[i] = v2[index];
		(*values)[i] = v3[index];
	}
}

template<typename I, typename T>
inline void readTuples(std::vector<I> *row_indices, std::vector<I> *col_indices, std::vector<T> *values, I nvals, FILE *f, bool read_values = true, bool zero_indexed_file = false) {
	I row_ind, col_ind;
	int_type row_ind_i, col_ind_i;
	double value;

	// Currently checks if there are fewer rows than promised
	// Could add check for edges in diagonal of adjacency matrix
	for (I i = 0; i < nvals; i++) {
		if (fscanf(f, "%u", &row_ind_i) == EOF) {
			std::cout << "Error: Not enough rows in mtx file!\n";
			return;
		} else {
			fscanf(f, "%u", &col_ind_i);
			if (read_values) {
				fscanf(f, "%lf", &value);
			} else {
				value = 1.0;
			}
			row_ind = (I) row_ind_i;
			col_ind = (I) col_ind_i;

			// Convert 1-based indexing MTX to 0-based indexing C++
			if (!zero_indexed_file) {
				row_ind--;
				col_ind--;
			}
			row_indices->push_back(row_ind);
			col_indices->push_back(col_ind);
			values->push_back((T) value);
		}
	}
}

template<typename I, typename T>
inline void undirect(std::vector<I> *row_indices, std::vector<I> *col_indices, std::vector<T> *values, I *nvals,
		bool skip_values = false) {
	for (I i = 0; i < *nvals; i++) {
		if ((*col_indices)[i] != (*row_indices)[i]) {
			row_indices->push_back((*col_indices)[i]);
			col_indices->push_back((*row_indices)[i]);
			if (!skip_values) {
				values->push_back((*values)[i]);
			}
		}
	}
	*nvals = row_indices->size();
}

/*!
 * Remove self-loops, duplicates and make graph undirected if option is set
 */
template<typename I, typename T>
inline void removeSelfloop(std::vector<I> *row_indices, std::vector<I> *col_indices, std::vector<T> *values, I *nvals) {
	// Sort
	customSort<I, T>(row_indices, col_indices, values);

	I curr = (*col_indices)[0];
	I last;
	I curr_row = (*row_indices)[0];
	I last_row;

	// Detect self-loops and duplicates
	for (I i = 0; i < *nvals; i++) {
		last = curr;
		last_row = curr_row;
		curr = (*col_indices)[i];
		curr_row = (*row_indices)[i];

		// Self-loops
		if (curr_row == curr)
			(*col_indices)[i] = -1;

		// Duplicates
		if (i > 0 && curr == last && curr_row == last_row)
			(*col_indices)[i] = -1;
	}

	I shift = 0;

	// Remove self-loops and duplicates marked -1.
	I back = 0;
	for (I i = 0; i + shift < *nvals; i++) {
		if ((*col_indices)[i] == -1) {
			for (; back <= *nvals; shift++) {
				back = i + shift;
				if ((*col_indices)[back] != -1) {
					(*col_indices)[i] = (*col_indices)[back];
					(*row_indices)[i] = (*row_indices)[back];
					(*col_indices)[back] = -1;
					break;
				}
			}
		}
	}

	*nvals = *nvals - shift;
	row_indices->resize(*nvals);
	col_indices->resize(*nvals);
	values->resize(*nvals);
}

template<typename I, typename T>
inline int readMtx(const char *fname, std::vector<I> *row_indices, std::vector<I> *col_indices, std::vector<T> *values,
		I* num_rows, I* num_cols, I* num_nnz, int directed = 1, bool read_values = true, bool debug = false, bool zero_indexed_file = false, bool sort_tuples = true) {
	int ret_code;
	MM_typecode matcode;
	FILE *f;

	I nrows = 0;
	I ncols = 0;
	I nvals = 0;
	bool mtxinfo = false;

	if ((f = fopen(fname, "r")) == NULL) {
		std::cerr << "File " << fname << " not found" << std::endl;
		std::cerr.flush();
		exit(1);
	}

	// Read MTX banner
	if (mm_read_banner(f, &matcode) != 0) {
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}

	// Read MTX Size
	if ((ret_code = mm_read_mtx_crd_size(f, &nrows, &ncols, &nvals)) != 0)
		exit(1);
	readTuples<I, T>(row_indices, col_indices, values, nvals, f, read_values, zero_indexed_file);

	bool is_undirected = mm_is_symmetric(matcode) || directed == 2;
	is_undirected = (directed == 1) ? false : is_undirected;
	if (is_undirected) {
		undirect(row_indices, col_indices, values, &nvals);
	}
	if (sort_tuples) customSort<I, T>(row_indices, col_indices, values);

	if (mtxinfo)
		mm_write_banner(stdout, matcode);
	if (mtxinfo)
		mm_write_mtx_crd_size(stdout, nrows, ncols, nvals);

	*num_rows = nrows;
	*num_cols = ncols;
	*num_nnz = nvals;

	return ret_code;
}

template<typename I, typename T>
inline void coo2csr(I *csrRowPtr, I *csrColInd, T *csrVal, const std::vector<I> &row_indices,
		const std::vector<I> &col_indices, const std::vector<T> &values, I nrows, I ncols, bool sort_tuples = true) {
	I temp, row, col, dest, cumsum = 0;
	I nvals = row_indices.size();

	std::vector<I> row_indices_t = row_indices;
	std::vector<I> col_indices_t = col_indices;
	std::vector<T> values_t = values;

	if (sort_tuples) customSort<I, T>(&row_indices_t, &col_indices_t, &values_t);

	// Set all rowPtr to 0
	for (I i = 0; i <= nrows; i++) {
		csrRowPtr[i] = 0;
	}

	// Go through all elements to see how many fall in each row
	for (I i = 0; i < nvals; i++) {
		row = row_indices_t[i];
		if (row >= nrows)
			std::cout << "Error: Index out of bounds!\n";
		csrRowPtr[row]++;
	}

	// Cumulative sum to obtain rowPtr
	for (I i = 0; i < nrows; i++) {
		temp = csrRowPtr[i];
		csrRowPtr[i] = cumsum;
		cumsum += temp;
	}
	csrRowPtr[nrows] = nvals;

	// Store colInd and val
	for (I i = 0; i < nvals; i++) {
		row = row_indices_t[i];
		dest = csrRowPtr[row];
		col = col_indices_t[i];
		if (col >= ncols) {
			std::cout << "Error: Index out of bounds!\n" << std::endl;
		}

		csrColInd[dest] = col;
		csrVal[dest] = values_t[i];
		csrRowPtr[row]++;
	}
	cumsum = 0;

	// Undo damage done to rowPtr
	for (I i = 0; i < nrows; i++) {
		temp = csrRowPtr[i];
		csrRowPtr[i] = cumsum;
		cumsum = temp;
	}

	temp = csrRowPtr[nrows];
	csrRowPtr[nrows] = cumsum;
	cumsum = temp;
}

///////////////////////////////
///////////////////////////////

template<typename I>
inline void customSortFast(std::vector<I> *row_indices, std::vector<I> *col_indices) {
	I nvals = row_indices->size();
	std::vector<std::tuple<I, I, I>> my_tuple;

	for (I i = 0; i < nvals; ++i)
		my_tuple.push_back(std::make_tuple((*row_indices)[i], (*col_indices)[i], i));

	std::sort(my_tuple.begin(), my_tuple.end(), compare<I>);

	std::vector<I> v1 = *row_indices;
	std::vector<I> v2 = *col_indices;

	for (I i = 0; i < nvals; ++i) {
		I index = std::get < 2 > (my_tuple[i]);
		(*row_indices)[i] = v1[index];
		(*col_indices)[i] = v2[index];
	}
}

template<typename I>
inline void readTuplesFast(std::vector<I> *row_indices, std::vector<I> *col_indices, I* nvals, FILE *f, bool zero_indexed_file = false) {
	I row_ind, col_ind;

	row_indices->resize(*nvals);
	col_indices->resize(*nvals);

	// Number of entries actually added, excluding self-loops;
	I effective_nvals = 0;

	// Currently checks if there are fewer rows than promised
	// Could add check for edges in diagonal of adjacency matrix;
	for (I i = 0; i < *nvals; i++) {
		fscanf(f, "%u %u", &row_ind, &col_ind);

		// Convert 1-based indexing MTX to 0-based indexing C++;
		if (row_ind != col_ind) {
			if (!zero_indexed_file) {
				row_ind--;
				col_ind--;
			}
			(*row_indices)[effective_nvals] = row_ind;
			(*col_indices)[effective_nvals] = col_ind;
			effective_nvals++;
		}
	}

	row_indices->resize(effective_nvals);
	col_indices->resize(effective_nvals);
	*nvals = effective_nvals;
}

// NOTE: duplicate vertices are marked using value INDEX_TYPE_MAX, i.e. the maximum value of index_type (e.g. 2^32 - 1).
// This will give issues if undirecting a graph whose number of vertices is the same as the index type maximum value;
template<typename I>
inline void undirectFast(std::vector<I> *row_indices, std::vector<I> *col_indices, I *nvals) {
	for (I i = 0; i < *nvals; i++) {
		if ((*col_indices)[i] != (*row_indices)[i]) {
			row_indices->push_back((*col_indices)[i]);
			col_indices->push_back((*row_indices)[i]);
		}
	}
	*nvals = row_indices->size();

	I curr = (*col_indices)[0];
	I last;
	I curr_row = (*row_indices)[0];
	I last_row;

	customSortFast(row_indices, col_indices);

	// Detect self-loops and duplicates
	for (I i = 0; i < *nvals; i++) {
		last = curr;
		last_row = curr_row;
		curr = (*col_indices)[i];
		curr_row = (*row_indices)[i];

		// Self-loops
		if (curr_row == curr)
			(*col_indices)[i] = INDEX_TYPE_MAX;

		// Duplicates
		if (i > 0 && curr == last && curr_row == last_row)
			(*col_indices)[i] = INDEX_TYPE_MAX;
	}

	I shift = 0;

	// Remove self-loops and duplicates marked -1.
	I back = 0;
	for (I i = 0; i + shift < *nvals; i++) {
		if ((*col_indices)[i] == INDEX_TYPE_MAX) {
			for (; back <= *nvals; shift++) {
				back = i + shift;
				if ((*col_indices)[back] != INDEX_TYPE_MAX) {
					(*col_indices)[i] = (*col_indices)[back];
					(*row_indices)[i] = (*row_indices)[back];
					(*col_indices)[back] = INDEX_TYPE_MAX;
					break;
				}
			}
		}
	}

	*nvals = *nvals - shift;
	row_indices->resize(*nvals);
	col_indices->resize(*nvals);
}

// Faster function to read an MTX matrix, it gives correct results assuming the following:
// - Values are all initialized to 1;
// - No duplicate entries are present;
// - Entries are already sorted by row and column;
// - Self-loops are skipped;
template<typename I>
inline int readMtxFast(const char *fname, std::vector<I> *row_indices, std::vector<I> *col_indices,
		I* num_vertices, I* num_edges, int directed = 1, bool debug = false, bool zero_indexed_file = false) {
	int ret_code;
	MM_typecode matcode;
	FILE *f;

	I nrows = 0;
	I ncols = 0;
	I nvals = 0;
	bool mtxinfo = false;

	if ((f = fopen(fname, "r")) == NULL) {
		std::cerr << "File " << fname << " not found" << std::endl;
		std::cerr.flush();
		exit(1);
	}

	// Read MTX banner
	if (mm_read_banner(f, &matcode) != 0) {
		printf("Could not process Matrix Market banner.\n");
		exit(1);
	}

	// Read MTX Size
	if ((ret_code = mm_read_mtx_crd_size(f, &nrows, &ncols, &nvals)) != 0)
		exit(1);

	readTuplesFast<I>(row_indices, col_indices, &nvals, f, zero_indexed_file);

	// Undirect the graph if required;
	bool is_undirected = mm_is_symmetric(matcode) || directed == 2;
	is_undirected = (directed == 1) ? false : is_undirected;
	if (is_undirected) {
		undirectFast(row_indices, col_indices, &nvals);
	}

	if (mtxinfo)
		mm_write_banner(stdout, matcode);
	if (mtxinfo)
		mm_write_mtx_crd_size(stdout, nrows, ncols, nvals);

	*num_vertices = nrows;
	*num_edges = nvals;

	return ret_code;
}

template<typename I>
inline void coo2csrFast(I *csrRowPtr, I *csrColInd, const std::vector<I> &row_indices,
		const std::vector<I> &col_indices, I nrows, I ncols) {
	I temp, row, col, dest, cumsum = 0;
	I nvals = row_indices.size();

	// Set all rowPtr to 0
	for (I i = 0; i <= nrows; i++)
		csrRowPtr[i] = 0;

	// Go through all elements to see how many fall in each row
	for (I i = 0; i < nvals; i++) {
		row = row_indices[i];
		if (row >= nrows)
			std::cout << "Error: Index out of bounds!\n";
		csrRowPtr[row]++;
	}

	// Cumulative sum to obtain rowPtr
	for (I i = 0; i < nrows; i++) {
		temp = csrRowPtr[i];
		csrRowPtr[i] = cumsum;
		cumsum += temp;
	}
	csrRowPtr[nrows] = nvals;

	// Store colInd and val
	for (I i = 0; i < nvals; i++) {
		row = row_indices[i];
		dest = csrRowPtr[row];
		col = col_indices[i];
		if (col >= ncols)
			std::cout << "Error: Index out of bounds!\n";
		csrColInd[dest] = col;
		csrRowPtr[row]++;
	}
	cumsum = 0;

	// Undo damage done to rowPtr
	for (I i = 0; i < nrows; i++) {
		temp = csrRowPtr[i];
		csrRowPtr[i] = cumsum;
		cumsum = temp;
	}
	temp = csrRowPtr[nrows];
	csrRowPtr[nrows] = cumsum;
	cumsum = temp;
}

///////////////////////////////
///////////////////////////////

template<typename I, typename T>
inline void coo2csc(I *cscColPtr, I *cscRowInd, T *cscVal, const std::vector<I> &row_indices,
		const std::vector<I> &col_indices, const std::vector<T> &values, I nrows, I ncols) {
	return coo2csr(cscColPtr, cscRowInd, cscVal, col_indices, row_indices, values, ncols, nrows);
}

template<typename I>
inline void coo2cscFast(I *cscColPtr, I *cscRowInd, const std::vector<I> &row_indices,
		const std::vector<I> &col_indices, I nrows, I ncols) {
	return coo2csrFast(cscColPtr, cscRowInd, col_indices, row_indices, ncols, nrows);
}


///////////////////////////////
///////////////////////////////

inline csc_t load_graph_csc(const std::string dir_path, bool debug) {

	if (debug) {
		std::cout << "Parsing " << dir_path << std::endl;
	}

	std::stringstream ss_val;
	std::stringstream ss_n_zero;
	std::stringstream ss_col_idx;

	std::ifstream val, non_zero, col_idx;

	std::vector<num_type> csc_col_val;
	std::vector<index_type> csc_col_ptr;
	std::vector<index_type> csc_col_idx;

	csc_t csc;

	ss_val << dir_path << "/" << "col_val.txt";
	ss_n_zero << dir_path << "/" << "col_ptr.txt";
	ss_col_idx << dir_path << "/" << "col_idx.txt";

	val.open(ss_val.str());
	non_zero.open(ss_n_zero.str());
	col_idx.open(ss_col_idx.str());

	if (!val || !non_zero || !col_idx) {
		if (debug) {
			std::cerr << "Error reading file" << std::endl;
			std::cerr.flush();
		}
		exit(1);
	}

	int tmp2;
	num_type tmp1;

	while (val >> tmp1) {
		csc_col_val.push_back(tmp1);
	}

	while (non_zero >> tmp2) {
		csc_col_ptr.push_back(tmp2);
	}

	while (col_idx >> tmp2) {
		csc_col_idx.push_back(tmp2);
	}

	csc.col_val = csc_col_val;
	csc.col_ptr = csc_col_ptr;
	csc.col_idx = csc_col_idx;

	return csc;
}

inline csc_t load_graph_mtx(const std::string path_to_mtx, bool debug = false, bool undirect_graph = false) {

	csc_t csc;
	index_type num_vertices = 0;
	index_type num_edges = 0;
	std::vector<index_type> row_indices;
	std::vector<index_type> col_indices;

	auto start_1 = clock_type::now();
	int res = readMtxFast<index_type>(path_to_mtx.c_str(), &row_indices, &col_indices, &num_vertices,
			&num_edges, (undirect_graph ? 2 : 0), debug);
	auto time_1 = chrono::duration_cast<chrono::milliseconds>(clock_type::now() - start_1).count();
	if (debug) {
		std::cout << "- read MTX: " << time_1 / 1000 << " sec" << std::endl;
	}
	// Allocate space for the CSC vectors;
	csc.col_idx.resize(num_edges);
	csc.col_val.resize(num_edges);
	csc.col_ptr.resize(num_vertices + 1);

	auto start_2 = clock_type::now();
	coo2cscFast<index_type>(csc.col_ptr.data(), csc.col_idx.data(), row_indices, col_indices,
			num_edges, num_vertices + 1);
	auto time_2 = chrono::duration_cast<chrono::milliseconds>(clock_type::now() - start_2).count();
	if (debug) {
		std::cout << "- coo2csc: " << time_2 / 1000 << " sec" << std::endl;
	}

	if (res) {
		if (debug) {
			std::cerr << "error reading mtx file" << std::endl;
			std::cerr.flush();
		}
		exit(1);
	}

	// Initialize the value vector to 1, then
	// normalize the value vector, divide it by the outdegree of each vertex;
	std::fill(csc.col_val.begin(), csc.col_val.end(), 1);
	index_type *outdegree = (index_type *)calloc(num_vertices, sizeof(index_type));
	for (index_type i = 0; i < num_edges; i++) {
		outdegree[csc.col_idx[i]]++;
	}
	for (index_type i = 0; i < num_edges; i++) {
		csc.col_val[i] /= outdegree[csc.col_idx[i]];
	}

	free(outdegree);
	return csc;
}

///////////////////////////////
///////////////////////////////

inline long get_file_size(std::string filename) {
	struct stat stat_buf;
	int rc = stat(filename.c_str(), &stat_buf);
	return rc == 0 ? stat_buf.st_size : -1;
}

