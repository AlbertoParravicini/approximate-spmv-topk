#pragma once

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <set>
#include <vector>
#include <algorithm>
#include <map>
#include <utility>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <tuple>

template<typename T>
inline std::vector<index_type> sort_pr(size_t DIM, T *pr) {

	std::map<index_type, T> pr_map;
	std::vector<std::pair<index_type, T>> sorted_pr;
	std::vector<index_type> sorted_pr_idxs;

	for (index_type i = 0; i < DIM; ++i) {
		sorted_pr.push_back( { i, pr[i] });
		pr_map[i] = pr[i];
	}

	std::sort(sorted_pr.begin(), sorted_pr.end(),
			[](const std::pair<index_type, num_type> &l, const std::pair<index_type, num_type> &r) {
				if (l.second != r.second)return l.second > r.second;
				else return l.first > r.first;
			});

	for (auto const &pair : sorted_pr) {
		sorted_pr_idxs.push_back(pair.first);
	}
	return sorted_pr_idxs;
}

template<typename I, typename V>
inline void sort_tuples(size_t DIM, I *idx, V* val) {

	std::map<I, V> tuples_map;
	std::vector<std::pair<I, V>> sorted_tuples;
	std::vector<I> sorted_idxs;

	for (uint i = 0; i < DIM; ++i) {
		sorted_tuples.push_back( { idx[i], val[i] });
		tuples_map[idx[i]] = val[i];
	}

	std::sort(sorted_tuples.begin(), sorted_tuples.end(),
			[](const std::pair<I, V> &l, const std::pair<I, V> &r) {
				if (l.second != r.second) return l.second > r.second;
				else return l.first > r.first;
			});

	for (int i = 0; i < sorted_tuples.size(); i++) {
		idx[i] = sorted_tuples[i].first;
		val[i] = sorted_tuples[i].second;
	}
}

template<typename T>
inline int compare_results(const size_t DIM, T *pr,
		std::string golden_result_path, bool debug = false) {

	auto sorted_pr_idxs = sort_pr(DIM, pr);

	if (debug) {
		std::cout << "Checking results..." << std::endl;
	}
	std::ifstream results;
	results.open(golden_result_path);

	int i = 0;
	int tmp = 0;
	int errors = 0;

	int prev_left_idx = 0;
	int prev_right_idx = 0;

	while (results >> tmp) {
		if (debug) {
			std::cout << "Comparing " << tmp << " ==? " << sorted_pr_idxs[i]
					<< std::endl;
		}

		if (tmp != sorted_pr_idxs[i]) {
			if (prev_left_idx != sorted_pr_idxs[i] || prev_right_idx != tmp) {
				errors++;
			}

			prev_left_idx = tmp;
			prev_right_idx = sorted_pr_idxs[i];

		}
		i++;
	}

	if (debug) {
		std::cout << "Percentage of error: "
				<< (((double) errors) / (DIM)) * 100 << "%\n" << std::endl;

		std::cout << "End of computation! Freeing memory..." << std::endl;
	}

	return errors;

}

inline double normalized_discounted_cumulative_gain(std::vector<int> golden,
		std::vector<int> test, const bool debug = false) {

	std::unordered_map<int, int> ranking;
	std::vector<std::pair<int, int>> golden_ranking;

	double dcg = 0.0;
	double idcg = 0.0;

	const int DIM = golden.size();

	for (uint i = 0; i < DIM; ++i) {
		ranking[test[i]] = DIM - i;
		golden_ranking.push_back(std::make_pair(golden[i], DIM - i));
	}

	for (uint i = 0; i < DIM; ++i) {
		int vertex = golden_ranking[i].first;
		int golden_rel = golden_ranking[i].second;

		auto next_ranking = ranking.find(vertex);
		int test_rel = 0;
		if (next_ranking != ranking.end()) {
			test_rel = ranking[vertex];
		}
//		if (debug) {
//			std::cout << "elem: " << elem << " golden: " << golden_elem
//					<< " i -> " << i << std::endl;
//		}

		dcg += (double) test_rel / log2((float) abs(golden_rel - DIM) + 2);
		idcg += (double) golden_rel / log2((float) abs(golden_rel - DIM) + 2);

	}

	return dcg / idcg;
}

/////////////////////////////
/////////////////////////////

inline std::vector<double> bounded_ndcg(std::vector<int> golden,
		std::vector<int> test, std::vector<int> bounds = std::vector<int> {10, 20, 50 }, bool debug = false) {

	std::vector<int> index_bounds = bounds;
	std::vector<double> ndcgs;

	// Perform ndcg on the sliced vectors
	for (auto index : index_bounds) {

		if (index > golden.size())
			break;

		auto bounded_golden = std::vector<int>(golden.begin(),
				golden.begin() + index);
		auto bounded_test = std::vector<int>(test.begin(),
				test.begin() + index);

		ndcgs.push_back(
				normalized_discounted_cumulative_gain(bounded_golden,
						bounded_test, debug));

	}

	if (debug) {
		for (int i = 0; i < ndcgs.size(); ++i) {
			std::cout << "top " << bounds[i] << ") " << "ndcgs[" << i << "] = "
					<< ndcgs[i] << std::endl;
		}
	}

	return ndcgs;

}

inline unsigned int edit_distance(const std::vector<int> &s1, const std::vector<int> &s2)
{
    const std::size_t len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<unsigned int>> d(len1 + 1, std::vector<unsigned int>(len2 + 1));

    d[0][0] = 0;
    for(unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
    for(unsigned int i = 1; i <= len2; ++i) d[0][i] = i;

    for(unsigned int i = 1; i <= len1; ++i)
        for(unsigned int j = 1; j <= len2; ++j)
            d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1) });
    return d[len1][len2];
}

inline std::vector<unsigned int> bounded_edit_distance(std::vector<int> golden,
		std::vector<int> test, std::vector<int> bounds = std::vector<int> {10, 20, 50 }, bool debug = false) {

	std::vector<int> index_bounds = bounds;
	std::vector<unsigned int> edit_distances;

	// Perform ndcg on the sliced vectors
	for (auto index : index_bounds) {

		if (index > golden.size())
			break;

		auto bounded_golden = std::vector<int>(golden.begin(),
				golden.begin() + index);
		auto bounded_test = std::vector<int>(test.begin(),
				test.begin() + index);

		edit_distances.push_back(
				edit_distance(bounded_golden,
						bounded_test));

	}

	if (debug) {
		for (int i = 0; i < edit_distances.size(); ++i) {
			std::cout << "top " << bounds[i] << ") " << "edit_distances[" << i << "] = "
					<< edit_distances[i] << std::endl;
		}
	}

	return edit_distances;

}
/////////////////////////////
/////////////////////////////

inline std::vector<int> bounded_count_errors(std::vector<int> golden,
		std::vector<int> test, std::vector<int> bounds =
				std::vector<int> { 10, 20, 50 }, bool debug = false) {

	std::vector<int> index_bounds = bounds;
	std::vector<int> errors;

	for (auto index : index_bounds) {

		if (index > golden.size())
			break;

		int tmp_errs = 0;
		for (int i = 0; i < index; ++i) {
			if (golden[i] != test[i])
				tmp_errs++;
		}

		errors.push_back(tmp_errs);

	}

	if (debug) {
		for (int i = 0; i < errors.size(); ++i) {
			std::cout << "top " << bounds[i] << ") " << "errors[" << i << "] = "
					<< errors[i] << std::endl;
		}
	}

	return errors;

}

/////////////////////////////
/////////////////////////////
template<typename T>
inline T mean(std::vector<T> x, int skip = 0) {
	T sum = 0;
	int fixed_size = x.size() - skip;
	if (fixed_size <= 0) return (T) 0;
	for (uint i = skip; i < x.size(); i++) {
		sum += x[i];
	}
	return sum / (x.size() - skip);
}

template<typename T>
inline T st_dev(std::vector<T> x, int skip = 0) {
	T mean = 0;
	T mean_sq = 0;
	int fixed_size = x.size() - skip;
	if (fixed_size <= 0) return (T) 0;
	for (uint i = skip; i < x.size(); i++) {
		mean += x[i];
		mean_sq += x[i] * x[i];
	}
	T diff = mean_sq - mean * mean / (fixed_size);
	diff = diff >= 0 ? diff : 0;
	return std::sqrt(diff / fixed_size);
}
