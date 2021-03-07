#pragma once

#include <iostream>
#include <vector>
// #include <ap_fixed.h>
#include "coo_matrix.hpp"
// #include "types.hpp"

template<typename I, typename T>
struct coo_fixed_fpga_t {
    std::vector<I> start;
    std::vector<I> end;
    std::vector<T> val;
    I N = 0;
    I E = 0;
    // Optionally, the COO contains diagonal value with value 0, to avoid skipping diagonal values;
    I E_fixed = 0;
    bool extra_self_loops_added = false;

    coo_fixed_fpga_t(coo_t<I, T> coo, bool add_missing_loops = false) : coo_fixed_fpga_t(coo.start, coo.end, coo.val, add_missing_loops) {}

    coo_fixed_fpga_t(std::vector<I> _start, std::vector<I> _end, std::vector<T> _val, bool add_missing_loops = false) {

    	E = _start.size();
    	I extra_N = 0;
		for (int i = 0; i < _start.size(); ++i){
			start.push_back(_start[i]);
			end.push_back(_end[i]);
			val.push_back(_val[i]);

			N = std::max(N, _start[i]);

			// Add a 0 diagonal value if this row has no non-zero values, to avoid skipping this row;
			if (i < _start.size() - 1 && add_missing_loops) {
				for (int j = _start[i] + 1; j < _start[i + 1]; j++) {
					start.push_back(j);
					end.push_back(j);
					val.push_back(0);
					extra_N++;
				}
			}
		}
		E_fixed = E + extra_N;
		extra_self_loops_added = true;
		N++;
	}

    void print_coo(bool compact = false, bool transposed = true) {
		if (compact) {
			int n = 0;
			// TODO: need to add transposed;
			I last_s = 0;
			I curr_e = 0;
			std::vector<I> neighbors;
			std::vector<I> vals;

			std::cout << "N: " << N << ", E: " << E << std::endl;

			for (int i = 0; i < start.size(); i++) {
				I curr_s = start[i];
				if (curr_s == last_s) {
					neighbors.push_back(end[curr_e]);
					vals.push_back(val[curr_e++]);
				} else {
					std::cout << n << ") degree: " << neighbors.size() << std::endl;
					std::cout << "  edges: ";
					for (auto e: neighbors) {
						std::cout << e << ", ";
					}
					std::cout << std::endl;
					std::cout << "  vals: ";
					for (auto v: vals) {
//						std::cout << v.to_float() << ", ";
					}
					std::cout << std::endl;

					last_s = curr_s;
					neighbors = { end[curr_e] };
					vals = { val[curr_e++] };
					n = curr_s;
				}
			}
			std::cout << n << ") degree: " << neighbors.size() << std::endl;
			std::cout << "  edges: ";
			for (auto e: neighbors) {
				std::cout << e << ", ";
			}
			std::cout << std::endl;
			std::cout << "  vals: ";
			for (auto v: vals) {
//				std::cout << v.to_float() << ", ";
			}
			std::cout << std::endl;
		} else {
			for (int i = 0; i < start.size(); i++) {
//				std::cout << start[i] << (transposed ? " <- " : " -> ") << end[i] << ": " << val[i].to_float() << std::endl;
			}
		}
	}
};
