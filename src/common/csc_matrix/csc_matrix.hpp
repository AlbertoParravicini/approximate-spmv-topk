//
// Created by fra on 19/04/19.
//

#pragma once

#include <iostream>
#include <vector>
#include <limits.h>

#define num_type double
#define fixed_num_type long long unsigned
typedef unsigned int index_type;
typedef unsigned int index_type_fpga;

// Maximum value for a graph index, used to mark invalid vertex values;
#define INDEX_TYPE_MAX UINT_MAX

typedef struct csc_t {
    std::vector<num_type> col_val;
    std::vector<index_type> col_ptr;
    std::vector<index_type> col_idx;
} csc_t;

typedef struct csc_fixed_t {
    std::vector<fixed_num_type> col_val;
    std::vector<index_type> col_ptr;
    std::vector<index_type> col_idx;
} csc_fixed_t;
