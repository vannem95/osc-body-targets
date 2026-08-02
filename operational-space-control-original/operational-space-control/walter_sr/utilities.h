#pragma once

#include <array>
#include <algorithm>

#include <Eigen/Dense>

#include "operational-space-control/walter_sr/autogen/autogen_functions.h"


namespace {
    typedef void (*func_incref)();
    typedef int (*func_checkout)();
    typedef int (*eval_func)(const double** args, double** res, casadi_int* iw, double* w, int mem);
    typedef void (*func_release)(int mem);
    typedef void (*func_decref)();
}

struct FunctionOperations {
    func_incref incref;
    func_checkout checkout;
    eval_func eval;
    func_release release;
    func_decref decref;
};

template<size_t sz_args, size_t sz_res, size_t sz_iw, size_t sz_w, int rows, int cols, size_t output_size, size_t N>
struct FunctionParams {
    static constexpr size_t args_size = sz_args;
    static constexpr size_t res_size = sz_res;
    static constexpr size_t iw_size = sz_iw;
    static constexpr size_t w_size = sz_w;
    static constexpr int matrix_rows = rows;
    static constexpr int matrix_cols = cols;
    static constexpr size_t out_size = output_size;
    static constexpr size_t num_args = N;
};

// Alias template for the return type
template<typename Params>
using ReturnType = Eigen::Matrix<double, Params::matrix_rows, Params::matrix_cols, Eigen::ColMajor>;

template<typename Params>
ReturnType<Params> evaluate_function(
    FunctionOperations& ops, 
    const std::array<double*, Params::num_args> arguments) {
    
    // Allocate Work Vectors:
    const double *args[Params::args_size];
    double *res[Params::res_size];
    casadi_int iw[Params::iw_size];
    double w[Params::w_size];

    // Place result pointer in the result array:
    double result[Params::out_size];
    res[0] = result;

    // Increase the reference count:
    ops.incref();

    // Copy arguments into args array:
    std::copy(arguments.begin(), arguments.end(), args);

    // Initialize Memory:
    int mem = ops.checkout();

    // Evaluate the Function:
    ops.eval(args, res, iw, w, mem);

    // Release Memory:
    ops.release(mem);

    // Decrease the reference count:
    ops.decref();

    return Eigen::Map<ReturnType<Params>>(result);
}
