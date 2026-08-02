#pragma once

#include "Eigen/Dense"

#include "operational-space-control/walter_sr/autogen/autogen_defines.h"

using namespace operational_space_controller::constants;


namespace operational_space_controller::constants {
    namespace optimization {
        // s_size : Size of fully spatial vector representation for all bodies
        constexpr int s_size = 6 * model::body_ids_size;
        // p_size : Size of translation component to a spatial vector:
        constexpr int p_size = 3 * model::body_ids_size;
        // r_size : Size of rotation component to a spatial vector:
        constexpr int r_size = 3 * model::body_ids_size;

        // Constraint Matrix Size:
        constexpr int constraint_matrix_rows = optimization::Aeq_rows + optimization::Aineq_rows + optimization::design_vector_size;
        constexpr int constraint_matrix_cols = optimization::design_vector_size;
        constexpr int bounds_size = optimization::beq_sz + optimization::bineq_sz + optimization::design_vector_size;
    }
}
