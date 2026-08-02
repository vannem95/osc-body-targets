#pragma once

#include "operational-space-control/walter_sr/aliases.h"
#include "operational-space-control/walter_sr/constants.h"
#include "operational-space-control/walter_sr/autogen/autogen_defines.h"

using namespace operational_space_controller::constants;
using namespace operational_space_controller::aliases;


namespace operational_space_controller {
    namespace containers {
        struct OSCData {
            Matrix<model::nv_size, model::nv_size> mass_matrix;    
            Vector<model::nv_size> coriolis_matrix;
            Matrix<model::nv_size, optimization::z_size> contact_jacobian;
            Matrix<optimization::s_size, model::nv_size> taskspace_jacobian;
            Vector<optimization::s_size> taskspace_bias;
            Vector<model::nq_size> previous_q;
            Vector<model::nv_size> previous_qd;
        };
        
        struct OptimizationData {
            MatrixColMajor<optimization::H_rows, optimization::H_cols> H;
            Vector<optimization::f_sz> f;
            MatrixColMajor<optimization::Aeq_rows, optimization::Aeq_cols> Aeq;
            Vector<optimization::beq_sz> beq;
            Matrix<optimization::Aineq_rows, optimization::Aineq_cols> Aineq;
            Vector<optimization::bineq_sz> bineq;
        };

        struct State {
            Vector<model::nu_size> motor_position;
            Vector<model::nu_size> motor_velocity;
            Vector<model::nu_size> motor_acceleration;
            Vector<model::nu_size> torque_estimate;
            Vector<4> body_rotation;
            Vector<3> linear_body_velocity;
            Vector<3> angular_body_velocity;
            Vector<3> linear_body_acceleration;
            Vector<model::contact_site_ids_size> contact_mask;
        };
    }
}

