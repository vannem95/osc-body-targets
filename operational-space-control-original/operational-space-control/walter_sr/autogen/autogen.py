from absl import app
from absl import flags

import os
import yaml

import casadi
import mujoco

from casadi import MX, DM

from python.runfiles import Runfiles


FLAGS = flags.FLAGS
flags.DEFINE_string("filepath", None, "Bazel filepath to the autogen folder (This should be automatically determinded by the genrule).")


class AutoGen():
    def __init__(self, mj_model: mujoco.MjModel):
        self.mj_model = mj_model

        # Parse Configuration YAML File:
        r = Runfiles.Create()
        with open(r.Rlocation("operational-space-controller/config/walter_sr/walter_sr_config.yaml"), "r") as file:
            config = yaml.safe_load(file)

        # Get Weight Configuration:
        self.weights_config = config['weights_config']

        # Get Body and Site IDs:
        self.body_list = [f'"{body}"sv' for body in config['body_list']]
        self.noncontact_site_list = [f'"{noncontact_site}"sv' for noncontact_site in config['noncontact_site_list']]
        self.contact_site_list = [f'"{contact_site}"sv' for contact_site in config['contact_site_list']]
        self.site_list = self.noncontact_site_list + self.contact_site_list
        self.num_body_ids = len(self.body_list)
        self.num_site_ids = len(self.site_list)
        self.num_noncontact_site_ids = len(self.noncontact_site_list)
        self.num_contact_site_ids = len(self.contact_site_list)
        self.mu = config['friction_coefficient']

        assert self.num_body_ids == self.num_site_ids, "Number of body IDs and site IDs must be equal."

        self.dv_size = self.mj_model.nv
        self.u_size = self.mj_model.nu
        self.z_size = self.num_contact_site_ids * 3
        self.design_vector_size = self.dv_size + self.u_size + self.z_size

        # self.dv_idx = self.dv_size
        # self.u_idx = self.dv_idx + self.u_size
        # self.z_idx = self.u_idx + self.z_size

        self.dv_idx = 0
        self.u_idx = self.dv_size
        self.z_idx = self.dv_size + self.u_size

        # To replace 6 from B matrix - the pos/rot xyz of the first non contact site
        body_dim = self.dv_size - self.u_size

        self.B: DM = casadi.vertcat(
            DM.zeros((body_dim, self.u_size)),
            # DM.zeros((6, self.u_size)),
            DM.eye(self.u_size),
        )

    def equality_constraints(
        self,
        q: MX,
        M: MX,
        C: MX,
        J_contact: MX,
    ) -> MX:
        """Equality constraints for the dynamics of a system.

        Args:
            q: design vector.
            M: The mass matrix.
            C: The Coriolis matrix.
            J_contact: Contact Jacobian.
            split_indx: The index at which the optimization
                variables are split in dv and u.

        Returns:
            The equality constraints.

            Dynamics:
            M @ dv + C - B @ u - J_contact.T @ z = 0
        """
        # Unpack Design Variables:
        # dv = q[:self.dv_idx]
        # u = q[self.dv_idx:self.u_idx]
        # z = q[self.u_idx:self.z_idx]

        dv = q[self.dv_idx : self.dv_idx + self.dv_size]
        u = q[self.u_idx : self.u_idx + self.u_size]
        z = q[self.z_idx : self.z_idx + self.z_size]

        # Dynamics:
        equality_constraints = M @ dv + C - self.B @ u - J_contact @ z

        return equality_constraints

    def inequality_constraints(
        self,
        q: MX,
    ) -> MX:
        """Compute inequality constraints for the Operational Space Controller.

        Args:
            q: design vector.

        Returns:
            MX: Inequality constraints.

            Friction Cone Constraints:
            |f_x| + |f_y| <= mu * f_z

        """
        # Unpack Design Variables:
        # dv = q[:self.dv_idx]
        # u = q[self.dv_idx:self.u_idx]
        # z = q[self.u_idx:self.z_idx]

        dv = q[self.dv_idx : self.dv_idx + self.dv_size]
        u = q[self.u_idx : self.u_idx + self.u_size]
        z = q[self.z_idx : self.z_idx + self.z_size]



        def translational_friction(x: MX) -> MX:
            constraint_1 = x[0] + x[1] - self.mu * x[2]
            constraint_2 = -x[0] + x[1] - self.mu * x[2]
            constraint_3 = x[0] - x[1] - self.mu * x[2]
            constraint_4 = -x[0] - x[1] - self.mu * x[2]
            return casadi.vertcat(constraint_1, constraint_2, constraint_3, constraint_4)

        contact_forces = casadi.vertsplit_n(z, self.num_contact_site_ids)

        inequality_constraints = []
        for i in range(self.num_contact_site_ids):
            contact_force = contact_forces[i]
            friction_constraints = translational_friction(contact_force)
            inequality_constraints.append(friction_constraints)

        inequality_constraints = casadi.vertcat(*inequality_constraints)

        return inequality_constraints

    def objective(
        self,
        q: MX,
        desired_task_ddx: MX,
        J_task: MX,
        task_bias: MX,
    ) -> MX:
        """Compute the Task Space Tracking Objective.

        Args:
            q: Design vector.
            desired_task_ddx: Desired task acceleration.
            J_task: Taskspace Jacobian.
            task_bias: Taskspace bias acceleration.

        Returns:
            MX: Objective function.

        """
        # Unpack Design Variables:
        # dv = q[:self.dv_idx]
        # u = q[self.dv_idx:self.u_idx]
        # z = q[self.u_idx:self.z_idx]

        dv = q[self.dv_idx : self.dv_idx + self.dv_size]
        u = q[self.u_idx : self.u_idx + self.u_size]
        z = q[self.z_idx : self.z_idx + self.z_size]


        # Compute Task Space Tracking Objective:
        ddx_task = J_task @ dv + task_bias

        # Split into Translational and Rotational components:
        ddx_task_p, ddx_task_r = casadi.vertsplit_n(ddx_task, 2)
        #-------------------------------------------------------------------------------------------------------------------------
        # edits
        #-------------------------------------------------------------------------------------------------------------------------
        ddx_torso_p, ddx_tls_p, ddx_trs_p,  ddx_hls_p,  ddx_hrs_p, ddx_tlh_p, ddx_trh_p,  ddx_hlh_p,  ddx_hrh_p, ddx_tlf_p, ddx_tlr_p, ddx_trf_p, ddx_trr_p, ddx_hlf_p, ddx_hlr_p, ddx_hrf_p, ddx_hrr_p = casadi.vertsplit_n(ddx_task_p, self.num_site_ids)
        ddx_torso_r, ddx_tls_r, ddx_trs_r,  ddx_hls_r,  ddx_hrs_r, ddx_tlh_r, ddx_trh_r,  ddx_hlh_r,  ddx_hrh_r, ddx_tlf_r, ddx_tlr_r, ddx_trf_r, ddx_trr_r, ddx_hlf_r, ddx_hlr_r, ddx_hrf_r, ddx_hrr_r = casadi.vertsplit_n(ddx_task_r, self.num_site_ids)
        # ddx_torso_p, ddx_tlf_p, ddx_tlr_p, ddx_trf_p, ddx_trr_p, ddx_hlf_p, ddx_hlr_p, ddx_hrf_p, ddx_hrr_p = casadi.vertsplit_n(ddx_task_p, self.num_site_ids)
        # ddx_torso_r, ddx_tlf_r, ddx_tlr_r, ddx_trf_r, ddx_trr_r, ddx_hlf_r, ddx_hlr_r, ddx_hrf_r, ddx_hrr_r = casadi.vertsplit_n(ddx_task_r, self.num_site_ids)

        # Split Desired Task Acceleration:
        desired_task_p, desired_task_r = casadi.horzsplit_n(desired_task_ddx, 2)
        desired_task_p = casadi.vertsplit_n(desired_task_p, self.num_site_ids)
        desired_task_r = casadi.vertsplit_n(desired_task_r, self.num_site_ids)
        desired_torso_p, desired_tls_p, desired_trs_p, desired_hls_p, desired_hrs_p, desired_tlh_p, desired_trh_p, desired_hlh_p, desired_hrh_p, desired_tlf_p, desired_tlr_p, desired_trf_p, desired_trr_p, desired_hlf_p, desired_hlr_p, desired_hrf_p, desired_hrr_p = map(lambda x: x.T, desired_task_p)
        desired_torso_r, desired_tls_r, desired_trs_r, desired_hls_r, desired_hrs_r, desired_tlh_r, desired_trh_r, desired_hlh_r, desired_hrh_r, desired_tlf_r, desired_tlr_r, desired_trf_r, desired_trr_r, desired_hlf_r, desired_hlr_r, desired_hrf_r, desired_hrr_r = map(lambda x: x.T, desired_task_r)
        # desired_torso_p, desired_tlf_p, desired_tlr_p, desired_trf_p, desired_trr_p, desired_hlf_p, desired_hlr_p, desired_hrf_p, desired_hrr_p = map(lambda x: x.T, desired_task_p)
        # desired_torso_r, desired_tlf_r, desired_tlr_r, desired_trf_r, desired_trr_r, desired_hlf_r, desired_hlr_r, desired_hrf_r, desired_hrr_r = map(lambda x: x.T, desired_task_r)

        # I could make this more general at the cost of readability...
        # ddx_task_p = casadi.vertsplit_n(ddx_task_p, self.num_site_ids)
        # ddx_task_r = casadi.vertsplit_n(ddx_task_r, self.num_site_ids)
        # desired_task_p = casadi.vertsplit_n(desired_task_p, self.num_site_ids)
        # desired_task_r = casadi.vertsplit_n(desired_task_r, self.num_site_ids)

        objective_terms = {
            'torso_translational_tracking': self._objective_tracking(
                ddx_torso_p,
                desired_torso_p,
            ),
            'torso_rotational_tracking': self._objective_tracking(
                ddx_torso_r,
                desired_torso_r,
            ),
            'tls_translational_tracking': self._objective_tracking(
                ddx_tls_p,
                desired_tls_p,
            ),
            'tls_rotational_tracking': self._objective_tracking(
                ddx_tls_r,
                desired_tls_r,
            ),
            'trs_translational_tracking': self._objective_tracking(
                ddx_trs_p,
                desired_trs_p,
            ),
            'trs_rotational_tracking': self._objective_tracking(
                ddx_trs_r,
                desired_trs_r,
            ),
            'hls_translational_tracking': self._objective_tracking(
                ddx_hls_p,
                desired_hls_p,
            ),
            'hls_rotational_tracking': self._objective_tracking(
                ddx_hls_r,
                desired_hls_r,
            ),
            'hrs_translational_tracking': self._objective_tracking(
                ddx_hrs_p,
                desired_hrs_p,
            ),
            'hrs_rotational_tracking': self._objective_tracking(
                ddx_hrs_r,
                desired_hrs_r,
            ),
            'tlh_translational_tracking': self._objective_tracking(
                ddx_tlh_p,
                desired_tlh_p,
            ),
            'tlh_rotational_tracking': self._objective_tracking(
                ddx_tlh_r,
                desired_tlh_r,
            ),
            'trh_translational_tracking': self._objective_tracking(
                ddx_trh_p,
                desired_trh_p,
            ),
            'trh_rotational_tracking': self._objective_tracking(
                ddx_trh_r,
                desired_trh_r,
            ),
            'hlh_translational_tracking': self._objective_tracking(
                ddx_hlh_p,
                desired_hlh_p,
            ),
            'hlh_rotational_tracking': self._objective_tracking(
                ddx_hlh_r,
                desired_hlh_r,
            ),
            'hrh_translational_tracking': self._objective_tracking(
                ddx_hrh_p,
                desired_hrh_p,
            ),
            'hrh_rotational_tracking': self._objective_tracking(
                ddx_hrh_r,
                desired_hrh_r,
            ),            
            'tlf_translational_tracking': self._objective_tracking(
                ddx_tlf_p,
                desired_tlf_p,
            ),
            'tlf_rotational_tracking': self._objective_tracking(
                ddx_tlf_r,
                desired_tlf_r,
            ),
            'tlr_translational_tracking': self._objective_tracking(
                ddx_tlr_p,
                desired_tlr_p,
            ),
            'tlr_rotational_tracking': self._objective_tracking(
                ddx_tlr_r,
                desired_tlr_r,
            ),
            'trf_translational_tracking': self._objective_tracking(
                ddx_trf_p,
                desired_trf_p,
            ),
            'trf_rotational_tracking': self._objective_tracking(
                ddx_trf_r,
                desired_trf_r,
            ),
            'trr_translational_tracking': self._objective_tracking(
                ddx_trr_p,
                desired_trr_p,
            ),
            'trr_rotational_tracking': self._objective_tracking(
                ddx_trr_r,
                desired_trr_r,
            ),
            'hlf_translational_tracking': self._objective_tracking(
                ddx_hlf_p,
                desired_hlf_p,
            ),
            'hlf_rotational_tracking': self._objective_tracking(
                ddx_hlf_r,
                desired_hlf_r,
            ),
            'hlr_translational_tracking': self._objective_tracking(
                ddx_hlr_p,
                desired_hlr_p,
            ),
            'hlr_rotational_tracking': self._objective_tracking(
                ddx_hlr_r,
                desired_hlr_r,
            ),
            'hrf_translational_tracking': self._objective_tracking(
                ddx_hrf_p,
                desired_hrf_p,
            ),
            'hrf_rotational_tracking': self._objective_tracking(
                ddx_hrf_r,
                desired_hrf_r,
            ),
            'hrr_translational_tracking': self._objective_tracking(
                ddx_hrr_p,
                desired_hrr_p,
            ),
            'hrr_rotational_tracking': self._objective_tracking(
                ddx_hrr_r,
                desired_hrr_r,
            ),
            'torque': self._objective_regularization(u),
            'regularization': self._objective_regularization(q),
        }

        objective_terms = {
            k: v * self.weights_config[k] for k, v in objective_terms.items()
        }
        objective_value = sum(objective_terms.values())

        return objective_value

    def _objective_tracking(
        self, q: MX, task_target: MX,
    ) -> MX:
        """Tracking Objective Function."""
        return casadi.sumsqr(q - task_target)

    def _objective_regularization(
        self, q: MX,
    ) -> MX:
        """Regularization Objective Function."""
        return casadi.sumsqr(q)

    def generate_functions(self):
        # Define symbolic variables:
        dv = casadi.MX.sym("dv", self.dv_size)
        u = casadi.MX.sym("u", self.u_size)
        z = casadi.MX.sym("z", self.z_size)

        design_vector = casadi.vertcat(dv, u, z)

        M = casadi.MX.sym("M", self.dv_size, self.dv_size)
        C = casadi.MX.sym("C", self.dv_size)
        J_contact = casadi.MX.sym("J_contact", self.dv_size, self.z_size)
        desired_task_ddx = casadi.MX.sym("desired_task_ddx", self.num_site_ids, 6)
        J_task = casadi.MX.sym("J_task", self.num_site_ids * 6, self.dv_size)
        task_bias = casadi.MX.sym("task_bias", self.num_site_ids * 6)

        equality_constraint_input = [
            design_vector,
            M,
            C,
            J_contact,
        ]

        inequality_constraint_input = [
            design_vector,
        ]

        objective_input = [
            design_vector,
            desired_task_ddx,
            J_task,
            task_bias,
        ]

        # Convert to CasADi Function:
        beq = casadi.Function(
            "beq",
            equality_constraint_input,
            [-self.equality_constraints(*equality_constraint_input)],
        )

        Aeq = casadi.Function(
            "Aeq",
            equality_constraint_input,
            [casadi.densify(casadi.jacobian(
                self.equality_constraints(*equality_constraint_input),
                design_vector,
            ))],
        )

        bineq = casadi.Function(
            "bineq",
            inequality_constraint_input,
            [-self.inequality_constraints(*inequality_constraint_input)],
        )

        Aineq = casadi.Function(
            "Aineq",
            inequality_constraint_input,
            [casadi.densify(casadi.jacobian(
                self.inequality_constraints(*inequality_constraint_input),
                design_vector,
            ))],
        )

        hessian, gradient = casadi.hessian(
            self.objective(*objective_input),
            design_vector,
        )

        H = casadi.Function(
            "H",
            objective_input,
            [casadi.densify(hessian)],
        )

        f = casadi.Function(
            "f",
            objective_input,
            [casadi.densify(gradient)],
        )

        beq_size = beq.size_out(0)
        self.beq_sz = beq_size[0] * beq_size[1]
        Aeq_size = Aeq.size_out(0)
        self.Aeq_sz = Aeq_size[0] * Aeq_size[1]
        self.Aeq_rows = Aeq_size[0]
        self.Aeq_cols = Aeq_size[1]
        bineq_size = bineq.size_out(0)
        self.bineq_sz = bineq_size[0] * bineq_size[1]
        Aineq_size = Aineq.size_out(0)
        self.Aineq_sz = Aineq_size[0] * Aineq_size[1]
        self.Aineq_rows = Aineq_size[0]
        self.Aineq_cols = Aineq_size[1]
        H_size = H.size_out(0)
        self.H_sz = H_size[0] * H_size[1]
        self.H_rows = H_size[0]
        self.H_cols = H_size[1]
        f_size = f.size_out(0)
        self.f_sz = f_size[0] * f_size[1]

        # Generate C++ Code:
        opts = {
            "cpp": True,
            "with_header": True,
        }
        filenames = [
            "autogen_functions",
        ]
        casadi_functions = [
            [beq, Aeq, bineq, Aineq, H, f],
        ]
        loop_iterables = zip(
            filenames,
            casadi_functions,
        )

        for filename, casadi_function in loop_iterables:
            generator = casadi.CodeGenerator(f"{filename}.cc", opts)
            for function in casadi_function:
                generator.add(function)
        generator.generate(FLAGS.filepath+"/")

    def generate_defines(self):
        cc_code = f"""#pragma once
#include <array>
#include <string_view>

using namespace std::string_view_literals;

namespace operational_space_controller::constants {{
    namespace model {{
        // Mujoco Model Constants:
        constexpr int nq_size  = {self.mj_model.nq};
        constexpr int nv_size = {self.mj_model.nv};
        constexpr int nu_size  = {self.mj_model.nu};
        constexpr int body_ids_size = {self.num_body_ids};
        constexpr int site_ids_size = {self.num_site_ids};
        constexpr int noncontact_site_ids_size = {self.num_noncontact_site_ids};
        constexpr int contact_site_ids_size = {self.num_contact_site_ids};
        constexpr std::array body_list = {{{", ".join(self.body_list)}}};
        constexpr std::array site_list = {{{", ".join(self.site_list)}}};
        constexpr std::array noncontact_site_list = {{{", ".join(self.noncontact_site_list)}}};
        constexpr std::array contact_site_list = {{{", ".join(self.contact_site_list)}}};
    }}
    namespace optimization {{
        // Optimization Constants:
        constexpr int dv_size = {self.dv_size};
        constexpr int u_size = {self.u_size};
        constexpr int z_size = {self.z_size};
        constexpr int design_vector_size = {self.design_vector_size};
        constexpr int dv_idx = {self.dv_idx};
        constexpr int u_idx = {self.u_idx};
        constexpr int z_idx = {self.z_idx};
        constexpr int beq_sz = {self.beq_sz};
        constexpr int Aeq_sz = {self.Aeq_sz};
        constexpr int Aeq_rows = {self.Aeq_rows};
        constexpr int Aeq_cols = {self.Aeq_cols};
        constexpr int bineq_sz = {self.bineq_sz};
        constexpr int Aineq_sz = {self.Aineq_sz};
        constexpr int Aineq_rows = {self.Aineq_rows};
        constexpr int Aineq_cols = {self.Aineq_cols};
        constexpr int H_sz = {self.H_sz};
        constexpr int H_rows = {self.H_rows};
        constexpr int H_cols = {self.H_cols};
        constexpr int f_sz = {self.f_sz};
    }}
}}
        """

        filepath = os.path.join(FLAGS.filepath, "autogen_defines.h")
        with open(filepath, "w") as f:
            f.write(cc_code)


def main(argv):
    # Initialize Mujoco Model:
    r = Runfiles.Create()
    mj_model = mujoco.MjModel.from_xml_path(
        r.Rlocation(
            path="mujoco-models/models/walter_sr/WaLTER_Senior_v2_ulim2.xml",
        )
    )

    # Generate Functions:
    autogen = AutoGen(mj_model)
    autogen.generate_functions()
    autogen.generate_defines()


if __name__ == "__main__":
    app.run(main)
