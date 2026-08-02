#include <filesystem>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <string>

#include "absl/status/status.h"
#include "absl/log/absl_check.h"
#include "rules_cc/cc/runfiles/runfiles.h"

#include "mujoco/mujoco.h"
#include "Eigen/Dense"
#include "GLFW/glfw3.h"
#include "osqp++.h" 

#include "operational-space-control/walter_sr/aliases.h"
#include "operational-space-control/walter_sr/containers.h"
#include "operational-space-control/walter_sr/constants.h"
#include "operational-space-control/walter_sr/operational_space_controller.h"

using namespace operational_space_controller::aliases;
using namespace operational_space_controller::containers;
using namespace operational_space_controller::constants;
using rules_cc::cc::runfiles::Runfiles;

class SimLogger {
private:
    std::map<std::string, std::vector<double>> data_map;
    std::vector<std::string> headers; 
    size_t rows = 0;

public:
    void reserve(size_t estimated_steps) {
        for (auto& pair : data_map) {
            pair.second.reserve(estimated_steps);
        }
    }

    void log(const std::string& name, double value) {
        if (data_map.find(name) == data_map.end()) {
            headers.push_back(name);
            data_map[name].reserve(rows + 1000); 
            data_map[name].resize(rows, 0.0); 
        }
        data_map[name].push_back(value);
    }
    
    void logArray(const std::string& prefix, const double* arr, int size) {
        for (int i = 0; i < size; ++i) {
            log(prefix + "_" + std::to_string(i), arr[i]);
        }
    }

    template <typename Derived>
    void logEigen(const std::string& prefix, const Eigen::MatrixBase<Derived>& vec) {
        for (int i = 0; i < vec.size(); ++i) {
            log(prefix + "_" + std::to_string(i), vec(i));
        }
    }

    void endStep() { rows++; }

    void save(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) return;
        for (size_t i = 0; i < headers.size(); ++i) {
            file << headers[i];
            if (i < headers.size() - 1) file << ",";
        }
        file << "\n";
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < headers.size(); ++c) {
                file << data_map[headers[c]][r];
                if (c < headers.size() - 1) file << ",";
            }
            file << "\n";
        }
        std::cout << "Data saved to " << filename << " (" << rows << " rows)" << std::endl;
    }
};

double get_propeller_leg_height(
    const Eigen::Quaterniond& body_quat, 
    double q_hip, double q_knee, 
    double q_hip_offset, double q_knee_offset,
    double L_thigh, double L_shin, double R_wheel) 
{
    double theta_hip_calibrated  = q_hip + q_hip_offset;
    double theta_knee_calibrated = q_knee + q_knee_offset;
    Eigen::Vector3d v_thigh(0, 0, -L_thigh);                                   
    double y_offset = -0.04675; 
    Eigen::Vector3d v_wheel_A( L_shin, y_offset, 0); 
    Eigen::Vector3d v_wheel_B(-L_shin, y_offset, 0); 
    double global_shin = theta_hip_calibrated + theta_knee_calibrated;         
    Eigen::AngleAxisd rot_thigh(theta_hip_calibrated, Eigen::Vector3d::UnitY());      
    Eigen::AngleAxisd rot_shin(global_shin,            Eigen::Vector3d::UnitY());
    Eigen::Vector3d p_knee = rot_thigh * v_thigh;                               
    Eigen::Vector3d p_wheel_A = p_knee + (rot_shin * v_wheel_A);               
    Eigen::Vector3d p_wheel_B = p_knee + (rot_shin * v_wheel_B);
    Eigen::Vector3d g_A = body_quat * p_wheel_A;                               
    Eigen::Vector3d g_B = body_quat * p_wheel_B;
    return std::max(-g_A.z(), -g_B.z()) + R_wheel;                             
}

template <typename T>
bool contains(const std::vector<T>& vec, const T& value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

std::vector<int> getSiteIdsOnSameBodyAsGeom(const mjModel* m, int geom_id) {
    std::vector<int> associated_site_ids; 
    if (geom_id < 0 || geom_id >= m->ngeom) return associated_site_ids; 
    int geom_body_id = m->geom_bodyid[geom_id];
    for (int i = 0; i < m->nsite; ++i) {
        if (m->site_bodyid[i] == geom_body_id) {
            associated_site_ids.push_back(i); 
        }
    }
    return associated_site_ids; 
}

std::vector<int> getBinaryRepresentation_std_find(const std::vector<int>& A, const std::vector<int>& B) {
    std::vector<int> C;
    C.reserve(B.size());
    for (int b_element : B) {
        auto it = std::find(A.begin(), A.end(), b_element);
        C.push_back((it != A.end()) ? 1 : 0);
    }
    return C;
}

int main(int argc, char** argv) {
    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error));
    std::filesystem::path osc_model_path = runfiles->Rlocation("mujoco-models/models/walter_sr/WaLTER_Senior_v2_ulim.xml");
    std::filesystem::path simulation_model_path = runfiles->Rlocation("mujoco-models/models/walter_sr/scene_walter_sr_v2_ulim.xml");

    char mj_error[1000];
    mjModel* mj_model = mj_loadXML(simulation_model_path.c_str(), nullptr, mj_error, 1000);
    if (!mj_model) { printf("%s\n", mj_error); return 1; }
    mjData* mj_data = mj_makeData(mj_model);

    mj_resetDataKeyframe(mj_model, mj_data, 5);
    mj_forward(mj_model, mj_data);

    mjvCamera cam; mjvPerturb pert; mjvOption opt; mjvScene scn; mjrContext con;
    glfwInit();
    GLFWwindow* window = glfwCreateWindow(1200, 600, "Sim + Plot", NULL, NULL); 
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    mjv_defaultCamera(&cam); mjv_defaultPerturb(&pert); mjv_defaultOption(&opt);
    mjv_defaultScene(&scn); mjr_defaultContext(&con);
    mjv_makeScene(mj_model, &scn, 1000); mjr_makeContext(mj_model, &con, mjFONTSCALE_300);

    // --- SETUP SINGLE GIANT PLOT FOR FORCES ---
    mjvFigure fig;
    mjv_defaultFigure(&fig);
    strcpy(fig.title, "Head Right Ground Force Fz (N)");
    strcpy(fig.xlabel, "Time (s)");
    fig.flg_extend = 0;        
    fig.range[0][0] = 0.0f;  fig.range[0][1] = 5.0f;  fig.gridsize[0] = 6;      
    strcpy(fig.xformat, "%.0f"); 
    fig.range[1][0] = -50.0f; fig.range[1][1] =  250.0f; fig.gridsize[1] = 11;     
    strcpy(fig.yformat, "%.0f"); 
    fig.flg_barplot = 0;     
    strcpy(fig.linename[0], "Solver Plan Fz");
    strcpy(fig.linename[1], "Actual Sim Fz");
    fig.linergb[0][0] = 1.0f; fig.linergb[0][1] = 0.0f; fig.linergb[0][2] = 0.0f; // Red (Plan)
    fig.linergb[1][0] = 0.0f; fig.linergb[1][1] = 0.0f; fig.linergb[1][2] = 1.0f; // Blue (Actual)
    
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    glfwSwapBuffers(window);
    glfwPollEvents();

    OperationalSpaceController controller(osc_model_path);

    Vector<model::nq_size> qpos = Eigen::Map<Vector<model::nq_size>>(mj_data->qpos);
    Vector<model::nv_size> qvel = Eigen::Map<Vector<model::nv_size>>(mj_data->qvel);
    Vector<model::nv_size> qfrc_actuator = Eigen::Map<Vector<model::nv_size>>(mj_data->qfrc_actuator);
    
    Vector<3> initial_position = qpos(Eigen::seqN(0, 3));
    Eigen::Matrix<double, model::site_ids_size, 3> site_data;
    Eigen::Matrix<double, model::site_ids_size, 3> initial_site_data;

    Eigen::Vector<double, model::contact_site_ids_size> contact_check2;
    Eigen::Vector<double, model::contact_site_ids_size> contact_check_temp_raw_phys;

    State initial_state;
    initial_state.motor_position = qpos(Eigen::seqN(7, model::nu_size));
    initial_state.motor_velocity = qvel(Eigen::seqN(6, model::nu_size));
    initial_state.torque_estimate = qfrc_actuator(Eigen::seqN(6, model::nu_size));
    initial_state.body_rotation = qpos(Eigen::seqN(3, 4));
    initial_state.linear_body_velocity = qvel(Eigen::seqN(0, 3));
    initial_state.angular_body_velocity = qvel(Eigen::seqN(3, 3));
    initial_state.contact_mask = Vector<model::contact_site_ids_size>::Constant(0.0);

    TaskspaceTargets taskspace_targets = Matrix<model::site_ids_size, 6>::Zero();

    absl::Status result;
    result.Update(controller.initialize(initial_state));
    result.Update(controller.initialize_optimization());
    ABSL_CHECK(result.ok()) << result.message();

    controller.update_taskspace_targets(taskspace_targets);
    result.Update(controller.initialize_thread());
    ABSL_CHECK(result.ok()) << result.message();

    double visualization_timer = mj_data->time;
    double visualization_start_time = visualization_timer;
    double visualization_interval = 0.01;
    double simulation_time = 10.0;
    auto current_time = mj_data->time;
    double last_time = current_time;

    std::vector<std::string> sites;
    std::vector<int> site_ids;
    std::vector<double> data_time;

    // Vectors for Plot (Forces)
    std::vector<double> data_fz_plan;
    std::vector<double> data_fz_act;
    data_fz_plan.reserve(30000); data_fz_act.reserve(30000);

    for(const std::string_view& site : model::site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model, mjOBJ_SITE, site_str.data());
        sites.push_back(site_str);
        site_ids.push_back(id);
    }

    initial_site_data = Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->site_xpos)(site_ids, Eigen::placeholders::all);

    double initial_tl_angular_position = mj_data->qpos[mj_model->jnt_qposadr[2]];
    double initial_tr_angular_position = mj_data->qpos[mj_model->jnt_qposadr[4]];
    double initial_hl_angular_position = mj_data->qpos[mj_model->jnt_qposadr[6]];
    double initial_hr_angular_position = mj_data->qpos[mj_model->jnt_qposadr[8]];
    
    double last_tl_angular_position = initial_tl_angular_position;
    double last_tr_angular_position = initial_tr_angular_position;
    double last_hl_angular_position = initial_hl_angular_position;
    double last_hr_angular_position = initial_hr_angular_position;

    Vector<3> last_tlh_linear_position = initial_site_data(5,Eigen::seqN(0, 3));
    Vector<3> last_trh_linear_position = initial_site_data(6,Eigen::seqN(0, 3));
    Vector<3> last_hlh_linear_position = initial_site_data(7,Eigen::seqN(0, 3));
    Vector<3> last_hrh_linear_position = initial_site_data(8,Eigen::seqN(0, 3));

    std::vector<int> wheel_site_ids_ref = {3, 4, 7, 8, 11, 12, 15, 16}; 
    std::vector<int> wheel_geom_ids = {4, 5, 8, 9, 13, 14, 17, 18};    

    SimLogger logger;
    logger.reserve(30000); 

    double soft_switch_max_force = 77.0*10.0;
    double soft_switch_ramp_time = 0.5; 
    bool enable_soft_switch = 0;

    Eigen::Vector<double, model::contact_site_ids_size> contact_start_times;
    contact_start_times.setConstant(-100.0); 
    Eigen::Vector<double, model::contact_site_ids_size> prev_contact_mask;
    prev_contact_mask.setZero();
    
    double L_THIGH = 0.1016; 
    double L_SHIN = 0.08255/1.0; 
    double L_WHEEL = 0.0635;

    // =========================================================================================
    // LOGIC LOOP
    // =========================================================================================
    while(current_time < simulation_time) {
        
        mj_step(mj_model, mj_data);
        mj_fwdPosition(mj_model, mj_data);
        mj_fwdVelocity(mj_model, mj_data);

        current_time = mj_data->time;
        visualization_timer = current_time - visualization_start_time;

        qpos = Eigen::Map<Vector<model::nq_size>>(mj_data->qpos);
        qvel = Eigen::Map<Vector<model::nv_size>>(mj_data->qvel);
        qfrc_actuator = Eigen::Map<Vector<model::nv_size>>(mj_data->qfrc_actuator);

        site_data = Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->site_xpos)(site_ids, Eigen::placeholders::all);

        std::vector<int> contact_site_ids_found;
        for (int i = 0; i < mj_data->ncon; ++i) {
            int geom1 = mj_data->contact[i].geom[0];
            int geom2 = mj_data->contact[i].geom[1];
            int active_wheel_geom_id = -1;
            if (contains(wheel_geom_ids, geom1)) active_wheel_geom_id = geom1;
            else if (contains(wheel_geom_ids, geom2)) active_wheel_geom_id = geom2;

            if (active_wheel_geom_id != -1) {
                std::vector<int> sites_on_body = getSiteIdsOnSameBodyAsGeom(mj_model, active_wheel_geom_id);
                if (!sites_on_body.empty()) contact_site_ids_found.push_back(sites_on_body[0]);
            }
        } 
        std::vector<int> contact_check_temp = getBinaryRepresentation_std_find(contact_site_ids_found, wheel_site_ids_ref);
        contact_check2 = Eigen::Map<Eigen::VectorXi>(contact_check_temp.data(), contact_check_temp.size()).cast<double>();
        contact_check_temp_raw_phys = Eigen::Map<Eigen::VectorXi>(contact_check_temp.data(), contact_check_temp.size()).cast<double>();

        double shin_rot_vel = 0.8*0.0; 

        // =========================================================
        // FORCE LIMITS (Sent to solver z_ub)
        // =========================================================
        Eigen::Vector<double, model::contact_site_ids_size> current_force_limits;
        for(int i=0; i < model::contact_site_ids_size; ++i) {
            bool is_contact = (contact_check2[i] > 0.5);
            bool was_contact = (prev_contact_mask[i] > 0.5);
            if (is_contact && !was_contact) contact_start_times[i] = current_time;

            double limit = 0.0;
            if (is_contact) {
                if (enable_soft_switch) {
                    double duration_in_contact = current_time - contact_start_times[i];
                    double ratio = std::clamp(duration_in_contact / soft_switch_ramp_time, 0.0, 1.0);
                    limit = ratio * soft_switch_max_force;
                } else {
                    limit = soft_switch_max_force;
                }                
            }
            current_force_limits[i] = limit;
        }
        prev_contact_mask = contact_check2;

        State state;
        state.motor_position = qpos(Eigen::seqN(7, model::nu_size));
        state.motor_velocity = qvel(Eigen::seqN(6, model::nu_size));
        state.torque_estimate = qfrc_actuator(Eigen::seqN(6, model::nu_size));
        state.body_rotation = qpos(Eigen::seqN(3, 4));
        state.linear_body_velocity = qvel(Eigen::seqN(0, 3));
        state.angular_body_velocity = qvel(Eigen::seqN(3, 3));
        state.contact_mask = contact_check_temp_raw_phys; 
        
        controller.update_state(state);
        // controller.update_max_contact_forces(current_force_limits); 
        
        TaskspaceTargets taskspace_targets = TaskspaceTargets::Zero();

        double phase_offset = 0.0;
        double tl_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[2]];        
        double tr_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[4]];        
        double hl_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[6]];        
        double hr_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[8]];

        double tl_angular_position = tl_shin_angle;
        double tr_angular_position = tr_shin_angle;
        double hl_angular_position = hl_shin_angle;
        double hr_angular_position = hr_shin_angle;

        double dt = current_time - last_time;
        if (dt == 0) dt = 0.0001;

        double tl_angular_velocity = (tl_angular_position - last_tl_angular_position)/dt;
        double tr_angular_velocity = (tr_angular_position - last_tr_angular_position)/dt;
        double hl_angular_velocity = (hl_angular_position - last_hl_angular_position)/dt;
        double hr_angular_velocity = (hr_angular_position - last_hr_angular_position)/dt;

        // Targets
        double tl_angular_position_target = initial_tl_angular_position + shin_rot_vel * current_time + phase_offset;
        double tr_angular_position_target = initial_tr_angular_position + shin_rot_vel * current_time + phase_offset;
        double hl_angular_position_target = initial_hl_angular_position + shin_rot_vel * current_time;
        double hr_angular_position_target = initial_hr_angular_position + shin_rot_vel * current_time;

        double tl_angular_position_error = (tl_angular_position_target - tl_angular_position);
        double tr_angular_position_error = (tr_angular_position_target - tr_angular_position);
        double hl_angular_position_error = (hl_angular_position_target - hl_angular_position);
        double hr_angular_position_error = (hr_angular_position_target - hr_angular_position);
        
        double tl_angular_velocity_error = (shin_rot_vel - tl_angular_velocity);
        double tr_angular_velocity_error = (shin_rot_vel - tr_angular_velocity);
        double hl_angular_velocity_error = (shin_rot_vel - hl_angular_velocity);
        double hr_angular_velocity_error = (shin_rot_vel - hr_angular_velocity);

        double shin_kp = 30.0*3.0; 
        double shin_kv = 3.0*3.0;

        double tl_angular_control = shin_kp * (tl_angular_position_error) + shin_kv * (tl_angular_velocity_error);
        double tr_angular_control = shin_kp * (tr_angular_position_error) + shin_kv * (tr_angular_velocity_error);
        double hl_angular_control = shin_kp * (hl_angular_position_error) + shin_kv * (hl_angular_velocity_error);
        double hr_angular_control = shin_kp * (hr_angular_position_error) + shin_kv * (hr_angular_velocity_error);
        
        last_tl_angular_position = tl_angular_position;
        last_tr_angular_position = tr_angular_position;
        last_hl_angular_position = hl_angular_position;
        last_hr_angular_position = hr_angular_position;

        Eigen::Vector<double, 6> cmd1 {0, 0, 0, 0, tl_angular_control, 0};        
        Eigen::Vector<double, 6> cmd2 {0, 0, 0, 0, tr_angular_control, 0};        
        Eigen::Vector<double, 6> cmd3 {0, 0, 0, 0, hl_angular_control, 0};        
        Eigen::Vector<double, 6> cmd4 {0, 0, 0, 0, hr_angular_control, 0};        

        taskspace_targets.row(1) = cmd1;
        taskspace_targets.row(2) = cmd2;
        taskspace_targets.row(3) = cmd3;
        taskspace_targets.row(4) = cmd4;

        // HEIGHT PD
        double thigh_lin_kp = 30.0*2.0; 
        double thigh_lin_kv = 3.0*2.0; 
        
        double HIP_OFFSET  = 0.0; 
        double KNEE_OFFSET = 0.0;     
        Eigen::Quaterniond body_quat(qpos(3), qpos(4), qpos(5), qpos(6));
        double hr_hip  = mj_data->qpos[13]; 
        double hr_knee = mj_data->qpos[14]; 
        double hl_hip  = mj_data->qpos[11]; 
        double hl_knee = mj_data->qpos[12]; 
        double tr_hip  = mj_data->qpos[9]; 
        double tr_knee = mj_data->qpos[10]; 
        double tl_hip  = mj_data->qpos[7]; 
        double tl_knee = mj_data->qpos[8]; 

        double h_hr_kinematic = get_propeller_leg_height(body_quat, hr_hip, hr_knee, HIP_OFFSET, KNEE_OFFSET, L_THIGH, L_SHIN, L_WHEEL);        
        double h_hl_kinematic = get_propeller_leg_height(body_quat, hl_hip, hl_knee, HIP_OFFSET, KNEE_OFFSET, L_THIGH, L_SHIN, L_WHEEL);
        double h_tr_kinematic = get_propeller_leg_height(body_quat, tr_hip, tr_knee, HIP_OFFSET, KNEE_OFFSET, L_THIGH, L_SHIN, L_WHEEL);
        double h_tl_kinematic = get_propeller_leg_height(body_quat, tl_hip, tl_knee, HIP_OFFSET, KNEE_OFFSET, L_THIGH, L_SHIN, L_WHEEL);

        Vector<3> tlh_linear_position = Vector<3>(0.0, 0.0, h_tl_kinematic);
        Vector<3> trh_linear_position = Vector<3>(0.0, 0.0, h_tr_kinematic);
        Vector<3> hlh_linear_position = Vector<3>(0.0, 0.0, h_hl_kinematic);
        Vector<3> hrh_linear_position = Vector<3>(0.0, 0.0, h_hr_kinematic);
        
        Vector<3> tlh_linear_velocity = (tlh_linear_position - last_tlh_linear_position)/dt;
        Vector<3> trh_linear_velocity = (trh_linear_position - last_trh_linear_position)/dt;
        Vector<3> hlh_linear_velocity = (hlh_linear_position - last_hlh_linear_position)/dt;
        Vector<3> hrh_linear_velocity = (hrh_linear_position - last_hrh_linear_position)/dt;

        double tlh_linear_velocity_target = 0.0;
        double trh_linear_velocity_target = 0.0;
        double hlh_linear_velocity_target = 0.0;
        double hrh_linear_velocity_target = 0.0;

        double thigh_height_increase_stairs = +0.01;
        double tlh_linear_position_target = initial_site_data(5,2) - 0.0 + thigh_height_increase_stairs;
        double trh_linear_position_target = initial_site_data(6,2) - 0.0 + thigh_height_increase_stairs;
        double hlh_linear_position_target = initial_site_data(7,2) - 0.0 + thigh_height_increase_stairs + 0.01*0.0;
        double hrh_linear_position_target = initial_site_data(8,2) - 0.0 + thigh_height_increase_stairs + 0.01*0.0;

        double tlh_linear_position_error = (tlh_linear_position_target - tlh_linear_position(2));
        double trh_linear_position_error = (trh_linear_position_target - trh_linear_position(2));
        double hlh_linear_position_error = (hlh_linear_position_target - hlh_linear_position(2));
        double hrh_linear_position_error = (hrh_linear_position_target - hrh_linear_position(2));

        double tlh_linear_velocity_error = (tlh_linear_velocity_target - tlh_linear_velocity(2));
        double trh_linear_velocity_error = (trh_linear_velocity_target - trh_linear_velocity(2));
        double hlh_linear_velocity_error = (hlh_linear_velocity_target - hlh_linear_velocity(2));
        double hrh_linear_velocity_error = (hrh_linear_velocity_target - hrh_linear_velocity(2));

        double tlh_linear_control = thigh_lin_kp * (tlh_linear_position_error) + thigh_lin_kv * (tlh_linear_velocity_error);
        double trh_linear_control = thigh_lin_kp * (trh_linear_position_error) + thigh_lin_kv * (trh_linear_velocity_error);
        double hlh_linear_control = thigh_lin_kp * (hlh_linear_position_error) + thigh_lin_kv * (hlh_linear_velocity_error);
        double hrh_linear_control = thigh_lin_kp * (hrh_linear_position_error) + thigh_lin_kv * (hrh_linear_velocity_error);

        last_tlh_linear_position = tlh_linear_position;
        last_trh_linear_position = trh_linear_position;
        last_hlh_linear_position = hlh_linear_position;
        last_hrh_linear_position = hrh_linear_position;
        last_time = current_time;

        Eigen::Vector<double, 6> cmd5 {0, 0, tlh_linear_control, 0, 0, 0};
        Eigen::Vector<double, 6> cmd6 {0, 0, trh_linear_control, 0, 0, 0};
        Eigen::Vector<double, 6> cmd7 {0, 0, hlh_linear_control, 0, 0, 0};
        Eigen::Vector<double, 6> cmd8 {0, 0, hrh_linear_control, 0, 0, 0};        

        taskspace_targets.row(5) = cmd5;
        taskspace_targets.row(6) = cmd6;
        taskspace_targets.row(7) = cmd7;
        taskspace_targets.row(8) = cmd8;        

        // move cam
        Vector<3> body_position = qpos(Eigen::seqN(0, 3));
        cam.lookat[0] = body_position(0);

        // -------------------------------------------------------------------------------------
        // SOLVE
        // -------------------------------------------------------------------------------------
        std::cout.setstate(std::ios_base::failbit);
        controller.update_taskspace_targets(taskspace_targets);
        Vector<model::nu_size> torque_command = controller.get_torque_command();
        std::cout.clear();

        int solver_status_int = controller.get_solver_status(); 
        osqp::OsqpExitCode solver_status = static_cast<osqp::OsqpExitCode>(solver_status_int);
        if (solver_status != osqp::OsqpExitCode::kOptimal) {
             std::cout << "Solver failed with status: " << (int)solver_status << " at time " << current_time << ". Exiting loop." << std::endl;
             break; 
        }

        Vector<optimization::design_vector_size> solution = controller.get_solution();

        // =====================================================================
        // SMOKING GUN: EXTRACT CONTACT FORCES (Plan vs Reality)
        // =====================================================================
        double plan_hrf_fz = solution[optimization::u_idx + 20];
        double plan_hrr_fz = solution[optimization::u_idx + 23];
        double curr_plan_total_hr_fz = plan_hrf_fz + plan_hrr_fz;

        double curr_act_hrf_fz = 0.0;
        double curr_act_hrr_fz = 0.0;
        int hrf_geom_id = mj_name2id(mj_model, mjOBJ_GEOM, "head_right_front_wheel_geom");
        int hrr_geom_id = mj_name2id(mj_model, mjOBJ_GEOM, "head_right_rear_wheel_geom");

        for (int i = 0; i < mj_data->ncon; ++i) {
            int g1 = mj_data->contact[i].geom[0];
            int g2 = mj_data->contact[i].geom[1];
            if (g1 == hrf_geom_id || g2 == hrf_geom_id) {
                double f_contact[6];
                mj_contactForce(mj_model, mj_data, i, f_contact);
                curr_act_hrf_fz += std::abs(f_contact[0]); 
            }
            if (g1 == hrr_geom_id || g2 == hrr_geom_id) {
                double f_contact[6];
                mj_contactForce(mj_model, mj_data, i, f_contact);
                curr_act_hrr_fz += std::abs(f_contact[0]);
            }
        }
        double curr_act_total_hr_fz = curr_act_hrf_fz + curr_act_hrr_fz;



// =====================================================================
        // SMOKING GUN: EXTRACT CONTACT FORCES (Plan vs Reality)
        // =====================================================================
        
        // 1. Array to hold the actual measured Fz for each of the 8 contacts
        std::vector<double> current_actual_fz(model::contact_site_ids_size, 0.0);

        // 2. Accumulate actual MuJoCo forces for all wheels
        for (int i = 0; i < mj_data->ncon; ++i) {
            int g1 = mj_data->contact[i].geom[0];
            int g2 = mj_data->contact[i].geom[1];
            
            for (size_t j = 0; j < wheel_geom_ids.size(); ++j) {
                if (g1 == wheel_geom_ids[j] || g2 == wheel_geom_ids[j]) {
                    double f_contact[6];
                    mj_contactForce(mj_model, mj_data, i, f_contact);
                    current_actual_fz[j] += std::abs(f_contact[0]); 
                }
            }
        }

        // 3. Array to hold the Solver's planned Fz for all wheels
        std::vector<double> current_plan_fz(model::contact_site_ids_size, 0.0);
        for (int j = 0; j < model::contact_site_ids_size; ++j) {
            current_plan_fz[j] = solution[optimization::u_idx + (3 * j) + 2];
        }



        // 6. Print the Side-by-Side Table to the Terminal
        static int table_counter = 0;
        if (table_counter++ % 50 == 0 && current_time > 0.01) { 
            std::cout << "\n=== ALL WHEELS Fz COMPARISON TABLE ===\n";
            std::cout << std::setw(8) << "Contact" 
                      << " | " << std::setw(12) << "Plan Fz (N)" 
                      << " | " << std::setw(12) << "Actual Fz (N)" 
                      << " | " << std::setw(12) << "Delta (N)" << "\n";
            std::cout << "--------------------------------------------------------\n";
            
            double total_plan = 0.0;
            double total_act = 0.0;

            for (int j = 0; j < model::contact_site_ids_size; ++j) {
                double plan = current_plan_fz[j];
                double act = current_actual_fz[j];
                double delta = std::abs(plan - act);
                
                total_plan += plan;
                total_act += act;

                std::cout << "   " << std::setw(5) << j 
                          << " | " << std::setw(12) << plan
                          << " | " << std::setw(12) << act
                          << " | " << std::setw(12) << delta << "\n";
            }
            std::cout << "--------------------------------------------------------\n";
            std::cout << "   TOTAL" 
                      << " | " << std::setw(12) << total_plan
                      << " | " << std::setw(12) << total_act
                      << " | " << std::setw(12) << std::abs(total_plan - total_act) << "\n";
        }        






        // Push data to plot vectors
        data_time.push_back(current_time);
        data_fz_plan.push_back(curr_plan_total_hr_fz); 
        data_fz_act.push_back(curr_act_total_hr_fz);

        static int diag_counter = 0;
        if (diag_counter++ % 50 == 0 && current_time > 0.01) { 
            std::cout << "\n=== Fz HALLUCINATION CHECK (Time: " << std::fixed << std::setprecision(3) << current_time << " s) ===" << std::endl;
            std::cout << "  Solver Plan Fz : " << std::setw(8) << curr_plan_total_hr_fz << " N" << std::endl;
            std::cout << "  Reality Sim Fz : " << std::setw(8) << curr_act_total_hr_fz << " N" << std::endl;
            std::cout << "  GAP (The Lie)  : " << std::setw(8) << std::abs(curr_plan_total_hr_fz - curr_act_total_hr_fz) << " N" << std::endl;
        }

        mj_data->ctrl = torque_command.data();
        logger.endStep();

        // -------------------------------------------------------------------------------------
        // VISUALIZATION (User's Exact Sequential Loop)
        // -------------------------------------------------------------------------------------
        if(visualization_timer > visualization_interval) {
            visualization_start_time = mj_data->time;

            int total_history = data_time.size();
            int max_points = 1000; 
            int count_to_plot = std::min(total_history, max_points);
            int start_idx = total_history - count_to_plot;

            // Plot sequentially backward from the latest data. No skipping indices.
            for (int k = 0; k < count_to_plot; ++k) {
                int i = start_idx + k; 

                fig.linedata[0][2*k]   = (float)data_time[i];      
                fig.linedata[0][2*k+1] = (float)data_fz_plan[i];     
                fig.linedata[1][2*k]   = (float)data_time[i];      
                fig.linedata[1][2*k+1] = (float)data_fz_act[i];     
            }
            
            fig.linepnt[0] = count_to_plot; 
            fig.linepnt[1] = count_to_plot; 

            // Scroll the X-axis continuously with the latest data
            if (!fig.flg_extend && count_to_plot > 0) {
                float latest_t = (float)data_time.back();
                float window_size = 5.0f; // Show last 5 seconds of data
                fig.range[0][0] = std::max(0.0f, latest_t - window_size); 
                fig.range[0][1] = std::max(window_size, latest_t);        
            }

            mjrRect viewport_sim = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport_sim.width, &viewport_sim.height);

            int plot_width = viewport_sim.width / 2; 
            viewport_sim.width -= plot_width;

            mjrRect viewport_plot1 = {viewport_sim.width, 0, plot_width, viewport_sim.height};

            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport_sim, &scn, &con);
            mjr_figure(viewport_plot1, &fig, &con);

            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << "Time: " << mj_data->time << " s";
            mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport_sim, ss.str().c_str(), 0, &con);
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    result.Update(controller.stop_thread());
    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);
    ABSL_CHECK(result.ok()) << result.message();

    logger.save("osc_test_data.csv"); 
    return 0;
}