/**
 * @file main.cpp
 * @brief Simulation environment and Operational Space Control (OSC) loop for the WaLTER Senior robot.
 */

//================================================================================= Includes =========================
#include <filesystem>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <cstdio>
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

//================================================================================= Utilities & Loggers =========================

class SimLogger {
private:
    std::map<std::string, std::vector<double>> data_map;
    std::vector<std::string> headers; 
    size_t rows = 0;

public:
    void reserve(size_t estimated_steps) {
        for (auto& pair : data_map) pair.second.reserve(estimated_steps);
    }

    void log(const std::string& name, double value) {
        if (data_map.find(name) == data_map.end()) {
            headers.push_back(name);
            data_map[name].reserve(rows + 1000); 
            data_map[name].resize(rows, 0.0); 
        }
        data_map[name].push_back(value);
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
        if (m->site_bodyid[i] == geom_body_id) associated_site_ids.push_back(i); 
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

//============================================================================================================
//================================================================================= MAIN =====================
//============================================================================================================

int main(int argc, char** argv) {
    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error));

    std::filesystem::path osc_model_path = runfiles->Rlocation("mujoco-models/models/walter_sr/WaLTER_Senior_v2_ulim2.xml");
    std::filesystem::path simulation_model_path = runfiles->Rlocation("mujoco-models/models/walter_sr/scene_walter_sr_v2_ulim2.xml");

    char mj_error[1000];
    mjModel* mj_model = mj_loadXML(simulation_model_path.c_str(), nullptr, mj_error, 1000);
    if (!mj_model) { 
        printf("Failed to load MuJoCo model: %s\n", mj_error); 
        return 1; 
    }
    mjData* mj_data = mj_makeData(mj_model);

    mj_resetDataKeyframe(mj_model, mj_data, 8);
    mj_forward(mj_model, mj_data);

    mjvCamera cam; mjvPerturb pert; mjvOption opt; mjvScene scn; mjrContext con;
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }
    
    int win_width = 1920; 
    int win_height = 1080;
    GLFWwindow* window = glfwCreateWindow(win_width, win_height, "Sim", NULL, NULL); 
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); 

    mjv_defaultCamera(&cam);
    mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt);
    
    opt.flags[mjVIS_CONTACTPOINT] = 1; 

    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(mj_model, &scn, 1000);
    mjr_makeContext(mj_model, &con, mjFONTSCALE_300);

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    glfwSwapBuffers(window);
    glfwPollEvents();

    OperationalSpaceController controller(osc_model_path);

    bool vid_record_flag = true;
    FILE* ffmpeg_pipe = nullptr;
    unsigned char* rgb_buffer = nullptr;

    if (vid_record_flag){
        const int fps = 100; 
        std::string ffmpeg_cmd = "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size " + 
                                    std::to_string(win_width) + "x" + std::to_string(win_height) + 
                                    " -framerate " + std::to_string(fps) + 
                                    " -i - -vf vflip -c:v h264_nvenc -preset hq -b:v 10M -pix_fmt yuv420p /home/vivek/sim_comp.mp4";    
        ffmpeg_pipe = popen(ffmpeg_cmd.c_str(), "w");
        if (!ffmpeg_pipe) {
            std::cerr << "Failed to open FFmpeg pipe." << std::endl;
            return 1;
        }
        rgb_buffer = new unsigned char[win_width * win_height * 3];    
    }

    Vector<model::nq_size> qpos = Eigen::Map<Vector<model::nq_size>>(mj_data->qpos);
    Vector<model::nv_size> qvel = Eigen::Map<Vector<model::nv_size>>(mj_data->qvel);
    Vector<model::nv_size> qfrc_actuator = Eigen::Map<Vector<model::nv_size>>(mj_data->qfrc_actuator);
    
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
    double simulation_time = 15.0;        
    auto current_time = mj_data->time;
    double last_time = current_time;

    // =========================================================================================
    // DYNAMIC ID RESOLUTION (No Hardcoding/Magic Numbers)
    // =========================================================================================
    
    // Safely lookup Joint Position Index (qpos), abort if not found
    auto get_qpos_idx = [&](const std::string& jnt_name) -> int {
        int id = mj_name2id(mj_model, mjOBJ_JOINT, jnt_name.c_str());
        ABSL_CHECK(id != -1) << "Fatal Error: Joint '" << jnt_name << "' not found in MuJoCo model for qpos mapping.";
        return mj_model->jnt_qposadr[id];
    };
    
    // Safely lookup Joint Degree of Freedom Index (dof/qvel/qacc), abort if not found
    auto get_dof_idx = [&](const std::string& jnt_name) -> int {
        int id = mj_name2id(mj_model, mjOBJ_JOINT, jnt_name.c_str());
        ABSL_CHECK(id != -1) << "Fatal Error: Joint '" << jnt_name << "' not found in MuJoCo model for dof mapping.";
        return mj_model->jnt_dofadr[id];
    };

    // Helper to safely extract a 3D Site position by string name
    auto get_site_pos = [&](const std::string& site_name) -> Vector<3> {
        int site_id = mj_name2id(mj_model, mjOBJ_SITE, site_name.c_str());
        ABSL_CHECK(site_id != -1) << "Fatal Error: Site '" << site_name << "' not found in MuJoCo model.";
        return Vector<3>(mj_data->site_xpos[3 * site_id + 0],
                         mj_data->site_xpos[3 * site_id + 1],
                         mj_data->site_xpos[3 * site_id + 2]);
    };

    // Mapped to wheel site names defined in the XML
    std::vector<std::string> wheel_site_names = {
        "tlf_wheel_site", "tlr_wheel_site", 
        "trf_wheel_site", "trr_wheel_site", 
        "hlf_wheel_site", "hlr_wheel_site", 
        "hrf_wheel_site", "hrr_wheel_site"
    };

    std::vector<int> wheel_site_ids_ref;
    std::vector<int> wheel_geom_ids;

    std::cout << "--- Resolving Wheel IDs ---" << std::endl;
    for (const auto& site_name : wheel_site_names) {
        int site_id = mj_name2id(mj_model, mjOBJ_SITE, site_name.c_str());
        ABSL_CHECK(site_id != -1) << "Fatal Error: Wheel Site [" << site_name << "] not found in model!";
        wheel_site_ids_ref.push_back(site_id);

        int parent_body_id = mj_model->site_bodyid[site_id];

        int target_geom_id = -1;
        for (int i = 0; i < mj_model->ngeom; i++) {
            if (mj_model->geom_bodyid[i] == parent_body_id) {
                target_geom_id = i;
                break;
            }
        }
        wheel_geom_ids.push_back(target_geom_id);

        std::cout << "Site ID [" << site_id << "] (" << site_name << ") maps to Geom ID [" 
                  << target_geom_id << "] via Body ID [" << parent_body_id << "]" << std::endl;
    }
    std::cout << "---------------------------" << std::endl;

    // -------------------------------------------------------------------------------------
    // Resolve Target for Proof 2 
    // -------------------------------------------------------------------------------------
    std::string target_wheel_name = "hrf_wheel_site";
    int target_wheel_idx = -1;
    
    auto target_it = std::find(wheel_site_names.begin(), wheel_site_names.end(), target_wheel_name);
    if (target_it != wheel_site_names.end()) {
        target_wheel_idx = std::distance(wheel_site_names.begin(), target_it);
    }
    int target_site_id = mj_name2id(mj_model, mjOBJ_SITE, target_wheel_name.c_str());
    ABSL_CHECK(target_wheel_idx != -1 && target_site_id != -1) 
        << "Fatal Error: Target wheel '" << target_wheel_name << "' could not be resolved for force diagnostics.";

    // Joint lookups strictly mapped to the exact strings in XML (No fallbacks!)
    int tl_knee_idx = get_qpos_idx("torso_left_thigh_shin_joint");
    int tr_knee_idx = get_qpos_idx("torso_right_thigh_shin_joint");
    int hl_knee_idx = get_qpos_idx("head_left_thigh_shin_joint");
    int hr_knee_idx = get_qpos_idx("head_right_thigh_shin_joint");

    // =========================================================================================
    // PROPER INITIALIZATION BEFORE LOOP
    // =========================================================================================
    double initial_tl_angular_position = mj_data->qpos[tl_knee_idx];
    double initial_tr_angular_position = mj_data->qpos[tr_knee_idx];
    double initial_hl_angular_position = mj_data->qpos[hl_knee_idx];
    double initial_hr_angular_position = mj_data->qpos[hr_knee_idx];

    double last_tl_angular_position = initial_tl_angular_position;
    double last_tr_angular_position = initial_tr_angular_position;
    double last_hl_angular_position = initial_hl_angular_position;
    double last_hr_angular_position = initial_hr_angular_position;

    // Retrieve Thigh Sites Dynamically via the Helper Lambda
    Vector<3> initial_tlh_linear_position = get_site_pos("tl_thigh_site");
    Vector<3> initial_trh_linear_position = get_site_pos("tr_thigh_site");
    Vector<3> initial_hlh_linear_position = get_site_pos("hl_thigh_site");
    Vector<3> initial_hrh_linear_position = get_site_pos("hr_thigh_site");

    Vector<3> last_tlh_linear_position = initial_tlh_linear_position;
    Vector<3> last_trh_linear_position = initial_trh_linear_position;
    Vector<3> last_hlh_linear_position = initial_hlh_linear_position;
    Vector<3> last_hrh_linear_position = initial_hrh_linear_position;

    SimLogger logger;
    logger.reserve(30000); 

    double soft_switch_max_force = 770.0;
    double soft_switch_ramp_time = 0.5; 
    Eigen::Vector<double, model::contact_site_ids_size> contact_start_times;
    contact_start_times.setConstant(-100.0); 
    Eigen::Vector<double, model::contact_site_ids_size> prev_contact_mask;
    prev_contact_mask.setZero();

    double L_THIGH = 0.1016; 
    double L_SHIN = 0.08255; 
    double L_WHEEL = 0.0635;

    // =========================================================================================
    // SIMULATION LOOP
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

        // ------------------------ CHECK CONTACTS ------------------------  
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
        Eigen::Vector<double, model::contact_site_ids_size> contact_check2 = 
            Eigen::Map<Eigen::VectorXi>(contact_check_temp.data(), contact_check_temp.size()).cast<double>();
        
        Eigen::VectorXi raw_physics = contact_check2.cast<int>(); 

        // =========================================================
        // DYNAMIC PIVOT SCHEDULER
        // =========================================================
        double shin_rot_vel = 1.5 * 1.0; 

        if (contact_check2[0] > 0.5) contact_check2[1] = 0.0; 
        if (contact_check2[2] > 0.5) contact_check2[3] = 0.0; 
        if (contact_check2[4] > 0.5) contact_check2[5] = 0.0; 
        if (contact_check2[6] > 0.5) contact_check2[7] = 0.0; 

        // =========================================================
        // SOFT SWITCH LOGIC
        // =========================================================
        Eigen::Vector<double, model::contact_site_ids_size> current_force_limits;
        for(int i=0; i < model::contact_site_ids_size; ++i) {
            bool is_contact = (contact_check2[i] > 0.5);
            bool was_contact = (prev_contact_mask[i] > 0.5);

            if (is_contact && !was_contact) contact_start_times[i] = current_time;

            double limit = 0.0;
            if (is_contact) {
                double duration_in_contact = current_time - contact_start_times[i];
                double ratio = std::clamp(duration_in_contact / soft_switch_ramp_time, 0.0, 1.0);
                limit = ratio * soft_switch_max_force;
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
        state.contact_mask = contact_check2;
        
        controller.update_state(state);

        // -------------------------------------------------------------------------------------
        // COMPUTE CONTROL TARGETS 
        // -------------------------------------------------------------------------------------
        TaskspaceTargets taskspace_targets = TaskspaceTargets::Zero();

        double tl_shin_angle = mj_data->qpos[tl_knee_idx];        
        double tr_shin_angle = mj_data->qpos[tr_knee_idx];        
        double hl_shin_angle = mj_data->qpos[hl_knee_idx];        
        double hr_shin_angle = mj_data->qpos[hr_knee_idx];        

        double dt = current_time - last_time;
        if (dt == 0) dt = 0.0001; 

        double tl_angular_velocity = (tl_shin_angle - last_tl_angular_position) / dt;
        double tr_angular_velocity = (tr_shin_angle - last_tr_angular_position) / dt;
        double hl_angular_velocity = (hl_shin_angle - last_hl_angular_position) / dt;
        double hr_angular_velocity = (hr_shin_angle - last_hr_angular_position) / dt;

        double tl_angular_position_target = initial_tl_angular_position + shin_rot_vel * current_time;
        double tr_angular_position_target = initial_tr_angular_position + shin_rot_vel * current_time;
        double hl_angular_position_target = initial_hl_angular_position + shin_rot_vel * current_time;
        double hr_angular_position_target = initial_hr_angular_position + shin_rot_vel * current_time;

        double shin_kp = 800.0; 
        double shin_kv = 80.0;

        double tl_angular_control = shin_kp * (tl_angular_position_target - tl_shin_angle) + shin_kv * (shin_rot_vel - tl_angular_velocity);
        double tr_angular_control = shin_kp * (tr_angular_position_target - tr_shin_angle) + shin_kv * (shin_rot_vel - tr_angular_velocity);
        double hl_angular_control = shin_kp * (hl_angular_position_target - hl_shin_angle) + shin_kv * (shin_rot_vel - hl_angular_velocity);
        double hr_angular_control = shin_kp * (hr_angular_position_target - hr_shin_angle) + shin_kv * (shin_rot_vel - hr_angular_velocity);
        
        last_tl_angular_position = tl_shin_angle;
        last_tr_angular_position = tr_shin_angle;
        last_hl_angular_position = hl_shin_angle;
        last_hr_angular_position = hr_shin_angle;

        taskspace_targets.row(1) = Eigen::Vector<double, 6> {0, 0, 0, 0, tl_angular_control, 0};        
        taskspace_targets.row(2) = Eigen::Vector<double, 6> {0, 0, 0, 0, tr_angular_control, 0};        
        taskspace_targets.row(3) = Eigen::Vector<double, 6> {0, 0, 0, 0, hl_angular_control, 0};        
        taskspace_targets.row(4) = Eigen::Vector<double, 6> {0, 0, 0, 0, hr_angular_control, 0};        

        double thigh_lin_vel = 0.0;
        double thigh_lin_kp = 100.0 * 8.0; 
        double thigh_lin_kv = 20.0 * 4.0; 

        Eigen::Quaterniond body_quat(qpos(3), qpos(4), qpos(5), qpos(6));
        
        int hr_hip = get_qpos_idx("head_right_thigh_joint"); int hr_knee = hr_knee_idx;
        int hl_hip = get_qpos_idx("head_left_thigh_joint");  int hl_knee = hl_knee_idx;
        int tr_hip = get_qpos_idx("torso_right_thigh_joint"); int tr_knee = tr_knee_idx;
        int tl_hip = get_qpos_idx("torso_left_thigh_joint");  int tl_knee = tl_knee_idx;

        double h_hr_kinematic = get_propeller_leg_height(body_quat, mj_data->qpos[hr_hip], mj_data->qpos[hr_knee], 0.0, 0.0, L_THIGH, L_SHIN, L_WHEEL);        
        double h_hl_kinematic = get_propeller_leg_height(body_quat, mj_data->qpos[hl_hip], mj_data->qpos[hl_knee], 0.0, 0.0, L_THIGH, L_SHIN, L_WHEEL);
        double h_tr_kinematic = get_propeller_leg_height(body_quat, mj_data->qpos[tr_hip], mj_data->qpos[tr_knee], 0.0, 0.0, L_THIGH, L_SHIN, L_WHEEL);
        double h_tl_kinematic = get_propeller_leg_height(body_quat, mj_data->qpos[tl_hip], mj_data->qpos[tl_knee], 0.0, 0.0, L_THIGH, L_SHIN, L_WHEEL);

        Vector<3> tlh_linear_position = Vector<3>(0.0, 0.0, h_tl_kinematic);
        Vector<3> trh_linear_position = Vector<3>(0.0, 0.0, h_tr_kinematic);
        Vector<3> hlh_linear_position = Vector<3>(0.0, 0.0, h_hl_kinematic);
        Vector<3> hrh_linear_position = Vector<3>(0.0, 0.0, h_hr_kinematic);
        
        Vector<3> tlh_linear_velocity = (tlh_linear_position - last_tlh_linear_position) / dt;
        Vector<3> trh_linear_velocity = (trh_linear_position - last_trh_linear_position) / dt;
        Vector<3> hlh_linear_velocity = (hlh_linear_position - last_hlh_linear_position) / dt;
        Vector<3> hrh_linear_velocity = (hrh_linear_position - last_hrh_linear_position) / dt;

        double thigh_height_increase_stairs = -0.00;
        
        // Z-axis targets are now driven directly from the fully resolved `initial_*_linear_position(2)`
        double tlh_linear_control = thigh_lin_kp * ((initial_tlh_linear_position(2) + thigh_height_increase_stairs) - tlh_linear_position(2)) + thigh_lin_kv * (thigh_lin_vel - tlh_linear_velocity(2));
        double trh_linear_control = thigh_lin_kp * ((initial_trh_linear_position(2) + thigh_height_increase_stairs) - trh_linear_position(2)) + thigh_lin_kv * (thigh_lin_vel - trh_linear_velocity(2));
        double hlh_linear_control = thigh_lin_kp * ((initial_hlh_linear_position(2) + thigh_height_increase_stairs) - hlh_linear_position(2)) + thigh_lin_kv * (thigh_lin_vel - hlh_linear_velocity(2));
        double hrh_linear_control = thigh_lin_kp * ((initial_hrh_linear_position(2) + thigh_height_increase_stairs) - hrh_linear_position(2)) + thigh_lin_kv * (thigh_lin_vel - hrh_linear_velocity(2));
        
        last_tlh_linear_position = tlh_linear_position;
        last_trh_linear_position = trh_linear_position;
        last_hlh_linear_position = hlh_linear_position;
        last_hrh_linear_position = hrh_linear_position;
        last_time = current_time;

        taskspace_targets.row(5) = Eigen::Vector<double, 6> {0, 0, tlh_linear_control, 0, 0, 0};
        taskspace_targets.row(6) = Eigen::Vector<double, 6> {0, 0, trh_linear_control, 0, 0, 0};
        taskspace_targets.row(7) = Eigen::Vector<double, 6> {0, 0, hlh_linear_control, 0, 0, 0};
        taskspace_targets.row(8) = Eigen::Vector<double, 6> {0, 0, hrh_linear_control, 0, 0, 0};        

        Vector<3> body_position = qpos(Eigen::seqN(0, 3));
        cam.lookat[0] = body_position(0);

        // -------------------------------------------------------------------------------------
        // OSQP SOLVER
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
        // CRITICAL FIX: THE TWO LINES THAT SYNCHRONIZE TIME
        // =====================================================================
        mj_data->ctrl = torque_command.data(); // Apply the new torque instantly
        mj_forward(mj_model, mj_data);         // Force MuJoCo to recalculate physics for THIS microsecond
        // =====================================================================
        
        // =====================================================================
        // PROOF 1: PLANNED VS ACTUAL JOINT ACCELERATION
        // =====================================================================
        auto dv_planned = solution(Eigen::seqN(0, optimization::dv_size));
        
        int hr_shin_dof = get_dof_idx("head_right_thigh_shin_joint"); 
        
        logger.log("time", current_time);
        logger.log("hr_shin_accel_plan", dv_planned(hr_shin_dof));
        logger.log("hr_shin_accel_actual", mj_data->qacc[hr_shin_dof]);


        // ===================================================================================== 
        // PROOF 2: THE "TREAD VS AXLE" MISMATCH & INTEGRATION MASS DIAGNOSTIC
        // ===================================================================================== 
        if (target_wheel_idx != -1 && target_site_id != -1) {
            
            // 1. Calculate exactly what OSQP PLANNED for contact torques.
            // (OSQP maps forces through the SITE Jacobian - the center of the axle)
            // 1. Calculate exactly what OSQP PLANNED for contact torques.
            Vector<model::nv_size> planned_contact_torque = Vector<model::nv_size>::Zero();
            for (int w = 0; w < 8; ++w) {
                std::string site_name = std::string(model::contact_site_list[w]);
                
                // FIX 1: Read from the z vector (contact forces), not the u vector (actuators)
                double fx_plan = solution[optimization::z_idx + (3 * w) + 0];
                double fy_plan = solution[optimization::z_idx + (3 * w) + 1];
                double fz_plan = solution[optimization::z_idx + (3 * w) + 2];
                Eigen::Vector3d f_plan_3d(fx_plan, fy_plan, fz_plan);

                int site_id = mj_name2id(mj_model, mjOBJ_SITE, site_name.c_str()); 
                
                // FIX 2: Force RowMajor memory layout to align with MuJoCo
                Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor> jacp = Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor>::Zero();
                Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor> jacr = Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor>::Zero();
                
                mj_jacSite(mj_model, mj_data, jacp.data(), jacr.data(), site_id);

                planned_contact_torque += jacp.transpose() * f_plan_3d;
            }         

            // 2. Extract EXACTLY what MuJoCo's physics engine ACTUALLY experienced.
            // qfrc_constraint contains true forces mapped through the tire tread collisions!
            Vector<model::nv_size> actual_contact_torque = Eigen::Map<Vector<model::nv_size>>(mj_data->qfrc_constraint);


            // Log the entire qfrc_constraint vector to see which joints are being pushed
            // std::cout << "[DIAGNOSTIC] qfrc_constraint dump: ";
            // for(int i = 0; i < mj_model->nv; ++i) {
            //     if (std::abs(mj_data->qfrc_constraint[i]) > 1e-3) { // Filter out near-zero noise
            //         std::cout << "DOF[" << i << "]=" << mj_data->qfrc_constraint[i] << " ";
            //     }
            // }
            // std::cout << std::endl;            
            

            // 3. The true total missing torque (now including the wheel-radius offset!)
            Vector<model::nv_size> total_missing_torque = actual_contact_torque - planned_contact_torque;

            // 4. Solve for the predicted acceleration error using the FREE-SPACE mass matrix
            Eigen::Matrix<double, model::nv_size, model::nv_size> M = Eigen::Matrix<double, model::nv_size, model::nv_size>::Zero();
            mj_fullM(mj_model, M.data(), mj_data->qM);
            Vector<model::nv_size> predicted_accel_error = M.llt().solve(total_missing_torque);
            
            // 5. Compare against the real acceleration error
            double real_accel_error = mj_data->qacc[hr_shin_dof] - dv_planned(hr_shin_dof);

            // 6. Calculate the "Ghost Mass" ratio caused by MuJoCo's semi-implicit integrator
            // If the predicted error is highly nonzero, divide real by predicted. 
            double inertia_ratio = 1.0;
            if (std::abs(predicted_accel_error(hr_shin_dof)) > 1e-5) {
                inertia_ratio = real_accel_error / predicted_accel_error(hr_shin_dof);
            }

            logger.log("accel_error_real", real_accel_error);
            logger.log("accel_error_predicted", predicted_accel_error(hr_shin_dof));
            logger.log("inertia_ratio", inertia_ratio);
        }

        // Correct mapping for the head_right_thigh_shin_joint
        int fr_knee_actuator_idx = 7; 

        logger.log("torque_planned", torque_command(fr_knee_actuator_idx));

        // For the qfrc_actuator, you use the DOF address, which is index-based
        int fr_knee_dof_addr = mj_name2id(mj_model, mjOBJ_JOINT, "head_right_thigh_shin_joint");
        logger.log("torque_actual", mj_data->qfrc_actuator[mj_model->jnt_dofadr[fr_knee_dof_addr]]);

        logger.endStep();

        // -------------------------------------------------------------------------------------
        // VISUALIZATION & RECORDING
        // -------------------------------------------------------------------------------------
        if(visualization_timer > visualization_interval) {
            visualization_start_time = mj_data->time;

            mjrRect viewport_full = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport_full.width, &viewport_full.height);

            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport_full, &scn, &con);
            
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << "Time: " << mj_data->time << " s";
            mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport_full, ss.str().c_str(), 0, &con);

            std::stringstream ss2;
            ss2 << "      [PHYS] [MASK]\n--------------------\n";
            auto row = [&](std::string label, int idx) {
                ss2 << label << ":   " << raw_physics[idx] << "      " << (contact_check2[idx] > 0.5) << "\n";
            };
            row("TL_F", 0); row("TL_R", 1); row("TR_F", 2); row("TR_R", 3);
            ss2 << "--------------------\n";
            row("HL_F", 4); row("HL_R", 5); row("HR_F", 6); row("HR_R", 7);
            mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, viewport_full, ss2.str().c_str(), 0, &con);

            if (vid_record_flag && ffmpeg_pipe != nullptr) {
                mjrRect record_viewport = {0, 0, win_width, win_height};
                mjr_readPixels(rgb_buffer, nullptr, record_viewport, &con);
                fwrite(rgb_buffer, 1, win_width * win_height * 3, ffmpeg_pipe);
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    if (vid_record_flag) {
        if (ffmpeg_pipe) pclose(ffmpeg_pipe);
        delete[] rgb_buffer;
        std::cout << "Video saved." << std::endl;        
    }

    result.Update(controller.stop_thread());
    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);
    ABSL_CHECK(result.ok()) << result.message();
    
    logger.save("osc_test_data.csv");

    return 0;
}