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

    bool vid_record_flag = false;
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

    // =========================================================================================
    // MANIPULATOR EQUATION DIAGNOSTIC SETUP
    // =========================================================================================
    std::ofstream diag_log("manipulator_diagnostics.csv");
    diag_log << "Time,M_sim,M_ctrl,C_sim,C_ctrl,Tau_sim,Tau_ctrl,qacc_sim,qacc_ctrl\n";
    
    // Look up DOFs once before the loop
    int hr_shin_dof = get_dof_idx("head_right_thigh_shin_joint");
    // int fr_knee_actuator_idx = 7;

    int fr_knee_actuator_idx = mj_name2id(mj_model, mjOBJ_ACTUATOR, "fr_knee");

    double soft_switch_max_force = 770.0;
    double soft_switch_ramp_time = 0.5; 
    Eigen::Vector<double, model::contact_site_ids_size> contact_start_times;
    contact_start_times.setConstant(-100.0); 
    Eigen::Vector<double, model::contact_site_ids_size> prev_contact_mask;
    prev_contact_mask.setZero();

    double L_THIGH = 0.1016; 
    double L_SHIN = 0.08255; 
    double L_WHEEL = 0.0635;

    Vector<model::nu_size> torque_command = Vector<model::nu_size>::Zero();

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

        // if (contact_check2[0] > 0.5) contact_check2[1] = 0.0; 
        // if (contact_check2[2] > 0.5) contact_check2[3] = 0.0; 
        // if (contact_check2[4] > 0.5) contact_check2[5] = 0.0; 
        // if (contact_check2[6] > 0.5) contact_check2[7] = 0.0; 

        // std::cout << "---contact check2---" << std::endl;
        // std::cout << contact_check2 << std::endl;


        // bool pivot_on_front = false; // Toggle this based on your gait/direction phase

        // if (pivot_on_front) {
        //     // Front wheels (0, 2, 4, 6) suppress Rear wheels (1, 3, 5, 7)
        //     if (contact_check2[0] > 0.5) contact_check2[1] = 0.0; 
        //     if (contact_check2[2] > 0.5) contact_check2[3] = 0.0; 
        //     if (contact_check2[4] > 0.5) contact_check2[5] = 0.0; 
        //     if (contact_check2[6] > 0.5) contact_check2[7] = 0.0; 
        // } else {
        //     // THE OTHER HALF: Rear wheels (1, 3, 5, 7) suppress Front wheels (0, 2, 4, 6)
        //     if (contact_check2[1] > 0.5) contact_check2[0] = 0.0; 
        //     if (contact_check2[3] > 0.5) contact_check2[2] = 0.0; 
        //     if (contact_check2[5] > 0.5) contact_check2[4] = 0.0; 
        //     if (contact_check2[7] > 0.5) contact_check2[6] = 0.0; 
        // }        


        

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


        // =======================================================
        // =======================================================
        // =======================================================
        // =======================================================
        // controller.update_mj_data();         
        // controller.update_osc_data();        
        // controller.update_optimization_data(); 
        // controller.update_optimization();
        // controller.solve_optimization();
        // =======================================================        
        // =======================================================
        // =======================================================
        // =======================================================

        // Vector<model::nu_size> torque_command = controller.get_torque_command();
        torque_command = controller.get_torque_command();
        std::cout.clear();

        int solver_status_int = controller.get_solver_status(); 
        osqp::OsqpExitCode solver_status = static_cast<osqp::OsqpExitCode>(solver_status_int);
        if (solver_status != osqp::OsqpExitCode::kOptimal) {
             std::cout << "Solver failed with status: " << (int)solver_status << " at time " << current_time << ". Exiting loop." << std::endl;
             break; 
        }

        Vector<optimization::design_vector_size> solution = controller.get_solution();

        // =======================================================
        // =======================================================
        // =======================================================
        // CRITICAL FIX: Extract the torques directly from the solution vector using u_idx
        // Vector<model::nu_size> torque_command = solution(Eigen::seqN(optimization::u_idx, model::nu_size));        
        // =======================================================
        // =======================================================
        // =======================================================

        // =====================================================================
        // EXTRACTION BUG PROOF
        // =====================================================================
        // std::cout << "\n--- EXTRACTION PROOF ---" << std::endl;

        // // 1. What the controller gave you to send to MuJoCo
        // std::cout << "Exported Torque Command: " << torque_command(0) << std::endl;

        // // 2. The raw planned acceleration (dv is at the start of the design vector, index 0)
        // std::cout << "Raw Solution [dv] value: " << solution(0) << std::endl;

        // // 3. The raw planned torque (u starts at u_idx)
        // std::cout << "Raw Solution [u]  value: " << solution(optimization::u_idx + 0) << std::endl;
        // std::cout << "------------------------\n" << std::endl;     


        // =====================================================================
        // CRITICAL FIX: THE TWO LINES THAT SYNCHRONIZE TIME
        // =====================================================================
        mj_data->ctrl = torque_command.data(); // Apply the new torque instantly
        mj_forward(mj_model, mj_data);         // Force MuJoCo to recalculate physics for THIS microsecond
        // =====================================================================
        // =====================================================================
        // =====================================================================
        // check indexing from here*****************************************************************************************************
        // =====================================================================
        // =====================================================================
        // =====================================================================
        
        // =====================================================================
        // MANIPULATOR EQUATION DIAGNOSTIC LOGGER
        // =====================================================================
        controller.log_manipulator_equation(mj_model, mj_data, diag_log, hr_shin_dof, fr_knee_actuator_idx);
        
        // =====================================================================
        // PROOF 1: PLANNED VS ACTUAL JOINT ACCELERATION
        // =====================================================================
        auto dv_planned = solution(Eigen::seqN(0, optimization::dv_size));
        
        logger.log("time", current_time);
        logger.log("hr_shin_accel_plan", dv_planned(hr_shin_dof));
        logger.log("hr_shin_accel_actual", mj_data->qacc[hr_shin_dof]);

        // ===================================================================================== 
        // ISOLATED CONTACT FORCE PROOF (CORRECTED API)
        // ===================================================================================== 

        // 1. Calculate Planned Contact Torque (What OSQP expected)
        Vector<model::nv_size> tau_contact_plan = Vector<model::nv_size>::Zero();
        for (int w = 0; w < 8; ++w) {
            std::string site_name = std::string(model::contact_site_list[w]);
            double fx_plan = solution[optimization::z_idx + (3 * w) + 0];
            double fy_plan = solution[optimization::z_idx + (3 * w) + 1];
            double fz_plan = solution[optimization::z_idx + (3 * w) + 2];
            Eigen::Vector3d f_plan_3d(fx_plan, fy_plan, fz_plan);

            int site_id = mj_name2id(mj_model, mjOBJ_SITE, site_name.c_str()); 
            // Eigen::Matrix<double, 3, model::nv_size> jacp = Eigen::Matrix<double, 3, model::nv_size>::Zero();

            // Matrix<3, model::nv_size> jacp = Matrix<3, model::nv_size>::Zero();            
            Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor> jacp = Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor>::Zero();
            
            mj_jacSite(mj_model, mj_data, jacp.data(), nullptr, site_id);

            tau_contact_plan += jacp.transpose() * f_plan_3d;
            // ===================================================================================== 
            // ===================================================================================== 
            // ===================================================================================== 
            // ===================================================================================== 

            // 1. Map MuJoCo's flat 1D array to a 3x3 Eigen matrix
            // Eigen::Matrix3d site_rot = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(mj_data->site_xmat + 9 * site_id);

            // 2. Rotate the planned local force into the global frame
            // Eigen::Vector3d f_plan_global = site_rot * f_plan_3d;

            // 3. Calculate torques using the global Jacobian and global force
            // tau_contact_plan += jacp.transpose() * f_plan_global;
            // ===================================================================================== 
            // ===================================================================================== 
            // ===================================================================================== 
            // ===================================================================================== 

        }            

        // 2. Calculate Actual Contact Torque (What MuJoCo actually did at the treads)
        Vector<model::nv_size> tau_contact_sim = Vector<model::nv_size>::Zero();

        // Loop through ONLY active collisions, ignoring joint limits/damping
        for (int i = 0; i < mj_data->ncon; ++i) {
            mjContact* contact = &(mj_data->contact[i]);
            
            int geom_id = contact->geom[0];
            int body_id = mj_model->geom_bodyid[geom_id];
            
            Eigen::Vector3d f_sim_contact;


            // ===================================================================================== 
            // ===================================================================================== 
            // ===================================================================================== 
            // ===================================================================================== 
            
            // // Safely get the string names of the colliding geometries
            // const char* geom0_name = mj_model->names + mj_model->name_geomadr[contact->geom[0]];
            // const char* geom1_name = mj_model->names + mj_model->name_geomadr[contact->geom[1]];

            // // MuJoCo standard: geom[1] acts upon geom[0]
            // std::cout << "Collision: Force is applied BY [" << geom1_name << "] TO [" << geom0_name << "]" << std::endl;
            // ===================================================================================== 
            // ===================================================================================== 
            // ===================================================================================== 
            // ===================================================================================== 




            // Ensure we are mapping the force TO the robot, not the floor.
            // Body 0 is always the world body.
            if (body_id == 0) { 
                geom_id = contact->geom[1];
                body_id = mj_model->geom_bodyid[geom_id];
                
                double force_6d[6] = {0};
                mj_contactForce(mj_model, mj_data, i, force_6d);
                // Invert the force (Newton's 3rd Law)
                f_sim_contact = Eigen::Vector3d(force_6d[0], force_6d[1], force_6d[2]); 
                // f_sim_contact = Eigen::Vector3d(-force_6d[0], -force_6d[1], -force_6d[2]); 
            } else {
                double force_6d[6] = {0};
                mj_contactForce(mj_model, mj_data, i, force_6d);
                f_sim_contact = Eigen::Vector3d(-force_6d[0], -force_6d[1], -force_6d[2]);
                // f_sim_contact = Eigen::Vector3d(force_6d[0], force_6d[1], force_6d[2]);
            }

            // Get the Jacobian at the exact physical collision point for the correct body
            // Eigen::Matrix<double, 3, model::nv_size> jacp = Eigen::Matrix<double, 3, model::nv_size>::Zero();
            Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor> jacp = Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor>::Zero();

            mj_jac(mj_model, mj_data, jacp.data(), nullptr, contact->pos, body_id);

            // MuJoCo contact forces are in a local frame; we need the contact frame matrix to rotate them
            Eigen::Matrix3d contact_frame = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(contact->frame);
            Eigen::Vector3d f_sim_global = contact_frame.transpose() * f_sim_contact;

            tau_contact_sim += jacp.transpose() * f_sim_global;
        }

        // 3. The Isolated Contact Torque Discrepancy
        Vector<model::nv_size> delta_tau_contact = tau_contact_sim - tau_contact_plan;

        // 4. Predict the Acceleration Error strictly from Contacts
        Eigen::Matrix<double, model::nv_size, model::nv_size> M = Eigen::Matrix<double, model::nv_size, model::nv_size>::Zero();
        mj_fullM(mj_model, M.data(), mj_data->qM);
        Vector<model::nv_size> predicted_delta_acc = M.llt().solve(delta_tau_contact);

        // 5. Calculate the True Acceleration Error
        Vector<model::nv_size> real_ddq = Eigen::Map<Vector<model::nv_size>>(mj_data->qacc);
        Vector<model::nv_size> real_delta_acc = real_ddq - dv_planned;


        // =====================================================================
        // PROOF OF VECTOR LAYOUT (SAFE DYNAMIC ACCESS)
        // =====================================================================
        // static int print_throttle = 0;
        // if (print_throttle++ % 100 == 0) { 
        //     Vector<optimization::design_vector_size> sol = controller.get_solution();
            
        //     std::cout << "\n--- LAYOUT CONFIGURATION ---" << std::endl;
        //     std::cout << "Target Vector Size: " << sol.size() << std::endl;
            
        //     // Print the constants to compare against sol.size()
        //     std::cout << "  dv: idx=" << optimization::dv_idx << ", size=" << optimization::dv_size << std::endl;
        //     std::cout << "  u:  idx=" << optimization::u_idx << ", size=" << optimization::u_size << std::endl;
        //     std::cout << "  z:  idx=" << optimization::z_idx << ", size=" << optimization::z_size << std::endl;

        //     // Check for overlaps or overflows
        //     if ((optimization::dv_idx + optimization::dv_size) > sol.size()) std::cout << "!! [dv] OVERFLOWS" << std::endl;
        //     if ((optimization::u_idx + optimization::u_size) > sol.size())   std::cout << "!! [u] OVERFLOWS" << std::endl;
        //     if ((optimization::z_idx + optimization::z_size) > sol.size())   std::cout << "!! [z] OVERFLOWS" << std::endl;

        //     std::cout << "\n--- DYNAMIC DATA DUMP ---" << std::endl;
            
        //     // Use dynamic segmenting (.segment(idx, size)) to avoid static assertion crashes
        //     if ((optimization::dv_idx + optimization::dv_size) <= sol.size()) {
        //         std::cout << "dv: " << sol.segment(optimization::dv_idx, optimization::dv_size).transpose() << std::endl;
        //     }
            
        //     if ((optimization::u_idx + optimization::u_size) <= sol.size()) {
        //         std::cout << "u:  " << sol.segment(optimization::u_idx, optimization::u_size).transpose() << std::endl;
        //     }
            
        //     if ((optimization::z_idx + optimization::z_size) <= sol.size()) {
        //         std::cout << "z:  " << sol.segment(optimization::z_idx, optimization::z_size).transpose() << std::endl;
        //     }
        //     std::cout << "----------------------------\n" << std::endl;
        // }


        // 6. The Final Verdict
        logger.log("delta_acc_real", real_delta_acc(hr_shin_dof));
        logger.log("delta_acc_predicted_from_contacts", predicted_delta_acc(hr_shin_dof));

        logger.log("torque_planned", torque_command(fr_knee_actuator_idx));

        // For the qfrc_actuator, you use the DOF address, which is index-based
        int fr_knee_dof_addr = mj_name2id(mj_model, mjOBJ_JOINT, "head_right_thigh_shin_joint");
        logger.log("torque_actual", mj_data->qfrc_actuator[mj_model->jnt_dofadr[fr_knee_dof_addr]]);


        // =====================================================================
        // THE 1-TO-1 LINEUP: Find the Ghost Force
        // =====================================================================

        // 1. Actuator Delta (Is MuJoCo scaling/clipping your torque?)
        double tau_act_sim = mj_data->qfrc_actuator[hr_shin_dof];
        double tau_act_plan = torque_command(fr_knee_actuator_idx); 
        double delta_actuator = tau_act_sim - tau_act_plan;

        // 2. Contact Delta (Is MuJoCo's soft floor fighting your rigid math?)
        // FIX: Using the newly named variables from your Isolated Contact Proof
        double tau_cont_sim_val = tau_contact_sim(hr_shin_dof); 
        double tau_cont_plan_val = tau_contact_plan(hr_shin_dof); 
        double delta_contact = tau_cont_sim_val - tau_cont_plan_val;

        // 3. Bias Delta (Does MuJoCo have damping/friction the QP lacks?)
        Eigen::Matrix<double, model::nv_size, model::nv_size> M_sim = Eigen::Matrix<double, model::nv_size, model::nv_size>::Zero();
        mj_fullM(mj_model, M_sim.data(), mj_data->qM);
        double inertial_plan = (M_sim * dv_planned)(hr_shin_dof);
        
        // FIX: Updated to use tau_cont_plan_val
        double c_ctrl_inferred = tau_act_plan + tau_cont_plan_val - inertial_plan; 

        double c_sim = mj_data->qfrc_bias[hr_shin_dof]; 
        double delta_bias = c_sim - c_ctrl_inferred;

        // Log the Deltas
        logger.log("Ghost_Delta_Actuator", delta_actuator);
        logger.log("Ghost_Delta_Contact", delta_contact);
        logger.log("Ghost_Delta_Bias", delta_bias);

        // Add these right above your qfrc_bias print
        // std::cout << "Joint Angle (qpos): " << mj_data->qpos[hr_knee] << std::endl;
        // std::cout << "Joint Velocity (qvel): " << mj_data->qvel[hr_shin_dof] << std::endl;


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

    // result.Update(controller.stop_thread());
    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);
    ABSL_CHECK(result.ok()) << result.message();
    
    logger.save("osc_test_data.csv");

    return 0;
}