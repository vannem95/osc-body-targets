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

//================================================================================= functions (sim logger) =========================

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
    void logArray(const std::string& prefix, const double* arr, int size) {
        for (int i = 0; i < size; ++i) log(prefix + "_" + std::to_string(i), arr[i]);
    }
    template <typename Derived>
    void logEigen(const std::string& prefix, const Eigen::MatrixBase<Derived>& vec) {
        for (int i = 0; i < vec.size(); ++i) log(prefix + "_" + std::to_string(i), vec(i));
    }
    void logMjData(const mjModel* m, const mjData* d) {
        log("time", d->time);
        logArray("qpos", d->qpos, m->nq);
        logArray("qvel", d->qvel, m->nv);
        logArray("qacc", d->qacc, m->nv);
        logArray("ctrl", d->ctrl, m->nu);
        logArray("sens", d->sensordata, m->nsensordata);
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
    if (!mj_model) { printf("%s\n", mj_error); return 1; }
    mjData* mj_data = mj_makeData(mj_model);

    mj_resetDataKeyframe(mj_model, mj_data, 8);
    mj_forward(mj_model, mj_data);

    mjvCamera cam; mjvPerturb pert; mjvOption opt; mjvScene scn; mjrContext con;
    glfwInit();
    int win_width = 1920; int win_height = 1080;
    GLFWwindow* window = glfwCreateWindow(win_width, win_height, "Sim + Plot", NULL, NULL); 
    glfwMakeContextCurrent(window);
    // glfwMaximizeWindow(window); 
    glfwSwapInterval(1);

    mjv_defaultCamera(&cam);
    mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt);
    
    // --- Enable 3D Contact Points and Force Arrows ---
    opt.flags[mjVIS_CONTACTPOINT] = 1; 
    // opt.flags[mjVIS_CONTACTFORCE] = 1; 

    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(mj_model, &scn, 1000);
    mjr_makeContext(mj_model, &con, mjFONTSCALE_300);

    // =====================================================================
    // GUI TOGGLE FLAG: Set to true for Kinematics, false for Force check
    // =====================================================================
    bool show_task_plots = false; 

    mjvFigure fig_fz, fig_shin, fig_height;
    mjv_defaultFigure(&fig_fz); mjv_defaultFigure(&fig_shin); mjv_defaultFigure(&fig_height);

    // --- SETUP FZ PLOT (Mode: FALSE) ---
    strcpy(fig_fz.title, "HR Front Contact Force Fz (N)");
    strcpy(fig_fz.xlabel, "Time (s)");
    fig_fz.flg_extend = 0; fig_fz.range[0][0] = 0.0f; fig_fz.range[0][1] = 10.0f; fig_fz.gridsize[0] = 6;      
    strcpy(fig_fz.xformat, "%.0f"); fig_fz.range[1][0] = -10.0f; fig_fz.range[1][1] = 150.0f; fig_fz.gridsize[1] = 9;     
    strcpy(fig_fz.yformat, "%.0f"); fig_fz.flg_barplot = 0;     
    // strcpy(fig_fz.linename[0], "Limit (Soft)"); 
    strcpy(fig_fz.linename[1], "QP Plan"); strcpy(fig_fz.linename[2], "Actual");
    // fig_fz.linergb[0][0]=1.0f; fig_fz.linergb[0][1]=0.0f; fig_fz.linergb[0][2]=0.0f; // Red
    fig_fz.linergb[1][0]=0.0f; fig_fz.linergb[1][1]=0.0f; fig_fz.linergb[1][2]=1.0f; // Blue
    fig_fz.linergb[2][0]=0.0f; fig_fz.linergb[2][1]=1.0f; fig_fz.linergb[2][2]=0.0f; // Green

    // --- SETUP SHIN PLOT (Mode: TRUE, Top) ---
    strcpy(fig_shin.title, "HR Shin Pos wrapped (rad)");
    strcpy(fig_shin.xlabel, "Time (s)");
    fig_shin.flg_extend = 0; fig_shin.range[0][0] = 0.0f; fig_shin.range[0][1] = 10.0f; fig_shin.gridsize[0] = 6;      
    strcpy(fig_shin.xformat, "%.0f"); fig_shin.range[1][0] = -4.0f; fig_shin.range[1][1] = 4.0f; fig_shin.gridsize[1] = 9;     
    strcpy(fig_shin.yformat, "%.1f"); fig_shin.flg_barplot = 0;     
    strcpy(fig_shin.linename[0], "Target"); strcpy(fig_shin.linename[1], "Actual");
    fig_shin.linergb[0][0]=1.0f; fig_shin.linergb[0][1]=0.0f; fig_shin.linergb[0][2]=0.0f; // Red
    fig_shin.linergb[1][0]=0.0f; fig_shin.linergb[1][1]=1.0f; fig_shin.linergb[1][2]=0.0f; // Green

    // --- SETUP HEIGHT PLOT (Mode: TRUE, Bottom) ---
    strcpy(fig_height.title, "HR Hip Height (m)");
    strcpy(fig_height.xlabel, "Time (s)");
    fig_height.flg_extend = 0; fig_height.range[0][0] = 0.0f; fig_height.range[0][1] = 10.0f; fig_height.gridsize[0] = 6;      
    strcpy(fig_height.xformat, "%.0f"); fig_height.range[1][0] = 0.14f; fig_height.range[1][1] = 0.16f; fig_height.gridsize[1] = 9;     
    strcpy(fig_height.yformat, "%.3f"); fig_height.flg_barplot = 0;     
    strcpy(fig_height.linename[0], "Target"); strcpy(fig_height.linename[1], "Actual");
    fig_height.linergb[0][0]=1.0f; fig_height.linergb[0][1]=0.0f; fig_height.linergb[0][2]=0.0f; // Red
    fig_height.linergb[1][0]=0.0f; fig_height.linergb[1][1]=1.0f; fig_height.linergb[1][2]=0.0f; // Green

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
                                    " -i - -vf vflip -c:v h264_nvenc -preset hq -b:v 10M -pix_fmt yuv420p /home/vivek/walter_sim_tumbling_3minthesis.mp4";    
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
    
    Vector<3> initial_position = qpos(Eigen::seqN(0, 3));
    Eigen::Matrix<double, model::site_ids_size, 3> site_data;
    Eigen::Matrix<double, model::site_ids_size, 3> initial_site_data;
    Eigen::Vector<double, model::contact_site_ids_size> contact_check2;
    Eigen::Matrix<double, model::site_ids_size, 9> site_rotational_data;

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
    std::vector<double> data1_4, data1t_4, data3_4, data3t_4;
    std::vector<double> data_time;
    std::vector<double> data_red, data_blue, data_green; // Used for Fz Plot

    data_time.reserve(30000); data_red.reserve(30000); data_blue.reserve(30000); data_green.reserve(30000);

    for(const std::string_view& site : model::site_list) {
        std::string site_str = std::string(site);
        sites.push_back(site_str);
        site_ids.push_back(mj_name2id(mj_model, mjOBJ_SITE, site_str.data()));
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

    std::vector<std::string> wheel_log_names;
    for(size_t i = 0; i < wheel_site_ids_ref.size(); ++i) {
        if (i < 4) wheel_log_names.push_back("torso_wheel_" + std::to_string(i + 1));
        else wheel_log_names.push_back("head_wheel_" + std::to_string(i - 3));
    }    

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
        site_data = Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->site_xpos)(site_ids, Eigen::placeholders::all);

        // =====================================================================
        // DIAGNOSTIC: THE ROLLING LEVER ARM TEST (FULL DURATION)
        // =====================================================================
        // static int trf_geom_id = mj_name2id(mj_model, mjOBJ_GEOM, "torso_right_front_wheel_geom");
        // static int trf_site_id = mj_name2id(mj_model, mjOBJ_SITE, "trf_wheel_site");
        
        // static int hrf_geom_id = mj_name2id(mj_model, mjOBJ_GEOM, "head_right_front_wheel_geom");
        // static int hrf_site_id = mj_name2id(mj_model, mjOBJ_SITE, "hrf_wheel_site");

        // // Throttle to prevent terminal lockup (prints every 10th tick / 50Hz)
        // static int print_throttle = 0;
        // print_throttle++;

        // if (print_throttle % 10 == 0) {
        //     for (int i = 0; i < mj_data->ncon; ++i) {
        //         int geom1 = mj_data->contact[i].geom[0];
        //         int geom2 = mj_data->contact[i].geom[1];

        //         // 1. Check Torso Right Front 
        //         if (geom1 == trf_geom_id || geom2 == trf_geom_id) {
        //             double contact_x = mj_data->contact[i].pos[0];
        //             double site_x = mj_data->site_xpos[3 * trf_site_id + 0];
        //             double x_error = contact_x - site_x;

        //             std::cout << std::fixed << std::setprecision(4);
        //             std::cout << "[Torso Pivot] Time: " << current_time 
        //                       << "s | Discrepancy: " << x_error * 100.0 << " cm" << std::endl;
        //         }

        //         // 2. Check Head Right Front
        //         if (geom1 == hrf_geom_id || geom2 == hrf_geom_id) {
        //             double contact_x = mj_data->contact[i].pos[0];
        //             double site_x = mj_data->site_xpos[3 * hrf_site_id + 0];
        //             double x_error = contact_x - site_x;

        //             std::cout << std::fixed << std::setprecision(4);
        //             std::cout << "[Head Pivot]  Time: " << current_time 
        //                       << "s | Discrepancy: " << x_error * 100.0 << " cm" << std::endl;
        //         }
        //     }
        // }
        // =====================================================================

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
        contact_check2 = Eigen::Map<Eigen::VectorXi>(contact_check_temp.data(), contact_check_temp.size()).cast<double>();
        Eigen::VectorXi raw_physics = contact_check2.cast<int>();

        // =========================================================
        // DYNAMIC PIVOT SCHEDULER (Unhinges feet to allow rolling)
        // =========================================================
        double shin_rot_vel = 1.5*1.0; 

        if (contact_check2[0] > 0.5) contact_check2[1] = 0.0; 
        if (contact_check2[2] > 0.5) contact_check2[3] = 0.0; 
        if (contact_check2[4] > 0.5) contact_check2[5] = 0.0; 
        if (contact_check2[6] > 0.5) contact_check2[7] = 0.0; 

        // =========================================================
        // SOFT SWITCH LOGIC (Time-based Ramp)
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
        // controller.update_max_contact_forces(current_force_limits);

        // -------------------------------------------------------------------------------------
        // COMPUTE CONTROL TARGETS
        // -------------------------------------------------------------------------------------
        TaskspaceTargets taskspace_targets = TaskspaceTargets::Zero();
        double phase_offset = 0.0;

        double tl_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[2]];        
        double tr_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[4]];        
        double hl_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[6]];        
        double hr_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[8]];        

        double dt = current_time - last_time;
        if (dt == 0) dt = 0.0001; 

        double tl_angular_velocity = (tl_shin_angle - last_tl_angular_position)/dt;
        double tr_angular_velocity = (tr_shin_angle - last_tr_angular_position)/dt;
        double hl_angular_velocity = (hl_shin_angle - last_hl_angular_position)/dt;
        double hr_angular_velocity = (hr_shin_angle - last_hr_angular_position)/dt;

        // Kinematic Targets
        double tl_angular_position_target = initial_tl_angular_position + shin_rot_vel * current_time + phase_offset;
        double tr_angular_position_target = initial_tr_angular_position + shin_rot_vel * current_time + phase_offset;
        double hl_angular_position_target = initial_hl_angular_position + shin_rot_vel * current_time;
        double hr_angular_position_target = initial_hr_angular_position + shin_rot_vel * current_time;

        double tl_angular_position_error = tl_angular_position_target - tl_shin_angle;
        double tr_angular_position_error = tr_angular_position_target - tr_shin_angle;
        double hl_angular_position_error = hl_angular_position_target - hl_shin_angle;
        double hr_angular_position_error = hr_angular_position_target - hr_shin_angle;
        
        double tl_angular_velocity_error = shin_rot_vel - tl_angular_velocity;
        double tr_angular_velocity_error = shin_rot_vel - tr_angular_velocity;
        double hl_angular_velocity_error = shin_rot_vel - hl_angular_velocity;
        double hr_angular_velocity_error = shin_rot_vel - hr_angular_velocity;

        // SHIN PD GAINS 
        double shin_kp = 800.0; 
        double shin_kv = 80.0;

        double tl_angular_control = shin_kp * tl_angular_position_error + shin_kv * tl_angular_velocity_error;
        double tr_angular_control = shin_kp * tr_angular_position_error + shin_kv * tr_angular_velocity_error;
        double hl_angular_control = shin_kp * hl_angular_position_error + shin_kv * hl_angular_velocity_error;
        double hr_angular_control = shin_kp * hr_angular_position_error + shin_kv * hr_angular_velocity_error;
        
        last_tl_angular_position = tl_shin_angle;
        last_tr_angular_position = tr_shin_angle;
        last_hl_angular_position = hl_shin_angle;
        last_hr_angular_position = hr_shin_angle;

        taskspace_targets.row(1) = Eigen::Vector<double, 6> {0, 0, 0, 0, tl_angular_control, 0};        
        taskspace_targets.row(2) = Eigen::Vector<double, 6> {0, 0, 0, 0, tr_angular_control, 0};        
        taskspace_targets.row(3) = Eigen::Vector<double, 6> {0, 0, 0, 0, hl_angular_control, 0};        
        taskspace_targets.row(4) = Eigen::Vector<double, 6> {0, 0, 0, 0, hr_angular_control, 0};        

        // THIGH HEIGHT PD GAINS
        double thigh_lin_vel = 0.0;
        double thigh_lin_kp = 100.0*10.0; 
        double thigh_lin_kv = 20.0*10.0; 

        Eigen::Quaterniond body_quat(qpos(3), qpos(4), qpos(5), qpos(6));
        double hr_hip = mj_data->qpos[13]; double hr_knee = mj_data->qpos[14]; 
        double hl_hip = mj_data->qpos[11]; double hl_knee = mj_data->qpos[12]; 
        double tr_hip = mj_data->qpos[9];  double tr_knee = mj_data->qpos[10]; 
        double tl_hip = mj_data->qpos[7];  double tl_knee = mj_data->qpos[8]; 

        double h_hr_kinematic = get_propeller_leg_height(body_quat, hr_hip, hr_knee, 0.0, 0.0, L_THIGH, L_SHIN, L_WHEEL);        
        double h_hl_kinematic = get_propeller_leg_height(body_quat, hl_hip, hl_knee, 0.0, 0.0, L_THIGH, L_SHIN, L_WHEEL);
        double h_tr_kinematic = get_propeller_leg_height(body_quat, tr_hip, tr_knee, 0.0, 0.0, L_THIGH, L_SHIN, L_WHEEL);
        double h_tl_kinematic = get_propeller_leg_height(body_quat, tl_hip, tl_knee, 0.0, 0.0, L_THIGH, L_SHIN, L_WHEEL);

        Vector<3> tlh_linear_position = Vector<3>(0.0, 0.0, h_tl_kinematic);
        Vector<3> trh_linear_position = Vector<3>(0.0, 0.0, h_tr_kinematic);
        Vector<3> hlh_linear_position = Vector<3>(0.0, 0.0, h_hl_kinematic);
        Vector<3> hrh_linear_position = Vector<3>(0.0, 0.0, h_hr_kinematic);
        
        Vector<3> tlh_linear_velocity = (tlh_linear_position - last_tlh_linear_position)/dt;
        Vector<3> trh_linear_velocity = (trh_linear_position - last_trh_linear_position)/dt;
        Vector<3> hlh_linear_velocity = (hlh_linear_position - last_hlh_linear_position)/dt;
        Vector<3> hrh_linear_velocity = (hrh_linear_position - last_hrh_linear_position)/dt;

        double thigh_height_increase_stairs = -0.00;
        double tlh_linear_position_target = initial_site_data(5,2) + thigh_height_increase_stairs;
        double trh_linear_position_target = initial_site_data(6,2) + thigh_height_increase_stairs;
        double hlh_linear_position_target = initial_site_data(7,2) + thigh_height_increase_stairs;
        double hrh_linear_position_target = initial_site_data(8,2) + thigh_height_increase_stairs;

        double tlh_linear_control = thigh_lin_kp * (tlh_linear_position_target - tlh_linear_position(2)) + thigh_lin_kv * (thigh_lin_vel - tlh_linear_velocity(2));
        double trh_linear_control = thigh_lin_kp * (trh_linear_position_target - trh_linear_position(2)) + thigh_lin_kv * (thigh_lin_vel - trh_linear_velocity(2));
        double hlh_linear_control = thigh_lin_kp * (hlh_linear_position_target - hlh_linear_position(2)) + thigh_lin_kv * (thigh_lin_vel - hlh_linear_velocity(2));
        double hrh_linear_control = thigh_lin_kp * (hrh_linear_position_target - hrh_linear_position(2)) + thigh_lin_kv * (thigh_lin_vel - hrh_linear_velocity(2));
        

        
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
        // COMPUTE TORQUES & SOLVE
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
        // RECORD LAGRANGE MULTIPLIERS (SHADOW PRICES)
        // =====================================================================
        // Extract the dual variables from the OSQP solver.
        // (Ensure OperationalSpaceController::get_dual_solution() is implemented 
        // and returns solver.dual_solution() from your OSQP wrapper).
        auto dual_solution = controller.get_dual_solution();

        // Log the exact Shadow Price for the Torso Z-axis constraint (Index 2)
        // This is the direct tension metric between your Height PD controller and Gravity.
        logger.log("shadow_price_z", dual_solution(2));        

        // =====================================================================
        // DIAGNOSTIC: THE FRICTION CONE HALLUCINATION TEST
        // =====================================================================
        // 1. Extract the contact forces segment (z) from the QP solution
        auto z_planned = solution(Eigen::seqN(optimization::u_idx, optimization::z_size));

        // 2. Extract Head Right Front (hrf) forces (Index 6 in YAML -> 18, 19, 20)
        double hrf_fx = z_planned(18);
        double hrf_fy = z_planned(19);
        double hrf_fz = z_planned(20);

        // 3. Extract Torso Right Front (trf) forces (Index 2 in YAML -> 6, 7, 8)
        double trf_fx = z_planned(6);
        double trf_fy = z_planned(7);
        double trf_fz = z_planned(8);

        // Calculate the friction ratio for the Head wheel
        if (hrf_fz > 1.0) { // Only check when the foot is actually bearing weight
            double horizontal_force = std::abs(hrf_fx) + std::abs(hrf_fy);
            double friction_ratio = horizontal_force / hrf_fz;
            
            // Adjust this time window to exactly when your 20N gap happens!
            if (current_time > 7.0 && current_time < 8.0) {
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "[Friction Check - HRF] Time: " << current_time 
                            << "s | Planned Fz: " << hrf_fz 
                            << "N | Ratio: " << friction_ratio 
                            << " (Limit: 0.8)" << std::endl;
            }
        }
        // =====================================================================      



        // -------------------------------------------------------------------------------------
        // EXTRACT CONTACT FORCES (Head Right Front Wheel)
        // -------------------------------------------------------------------------------------
        int target_wheel_idx = 6; 
        int target_geom = wheel_geom_ids[target_wheel_idx];

        data_red.push_back(current_force_limits[target_wheel_idx]); // Limit
        data_blue.push_back(solution[optimization::u_idx + (3 * target_wheel_idx) + 2]); // QP Plan Fz

        double hrf_actual_fz = 0.0;
        for (int i = 0; i < mj_data->ncon; ++i) {
            if (mj_data->contact[i].geom[0] == target_geom || mj_data->contact[i].geom[1] == target_geom) {
                double f_contact[6];
                mj_contactForce(mj_model, mj_data, i, f_contact);
                hrf_actual_fz += std::abs(f_contact[0]); 
            }
        }
        data_green.push_back(hrf_actual_fz); // Actual Fz

        // Store Task Targets for GUI
        // data1_4.push_back(hr_shin_angle);
        // data1t_4.push_back(hr_angular_position_target);

        // -------------------------------------------------------------------------------------
        // PREPARE TASK TARGET DATA FOR GUI
        // -------------------------------------------------------------------------------------
        // Helper lambda to wrap angles seamlessly between -Pi and Pi
        auto wrap_angle = [](double angle) {
            return std::atan2(std::sin(angle), std::cos(angle));
        };

        // Store Wrapped Task Targets for GUI
        data1_4.push_back(wrap_angle(hr_shin_angle));
        data1t_4.push_back(wrap_angle(hr_angular_position_target));
        

        
        data3_4.push_back(hrh_linear_position(2));
        data3t_4.push_back(hrh_linear_position_target);
        data_time.push_back(current_time);

        // -------------------------------------------------------------------------------------
        // LOGGING FOR MATLAB PLOTS
        // -------------------------------------------------------------------------------------
        logger.log("time", current_time);

        // 1. Task Space: Shin Angular Position
        logger.log("tl_shin_angle", tl_shin_angle); logger.log("tl_ang_pos_targ", tl_angular_position_target);
        logger.log("tr_shin_angle", tr_shin_angle); logger.log("tr_ang_pos_targ", tr_angular_position_target);
        logger.log("hl_shin_angle", hl_shin_angle); logger.log("hl_ang_pos_targ", hl_angular_position_target);
        logger.log("hr_shin_angle", hr_shin_angle); logger.log("hr_ang_pos_targ", hr_angular_position_target);

        // 2. Task Space: Shin Angular Velocity
        logger.log("tl_shin_ang_vel", tl_angular_velocity); logger.log("tl_ang_vel_targ", shin_rot_vel);
        logger.log("tr_shin_ang_vel", tr_angular_velocity); logger.log("tr_ang_vel_targ", shin_rot_vel);
        logger.log("hl_shin_ang_vel", hl_angular_velocity); logger.log("hl_ang_vel_targ", shin_rot_vel);
        logger.log("hr_shin_ang_vel", hr_angular_velocity); logger.log("hr_ang_vel_targ", shin_rot_vel);

        // 3. Task Space: Thigh Z-Height Position
        logger.log("tlh_z", tlh_linear_position(2)); logger.log("tlh_z_targ", tlh_linear_position_target);
        logger.log("trh_z", trh_linear_position(2)); logger.log("trh_z_targ", trh_linear_position_target);
        logger.log("hlh_z", hlh_linear_position(2)); logger.log("hlh_z_targ", hlh_linear_position_target);
        logger.log("hrh_z", hrh_linear_position(2)); logger.log("hrh_z_targ", hrh_linear_position_target);

        // 4. Task Space: Thigh Z-Velocity
        logger.log("tlh_z_vel", tlh_linear_velocity(2)); logger.log("tlh_z_vel_targ", thigh_lin_vel);
        logger.log("trh_z_vel", trh_linear_velocity(2)); logger.log("trh_z_vel_targ", thigh_lin_vel);
        logger.log("hlh_z_vel", hlh_linear_velocity(2)); logger.log("hlh_z_vel_targ", thigh_lin_vel);
        logger.log("hrh_z_vel", hrh_linear_velocity(2)); logger.log("hrh_z_vel_targ", thigh_lin_vel);

        // 5. Task Space: Forward Body Velocity
        // Target is set to 0.0 since a specific forward target isn't explicitly tracked in this loop phase
        logger.log("current_forward_velocity", state.linear_body_velocity(0)); 
        logger.log("forward_velocity_target", 0.0); 

        // 6. Commanded (QP) vs Actual (Physics) Torques
        // Using `torque_command` for commanded and `state.torque_estimate` for actual physics.
        // Assumes typical 8-actuator layout mapping: [0:1] TL, [2:3] TR, [4:5] HL, [6:7] HR

        // Torso Left (bl)
        logger.log("bl_hip_cmd", torque_command[0]); logger.log("bl_hip_actual", state.torque_estimate[0]);
        logger.log("bl_knee_cmd", torque_command[1]); logger.log("bl_knee_actual", state.torque_estimate[1]);
        
        // Torso Right (br)
        logger.log("br_hip_cmd", torque_command[2]); logger.log("br_hip_actual", state.torque_estimate[2]);
        logger.log("br_knee_cmd", torque_command[3]); logger.log("br_knee_actual", state.torque_estimate[3]);
        
        // Head Left (fl)
        logger.log("fl_hip_cmd", torque_command[4]); logger.log("fl_hip_actual", state.torque_estimate[4]);
        logger.log("fl_knee_cmd", torque_command[5]); logger.log("fl_knee_actual", state.torque_estimate[5]);
        
        // Head Right (fr)
        logger.log("fr_hip_cmd", torque_command[6]); logger.log("fr_hip_actual", state.torque_estimate[6]);
        logger.log("fr_knee_cmd", torque_command[7]); logger.log("fr_knee_actual", state.torque_estimate[7]);        

        // -------------------------------------------------------------------------------------
        // APPLY CONTROL & LOG
        // -------------------------------------------------------------------------------------
        
        mj_data->ctrl = torque_command.data();
        logger.endStep();

        // -------------------------------------------------------------------------------------
        // VISUALIZATION
        // -------------------------------------------------------------------------------------
        if(visualization_timer > visualization_interval) {
            visualization_start_time = mj_data->time;

            int max_points = 1000; 
            int total_history = data_time.size();
            int count_to_plot = std::min(total_history, max_points);

            for (int k = 0; k < count_to_plot; ++k) {
                int i = (int)((double)k / (count_to_plot - 1) * (total_history - 1));

                if (show_task_plots) {
                    // Populate Shin Plot (Top)
                    fig_shin.linedata[0][2*k] = (float)data_time[i]; fig_shin.linedata[0][2*k+1] = (float)data1t_4[i]; // Target
                    fig_shin.linedata[1][2*k] = (float)data_time[i]; fig_shin.linedata[1][2*k+1] = (float)data1_4[i];  // Actual
                    
                    // Populate Height Plot (Bottom)
                    fig_height.linedata[0][2*k] = (float)data_time[i]; fig_height.linedata[0][2*k+1] = (float)data3t_4[i]; // Target
                    fig_height.linedata[1][2*k] = (float)data_time[i]; fig_height.linedata[1][2*k+1] = (float)data3_4[i];  // Actual
                } else {
                    // Populate Fz Plot (Full)
                    // fig_fz.linedata[0][2*k] = (float)data_time[i]; fig_fz.linedata[0][2*k+1] = (float)data_red[i];   // Limit
                    fig_fz.linedata[1][2*k] = (float)data_time[i]; fig_fz.linedata[1][2*k+1] = (float)data_blue[i];  // QP Plan
                    fig_fz.linedata[2][2*k] = (float)data_time[i]; fig_fz.linedata[2][2*k+1] = (float)data_green[i]; // Actual
                }
            }
            
            fig_shin.linepnt[0] = count_to_plot; fig_shin.linepnt[1] = count_to_plot;
            fig_height.linepnt[0] = count_to_plot; fig_height.linepnt[1] = count_to_plot;
            // fig_fz.linepnt[0] = count_to_plot; 
            fig_fz.linepnt[1] = count_to_plot; fig_fz.linepnt[2] = count_to_plot;

            float min_t = std::max(0.0, current_time - 5.0); 
            float max_t = std::max(5.0, current_time);       
            fig_shin.range[0][0] = min_t; fig_shin.range[0][1] = max_t;
            fig_height.range[0][0] = min_t; fig_height.range[0][1] = max_t;
            fig_fz.range[0][0] = min_t; fig_fz.range[0][1] = max_t;

            mjrRect viewport_full = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport_full.width, &viewport_full.height);

            mjrRect viewport_sim = viewport_full;
            int plot_width = viewport_full.width / 3;
            viewport_sim.width -= plot_width;

            mjrRect viewport_plot_top = {viewport_sim.width, viewport_full.height / 2, plot_width, viewport_full.height / 2};
            mjrRect viewport_plot_bot = {viewport_sim.width, 0, plot_width, viewport_full.height / 2};
            mjrRect viewport_plot_full = {viewport_sim.width, 0, plot_width, viewport_full.height};

            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport_sim, &scn, &con);
            
            if (show_task_plots) {
                mjr_figure(viewport_plot_top, &fig_shin, &con);
                mjr_figure(viewport_plot_bot, &fig_height, &con);
            } else {
                mjr_figure(viewport_plot_full, &fig_fz, &con);
            }

            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << "Time: " << mj_data->time << " s";
            mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport_sim, ss.str().c_str(), 0, &con);

            std::stringstream ss2;
            ss2 << "      [PHYS] [MASK]\n--------------------\n";
            auto row = [&](std::string label, int idx) {
                ss2 << label << ":   " << raw_physics[idx] << "      " << (contact_check2[idx] > 0.5) << "\n";
            };
            row("TL_F", 0); row("TL_B", 1); row("TR_F", 2); row("TR_B", 3);
            ss2 << "--------------------\n";
            row("HL_F", 4); row("HL_B", 5); row("HR_F", 6); row("HR_B", 7);
            mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, viewport_sim, ss2.str().c_str(), 0, &con);

            if (vid_record_flag && ffmpeg_pipe != nullptr) {
                mjrRect record_viewport = {0, 0, win_width, win_height};
                mjr_readPixels(rgb_buffer, nullptr, record_viewport, &con);
                fwrite(rgb_buffer, 1, win_width * win_height * 3, ffmpeg_pipe);
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    // Clean up
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