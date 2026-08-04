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
#include "osqp++.h" // Required for OsqpExitCode

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
    // Stores data: Map key = "Column Name", Value = Vector of data over time
    std::map<std::string, std::vector<double>> data_map;
    std::vector<std::string> headers; // To keep column order consistent
    size_t rows = 0;

public:
    // Reserve memory to prevent lag during simulation resizing
    void reserve(size_t estimated_steps) {
        for (auto& pair : data_map) {
            pair.second.reserve(estimated_steps);
        }
    }

    // --- GENERIC LOGGING FUNCTIONS ---

    // Log a single double
    void log(const std::string& name, double value) {
        if (data_map.find(name) == data_map.end()) {
            headers.push_back(name);
            data_map[name].reserve(rows + 1000); // Catch up reserve
            // Fill previous rows with zeros if a new variable appears late (optional safety)
            data_map[name].resize(rows, 0.0); 
        }
        data_map[name].push_back(value);
    }

    // Log a MuJoCo Array (like qpos, qvel, ctrl)
    void logArray(const std::string& prefix, const double* arr, int size) {
        for (int i = 0; i < size; ++i) {
            log(prefix + "_" + std::to_string(i), arr[i]);
        }
    }

    // Log an Eigen Vector
    template <typename Derived>
    void logEigen(const std::string& prefix, const Eigen::MatrixBase<Derived>& vec) {
        for (int i = 0; i < vec.size(); ++i) {
            log(prefix + "_" + std::to_string(i), vec(i));
        }
    }

    // --- MUJOCO SPECIFIC HELPER ---
    // Logs qpos, qvel, qacc, ctrl, and sensordata in one line
    void logMjData(const mjModel* m, const mjData* d) {
        log("time", d->time);
        logArray("qpos", d->qpos, m->nq);
        logArray("qvel", d->qvel, m->nv);
        logArray("qacc", d->qacc, m->nv);
        logArray("ctrl", d->ctrl, m->nu);
        logArray("sens", d->sensordata, m->nsensordata);
    }

    // Call this at the VERY END of the loop step to confirm row count
    void endStep() {
        rows++;
    }

    // Save to CSV for MATLAB
    void save(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        // Write Header
        for (size_t i = 0; i < headers.size(); ++i) {
            file << headers[i];
            if (i < headers.size() - 1) file << ",";
        }
        file << "\n";

        // Write Data
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

// checks if the <value> is in the <vector>
template <typename T>
bool contains(const std::vector<T>& vec, const T& value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

// find sites on the same geom id
std::vector<int> getSiteIdsOnSameBodyAsGeom(const mjModel* m, int geom_id) {
    std::vector<int> associated_site_ids; 

    if (geom_id < 0 || geom_id >= m->ngeom) {
        std::cerr << "Error: Invalid geom ID: " << geom_id << std::endl;
        return associated_site_ids; 
    }

    int geom_body_id = m->geom_bodyid[geom_id];

    for (int i = 0; i < m->nsite; ++i) {
        if (m->site_bodyid[i] == geom_body_id) {
            associated_site_ids.push_back(i); 
        }
    }
    return associated_site_ids; 
}

// outputs vector C as binary of if vector A elements are in vector B
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
    // Use runfiles to find the path to the model file
    std::string error;
    std::unique_ptr<Runfiles> runfiles(
        Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error)
    );

    std::filesystem::path osc_model_path = 
        runfiles->Rlocation("mujoco-models/models/walter_sr/WaLTER_Senior_updated_noda.xml");
    
    std::filesystem::path simulation_model_path = 
        runfiles->Rlocation("mujoco-models/models/walter_sr/scene_walter_sr_updated_noda_stairs.xml");

    // Load Simulation Model
    char mj_error[1000];
    mjModel* mj_model = mj_loadXML(simulation_model_path.c_str(), nullptr, mj_error, 1000);
    if (!mj_model) {
        printf("%s\n", mj_error);
        return 1;
    }
    mjData* mj_data = mj_makeData(mj_model);

    // Reset Data to match Keyframe 2
    mj_resetDataKeyframe(mj_model, mj_data, 3);
    mj_forward(mj_model, mj_data);



    // std::cout << "\n================= MODEL DEBUG INFO =================" << std::endl;
    // std::cout << "--- SITES ---" << std::endl;
    // for (int i = 0; i < mj_model->nsite; ++i) {
    //     const char* name = mj_id2name(mj_model, mjOBJ_SITE, i);
    //     std::cout << "Site ID: " << i 
    //               << " | Name: " << (name ? name : "UNNAMED") 
    //               << " | Body ID: " << mj_model->site_bodyid[i] << std::endl;
    // }

    // std::cout << "\n--- GEOMS ---" << std::endl;
    // for (int i = 0; i < mj_model->ngeom; ++i) {
    //     const char* name = mj_id2name(mj_model, mjOBJ_GEOM, i);
    //     std::cout << "Geom ID: " << i 
    //               << " | Name: " << (name ? name : "UNNAMED") 
    //               << " | Body ID: " << mj_model->geom_bodyid[i] << std::endl;
    // }
    // std::cout << "====================================================\n" << std::endl;    

    

    // Visualization:
    mjvCamera cam;
    mjvPerturb pert;
    mjvOption opt;
    mjvScene scn;
    mjrContext con;

    // Create GLFW window
    glfwInit();

    GLFWmonitor* primary_monitor = glfwGetPrimaryMonitor();
    //===================
    int win_width = 1920;
    int win_height = 1080;
    GLFWwindow* window = glfwCreateWindow(win_width, win_height, "Sim + Plot", primary_monitor, NULL); 
    //===================
    
    // GLFWwindow* window = glfwCreateWindow(1200, 600, "Sim + Plot", NULL, NULL); 
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Initialize visualization data structures:
    mjv_defaultCamera(&cam);
    mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    mjv_makeScene(mj_model, &scn, 1000);
    mjr_makeContext(mj_model, &con, mjFONTSCALE_150);

    // INITIALIZE THE FIGURE
    mjvFigure fig;
    mjv_defaultFigure(&fig);

    // Setup Titles and Options
    strcpy(fig.title, "Ang Acc (rad/s^2) - Torso left knee");
    strcpy(fig.xlabel, "Time (s)");
    fig.flg_extend = 0;       
    fig.range[0][0] = 0.0f; 
    fig.range[0][1] = 30.0f; 
    fig.gridsize[0] = 6;      
    strcpy(fig.xformat, "%.0f"); 
    fig.range[1][0] = -500.0f; 
    fig.range[1][1] =  500.0f;
    fig.gridsize[1] = 10;     
    strcpy(fig.yformat, "%.1f"); 
    fig.flg_barplot = 0;     
    
    strcpy(fig.linename[0], "PD");
    strcpy(fig.linename[1], "QP");
    strcpy(fig.linename[2], "Real");
    
    fig.linergb[0][0] = 1.0f; fig.linergb[0][1] = 0.0f; fig.linergb[0][2] = 0.0f; // Red
    fig.linergb[1][0] = 0.0f; fig.linergb[1][1] = 0.0f; fig.linergb[1][2] = 1.0f; // Blue
    fig.linergb[2][0] = 0.0f; fig.linergb[2][1] = 1.0f; fig.linergb[2][2] = 0.0f; // Greenish

    // Framebuffer viewport:
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    glfwSwapBuffers(window);
    glfwPollEvents();

    // Initialize Operational Space Controller
    OperationalSpaceController controller(osc_model_path);

    // Initial Data Setup
    Vector<model::nq_size> qpos = Eigen::Map<Vector<model::nq_size>>(mj_data->qpos);
    Vector<model::nv_size> qvel = Eigen::Map<Vector<model::nv_size>>(mj_data->qvel);
    Vector<model::nv_size> qfrc_actuator = Eigen::Map<Vector<model::nv_size>>(mj_data->qfrc_actuator);
    
    Vector<3> initial_position = qpos(Eigen::seqN(0, 3));
    Eigen::Matrix<double, model::site_ids_size, 3> site_data;
    Eigen::Matrix<double, model::site_ids_size, 3> initial_site_data;

    Eigen::Vector<double, model::contact_site_ids_size> contact_check2;
    Eigen::Matrix<double, model::site_ids_size, 9> site_rotational_data;
    Eigen::Matrix<double, model::site_ids_size, 9> initial_site_rotational_data;

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

    // Time setup
    double visualization_timer = mj_data->time;
    double visualization_start_time = visualization_timer;
    double visualization_interval = 0.01;
    double simulation_time = 30.0;
    auto current_time = mj_data->time;
    double last_time = current_time;

    // Data recording vectors
    std::vector<std::string> sites;
    std::vector<int> site_ids;
    // ... [Vectors omitted for brevity, same as original] ... 
    // (Ensure you keep your vector definitions here as in your original code)
    std::vector<double> data1_1, data1_2, data1_3, data1_4, data1t_1, data1t_2, data1t_3, data1t_4;
    std::vector<double> data2_1, data2_2, data2_3, data2_4, data2t_1, data2t_2, data2t_3, data2t_4;    
    std::vector<double> data3_1, data3_2, data3_3, data3_4, data3t_1, data3t_2, data3t_3, data3t_4;    
    std::vector<double> data4_1, data4_2, data4_3, data4_4, data4t_1, data4t_2, data4t_3, data4t_4;    
    std::vector<double> data5_1, data5_2, data5_3, data5t_1, data5t_2, data5t_3;
    std::vector<double> data6_1, data6_2, data6_3, data6t_1, data6t_2, data6t_3, data6t_4, data6t_5, data6t_6, data6t_7, data6t_8, data6t_9;
    std::vector<double> data7_1, data7_2, data7_3, data7_4, data7_5, data7_6, data7_7, data7_8;
    std::vector<double> data_time;

    // Get site IDs
    for(const std::string_view& site : model::site_list) {
        std::string site_str = std::string(site);
        int id = mj_name2id(mj_model, mjOBJ_SITE, site_str.data());
        sites.push_back(site_str);
        site_ids.push_back(id);
    }

    initial_site_data = Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->site_xpos)(site_ids, Eigen::placeholders::all);
    initial_site_rotational_data = Eigen::Map<Matrix<model::site_ids_size, 9>>(mj_data->site_xmat)(site_ids, Eigen::placeholders::all);

    // Initial Angles
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
    // std::vector<int> wheel_geom_ids = {4, 5, 8, 9, 13, 14, 17, 18};    
    std::vector<int> wheel_geom_ids = {26, 27, 30, 31, 35, 36, 39, 40}; // Updated IDs    

    // Initialize Logger
    SimLogger logger;
    logger.reserve(30000); // Reserve memory for 30s @ 1kHz (approx)


    // --- TIED STATE TRACKING ---
    // Torso (Indices 0-3)
    bool prev_torso_front_any = false;
    bool prev_torso_back_any  = false;
    int active_torso_mode = 0; // 0=All, 1=Front Only, 2=Back Only
    int torso_flight_timer = 0;

    // Head (Indices 4-7)
    bool prev_head_front_any = false;
    bool prev_head_back_any  = false;
    int active_head_mode = 0; // 0=All, 1=Front Only, 2=Back Only
    int head_flight_timer = 0;
    
    const int FLIGHT_THRESHOLD = 200; // 0.2s hysteresis -- try 50 and 150?


    // Initialize FFmpeg pipe for 4K recording
    const int fps = 100; // Matches your visualization_interval of 0.01
    std::string ffmpeg_cmd = "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size " + 
                                std::to_string(win_width) + "x" + std::to_string(win_height) + 
                                " -framerate " + std::to_string(fps) + 
                                " -i - -vf vflip -c:v h264_nvenc -preset hq -b:v 10M -pix_fmt yuv420p /home/vivek/walter_sim_1080p.mp4";    // Note: requires <cstdio>
    FILE* ffmpeg_pipe = popen(ffmpeg_cmd.c_str(), "w");
    if (!ffmpeg_pipe) {
        std::cerr << "Failed to open FFmpeg pipe. Ensure FFmpeg is installed and in your system PATH." << std::endl;
        return 1;
    }
    
    // Allocate memory for the 4K RGB buffer
    unsigned char* rgb_buffer = new unsigned char[win_width * win_height * 3];    

    // =========================================================================================
    // CORRECTED LOGIC LOOP: Step -> Kinematics -> Control
    // =========================================================================================
    while(current_time < simulation_time) {
        
        // 1. ADVANCE STATE (t -> t+1) using previous control
        mj_step(mj_model, mj_data);

        // 2. UPDATE KINEMATICS & DYNAMICS (t+1)
        //    mj_fwdPosition: updates xpos, xmat, geometry positions, and CONTACTS for t+1
        //    mj_fwdVelocity: updates bias forces (qfrc_bias) for t+1
        mj_fwdPosition(mj_model, mj_data);
        mj_fwdVelocity(mj_model, mj_data);

        // 3. UPDATE TIME VARIABLES
        current_time = mj_data->time;
        visualization_timer = current_time - visualization_start_time;

        // -------------------------------------------------------------------------------------
        // SENSOR READINGS & STATE ESTIMATION
        // -------------------------------------------------------------------------------------
        qpos = Eigen::Map<Vector<model::nq_size>>(mj_data->qpos);
        qvel = Eigen::Map<Vector<model::nv_size>>(mj_data->qvel);
        qfrc_actuator = Eigen::Map<Vector<model::nv_size>>(mj_data->qfrc_actuator);

        site_data = Eigen::Map<Matrix<model::site_ids_size, 3>>(mj_data->site_xpos)(site_ids, Eigen::placeholders::all);
        site_rotational_data = Eigen::Map<Matrix<model::site_ids_size, 9>>(mj_data->site_xmat)(site_ids, Eigen::placeholders::all);

        // ------------------------ CHECK CONTACTS (Using valid mj_data->contact from mj_fwdPosition) ------------------------  
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

        // Debug print (optional)
        std::cout << "Contact Mask: " << contact_check2.transpose() << std::endl;    



        // --- contact prescription ---
        // ... inside while loop, after calculating contact_check2 ...

        // MAPPING: 
        // i=0 (Torso Left)  -> Front: 0, Back: 1
        // i=1 (Torso Right) -> Front: 2, Back: 3
        // i=2 (Head Left)   -> Front: 4, Back: 5
        // i=3 (Head Right)  -> Front: 6, Back: 7

        // ... inside while loop ...

        // =========================================================
        // LOGIC A: TORSO (Left & Right Tied Together)
        // Indices: Front=[0, 2], Back=[1, 3]
        // =========================================================
        
        // 1. Aggregated Sensing: Is "The Front" or "The Back" touching?
        bool torso_front_phys = (contact_check2[0] > 0.5 || contact_check2[2] > 0.5);
        bool torso_back_phys  = (contact_check2[1] > 0.5 || contact_check2[3] > 0.5);

        // 2. Rising Edge Detection
        bool torso_front_hit = torso_front_phys && !prev_torso_front_any;
        bool torso_back_hit  = torso_back_phys  && !prev_torso_back_any;

        // 3. Mode Selection
        if (torso_front_hit && torso_back_hit) {
            // Flat Landing: Reset to use ALL wheels to prevent tipping
            active_torso_mode = 0; 
            torso_flight_timer = 0;
        }
        else if (torso_front_hit) {
            // Front Hit: Pivot on Front (Mask Back)
            active_torso_mode = 1; 
            torso_flight_timer = 0;
        }
        else if (torso_back_hit) {
            // Back Hit: Pivot on Back (Mask Front)
            active_torso_mode = 2; 
            torso_flight_timer = 0;
        }

        // 4. Flight Hysteresis (Reset only after prolonged airtime)
        // if (!torso_front_phys && !torso_back_phys) {
        //     torso_flight_timer++;
        //     if (torso_flight_timer > FLIGHT_THRESHOLD) active_torso_mode = 0;
        // } else {
        //     torso_flight_timer = 0;
        // }

        // 5. Apply Mask (The "Nudge")
        if (active_torso_mode == 1) {
            // Front Mode: Zero out BOTH Back wheels
            contact_check2[1] = 0.0; // Left Rear
            contact_check2[3] = 0.0; // Right Rear
        } 
        else if (active_torso_mode == 2) {
            // Back Mode: Zero out BOTH Front wheels
            contact_check2[0] = 0.0; // Left Front
            contact_check2[2] = 0.0; // Right Front
        }

        prev_torso_front_any = torso_front_phys;
        prev_torso_back_any  = torso_back_phys;


        // =========================================================
        // LOGIC B: HEAD (Left & Right Tied Together)
        // Indices: Front=[4, 6], Back=[5, 7]
        // =========================================================

        bool head_front_phys = (contact_check2[4] > 0.5 || contact_check2[6] > 0.5);
        bool head_back_phys  = (contact_check2[5] > 0.5 || contact_check2[7] > 0.5);

        bool head_front_hit = head_front_phys && !prev_head_front_any;
        bool head_back_hit  = head_back_phys  && !prev_head_back_any;

        if (head_front_hit && head_back_hit) {
            active_head_mode = 0; 
            head_flight_timer = 0;
        }
        else if (head_front_hit) {
            active_head_mode = 1; 
            head_flight_timer = 0;
        }
        else if (head_back_hit) {
            active_head_mode = 2; 
            head_flight_timer = 0;
        }

        if (!head_front_phys && !head_back_phys) {
            head_flight_timer++;
            if (head_flight_timer > FLIGHT_THRESHOLD) active_head_mode = 0;
        } else {
            head_flight_timer = 0;
        }

        if (active_head_mode == 1) {
            // Front Mode: Mask Back
            contact_check2[5] = 0.0; 
            contact_check2[7] = 0.0; 
        } 
        else if (active_head_mode == 2) {
            // Back Mode: Mask Front
            contact_check2[4] = 0.0; 
            contact_check2[6] = 0.0; 
        }

        prev_head_front_any = head_front_phys;
        prev_head_back_any  = head_back_phys;
        
        // ... continue to state update ...
        // =========================================================






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

        double shin_rot_vel = 0.1*8.0*1.0*0.1; 

        double phase_offset = 3.14159*0.0 / 2.0;

        double tl_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[2]];        
        double tr_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[4]];        
        double hl_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[6]];        
        double hr_shin_angle = mj_data->qpos[mj_model->jnt_qposadr[8]];        
        
        double tl_angular_position = tl_shin_angle;
        double tr_angular_position = tr_shin_angle;
        double hl_angular_position = hl_shin_angle;
        double hr_angular_position = hr_shin_angle;

        double dt = current_time - last_time;
        if (dt == 0) dt = 0.0001; // Avoid div by zero on first step if logic allows

        double tl_angular_velocity = (tl_angular_position - last_tl_angular_position)/dt;
        double tr_angular_velocity = (tr_angular_position - last_tr_angular_position)/dt;
        double hl_angular_velocity = (hl_angular_position - last_hl_angular_position)/dt;
        double hr_angular_velocity = (hr_angular_position - last_hr_angular_position)/dt;

        // targets
        double tl_angular_position_target = initial_tl_angular_position + shin_rot_vel * current_time + phase_offset;
        double tr_angular_position_target = initial_tr_angular_position + shin_rot_vel * current_time + phase_offset;
        double hl_angular_position_target = initial_hl_angular_position + shin_rot_vel * current_time;
        double hr_angular_position_target = initial_hr_angular_position + shin_rot_vel * current_time;

        double tl_angular_velocity_target = shin_rot_vel;
        double tr_angular_velocity_target = shin_rot_vel;
        double hl_angular_velocity_target = shin_rot_vel;
        double hr_angular_velocity_target = shin_rot_vel;

        double tl_angular_position_error = (tl_angular_position_target - tl_angular_position);
        double tr_angular_position_error = (tr_angular_position_target - tr_angular_position);
        double hl_angular_position_error = (hl_angular_position_target - hl_angular_position);
        double hr_angular_position_error = (hr_angular_position_target - hr_angular_position);
        
        double tl_angular_velocity_error = (tl_angular_velocity_target - tl_angular_velocity);
        double tr_angular_velocity_error = (tr_angular_velocity_target - tr_angular_velocity);
        double hl_angular_velocity_error = (hl_angular_velocity_target - hl_angular_velocity);
        double hr_angular_velocity_error = (hr_angular_velocity_target - hr_angular_velocity);

        // ---------------------------------------------------------------------------------------------------------------------------------------------
        double shin_kp = 1.0*1000.0; 
        double shin_kv = 0.5*400.0;
        // ---------------------------------------------------------------------------------------------------------------------------------------------

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

        // ---------------------------------------------------------------------------------------------------------------------------------------------
        double thigh_lin_vel = 0.0;
        double thigh_lin_kp = 1.0*500.0; 
        double thigh_lin_kv = 0.5*500.0;
        // ---------------------------------------------------------------------------------------------------------------------------------------------

        Vector<3> tlh_linear_position = site_data(5,Eigen::seqN(0, 3));
        Vector<3> trh_linear_position = site_data(6,Eigen::seqN(0, 3));
        Vector<3> hlh_linear_position = site_data(7,Eigen::seqN(0, 3));
        Vector<3> hrh_linear_position = site_data(8,Eigen::seqN(0, 3));
    
        Vector<3> tlh_linear_velocity = (tlh_linear_position - last_tlh_linear_position)/dt;
        Vector<3> trh_linear_velocity = (trh_linear_position - last_trh_linear_position)/dt;
        Vector<3> hlh_linear_velocity = (hlh_linear_position - last_hlh_linear_position)/dt;
        Vector<3> hrh_linear_velocity = (hrh_linear_position - last_hrh_linear_position)/dt;

        // targets
        double tlh_linear_velocity_target = thigh_lin_vel;
        double trh_linear_velocity_target = thigh_lin_vel;
        double hlh_linear_velocity_target = thigh_lin_vel;
        double hrh_linear_velocity_target = thigh_lin_vel;
        Vector<3> body_position = qpos(Eigen::seqN(0, 3));

        double thigh_height_increase_stairs = -0.01;

        double tlh_linear_position_target = initial_site_data(5,2) - 0.0 + thigh_height_increase_stairs;
        double trh_linear_position_target = initial_site_data(6,2) - 0.0 + thigh_height_increase_stairs;
        double hlh_linear_position_target = initial_site_data(7,2) - 0.0 + thigh_height_increase_stairs;
        double hrh_linear_position_target = initial_site_data(8,2) - 0.0 + thigh_height_increase_stairs;


        double tlh_linear_position_error = ( (tlh_linear_position_target) - tlh_linear_position(2));
        double trh_linear_position_error = ( (trh_linear_position_target) - trh_linear_position(2));
        double hlh_linear_position_error = ( (hlh_linear_position_target) - hlh_linear_position(2));
        double hrh_linear_position_error = ( (hrh_linear_position_target) - hrh_linear_position(2));

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

        // Forward Velocity Control
        double forward_velocity_target = 0.1; 
        double current_forward_velocity = state.linear_body_velocity(0); 
        double forward_kp = 200.0*0.1; 
        // double x_accel_cmd = forward_kp * (forward_velocity_target - current_forward_velocity);
        double x_accel_cmd = 0.0;

        Eigen::Vector<double, 6> cmd5 {x_accel_cmd, 0, tlh_linear_control, 0, 0, 0};
        Eigen::Vector<double, 6> cmd6 {x_accel_cmd, 0, trh_linear_control, 0, 0, 0};
        Eigen::Vector<double, 6> cmd7 {x_accel_cmd, 0, hlh_linear_control, 0, 0, 0};
        Eigen::Vector<double, 6> cmd8 {x_accel_cmd, 0, hrh_linear_control, 0, 0, 0};        

        taskspace_targets.row(5) = cmd5;
        taskspace_targets.row(6) = cmd6;
        taskspace_targets.row(7) = cmd7;
        taskspace_targets.row(8) = cmd8;        

        // Torso Orientation / Height Control
        double xvel = 0.1;
        Vector<3> position_target = Vector<3>(
            initial_position(0) + xvel*current_time, initial_position(1), initial_position(2)
        );
        Vector<3> velocity_target = Vector<3>(xvel,0.0,0.0);

        Eigen::Quaternion<double> body_rotation = Eigen::Quaternion<double>(state.body_rotation(0), state.body_rotation(1), state.body_rotation(2), state.body_rotation(3));
        Vector<3> position_error = position_target - body_position;
        Vector<3> velocity_error = velocity_target - state.linear_body_velocity;
        Vector<3> rotation_error = (Eigen::Quaternion<double>(1, 0, 0, 0) * body_rotation.conjugate()).vec();
        Vector<3> angular_velocity_error = Vector<3>::Zero() - state.angular_body_velocity;

        double torso_lin_kp = 10.0;
        double torso_lin_kv = 1.0;
        double torso_ang_kp = 0.0;
        double torso_ang_kv = 0.0;        

        Vector<3> linear_control = torso_lin_kp * (position_error) + torso_lin_kv * (velocity_error);
        Vector<3> angular_control = torso_ang_kp * (rotation_error) + torso_ang_kv * (angular_velocity_error);

        // Update Camera
        cam.lookat[0] = body_position(0);

        // -------------------------------------------------------------------------------------
        // COMPUTE TORQUES (OSC_Compute)
        // -------------------------------------------------------------------------------------
        // Now using valid M, J, h from mj_fwdPosition/Velocity
        controller.update_taskspace_targets(taskspace_targets);
        
        Vector<model::nu_size> torque_command = controller.get_torque_command();

        // -------------------------------------------------------------------------------------
        // CHECK SOLVER STATUS (Exit Loop on Failure)
        // -------------------------------------------------------------------------------------
        // Retrieve internal solver status (casted to int from OsqpExitCode in controller)
        int solver_status_int = controller.get_solver_status(); 
        
        // Cast to osqp::OsqpExitCode for comparison
        osqp::OsqpExitCode solver_status = static_cast<osqp::OsqpExitCode>(solver_status_int);

        // Check if optimal (solved). IF NOT OPTIMAL, BREAK.
        // This ensures "Solved Inaccurate" or "Max Iterations" triggers an immediate save.
        if (solver_status != osqp::OsqpExitCode::kOptimal) {
             std::cout << "Solver failed with status: " << (int)solver_status 
                       << " at time " << current_time << ". Exiting loop and saving data." << std::endl;
             // Loop exit triggers data save below
             break; 
        }

        double current_qp_obj = controller.get_last_qp_objective_value();

        int joint_id = mj_name2id(mj_model, mjOBJ_JOINT, "torso_left_thigh_shin_joint"); 
        int qp_index = mj_model->jnt_dofadr[joint_id];
        Vector<optimization::design_vector_size> solution = controller.get_solution();

        // -------------------------------------------------------------------------------------
        // DATA LOGGING (Moved after Solver Check to prevent partial data writes)
        // -------------------------------------------------------------------------------------
        // shin angular position 
        data1_1.push_back(tl_angular_position);
        data1_2.push_back(tr_angular_position);
        data1_3.push_back(hl_angular_position);
        data1_4.push_back(hr_angular_position);
        data1t_1.push_back(tl_angular_position_target);
        data1t_2.push_back(tr_angular_position_target);
        data1t_3.push_back(hl_angular_position_target);
        data1t_4.push_back(hr_angular_position_target);

        // shin angular velocity
        data2_1.push_back(tl_angular_velocity);
        data2_2.push_back(tr_angular_velocity);
        data2_3.push_back(hl_angular_velocity);
        data2_4.push_back(hr_angular_velocity);
        data2t_1.push_back(tl_angular_velocity_target);
        data2t_2.push_back(tr_angular_velocity_target);
        data2t_3.push_back(hl_angular_velocity_target);
        data2t_4.push_back(hr_angular_velocity_target);

        // thigh z position
        data3_1.push_back(tlh_linear_position(2));
        data3_2.push_back(trh_linear_position(2));
        data3_3.push_back(hlh_linear_position(2));
        data3_4.push_back(hrh_linear_position(2));
        data3t_1.push_back(initial_site_data(5,2) - 0.0 + thigh_height_increase_stairs);
        data3t_2.push_back(initial_site_data(6,2) - 0.0 + thigh_height_increase_stairs);
        data3t_3.push_back(initial_site_data(7,2) - 0.0 + thigh_height_increase_stairs);
        data3t_4.push_back(initial_site_data(8,2) - 0.0 + thigh_height_increase_stairs);

        // thigh z velocity
        data4_1.push_back(tlh_linear_velocity(2));
        data4_2.push_back(trh_linear_velocity(2));
        data4_3.push_back(hlh_linear_velocity(2));
        data4_4.push_back(hrh_linear_velocity(2));
        data4t_1.push_back(tlh_linear_velocity_target);
        data4t_2.push_back(trh_linear_velocity_target);
        data4t_3.push_back(hlh_linear_velocity_target);
        data4t_4.push_back(hrh_linear_velocity_target);

        // rotation error
        data5_1.push_back(rotation_error(0));
        data5_2.push_back(rotation_error(1));
        data5_3.push_back(rotation_error(2));

        // angular velocity error
        data6_1.push_back(angular_velocity_error(0));
        data6_2.push_back(angular_velocity_error(1));
        data6_3.push_back(angular_velocity_error(2));

        // time
        data_time.push_back(current_time);

        // 1. EXTRACT CONTACT FORCES (z)
        // The z vector starts at u_idx. It contains 3 doubles (Fx, Fy, Fz) for each contact site.
        // Vector<optimization::z_size> all_forces = solution.segment<optimization::z_size>(optimization::u_idx);

        // 2. MAP FORCES TO WHEELS
        // Based on your python objective splitting, the order is:
        // [TLF, TLR, TRF, TRR, HLF, HLR, HRF, HRR]
        // Vector<3> f_torso_LF = all_forces.segment<3>(0);  // Torso Left Front
        // Vector<3> f_torso_LR = all_forces.segment<3>(3);  // Torso Left Rear
        // Vector<3> f_torso_RF = all_forces.segment<3>(6);  // Torso Right Front
        // Vector<3> f_torso_RR = all_forces.segment<3>(9);  // Torso Right Rear

        // Vector<3> f_head_LF  = all_forces.segment<3>(12); // Head Left Front
        // Vector<3> f_head_LR  = all_forces.segment<3>(15); // Head Left Rear
        // Vector<3> f_head_RF  = all_forces.segment<3>(18); // Head Right Front
        // Vector<3> f_head_RR  = all_forces.segment<3>(21); // Head Right Rear

        // 3. PRINT FORCES (Optional: Print only periodically to avoid spamming console)
        // if (visualization_timer > visualization_interval) {
            // std::cout << "--- Contact Forces (N) [x, y, z] ---" << std::endl;
            // std::cout << "Torso LF: " << f_torso_LF.transpose() << std::endl;
            // std::cout << "Torso LR: " << f_torso_LR.transpose() << std::endl;
            // std::cout << "Torso RF: " << f_torso_RF.transpose() << std::endl;
            // std::cout << "Torso RR: " << f_torso_RR.transpose() << std::endl;
            // std::cout << "Head  LF: " << f_head_LF.transpose()  << std::endl;
            // std::cout << "Head  LR: " << f_head_LR.transpose()  << std::endl;
            // std::cout << "Head  RF: " << f_head_RF.transpose()  << std::endl;
            // std::cout << "Head  RR: " << f_head_RR.transpose()  << std::endl;
            // std::cout << "------------------------------------" << std::endl;
        // }
        

        
        double tl_shin_qp_accel = solution(qp_index);
        double tl_shin_real_accel = mj_data->qacc[qp_index];        

        data5t_1.push_back(tl_angular_control);  // Desired (PD)
        data5t_2.push_back(tl_shin_qp_accel);    // Planned (QP)
        data5t_3.push_back(tl_shin_real_accel);  // Actual (Sim)

        int tl_joint_id = mj_name2id(mj_model, mjOBJ_JOINT, "torso_left_thigh_shin_joint");
        int tl_dof_adr = mj_model->jnt_dofadr[tl_joint_id];
        double tl_angular_velocity2 = mj_data->qvel[tl_dof_adr];       

        data6t_1.push_back(tl_angular_velocity2);
        data6t_2.push_back(torque_command[1]);
        data6t_3.push_back(torque_command[2]);        
        data6t_4.push_back(torque_command[3]);
        data6t_5.push_back(torque_command[4]);
        data6t_6.push_back(torque_command[5]);        
        data6t_7.push_back(torque_command[6]);
        data6t_8.push_back(torque_command[7]);
        data6t_9.push_back(current_qp_obj);        

        data7_1.push_back(mj_data->qacc[6]);
        data7_2.push_back(mj_data->qacc[7]);
        data7_3.push_back(mj_data->qacc[8]);        
        data7_4.push_back(mj_data->qacc[9]);
        data7_5.push_back(mj_data->qacc[10]);
        data7_6.push_back(mj_data->qacc[11]);        
        data7_7.push_back(mj_data->qacc[12]);
        data7_8.push_back(current_forward_velocity);        

        // -------------------------------------------------------------------------------------
        // 4. APPLY CONTROL
        // -------------------------------------------------------------------------------------
        // Set the control for the NEXT mj_step (top of the loop)
        mj_data->ctrl = torque_command.data();

        // =============================================================== LOGGING ->
        // logger.log(mj_data->time, current_qp_obj, 
        //        torque_command.data(), torque_command.size(), 
        //        mj_data->qacc, 12);
        // -------------------------------------------------------------------------------------
        // DATA LOGGING
        // -------------------------------------------------------------------------------------
        
        // 1. Log All Standard MuJoCo States (Time, Qpos, Qvel, Qacc, Ctrl)
        // logger.logMjData(mj_model, mj_data);

        // 2. Log Specific Targets (Scalars)

        // 3. Log Eigen Vectors (e.g., Rotation Error)
        // logger.logEigen("rot_err", rotation_error);       // automatically logs rot_err_0, rot_err_1, etc.
        // logger.logEigen("ang_vel_err", angular_velocity_error);

        logger.log("time", current_time);

        // 4. Log Specific OSC Data
        logger.log("qp_obj", current_qp_obj);
        logger.log("fwd_vel_targ", forward_velocity_target);
        logger.logEigen("cmd_torque", torque_command); // Logs entire torque vector

        
        logger.log("tl_shin_angle", tl_shin_angle);
        logger.log("tr_shin_angle", tr_shin_angle);
        logger.log("hl_shin_angle", hl_shin_angle);
        logger.log("hr_shin_angle", hr_shin_angle);

        logger.log("tl_ang_pos_targ", tl_angular_position_target);
        logger.log("tr_ang_pos_targ", tr_angular_position_target);
        logger.log("hl_ang_pos_targ", hl_angular_position_target);
        logger.log("hr_ang_pos_targ", hr_angular_position_target);

        logger.log("tl_shin_ang_vel", tl_angular_velocity);
        logger.log("tr_shin_ang_vel", tr_angular_velocity);
        logger.log("hl_shin_ang_vel", hl_angular_velocity);
        logger.log("hr_shin_ang_vel", hr_angular_velocity);        
        
        logger.log("tl_ang_vel_targ", tl_angular_velocity_target);
        logger.log("tr_ang_vel_targ", tr_angular_velocity_target);
        logger.log("hl_ang_vel_targ", hl_angular_velocity_target);
        logger.log("hr_ang_vel_targ", hr_angular_velocity_target);

        logger.log("tlh_z", tlh_linear_position(2));
        logger.log("trh_z", trh_linear_position(2));
        logger.log("hlh_z", hlh_linear_position(2));
        logger.log("hrh_z", hrh_linear_position(2));

        logger.log("tlh_z_targ", tlh_linear_position_target);
        logger.log("trh_z_targ", trh_linear_position_target);
        logger.log("hlh_z_targ", hlh_linear_position_target);
        logger.log("hrh_z_targ", hrh_linear_position_target);

        logger.log("tlh_z_vel", tlh_linear_velocity(2));
        logger.log("trh_z_vel", trh_linear_velocity(2));
        logger.log("hlh_z_vel", hlh_linear_velocity(2));
        logger.log("hrh_z_vel", hrh_linear_velocity(2));

        logger.log("tlh_z_vel_targ", tlh_linear_velocity_target);
        logger.log("trh_z_vel_targ", trh_linear_velocity_target);
        logger.log("hlh_z_vel_targ", hlh_linear_velocity_target);
        logger.log("hrh_z_vel_targ", hrh_linear_velocity_target);

        logger.log("forward_velocity_target", forward_velocity_target);
        logger.log("current_forward_velocity", current_forward_velocity);


        // logger.logEigen("f_torso_LF", f_torso_LF); 
        // logger.logEigen("f_torso_LR", f_torso_LR); 
        // logger.logEigen("f_torso_RF", f_torso_RF); 
        // logger.logEigen("f_torso_RR", f_torso_RR); 
        // logger.logEigen("f_head_LF",  f_head_LF); 
        // logger.logEigen("f_head_LR",  f_head_LR); 
        // logger.logEigen("f_head_RF",  f_head_RF); 
        // logger.logEigen("f_head_RR",  f_head_RR);


        // 5. Log Contacts (Converting your int contact check to double)
        logger.logEigen("contacts", contact_check2); 

        // CRITICAL: Tell logger this timestep is done
        logger.endStep();
        // =============================================================== LOGGING ->

        // -------------------------------------------------------------------------------------
        // VISUALIZATION
        // -------------------------------------------------------------------------------------
        if(visualization_timer > visualization_interval) {
            visualization_start_time = mj_data->time;

            int max_points = 1000; 
            int total_history = data_time.size();
            int count_to_plot = std::min(total_history, max_points);
            
            // for (int k = 0; k < count_to_plot; ++k) {
            //     int i = (int)((double)k / (count_to_plot - 1) * (total_history - 1));

            //     // --- LINE 0: Red (Desired) ---
            //     fig.linedata[0][2*k]   = (float)data_time[i];      
            //     fig.linedata[0][2*k+1] = (float)data5t_1[i];     
                
            //     // --- LINE 1: Blue (Planned QP) ---
            //     fig.linedata[1][2*k]   = (float)data_time[i];      
            //     fig.linedata[1][2*k+1] = (float)data5t_2[i];     

            //     // --- LINE 2: Green (Actual Sim) ---
            //     fig.linedata[2][2*k]   = (float)data_time[i];      
            //     fig.linedata[2][2*k+1] = (float)data5t_3[i];     
            // }
            
            // fig.linepnt[0] = count_to_plot;
            // fig.linepnt[1] = count_to_plot;
            // fig.linepnt[2] = count_to_plot;

            // if (!fig.flg_extend) {
            //     fig.range[0][0] = std::max(0.0, current_time - 5.0); 
            //     fig.range[0][1] = std::max(5.0, current_time);       
            // }

            mjrRect viewport_sim = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport_sim.width, &viewport_sim.height);

            // mjrRect viewport_plot = viewport_sim;
            // viewport_plot.width = viewport_sim.width / 3;
            // viewport_plot.left = viewport_sim.width - viewport_plot.width; 
            // viewport_sim.width -= viewport_plot.width;

            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport_sim, &scn, &con);
            // mjr_figure(viewport_plot, &fig, &con);

            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << "Time: " << mj_data->time << " s";
            mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport_sim, ss.str().c_str(), 0, &con);

            std::stringstream ss2;
            ss2 << std::fixed << std::setprecision(3) << "torso_left_front_wheel: " << (int)contact_check2[0] << "\n";
            ss2 << std::fixed << std::setprecision(3) << "torso_left_rear_wheel: " << (int)contact_check2[1] << "\n";
            ss2 << std::fixed << std::setprecision(3) << "torso_right_front_wheel: " << (int)contact_check2[2] << "\n";
            ss2 << std::fixed << std::setprecision(3) << "torso_right_rear_wheel: " << (int)contact_check2[3] << "\n";
            ss2 << std::fixed << std::setprecision(3) << "head_left_front_wheel: " << (int)contact_check2[4] << "\n";
            ss2 << std::fixed << std::setprecision(3) << "head_left_rear_wheel: " << (int)contact_check2[5] << "\n";
            ss2 << std::fixed << std::setprecision(3) << "head_right_front_wheel: " << (int)contact_check2[6] << "\n";
            ss2 << std::fixed << std::setprecision(3) << "head_right_rear_wheel: " << (int)contact_check2[7] << "\n";

            mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, viewport_sim, ss2.str().c_str(), 0, &con);
            glfwSwapBuffers(window);
            glfwPollEvents();

            // --- ADD THIS BLOCK TO RECORD ---
            // Read pixels from the entire 4K framebuffer
            mjrRect record_viewport = {0, 0, win_width, win_height};
            mjr_readPixels(rgb_buffer, nullptr, record_viewport, &con);
            
            // Write the raw RGB data into the FFmpeg pipe
            fwrite(rgb_buffer, 1, win_width * win_height * 3, ffmpeg_pipe);
            // --------------------------------
            
        }
    }

    // Clean up visualization:
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Close FFmpeg pipe and free buffer
    if (ffmpeg_pipe) pclose(ffmpeg_pipe);
    delete[] rgb_buffer;
    std::cout << "Video saved as walter_sim_4k.mp4" << std::endl;    


    // Stop Threads and Clean up:
    result.Update(controller.stop_thread());
    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);
    ABSL_CHECK(result.ok()) << result.message();

    // save data to file
    std::ofstream outfile("osc_test_slowtumbling.txt");
    if (outfile.is_open()) {
        outfile << "time data1_1 data1_2 data1_3 data1_4 data1t_1 data1t_2 data1t_3 data1t_4 "
            << "data2_1 data2_2 data2_3 data2_4 data2t_1 data2t_2 data2t_3 data2t_4 "
            << "data3_1 data3_2 data3_3 data3_4 data3t_1 data3t_2 data3t_3 data3t_4 "
            << "data4_1 data4_2 data4_3 data4_4 data4t_1 data4t_2 data4t_3 data4t_4 "
            << "data5_1 data5_2 data5_3 data5t_1 data5t_2 data5t_3 "
            << "data6_1 data6_2 data6_3 data6t_1 data6t_2 data6t_3 data6t_4 data6t_5 data6t_6 data6t_7 data6t_8 data6t_9 "
            << "data7_1 data7_2 data7_3 data7_4 data7_5 data7_6 data7_7 data7_8 " << std::endl;

        for (size_t i = 0; i < data1_1.size(); ++i) {
            outfile << data_time[i] << " " << 
                       data1_1[i] << " " << data1_2[i] << " " << data1_3[i] << " " << data1_4[i] << " "
                    << data1t_1[i] << " " << data1t_2[i] << " " << data1t_3[i] << " " << data1t_4[i] << " "
                    << data2_1[i] << " " << data2_2[i] << " " << data2_3[i] << " " << data2_4[i] << " "
                    << data2t_1[i] << " " << data2t_2[i] << " " << data2t_3[i] << " " << data2t_4[i] << " "
                    << data3_1[i] << " " << data3_2[i] << " " << data3_3[i] << " " << data3_4[i] << " "
                    << data3t_1[i] << " " << data3t_2[i] << " " << data3t_3[i] << " " << data3t_4[i] << " "
                    << data4_1[i] << " " << data4_2[i] << " " << data4_3[i] << " " << data4_4[i] << " "
                    << data4t_1[i] << " " << data4t_2[i] << " " << data4t_3[i] << " " << data4t_4[i] << " "
                    << data5_1[i] << " " << data5_2[i] << " " << data5_3[i] << " "
                    << data5t_1[i] << " " << data5t_2[i] << " " << data5t_3[i] << " "
                    << data6_1[i] << " " << data6_2[i] << " " << data6_3[i]  << " "
                    << data6t_1[i] << " " << data6t_2[i] << " " << data6t_3[i] << " " << data6t_4[i] << " " << data6t_5[i] << " " << data6t_6[i] << " " << data6t_7[i] << " " << data6t_8[i] << " " << data6t_9[i]  << " "            
                    << data7_1[i] << " " << data7_2[i] << " " << data7_3[i] << " " << data7_4[i] << " " << data7_5[i] << " " << data7_6[i] << " " << data7_7[i] << " " << data7_8[i] << std::endl;               
        }
        outfile.close();
        std::cout << "Data saved to data.txt" << std::endl;
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
    // Save data to file
    logger.save("osc_test_data.csv"); // Use .csv extension!    

    return 0;
}