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
#include <deque>
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
// Class to log sim data
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




// -----------------------------------------------------------------------------------------------------

/**
 * @class TelemetryVisualizer
 * @brief Generic visualizer for rendering real-time dual-line telemetry in MuJoCo.
 * 
 * Manages a dynamic number of plots, each rendering exactly two data streams against time.
 * Automatically handles memory buffering and scales viewports based on the number of active plots.
 */
class TelemetryVisualizer {
private:
    struct PlotContext {
        mjvFigure fig;
        std::deque<float> time_buf;
        std::deque<float> line1_buf;
        std::deque<float> line2_buf;
    };

    size_t max_history_;
    std::vector<PlotContext> plots_;

public:
    /**
     * @brief Construct a new Telemetry Visualizer object.
     * @param max_history Maximum number of data points to retain in memory.
     */
    TelemetryVisualizer(size_t max_history = 1000) 
        : max_history_(max_history) {}

    /**
     * @brief Initializes a new 2-line plot and returns its index.
     * 
     * @param title Title of the plot.
     * @param line1_name Legend name for the first data stream (Red).
     * @param line2_name Legend name for the second data stream (Green).
     * @param y_min Minimum value for the Y-axis.
     * @param y_max Maximum value for the Y-axis.
     * @return size_t The integer index used to update this specific plot later.
     */
    size_t addPlot(const std::string& title, 
                   const std::string& line1_name, 
                   const std::string& line2_name, 
                   float y_min, float y_max) {
        
        PlotContext ctx;
        mjv_defaultFigure(&ctx.fig);
        
        ctx.fig.flg_extend = 0;

        // Configure text and labels safely for C-style arrays
        strncpy(ctx.fig.title, title.c_str(), sizeof(ctx.fig.title) - 1);
        strncpy(ctx.fig.xlabel, "Time (s)", sizeof(ctx.fig.xlabel) - 1);
        
        strncpy(ctx.fig.linename[0], line1_name.c_str(), sizeof(ctx.fig.linename[0]) - 1);
        strncpy(ctx.fig.linename[1], line2_name.c_str(), sizeof(ctx.fig.linename[1]) - 1);
        
        // Configure standard contrasting colors
        ctx.fig.linergb[0][0]=1.0f; ctx.fig.linergb[0][1]=0.0f; ctx.fig.linergb[0][2]=0.0f; // Red
        ctx.fig.linergb[1][0]=0.0f; ctx.fig.linergb[1][1]=1.0f; ctx.fig.linergb[1][2]=0.0f; // Green
        
        // Lock Y-axis bounds
        ctx.fig.range[1][0] = y_min; 
        ctx.fig.range[1][1] = y_max;

        plots_.push_back(std::move(ctx));
        return plots_.size() - 1;
    }

    /**
     * @brief Pushes a new frame of data to a specific plot.
     * 
     * @param plot_idx The index of the plot (returned by addPlot).
     * @param time Current simulation time.
     * @param val1 Value for the first line.
     * @param val2 Value for the second line.
     */
    void updateData(size_t plot_idx, double time, double val1, double val2) {
        if (plot_idx >= plots_.size()) return;
        
        PlotContext& plot = plots_[plot_idx];
        
        pushToBuffer(plot.time_buf, static_cast<float>(time));
        pushToBuffer(plot.line1_buf, static_cast<float>(val1));
        pushToBuffer(plot.line2_buf, static_cast<float>(val2));
    }

    /**
     * @brief Renders all registered plots to the GLFW window context.
     * 
     * @param window Pointer to the active GLFW window.
     * @param con Pointer to the MuJoCo OpenGL context.
     */
    void render(GLFWwindow* window, mjrContext* con) {
        if (plots_.empty()) return;

        for (auto& plot : plots_) {
            if (plot.time_buf.empty()) continue;
            populateLineData(plot);
            updateXAxisRange(plot);
        }

        drawViewports(window, con);
    }

private:
    /**
     * @brief Helper to manage fixed-size circular buffers.
     */
    void pushToBuffer(std::deque<float>& buffer, float val) {
        buffer.push_back(val);
        if (buffer.size() > max_history_) {
            buffer.pop_front();
        }
    }

    /**
     * @brief Maps deque data into MuJoCo's C-style float arrays for rendering.
     */
    void populateLineData(PlotContext& plot) {
        int count = plot.time_buf.size();
        for (int k = 0; k < count; ++k) {
            // Line 0
            plot.fig.linedata[0][2*k]   = plot.time_buf[k];
            plot.fig.linedata[0][2*k+1] = plot.line1_buf[k];
            
            // Line 1
            plot.fig.linedata[1][2*k]   = plot.time_buf[k];
            plot.fig.linedata[1][2*k+1] = plot.line2_buf[k];
        }
        plot.fig.linepnt[0] = count;
        plot.fig.linepnt[1] = count;
    }

    /**
     * @brief Creates a scrolling X-axis window showing the last 5 seconds.
     */
    void updateXAxisRange(PlotContext& plot) {
        float current_time = plot.time_buf.back();
        // Change this value to adjust your X-axis limits (e.g., 10.0f for a 10-second view)
        float x_axis_window = 2.0f; 
        
        float min_t = std::max(0.0f, current_time - x_axis_window);
        float max_t = std::max(x_axis_window, current_time);

        plot.fig.range[0][0] = min_t; 
        plot.fig.range[0][1] = max_t;
    }

    /**
     * @brief Automatically stacks plots vertically on the right side of the screen.
     */
    void drawViewports(GLFWwindow* window, mjrContext* con) {
        mjrRect viewport_full = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport_full.width, &viewport_full.height);

        int plot_width = viewport_full.width / 3; // Occupy right 33% of the window
        int num_plots = plots_.size();
        int plot_height = viewport_full.height / num_plots;

        for (int i = 0; i < num_plots; ++i) {
            // Calculate vertically stacked layout (bottom-up rendering in OpenGL)
            mjrRect viewport_plot = {
                viewport_full.width - plot_width, 
                viewport_full.height - (i + 1) * plot_height, 
                plot_width, 
                plot_height
            };
            mjr_figure(viewport_plot, &plots_[i].fig, con);
        }
    }
};



/**
 * @class ProprioceptiveContactEstimator
 * @brief Estimates Z-axis ground reaction forces using 1D Jacobian projection.
 *        Uses terrain-agnostic relative proprioception to handle kinematic blind spots.
 */
class ProprioceptiveContactEstimator {
private:
    double damping_;
    double force_threshold_high_;
    double force_threshold_low_;
    Eigen::VectorXd mask_state_;
    Eigen::VectorXd estimated_forces_; 
    int debug_counter_; 

public:
    ProprioceptiveContactEstimator(int num_wheels, double high_N = 14.0, double low_N = 4.0, double damping = 0.01) 
        : damping_(damping), force_threshold_high_(high_N), force_threshold_low_(low_N) {
        mask_state_ = Eigen::VectorXd::Zero(num_wheels);
        estimated_forces_ = Eigen::VectorXd::Zero(num_wheels);
        debug_counter_ = 0;
    }

    Eigen::VectorXd update(const mjModel* m, mjData* d, const std::vector<int>& wheel_site_ids, const Eigen::VectorXd& tau_meas) {
        int nv = m->nv;
        int num_wheels = wheel_site_ids.size();
        
        bool print_debug = (debug_counter_ % 100 == 0); 
        debug_counter_++;

        if (print_debug) std::cout << "\n--- Contact Estimator Debug (Tick: " << debug_counter_ << ") ---\n";

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M(nv, nv);
        mj_fullM(m, M.data(), d->qM);
        Eigen::VectorXd qacc = Eigen::Map<const Eigen::VectorXd>(d->qacc, nv);
        Eigen::VectorXd qvel = Eigen::Map<const Eigen::VectorXd>(d->qvel, nv); 
        
        Eigen::VectorXd tau_expected = (M * qacc) 
                                     + Eigen::Map<const Eigen::VectorXd>(d->qfrc_bias, nv) 
                                     - Eigen::Map<const Eigen::VectorXd>(d->qfrc_passive, nv);

        Eigen::VectorXd tau_dist = tau_expected - tau_meas;
        int act_dofs = nv - 6; 
        Eigen::VectorXd tau_dist_act = tau_dist.tail(act_dofs);

        // =========================================================
        // LOOP 1: Calculate raw physics and kinematic overrides
        // =========================================================
        for (int i = 0; i < num_wheels; ++i) {
            int site_id = wheel_site_ids[i];
            
            // 1. 1D Projection Math
            Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp(3, nv);
            mj_jacSite(m, d, jacp.data(), nullptr, site_id);
            Eigen::VectorXd J_z = jacp.block(2, 6, 1, act_dofs).transpose(); 
            
            double z_leverage = J_z.squaredNorm();
            double dot_product = J_z.dot(tau_dist_act);
            double raw_fz = 0.0;

            if (z_leverage >= 0.01) { 
                raw_fz = dot_product / (z_leverage + damping_);
            }

            // ---------------------------------------------------------
            // 2. KINEMATIC OVERRIDES (Terrain-Agnostic)
            // ---------------------------------------------------------
            // Traverse the MuJoCo physics tree dynamically to find the knee joint for this specific wheel.
            int body_wheel = m->site_bodyid[site_id];
            int body_shin = m->body_parentid[body_wheel];
            int jnt_knee = m->body_jntadr[body_shin];

            // Extract the gravity-aligned vertical positions.
            double wheel_z = d->site_xpos[3 * site_id + 2];
            double knee_z = d->xanchor[3 * jnt_knee + 2]; 

            // GUARD A: Skyhook Guard 
            // If the wheel is more than 2cm higher than the knee, it is pointing at the ceiling or tucked.
            // It cannot possibly be bearing weight against the ground.
            if (wheel_z > (knee_z + 0.02)) {
                raw_fz = 0.0;
                mask_state_(i) = 0.0;
            }
            // GUARD B: Singularity Hold
            // If the motors lose mechanical leverage (e.g., shin is perfectly horizontal), the sensors go blind.
            // Instead of guessing where the floor is, we freeze the contact mask to whatever it was 1 tick ago.
            else if (z_leverage < 0.01) {
                if (mask_state_(i) > 0.5) {
                    raw_fz = force_threshold_high_ + 5.0; // Keep rendering a force for the UI
                } else {
                    raw_fz = 0.0;
                }
            }
            // STANDARD OPERATION: The leg is angled downward and has torque leverage.
            else {
                if (raw_fz > force_threshold_high_) {
                    mask_state_(i) = 1.0;
                } else if (raw_fz < force_threshold_low_) {
                    mask_state_(i) = 0.0;
                }
            }

            estimated_forces_(i) = raw_fz; 

            if (print_debug && (i == 0 || i == 4)) {
                std::string label = (i == 0) ? "Torso W[0]" : "Head  W[4]";
                std::cout << label << " | "
                          << "Lev: " << std::fixed << std::setprecision(4) << std::setw(6) << z_leverage 
                          << " | WheelZ: " << std::setw(6) << wheel_z
                          << " | KneeZ: " << std::setw(6) << knee_z
                          << " | Fz Est: " << std::setw(8) << raw_fz 
                          << " | Mask: " << mask_state_(i) << "\n";
            }
        }

        // =========================================================
        // LOOP 2: The Velocity-Based Mutually Exclusive Guard
        // =========================================================
        for (int i = 0; i < num_wheels; ++i) {
            if (mask_state_(i) < 0.5) continue; 

            int site_i = wheel_site_ids[i];
            int body_i = m->site_bodyid[site_i];
            int shin_i = m->body_parentid[body_i]; 

            for (int j = i + 1; j < num_wheels; ++j) {
                if (mask_state_(j) < 0.5) continue; 
                
                int site_j = wheel_site_ids[j];
                int body_j = m->site_bodyid[site_j];
                int shin_j = m->body_parentid[body_j];

                if (shin_i == shin_j) {
                    
                    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp_i(3, nv);
                    mj_jacSite(m, d, jacp_i.data(), nullptr, site_i);
                    double v_z_i = jacp_i.row(2).dot(qvel);
                    
                    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor> jacp_j(3, nv);
                    mj_jacSite(m, d, jacp_j.data(), nullptr, site_j);
                    double v_z_j = jacp_j.row(2).dot(qvel);
                    
                    // VELOCITY TIE-BREAKER: Momentum decides which wheel claims the ground.
                    if (std::abs(v_z_i - v_z_j) > 0.05) { 
                        if (v_z_i < v_z_j) {
                            mask_state_(j) = 0.0; 
                        } else {
                            mask_state_(i) = 0.0;
                        }
                    } 
                    // POSITION TIE-BREAKER: Gravity-aligned depth check.
                    else {
                        if (d->site_xpos[3 * site_i + 2] < d->site_xpos[3 * site_j + 2]) {
                            mask_state_(j) = 0.0; 
                        } else {
                            mask_state_(i) = 0.0;
                        }
                    }
                }
            }
        }
        
        if (print_debug) std::cout << "--------------------------------------------------\n";
        
        return mask_state_;
    }

    const Eigen::VectorXd& get_estimated_forces() const {
        return estimated_forces_;
    }
};



//  Function to estimate hip height kinematically
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

// Function to check if input value is in input vector
template <typename T>
bool contains(const std::vector<T>& vec, const T& value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
}

// Function to list all site-ids on same body as geom
std::vector<int> getSiteIdsOnSameBodyAsGeom(const mjModel* m, int geom_id) {
    std::vector<int> associated_site_ids; 
    if (geom_id < 0 || geom_id >= m->ngeom) return associated_site_ids; 
    
    int geom_body_id = m->geom_bodyid[geom_id];
    for (int i = 0; i < m->nsite; ++i) {
        if (m->site_bodyid[i] == geom_body_id) associated_site_ids.push_back(i); 
    }
    return associated_site_ids; 
}

// Check binarily what values of A are in B
std::vector<int> getBinaryRepresentation_std_find(const std::vector<int>& A, const std::vector<int>& B) {
    std::vector<int> C;
    C.reserve(B.size());
    for (int b_element : B) {
        auto it = std::find(A.begin(), A.end(), b_element);
        C.push_back((it != A.end()) ? 1 : 0);
    }
    return C;
}

//================================================================================= MAIN =====================

int main(int argc, char** argv) {

    // Check to see if bazel repo is constructed correctly
    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::Create(argv[0], BAZEL_CURRENT_REPOSITORY, &error));

    // Gets robot mujoco model and scene
    std::filesystem::path osc_model_path = runfiles->Rlocation("mujoco-models/models/walter_sr/WaLTER_Senior_v2_ulim2.xml");
    std::filesystem::path simulation_model_path = runfiles->Rlocation("mujoco-models/models/walter_sr/scene_walter_sr_v2_ulim2.xml");

    // Import model and create mjmodel and mjdata objects
    char mj_error[1000];
    mjModel* mj_model = mj_loadXML(simulation_model_path.c_str(), nullptr, mj_error, 1000);
    if (!mj_model) { 
        printf("Failed to load MuJoCo model: %s\n", mj_error); 
        return 1; 
    }
    mjData* mj_data = mj_makeData(mj_model);

    // Select the correct keyframe/starting position of the robot (originally 8)
    mj_resetDataKeyframe(mj_model, mj_data, 9);

    // Runs forward dynamics and updates the variables after keyframe update
    mj_forward(mj_model, mj_data);

    // Initializes the simulation scene
    mjvCamera cam; mjvPerturb pert; mjvOption opt; mjvScene scn; mjrContext con;
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }

    // Sets sim graphics window settings
    int win_width = 1920; 
    int win_height = 1080;
    GLFWwindow* window = glfwCreateWindow(win_width, win_height, "Sim", NULL, NULL); 
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); 
    mjv_defaultCamera(&cam);
    mjv_defaultPerturb(&pert);
    mjv_defaultOption(&opt);

    // Visualizes contact points
    opt.flags[mjVIS_CONTACTPOINT] = 1; 

    // Additional sim window settings
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

    // Create OSC object
    OperationalSpaceController controller(osc_model_path);

    // Set video record option and create object to store
    bool vid_record_flag = true;
    FILE* ffmpeg_pipe = nullptr;
    unsigned char* rgb_buffer = nullptr;
    if (vid_record_flag){
        const int fps = 100; 
        std::string ffmpeg_cmd = "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size " + 
                                    std::to_string(win_width) + "x" + std::to_string(win_height) + 
                                    " -framerate " + std::to_string(fps) + 
                                    " -i - -vf vflip -c:v h264_nvenc -preset hq -b:v 10M -pix_fmt yuv420p /home/vivek/osc_contact.mp4";    
        ffmpeg_pipe = popen(ffmpeg_cmd.c_str(), "w");
        if (!ffmpeg_pipe) {
            std::cerr << "Failed to open FFmpeg pipe." << std::endl;
            return 1;
        }
        rgb_buffer = new unsigned char[win_width * win_height * 3];    
    }

    // Grab initial state of the robot
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

    // Initialize task space targets object
    TaskspaceTargets taskspace_targets = Matrix<model::site_ids_size, 6>::Zero();

    // initialize Abseil status object for controller solution verification - Checks if there were any errors
    absl::Status result;
    result.Update(controller.initialize(initial_state));
    result.Update(controller.initialize_optimization());
    ABSL_CHECK(result.ok()) << result.message();

    // Send taskspace targets to controller
    controller.update_taskspace_targets(taskspace_targets);

    // Initialize controller thread
    result.Update(controller.initialize_thread());
    ABSL_CHECK(result.ok()) << result.message();

    // Start sim timer
    double visualization_timer = mj_data->time;
    double visualization_start_time = visualization_timer;
    double visualization_interval = 0.01; 
    auto current_time = mj_data->time;
    double last_time = current_time;

    // Set total sim time
    double simulation_time = 30.0; // originally 15       


    // Functions to get indexes so we don't use hardcoded numbers
    
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

    // Mapping and printing wheel geoms and sites
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
    // PROPER INITIALIZATION BEFORE LOOP - Specifically for task tracking for the OSC
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

    // Initializing logger for data logging
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

    // For soft switching contact - not used here
    double soft_switch_max_force = 770.0;
    double soft_switch_ramp_time = 0.5; 

    // For timing based contact switching
    Eigen::Vector<double, model::contact_site_ids_size> contact_start_times;
    contact_start_times.setConstant(-100.0); 
    Eigen::Vector<double, model::contact_site_ids_size> prev_contact_mask;
    prev_contact_mask.setZero();

    // Lengths for kinematic hip height calculation
    double L_THIGH = 0.1016; 
    double L_SHIN = 0.08255; 
    double L_WHEEL = 0.0635;

    // Initializing variable
    Vector<model::nu_size> torque_command = Vector<model::nu_size>::Zero();

    // Initial torso pos
    double initial_torso_pos_x = mj_data->qpos[0];



    TelemetryVisualizer visualizer(1000);

    // Initialize your generic plots
    size_t plot_a_id = visualizer.addPlot("Task 1: Torso velocity (x)", "Target", "Actual", -1.0f, 1.0f);
    size_t plot_b_id = visualizer.addPlot("Task 2: Hip height (z)(Torso Right)", "Target", "Actual", 0.1f, 0.2f);    


    ProprioceptiveContactEstimator contact_estimator(model::contact_site_ids_size, 14.0, 0.1); // 14/4


    // =========================================================================================
    // SIMULATION LOOP
    // =========================================================================================
    while(current_time < simulation_time) {

        

        Eigen::VectorXd tau_measured = Eigen::Map<const Eigen::VectorXd>(mj_data->qfrc_actuator, mj_model->nv);
        // Update the estimator. (Pass true for the last argument when you move to physical hardware)
        Eigen::VectorXd proprioceptive_mask = contact_estimator.update(mj_model, mj_data, wheel_site_ids_ref, tau_measured);




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
        // state.contact_mask = proprioceptive_mask;
        
        controller.update_state(state);

        // -------------------------------------------------------------------------------------
        // COMPUTE CONTROL TARGETS 
        // -------------------------------------------------------------------------------------
        TaskspaceTargets taskspace_targets = TaskspaceTargets::Zero();

        double dt = current_time - last_time;
        if (dt == 0) dt = 0.0001; 



        // -------------------------------------------------------------------------------------
        // Torso forward velocity - Task space target computation
        double torso_pos_x = mj_data->qpos[0];        
        double torso_vel_x = mj_data->qvel[0];
        
        double torso_vel_x_target = 0.25;

        double torso_pos_x_target = initial_torso_pos_x + torso_vel_x_target * current_time;

        double torso_pos_x_kp = 1000.0; 
        double torso_pos_x_kv = 100.0;

        double torso_x_control = torso_pos_x_kp * (torso_pos_x_target - torso_pos_x) + torso_pos_x_kv * (torso_vel_x_target - torso_vel_x);
        
        // taskspace_targets.row(0) = Eigen::Vector<double, 6> {torso_position_control, 0, 0, 0, 0, 0};        
        
        // NOTES: 
        // 0.05,100,10 worked for one tumble         
        // 0.5,1000,100 - springs slowly into tumbling but goes crazy after 15 seconds


        Eigen::Quaterniond body_quat(qpos(3), qpos(4), qpos(5), qpos(6));

        // --- NEW: Torso Pitch (Ry) Constraint ---
        // 1. Extract Pitch from body_quat (w, x, y, z), Pitch  = arcsin(2*(wy - zx))
        double sinp = 2.0 * (body_quat.w() * body_quat.y() - body_quat.z() * body_quat.x());
        double torso_pitch = std::asin(std::clamp(sinp, -1.0, 1.0));
        double torso_pitch_vel = state.angular_body_velocity(1); 

        // 2. Critically Damped Pitch Gains
        double pitch_kp = 400.0;
        double pitch_kv = 40.0; // 2 * sqrt(400)
        double pitch_target = 0.0; // Force it to stay level
        double torso_pitch_control = pitch_kp * (pitch_target - torso_pitch) - pitch_kv * torso_pitch_vel;






        // --- NEW: Torso Roll (Rx) ---
        // Extract Roll from body_quat (w, x, y, z), Roll  = atan2(2*(wx + yz),1-2*(x^2 + y^2))
        double sinr_cosp = 2.0 * (body_quat.w() * body_quat.x() + body_quat.y() * body_quat.z());
        double cosr_cosp = 1.0 - 2.0 * (body_quat.x() * body_quat.x() + body_quat.y() * body_quat.y());
        double torso_roll = std::atan2(sinr_cosp, cosr_cosp);
        double torso_roll_vel = state.angular_body_velocity(0); 

        // Critically Damped Roll Gains
        double roll_kp = 1000.0;
        double roll_kv = 100.0; 
        double torso_roll_control = roll_kp * (0.0 - torso_roll) - roll_kv * torso_roll_vel;






        // 3. Inject both X-translation and Y-rotation into the Torso Task (Row 0)
        // Format assumes: [x, y, z, rx, ry, rz]
        taskspace_targets.row(0) = Eigen::Vector<double, 6> {torso_x_control, 0, 0, torso_roll_control, torso_pitch_control, 0};        
        // taskspace_targets.row(0) = Eigen::Vector<double, 6> {torso_x_control, 0, 0, torso_roll_control, 0, 0};        






        // -------------------------------------------------------------------------------------
        // Shin velocity and position - Task space target computation
        double tl_shin_angle = mj_data->qpos[tl_knee_idx];        
        double tr_shin_angle = mj_data->qpos[tr_knee_idx];        
        double hl_shin_angle = mj_data->qpos[hl_knee_idx];        
        double hr_shin_angle = mj_data->qpos[hr_knee_idx];        

        // double dt = current_time - last_time;
        // if (dt == 0) dt = 0.0001; 

        double tl_angular_velocity = (tl_shin_angle - last_tl_angular_position) / dt;
        double tr_angular_velocity = (tr_shin_angle - last_tr_angular_position) / dt;
        double hl_angular_velocity = (hl_shin_angle - last_hl_angular_position) / dt;
        double hr_angular_velocity = (hr_shin_angle - last_hr_angular_position) / dt;

        double tl_angular_position_target = initial_tl_angular_position + shin_rot_vel * current_time;
        double tr_angular_position_target = initial_tr_angular_position + shin_rot_vel * current_time;
        double hl_angular_position_target = initial_hl_angular_position + shin_rot_vel * current_time;
        double hr_angular_position_target = initial_hr_angular_position + shin_rot_vel * current_time;

        double shin_kp = 800.0; 
        double shin_kv = 800.0;

        double tl_angular_control = shin_kp * (tl_angular_position_target - tl_shin_angle) + shin_kv * (shin_rot_vel - tl_angular_velocity);
        double tr_angular_control = shin_kp * (tr_angular_position_target - tr_shin_angle) + shin_kv * (shin_rot_vel - tr_angular_velocity);
        double hl_angular_control = shin_kp * (hl_angular_position_target - hl_shin_angle) + shin_kv * (shin_rot_vel - hl_angular_velocity);
        double hr_angular_control = shin_kp * (hr_angular_position_target - hr_shin_angle) + shin_kv * (shin_rot_vel - hr_angular_velocity);
        
        last_tl_angular_position = tl_shin_angle;
        last_tr_angular_position = tr_shin_angle;
        last_hl_angular_position = hl_shin_angle;
        last_hr_angular_position = hr_shin_angle;

        // taskspace_targets.row(1) = Eigen::Vector<double, 6> {0, 0, 0, 0, tl_angular_control, 0};        
        // taskspace_targets.row(2) = Eigen::Vector<double, 6> {0, 0, 0, 0, tr_angular_control, 0};        
        // taskspace_targets.row(3) = Eigen::Vector<double, 6> {0, 0, 0, 0, hl_angular_control, 0};        
        // taskspace_targets.row(4) = Eigen::Vector<double, 6> {0, 0, 0, 0, hr_angular_control, 0};        

        taskspace_targets.row(1) = Eigen::Vector<double, 6> {0, 0, 0, 0, 0, 0};        
        taskspace_targets.row(2) = Eigen::Vector<double, 6> {0, 0, 0, 0, 0, 0};        
        taskspace_targets.row(3) = Eigen::Vector<double, 6> {0, 0, 0, 0, 0, 0};        
        taskspace_targets.row(4) = Eigen::Vector<double, 6> {0, 0, 0, 0, 0, 0};        

        // -------------------------------------------------------------------------------------


        // -------------------------------------------------------------------------------------
        // Hip height - task space target computation (originally 800,80)
        double thigh_lin_vel = 0.0;
        // double thigh_lin_kp = 100.0 * 8.0; 
        // double thigh_lin_kv = 20.0 * 4.0;
        
        double thigh_lin_kp = 800.0; 
        double thigh_lin_kv = 80.0;
        

        // Eigen::Quaterniond body_quat(qpos(3), qpos(4), qpos(5), qpos(6));
        
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
        // -------------------------------------------------------------------------------------

        // Get robot position so camera can track it
        Vector<3> body_position = qpos(Eigen::seqN(0, 3));
        cam.lookat[0] = body_position(0);

        // -------------------------------------------------------------------------------------
        // OSQP SOLVER
        // -------------------------------------------------------------------------------------
        //console MUTE std::cout.setstate(std::ios_base::failbit); console MUTE
        controller.update_taskspace_targets(taskspace_targets);


        // Vector<model::nu_size> torque_command = controller.get_torque_command();
        torque_command = controller.get_torque_command();
        // console UNMUTE std::cout.clear(); console UNMUTE

        // Print solver status - especially if the solver failed to find a solution
        int solver_status_int = controller.get_solver_status(); 
        osqp::OsqpExitCode solver_status = static_cast<osqp::OsqpExitCode>(solver_status_int);
        if (solver_status != osqp::OsqpExitCode::kOptimal) {
             std::cout << "Solver failed with status: " << (int)solver_status << " at time " << current_time << ". Exiting loop." << std::endl;
             break; 
        }

        // Extract solution from controller
        Vector<optimization::design_vector_size> solution = controller.get_solution();

        
        // =====================================================================
        // CRITICAL FIX: THE TWO LINES THAT SYNCHRONIZE TIME
        // =====================================================================
        mj_data->ctrl = torque_command.data(); // Apply the new torque instantly
        mj_forward(mj_model, mj_data);         // Force MuJoCo to recalculate physics for THIS microsecond
        // =====================================================================

        // =====================================================================
        // MANIPULATOR EQUATION DIAGNOSTIC LOGGER
        // =====================================================================
        controller.log_manipulator_equation(mj_model, mj_data, diag_log, hr_shin_dof, fr_knee_actuator_idx);
        
        // =====================================================================
        // PROOF 1: PLANNED VS ACTUAL JOINT ACCELERATION
        // =====================================================================
        // Extract the solver's target joint accelerations (dv) from the optimal design vector.
        auto dv_planned = solution(Eigen::seqN(0, optimization::dv_size));
        
        // Log 1. time, 2. planned acc at head right knee joint, 3. sim sensed acc at head right knee joint
        logger.log("time", current_time);
        logger.log("hr_shin_accel_plan", dv_planned(hr_shin_dof));
        logger.log("hr_shin_accel_actual", mj_data->qacc[hr_shin_dof]);

        // ===================================================================================== 
        // ISOLATED CONTACT FORCE PROOF (CORRECTED API)
        // ===================================================================================== 

        // -------------------------------------------------------------------------------------
        // 1. Compute Planned Contact Torques
        // Maps the OSQP solver's planned 3D ground reaction forces (GRFs) into joint space.
        // Uses the relationship: tau = J^T * F
        // -------------------------------------------------------------------------------------
        Vector<model::nv_size> tau_contact_plan = Vector<model::nv_size>::Zero();
        for (int w = 0; w < 8; ++w) {
            std::string site_name = std::string(model::contact_site_list[w]);
            
            // Extract the planned 3D linear force vector for this specific contact site
            double fx_plan = solution[optimization::z_idx + (3 * w) + 0];
            double fy_plan = solution[optimization::z_idx + (3 * w) + 1];
            double fz_plan = solution[optimization::z_idx + (3 * w) + 2];
            Eigen::Vector3d f_plan_3d(fx_plan, fy_plan, fz_plan);

            int site_id = mj_name2id(mj_model, mjOBJ_SITE, site_name.c_str()); 
            
            // Initialize RowMajor Jacobian to prevent memory scrambling from MuJoCo's C-API
            // Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor> jacp = Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor>::Zero();
            Matrix<3, model::nv_size> jacp = Matrix<3, model::nv_size>::Zero();
            // Eigen::Matrix<double, 3, model::nv_size> jacp = Eigen::Matrix<double, 3, model::nv_size>::Zero();
            mj_jacSite(mj_model, mj_data, jacp.data(), nullptr, site_id);

            // Accumulate planned joint torques
            tau_contact_plan += jacp.transpose() * f_plan_3d;
        }            

        // -------------------------------------------------------------------------------------
        // 2. Compute Actual Contact Torques (Simulator Ground Truth)
        // Extracts real collision dynamics from MuJoCo's penalty-based (soft) contact model.
        // -------------------------------------------------------------------------------------
        Vector<model::nv_size> tau_contact_sim = Vector<model::nv_size>::Zero();

        // Iterate strictly over active simulator contacts, bypassing internal joint limits/damping
        for (int i = 0; i < mj_data->ncon; ++i) {
            mjContact* contact = &(mj_data->contact[i]);
            
            int geom_id = contact->geom[0];
            int body_id = mj_model->geom_bodyid[geom_id];
            
            Eigen::Vector3d f_sim_contact;

            // Apply Newton's 3rd Law based on collision geometry hierarchy.
            // MuJoCo standard convention: geom[0] applies force TO geom[1].
            if (body_id == 0) { 
                // Case A: geom[0] is the world body (floor).
                // The force is already applied TO the robot. No inversion is necessary.
                geom_id = contact->geom[1];
                body_id = mj_model->geom_bodyid[geom_id];
                
                double force_6d[6] = {0};
                mj_contactForce(mj_model, mj_data, i, force_6d);
                f_sim_contact = Eigen::Vector3d(force_6d[0], force_6d[1], force_6d[2]); 
            } else {
                // Case B: geom[0] is the robot.
                // The robot applies force TO the floor. We must invert to find the reaction force.
                double force_6d[6] = {0};
                mj_contactForce(mj_model, mj_data, i, force_6d);
                f_sim_contact = Eigen::Vector3d(-force_6d[0], -force_6d[1], -force_6d[2]);
            }

            // Extract the translational Jacobian at the exact spatial point of collision
            Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor> jacp = Eigen::Matrix<double, 3, model::nv_size, Eigen::RowMajor>::Zero();
            // Eigen::Matrix<double, 3, model::nv_size> jacp = Eigen::Matrix<double, 3, model::nv_size>::Zero();            
            mj_jac(mj_model, mj_data, jacp.data(), nullptr, contact->pos, body_id);

            // MuJoCo returns contact forces in a local contact frame. 
            // We must rotate them into the global coordinate frame before multiplying by the global Jacobian.
            Eigen::Matrix3d contact_frame = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(contact->frame);
            Eigen::Vector3d f_sim_global = contact_frame.transpose() * f_sim_contact;

            // Accumulate actual joint torques
            tau_contact_sim += jacp.transpose() * f_sim_global;
        }

        // -------------------------------------------------------------------------------------
        // 3. Contact Torque Discrepancy
        // Quantifies the difference between rigid-body assumptions (QP) and soft-contact physics.
        // -------------------------------------------------------------------------------------
        Vector<model::nv_size> delta_tau_contact = tau_contact_sim - tau_contact_plan;

        // -------------------------------------------------------------------------------------
        // 4. Accelerative Impact Prediction
        // Solves the forward dynamics (M * ddq = tau) to isolate how much joint acceleration
        // error is caused *exclusively* by the contact force discrepancy.
        // NOTE: M must be RowMajor to interface safely with mj_fullM.
        // -------------------------------------------------------------------------------------
        Eigen::Matrix<double, model::nv_size, model::nv_size, Eigen::RowMajor> M = Eigen::Matrix<double, model::nv_size, model::nv_size, Eigen::RowMajor>::Zero();
        mj_fullM(mj_model, M.data(), mj_data->qM);
        Vector<model::nv_size> predicted_delta_acc = M.llt().solve(delta_tau_contact);

        // -------------------------------------------------------------------------------------
        // 5. Total Acceleration Error
        // Computes the absolute divergence between the controller's plan and physical reality.
        // -------------------------------------------------------------------------------------
        Vector<model::nv_size> real_ddq = Eigen::Map<Vector<model::nv_size>>(mj_data->qacc);
        Vector<model::nv_size> real_delta_acc = real_ddq - dv_planned;


        // 6. Log the Verdict
        logger.log("delta_acc_real", real_delta_acc(hr_shin_dof));
        logger.log("delta_acc_predicted_from_contacts", predicted_delta_acc(hr_shin_dof));
        logger.log("torque_planned", torque_command(fr_knee_actuator_idx));

        // Use the mapped DOF address to safely extract actuator torque
        int fr_knee_dof_addr = mj_name2id(mj_model, mjOBJ_JOINT, "head_right_thigh_shin_joint");
        logger.log("torque_actual", mj_data->qfrc_actuator[mj_model->jnt_dofadr[fr_knee_dof_addr]]);


        // =====================================================================
        // THE 1-TO-1 LINEUP: ISOLATING THE GHOST FORCES
        // Decomposes the full manipulator equation to pinpoint the source of tracking errors.
        // Equation: M*ddq + c = tau_act + J^T*F_ext
        // =====================================================================

        // 1. Actuator Delta: Identifies torque clipping, gear friction, or command latency.
        double tau_act_sim = mj_data->qfrc_actuator[hr_shin_dof];
        double tau_act_plan = torque_command(fr_knee_actuator_idx); 
        double delta_actuator = tau_act_sim - tau_act_plan;

        // 2. Contact Delta: Identifies leverage mismatch or penetration penalty discrepancies.
        double tau_cont_sim_val = tau_contact_sim(hr_shin_dof); 
        double tau_cont_plan_val = tau_contact_plan(hr_shin_dof); 
        double delta_contact = tau_cont_sim_val - tau_cont_plan_val;

        // 3. Bias Delta: Identifies unmodeled Coriolis effects, joint damping, or gravity mismatch.
        // NOTE: M_sim must be RowMajor.
        Eigen::Matrix<double, model::nv_size, model::nv_size, Eigen::RowMajor> M_sim = Eigen::Matrix<double, model::nv_size, model::nv_size, Eigen::RowMajor>::Zero();
        mj_fullM(mj_model, M_sim.data(), mj_data->qM);
        
        // Calculate expected internal dynamics based on the planned acceleration
        double inertial_plan = (M_sim * dv_planned)(hr_shin_dof);
        
        // Infer what the controller believed the bias forces (c) were based on its torque output
        double c_ctrl_inferred = tau_act_plan + tau_cont_plan_val - inertial_plan; 

        // Extract the simulator's true calculation for bias forces
        double c_sim = mj_data->qfrc_bias[hr_shin_dof]; 
        double delta_bias = c_sim - c_ctrl_inferred;

        // Log the decomposed deltas to trace the exact origin of the mathematical ghost
        logger.log("Ghost_Delta_Actuator", delta_actuator);
        logger.log("Ghost_Delta_Contact", delta_contact);
        logger.log("Ghost_Delta_Bias", delta_bias);

        // Log linear body velocity
        logger.log("body_x_vel", state.linear_body_velocity[0]);

        // Log linear body velocity
        logger.log("body_rw", state.body_rotation[0]);
        logger.log("body_rx", state.body_rotation[1]);
        logger.log("body_ry", state.body_rotation[2]);
        logger.log("body_rz", state.body_rotation[3]);

        logger.endStep();

        // -------------------------------------------------------------------------------------
        // VISUALIZATION & RECORDING
        // -------------------------------------------------------------------------------------


        if(visualization_timer > visualization_interval) {

            // Feed data using the IDs
            visualizer.updateData(plot_a_id, current_time, torso_vel_x_target, torso_vel_x);
            visualizer.updateData(plot_b_id, current_time, initial_trh_linear_position(2), h_tr_kinematic);

            visualization_start_time = mj_data->time;

            // Call sim window frame object
            mjrRect viewport_full = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport_full.width, &viewport_full.height);

            // 1. Shrink the Sim Viewport to make room for the Visualizer
            mjrRect viewport_sim = viewport_full;
            viewport_sim.width -= (viewport_full.width / 3);

            // 2. Render 3D Scene to the shrunken viewport
            mjv_updateScene(mj_model, mj_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport_sim, &scn, &con); // <--- Use viewport_sim, not viewport_full

            visualizer.render(window, &con);

            // Add time at top left of sim graphics window
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << "Time: " << mj_data->time << " s";
            mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport_full, ss.str().c_str(), 0, &con);




            // // Add contact mask of each wheel at bottom left
            // std::stringstream ss2;
            // ss2 << "      [PHYS] [MASK]\n--------------------\n";
            // auto row = [&](std::string label, int idx) {
            //     ss2 << label << ":   " << raw_physics[idx] << "      " << (contact_check2[idx] > 0.5) << "\n";
            // };
            // row("TL_F", 0); row("TL_R", 1); row("TR_F", 2); row("TR_R", 3);
            // ss2 << "--------------------\n";
            // row("HL_F", 4); row("HL_R", 5); row("HR_F", 6); row("HR_R", 7);
            // mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, viewport_full, ss2.str().c_str(), 0, &con);




            // Add contact mask of each wheel at bottom left
            std::stringstream ss2;
            ss2 << std::fixed << std::setprecision(1); // Lock decimal places
            ss2 << "      [PHYS] [MASK] [PROP] [FORCE (N)]\n----------------------------------------\n";
            
            // Retrieve the raw forces from the estimator
            Eigen::VectorXd current_est_forces = contact_estimator.get_estimated_forces();
            
            auto row = [&](std::string label, int idx) {
                ss2 << label << ":   " << raw_physics[idx] 
                    << "      " << (contact_check2[idx] > 0.5) 
                    << "      " << (proprioceptive_mask[idx] > 0.5) 
                    << "      " << std::setw(6) << current_est_forces[idx] << "\n";
            };
            
            row("TL_F", 0); row("TL_R", 1); row("TR_F", 2); row("TR_R", 3);
            ss2 << "----------------------------------------\n";
            row("HL_F", 4); row("HL_R", 5); row("HR_F", 6); row("HR_R", 7);
            mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, viewport_full, ss2.str().c_str(), 0, &con);




            // Record video to file if record flag is true
            if (vid_record_flag && ffmpeg_pipe != nullptr) {
                mjrRect record_viewport = {0, 0, win_width, win_height};
                mjr_readPixels(rgb_buffer, nullptr, record_viewport, &con);
                fwrite(rgb_buffer, 1, win_width * win_height * 3, ffmpeg_pipe);
            }

            // Graphics window update
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    // Close graphics window after sim 
    glfwTerminate();
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Write video file
    if (vid_record_flag) {
        if (ffmpeg_pipe) pclose(ffmpeg_pipe);
        delete[] rgb_buffer;
        std::cout << "Video saved." << std::endl;        
    }

    // Delete mj model and data objects
    // result.Update(controller.stop_thread());
    mj_deleteData(mj_data);
    mj_deleteModel(mj_model);
    ABSL_CHECK(result.ok()) << result.message();

    // Write log data to file
    logger.save("osc_test_data.csv");

    // End
    return 0;
}