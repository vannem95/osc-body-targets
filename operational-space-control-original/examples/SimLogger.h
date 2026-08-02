#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <mujoco/mujoco.h>
#include "Eigen/Dense"

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