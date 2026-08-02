#pragma once

#include <array>
#include <vector>
#include <algorithm>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/SparseCore>


namespace matrix_utils {

    constexpr bool RowMajor = true;
    constexpr bool ColumnMajor = false; 

    template<typename T, std::size_t Rows, std::size_t Cols, bool ToRowMajor>
    constexpr std::array<T, Rows * Cols> transformMatrix(const T* arr) {
        std::array<T, Rows * Cols> result;

        if constexpr (ToRowMajor) {
            // Convert from Column Major to Row Major
            for (std::size_t i = 0; i < Rows; ++i) {
                for (std::size_t j = 0; j < Cols; ++j) {
                    result[i * Cols + j] = arr[j * Rows + i];
                }
            }
        } else {
            // Convert from Row Major to Column Major
            for (std::size_t i = 0; i < Rows; ++i) {
                for (std::size_t j = 0; j < Cols; ++j) {
                    result[j * Rows + i] = arr[i * Cols + j];
                }
            }
        }

        return result;
    }

}
