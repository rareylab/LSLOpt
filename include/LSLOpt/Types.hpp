#pragma once

#include <Eigen/Dense>


namespace LSLOpt {

/**
 * @brief Vector type used.
 * @tparam Scalar The scalar type of the coefficients.
 *
 * This is a column vector (an `Eigen::Matrix`).
 */
template<typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

/**
 * @brief Matrix type used.
 * @tparam Scalar The scalar type of the coefficients.
 *
 * This is a matrix (an `Eigen::Matrix`).
 */
template<typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * @brief Diagonal matrix type used.
 * @tparam Scalar The scalar type of the coefficients.
 *
 * This is a diagonal matrix (an `Eigen::DiagonalMatrix`).
 */
template<typename Scalar>
using DiagonalMatrix = Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

}
