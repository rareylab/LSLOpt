/**
 * @brief Gradient descent optimizations.
 *
 * This file contains functions for the optimization
 * using the gradient descent algorithm with and without the
 * limitation of step lengths.
 *
 * @warning
 * These functions are used for comparison only, the various
 * BFGS implementations are usually far superior.
 */

#pragma once

#include "Implementation/GDStorage.hpp"
#include "Implementation/ProblemTraits.hpp"
#include "Implementation/QuasiNewton.hpp"
#include "Implementation/UnconstrainedAlgorithm.hpp"
#include "OptimizationParameters.hpp"
#include "OptimizationResult.hpp"
#include "OutputUtils.hpp"
#include "Types.hpp"


namespace LSLOpt {

/**
 * @brief Perform a gradient descent optimization with identification of the maximum step length.
 *
 * @param problem Objective function to minimize.
 * @param x0 Starting point for optimization.
 * @param params BFGS parameters.
 * @param output_function Function for output of status and error messages.
 *
 * This performs a gradient descent minimization of the given objective function.
 *
 * The problem must provide the following functions, calculating the value at
 * a given point, the gradient at a given point, the initially allowed
 * step length at a given point and a given direction as well as a function
 * checking if a change from one point to the other is acceptable:
 *
 * `Scalar value(const Vector<Scalar>& x);`
 *
 * `Vector<Scalar> gradient(const Vector<Scalar>& x);`
 *
 * `double initial_step_length(const Eigen::VectorXd& x, const Eigen::VectorXd& p);`
 *
 * `bool is_change_acceptable(const Eigen::VectorXd& x, const Eigen::VectorXd& xp);`
 *
 */
template<typename Problem, typename Scalar, typename OutputFunction = NoOutput>
OptimizationResult<Scalar> lsl_gd(
    Problem& problem,
    const Vector<Scalar>& x0,
    const OptimizationParameters<Scalar>& params,
    OutputFunction& output_function = OutputFunction())
{
  return Implementation::quasi_newton<
      Problem,
      Scalar,
      Implementation::GDStorage,
      Implementation::UnconstrainedAlgorithm,
      OutputFunction,
      Implementation::MaxStepLengthIdentifierTrait>(
          problem, x0, params, output_function);
}

/**
 * @brief Perform a gradient descent optimization without identification of the maximum step length.
 *
 * @param problem Objective function to minimize.
 * @param x0 Starting point for optimization.
 * @param params BFGS parameters.
 * @param output_function Function for output of status and error messages.
 *
 * This performs a gradient descent minimization of the given objective function.
 *
 * The problem must provide the following functions, calculating the value at
 * a given point, the gradient at a given point as well as the maximum allowed
 * step length at a given point and a given direction:
 *
 * `Scalar value(const Vector<Scalar>& x);`
 *
 * `Vector<Scalar> gradient(const Vector<Scalar>& x);`
 *
 * @note
 * This is a convenience function to perform a standard
 * gradient descent without maximum step length identification.
 */
template<typename Problem, typename Scalar, typename OutputFunction = NoOutput>
OptimizationResult<Scalar> gd(
    Problem& problem,
    const Vector<Scalar>& x0,
    const OptimizationParameters<Scalar>& params,
    OutputFunction& output_function = OutputFunction())
{
  return Implementation::quasi_newton<
      Problem,
      Scalar,
      Implementation::GDStorage,
      Implementation::UnconstrainedAlgorithm,
      OutputFunction,
      Implementation::NoMaxStepLengthIdentifierTrait>(
          problem, x0, params, output_function);
}

}

#include "Implementation/UndefOutputMacros.hpp"
