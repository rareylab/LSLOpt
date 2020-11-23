/**
 * @brief BFGS optimizations.
 *
 * This file contains functions for the optimization
 * using the BFGS algorithm with and without the
 * limitation of step lengths.
 *
 * @warning
 * These functions are not suited for large-scale optimization.
 */

#pragma once

#include "Implementation/BFGSStorage.hpp"
#include "Implementation/ProblemTraits.hpp"
#include "Implementation/QuasiNewton.hpp"
#include "Implementation/UnconstrainedAlgorithm.hpp"
#include "OptimizationParameters.hpp"
#include "OptimizationResult.hpp"
#include "OutputUtils.hpp"
#include "Types.hpp"


namespace LSLOpt {

/**
 * @brief Perform a BFGS optimization with identification of the maximum step length
 *        (LSL-BFGS).
 *
 * @param problem Objective function to minimize.
 * @param x0 Starting point for optimization.
 * @param params BFGS parameters.
 * @param output_function Function for output of status and error messages.
 *
 * This performs a BFGS minimization of the given objective function.
 * Internally, it uses an backtracking line search and uses a damped
 * update strategy to ensure the positive definiteness of the Hessian approximation
 * (and it's inverse).
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
 * `Scalar initial_step_length(const Vector<Scalar>& x, const Vector<Scalar>& p);`
 *
 * `Scalar change_acceptable(const Vector<Scalar>& x, const Vector<Scalar>& xp);`
 * (should return a value <= 0.0 if the change is acceptable, > 0.0 otherwise).
 *
 * @warning This is a BFGS implementation (NOT L-BFGS) and therefore not suited
 * for large optimization problems.
 *
 */
template<typename Problem, typename Scalar, typename OutputFunction = NoOutput>
OptimizationResult<Scalar> lsl_bfgs(
    Problem& problem,
    const Vector<Scalar>& x0,
    const OptimizationParameters<Scalar>& params,
    OutputFunction& output_function = OutputFunction())
{
  return Implementation::quasi_newton<
      Problem,
      Scalar,
      Implementation::BFGSStorage,
      Implementation::UnconstrainedAlgorithm,
      OutputFunction,
      Implementation::MaxStepLengthIdentifierTrait>(
          problem, x0, params, output_function);
}

/**
 * @brief Perform a BFGS optimization without identification of the maximum step length
 *        (BFGS).
 *
 * @param problem Objective function to minimize.
 * @param x0 Starting point for optimization.
 * @param params BFGS parameters.
 * @param output_function Function for output of status and error messages.
 *
 * This performs a BFGS minimization of the given objective function.
 * Internally, it uses an backtracking line search and uses a damped
 * update strategy to ensure the positive definiteness of the Hessian approximation
 * (and it's inverse).
 *
 * The problem must provide the following functions, calculating the value at
 * a given point, the gradient at a given point as well as the maximum allowed
 * step length at a given point and a given direction:
 *
 * `Scalar value(const Vector<Scalar>& x);`
 *
 * `Vector<Scalar> gradient(const Vector<Scalar>& x);`
 *
 * @warning This is a BFGS implementation (NOT L-BFGS) and therefore not suited
 * for large optimization problems.
 *
 * @note
 * This is a convenience function to perform a standard
 * BFGS without maximum step length identification.
 */
template<typename Problem, typename Scalar, typename OutputFunction = NoOutput>
OptimizationResult<Scalar> bfgs(
    Problem& problem,
    const Vector<Scalar>& x0,
    const OptimizationParameters<Scalar>& params,
    OutputFunction& output_function = OutputFunction())
{
  return Implementation::quasi_newton<
      Problem,
      Scalar,
      Implementation::BFGSStorage,
      Implementation::UnconstrainedAlgorithm,
      OutputFunction,
      Implementation::NoMaxStepLengthIdentifierTrait>(
          problem, x0, params, output_function);
}

}

#include "Implementation/UndefOutputMacros.hpp"
