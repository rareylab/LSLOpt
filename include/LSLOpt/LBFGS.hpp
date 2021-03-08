/**
 * @brief L-BFGS optimizations.
 *
 * This file contains functions for the optimization
 * using the L-BFGS algorithm with and without the
 * limitation of step lengths.
 */

#pragma once

#include "Implementation/LBFGSStorage.hpp"
#include "Implementation/ProblemTraits.hpp"
#include "Implementation/QuasiNewton.hpp"
#include "Implementation/UnconstrainedAlgorithm.hpp"
#include "OptimizationParameters.hpp"
#include "OptimizationResult.hpp"
#include "OutputUtils.hpp"
#include "Types.hpp"


namespace LSLOpt {

/**
 * @brief Perform an L-BFGS optimization with identification of the maximum step length
 *        (LSL-L-BFGS).
 *
 * @param problem Objective function to minimize.
 * @param x0 Starting point for optimization.
 * @param params BFGS parameters.
 * @param output_function Function for output of status and error messages.
 *
 * This performs a L-BFGS minimization of the given objective function.
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
 * @note
 * This is a L-BFGS implementation and therefore suited
 * for large optimization problems.
 *
 */
template<typename Problem, typename Scalar, typename OutputFunction = NoOutput>
OptimizationResult<Scalar> lsl_lbfgs(
    Problem& problem,
    const Vector<Scalar>& x0,
    const OptimizationParameters<Scalar>& params,
    OutputFunction& output_function = OutputFunction())
{
  return Implementation::quasi_newton<
      Problem, // the actual optimization problem
      Scalar,
      Implementation::LBFGSStorage, // storage type (e.g. limited memory or full)
      Implementation::UnconstrainedAlgorithm, // the algorithm for constraint handling
      OutputFunction, // the output function
      Implementation::MaxStepLengthIdentifierTrait>(
      problem, x0,  params, output_function);
}


/**
 * @brief Perform an L-BFGS optimization without identification of the maximum step length
 *        (L-BFGS).
 *
 * @param problem Objective function to minimize.
 * @param x0 Starting point for optimization.
 * @param params BFGS parameters.
 * @param output_function Function for output of status and error messages.
 *
 * This performs a L-BFGS minimization of the given objective function.
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
 * @note
 * This is a L-BFGS implementation and therefore suited
 * for large optimization problems.
 *
 * @note
 * This is a convenience function to perform a standard
 * L-BFGS without maximum step length identification.
 */
template<typename Problem, typename Scalar, typename OutputFunction = NoOutput>
OptimizationResult<Scalar> lbfgs(
    Problem& problem,
    const Vector<Scalar>& x0,
    const OptimizationParameters<Scalar>& params,
    OutputFunction& output_function = OutputFunction())
{
  return Implementation::quasi_newton<
      Problem,
      Scalar,
      Implementation::LBFGSStorage,
      Implementation::UnconstrainedAlgorithm,
      OutputFunction,
      Implementation::NoMaxStepLengthIdentifierTrait>(
      problem, x0, params, output_function);
}

}

#include "Implementation/UndefOutputMacros.hpp"
