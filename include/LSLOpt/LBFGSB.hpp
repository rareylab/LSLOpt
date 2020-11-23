/**
 * @brief L-BFGS-B optimizations.
 *
 * This file contains functions for the optimization
 * using the L-BFGS-B algorithm with and without the
 * limitation of step lengths.
 */

#pragma once

#include "Implementation/BoxConstrainedAlgorithm.hpp"
#include "Implementation/LBFGSStorage.hpp"
#include "Implementation/ProblemTraits.hpp"
#include "Implementation/QuasiNewton.hpp"
#include "OptimizationParameters.hpp"
#include "OptimizationResult.hpp"
#include "OutputUtils.hpp"
#include "Types.hpp"


namespace LSLOpt {

/**
 * @brief Perform an L-BFGS optimization with box constraints and
 *        identification of the maximum step length (LSL-L-BFGS-B).
 *
 * @param problem Objective function to minimize.
 * @param x0 Starting point for optimization.
 * @param params BFGS parameters.
 * @param output_function Function for output of status and error messages.
 *
 * This performs a L-BFGS-B minimization of the given objective function.
 * Internally, it uses an backtracking line search and uses a damped
 * update strategy to ensure the positive definiteness of the Hessian approximation
 * (and it's inverse).
 *
 * The problem must provide the following functions, calculating the value at
 * a given point, the gradient at a given point, the initially allowed
 * step length at a given point and a given direction as well as a function
 * checking if a change from one point to the other is acceptable
 * and the upper and lower bounds on the parameters:
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
 * `const Vector<Scalar>& lower_bounds();`
 *
 * `const Vector<Scalar>& upper_bounds();`
 *
 * @note
 * This is a L-BFGS-B implementation and therefore suited
 * for large optimization problems.
 *
 * Byrd, R.H., Lu, P., Nocedal, J., Zhu, C.
 * A Limited Memory Algorithm for Bound Constrained Optimization.
 * 1995. SIAM Journal of Scientific and Statistical Computing. 16(5). 1190-1208.
 *
 * Morales, J.L, Nocedal, J.
 * L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for
 * large scale bound constrained optimization.
 * 2011. ACM Transactions on Mathematical Software. 38(1). 7:1-7:4.
 */
template<typename Problem, typename Scalar, typename OutputFunction = NoOutput>
OptimizationResult<Scalar> lsl_lbfgs_b(
    Problem& problem,
    const Vector<Scalar>& x0,
    const OptimizationParameters<Scalar>& params,
    OutputFunction& output_function = OutputFunction())
{
  return Implementation::quasi_newton<
      Problem,
      Scalar,
      Implementation::LBFGSStorage,
      Implementation::BoxConstrainedAlgorithm,
      OutputFunction,
      Implementation::MaxStepLengthIdentifierTrait>(
          problem, x0, params, output_function);
}

/**
 * @brief Perform an L-BFGS optimization with box constraints and
 *        without identification of the maximum step length (L-BFGS-B).
 *
 * @param problem Objective function to minimize.
 * @param x0 Starting point for optimization.
 * @param params BFGS parameters.
 * @param output_function Function for output of status and error messages.
 *
 * This performs a L-BFGS-B minimization of the given objective function.
 * Internally, it uses an backtracking line search and uses a damped
 * update strategy to ensure the positive definiteness of the Hessian approximation
 * (and it's inverse).
 *
 * The problem must provide the following functions, calculating the value at
 * a given point, the gradient at a given point as well as the maximum allowed
 * step length at a given point and a given direction and the upper and lower bounds
 * on the parameters:
 *
 * `Scalar value(const Vector<Scalar>& x);`
 *
 * `Vector<Scalar> gradient(const Vector<Scalar>& x);`
 *
 * `const Vector<Scalar>& lower_bounds();`
 *
 * `const Vector<Scalar>& upper_bounds();`
 *
 * @note
 * This is a L-BFGS-B implementation and therefore suited
 * for large optimization problems.
 *
 * @note
 * This is a convenience function to perform a standard
 * L-BFGS-B without maximum step length identification.
 *
 * Byrd, R.H., Lu, P., Nocedal, J., Zhu, C.
 * A Limited Memory Algorithm for Bound Constrained Optimization.
 * 1995. SIAM Journal of Scientific and Statistical Computing. 16(5). 1190-1208.
 *
 * Morales, J.L, Nocedal, J.
 * L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for
 * large scale bound constrained optimization.
 * 2011. ACM Transactions on Mathematical Software. 38(1). 7:1-7:4.
 */
template<typename Problem, typename Scalar, typename OutputFunction = NoOutput>
OptimizationResult<Scalar> lbfgs_b(
    Problem& problem,
    const Vector<Scalar>& x0,
    const OptimizationParameters<Scalar>& params,
    OutputFunction& output_function = OutputFunction())
{
  return Implementation::quasi_newton<
      Problem,
      Scalar,
      Implementation::LBFGSStorage,
      Implementation::BoxConstrainedAlgorithm,
      OutputFunction,
      Implementation::NoMaxStepLengthIdentifierTrait>(
          problem, x0, params, output_function);
}

}

#include "Implementation/UndefOutputMacros.hpp"
