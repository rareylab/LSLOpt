#pragma once

#include <limits>

#include "OptimizationResult.hpp"
#include "ScalarTraits.hpp"


namespace LSLOpt {

/// @brief Line search algorithm.
enum class Linesearch {
    /// Armijo backtracking
    Backtracking,
};

/// @brief Selection of initial step length alpha0
enum class Alpha0Policy {
  Constant, /// use max alpha from algorithm (or 1.0)
  ConstantScaling, /// scale old alpha by phi'_{k-1}(0) / phi'_{k}(0)
  Interpolation, /// set alpha0 to 2 * (f_k - f_{k-1}) / phi'_{k}(0)
};


/**
 * @brief Parameters for limited step length BFGS algorithm.
 *
 * @warning
 * You can't instantiate this class directly, but
 * you have to use \ref getOptimizationParameters.
 *
 * @note
 * If the initialization of the scalar constants is not
 * appropriate for your scalar you can specialize the
 * complete parameter template.
 *
 * @note
 * We provide five different termination criteria.
 * The first fulfilled criterion triggers the termination.
 * 1. Maximum number of iterations (default: 10000).
 *    The algorithm terminates after the given number
 *    of iterations of the Quasi-Newton method.
 * 2. Gradient norm (default for `double`: 1e-6).
 *    The algorithm terminates the gradient norm fulfills
 *    ||grad|| <= tolerance * max(1.0, ||x||)
 *    This is the classical termination criterion.
 * 3. Minimum change (default for `double`: 0.0).
 *    The algorithm terminates if the relative change
 *    in the function value is smaller than this value.
 *    Please note that this criterion is also active
 *    even if it is set to 0.0: In this case the algorithm
 *    terminates if the relative change in the objective function
 *    value is so small that it can't be detected anymore.
 *    To deactivate this criterion completely, set the
 *    value `min_iterations_with_no_change` to a large
 *    value (e.g. `UINT_MAX`).
 * 4. Minimum function value (default: -infinity).
 *    The algorithm terminates if the objective function
 *    value falls below the minimum function value.
 *    This criterion is turned off by default. It is mainly
 *    useful for comparing optimization algorithms.
 * 5. Minimum change of gradient or parameters
 *    (default for `double`: machine epsilon).
 *    The algorithm terminates if the L2-norm of the gradient change
 *    or the parameter change is less than this parameter.
 *    If set to a small value (e.g. machine epsilon)
 *    it is rarely triggered, however, prevents numerical problems
 *    arising from very small changes of the gradient or the parameter set.
 */
template<typename Scalar>
struct OptimizationParameters {
    /// machine epsilon adjust if needed
    Scalar machine_epsilon;

    /// maximum number of iterations of the BFGS algorithm
    unsigned max_iterations = 10000;
    /// parameter for convergence check;
    /// BFGS is converged if ||grad|| <= tolerance * max(1.0, ||x||)
    Scalar gradient_tolerance;
    /// minimum allowed step length
    Scalar alpha_min;

    /// c1 for first Wolfe condition (sufficient decrease)
    Scalar wolfe_c1 {1e-4};
    /// c2 for second Wolfe condition (curvature condition)
    Scalar wolfe_c2 {0.9};

    /// parameter sigma2 for damped BFGS update
    Scalar sigma2 {0.6};
    /// parameter sigma3 for damped BFGS update
    Scalar sigma3 {3.0};
    /// parameter sigma4 for damped BFGS update
    Scalar sigma4 {0.0};

    /// parameter for checking the numerical stability
    Scalar check_epsilon;

    /// termination criterion for minimal change of function value; BFGS is terminated if
    /// (f_{i-1} - f_{i}) / (0.5 * (|f_{i-1}| + |f_{i}|)) <= min_change
    Scalar min_change;
    /// the number of iterations the criterion above must hold
    unsigned min_iterations_with_no_change = 2;

    /// termination criterion of minimal of gradient or parameters; BFGS is terminated if
    /// ||s|| or ||r|| are smaller than this value.
    Scalar min_grad_param;

    /// termination criterion for minimal function value (stopping value);
    // BFGS is terminated if f_{i} <= min_value
    Scalar min_value;

    /// line search method (currently no choice)
    Linesearch linesearch = Linesearch::Backtracking;

    /// maximum number of linesearch iterations
    unsigned max_linesearch_iterations = std::numeric_limits<unsigned>::max();

    /// maximum history to store in limited memory methods
    unsigned m = 10;

    /// allow the restarting in case of errors
    bool allow_restarts = true;

    /// how to select alpha0 for start of line search
    Alpha0Policy alpha0 = Alpha0Policy::Constant;

  private:
    // we prevent instantiation without meaningful default parameters
    OptimizationParameters() = default;
    template<typename U>
    friend OptimizationParameters<U> getOptimizationParameters();

};

/**
 * @brief Get optimization parameters for this type.
 * @tparam Scalar Type of scalar values.
 */
template<typename Scalar>
OptimizationParameters<Scalar> getOptimizationParameters()
{
  return OptimizationParameters<Scalar>();
}

/**
 * @brief Default optimization parameters for `double`.
 */
template<>
inline OptimizationParameters<double> getOptimizationParameters<double>()
{
  OptimizationParameters<double> params;

  params.machine_epsilon = scalar_traits<double>::epsilon();
  params.gradient_tolerance = 1e-6;
  params.alpha_min = params.machine_epsilon;
  params.check_epsilon = 1e-10;
  params.min_change = 0.0;
  params.min_value = -scalar_traits<double>::infinity();
  params.min_grad_param = params.machine_epsilon * params.machine_epsilon;

  return params;
};

}
