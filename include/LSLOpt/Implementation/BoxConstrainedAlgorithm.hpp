#pragma once

#include <vector>

#include "../ScalarTraits.hpp"
#include "../Types.hpp"
#include "DefineOutputMacros.hpp"
#include "ProblemTraits.hpp"
#include "SubspaceUtils.hpp"


namespace LSLOpt {

namespace Implementation {

/**
 * @brief Generic implementation of box constrained quasi-Newton algorithms.
 * @tparam Problem The optimization problem type.
 * @tparam Scalar The scalar type of vector/matrix coefficients.
 * @tparam Storage The algorithm storage, i.e. BFGS, L-BFGS etc.
 */
template<typename Problem, typename Scalar, typename Storage>
struct BoxConstrainedAlgorithm {

    // we must have an lower bound
    static_assert(problem_traits<Problem, Scalar>::has_lower_bound,
        "Problem must provide lower bounds function: Vector<Scalar> lower_bounds()");

    // we must have an upper bound
    static_assert(problem_traits<Problem, Scalar>::has_upper_bound,
        "Problem must provide upper bounds function: Vector<Scalar> upper_bounds()");

    /**
     * @brief Verify that the bounds are correct.
     * @param problem The minimization problem (including box constraints).
     * @returns `true` if the bounds are ok, `false` otherwise
     */
    static bool check_bounds(Problem& problem);

    /**
     * @brief Truncate vector such that it fulfills the box constraints.
     * @param problem The minimization problem (including box constraints).
     * @param x Vector to truncate.
     * @returns The truncated x vector.
     */
    static Vector<Scalar> truncate_x(
        Problem& problem,
        const Vector<Scalar>& x);

    /**
     * @brief Calculate the projected gradient norm (\f$ L_\infty \f$ norm)
     * @param problem The minimization problem (including box constraints).
     * @param x Current \f$ \mathbf{x} \f$ value.
     * @param g Current gradient for gradient norm calculation.
     * @returns The projected gradient norm (\f$ L_\infty \f$ norm).
     */
    static Scalar gradient_norm(
        Problem& problem,
        const Vector<Scalar>& x,
        const Vector<Scalar>& g);

    /**
     * @brief Calculate the maximum allowed step length.
     * @param problem The minimization problem (including box constraints).
     * @param x Current \f$ \mathbf{x} \f$ value.
     * @param p Current search direction.
     * @return Maximum allowed step length that does not violate the bounds.
     */
    static Scalar max_step_length(
        Problem& problem,
        const Vector<Scalar>& x,
        const Vector<Scalar>& p);

    /**
     * @brief Calculate the new search direction.
     * @param problem The minimization problem (including box constraints).
     * @param storage The storage for calculation of Hessian vector products.
     * @param x Current \f$ \mathbf{x} \f$ value.
     * @param g Current gradient.
     * @param check_epsilon Small value to check numerical stability.
     * @param machine_epsilon Small value with machine accuracy.
     * @param output_function Function for status output.
     *
     * This function first calculates the generalized cauchy point
     * and then performs a direct primal subspace minimization.
     *
     * This technique is described in:
     *
     * Byrd, R.H., Lu, P., Nocedal, J., Zhu, C.
     * A Limited Memory Algorithm for Bound Constrained Optimization. 1995.
     * SIAM Journal of Scientific and Statistical Computing. 16(5). 1190-1208.
     *
     */
    template<typename OutputFunction>
    static Vector<Scalar> search_direction(
        Problem& problem,
        Storage& storage,
        const Vector<Scalar>& x,
        const Vector<Scalar>& g,
        const Scalar& check_epsilon,
        const Scalar& machine_epsilon,
        OutputFunction& output_function);
};

template<typename Problem, typename Scalar, typename Storage>
bool BoxConstrainedAlgorithm<Problem, Scalar, Storage>::check_bounds(Problem& problem)
{
  return (problem.lower_bounds().array() <= problem.upper_bounds().array()).all();
}

template<typename Problem, typename Scalar, typename Storage>
Vector<Scalar> BoxConstrainedAlgorithm<Problem, Scalar, Storage>::truncate_x(
    Problem& problem,
    const Vector<Scalar>& x)
{
  Vector<Scalar> r(x.size());

  auto lb = problem.lower_bounds();
  auto ub = problem.upper_bounds();

  // ensure that no bound is violated
  r.array() = x.array().max(lb.array()).min(ub.array());

  return r;
}

template<typename Problem, typename Scalar, typename Storage>
Scalar BoxConstrainedAlgorithm<Problem, Scalar, Storage>::gradient_norm(
    Problem& problem,
    const Vector<Scalar>& x,
    const Vector<Scalar>& g)
{
  // calculate projected gradient norm
  return projected_gradient_norm(x, g, problem.lower_bounds(), problem.upper_bounds());
}

template<typename Problem, typename Scalar, typename Storage>
Scalar BoxConstrainedAlgorithm<Problem, Scalar, Storage>::max_step_length(
    Problem& problem,
    const Vector<Scalar>& x,
    const Vector<Scalar>& p)
{
  Scalar alpha_max = scalar_traits<Scalar>::infinity();

  typename problem_traits<Problem, Scalar>::lower_bound_type lower_bounds {problem.lower_bounds()};
  typename problem_traits<Problem, Scalar>::upper_bound_type upper_bounds {problem.upper_bounds()};

  for (Eigen::Index i = 0; i < p.size(); ++i) {
    const Scalar& pi = p(i);

    if (pi < Scalar{0}) {
      Scalar lower_dist = lower_bounds(i) - x(i);
      // if value is at or over bound, no progress in this direction
      if (lower_dist >= Scalar{0}) {
        alpha_max = Scalar{0};
      }
      else if (pi * alpha_max < lower_dist) {
        alpha_max = lower_dist / pi;
      }
    }
    else if (pi > Scalar{0}) {
      Scalar upper_dist = upper_bounds(i) - x(i);
      // if value is at or over bound, no progress in this direction
      if (upper_dist <= Scalar{0}) {
        alpha_max = Scalar{0};
      }
      else if (pi * alpha_max > upper_dist) {
        alpha_max = upper_dist / pi;
      }
    }
  }

  /*
   * We need this additional check so the function is really never
   * evaluated outside the bounds!
   *
   * The alpha_max calculation above does not guarantee
   * the conditions here because of numerical inaccuracies.
   */
  for (Eigen::Index i = 0; i < p.size(); ++i) {
    const Scalar& pi = p(i);
    if (pi < Scalar{0}) {
      if (x(i) + alpha_max * p(i) < lower_bounds(i)) {
        alpha_max /= Scalar{2};
      }
    }
    else if (pi > Scalar{0}) {
      if (x(i) + alpha_max * p(i) > upper_bounds(i)) {
        alpha_max /= Scalar{2};
      }
    }
  }

  return alpha_max;
}

template<typename Problem, typename Scalar, typename Storage>
template<typename OutputFunction>
Vector<Scalar> BoxConstrainedAlgorithm<Problem, Scalar, Storage>::search_direction(
    Problem& problem,
    Storage& storage,
    const Vector<Scalar>& x,
    const Vector<Scalar>& g,
    const Scalar& check_epsilon,
    const Scalar& machine_epsilon,
    OutputFunction& output_function)
{
  // calculate the generalized cauchy point
  Vector<Scalar> x_cp = cauchy_point(x, g, storage,
      problem.lower_bounds(), problem.upper_bounds(), machine_epsilon);
  // calculate the subspace minimum in the direction of the cauchy point
  Vector<Scalar> ssm = subspace_min(x, g, x_cp, storage,
      problem.lower_bounds(), problem.upper_bounds(), check_epsilon);

  // The subspace minimum has been set to NaN if numerical problems
  // occurred during subspace minimization, see \ref subspace_min for details.
  if (ssm.unaryExpr(&scalar_traits<Scalar>::is_nan).any()) {
    LSL_OUTPUT(output_function, OutputLevel::Warning,
        "Detected numerical instability during subspace minimization!");
  }

  // the search direction is the vector from the current x to the subspace minimum
  Vector<Scalar> p = ssm - x;

  return p;
}

}
}
