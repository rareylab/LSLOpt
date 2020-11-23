#pragma once

#include "../ScalarTraits.hpp"
#include "../OptimizationStatus.hpp"
#include "../OptimizationParameters.hpp"
#include "../Types.hpp"


namespace LSLOpt {

namespace Implementation {
/**
 * @brief Function that checks sufficient decrease.
 *
 * @param f Current function value.
 * @param alpha Current step length.
 * @param f0 Initial function value (alpha=0).
 * @param m0 Initial slope (alpha=0).
 * @param c1 Parameter of the sufficient decrease test.
 *
 * Check if the sufficient decrease condition
 * (first Wolfe condition) is fulfilled for the
 * current step length.
 */
template<typename Scalar>
bool is_sufficient_decrease(
    const Scalar& f,
    const Scalar& alpha,
    const Scalar& f0,
    const Scalar& m0,
    const Scalar& c1)
{
  return f <= f0 + c1 * alpha * m0;
}

/**
 * @brief Function that checks the curvature condition.
 *
 * @param m Current slope..
 * @param m0 Initial slope (alpha=0).
 * @param c2 Parameter of the curvature condition test.
 *
 * Check if the curvature condition
 * (second Wolfe condition) is fulfilled for the
 * current step length.
 */
template<typename Scalar>
bool is_curvature_condition(
    const Scalar& m,
    const Scalar& m0,
    const Scalar& c2)
{
  return -m <= -c2 * m0;
}

/**
 * @brief Function that checks the strong wolfe condition.
 *
 * @param m Current slope..
 * @param m0 Initial slope (alpha=0).
 * @param c2 Parameter of the curvature condition test.
 *
 * Check if the strong Wolfe condition is fulfilled for the
 * current step length.
 */
template<typename Scalar>
bool is_strong_wolfe(
    const Scalar& m,
    const Scalar& m0,
    const Scalar& c2)
{
  using std::abs; // we need this for accessing abs()
  return abs(m) <= c2 * abs(m0);
}

template<
  typename Problem,
  typename Scalar,
  template<typename, typename> class MaxStepLengthIdentifierTrait>
Status no_line_search(
    Problem& problem,
    Scalar& alpha,
    Vector<Scalar>& x,
    Vector<Scalar>& g,
    Scalar& f,
    bool& sufficient_decrease,
    bool& curvature_condition,
    bool& strong_wolfe,
    const Vector<Scalar>& x0,
    const Vector<Scalar>& g0,
    const Scalar& f0,
    const Vector<Scalar>& p,
    const Scalar& alpha_min,
    const Scalar& alpha_max,
    const OptimizationParameters<Scalar>& params)
{
  sufficient_decrease = false;
  curvature_condition = false;
  strong_wolfe = false;

  return Status::GeneralFailure;
}

/**
 * @brief Perform a backtracking line search.
 *
 * @param problem Function to evaluate
 * @param alpha Result of the line search; will be changed!
 * @param x_new New x vector after applying the new step length; will be changed!
 * @param g_new New gradient vector after applying the new step length; will be changed!
 * @param f_new New function value after applying the new step length; will be changed!
 * @param x0 Initial x vector.
 * @param g0 initial gradient vector.
 * @param f0 initial function value.
 * @param p Optimization direction.
 * @param alpha_min Minimal step length, ensures numerical stability.
 * @param alpha_max Maximal step length allowed.
 * @param params User defined parameters for the optimization.
 * @return the status of the line search, see \ref Status for documentation.
 *
 * Please note:
 *   - If the scalar product between gradient and search direction is positive,
 *     we would go uphill and the function terminates with error
 *     \ref Status::WrongSearchDirection.
 *   - The first step length is the step length specified in the function call
 *     but never larger than the maximum provided step length.
 *   - If the first Wolfe condition (or Armijo condition) is met,
 *     the function returns with \ref Status::Success.
 *   - However, if the step length is smaller than the minimum step length or if the
 *     maximum number of iterations is exceeded, the function returns
 *     \ref Status::LineSearchMinimum or \ref Status::LineSearchMaxIterations, respectively.
 *
 *   @warning
 *   Please check the return value of the line search, and especially the
 *   `sufficient_decrease` and the `curvature_condition` property!
 *
 *   @warning
 *   If `sufficient_decrease` is `true` you can be sure that the sufficient decrease
 *   condition is met. However, `alpha` may be very small.
 *
 *   @warning
 *   If `curvature_condition` is `true` you can be sure that the curvature condition
 *   is met. If it is `false`, you can still use the result of the line search,
 *   but you have to take precautions to ensure the positive semi definiteness of
 *   the matrices.
 *
 *   @warning
 *   The result of the line search should only be used if the status is non-negative.
 *
 *   @warning
 *   This line search only ensures the first Wolfe condition, but not the second Wolfe
 *   condition (and not to mention the strong Wolfe conditions).
 *   If used in BFGS, this line search alone cannot guarantee the positive semi-definiteness
 *   of the Hessian approximation!
 *
 *   @warning
 *   The output variables f, x and g are only set
 *   if at least one iteration is performed. It is recommended to set these values
 *   to the old ones before calling this function.
 *
 *   See:
 *   Nodecal and Wright. 2006. Numerical Optimization, 2nd ed. p. 33ff.
 */
template<
  typename Problem,
  typename Scalar,
  template<typename, typename> class MaxStepLengthIdentifierTrait>
Status armijo_line_search(
    Problem& problem,
    Scalar& alpha,
    Vector<Scalar>& x,
    Vector<Scalar>& g,
    Scalar& f,
    bool& sufficient_decrease,
    bool& curvature_condition,
    bool& strong_wolfe,
    const Vector<Scalar>& x0,
    const Vector<Scalar>& g0,
    const Scalar& f0,
    const Vector<Scalar>& p,
    const Scalar& alpha_min,
    const Scalar& alpha_max,
    const OptimizationParameters<Scalar>& params)
{
  sufficient_decrease = false;
  curvature_condition = false;
  strong_wolfe = false;

  const Scalar tau = Scalar{1} / Scalar{2};

  Scalar m0 = p.transpose() * g0;

  // we would go uphill, that's an error
  if (m0 > Scalar{0}) {
    // this should never happen because of the measures taken in the quasi newton implementation
    return Status::WrongSearchDirection;
  }

  // initial step length is the maximum step length, but never exceeds 1.0
  alpha = std::min<Scalar>(alpha, alpha_max);

  bool change_acceptable = false;

  unsigned iterations = 0;
  /*
   * This for loop runs as long
   *  - the number of iterations doesn't get too large
   *  - alpha doesn't get too small
   *  - the step length is not acceptable because it violates the first Wolfe
   *    or the externally defined condition
   */
  for (; iterations < params.max_linesearch_iterations
         && (!change_acceptable || !sufficient_decrease) && alpha >= alpha_min; alpha *= tau) {

    // recalculate function value
    x = x0 + alpha * p;

    if (MaxStepLengthIdentifierTrait<Problem, Scalar>::is_check_acceptable_first(problem)) {
      change_acceptable = MaxStepLengthIdentifierTrait<Problem, Scalar>::change_acceptable(
          problem, x0, x) <= 0.0;
      // evaluate function only if change is acceptable
      if (change_acceptable) {
        f = problem.value(x);
        // check first Wolfe condition
        sufficient_decrease = is_sufficient_decrease(f, alpha, f0, m0, params.wolfe_c1);
        ++iterations;
      }
    }
    else {
      f = problem.value(x);
      // check first Wolfe condition
      sufficient_decrease = is_sufficient_decrease(f, alpha, f0, m0, params.wolfe_c1);
      ++iterations;
      // check the acceptance only if sufficient decrease is true
      if (sufficient_decrease) {
        change_acceptable = MaxStepLengthIdentifierTrait<Problem, Scalar>::change_acceptable(
            problem, x0, x) <= 0.0;
      }
    }
  }

  // was multiplied with tau before termination
  alpha /= tau;

  // Recalculate gradient only here, we don't need it for the first Wolfe condition!
  // And only if at least one iteration was performed (and x_new was changed).
  if (iterations > 0) {
    g = problem.gradient(x);
  }

  Scalar m = p.dot(g);

  // check the other Wolfe conditions; we don't need them but it is cheap
  curvature_condition = is_curvature_condition(m, m0, params.wolfe_c2);
  strong_wolfe = is_strong_wolfe(m, m0, params.wolfe_c2);

  // if step length is getting too small, notify
  if (alpha * tau < alpha_min) {
    return Status::LineSearchMinimum;
  }

  // if step length is not too small and Armijo condition is fulfilled => success
  if (sufficient_decrease) {
    return Status::Success;
  }

  // if armijo condition is not fulfilled, and the max number of iterations was reached
  if (iterations == params.max_linesearch_iterations) {
    return Status::LineSearchMaxIterations;
  }

  // general failure, we should never end up here
  return Status::LineSearchFailed;
}

/**
 * @brief Select a line search.
 * @param ls The line search algorithm.
 * @returns The corresponding line search.
 *
 * @todo
 * This is currently of little use. It becomes
 * important if multiple line searches are available.
 */
template<
  typename Problem,
  typename Scalar,
  template<typename, typename> class MaxStepLengthIdentifierTrait>
auto select_line_search(const Linesearch& ls)
{
  switch(ls) {
    case Linesearch::Backtracking:
      return armijo_line_search<Problem, Scalar, MaxStepLengthIdentifierTrait>;
    default:
      return no_line_search<Problem, Scalar, MaxStepLengthIdentifierTrait>;
  }
}

}
}
