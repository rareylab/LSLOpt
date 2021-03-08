#pragma once

#include <cmath>

#include "../OutputUtils.hpp"
#include "../OptimizationParameters.hpp"
#include "../OptimizationResult.hpp"
#include "../ScalarTraits.hpp"
#include "../Types.hpp"
#include "DefineOutputMacros.hpp"
#include "LineSearches.hpp"
#include "ProblemTraits.hpp"


namespace LSLOpt {

namespace Implementation {

/**
 * @brief This is a generic implementation of quasi-Newton methods.
 *
 * @param f objective function to minimize
 * @param x0 starting point for optimization
 * @param params optimization parameters
 * @param output_function function to use for error and status messages
 *
 * This function can be used for BFGS, L-BFGS or even gradient descent.
 * It executes the main steps of each quasi-Newton algorithm.
 *
 * - calculation of function value and gradient
 * - check of termination conditions (see below)
 * - calculation of new search directions - this depends on the actual algorithm used!
 * - line search (here: backtracking line search)
 * - damped(!) update of the (inverse) Hessian approximation
 *
 * Search direction determination:
 * This depends on the actual algorithm used. It is usually given as
 * -Hv, the algorithm defines how this is calculated.
 *
 * On the choice of the line search:
 * The use of the limited step length can often result in steps not satisfying
 * the curvature condition. Therefore we use the easy to implement backtracking
 * line search that can also be extended to accommodate for the limited step lengths
 * and combine it with a damped update strategy (see below).
 *
 * Damped update:
 * The backtracking line search does not guarantee the curvature condition
 * to be satisfied. Therefore we perform a damped BFGS update as described in
 * Al-Baali, M., Grandinetti, L., Psacane, O.
 * Damped Techniques for the Limited memory BFGS Method for Large Scale
 * Optimization. 2015. J Optim Theory Appl. 161. 688-699.
 *
 * This update technique changes the y vector such that the update results
 * in positive semi-definite matrices even if the original (s,y) would not.
 */
template<
  typename Problem,
  typename Scalar,
  template<typename, typename> class Storage,
  template<typename, typename, typename> class ConstrainedAlgorithm,
  typename OutputFunction,
  template<typename, typename> class MaxStepLengthIdentifierTrait>
OptimizationResult<Scalar> quasi_newton(
    Problem& problem,
    const Vector<Scalar>& x0,
    const OptimizationParameters<Scalar>& params,
    OutputFunction& output_function)
{
  using std::abs; // we need this for accessing abs()

  using ThisVector = Vector<Scalar>;
  using ThisStorage = Storage<Scalar, OutputFunction>;
  using ThisConstrainedAlgorithm = ConstrainedAlgorithm<Problem, Scalar, ThisStorage>;

  static_assert(
      problem_traits<Problem, Scalar>::has_value,
      "Problem must provide value function: Scalar value(const Vector<Scalar>&)");
  static_assert(
      problem_traits<Problem, Scalar>::has_gradient,
      "Problem must provide gradient function: Vector<Scalar> gradient(const Vector<Scalar>&)");

  Status status = Status::Running;

  auto line_search = select_line_search<Problem, Scalar, MaxStepLengthIdentifierTrait>(
      params.linesearch);

  const Eigen::Index n = x0.size();

  unsigned nof_restarts = 0;

  if (!ThisConstrainedAlgorithm::check_bounds(problem)) {
    LSL_OUTPUT(output_function, OutputLevel::Error, "Invalid bounds specified.");
    status = Status::InvalidBounds;

    LSL_OUTPUT(output_function, OutputLevel::Status, "Finishing with status: " << status);

    return OptimizationResult<Scalar>{
      scalar_traits<Scalar>::nan(),
      Vector<Scalar>::Constant(n, scalar_traits<Scalar>::nan()),
      Vector<Scalar>::Constant(n, scalar_traits<Scalar>::nan()),
      scalar_traits<Scalar>::nan(),
      status,
      0,
      nof_restarts};
  }

  ThisVector x = ThisConstrainedAlgorithm::truncate_x(problem, x0);

  /*
   * used notation:
   * B is the approximation of the Hessian matrix
   * H is the inverse of the approximation (i.e. B^-1)
   *
   * x is the parameter set
   * p is the search direction
   * grad is the gradient
   *
   * We store approximations to both the inverse of the Hessian matrix
   * and the Hessian matrix itself. This seems to be
   * inefficient on first glance, but it saves the O(n^3)
   * step for the inversion of the matrix or an LU decomposition.
   */

  // initial calculation (the rest is performed in the line search)
  Scalar f = problem.value(x);
  ThisVector g = problem.gradient(x);

  Scalar g_norm = scalar_traits<Scalar>::infinity();

  ThisStorage storage (n, params.m, params.check_epsilon, output_function);

  // the last change in x values
  ThisVector s = Vector<Scalar>::Constant(n, scalar_traits<Scalar>::nan());
  // the last damped change in gradient values
  ThisVector r = Vector<Scalar>::Constant(n, scalar_traits<Scalar>::nan());

  unsigned iterations = 0;

  unsigned iterations_with_no_change = 0;
  unsigned iterations_since_last_reset = 0;

  Scalar alpha;
  Scalar f_old = 0.0;
  Scalar phi0_old = 0.0;

  for (; iterations < params.max_iterations && status == Status::Running;
      ++iterations) {

    g_norm = ThisConstrainedAlgorithm::gradient_norm(problem, x, g);

    LSL_OUTPUT(output_function, OutputLevel::Status,
        iterations << ": VALUE = " << f << ", GRADIENT_NORM = " << g_norm);
    Scalar x_norm = x.norm();

    /*
     * Check convergence.
     */
    if (g_norm <= params.gradient_tolerance * std::max<Scalar>(Scalar{1}, x_norm)) {
      LSL_OUTPUT(output_function, OutputLevel::Status,
          iterations << ": Convergence reached.");
      status = Status::Converged;
    }
    else if (f == -scalar_traits<Scalar>::infinity()) {
      LSL_OUTPUT(output_function, OutputLevel::Status,
          iterations << ": Terminating with objective function -inf.");
      status = Status::MinusInfinity;
    }
    else if (f <= params.min_value) {
      LSL_OUTPUT(output_function, OutputLevel::Status,
          iterations << " Terminating with objective function <= min_value.");
      status = Status::MinValue;
    }
    else if (iterations_with_no_change >= params.min_iterations_with_no_change) {
      LSL_OUTPUT(output_function, OutputLevel::Status,
          iterations << ": Terminating because objective function did not change for "
              << iterations_with_no_change << " iterations.");
      status = Status::MinChange;
    }
    // check that the scalar product s'r and r'r are not too close to zero
    // (here the check is performed on ||s|| and ||r|| because the actual problem does
    //  not arise from orthogonal vectors r and s, but too short vectors r and s).
    else if (iterations_since_last_reset != 0
        && (s.norm() < params.min_grad_param || r.norm() < params.min_grad_param)) {
      LSL_OUTPUT(output_function, OutputLevel::Status,
          iterations << ": ||s|| or ||r|| are getting too small.");
      status = Status::XGradChange;
    }
    else {
      if (iterations_since_last_reset == 0) {
        // reset optimization, i.e. reset H and B to identity
        storage.reset();

        s = ThisVector::Constant(n, scalar_traits<Scalar>::nan());
        r = ThisVector::Constant(n, scalar_traits<Scalar>::nan());

        if (iterations > 0) {
          if (params.allow_restarts) {
            ++nof_restarts;
            LSL_OUTPUT(output_function, OutputLevel::Warning,
                iterations << ": Optimization was restarted!");
          }
          else {
            LSL_OUTPUT(output_function, OutputLevel::Warning,
                iterations << ": Restart not allowed, terminating.");
            status = Status::GeneralFailure;
            continue;
          }
        }

        // reset the counter with iterations with no change
        iterations_with_no_change = 0;
      }
      else {
        bool success = storage.update(s, r, g);
        if (!success) {
          LSL_OUTPUT(output_function, OutputLevel::Warning,
              iterations << ": B or H Update failed!");

          if (params.allow_restarts) {
            // if resetting didn't help the last time, this is bound to fail;
            // we also check that it was not reset the iteration before
            if (iterations_since_last_reset <= 1) {
              LSL_OUTPUT(output_function, OutputLevel::Error,
                  iterations << ": Resetting won't help, we already did this!");
              status = Status::GeneralFailure;
            }
            iterations_since_last_reset = 0;
          }
          else {
            LSL_OUTPUT(output_function, OutputLevel::Status,
                iterations << ": Restart not allowed, terminating.");
            status = Status::GeneralFailure;
            continue;
          }

          continue;
        }
      }

      Vector<Scalar> p = ThisConstrainedAlgorithm::search_direction(
              problem, storage, x, g, params.check_epsilon,
              params.machine_epsilon,
              output_function);

      if (p.unaryExpr(&scalar_traits<Scalar>::is_nan).any()) {
        LSL_OUTPUT(output_function, OutputLevel::Warning,
            iterations << ": Failed to retrieve search direction!");
        // if resetting didn't help the last time, this is bound to fail;
        // we also check that it was not reset the iteration before
        if (params.allow_restarts) {
          if (iterations_since_last_reset <= 1) {
            LSL_OUTPUT(output_function, OutputLevel::Error,
                iterations << ": Resetting won't help, we already did this!");
            status = Status::GeneralFailure;
          }
          iterations_since_last_reset = 0;
        }
        else {
          LSL_OUTPUT(output_function, OutputLevel::Status,
              iterations << ": Restart not allowed, terminating.");
          status = Status::GeneralFailure;
        }

        continue;
      }

      // pre-scale the direction on first iteration
      // -- this is equivalent to scaling the identity matrix for first hessian approximation
      if (iterations_since_last_reset == 0) {
        Scalar prescale = g.norm() > Scalar{1} ? Scalar{1} / g.norm() : Scalar{1};
        p *= prescale;
      }

      // initially allowed step length in this step
      Scalar alpha_max = MaxStepLengthIdentifierTrait<Problem, Scalar>::initial_step_length(
          problem, x, p);

      if (scalar_traits<Scalar>::is_nan(alpha_max)) {
        status = Status::InvalidMaximumValue;
        continue;
      }

      // ensure that the maximum step length does not violate possible box constraints
      alpha_max = std::min<Scalar>(alpha_max,
          ThisConstrainedAlgorithm::max_step_length(problem, x, p));

      LSL_OUTPUT(output_function, OutputLevel::Status,
          iterations << ": Maximum step length: " << alpha_max);

      Scalar phi0 = p.transpose() * g;
      if (iterations_since_last_reset == 0) {
        alpha = Scalar{1};
      }
      else {
        switch (params.alpha0) {
          case Alpha0Policy::Constant:
            alpha = 1.0; // start with 1.0, will be compare to alpha_max below
            break;
          case Alpha0Policy::ConstantScaling:
            alpha = std::max<Scalar>(alpha * phi0_old / phi0, params.alpha_min);
            break;
          case Alpha0Policy::Interpolation:
            alpha = std::max<Scalar>(Scalar{2} * (f - f_old) / phi0, params.alpha_min);
            alpha = std::min<Scalar>(Scalar{1}, Scalar{1.01} * alpha);
            break;
          default:
            alpha = 1.0;
        }

        alpha = std::min<Scalar>(alpha, alpha_max);
      }
      phi0_old = phi0;
      f_old = f;

      LSL_OUTPUT(output_function, OutputLevel::Status,
          iterations << ": Initial step length: " << alpha);

      Scalar f_new = f;
      ThisVector x_new(x);
      ThisVector g_new(g);

      // try line-search in the direction of p
      Status line_search_status;
      bool sufficient_decrease = false;
      bool curvature_condition = false;
      bool strong_wolfe = false;

      line_search_status = line_search(
        problem, alpha,
        x_new, g_new, f_new,
        sufficient_decrease, curvature_condition, strong_wolfe,
        x, g, f, p,
        params.alpha_min, alpha_max, params);

      LSL_OUTPUT(output_function, OutputLevel::Status,
          iterations << ": Line search terminating with status " << line_search_status
          << " and final step length " << alpha);

      if (line_search_status != Status::Success) {
        // if line search does not terminate with
        // Success status, make debug output
        if (line_search_status == Status::LineSearchMaxIterations) {
          LSL_OUTPUT(output_function, OutputLevel::Warning,
              iterations << ": Line search could not find minimizer within "
              << params.max_linesearch_iterations << " iterations.");
        }
        else if (line_search_status == Status::LineSearchMinimum) {
          LSL_OUTPUT(output_function, OutputLevel::Warning,
              iterations << ": Line search could not find minimizer larger than "
              << params.alpha_min << ".");
        }
        else if (line_search_status == Status::WrongSearchDirection) {
          LSL_OUTPUT(output_function, OutputLevel::Warning,
              iterations << ": Wrong (uphill) search direction.");
        }
        else if (line_search_status == Status::LineSearchFailed) {
          LSL_OUTPUT(output_function, OutputLevel::Warning,
              iterations << ": Line search failed!");
        }
        else {
          LSL_OUTPUT(output_function, OutputLevel::Error,
              iterations << ": Line search failed with unknown error code "
              << static_cast<unsigned>(line_search_status) << "!");
        }
      }

      if (!sufficient_decrease) {
        // this is nasty, we need a sufficient decrease
        LSL_OUTPUT(output_function, OutputLevel::Warning,
            iterations << ": Line search failed, no step with sufficient decrease found!");
        /*
         * Fallback: if the line search failed in the given direction,
         * reset the Hessian approximation to the beginning (Hessian is identity
         * and search direction is gradient) and retry line search.
         *
         * This is equivalent to one iteration of gradient descent.
         *
         * If this already failed recently, we fail completely.
         */

        // if resetting didn't help the last time, this is bound to fail;
        // we also check for the iteration before.
        if (params.allow_restarts) {
          if (iterations_since_last_reset <= 1) {
            LSL_OUTPUT(output_function, OutputLevel::Error,
                iterations << ": Resetting won't help, we already did this!");
            status = line_search_status == Status::Success
                   ? Status::GeneralFailure : line_search_status;
          }
          iterations_since_last_reset = 0;
        }
        else {
          LSL_OUTPUT(output_function, OutputLevel::Status,
              iterations << ": Restart not allowed, terminating.");
          status = line_search_status == Status::Success
                 ? Status::GeneralFailure : line_search_status;
        }

        // go to the beginning
        continue;
      }
      else {
        // This informs only about the state of the second and strong Wolfe conditions.
        // There are no consequences when these are violated, we actually expect this.
        if (!curvature_condition) {
          LSL_OUTPUT(output_function, OutputLevel::Status,
              iterations << ": Line search terminated with step length NOT fulfilling "
                         << "the curvature condition!");
        }
        else {
          LSL_OUTPUT(output_function, OutputLevel::Status,
              iterations << ": Line search terminated with step length fulfilling "
                         << "the curvature condition.");
          if (!strong_wolfe) {
            LSL_OUTPUT(output_function, OutputLevel::Status,
                iterations << ": Line search terminated with step length NOT fulfilling "
                           << "the strong wolfe conditions.");
          }
          else {
            LSL_OUTPUT(output_function, OutputLevel::Status,
                iterations << ": Line search terminated with step length fulfilling "
                           << "the strong wolfe conditions.");
          }
        }

        /**
         * Here, reparameterization is applied to improve the gradients. This can be used to avoid
         * the gimbal lock phenomenon with the exponential map rotation.
         * Two important assumptions are made:
         *   1. The reparameterization does never change the objective function value, but only
         *      the derivatives.
         *   2. The reparameterization does never move the x value outside of the box constraints,
         *      if any.
         *
         * Obviously, changing the x value and the derivatives has an influence on the
         * vectors s and y below. However, the damped update always ensures that the
         * update pairs preserve the positive definiteness!
         */
        if (ReparameterizeTrait<Problem, Scalar>::reparameterize(problem, x_new, g_new)) {
          LSL_OUTPUT(output_function, OutputLevel::Status,
              iterations << ": x value was reparameterized and gradient recalculated!");
        }

        // change in x value
        s = x_new - x;
        // change in gradient
        ThisVector y = g_new - g;

        // here the damping technique is applied
        ThisVector Bs = storage.calculate_Bv(s);

        Scalar sTy = s.dot(y);
        Scalar tau = sTy / s.dot(Bs);
        Scalar h = y.dot(storage.calculate_Hv(y)) / sTy;

        Scalar phi {1};

        if (tau <= Scalar{0} || tau < std::min<Scalar>(Scalar{1} - params.sigma2,
            (Scalar{1} - params.sigma4) * h)) {
          phi = params.sigma2 / (Scalar{1} - tau);
        }
        else if (tau > Scalar{1} + params.sigma3 && tau < (Scalar{1} - params.sigma4) * h) {
          phi = params.sigma3 / (tau - Scalar{1});
        }

        r = phi * y + (Scalar{1} - phi) * Bs;

        // Now (s, r) is our (damped) update pair that guarantees positive
        // definiteness of the updated (inverse) Hessian approximation.

        x = x_new;
        g = g_new;

        // calculate relative change.
        Scalar relative_change = (f - f_new) / ((abs(f) + abs(f_new)) / Scalar{2});

        // this criterion is very useful even if min_change = 0!
        // if the change is so small that there is no change in the function
        // value anymore (numerically at least) we terminate
        if (relative_change <= params.min_change) {
          ++iterations_with_no_change;
        }
        else {
          iterations_with_no_change = 0;
        }

        f = f_new;
        ++iterations_since_last_reset;
      }
    }
  }

  // if the optimization is not yet converged, but we have reached the maximum number of iterations
  if (status == Status::Running && iterations == params.max_iterations) {
    status = Status::MaxIterations;
  }

  // this should not happen
  if (status == Status::Running) {
    status = Status::GeneralFailure;
    LSL_OUTPUT(output_function, OutputLevel::Error, "Fatal internal error!");
  }

  LSL_OUTPUT(
      output_function,
      (status >= Status::Converged ? OutputLevel::Status : OutputLevel::Error),
      "Finishing with status: " << status);

  return OptimizationResult<Scalar>{f, x, g, g_norm, status, iterations, nof_restarts};
}

}

}
