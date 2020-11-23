#pragma once

#include "../ScalarTraits.hpp"
#include "DefineOutputMacros.hpp"


namespace LSLOpt {

namespace Implementation {

/**
 * @brief Generic implementation of unconstrained quasi-Newton algorithms.
 */
template<typename Problem, typename Scalar, typename Storage>
struct UnconstrainedAlgorithm {

    /**
     * @brief Verify that the bounds are correct.
     * @param problem The minimization problem (including box constraints).
     * @returns `true`
     */
    static bool check_bounds(Problem& problem);

    /**
     * @brief Truncate vector such that it fulfills the constraints.
     * @param problem The minimization problem.
     * @param x Vector to truncate.
     * @returns The \f$ \mathbf{x} \f$ vector itself.
     *
     * This function doesn't change \f$ \mathbf{x} \f$, no constraints!
     */
    static Vector<Scalar> truncate_x(
        Problem& problem,
        const Vector<Scalar>& x);

    /**
     * @brief Calculate the gradient norm
     * @param problem The minimization problem.
     * @param x Current \f$ \mathbf{x} \f$ value.
     * @param g Current gradient for gradient norm calculation.
     * @returns The gradient \f$ L_2 \f$-norm.
     */
    static Scalar gradient_norm(
        Problem& problem,
        const Vector<Scalar>& x,
        const Vector<Scalar>& g);

    /**
     * @brief Calculate the maximum allowed step length.
     * @param problem The minimization problem.
     * @param x Current \f$ \mathbf{x} \f$ value.
     * @param p Current search direction.
     * @return \f$ \infty \f$ because the problem is unconstrained.
     */
    static Scalar max_step_length(
        Problem& problem,
        const Vector<Scalar>& x,
        const Vector<Scalar>& p);

    /**
     * @brief Calculate the new search direction.
     * @param problem The minimization problem.
     * @param storage The storage for calculation of Hessian vector products.
     * @param x Current \f$ \mathbf{x} \f$ value.
     * @param g Current gradient.
     * @param check_epsilon Small value to check numerical stability.
     * @param machine_epsilon Small value with machine accuracy.
     * @param output_function Function for status output.
     *
     * This function calculates \f$ -\mathbf{H} \mathbf{g} \f$.
     * The exact calculations and therefore
     * the runtime depends on the used storage.
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
bool UnconstrainedAlgorithm<Problem, Scalar, Storage>::check_bounds(Problem& problem)
{
  return true;
}

template<typename Problem, typename Scalar, typename Storage>
Vector<Scalar> UnconstrainedAlgorithm<Problem, Scalar, Storage>::truncate_x(
    Problem& problem,
    const Vector<Scalar>& x)
{
  return x;
}

template<typename Problem, typename Scalar, typename Storage>
Scalar UnconstrainedAlgorithm<Problem, Scalar, Storage>::gradient_norm(
    Problem& problem,
    const Vector<Scalar>& x,
    const Vector<Scalar>& g)
{
  return g.norm();
}

template<typename Problem, typename Scalar, typename Storage>
Scalar UnconstrainedAlgorithm<Problem, Scalar, Storage>::max_step_length(
    Problem& problem,
    const Vector<Scalar>& x,
    const Vector<Scalar>& p)
{
  // there is no limitation in unconstrained optimization
  return scalar_traits<Scalar>::infinity();
}

template<typename Problem, typename Scalar, typename Storage>
template<typename OutputFunction>
Vector<Scalar> UnconstrainedAlgorithm<Problem, Scalar, Storage>::search_direction(
    Problem& problem,
    Storage& storage,
    const Vector<Scalar>& x,
    const Vector<Scalar>& g,
    const Scalar& check_epsilon,
    const Scalar& machine_epsilon,
    OutputFunction& output_function)
{
  Vector<Scalar> Hg = storage.calculate_Hv(g);

  // this should again give g (if everything was perfect)
  Vector<Scalar> BHg = storage.calculate_Bv(Hg);

  Scalar relative_error{0};
  // if the absolute error is != 0.0, then one of these values is also != 0.0
  if (g.norm() != Scalar{0}) {
    relative_error = (g - BHg).norm() / g.norm();
  }
  else if (BHg.norm() != Scalar{0}) {
    relative_error = (g - BHg).norm() / BHg.norm();
  }

  if (relative_error > check_epsilon) {
    /*
     * Currently we do nothing in this case because it is not a huge problem.
     * Only if no progress can be made in the new direction (because it is e.g.
     * uphill direction) we have a problem. And this is handled anyway by restarting.
     */
    LSL_OUTPUT(output_function, OutputLevel::Status,
        "Relative error of search direction determination is " << relative_error
          << " and larger than " << check_epsilon);
  }
  else {
    LSL_OUTPUT(output_function, OutputLevel::Debug,
        "Relative error of search direction determination ist " << relative_error
          << " and smaller or equal than " << check_epsilon);
  }

  return -Hg;
}

}

}
