#pragma once

#include <cmath>

#include "Implementation/ProblemTraits.hpp"
#include "Types.hpp"


namespace LSLOpt {

/**
 * @brief Debug wrapper for any problem.
 *
 * This wrapper class compares the provided
 * gradient function to the numerically determined
 * gradient by finite differences.
 *
 * @warning
 * This makes the optimization slow and is not
 * intended for production use.
 */
template<typename Problem, typename Scalar, typename ErrorHandler>
struct DebugProblem : Problem {

  /**
   * @brief Construct a new debug problem.
   * @param error_handler Functor that is called in case of errors.
   * @param args Arguments are passed through to the constructor call of the problem.
   *
   * @note
   * The epsilon values may have to be adapted for your problem.
   */
  template <typename ... Args>
  DebugProblem(
      ErrorHandler& error_handler,
      Args&&... args)
  : Problem(std::forward<Args>(args)...)
  , m_error_handler(error_handler)
  {
    this->setDerivativeEpsilon();
    this->setCompareEpsilon();
  }

  /**
   * @brief Set the epsilon value for numerical derivative calculation.
   * @param derivative_epsilon Epsilon for numerical derivative calculation.
   *
   * @note
   * Meaningful values depend on the problem at hand!
   */
  void setDerivativeEpsilon(const Scalar& derivative_epsilon = 1e-6)
  {
    m_derivative_epsilon = derivative_epsilon;
  }

  /**
   * @brief Set the epsilon value for comparison of analytical and numerical derivative.
   * @param compare_epsilon Epsilon for comparison of analytical and numerical derivative.
   *
   * @note
   * Meaningful values depend on the problem at hand!
   */
  void setCompareEpsilon(const Scalar& compare_epsilon = 1e-3)
  {
    m_compare_epsilon = compare_epsilon;
  }

  /**
   * @brief Get the epsilon value for numerical derivative calculation.
   */
  const Scalar& getDerivativeEpsilon() const
  {
    return m_derivative_epsilon;
  }

  /**
   * @brief Get the epsilon value for comparison of analytical and numerical derivative.
   */
  const Scalar& getCompareEpsilon() const
  {
    return m_compare_epsilon;
  }

  /**
   * @brief Calculate and check the gradient.
   * @param x Point where to evaluate the function.
   * @returns Gradient at x.
   *
   * This function checks the calculated gradient
   * against the numerically determined gradient using
   * the finite difference method.
   *
   * If the deviation is too large, the error functor
   * is called.
   */
  typename Implementation::problem_traits<Problem, Scalar>::gradient_type
  gradient(const Vector<Scalar>& x)
  {
    using std::abs; // we need this for accessing abs()

    typename Implementation::problem_traits<Problem, Scalar>::gradient_type g
        = Problem::gradient(x);

    bool equal = true;
    Vector<Scalar> numerical_gradient(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
      Vector<Scalar> x_plus(x);
      x_plus(i) += m_derivative_epsilon;

      Vector<Scalar> x_minus(x);
      x_minus(i) -= m_derivative_epsilon;

      if (!x_value_is_ok<Problem>(x_minus, i) || !x_value_is_ok<Problem>(x_plus, i)) {
        continue;
      }

      Scalar score_plus = Problem::value(x_plus);

      Scalar score_minus = Problem::value(x_minus);

      Scalar numeric_first_derivative =
          (score_plus - score_minus) / (Scalar{2} * m_derivative_epsilon);
      numerical_gradient(i) = numeric_first_derivative;

      if (abs(numeric_first_derivative - g(i)) > m_compare_epsilon) {
        equal = false;
        // do not break, we want to calculate and see the rest for debugging
      }
    }

    if (!equal) {
      m_error_handler(x, numerical_gradient, g);
    }


    return g;
  }

  private:

    template<typename U,
        typename std::enable_if<!Implementation::problem_traits<U, Scalar>::has_lower_bound,
            U>::type* = nullptr>
    bool x_value_is_ok(
        const Vector<Scalar>& x,
        Eigen::Index i)
    {
      return true;
    }

    template<typename U,
        typename std::enable_if<Implementation::problem_traits<U, Scalar>::has_lower_bound,
            void>::type* = nullptr>
    bool x_value_is_ok(
        const Vector<Scalar>& x,
        Eigen::Index i)
    {
      return x(i) >= this->lower_bounds()(i) && x(i) <= this->upper_bounds()(i);
    }

    ErrorHandler& m_error_handler;

    Scalar m_derivative_epsilon;
    Scalar m_compare_epsilon;
};

}
