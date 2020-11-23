#pragma once

#include <cmath>

#include "../ScalarTraits.hpp"
#include "../Types.hpp"
#include "DefineOutputMacros.hpp"


namespace LSLOpt {

namespace Implementation {

enum class TriangleMatrixType {
    Upper,
    Lower
};

/**
 * @brief Solve a triangular linear equation system
 *        \f$ \mathbf{M} \mathbf{x} = \mathbf{v} \f$.
 * @param m Matrix
 * @param v Vector
 * @param tmt Type of triangular matrix (upper/lower).
 * @param epsilon Relative error check.
 * @returns The solution of the system of linear equations.
 *
 * @warning
 * If the relative error exceeds the given epsilon the
 * system is deemed to be (not stable) solvable and
 * a NaN vector is returned.
 *
 * @warning
 * This function does not check if the matrix is really
 * a triangular matrix.
 */
template<typename DerivedM, typename DerivedV, typename Scalar, typename OutputFunction>
Vector<Scalar> solve_triangular_system_and_check(
    const Eigen::MatrixBase<DerivedM>& m,
    const Eigen::MatrixBase<DerivedV>& v,
    TriangleMatrixType tmt,
    const Scalar epsilon,
    OutputFunction& output_function)
{
  static_assert(DerivedV::IsVectorAtCompileTime, "Second argument must be vector!");

  const Eigen::Index n = m.rows();

  Vector<Scalar> result = Vector<Scalar>::Zero(n);

  if (tmt == TriangleMatrixType::Upper) {
    result = m.template triangularView<Eigen::Upper>().solve(v);
  }
  else if (tmt == TriangleMatrixType::Lower) {
    result = m.template triangularView<Eigen::Lower>().solve(v);
  }

  // check if system is stable solvable
  Scalar re{0};

  // if the absolute error is != 0.0, then one of the values also is != 0.0
  if (v.norm() != Scalar{0}) {
    re = (m * result - v).norm() / v.norm();
  }
  else if (result.norm() != Scalar{0}) {
    re = (m * result - v).norm() / result.norm();
  }

  if (re > epsilon) {
    LSL_OUTPUT(output_function, OutputLevel::Warning,
        "Relative error of triangular system solution is " << re
          << " and larger than " << epsilon);
    return Vector<Scalar>::Constant(v.size(), scalar_traits<Scalar>::nan());
  }
  else {
    LSL_OUTPUT(output_function, OutputLevel::Debug,
        "Relative error of triangular system solution is " << re
          << " and smaller or equal than " << epsilon);
    return result;
  }
}

/**
 * @brief Calculates the \f$ L_\infty \f$ norm
 * of the projected gradient.
 * @param x Current x value.
 * @param g Current gradient.
 * @param lb Lower bounds.
 * @param ub Upper bounds.
 */
template<typename Scalar>
Scalar projected_gradient_norm(
    const Vector<Scalar>& x,
    const Vector<Scalar>& g,
    const Vector<Scalar>& lb,
    const Vector<Scalar>& ub)
{
  using std::abs; // we need this for accessing abs()
  Scalar norm{0};

  for (Eigen::Index i = 0; i < x.size(); ++i) {
    Scalar gi;
    if (g(i) < Scalar{0}) {
      gi = std::max<Scalar>(x(i)-ub(i), g(i));
    }
    else {
      gi = std::min<Scalar>(x(i)-lb(i), g(i));
    }
    norm = std::max<Scalar>(norm, abs(gi));
  }

  return norm;
}

}

}
