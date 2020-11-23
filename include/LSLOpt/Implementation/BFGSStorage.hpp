#pragma once

#include "../ScalarTraits.hpp"
#include "../Types.hpp"


namespace LSLOpt {

namespace Implementation {

/**
 * @brief BFGS storage.
 *
 * This is the implementation of the BFGS algorithm.
 * It directly stores the approximate Hessian \f$ \mathbf{B} \f$ and the inverse
 * approximate inverse Hessian \f$ \mathbf{H} \f$.
 *
 * @tparam Scalar The scalar type of vector/matrix coefficients.
 *
 * It requires \f$ \Theta(n^2) \f$ storage!
 */
template<typename Scalar, typename OutputFunction>
struct BFGSStorage {
    /**
     * @brief Construct a BFGS storage.
     * @param n Dimensionality of the problem.
     * @param output_function Output function for status messages.
     *
     * The second and third parameter are unused.
     */
    BFGSStorage(
        Eigen::Index n,
        Eigen::Index,
        Scalar,
        OutputFunction& output_function);

    /**
     * @brief Reset the (inverse) Hessian approximation to
     * identity matrix.
     *
     * Runtime \f$ \Theta(n^2) \f$.
     */
    void reset();

    /**
     * @brief Calculate product of inverse Hessian approximation
     *        \f$ \mathbf{H} \f$ with vector \f$ \mathbf{v} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The product \f$ \mathbf{H} \mathbf{v} \f$.
     *
     * Runtime \f$ \Theta(n^2) \f$.
     */
    Vector<Scalar> calculate_Hv(
        const Vector<Scalar>& v);

    /**
     * @brief Calculate normalized scalar product of vector \f$ \mathbf{v} \f$
     *        with inverse Hessian approximation \f$ \mathbf{H} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The normalized scalar product \f$ \mathbf{v}^\intercal \mathbf{H} \mathbf{v} \f$.
     *
     * Runtime \f$ \Theta(n^2) \f$.
     */
    Scalar calculate_vHv(
        const Vector<Scalar>& v);

    /**
     * @brief Access the inverse Hessian approximation \f$ \mathbf{H} \f$.
     *
     * Runtime \f$ \Theta(1) \f$.
     */
    const Matrix<Scalar>& calculate_H();

    /**
     * @brief Calculate product of Hessian approximation \f$ \mathbf{B} \f$
     *        with vector \f$ \mathbf{v} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The product \f$ \mathbf{B} \mathbf{v} \f$.
     *
     * Runtime \f$ \Theta(n^2) \f$.
     */
    Vector<Scalar> calculate_Bv(
        const Vector<Scalar>& v);

    /**
     * @brief Calculate normalized scalar product of vector \f$ \mathbf{v} \f$
     *        with Hessian approximation \f$ \mathbf{B} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The normalized scalar product \f$ \mathbf{v}^\intercal \mathbf{B} \mathbf{v} \f$.
     *
     * Runtime \f$ \Theta(n^2) \f$.
     */
    Scalar calculate_vBv(
        const Vector<Scalar>& v);

    /**
     * @brief Access the Hessian approximation \f$ \mathbf{B} \f$.
     *
     * Runtime \f$ \Theta(1) \f$.
     */
    const Matrix<Scalar>& calculate_B();

    /**
     * @brief Update the (inverse) Hessian approximation.
     * @param s Change in x coordinate.
     * @param y Change in gradient.
     * @param g New gradient.
     * @returns `true` if successful, `false` otherwise
     *
     * Runtime \f$ \omega(n^2) \f$ (depends on the matrix multiplication
     * algorithm in `Eigen`).
     */
    bool update(
        const Vector<Scalar>& s,
        const Vector<Scalar>& y,
        const Vector<Scalar>& g);

    /// dimensionality of the problem
    Eigen::Index n;

    /// inverse Hessian approximation
    Matrix<Scalar> H;
    /// Hessian approximation
    Matrix<Scalar> B;
    /// is this the initial approximation?
    bool initial;

    /// output function for status messages
    OutputFunction& output_function;
};

template<typename Scalar, typename OutputFunction>
BFGSStorage<Scalar, OutputFunction>::BFGSStorage(
    Eigen::Index n,
    Eigen::Index,
    Scalar,
    OutputFunction& output_function)
: n(n)
, initial(true)
, output_function(output_function)
{

}

template<typename Scalar, typename OutputFunction>
void BFGSStorage<Scalar, OutputFunction>::reset()
{
  H = Matrix<Scalar>::Identity(n, n);
  B = Matrix<Scalar>::Identity(n, n);
  initial = true;
}

template<typename Scalar, typename OutputFunction>
Vector<Scalar> BFGSStorage<Scalar, OutputFunction>::calculate_Hv(
    const Vector<Scalar>& v)
{
  // matrix vector product
  return H * v;
}

template<typename Scalar, typename OutputFunction>
Scalar BFGSStorage<Scalar, OutputFunction>::calculate_vHv(
    const Vector<Scalar>& v)
{
  // matrix vector and dot product
  return v.dot(calculate_Hv(v));
}

template<typename Scalar, typename OutputFunction>
const Matrix<Scalar>& BFGSStorage<Scalar, OutputFunction>::calculate_H()
{
  return H;
}

template<typename Scalar, typename OutputFunction>
Vector<Scalar> BFGSStorage<Scalar, OutputFunction>::calculate_Bv(
    const Vector<Scalar>& v)
{
  // matrix vector product
  return B * v;
}

template<typename Scalar, typename OutputFunction>
Scalar BFGSStorage<Scalar, OutputFunction>::calculate_vBv(
    const Vector<Scalar>& v)
{
  // matrix vector and dot product
  return v.dot(calculate_Bv(v));
}

template<typename Scalar, typename OutputFunction>
const Matrix<Scalar>& BFGSStorage<Scalar, OutputFunction>::calculate_B()
{
  return B;
}

template<typename Scalar, typename OutputFunction>
bool BFGSStorage<Scalar, OutputFunction>::update(
    const Vector<Scalar>& s,
    const Vector<Scalar>& y,
    const Vector<Scalar>& g)
{
  if (initial) {
    // scale the initial (inverse) Hessian
    H *= y.dot(s) / y.dot(y);
    B *= y.dot(y) / y.dot(s);
    initial = false;
  }

  // update the Hessian approximation
  B = B - (B * s * s.transpose() * B)/(s.transpose() * B * s)
      + (y * y.transpose()) / (s.transpose() * y);

  Scalar rho = Scalar{1} / y.dot(s);

  // update the inverse Hessian approximation
  H = (Matrix<Scalar>::Identity(n, n) - s * y.transpose() * rho)
      * H * (Matrix<Scalar>::Identity(n, n) - y * s.transpose() * rho)
      + rho * s * s.transpose();
  /**
   * @todo
   * Here we could also introduce a check if the product B * H
   * really is the identity.
   */

  return !H.unaryExpr(&scalar_traits<Scalar>::is_nan).any()
      && !B.unaryExpr(&scalar_traits<Scalar>::is_nan).any();
}

}

}
