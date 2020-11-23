#pragma once

#include "../Types.hpp"


namespace LSLOpt {

namespace Implementation {

/**
 * @brief Storage for simply gradient descent optimization.
 * @tparam Scalar The scalar type of vector/matrix coefficients.

 * @warning This is only used for comparison.
 *
 * It requires \f$ \Theta(1) \f$ storage.
 */
template<typename Scalar, typename OutputFunction>
struct GDStorage {
    /**
     * @brief Construct a BFGS storage.
     * @param n Dimensionality of the problem.
     * @param output_function Output function for status messages.
     *
     * The second and third parameter is unused.
     */
    GDStorage(
        Eigen::Index n,
        Eigen::Index,
        Scalar,
        OutputFunction& output_function);

    /**
     * @brief Reset method.
     */
    void reset();

    /**
     * @brief Calculate product of inverse Hessian \f$ \mathbf{H} \f$ approximation
     *        with vector \f$ \mathbf{v} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The vector \f$ \gamma \mathbf{v} \f$.
     */
    Vector<Scalar> calculate_Hv(
        const Vector<Scalar>& v);

    /**
     * @brief Calculate normalized scalar product of vector \f$ \mathbf{v} \f$
     *        with inverse Hessian approximation \f$ \mathbf{H} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The scalar product \f$ \gamma \mathbf{v}^\intercal \mathbf{v} \f$.
     */
    Scalar calculate_vHv(
        const Vector<Scalar>& v);

    /**
     * @brief Access the inverse Hessian approximation \f$ \mathbf{H} \f$.
     *
     * Runtime \f$ \Theta(n^2) \f$.
     *
     * @warning
     * This method should only be used for debugging,
     * it is the scaled identity matrix anyway!
     */
    [[deprecated]] Matrix<Scalar> calculate_H();

    /**
     * @brief Calculate product of Hessian approximation \f$ \mathbf{B} \f$
     *        with vector \f$ \mathbf{v} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The vector \f$ \gamma \mathbf{v} \f$.
     */
    Vector<Scalar> calculate_Bv(
        const Vector<Scalar>& v);

    /**
     * @brief Calculate normalized scalar product of vector \f$ \mathbf{c} \f$
     *        with Hessian approximation \f$ \mathbf{B} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The scalar product \f$ \gamma \mathbf{v}^\intercal \mathbf{v} \f$.
     */
    Scalar calculate_vBv(
        const Vector<Scalar>& v);

    /**
     * @brief Access the Hessian approximation \f$ \mathbf{B} \f$.
     *
     * Runtime \f$ \Theta(n^2) \f$.
     *
     * @warning
     * This method should only be used for debugging,
     * it is the scaled identity matrix anyway!
     */
    [[deprecated]] Matrix<Scalar> calculate_B();

    /**
     * @brief Update the (inverse) Hessian approximation.
     * @param s Change in x coordinate.
     * @param y Change in gradient.
     * @param g New gradient.
     * @returns `true` if successful, `false` otherwise
     *
     * Runtime \f$ \Theta(1) \f$
     */
    bool update(
        const Vector<Scalar>& s,
        const Vector<Scalar>& y,
        const Vector<Scalar>& g);

    /// dimensionality of the problem
    Eigen::Index n;

    /// the output function for status messages
    OutputFunction& output_function;

    /// the scaling of the (initial) approximations
    Scalar gamma{1};
};

template<typename Scalar, typename OutputFunction>
GDStorage<Scalar, OutputFunction>::GDStorage(
    Eigen::Index n,
    Eigen::Index,
    Scalar,
    OutputFunction& output_function)
  : n(n)
  , output_function(output_function)
{
}

template<typename Scalar, typename OutputFunction>
void GDStorage<Scalar, OutputFunction>::reset()
{
  gamma = Scalar{1};
}

template<typename Scalar, typename OutputFunction>
Vector<Scalar> GDStorage<Scalar, OutputFunction>::calculate_Hv(
    const Vector<Scalar>& v)
{
  return gamma * v;
}

template<typename Scalar, typename OutputFunction>
Scalar GDStorage<Scalar, OutputFunction>::calculate_vHv(
    const Vector<Scalar>& v)
{
  return gamma * v.dot(v);
}

template<typename Scalar, typename OutputFunction>
Matrix<Scalar> GDStorage<Scalar, OutputFunction>::calculate_H()
{
  return gamma * Matrix<Scalar>::Identity(n, n);
}

template<typename Scalar, typename OutputFunction>
Vector<Scalar> GDStorage<Scalar, OutputFunction>::calculate_Bv(
    const Vector<Scalar>& v)
{
  return Scalar{1} / gamma * v;
}

template<typename Scalar, typename OutputFunction>
Scalar GDStorage<Scalar, OutputFunction>::calculate_vBv(
    const Vector<Scalar>& v)
{
  return Scalar{1} / gamma * v.dot(v);
}

template<typename Scalar, typename OutputFunction>
Matrix<Scalar> GDStorage<Scalar, OutputFunction>::calculate_B()
{
  return Scalar{1} / gamma * Matrix<Scalar>::Identity(n, n);
}

template<typename Scalar, typename OutputFunction>
bool GDStorage<Scalar, OutputFunction>::update(
    const Vector<Scalar>& s,
    const Vector<Scalar>& y,
    const Vector<Scalar>& g)
{
  // this scaling is always performed!
  gamma = Scalar{1} / g.norm();

  return true;
}

}

}
