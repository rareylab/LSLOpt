#pragma once

#include "DefineOutputMacros.hpp"
#include "Utils.hpp"


namespace LSLOpt {

namespace Implementation {

/**
 * @brief L-BFGS storage.
 * @tparam Scalar The scalar type of vector/matrix coefficients.
 *
 * This is the implementation of the limited memory BFGS algorithm.
 * Here, the approximation of the (inverse) Hessian is stored
 * as the last m update pairs.
 *
 * It requires \f$ \Theta(nm + m^2) \f$ storage.
 *
 * We use the matrix representation described in
 *
 * Byrd, R.H., Nocedal, J., Schnable, R.B.
 * Representation of quasi-Newton matrices and their use in
 * limited memory methods. 1994. Mathematical Programming. 63. 129-156.
 *
 * Byrd, R.H., Lu, P., Nocedal, J., Zhu, C.
 * A Limited Memory Algorithm for Bound Constrained Optimization. 1995.
 * SIAM Journal of Scientific and Statistical Computing. 16(5). 1190-1208.
 */
template<typename Scalar, typename OutputFunction>
struct LBFGSStorage {

    /**
     * @brief Construct a BFGS storage.
     * @param n Dimensionality of the problem.
     * @param m Number \f$ (\mathbf{s}, \mathbf{y}) \f$ update pairs to store.
     * @param epsilon Small value for numerical stability check.
     * @param output_function Output function for status messages.
     */
    LBFGSStorage(
        Eigen::Index n,
        Eigen::Index m,
        Scalar epsilon,
        OutputFunction& output_function);

    /**
     * @brief Reset the (inverse) Hessian approximation to
     * identity matrix.
     *
     * This function deletes all \f$ (\mathbf{s}, \mathbf{y}) \f$ update pairs.
     *
     * Runtime \f$ \Theta(n) \f$.
     */
    void reset();

    /**
     * @brief Calculate product of inverse Hessian approximation \f$ \mathbf{H} \f$
     *        with vector \f$ \mathbf{v} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @param STv Product of the \f$ \mathbf{S}^\intercal \f$ matrix with \f$ \mathbf{v} \f$
     * @param YTv Product of the \f$ \mathbf{S}^\intercal \f$ matrix with \f$ \mathbf{v} \f$
     * @returns The product \f$ \mathbf{H} \mathbf{v} \f$.
     *
     * Runtime \f$ \Theta(mn) \f$.
     */
    Vector<Scalar> calculate_Hv(
        const Vector<Scalar>& v,
        const Vector<Scalar>& STv,
        const Vector<Scalar>& YTv);

    /**
     * @brief Calculate product of inverse Hessian approximation \f$ \mathbf{H} \f$
     *        with vector \f$ \mathbf{v} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The product \f$ \mathbf{H} \mathbf{v} \f$.
     *
     * Runtime \f$ \Theta(mn) \f$.
     */
    Vector<Scalar> calculate_Hv(const Vector<Scalar>& v);

    /**
     * @brief Calculate normalized scalar product of vector \f$ \mathbf{v} \f$
     *        with inverse Hessian approximation \f$ \mathbf{H} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The normalized scalar product \f$ \mathbf{v}^\intercal \mathbf{H} \mathbf{v} \f$.
     *
     * Runtime \f$ \Theta(mn) \f$.
     */
    Scalar calculate_vHv(const Vector<Scalar>& v);

    /**
     * @brief Calculate product of Hessian approximation \f$ \mathbf{B} \f$
     *        with vector \f$ \mathbf{v} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @param STv Product of the \f$ \mathbf{S}^\intercal \f$ matrix with \f$ \mathbf{v} \f$
     * @param YTv Product of the \f$ \mathbf{S}^\intercal \f$ matrix with \f$ \mathbf{v} \f$
     * @returns The product \f$ \mathbf{B} \mathbf{v} \f$.
     *
     * Runtime \f$ \Theta(mn) \f$.
     */
    Vector<Scalar> calculate_Bv(
        const Vector<Scalar>& v,
        const Vector<Scalar>& STv,
        const Vector<Scalar>& YTv);

    /**
     * @brief Calculate product of Hessian approximation \f$ \mathbf{B} \f$
     *        with vector \f$ \mathbf{v} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The product \f$ \mathbf{B} \mathbf{v} \f$.
     *
     * Runtime \f$ \Theta(mn) \f$.
     */
    Vector<Scalar> calculate_Bv(const Vector<Scalar>& v);

    /**
     * @brief Calculate normalized scalar product of vector \f$ \mathbf{v} \f$
     *        with Hessian approximation \f$ \mathbf{B} \f$.
     * @param v Vector \f$ \mathbf{v} \f$ for calculation.
     * @returns The normalized scalar product \f$ \mathbf{v}^\intercal \mathbf{B} \mathbf{v} \f$.
     *
     * Runtime \f$ \Theta(mn) \f$.
     *
     * @todo There is a formulation that calculates this in \f$ \Theta(m^2) \f$.
     */
    Scalar calculate_vBv(const Vector<Scalar>& v);

    /**
     * @brief Calculate the Hessian matrix approximation \f$ \mathbf{B} \f$.
     * @returns The Hessian matrix approximation \f$ \mathbf{B} \f$.
     *
     * @warning
     * This is only for debugging and testing. This can get
     * very large and it undermines the limited-memory concept!
     */
    [[deprecated]] Matrix<Scalar> calculate_B();

    /**
     * @brief Update the (inverse) Hessian approximation.
     * @param s Change in x coordinate.
     * @param y Change in gradient.
     * @param g New gradient.
     * @returns `true` if successful, `false` otherwise

     * The runtime of `update` is \f$ 4*n*m + \Theta(m^3) \f$
     *
     * The \f$ \Theta(m^3) \f$ part stems from the Cholesky decomposition
     * and the inversion of \f$ \mathbf{M} \f$.
     */
    bool update(
        const Vector<Scalar>& s,
        const Vector<Scalar>& y,
        const Vector<Scalar>& g);

    /**
     * @brief Function that resizes the storage to `b`.
     * @param b New size of the storage.
     *
     * If \f$ b < m \f$, then the size of the storage is increased.
     * Otherwise the oldest update pair is deleted.
     *
     * Runtime \f$ \Theta(mn + m^2) \f$.
     */
    void resize(Eigen::Index b);

    /// Dimensionality of the problem.
    Eigen::Index n;
    /// Maximal number of update pairs to store.
    Eigen::Index m;
    /// Current number of stored update pairs.
    Eigen::Index b;

    /// \f$ 2n*m \f$ working matrix
    Matrix<Scalar> W;
    /// \f$ 2m*2m \f$ working matrix
    Matrix<Scalar> M;

    /// \f$ n*m \f$ matrix storing the last \f$ \mathbf{s} \f$ vectors
    Matrix<Scalar> S;
    /// \f$ n*m \f$ matrix storing the last \f$ \mathbf{y} \f$ vectors
    Matrix<Scalar> Y;
    /// \f$ m*m \f$ helper matrix
    Matrix<Scalar> R;
    /// \f$ m*m \f$ helper matrix
    Matrix<Scalar> L;
    /// \f$ m*m \f$ helper matrix
    Matrix<Scalar> D;
    /// \f$ m*m \f$ matrix storing \f$ \mathbf{Y}^\intercal \mathbf{Y} \f$
    Matrix<Scalar> YTY;
    /// \f$ m*m \f$ matrix storing \f$ \mathbf{S}^\intercal \mathbf{S} \f$
    Matrix<Scalar> STS;

    /// \f$ 2m*2m \f$ working matrix
    Matrix<Scalar> LOW;
    /// \f$ 2m*2m \f$ working matrix
    Matrix<Scalar> UPP;

    /// current scaling of the inverse Hessian
    Scalar gamma = Scalar{1};

    /// numerical stability check epsilon
    Scalar epsilon;

    /// output function for status messages.
    OutputFunction& output_function;
};

template<typename Scalar, typename OutputFunction>
LBFGSStorage<Scalar, OutputFunction>::LBFGSStorage(
    Eigen::Index n,
    Eigen::Index m,
    Scalar epsilon,
    OutputFunction& output_function)
: n(n)
, m(m)
, b(0)
, epsilon(epsilon)
, output_function(output_function)
{
  this->reset();
}

template<typename Scalar, typename OutputFunction>
void LBFGSStorage<Scalar, OutputFunction>::reset()
{
  b = 0;

  W = Matrix<Scalar>::Zero(n, 0);
  M = Matrix<Scalar>::Zero(n, 0);

  S = Matrix<Scalar>::Zero(n, 0);
  Y = Matrix<Scalar>::Zero(n, 0);
  R = Matrix<Scalar>::Zero(0, 0);
  L = Matrix<Scalar>::Zero(0, 0);
  D = Matrix<Scalar>::Zero(0, 0);
  YTY = Matrix<Scalar>::Zero(0, 0);
  STS = Matrix<Scalar>::Zero(0, 0);

  LOW = Matrix<Scalar>::Zero(0, 0);
  UPP = Matrix<Scalar>::Zero(0, 0);

  // initial scaling of H and B is 1.0
  gamma = Scalar{1};

}

template<typename Scalar, typename OutputFunction>
Vector<Scalar> LBFGSStorage<Scalar, OutputFunction>::calculate_Hv(
    const Vector<Scalar>& v,
    const Vector<Scalar>& STv,
    const Vector<Scalar>& YTv)
{
  // O(m^2)
  Vector<Scalar> RSTv = solve_triangular_system_and_check(
      R, STv, TriangleMatrixType::Upper, epsilon, output_function);

  // we need these static casts if we're not using `double`
  Vector<Scalar> RTYTv = solve_triangular_system_and_check(
      R.transpose(),
      YTv, TriangleMatrixType::Lower, epsilon, output_function);

  Vector<Scalar> tmp = solve_triangular_system_and_check(
      R.transpose(),
      (D + gamma * YTY) * RSTv,
      TriangleMatrixType::Lower, epsilon, output_function);

  // 2*n*m
  Vector<Scalar> Hv = gamma * v + S * (tmp - gamma * RTYTv) + gamma * Y * (-RSTv);

  return Hv;
}

template<typename Scalar, typename OutputFunction>
Vector<Scalar> LBFGSStorage<Scalar, OutputFunction>::calculate_Hv(const Vector<Scalar>& v)
{
  return calculate_Hv(v, S.transpose() * v, Y.transpose() * v);
}

template<typename Scalar, typename OutputFunction>
Scalar LBFGSStorage<Scalar, OutputFunction>::calculate_vHv(const Vector<Scalar>& v)
{
  return v.dot(calculate_Hv(v));
}

template<typename Scalar, typename OutputFunction>
Vector<Scalar> LBFGSStorage<Scalar, OutputFunction>::calculate_Bv(
    const Vector<Scalar>& v,
    const Vector<Scalar>& STv,
    const Vector<Scalar>& YTv)
{
  // O(m)
  Vector<Scalar> p (2*b);
  p << YTv.head(b), Scalar{1} / gamma * STv.head(b);

  Vector<Scalar> p_ = solve_triangular_system_and_check(
      LOW, p, TriangleMatrixType::Lower, epsilon, output_function);

  // O(m^2)
  p = solve_triangular_system_and_check(
      UPP, p_, TriangleMatrixType::Upper, epsilon, output_function);

  // 2*n*m
  Vector<Scalar> Bv = Scalar{1} / gamma * v - Y * p.head(b) - Scalar{1} / gamma * S * p.tail(b);

  return Bv;
}

template<typename Scalar, typename OutputFunction>
Vector<Scalar> LBFGSStorage<Scalar, OutputFunction>::calculate_Bv(const Vector<Scalar>& v)
{
  return calculate_Bv(v, S.transpose() * v, Y.transpose() * v);
}

template<typename Scalar, typename OutputFunction>
Scalar LBFGSStorage<Scalar, OutputFunction>::calculate_vBv(const Vector<Scalar>& v)
{
  return v.dot(calculate_Bv(v));
}

template<typename Scalar, typename OutputFunction>
[[deprecated]] Matrix<Scalar> LBFGSStorage<Scalar, OutputFunction>::calculate_B()
{
  return Scalar{1} / gamma * Matrix<Scalar>::Identity(n, n) - W * M * W.transpose();
}

template<typename Scalar, typename OutputFunction>
bool LBFGSStorage<Scalar, OutputFunction>::update(
    const Vector<Scalar>& s,
    const Vector<Scalar>& y,
    const Vector<Scalar>& g)
{
  // b is the actual size of the history
  b = std::min(m, b + 1);
  // if the history is not full (yet), increase the size
  // if the history is already full, move the content
  this->resize(b);

  // O(n)
  S.col(b-1) = s;
  Y.col(b-1) = y;

  // O(n*m)
  // set the b-1 column of R to the scalar product of
  // each column of S with the b-1 column of Y
  R.row(b-1).head(b-1).setZero();
  R.col(b-1).noalias() = S.transpose() * Y.col(b-1);

  // O(n*m)
  // set the b-1 row of L to the scalar product of
  // each column of Y with the b-1 column of S
  // (the diagonal is 0.0 !)
  L.row(b-1).head(b-1).noalias() = Y.transpose().topLeftCorner(b-1, n) * S.col(b-1);
  L.col(b-1).setZero();

  // O(n*m)
  YTY.col(b-1).noalias() = Y.transpose() * Y.col(b-1);
  YTY.row(b-1).head(b-1) = YTY.col(b-1).head(b-1).eval();

  // O(m)
  D.col(b-1).setZero();
  D.row(b-1).setZero();
  D(b - 1, b - 1) = R(b - 1, b - 1);

  // O(n*m)
  STS.col(b-1).noalias() = S.transpose() * S.col(b-1);
  STS.row(b-1).head(b-1) = STS.col(b-1).head(b-1).eval();

  // O(1)
  gamma = R(b - 1, b - 1) / YTY(b - 1, b - 1);

  // convert the matrix to an explicit diagonal matrix to make the inverse efficient
  DiagonalMatrix<Scalar> dD(D.diagonal());

  // O(m)
  DiagonalMatrix<Scalar> dDI = dD.inverse();

  // O(m)
  DiagonalMatrix<Scalar> dD_sq(b);
  dD_sq.diagonal().array() = dD.diagonal().array().sqrt();

  // O(m), inverting diagonal matrix should be more efficient than sqrt
  DiagonalMatrix<Scalar> dD_sqI = dD_sq.inverse();

  Matrix<Scalar> to_factorize = Scalar{1} / gamma * STS + L * dDI * L.transpose();
  // compute cholesky factorization, O(m^3)
  Eigen::LLT<Matrix<Scalar>> llt (to_factorize);
  Matrix<Scalar> J = llt.matrixL();
  Matrix<Scalar> JT = llt.matrixU();

  /*
   * This is a step that can potentially suffer from (and also uncover)
   * numerical instabilities and problems, like indefinite matrices
   * (i.e. the initial matrix could have very small negative or very small
   *  complex eigenvalues, that should not be there).
   */

  // check if cholesky factorization was successful
  Matrix<Scalar> JJT = J * JT;
  Scalar re{0};
  // if the absolute error is != 0.0, then one of these values is also != 0.0
  if (to_factorize.norm() != Scalar{0}) {
    re = (to_factorize - JJT).norm() / to_factorize.norm();
  }
  else if (JJT.norm() != Scalar{0}) {
    re = (to_factorize - JJT).norm() / JJT.norm();
  }

  if (re > epsilon) {
    this->reset();
    LSL_OUTPUT(output_function, OutputLevel::Warning,
        "Relative error of Cholesky decomposition is " << re
          << " and larger than " << epsilon);
    // reset and terminate with error if we have a problem
    return false;
  }
  else {
    LSL_OUTPUT(output_function, OutputLevel::Debug,
        "Relative error of Cholesky decomposition is " << re
          << " and smaller or equal than " << epsilon);
  }

  // o(m^2) (because diagonal)
  Matrix<Scalar> D_sqILT = dD_sqI * L.transpose();

  // o(m^2) (because diagonal)
  Matrix<Scalar> LD_sqI = L * dD_sqI;

  // replace this by map!

  // O(m^2)
  LOW = Matrix<Scalar>::Zero(2*b, 2*b);
  LOW.topLeftCorner(b, b).diagonal() = dD_sq.diagonal();
  LOW.bottomLeftCorner(b, b) = -LD_sqI;
  LOW.bottomRightCorner(b, b) = J;

  // O(m^2)
  UPP = Matrix<Scalar>::Zero(2*b, 2*b);
  UPP.topLeftCorner(b, b).diagonal() = -dD_sq.diagonal();
  UPP.topRightCorner(b, b) = D_sqILT;
  UPP.bottomRightCorner(b, b) = JT;

  // O(m^2)
  W = Matrix<Scalar>::Zero(n, 2 * b);
  W.topLeftCorner(n, b) = Y;
  W.topRightCorner(n, b) = 1.0 / gamma * S;

  // O(m^2)
  Matrix<Scalar> M_ = Matrix<Scalar>::Zero(2 * b, 2 * b);
  M_.topLeftCorner(b, b) = -D;
  M_.topRightCorner(b, b) = L.transpose();
  M_.bottomLeftCorner(b, b) = L;
  M_.bottomRightCorner(b, b) = 1.0 / gamma * STS;

  /*
   * This is a step that can potentially suffer from (and also uncover)
   * numerical instabilities and problems, like indefinite matrices
   * (i.e. the initial matrix could have very small negative or very small
   *  complex eigenvalues, that should not be there).
   */

  // O(m^3)
  M = Eigen::FullPivLU<Matrix<Scalar>>(M_).inverse();

  Matrix<Scalar> identity = Matrix<Scalar>::Identity(2*b, 2*b);
  Matrix<Scalar> should_be_identity = M * M_;

  re = (identity - should_be_identity).norm() / identity.norm();

  if (re > epsilon) {
    this->reset();
    LSL_OUTPUT(output_function, OutputLevel::Warning,
        "Relative error of matrix inversion is " << re
          << " and larger than " << epsilon);
    // reset and return false in case of such an error
    return false;
  }
  else {
    LSL_OUTPUT(output_function, OutputLevel::Debug,
        "Relative error of matrix inversion is " << re
          << " and smaller or equal than " << epsilon);
  }

  return true;
}

template<typename Scalar, typename OutputFunction>
void LBFGSStorage<Scalar, OutputFunction>::resize(Eigen::Index b)
{
  if (S.cols() < b) {
    this->b = b;
    // S is a n x b matrix
    S.conservativeResize(Eigen::NoChange, b);
    // Y is a n x b matrix
    Y.conservativeResize(Eigen::NoChange, b);
    // R is a b x b matrix (and upper triangle)
    R.conservativeResize(b, b);
    // L is a b x b matrix (and lower triangle)
    L.conservativeResize(b, b);
    // D is a b x b matrix
    D.conservativeResize(b, b);
    // YY is a b x b matrix
    YTY.conservativeResize(b, b);
    // SS is a b x b matrix
    STS.conservativeResize(b, b);
  }
  else {
    S.topLeftCorner(n, b-1) = S.topRightCorner(n, b-1);
    Y.topLeftCorner(n, b-1) = Y.topRightCorner(n, b-1);

    R.topLeftCorner(b-1, b-1) = R.bottomRightCorner(b-1, b-1);
    L.topLeftCorner(b-1, b-1) = L.bottomRightCorner(b-1, b-1);
    D.topLeftCorner(b-1, b-1) = D.bottomRightCorner(b-1, b-1);
    YTY.topLeftCorner(b-1, b-1) = YTY.bottomRightCorner(b-1, b-1);
    STS.topLeftCorner(b-1, b-1) = STS.bottomRightCorner(b-1, b-1);
  }
}

}

}
