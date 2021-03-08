#pragma once

#include <vector>

#include "../ScalarTraits.hpp"
#include "BFGSStorage.hpp"
#include "GDStorage.hpp"
#include "LBFGSStorage.hpp"


namespace LSLOpt {

namespace Implementation {

/**
 * @brief Calculation of the generalized cauchy point.
 * @param x Current x value.
 * @param g Current gradient.
 * @param storage Storage for calculation of matrix products.
 * @param lb Lower bounds.
 * @param ub Upper bounds.
 *
 * This algorithm was described in
 *
 * Byrd, R.H., Lu, P., Nocedal, J., Zhu, C.
 * A Limited Memory Algorithm for Bound Constrained Optimization.
 * 1995. SIAM Journal of Scientific and Statistical Computing. 16(5). 1190-1208.
 */
template<
  typename Scalar,
  typename OutputFunction,
  template<typename, typename> class Storage>
Vector<Scalar> cauchy_point(
    const Vector<Scalar>& x,
    const Vector<Scalar>& g,
    Storage<Scalar, OutputFunction>& storage,
    const Vector<Scalar>& lb,
    const Vector<Scalar>& ub,
    const Scalar& machine_epsilon)
{
  const Eigen::Index n = x.size();
  Vector<Scalar> t = Vector<Scalar>::Zero(n);
  Vector<Scalar> d = Vector<Scalar>::Zero(n);

  for (Eigen::Index i = 0; i < n; ++i) {
    if (g(i) < Scalar{0}) {
      t(i) = (x(i) - ub(i)) / g(i);
    }
    else if (g(i) > Scalar{0}) {
      t(i) = (x(i) - lb(i)) / g(i);
    }
    else {
      t(i) = scalar_traits<Scalar>::infinity();
    }
    if (t(i) == Scalar{0}) {
      d(i) = Scalar{0};
    }
    else {
      d(i) = -g(i);
    }
  }

  std::vector<std::pair<Eigen::Index, Scalar>> t_values;
  for (Eigen::Index i = 0; i < n; ++i) {
    t_values.push_back({i, t(i)});
  }

  std::sort(t_values.begin(), t_values.end(),
      [] (const auto& p1, const auto& p2) {return p1.second < p2.second;});

  auto first_non_zero = std::find_if(t_values.begin(), t_values.end(),
      [] (const auto& p) {return p.second != Scalar{0};});

  // if there is no active bound => return initial point
  if (first_non_zero == t_values.end()) {
    return x;
  }

  // initialize values
  Scalar df = -d.dot(d);
  Scalar d2f = storage.calculate_vBv(d);;
  Scalar d2f_init = d2f;
  Scalar dt_min = - df / d2f;
  Scalar t_old {0};

  Vector<Scalar> z = Vector<Scalar>::Zero(n);

  Vector<Scalar> x_cp = x;

  auto it = first_non_zero;

  // first_non_zero is now guaranteed to be valid
  Eigen::Index b = it->first;
  Scalar tv = it->second;
  Scalar dt = tv;

  while (dt_min >= dt && it != t_values.end()) {
    if (d(b) > Scalar{0}) {
      x_cp(b) = ub(b);
    }
    else if (d(b) < Scalar{0}) {
      x_cp(b) = lb(b);
    }
    else {
      // this can't happen (see above)
      x_cp(b) = scalar_traits<Scalar>::nan();
    }

    // update z^(j-1) => z^(j)
    // z^(j) = z^(j-1) + dt^(j-1)*d^(j-1)
    z = z + dt * d;

    // update d^(j-1) => d(j)
    // => zero out d(b)
    d(b) = Scalar{0};

    // update f'^(j-1) => f'^(j)
    // f'^(j) = g' * d^(j) + d(j)' * B * z^(j)
    df = g.dot(d) + d.dot(storage.calculate_Bv(z));

    // update f''^(j-1) => f''^(j)
    // f''^(j) = d^(j)' * B * d^(j)
    d2f = storage.calculate_vBv(d);

    // here we need the machine epsilon:
    // if d2f were 0.0 the following update would fail, so
    // we replace it with a very small value to be able to continue
    d2f = std::max<Scalar>(d2f_init * machine_epsilon, d2f);

    // update dt_min^(j-1) => dt_min^(j)
    dt_min = - df/d2f;

    // update t_old^(j) => t_old^(j + 1)
    t_old = tv;
    ++it;
    if (it != t_values.end()) {
      b = it->first;
      tv = it->second;

      // update dt^(j-1) => dt^(j)
      dt = tv - t_old;
    }
    else {
      dt_min = Scalar{0};
    }
  }

  dt_min = std::max<Scalar>(dt_min, Scalar{0});
  t_old = t_old + dt_min;

  for (Eigen::Index i = 0; i < n; ++i) {
    if (t(i) >= tv) {
      x_cp(i) = x(i) + t_old * d(i);
    }
  }

  return x_cp;
}

/**
 * @brief Calculation of the minimizing subspace direction for BFGS.
 * @param x Current x value.
 * @param g Current gradient.
 * @param x_cp Generalized cauchy point.
 * @param Z The projection matrix defining the free variables.
 * @param storage The BFGS storage for matrix products.
 * @returns Minimizing direction.
 *
 * Runtime \Theta(n^3) because of the solution of the linear system.
 *
 * @warning
 * If the solution of linear equation is unstable, a NaN vector
 * is returned that should trigger a total restart!
 *
 * This is described in
 *
 * Byrd, R.H., Lu, P., Nocedal, J., Zhu, C.
 * A Limited Memory Algorithm for Bound Constrained Optimization.
 * 1995. SIAM Journal of Scientific and Statistical Computing. 16(5). 1190-1208.
 */
template<typename Scalar, typename OutputFunction>
Vector<Scalar> subspace_min_direction(
    const Vector<Scalar>& x,
    const Vector<Scalar>& g,
    const Vector<Scalar>& x_cp,
    const Matrix<Scalar>& Z,
    BFGSStorage<Scalar, OutputFunction>& storage,
    const Scalar& check_epsilon)
{
  Vector<Scalar> r_c = Z.transpose() * (g + storage.calculate_Bv(x_cp - x));
  Matrix<Scalar> rB = Z.transpose() * storage.calculate_B() * Z;

  Vector<Scalar> d = -Eigen::FullPivLU<Matrix<Scalar>>(rB).solve(r_c);
  Vector<Scalar> rBd = -rB * d;

  /**
   * @todo
   * This is a step that can potentially suffer from (and also uncover)
   * numerical instabilities and problems, like indefinite matrices
   * (i.e. the initial matrix could have very small negative or very small
   *  complex eigenvalues, that should not be there).
   * It should be checked if this just happens or if there is some
   * subtle bug.
   *
   * If the error is to large, we return NaN that triggers
   * an optimization restart.
   */
  Scalar relative_error {0};
  // if the absolute error is != 0.0, than one of the values is != 0.0
  if (r_c.norm() != Scalar{0}) {
    relative_error = (rBd - r_c).norm() / r_c.norm();
  }
  else if (rBd.norm() != Scalar{0}) {
    relative_error = (rBd - r_c).norm() / rBd.norm();
  }

  if (relative_error > check_epsilon) {
    return Vector<Scalar>::Constant(x.size(), scalar_traits<Scalar>::nan());
  }

  return d;
}

/**
 * @brief Calculation of the miminizing subspace direction for BFGS.
 * @param x Current x value.
 * @param g Current gradient.
 * @param x_cp Generalized cauchy point.
 * @param Z The projection matrix defining the free variables.
 * @param storage The L-BFGS storage for matrix products.
 * @returns Minimizing direction.
 *
 * Runtime \Theta(mn + m^3) because of the solution of the linear system.
 *
 * @warning
 * If the solution of linear equation is unstable, a NaN vector
 * is returned that should trigger a total restart!
 *
 * This is described in
 *
 * Byrd, R.H., Lu, P., Nocedal, J., Zhu, C.
 * A Limited Memory Algorithm for Bound Constrained Optimization.
 * 1995. SIAM Journal of Scientific and Statistical Computing. 16(5). 1190-1208.
 */
template<typename Scalar, typename OutputFunction>
Vector<Scalar> subspace_min_direction(
    const Vector<Scalar>& x,
    const Vector<Scalar>& g,
    const Vector<Scalar>& x_cp,
    const Matrix<Scalar>& Z,
    LBFGSStorage<Scalar, OutputFunction>& storage,
    const Scalar& check_epsilon)
{
  // reduced gradient at the cauchy point
  Vector<Scalar> r_c = Z.transpose() * (g + storage.calculate_Bv(x_cp - x));

  Vector<Scalar> d = storage.gamma * r_c;

  // if we are in the initial phase the calculations are not only unnecessary
  // but actually fail => skip them!
  if (storage.M.size() != 0) {

    Vector<Scalar> v0 = storage.W.transpose() * Z * r_c;
    v0 = storage.M * v0;

    Matrix<Scalar> N = storage.gamma * storage.W.transpose() * Z * Z.transpose() * storage.W;
    N = Matrix<Scalar>::Identity(N.rows(), N.rows()) - storage.M * N;

    Vector<Scalar> v = Eigen::FullPivLU<Matrix<Scalar>>(N).solve(v0);

    Vector<Scalar> Nv = N * v;

    /**
     * @todo
     * This is a step that can potentially suffer from (and also uncover)
     * numerical instabilities and problems, like indefinite matrices
     * (i.e. the initial matrix could have very small negative or very small
     *  complex eigenvalues, that should not be there).
     * It should be checked if this just happens or if there is some
     * subtle bug.
     *
     * If the error is to large, we return NaN that triggers
     * an optimization restart.
     */
    Scalar relative_error{0};
    // if the absolute error is != 0.0, than one of the values is != 0.0
    if (v0.norm() != Scalar{0}) {
      relative_error = (Nv - v0).norm() / v0.norm();
    }
    else if (Nv.norm() != Scalar{0}) {
      relative_error = (Nv - v0).norm() / Nv.norm();
    }

    if (relative_error > check_epsilon) {
      return Vector<Scalar>::Constant(x.size(), scalar_traits<Scalar>::nan());
    }

    d = d + (storage.gamma * storage.gamma) * Z.transpose() * storage.W * v;
  }

  return -d;
}

/**
 * @brief Calculation of the subspace minimizer.
 * @param x Current x value.
 * @param g Current gradient.
 * @param x_cp Generalized cauchy point.
 * @param storage The storage for matrix products.
 * @param lb Lower bounds.
 * @param ub Upper bounds.
 * @returns Subspace minimum.
 *
 * Runtime depends on the used storage!
 *
 * @warning
 * If the solution of linear equation is unstable, a NaN vector
 * is returned that should trigger a total restart!
 *
 * This is described in
 *
 * Byrd, R.H., Lu, P., Nocedal, J., Zhu, C.
 * A Limited Memory Algorithm for Bound Constrained Optimization.
 * 1995. SIAM Journal of Scientific and Statistical Computing. 16(5). 1190-1208.
 *
 * and
 *
 * Morales, J.L, Nocedal, J.
 * L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for
 * large scale bound constrained optimization.
 * 2011. ACM Transactions on Mathematical Software. 38(1). 7:1-7:4.
 *
 */

template<
  typename Scalar,
  typename OutputFunction,
  template<typename, typename> class Storage>
Vector<Scalar> subspace_min(
    const Vector<Scalar>& x,
    const Vector<Scalar>& g,
    const Vector<Scalar>& x_cp,
    Storage<Scalar, OutputFunction>& storage,
    const Vector<Scalar>& lb,
    const Vector<Scalar>& ub,
    const Scalar& check_epsilon)
{
  const Eigen::Index n = x.size();

  // determine the fixed and free variables

  Vector<bool> fixed = (x_cp.array() <= lb.array() || x_cp.array() >= ub.array());
  unsigned n_free = fixed.rows() - fixed.sum();

  // if there is nothing left in the reduced space
  if (n_free == 0) {
    return x_cp;
  }

  // Z projects into the space of free variables, hence "subspace minimzation"
  Matrix<Scalar> Z = Matrix<Scalar>::Zero(n, n_free);
  for (Eigen::Index i = 0, k = 0; k < n; ++k) {
    if (!fixed(k)) {
      Z(k,i) = Scalar{1};
      ++i;
    }
  }

  // determine the minimizing direction depending on the storage
  Vector<Scalar> d = subspace_min_direction(x, g, x_cp, Z, storage, check_epsilon);

  // pass an NaN result if the subspace minimization failed
  if (d.unaryExpr(&scalar_traits<Scalar>::is_nan).any()) {
    return Vector<Scalar>::Constant(n, scalar_traits<Scalar>::nan());
  }

  /*
   * These are the extensions described in
   * Morales, J.L, Nocedal, J.
   * L-BFGS-B: Remark on Algorithm 778: L-BFGS-B, FORTRAN routines for
   * large scale bound constrained optimization.
   * 2011. ACM Transactions on Mathematical Software. 38(1). 7:1-7:4.
   *
   * First, we try an orthogonal projection. If that fails
   * (i.e., if the search direction points in the opposite direction
   * of the gradient) we use the backtracking approach as fallback
   */

  Vector<Scalar> ssm = x_cp;

  // try orthogonal projection
  for (Eigen::Index i = 0, k = 0; k < n; ++k) {
    if (!fixed(k)) {
      ssm(k) = std::max<Scalar>(ssm(k) + d(i), lb(k));
      ssm(k) = std::min<Scalar>(ssm(k), ub(k));
      ++i;
    }
  }

  Scalar m = (ssm - x).dot(g);

  if (m > Scalar{0}) {
    // moving in current direction leads uphill, try the backtracking approach

    ssm = x_cp;

    Eigen::Index i_bound = std::numeric_limits<Eigen::Index>::max();
    Eigen::Index k_bound = std::numeric_limits<Eigen::Index>::max();

    Scalar alpha{1};
    Scalar temp1 = alpha;

    for (Eigen::Index i = 0, k = 0; k < n; ++k) {
      if (!fixed(k)) {
        Scalar dk = d(i);
        Scalar temp2;
        if (dk < Scalar{0}) {
          temp2 = lb(k) - ssm(k);
          if (temp2 >= Scalar{0}) {
            temp1 = Scalar{0};
          }
          else if (dk * alpha < temp2) {
            temp1 = temp2 / dk;
          }
        }
        else if (dk > Scalar{0}) {
          temp2 = ub(k) - ssm(k);
          if (temp2 <= Scalar{0}) {
            temp1 = Scalar{0};
          }
          else if (dk * alpha > temp2){
            temp1 = temp2 / dk;
          }
        }
        if (temp1 < alpha) {
          alpha = temp1;
          i_bound = i;
          k_bound = k;
        }
        ++i;
      }
    }

    // force the vector to lie on bounds!
    if (i_bound != std::numeric_limits<Eigen::Index>::max()) {
      Scalar dk = d(i_bound);
      if (dk > Scalar{0}) {
        ssm(k_bound) = ub(k_bound);
        d(i_bound) = Scalar{0};
      }
      else if (dk < Scalar{0}) {
        ssm(k_bound) = lb(k_bound);
        d(i_bound) = Scalar{0};
      }
    }

    for (Eigen::Index i = 0, k = 0; k < n; ++k) {
      if (!fixed(k)) {
        ssm(k) += alpha * d(i);
        ++i;
      }
    }
  }

  return ssm;
}

}

}
