#pragma once

#include "LSLOpt/ScalarTraits.hpp"
#include "LSLOpt/Types.hpp"


namespace Testing {

template<typename Scalar>
struct Rosenbrock {
  Rosenbrock(unsigned n, bool limit, int box, bool wrong, bool check_acceptable_first)
  : n(n)
  , limit(limit)
  , box(box)
  , wrong(wrong)
  , check_acceptable_first(check_acceptable_first)
  {
  }

  Rosenbrock() = default;

  Scalar value(const LSLOpt::Vector<Scalar>& x)
  {
    if (!error && box > 0) {
      error |= ((x.array() - upper_bounds().array() > Scalar{0}).any());
      error |= ((x.array() - lower_bounds().array() < Scalar{0}).any());
    }

    Scalar result{0};

    for (unsigned i = 0; i < n-2; ++i) {
      result += Scalar{100} * (x(i+1) - x(i) * x(i)) * (x(i+1) - x(i) * x(i)) + (Scalar{1} - x(i)) * (Scalar{1} - x(i));
    }

    return result;
  }

  LSLOpt::Vector<Scalar> gradient(const LSLOpt::Vector<Scalar>& x)
  {
    if (!error && box > 0) {
      error |= ((x.array() - upper_bounds().array() > Scalar{0}).any());
      error |= ((x.array() - lower_bounds().array() < Scalar{0}).any());
    }

    LSLOpt::Vector<Scalar> gradient = LSLOpt::Vector<Scalar>::Zero(x.size());

    for (unsigned i = 0; i < n-1; ++i) {
      if (i < n - 2) {
        gradient(i) += -Scalar{400} * x(i) * (x(i+1) - x(i)*x(i)) - Scalar{2} * (Scalar{1} - x(i));
      }
      if (i > 0) {
        gradient(i) += Scalar{200} * (x(i) - x(i-1)*x(i-1));
      }
    }

    if (wrong) {
      return -gradient;
    }
    else {
      return gradient;
    }
  }

  Scalar initial_step_length(const LSLOpt::Vector<Scalar>& x, const LSLOpt::Vector<Scalar>& p)
  {
    if (limit) {
      Scalar max_coeff = p.array().abs().maxCoeff();
      return Scalar{1} / max_coeff;
    }
    else {
      return LSLOpt::scalar_traits<Scalar>::infinity();
    }
  }

  Scalar change_acceptable(const LSLOpt::Vector<Scalar>& x, const LSLOpt::Vector<Scalar>& xp)
  {
    if (limit) {
      return (xp - x).array().abs().maxCoeff() - (Scalar{1} + Scalar{1e-6});
    }
    else {
      return Scalar{0};
    }
  }

  bool is_check_acceptable_first()
  {
    return check_acceptable_first;
  }

  LSLOpt::Vector<Scalar> lower_bounds()
  {
    if (box == 1){
      return LSLOpt::Vector<Scalar>::Constant(n, -Scalar{3});
    }
    else if (box == 2) {
      return LSLOpt::Vector<Scalar>::Constant(n, Scalar{2});
    }
    return LSLOpt::Vector<Scalar>::Constant(n, -LSLOpt::scalar_traits<Scalar>::infinity());
  }

  LSLOpt::Vector<Scalar> upper_bounds()
  {
    if (box == 1){
      return LSLOpt::Vector<Scalar>::Constant(n, Scalar{3});
    }
    else if (box == 2) {
      return LSLOpt::Vector<Scalar>::Constant(n, Scalar{3});
    }
    return LSLOpt::Vector<Scalar>::Constant(n, LSLOpt::scalar_traits<Scalar>::infinity());
  }

  unsigned n = 2;
  bool limit = false;
  int box = 0;
  bool wrong = false;

  bool error = false;

  bool check_acceptable_first = true;
};

}
