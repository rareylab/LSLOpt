#pragma once

#include <boost/multiprecision/cpp_dec_float.hpp>

#include "ExecuteOptimization.hpp"


namespace Testing {
using Multiprecision = boost::multiprecision::cpp_dec_float_100;
using MultiprecisionVector = LSLOpt::Vector<Multiprecision>;
}

namespace LSLOpt {

template<>
struct scalar_traits<Testing::Multiprecision> {
    // we use the double epsilon anyway!
    static Testing::Multiprecision epsilon() {return std::numeric_limits<double>::epsilon();}
    static Testing::Multiprecision infinity() {return std::numeric_limits<Testing::Multiprecision>::infinity();}
    static Testing::Multiprecision nan() {return std::numeric_limits<Testing::Multiprecision>::quiet_NaN();}

    static bool is_nan(const Testing::Multiprecision& d) {return !(d == d);}
};

template<>
OptimizationParameters<Testing::Multiprecision> getOptimizationParameters<Testing::Multiprecision>()
{
  OptimizationParameters<Testing::Multiprecision> params;
  params.machine_epsilon = scalar_traits<Testing::Multiprecision>::epsilon();
  params.gradient_tolerance = 1e-6;
  params.alpha_min = params.machine_epsilon;
  params.check_epsilon = 1e-10;
  params.min_change = 0.0;
  params.min_value = -scalar_traits<Testing::Multiprecision>::infinity();
  params.min_grad_param = params.machine_epsilon;

  return params;
}

}
