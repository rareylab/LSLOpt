#pragma once

#include "LSLOpt/BFGS.hpp"
#include "LSLOpt/BFGSB.hpp"
#include "LSLOpt/GD.hpp"
#include "LSLOpt/LBFGS.hpp"
#include "LSLOpt/LBFGSB.hpp"


namespace Testing {

enum class Algorithm {
    BFGS,
    LSL_BFGS,
    L_BFGS,
    LSL_L_BFGS,
    BFGS_B,
    LSL_BFGS_B,
    L_BFGS_B,
    LSL_L_BFGS_B,
    GD,
    LSL_GD
};

template<typename Problem,
typename Scalar,
typename OutputFunction>
LSLOpt::OptimizationResult<Scalar> execute_optimization(
    Problem& problem,
    const LSLOpt::Vector<Scalar>& x0,
    const LSLOpt::OptimizationParameters<Scalar>& params,
    Algorithm algorithm,
    OutputFunction& output_function)
{
  switch (algorithm) {
    case Algorithm::BFGS:
      return LSLOpt::bfgs(problem, x0, params, output_function);
    case Algorithm::LSL_BFGS:
      return LSLOpt::lsl_bfgs(problem, x0, params, output_function);
    case Algorithm::L_BFGS:
      return LSLOpt::lbfgs(problem, x0, params, output_function);
    case Algorithm::LSL_L_BFGS:
      return LSLOpt::lsl_lbfgs(problem, x0, params, output_function);
    case Algorithm::BFGS_B:
      return LSLOpt::bfgs_b(problem, x0, params, output_function);
    case Algorithm::LSL_BFGS_B:
      return LSLOpt::lsl_bfgs_b(problem, x0, params, output_function);
    case Algorithm::L_BFGS_B:
      return LSLOpt::lbfgs_b(problem, x0, params, output_function);
    case Algorithm::LSL_L_BFGS_B:
      return LSLOpt::lsl_lbfgs_b(problem, x0, params, output_function);
    case Algorithm::GD:
      return LSLOpt::gd(problem, x0, params, output_function);
    case Algorithm::LSL_GD:
      return LSLOpt::lsl_gd(problem, x0, params, output_function);
    default:
      return LSLOpt::OptimizationResult<Scalar>{};
  }
}

}
