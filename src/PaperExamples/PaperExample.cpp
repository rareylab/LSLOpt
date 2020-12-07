#include <iostream>
#include <Eigen/Dense>

#include "LSLOpt/BFGS.hpp"
#include "ModelSystem.hpp"


int main(int argc, char* argv[])
{
  ModelSystem modelSystem;

  // start with gamme = 67.5 deg
  Eigen::VectorXd x0 = Eigen::VectorXd::Constant(1, 3*M_PI/8);

  LSLOpt::OptimizationParameters<double> params
      = LSLOpt::getOptimizationParameters<double>();
  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Status, std::cout};

  auto result = LSLOpt::lsl_bfgs(modelSystem, x0, params, output);

  // access the optimal parameters
  std::cout << "gamma: " << result.x[0] << std::endl;
  // access the final function value
  std::cout << "score: " << result.function_value << std::endl;

  return 0;
}
