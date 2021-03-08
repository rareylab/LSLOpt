#include <iostream>
#include <iomanip>

#include <Eigen/Dense>

#include "LSLOpt/BFGS.hpp"

#include "ModelSystem.hpp"


int main(int argc, char* argv[])
{
  int mode;
  if (argc == 0) {
    mode = 0;
  }
  else {
    try {
      mode = std::stoi(argv[1]);
    }
    catch (...) {
      mode = 0;
    }
  }

  ModelSystem modelSystem;

  std::cerr << std::setprecision(16);

  double angle_eps = 0.1 * M_PI / 180.0;

  unsigned n_steps = static_cast<unsigned>((2 * M_PI) / angle_eps) + 1;

  for (unsigned i = 0; i < n_steps; ++i) {
    double angle = angle_eps * i;
    double x = modelSystem.x0.norm() * std::sin(angle);
    double y = modelSystem.x0.norm() * std::cos(angle);
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(1);
    x0[0] = angle;

    LSLOpt::OptimizationParameters<double> params
        = LSLOpt::getOptimizationParameters<double>();
    LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Status, std::cout};

    LSLOpt::OptimizationResult<double> result;
    if (mode == 0) {
      result = LSLOpt::lsl_bfgs(modelSystem, x0, params, output);
    }
    else {
      result = LSLOpt::bfgs(modelSystem, x0, params, output);
    }

    double dist = sqrt(pow(modelSystem.to_cartesian(result.x)[0] - modelSystem.to_cartesian(x0)[0], 2)
                     + pow(modelSystem.to_cartesian(result.x)[1] - modelSystem.to_cartesian(x0)[1], 2));
    std::cerr << angle << ";" << x << ";" << y << ";" << result.function_value << ";" << dist << std::endl;
  }

  return 0;
}
