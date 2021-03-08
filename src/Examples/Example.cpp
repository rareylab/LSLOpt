#include <iostream>

#include <Eigen/Dense>

#include "LSLOpt/BFGS.hpp"


struct Parabola {
    double value(const Eigen::VectorXd& x)
    {
      return x.dot(x);
    }

    Eigen::VectorXd gradient(const Eigen::VectorXd& x)
    {
      return 2 * x;
    }

    double initial_step_length(const Eigen::VectorXd& x, const Eigen::VectorXd& p)
    {
      double max_change = p.array().abs().maxCoeff();

      return 0.2 / max_change;
    }

    double change_acceptable(const Eigen::VectorXd& x, const Eigen::VectorXd& xp) {
      return (xp - x).array().abs().maxCoeff() - (0.2 + 1e-6);
    }
};


int main()
{
  Parabola parabola;
  Eigen::VectorXd x0 = Eigen::VectorXd::Constant(2, 1.0);
  LSLOpt::OptimizationParameters<double> params
      = LSLOpt::getOptimizationParameters<double>();
  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Status, std::cerr};

  auto result = LSLOpt::lsl_bfgs(parabola, x0, params, output);

  std::cout << "STATUS " << result.status << std::endl;
  std::cout << "F      " << result.function_value << std::endl;
  std::cout << "X      " << result.x << std::endl;

  return 0;
}
