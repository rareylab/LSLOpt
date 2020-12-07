#include <iostream>
#include <iomanip>

#include <Eigen/Dense>

#include "LSLOpt/BFGS.hpp"

#include "ModelSystem.hpp"


int main(int argc, char* argv[])
{
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

    double val = modelSystem.value(x0);

    std::cerr << angle << ";" << x << ";" << y << ";" << val << std::endl;
  }
  return 0;
}
