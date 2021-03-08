#include <iostream>
#include <iomanip>

#include <Eigen/Dense>

#include "LSLOpt/BFGS.hpp"

#include "ModelSystem.hpp"


int main(int argc, char* argv[])
{
  ModelSystem modelSystem;

  std::cerr << std::setprecision(16);

  double max_x = 2.0;
  double step = 0.001;
  unsigned n = static_cast<unsigned>(max_x / step) + 1;

  for (unsigned i = 0; i < n; ++i) {
    double x = step * i;
    double v = modelSystem.spline.eval<0>(x);
    std::cerr << x << ";" << v << std::endl;
  }

  return 0;
}
