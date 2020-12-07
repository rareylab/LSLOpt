#include "ModelSystem.hpp"


double ModelSystem::value(const Eigen::VectorXd& x) const
{
  double d = (to_cartesian(x) - x0).norm();
  return spline.eval<0>(d);
}


Eigen::VectorXd ModelSystem::gradient(const Eigen::VectorXd& x) const
{
  double d = (to_cartesian(x) - x0).norm();
  Eigen::VectorXd gradient_dd_dx = Eigen::VectorXd::Zero(1);
  if (d > 1e-6) {
    gradient_dd_dx[0] =
        1.0 / d * (2 * r * cos(x[0]) * (r * sin(x[0]) - x0[1])
                 - 2 * r * sin(x[0]) * (r * cos(x[0]) - x0[0]));
  }
  return gradient_dd_dx * spline.eval<1>(d);
}

double ModelSystem::initial_step_length(
    const Eigen::VectorXd& x, // the current set of parameters
    const Eigen::VectorXd& p  // the current search direction
  ) const
{
  // changes of more than Pi in any direction are meaningless
  return (std::abs(p[0]) > M_PI) ? M_PI / std::abs(p[0]) : 1.0;
}

double ModelSystem::change_acceptable(
      const Eigen::VectorXd& x_old, // the current set of parameters
      const Eigen::VectorXd& x_new  // the new set of parameters
  ) const
{
  double change = (to_cartesian(x_old) - to_cartesian(x_new)).norm();
  return (change > d_max) ? pow(change - d_max, 2) : 0.0;
}

Eigen::Vector2d ModelSystem::to_cartesian(const Eigen::VectorXd& x) const
{
  return Eigen::Vector2d {r * cos(x[0]), r * sin(x[0])};
}

ModelSystem::ModelSystem(const Eigen::Vector2d& x0)
  : x0(x0)
  , spline(4.0, 1.0, pow(2.0, 1.0/6.0), -0.1, 1.7)
  , r(x0.norm())
{
}
