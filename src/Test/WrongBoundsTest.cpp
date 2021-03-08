#include <memory>

#include <gtest/gtest.h>

#include "ExecuteOptimization.hpp"


struct WrongBounds {
    WrongBounds(unsigned n)
  : n(n)
  {
  }

  double value(const LSLOpt::Vector<double>& x)
  {
    return x.array().square().sum();
  }

  LSLOpt::Vector<double> gradient(const LSLOpt::Vector<double>& x)
  {
    return 2 * x;
  }

  double change_acceptable(const LSLOpt::Vector<double>&, const LSLOpt::Vector<double>&)
  {
    return 0.0;
  }

  double initial_step_length(const LSLOpt::Vector<double>&, const LSLOpt::Vector<double>&)
  {
    return LSLOpt::scalar_traits<double>::infinity();
  }

  LSLOpt::Vector<double> upper_bounds()
  {
    return Eigen::VectorXd::Constant(n, -1.0);
  }

  LSLOpt::Vector<double> lower_bounds()
  {
    // wrong: lower bound is larger than upper bound
    return Eigen::VectorXd::Constant(n, 1.0);
  }

  unsigned n;
};

struct WrongBoundsTestRow {
  unsigned dim;
  Eigen::VectorXd x0;
  Testing::Algorithm algorithm;
};

class WrongBoundsTest : public ::testing::TestWithParam<WrongBoundsTestRow> {
  protected:
    void SetUp() override
    {
      wrong_bounds =
          std::make_unique<WrongBounds>(GetParam().dim);
    }

    void TearDown() override
    {
      wrong_bounds.reset();
    }

    std::unique_ptr<WrongBounds> wrong_bounds;
};

TEST_P(WrongBoundsTest, TestOptimization) {
  auto test_params = GetParam();

  LSLOpt::OptimizationParameters<double> params
      = LSLOpt::getOptimizationParameters<double>();

  std::stringstream ss;

  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Debug, ss};

  auto result = Testing::execute_optimization(
      *wrong_bounds, test_params.x0,
      params, test_params.algorithm,
      output);

  bool error_somewhere = false;
  for (std::string line; std::getline(ss, line);) {
    error_somewhere |= (line.find("ERROR") != std::string::npos);
  }

  if (test_params.algorithm == Testing::Algorithm::BFGS_B
      || test_params.algorithm == Testing::Algorithm::LSL_BFGS_B
      || test_params.algorithm == Testing::Algorithm::L_BFGS_B
      || test_params.algorithm == Testing::Algorithm::LSL_L_BFGS_B) {
    EXPECT_EQ(result.status, LSLOpt::Status::InvalidBounds);
    EXPECT_EQ(error_somewhere, true);
  }
  else {
    // for the non-constrained variants the bounds are ignored
    EXPECT_EQ(result.status, LSLOpt::Status::Success);
    EXPECT_EQ(error_somewhere, false);
  }
}

INSTANTIATE_TEST_CASE_P(WrongBoundsTester, WrongBoundsTest, ::testing::Values(
    WrongBoundsTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::BFGS},
    WrongBoundsTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::L_BFGS},
    WrongBoundsTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::GD},

    WrongBoundsTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::BFGS},
    WrongBoundsTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::L_BFGS},

    WrongBoundsTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS},

    WrongBoundsTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::BFGS},
    WrongBoundsTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::L_BFGS},

    WrongBoundsTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::BFGS},
    WrongBoundsTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::L_BFGS},

    WrongBoundsTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::L_BFGS}
));

INSTANTIATE_TEST_CASE_P(WrongBoundsTesterLSL, WrongBoundsTest, ::testing::Values(
    WrongBoundsTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_BFGS},
    WrongBoundsTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_L_BFGS},
    WrongBoundsTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_GD},

    WrongBoundsTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_BFGS},
    WrongBoundsTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_L_BFGS},

    WrongBoundsTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS},

    WrongBoundsTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS},
    WrongBoundsTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS},

    WrongBoundsTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS},
    WrongBoundsTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS},

    WrongBoundsTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS}
));

INSTANTIATE_TEST_CASE_P(WrongBoundsTesterBox, WrongBoundsTest, ::testing::Values(
    WrongBoundsTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::BFGS_B},
    WrongBoundsTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::L_BFGS_B},

    WrongBoundsTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::BFGS_B},
    WrongBoundsTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::L_BFGS_B},

    WrongBoundsTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS_B},

    WrongBoundsTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::BFGS_B},
    WrongBoundsTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B},

    WrongBoundsTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::BFGS_B},
    WrongBoundsTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B},

    WrongBoundsTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(WrongBoundsTesterLSLBox, WrongBoundsTest, ::testing::Values(
    WrongBoundsTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_BFGS_B},
    WrongBoundsTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_L_BFGS_B},

    WrongBoundsTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_BFGS_B},
    WrongBoundsTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_L_BFGS_B},

    WrongBoundsTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS_B},

    WrongBoundsTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS_B},
    WrongBoundsTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B},

    WrongBoundsTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS_B},
    WrongBoundsTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B},

    WrongBoundsTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B}
));
