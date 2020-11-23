#include <memory>

#include <gtest/gtest.h>

#include "ExecuteOptimization.hpp"


struct WrongDerivative {
    WrongDerivative(unsigned n, unsigned error_type)
  : n(n)
  , error_type(error_type)
  {
  }

  double value(const LSLOpt::Vector<double>& x)
  {
    return x.array().sum();
  }

  LSLOpt::Vector<double> gradient(const LSLOpt::Vector<double>& x)
  {
    // some wrong derivative
    if (error_type == 0) {
      // this is in the wrong direction but small
      return -Eigen::VectorXd::Constant(n, 1);
    }
    else {
      // this is in the wrong direction and large
      return -Eigen::VectorXd::Constant(n, 1e10);
    }
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
    return Eigen::VectorXd::Constant(n, LSLOpt::scalar_traits<double>::infinity());
  }

  LSLOpt::Vector<double> lower_bounds()
  {
    return Eigen::VectorXd::Constant(n, -LSLOpt::scalar_traits<double>::infinity());
  }

  unsigned n;
  unsigned error_type;
};

struct WrongDerivativeTestRow {
  unsigned dim;
  Eigen::VectorXd x0;
  Testing::Algorithm algorithm;
  unsigned error_type = 0;
  bool allow_restarts = true;
};

class WrongDerivativeTest : public ::testing::TestWithParam<WrongDerivativeTestRow> {
  protected:
    void SetUp() override
    {
      wrong_derivative =
          std::make_unique<WrongDerivative>(GetParam().dim, GetParam().error_type);
    }

    void TearDown() override
    {
      wrong_derivative.reset();
    }

    std::unique_ptr<WrongDerivative> wrong_derivative;
};

TEST_P(WrongDerivativeTest, TestOptimization) {
  auto test_params = GetParam();

  LSLOpt::OptimizationParameters<double> params
      = LSLOpt::getOptimizationParameters<double>();
  params.allow_restarts = test_params.allow_restarts;

  std::stringstream ss;

  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Debug, ss};

  auto result = Testing::execute_optimization(
      *wrong_derivative, test_params.x0,
      params, test_params.algorithm,
      output);

  EXPECT_TRUE(result.status == LSLOpt::Status::LineSearchMaxIterations
           || result.status == LSLOpt::Status::LineSearchMinimum);

  bool error_somewhere = false;
  std::string last_line;
  for (std::string line; std::getline(ss, line);) {
    error_somewhere |= (line.find("ERROR") != std::string::npos);
    last_line = line;
  }

  if (error_somewhere) {
    EXPECT_EQ(last_line.substr(0, 10), "[ERROR   ]");
  }
  else {
    EXPECT_EQ(last_line.substr(0, 10), "[STATUS  ]");
  }

  EXPECT_EQ(error_somewhere, true);
}

INSTANTIATE_TEST_CASE_P(WrongDerivativeTester, WrongDerivativeTest, ::testing::Values(
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::BFGS},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::L_BFGS},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::GD},

    WrongDerivativeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::BFGS},
    WrongDerivativeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::L_BFGS},

    WrongDerivativeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS},
    WrongDerivativeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS, 1},

    WrongDerivativeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::BFGS},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::L_BFGS},

    WrongDerivativeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::BFGS},
    WrongDerivativeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::L_BFGS},

    WrongDerivativeTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::L_BFGS}
));

INSTANTIATE_TEST_CASE_P(WrongDerivativeTesterNoRestart, WrongDerivativeTest, ::testing::Values(
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::BFGS, 0, false},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::L_BFGS, 0, false},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::GD, 0, false},

    WrongDerivativeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::BFGS, 0, false},
    WrongDerivativeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::L_BFGS, 0, false},

    WrongDerivativeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS, 0, false},
    WrongDerivativeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS, 1, false},

    WrongDerivativeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::BFGS, 0, false},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::L_BFGS, 0, false},

    WrongDerivativeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::BFGS, 0, false},
    WrongDerivativeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::L_BFGS, 0, false},

    WrongDerivativeTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::L_BFGS, 0, false}
));

INSTANTIATE_TEST_CASE_P(WrongDerivativeTesterLSL, WrongDerivativeTest, ::testing::Values(
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_BFGS},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_L_BFGS},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_GD},

    WrongDerivativeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_BFGS},
    WrongDerivativeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_L_BFGS},

    WrongDerivativeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS},
    WrongDerivativeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS, 1},

    WrongDerivativeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS},

    WrongDerivativeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS},
    WrongDerivativeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS},

    WrongDerivativeTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS}
));

INSTANTIATE_TEST_CASE_P(WrongDerivativeTesterBox, WrongDerivativeTest, ::testing::Values(
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::BFGS_B},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::L_BFGS_B},

    WrongDerivativeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::BFGS_B},
    WrongDerivativeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::L_BFGS_B},

    WrongDerivativeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS_B},
    WrongDerivativeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS_B, 1},

    WrongDerivativeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::BFGS_B},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B},

    WrongDerivativeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::BFGS_B},
    WrongDerivativeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B},

    WrongDerivativeTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(WrongDerivativeTesterLSLBox, WrongDerivativeTest, ::testing::Values(
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_BFGS_B},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_L_BFGS_B},

    WrongDerivativeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_BFGS_B},
    WrongDerivativeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_L_BFGS_B},

    WrongDerivativeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS_B},
    WrongDerivativeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS_B, 1},

    WrongDerivativeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS_B},
    WrongDerivativeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B},

    WrongDerivativeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS_B},
    WrongDerivativeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B},

    WrongDerivativeTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B}
));
