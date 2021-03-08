#include <memory>

#include <gtest/gtest.h>

#include "ExecuteOptimization.hpp"


struct MinusInfinity {
    MinusInfinity(unsigned n)
  : n(n)
  {
  }

  double value(const LSLOpt::Vector<double>& x)
  {
    return -LSLOpt::scalar_traits<double>::infinity();
  }

  LSLOpt::Vector<double> gradient(const LSLOpt::Vector<double>& x)
  {
    return Eigen::VectorXd::Constant(n, 3);
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
};

struct MinusInfinityTestRow {
  unsigned dim;
  Eigen::VectorXd x0;
  Testing::Algorithm algorithm;
};

class MinusInfinityTest : public ::testing::TestWithParam<MinusInfinityTestRow> {
  protected:
    void SetUp() override
    {
      minus_infinity =
          std::make_unique<MinusInfinity>(GetParam().dim);
    }

    void TearDown() override
    {
      minus_infinity.reset();
    }

    std::unique_ptr<MinusInfinity> minus_infinity;
};

TEST_P(MinusInfinityTest, TestOptimization) {
  auto test_params = GetParam();

  LSLOpt::OptimizationParameters<double> params
      = LSLOpt::getOptimizationParameters<double>();

  std::stringstream ss;

  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Debug, ss};

  auto result = Testing::execute_optimization(
      *minus_infinity, test_params.x0,
      params, test_params.algorithm,
      output);

  EXPECT_EQ(result.status, LSLOpt::Status::MinusInfinity);

  for (std::string line; std::getline(ss, line);) {
    std::cerr << line << std::endl;
    EXPECT_EQ(line.find("ERROR"), std::string::npos);
  }

  EXPECT_TRUE(result.function_value == -LSLOpt::scalar_traits<double>::infinity());
}

INSTANTIATE_TEST_CASE_P(MinusInfinityTester, MinusInfinityTest, ::testing::Values(
    MinusInfinityTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::BFGS},
    MinusInfinityTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::L_BFGS},
    MinusInfinityTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::GD},

    MinusInfinityTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::BFGS},
    MinusInfinityTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::L_BFGS},

    MinusInfinityTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS},

    MinusInfinityTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::BFGS},
    MinusInfinityTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::L_BFGS},

    MinusInfinityTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::BFGS},
    MinusInfinityTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::L_BFGS},

    MinusInfinityTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::L_BFGS}
));

INSTANTIATE_TEST_CASE_P(MinusInfinityTesterLSL, MinusInfinityTest, ::testing::Values(
    MinusInfinityTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_BFGS},
    MinusInfinityTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_L_BFGS},
    MinusInfinityTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_GD},

    MinusInfinityTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_BFGS},
    MinusInfinityTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_L_BFGS},

    MinusInfinityTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS},

    MinusInfinityTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS},
    MinusInfinityTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS},

    MinusInfinityTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS},
    MinusInfinityTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS},

    MinusInfinityTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS}
));

INSTANTIATE_TEST_CASE_P(MinusInfinityTesterBox, MinusInfinityTest, ::testing::Values(
    MinusInfinityTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::BFGS_B},
    MinusInfinityTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::L_BFGS_B},

    MinusInfinityTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::BFGS_B},
    MinusInfinityTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::L_BFGS_B},

    MinusInfinityTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS_B},

    MinusInfinityTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::BFGS_B},
    MinusInfinityTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B},

    MinusInfinityTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::BFGS_B},
    MinusInfinityTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B},

    MinusInfinityTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(MinusInfinityTesterLSLBox, MinusInfinityTest, ::testing::Values(
    MinusInfinityTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_BFGS_B},
    MinusInfinityTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_L_BFGS_B},

    MinusInfinityTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_BFGS_B},
    MinusInfinityTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_L_BFGS_B},

    MinusInfinityTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS_B},

    MinusInfinityTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS_B},
    MinusInfinityTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B},

    MinusInfinityTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS_B},
    MinusInfinityTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B},

    MinusInfinityTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B}
));
