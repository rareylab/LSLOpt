#include <memory>

#include <gtest/gtest.h>

#include "ExecuteOptimization.hpp"


struct Unbounded {
    Unbounded(unsigned n)
  : n(n)
  {
  }

  double value(const LSLOpt::Vector<double>& x)
  {
    return x.array().sum();
  }

  LSLOpt::Vector<double> gradient(const LSLOpt::Vector<double>& x)
  {
    return Eigen::VectorXd::Constant(n, 1);
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

struct UnboundedTestRow {
  unsigned dim;
  Eigen::VectorXd x0;
  Testing::Algorithm algorithm;
};

class UnboundedTest : public ::testing::TestWithParam<UnboundedTestRow> {
  protected:
    void SetUp() override
    {
      unbounded =
          std::make_unique<Unbounded>(GetParam().dim);
    }

    void TearDown() override
    {
      unbounded.reset();
    }

    std::unique_ptr<Unbounded> unbounded;
};

TEST_P(UnboundedTest, TestOptimization) {
  auto test_params = GetParam();

  LSLOpt::OptimizationParameters<double> params
      = LSLOpt::getOptimizationParameters<double>();

  std::stringstream ss;

  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Debug, ss};

  auto result = Testing::execute_optimization(
      *unbounded, test_params.x0,
      params, test_params.algorithm,
      output);

  if (test_params.algorithm == Testing::Algorithm::GD
      || test_params.algorithm == Testing::Algorithm::LSL_GD) {
    EXPECT_EQ(result.status, LSLOpt::Status::MaxIterations);
  }
  else {
    // this succeeds only because of numerical reasons:
    // the termination criterion leads to a successful termination
    // because ||x|| gets exceedingly large
    EXPECT_EQ(result.status, LSLOpt::Status::Success);
  }
}

INSTANTIATE_TEST_CASE_P(UnboundedTester, UnboundedTest, ::testing::Values(
    UnboundedTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::BFGS},
    UnboundedTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::L_BFGS},
    UnboundedTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::GD},

    UnboundedTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::BFGS},
    UnboundedTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::L_BFGS},

    UnboundedTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS},

    UnboundedTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::BFGS},
    UnboundedTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::L_BFGS},

    UnboundedTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::BFGS},
    UnboundedTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::L_BFGS},

    UnboundedTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::L_BFGS}
));

INSTANTIATE_TEST_CASE_P(UnboundedTesterLSL, UnboundedTest, ::testing::Values(
    UnboundedTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_BFGS},
    UnboundedTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_L_BFGS},
    UnboundedTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_GD},

    UnboundedTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_BFGS},
    UnboundedTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_L_BFGS},

    UnboundedTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS},

    UnboundedTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS},
    UnboundedTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS},

    UnboundedTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS},
    UnboundedTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS},

    UnboundedTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS}
));

INSTANTIATE_TEST_CASE_P(UnboundedTesterBox, UnboundedTest, ::testing::Values(
    UnboundedTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::BFGS_B},
    UnboundedTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::L_BFGS_B},

    UnboundedTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::BFGS_B},
    UnboundedTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::L_BFGS_B},

    UnboundedTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS_B},

    UnboundedTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::BFGS_B},
    UnboundedTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B},

    UnboundedTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::BFGS_B},
    UnboundedTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B},

    UnboundedTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(UnboundedTesterLSLBox, UnboundedTest, ::testing::Values(
    UnboundedTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_BFGS_B},
    UnboundedTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_L_BFGS_B},

    UnboundedTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_BFGS_B},
    UnboundedTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_L_BFGS_B},

    UnboundedTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS_B},

    UnboundedTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS_B},
    UnboundedTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B},

    UnboundedTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS_B},
    UnboundedTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B},

    UnboundedTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B}
));
