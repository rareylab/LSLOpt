#include <memory>

#include <gtest/gtest.h>

#include "ExecuteOptimization.hpp"


struct Reparameterize {
    Reparameterize(unsigned n)
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
    return Eigen::VectorXd::Constant(n, LSLOpt::scalar_traits<double>::infinity());
  }

  LSLOpt::Vector<double> lower_bounds()
  {
    return Eigen::VectorXd::Constant(n, -LSLOpt::scalar_traits<double>::infinity());
  }

  bool reparameterize(LSLOpt::Vector<double>& x, LSLOpt::Vector<double>& g)
  {
    x = -x;
    g = this->gradient(x);
    return true;
  }

  unsigned n;
};

struct ReparameterizeTestRow {
  unsigned dim;
  Eigen::VectorXd x0;
  Testing::Algorithm algorithm;
};

class ReparameterizeTest : public ::testing::TestWithParam<ReparameterizeTestRow> {
  protected:
    void SetUp() override
    {
      reparameterize =
          std::make_unique<Reparameterize>(GetParam().dim);
    }

    void TearDown() override
    {
      reparameterize.reset();
    }

    std::unique_ptr<Reparameterize> reparameterize;
};

TEST_P(ReparameterizeTest, TestOptimization) {
  auto test_params = GetParam();

  LSLOpt::OptimizationParameters<double> params
      = LSLOpt::getOptimizationParameters<double>();
  // demonstrate that this works without failures
  params.allow_restarts = false;

  std::stringstream ss;

  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Debug, ss};

  auto result = Testing::execute_optimization(
      *reparameterize, test_params.x0,
      params, test_params.algorithm,
      output);

  bool error_somewhere = false;
  for (std::string line; std::getline(ss, line);) {
    error_somewhere |= (line.find("ERROR") != std::string::npos);
  }

  EXPECT_EQ(result.status, LSLOpt::Status::Success);
  EXPECT_EQ(error_somewhere, false);
}

INSTANTIATE_TEST_CASE_P(ReparameterizeTester, ReparameterizeTest, ::testing::Values(
    ReparameterizeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::BFGS},
    ReparameterizeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::L_BFGS},
    ReparameterizeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::GD},

    ReparameterizeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::BFGS},
    ReparameterizeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::L_BFGS},

    ReparameterizeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS},

    ReparameterizeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::BFGS},
    ReparameterizeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::L_BFGS},

    ReparameterizeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::BFGS},
    ReparameterizeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::L_BFGS},

    ReparameterizeTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::L_BFGS}
));

INSTANTIATE_TEST_CASE_P(ReparameterizeTesterLSL, ReparameterizeTest, ::testing::Values(
    ReparameterizeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_BFGS},
    ReparameterizeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_L_BFGS},
    ReparameterizeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_GD},

    ReparameterizeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_BFGS},
    ReparameterizeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_L_BFGS},

    ReparameterizeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS},

    ReparameterizeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS},
    ReparameterizeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS},

    ReparameterizeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS},
    ReparameterizeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS},

    ReparameterizeTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS}
));

INSTANTIATE_TEST_CASE_P(ReparameterizeTesterBox, ReparameterizeTest, ::testing::Values(
    ReparameterizeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::BFGS_B},
    ReparameterizeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::L_BFGS_B},

    ReparameterizeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::BFGS_B},
    ReparameterizeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::L_BFGS_B},

    ReparameterizeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::L_BFGS_B},

    ReparameterizeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::BFGS_B},
    ReparameterizeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B},

    ReparameterizeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::BFGS_B},
    ReparameterizeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B},

    ReparameterizeTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(ReparameterizeTesterLSLBox, ReparameterizeTest, ::testing::Values(
    ReparameterizeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_BFGS_B},
    ReparameterizeTestRow{3u, Eigen::VectorXd::Zero(3u), Testing::Algorithm::LSL_L_BFGS_B},

    ReparameterizeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_BFGS_B},
    ReparameterizeTestRow{10u, Eigen::VectorXd::Zero(10u), Testing::Algorithm::LSL_L_BFGS_B},

    ReparameterizeTestRow{100u, Eigen::VectorXd::Zero(100u), Testing::Algorithm::LSL_L_BFGS_B},

    ReparameterizeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS_B},
    ReparameterizeTestRow{3u, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B},

    ReparameterizeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_BFGS_B},
    ReparameterizeTestRow{10u, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B},

    ReparameterizeTestRow{100u, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), Testing::Algorithm::LSL_L_BFGS_B}
));
