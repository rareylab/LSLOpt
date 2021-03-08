#include <memory>

#include <gtest/gtest.h>

#include "ExecuteOptimization.hpp"
#include "Rosenbrock.hpp"


struct RosenbrockTestRow {
  unsigned dim;
  bool limit;
  Eigen::VectorXd x0;
  bool global_opt;
  int box;
  Testing::Algorithm algorithm;
  bool check_acceptable_first = true;
  bool no_line_search = false;
};

class RosenbrockTest : public ::testing::TestWithParam<RosenbrockTestRow> {
  protected:
    void SetUp() override
    {
      rosenbrock =
          std::make_unique<Testing::Rosenbrock<double>>(
              GetParam().dim, GetParam().limit, GetParam().box, false,
              GetParam().check_acceptable_first);
    }

    void TearDown() override
    {
      rosenbrock.reset();
    }

    std::unique_ptr<Testing::Rosenbrock<double>> rosenbrock;
};

TEST_P(RosenbrockTest, TestOptimization) {
  auto test_params = GetParam();

  LSLOpt::OptimizationParameters<double> params
      = LSLOpt::getOptimizationParameters<double>();

  if (test_params.no_line_search) {
    // force some non-existing line search
    params.linesearch = static_cast<LSLOpt::Linesearch>(99);
  }

  std::stringstream ss;

  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Status, ss};

  auto result = Testing::execute_optimization(
      *rosenbrock, test_params.x0,
      params, test_params.algorithm,
      output);

  if (!test_params.no_line_search) {
    EXPECT_EQ(result.status, LSLOpt::Status::Success);
    EXPECT_FALSE(rosenbrock->error);

    for (std::string line; std::getline(ss, line);) {
      std::cerr << line << std::endl;
      EXPECT_EQ(line.find("ERROR"), std::string::npos);
    }

    if (test_params.global_opt) {
      EXPECT_TRUE(std::fabs(result.function_value) < 1e-6);
    }
  }
  else {
    EXPECT_EQ(result.status, LSLOpt::Status::GeneralFailure);
  }
}

INSTANTIATE_TEST_CASE_P(RosenbrockTester, RosenbrockTest, ::testing::Values(
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::BFGS},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::L_BFGS},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::BFGS},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::L_BFGS},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::Zero(100u), true, 0, Testing::Algorithm::L_BFGS},

    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::BFGS},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::L_BFGS},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::BFGS},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), false, 0, Testing::Algorithm::L_BFGS},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::L_BFGS}
));

// no line search, this leads to an error
INSTANTIATE_TEST_CASE_P(RosenbrockTesterNoLS, RosenbrockTest, ::testing::Values(
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::BFGS, true, true}
));

// this test inverts the order in which step length and first Wolfe condition are checked
INSTANTIATE_TEST_CASE_P(RosenbrockTesterLSLInvCheck, RosenbrockTest, ::testing::Values(
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS, false},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS, false},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS, false},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS, false},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::LSL_BFGS, false},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::LSL_BFGS, false},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::LSL_L_BFGS, false},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::LSL_L_BFGS, false},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::Zero(100u), true, 0, Testing::Algorithm::LSL_L_BFGS, false},
    RosenbrockTestRow{100u,  true, Eigen::VectorXd::Zero(100u), true, 0, Testing::Algorithm::LSL_L_BFGS, false},

    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS, false},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS, false},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS, false},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS, false},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS, false},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS, false},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS, false},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS, false},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS, false},
    RosenbrockTestRow{100u,  true, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS, false}
));

INSTANTIATE_TEST_CASE_P(RosenbrockTesterBox, RosenbrockTest, ::testing::Values(
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::BFGS_B},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::Zero(100u), true, 0, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::BFGS_B},
    // this ends up in a local optimum
    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), false, 0, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockTesterBoxConstrained, RosenbrockTest, ::testing::Values(
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1, Testing::Algorithm::BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 1, Testing::Algorithm::BFGS_B},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 1, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::Zero(100u), true, 1, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::BFGS_B},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), true, 1, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockTesterBoxConstrained2, RosenbrockTest, ::testing::Values(
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), false, 2, Testing::Algorithm::BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), false, 2, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), false, 2, Testing::Algorithm::BFGS_B},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), false, 2, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::Zero(100u), false, 2, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::BFGS_B},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), false, 2, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockTesterLSLBox, RosenbrockTest, ::testing::Values(
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::Zero(10u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::Zero(100u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{100u,  true, Eigen::VectorXd::Zero(100u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS_B},
    // this ends up in a local optimum
    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), false, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{100u,  true, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockTesterLSLBoxConstrained, RosenbrockTest, ::testing::Values(
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::Zero(10u), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::Zero(10u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::Zero(100u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{100u,  true, Eigen::VectorXd::Zero(100u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{100u,  true, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockTesterLSLBoxConstrained2, RosenbrockTest, ::testing::Values(
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::Zero(3u), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::Zero(3u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::Zero(3u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::Zero(10u), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::Zero(10u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::Zero(10u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::Zero(100u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{100u,  true, Eigen::VectorXd::Zero(100u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{3u, false, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{3u,  true, Eigen::VectorXd::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockTestRow{10u, false, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{10u,  true, Eigen::VectorXd::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockTestRow{100u, false, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockTestRow{100u,  true, Eigen::VectorXd::LinSpaced(100u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B}
));
