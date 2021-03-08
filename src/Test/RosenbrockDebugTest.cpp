#include <memory>

#include <gtest/gtest.h>

#include "LSLOpt/DebugProblem.hpp"
#include "ExecuteOptimization.hpp"
#include "Rosenbrock.hpp"


struct RosenbrockDebugTestRow {
  unsigned dim;
  bool limit;
  Eigen::VectorXd x0;
  bool global_opt;
  int box;
  bool wrong;
  Testing::Algorithm algorithm;
};

struct ErrorHandler {
    bool error = false;
    void operator() (const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)
    {
      error = true;
    }
};

class RosenbrockDebugTest : public ::testing::TestWithParam<RosenbrockDebugTestRow> {
  protected:
    void SetUp() override
    {
      rosenbrock
          = std::make_unique<LSLOpt::DebugProblem<Testing::Rosenbrock<double>, double, ErrorHandler>>(
                error_handler, GetParam().dim, GetParam().limit,
                GetParam().box, GetParam().wrong, false);
    }

    void TearDown() override
    {
      rosenbrock.reset();
    }

    std::unique_ptr<LSLOpt::DebugProblem<Testing::Rosenbrock<double>, double, ErrorHandler>> rosenbrock;
    ErrorHandler error_handler;
};


TEST_P(RosenbrockDebugTest, TestOptimization) {
  auto test_params = GetParam();

  LSLOpt::OptimizationParameters<double> params
      = LSLOpt::getOptimizationParameters<double>();

  std::stringstream ss;

  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Status, ss};

  auto result = Testing::execute_optimization(
      *rosenbrock, test_params.x0,
      params, test_params.algorithm,
      output);

  if (!test_params.wrong) {
    EXPECT_EQ(result.status, LSLOpt::Status::Success);
    EXPECT_FALSE(rosenbrock->error);
    EXPECT_FALSE(error_handler.error);
    if (test_params.global_opt) {
      EXPECT_TRUE(std::fabs(result.function_value) < 1e-6);
    }
  }
  else {
    // even if the derivative is wrong, do not evaluate outside bounds
    EXPECT_FALSE(rosenbrock->error);
    // we must detect the errors
    EXPECT_TRUE(error_handler.error);
    // and it should consistently fail because derivative is totally off
    EXPECT_NE(result.status, LSLOpt::Status::Success);
    EXPECT_LT(static_cast<int>(result.status), static_cast<int>(LSLOpt::Status::Success));
  }
}

INSTANTIATE_TEST_CASE_P(RosenbrockDebugTester, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::BFGS},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_BFGS},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_BFGS},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::L_BFGS},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_L_BFGS},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_L_BFGS}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugWrongTester, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::BFGS},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_BFGS},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_BFGS},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::L_BFGS},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_L_BFGS},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_L_BFGS}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugTesterLSL, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_BFGS},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_BFGS},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_L_BFGS},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_L_BFGS}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugWrongTesterLSL, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_BFGS},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_BFGS},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_L_BFGS},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_L_BFGS}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugTesterBox, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::BFGS_B},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugWrongTesterBox, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::BFGS_B},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugTesterBoxConstrained, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1, false, Testing::Algorithm::BFGS_B},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1, false, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugWrongTesterBoxConstrained, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1,  true, Testing::Algorithm::BFGS_B},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1,  true, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugTesterBoxConstrained2, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), false, 2, false, Testing::Algorithm::BFGS_B},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), false, 2, false, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugTesterLSLBox, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0, false, Testing::Algorithm::LSL_L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugWrongTesterLSLBox, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 0,  true, Testing::Algorithm::LSL_L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugTesterLSLBoxConstrained, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1, false, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 1, false, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1, false, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 1, false, Testing::Algorithm::LSL_L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugWrongTesterLSLBoxConstrained, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1,  true, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 1,  true, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), true, 1,  true, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), true, 1,  true, Testing::Algorithm::LSL_L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockDebugTesterLSLBoxConstrained2, RosenbrockDebugTest, ::testing::Values(
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), false, 2, false, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), false, 2, false, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockDebugTestRow{3u, false, Eigen::VectorXd::Zero(3u), false, 2, false, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockDebugTestRow{3u,  true, Eigen::VectorXd::Zero(3u), false, 2, false, Testing::Algorithm::LSL_L_BFGS_B}
));
