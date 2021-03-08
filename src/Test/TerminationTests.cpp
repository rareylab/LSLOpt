#include <memory>

#include <gtest/gtest.h>

#include "ExecuteOptimization.hpp"
#include "Rosenbrock.hpp"


struct TerminationTestRow {
  unsigned dim;
  bool limit;
  Eigen::VectorXd x0;
  int box;
  Testing::Algorithm algorithm;
};

class TerminationTest : public ::testing::TestWithParam<TerminationTestRow> {
  protected:
    void SetUp() override
    {
      rosenbrock =
          std::make_unique<Testing::Rosenbrock<double>>(
              GetParam().dim, GetParam().limit, GetParam().box, false,
              true);
    }

    void TearDown() override
    {
      rosenbrock.reset();
    }

    std::unique_ptr<Testing::Rosenbrock<double>> rosenbrock;
};

TEST_P(TerminationTest, TestOptimization) {
  auto test_params = GetParam();

  LSLOpt::OptimizationParameters<double> params
      = LSLOpt::getOptimizationParameters<double>();

  {
  std::stringstream ss;
  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Status, ss};

  // check the termination by iterations
  LSLOpt::OptimizationParameters<double> this_params = params;
  // small number of iterations
  this_params.max_iterations = 1;
  auto result_it = Testing::execute_optimization(
      *rosenbrock, test_params.x0,
      this_params, test_params.algorithm,
      output);

  EXPECT_EQ(result_it.status, LSLOpt::Status::MaxIterations);
  EXPECT_FALSE(rosenbrock->error);

  for (std::string line; std::getline(ss, line);) {
    std::cerr << line << std::endl;
    EXPECT_EQ(line.find("ERROR"), std::string::npos);
  }
  // we need a new ss after that because std::getline drives it into EOF
  }

  rosenbrock->error = false;

  {
  std::stringstream ss;
  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Status, ss};

  // check the termination by minimum value
  LSLOpt::OptimizationParameters<double> this_params = params;
  // arbitrary minimum value
  this_params.min_value = 10.0;
  auto result_mv = Testing::execute_optimization(
      *rosenbrock, test_params.x0,
      this_params, test_params.algorithm,
      output);

  EXPECT_EQ(result_mv.status, LSLOpt::Status::MinValue);
  EXPECT_FALSE(rosenbrock->error);

  for (std::string line; std::getline(ss, line);) {
    std::cerr << line << std::endl;
    EXPECT_EQ(line.find("ERROR"), std::string::npos);
  }

  EXPECT_TRUE(result_mv.function_value <= this_params.min_value);
  }

  rosenbrock->error = false;

  {
  std::stringstream ss;
  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Status, ss};

  // check the termination by minimum change
  LSLOpt::OptimizationParameters<double> this_params = params;
  // set to termination if less than 50% change over 2 iterations
  // (of course an extreme value, but we want to force it)
  this_params.min_change = 0.5;
  auto result_mc = Testing::execute_optimization(
      *rosenbrock, test_params.x0,
      this_params, test_params.algorithm,
      output);

  EXPECT_EQ(result_mc.status, LSLOpt::Status::MinChange);
  EXPECT_FALSE(rosenbrock->error);

  for (std::string line; std::getline(ss, line);) {
    std::cerr << line << std::endl;
    EXPECT_EQ(line.find("ERROR"), std::string::npos);
  }
  }
}

INSTANTIATE_TEST_CASE_P(TerminationTester, TerminationTest, ::testing::Values(
    TerminationTestRow{3u, false, Eigen::VectorXd::Zero(3u), 0, Testing::Algorithm::BFGS},
    TerminationTestRow{3u, false, Eigen::VectorXd::Zero(3u), 0, Testing::Algorithm::L_BFGS},
    TerminationTestRow{3u, false, Eigen::VectorXd::Zero(3u), 0, Testing::Algorithm::GD},

    TerminationTestRow{10u, false, Eigen::VectorXd::Zero(10u), 0, Testing::Algorithm::BFGS},
    TerminationTestRow{10u, false, Eigen::VectorXd::Zero(10u), 0, Testing::Algorithm::L_BFGS}
));

INSTANTIATE_TEST_CASE_P(TerminationTesterBox, TerminationTest, ::testing::Values(
    TerminationTestRow{3u, false, Eigen::VectorXd::Zero(3u), 0, Testing::Algorithm::BFGS_B},
    TerminationTestRow{3u, false, Eigen::VectorXd::Zero(3u), 0, Testing::Algorithm::L_BFGS_B},

    TerminationTestRow{10u, false, Eigen::VectorXd::Zero(10u), 0, Testing::Algorithm::BFGS_B},
    TerminationTestRow{10u, false, Eigen::VectorXd::Zero(10u), 0, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(TerminationTesterBoxConstrained, TerminationTest, ::testing::Values(
    TerminationTestRow{3u, false, Eigen::VectorXd::Zero(3u), 1, Testing::Algorithm::BFGS_B},
    TerminationTestRow{3u, false, Eigen::VectorXd::Zero(3u), 1, Testing::Algorithm::L_BFGS_B},

    TerminationTestRow{10u, false, Eigen::VectorXd::Zero(10u), 1, Testing::Algorithm::BFGS_B},
    TerminationTestRow{10u, false, Eigen::VectorXd::Zero(10u), 1, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(TerminationTesterLSLBox, TerminationTest, ::testing::Values(
    TerminationTestRow{3u, false, Eigen::VectorXd::Zero(3u), 0, Testing::Algorithm::LSL_BFGS_B},
    TerminationTestRow{3u,  true, Eigen::VectorXd::Zero(3u), 0, Testing::Algorithm::LSL_BFGS_B},
    TerminationTestRow{3u, false, Eigen::VectorXd::Zero(3u), 0, Testing::Algorithm::LSL_L_BFGS_B},
    TerminationTestRow{3u,  true, Eigen::VectorXd::Zero(3u), 0, Testing::Algorithm::LSL_L_BFGS_B},

    TerminationTestRow{10u, false, Eigen::VectorXd::Zero(10u), 0, Testing::Algorithm::LSL_BFGS_B},
    TerminationTestRow{10u,  true, Eigen::VectorXd::Zero(10u), 0, Testing::Algorithm::LSL_BFGS_B},
    TerminationTestRow{10u, false, Eigen::VectorXd::Zero(10u), 0, Testing::Algorithm::LSL_L_BFGS_B},
    TerminationTestRow{10u,  true, Eigen::VectorXd::Zero(10u), 0, Testing::Algorithm::LSL_L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(TerminationTesterLSLBoxConstrained, TerminationTest, ::testing::Values(
    TerminationTestRow{3u, false, Eigen::VectorXd::Zero(3u), 1, Testing::Algorithm::LSL_BFGS_B},
    TerminationTestRow{3u,  true, Eigen::VectorXd::Zero(3u), 1, Testing::Algorithm::LSL_BFGS_B},
    TerminationTestRow{3u, false, Eigen::VectorXd::Zero(3u), 1, Testing::Algorithm::LSL_L_BFGS_B},
    TerminationTestRow{3u,  true, Eigen::VectorXd::Zero(3u), 1, Testing::Algorithm::LSL_L_BFGS_B},

    TerminationTestRow{10u, false, Eigen::VectorXd::Zero(10u), 1, Testing::Algorithm::LSL_BFGS_B},
    TerminationTestRow{10u,  true, Eigen::VectorXd::Zero(10u), 1, Testing::Algorithm::LSL_BFGS_B},
    TerminationTestRow{10u, false, Eigen::VectorXd::Zero(10u), 1, Testing::Algorithm::LSL_L_BFGS_B},
    TerminationTestRow{10u,  true, Eigen::VectorXd::Zero(10u), 1, Testing::Algorithm::LSL_L_BFGS_B}
));
