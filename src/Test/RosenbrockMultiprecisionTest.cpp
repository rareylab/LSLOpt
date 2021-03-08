#include <memory>

#include <gtest/gtest.h>

#include "ExecuteOptimization.hpp"
#include "Multiprecision.hpp"
#include "Rosenbrock.hpp"


struct RosenbrockMultiprecisionTestRow {
  unsigned dim;
  bool limit;
  Testing::MultiprecisionVector x0;
  bool global_opt;
  int box;
  Testing::Algorithm algorithm;
};

class RosenbrockMultiprecisionTest : public ::testing::TestWithParam<RosenbrockMultiprecisionTestRow> {
  protected:
    void SetUp() override
    {
      rosenbrock =
          std::make_unique<Testing::Rosenbrock<Testing::Multiprecision>>(
              GetParam().dim, GetParam().limit, GetParam().box, false, false);
    }

    void TearDown() override
    {
      rosenbrock.reset();
    }

    std::unique_ptr<Testing::Rosenbrock<Testing::Multiprecision>> rosenbrock;
};

TEST_P(RosenbrockMultiprecisionTest, TestOptimization) {
  auto test_params = GetParam();

  LSLOpt::OptimizationParameters<Testing::Multiprecision> params
      = LSLOpt::getOptimizationParameters<Testing::Multiprecision>();

  std::stringstream ss;

  LSLOpt::OstreamOutput output{LSLOpt::OutputLevel::Warning, ss};

  auto result = Testing::execute_optimization(
      *rosenbrock, test_params.x0,
      params, test_params.algorithm,
      output);

  EXPECT_EQ(result.status, LSLOpt::Status::Success);
  EXPECT_FALSE(rosenbrock->error);

  for (std::string line; std::getline(ss, line);) {
    std::cerr << line << std::endl;
    // for the multiprecision test, there should be no warnings
    EXPECT_EQ(line.find("WARNING"), std::string::npos);
    EXPECT_EQ(line.find("ERROR"), std::string::npos);
  }

  if (test_params.global_opt) {
    EXPECT_TRUE(abs(result.function_value) < 1e-6);
  }
}

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterDefault, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::BFGS},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::L_BFGS},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::BFGS},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::L_BFGS},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::Zero(100u), true, 0, Testing::Algorithm::L_BFGS},

    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::BFGS},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::L_BFGS},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::BFGS},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), false, 0, Testing::Algorithm::L_BFGS},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::L_BFGS}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterLSLDefault, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterBoxDefault, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterBoxConstrainedDefault, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterBoxConstrained2Default, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterLSLBoxDefault, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterLSLBoxConstrainedDefault, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::LSL_L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterLSLBoxConstrained2Default, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::LSL_L_BFGS_B}
));


INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterLSL, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::LSL_BFGS},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::LSL_BFGS},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::LSL_L_BFGS},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::LSL_L_BFGS},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::Zero(100u), true, 0, Testing::Algorithm::LSL_L_BFGS},
    RosenbrockMultiprecisionTestRow{100u,  true, Testing::MultiprecisionVector::Zero(100u), true, 0, Testing::Algorithm::LSL_L_BFGS},

    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS},
    RosenbrockMultiprecisionTestRow{100u,  true, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterBox, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::Zero(100u), true, 0, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::BFGS_B},
    // this ends up in a local optimum
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), false, 0, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterBoxConstrained, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 1, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 1, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::Zero(100u), true, 1, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), true, 1, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterBoxConstrained2, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), false, 2, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), false, 2, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::Zero(100u), false, 2, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::BFGS_B},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), false, 2, Testing::Algorithm::L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterLSLBox, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::Zero(10u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::Zero(100u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{100u,  true, Testing::MultiprecisionVector::Zero(100u), true, 0, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_BFGS_B},
    // this ends up in a local optimum
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), false, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{100u,  true, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), true, 0, Testing::Algorithm::LSL_L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterLSLBoxConstrained, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::Zero(10u), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::Zero(10u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::Zero(100u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{100u,  true, Testing::MultiprecisionVector::Zero(100u), true, 1, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{100u,  true, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), true, 1, Testing::Algorithm::LSL_L_BFGS_B}
));

INSTANTIATE_TEST_CASE_P(RosenbrockMultiprecisionTesterLSLBoxConstrained2, RosenbrockMultiprecisionTest, ::testing::Values(
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::Zero(3u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::Zero(10u), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::Zero(10u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::Zero(10u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::Zero(100u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{100u,  true, Testing::MultiprecisionVector::Zero(100u), false, 2, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u, false, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{3u,  true, Testing::MultiprecisionVector::LinSpaced(3u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u, false, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{10u,  true, Testing::MultiprecisionVector::LinSpaced(10u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B},

    RosenbrockMultiprecisionTestRow{100u, false, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B},
    RosenbrockMultiprecisionTestRow{100u,  true, Testing::MultiprecisionVector::LinSpaced(100u, -42.0, 42.0), false, 2, Testing::Algorithm::LSL_L_BFGS_B}
));
