#pragma once

#include <limits>

#include "Implementation/ProblemTraits.hpp"
#include "OptimizationStatus.hpp"
#include "ScalarTraits.hpp"
#include "Types.hpp"


namespace LSLOpt {

/**
 * @brief The optimization result.
 */
template<typename Scalar>
struct OptimizationResult {
    /// final function value
    Scalar function_value = Scalar{0};
    /// final set of parameters
    Vector<Scalar> x;
    /// final gradient
    Vector<Scalar> g;
    /// final gradient norm
    Scalar gradient_norm = Scalar{0};
    /// status of the optimization, see \ref Status
    Status status = Status::GeneralFailure;
    /// number of iterations performed
    unsigned iterations = 0;
    /// number of restarts
    unsigned nof_restarts = 0;
};

}
