#pragma once

#include <cmath>
#include <limits>


namespace LSLOpt {

/**
 * @brief Traits for scalar values.
 *
 * Needs:
 * `epsilon`, `infinity`, `nan`, `is_nan`
 */
template<typename Scalar>
struct scalar_traits {

};

/// @brief Scalar for `double` values.
template<>
struct scalar_traits<double> {
    /// @brief Machine epsilon for `double`.
    static constexpr double epsilon() {return std::numeric_limits<double>::epsilon();}
    /// @brief Infinity value for `double`.
    static constexpr double infinity() {return std::numeric_limits<double>::infinity();}
    /// @brief NaN value for `double`.
    static constexpr double nan() {return std::numeric_limits<double>::quiet_NaN();}

    /**
     * @brief Check if value is NaN
     * @param d Value for NaN checking.
     */
    static bool is_nan(const double& d) {return std::isnan(d);}
};

}
