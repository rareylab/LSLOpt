#pragma once

#include <ostream>


namespace LSLOpt {

/**
 * @brief Status of the optimizer
 *
 * Non-negative values indicate success or regular
 * termination.
 *
 * Negative values indicate failures.
 */
enum class Status : int {
    // general errors
    GeneralFailure           = -256, /// general failure
    // line search errors
    InvalidMaximumValue      = -129, /// no valid max alpha obtained
    LineSearchFailed         = -128, /// general line search failure
    LineSearchMinimum        = -127, /// step length too small
    LineSearchMaxIterations  = -126, /// maximum number of line search iterations reached
    WrongSearchDirection     = -125, /// optimizer going uphill, probably wrong gradient!
    InvalidBounds            = -124, /// the bounds are invalid

    // normal termination
    Converged                = 0,    /// optimization converged
    Success                  = 0,    /// optimization successful, the same as `Converged`
    // abnormal, but correct termination
    MaxIterations            = 1,    /// maximum number of iterations reached
    MinChange                = 2,    /// objective function doesn't change anymore
    MinusInfinity            = 3,    /// function reached minus infinity
    MinValue                 = 4,    /// function value reached stopping value
    XGradChange              = 5,    /// the gradient or the parameters don't change anymore

    // special values, should NEVER be observed outside of the implementation
    Running                  = 256   /// special value used during optimization, never returned!
};

/**
 * @brief Output of \ref Status on `std::ostream`.
 * @param os Output stream.
 * @param status Status to output.
 * @returns The output stream `os`.
 */
inline std::ostream& operator<< (std::ostream& os, const Status& status)
{
  os << "{ " << static_cast<int>(status) << "; ";
  switch (status) {
    case Status::GeneralFailure:
      os << "General failure";
      break;
    case Status::InvalidMaximumValue:
      os << "No valid maximum step length obtained";
      break;
    case Status::LineSearchFailed:
      os << "Line search: general failure";
      break;
    case Status::LineSearchMinimum:
      os << "Line search: minimum reached";
      break;
    case Status::LineSearchMaxIterations:
      os << "Line search: maximum iteration number reached";
      break;
    case Status::WrongSearchDirection:
      os << "Wrong search direction";
      break;
    case Status::InvalidBounds:
      os << "Invalid bounds";
      break;
    case Status::Success:
      os << "Success";
      break;
    case Status::MaxIterations:
      os << "Maximum iteration number reached";
      break;
    case Status::MinChange:
      os << "Minimum change of function value";
      break;
    case Status::MinusInfinity:
      os << "Function is unbounded below";
      break;
    case Status::MinValue:
      os << "Minimum function value reached";
      break;
    case Status::XGradChange:
      os << "Minimum change of parameters or gradient.";
      break;
    case Status::Running:
      os << "Running";
      break;
    default:
      os << "Unknown";
  }
  os << "}";
  return os;
}

}
