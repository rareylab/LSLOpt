#pragma once

#include <ostream>


namespace LSLOpt {

/**
 * @brief The output level for filtering.
 */
enum class OutputLevel {
    Nothing, /// not output
    Error,   /// show fatal errors
    Warning, /// show warnings
    Status,  /// show status messages
    Debug,   /// show all messages
};

/**
 * @brief Output operator for textual level output.
 * @param os Output stream.
 * @param level The output leve.
 * @returns The output stream.
 */
inline std::ostream& operator<< (std::ostream& os, const OutputLevel& level)
{
  switch (level) {
    case OutputLevel::Error:
      os << "[ERROR   ] ";
      break;
    case OutputLevel::Warning:
      os << "[WARNING ] ";
      break;
    case OutputLevel::Status:
      os << "[STATUS  ] ";
      break;
    case OutputLevel::Debug:
      os << "[DEBUG   ] ";
      break;
    default:
      os << "[????????] ";
  }

  return os;
}

/**
 * @brief NoOutput struct.
 *
 * If this is used as output function,
 * nothing is ever shown.
 */
struct NoOutput {
    /**
     * @brief Output operator.
     * @param t Variable to output.
     * @returns This object.
     *
     * The output variable can be of any
     * type (except `void`) as it is not
     * used anyway.
     */
    template<typename T>
    NoOutput& operator<<(const T& t)
    {
      return *this;
    }

    /// The current output level. Set to Nothing.
    const OutputLevel output_level = OutputLevel::Nothing;
};

/**
 * @brief Class for the output to an `std::ostream`.
 *
 * The output level can be set, i.e. it can be specified
 * which messages are to be shown.
 */
struct OstreamOutput {
    /**
     * @brief Construct OstreamOutput.
     * @param output_level Level where output is starting.
     * @param os The `std::ostream`.
     */
    OstreamOutput(OutputLevel output_level, std::ostream& os)
    : output_level(output_level)
    , current_level(OutputLevel::Nothing)
    , os(os)
    {

    }

    /**
     * @brief Set and output the level.
     * @param level The output to set and output.
     */
    OstreamOutput& operator<<(const OutputLevel& level)
    {
      current_level = level;
      output(level);
      return *this;
    }

    /**
     * @brief Output variable to the output stream.
     * @param t The variable to output.
     *
     * Using this function any type can be output that
     * can be written to an `std::ostream`.
     */
    template<typename T>
    OstreamOutput& operator<<(const T& t)
    {
      output(t);
      return *this;
    }

    /// The current output level.
    OutputLevel output_level;

  private:
    template<typename T>
    void output(const T& t)
    {
      if (current_level != OutputLevel::Nothing &&
          static_cast<unsigned>(current_level) <= static_cast<unsigned>(output_level)) {
        os << t;
      }
    }

    OutputLevel current_level;
    std::ostream& os;
};

/**
 * Check if the output function is activated at this level.
 * This uses SFINAE, i.e. if the output_function does not provide
 * the `output_level` attribute, this function is not used.
 *
 * @param output_function The output function.
 * @param curr_level Current leve of message.
 */
template<typename OutputFunction>
bool is_output_enabled(
    OutputFunction& output_function,
    OutputLevel curr_level,
    decltype(output_function.output_level)* = nullptr)
{
  OutputLevel level = output_function.output_level;

  if (std::is_same<OutputFunction, NoOutput>::value) {
    return false;
  }

  return curr_level != OutputLevel::Nothing
      && static_cast<unsigned>(curr_level) <= static_cast<unsigned>(level);
}

/**
 * @brief Catch-all function.
 */
inline bool is_output_enabled(...)
{
  return true;
}

}
