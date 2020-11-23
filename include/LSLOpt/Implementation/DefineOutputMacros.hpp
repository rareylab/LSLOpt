#ifndef _LSL_OUTPUT_MACROS
#define _LSL_OUTPUT_MACROS

#include "../OutputUtils.hpp"

// This macro ensures that the output expression is only
// evaluated if the level in the output function is set accordingly.
#define LSL_OUTPUT(output_function, level, arg) \
  if (is_output_enabled(output_function, level)) { \
    output_function << level << arg << "\n"; \
  }

#endif
