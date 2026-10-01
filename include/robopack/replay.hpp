#pragma once
#include "robopack/types.hpp"
#include <iosfwd>
namespace robopack {
bool readFrame(std::istream& input, Frame& frame);
void writeFrame(std::ostream& output, const Frame& frame);
}
