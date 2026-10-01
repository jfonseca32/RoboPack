#pragma once
#include "robopack/config.hpp"
namespace robopack {
struct Circle { Point center; double radius; };
Scan simulateScan(const Config& config,Pose robot,double stamp,
                  const std::vector<Circle>& obstacles = {});
std::vector<Range> simulateRanges(const Config& config,Point target,double stamp);
}
