#pragma once
#include "robopack/types.hpp"
namespace robopack {
Pose advanceBicycle(Pose pose, double speed, double steering, double wheelbase, double dt);
double slew(double requested, double measured, double rate, double dt);
}
