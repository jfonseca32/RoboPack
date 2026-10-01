#include "throttle.hpp"
#include <algorithm>
#include <cmath>
namespace robopack {
double stoppingDistance(double v, double braking, double reaction_time) {
    return v*reaction_time+v*v/(2*braking);
}
double approachSpeed(double error, double limit, double braking) {
    return std::min(limit,std::sqrt(2*braking*std::max(0.0,error)));
}
}
