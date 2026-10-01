#include "steering.hpp"
#include <algorithm>
namespace robopack {
double slew(double requested, double measured, double rate, double dt) {
    return std::clamp(requested,measured-rate*dt,measured+rate*dt);
}
Pose advanceBicycle(Pose p, double v, double steering, double wheelbase, double dt) {
    const double turn=v*std::tan(steering)/wheelbase;
    if (std::abs(turn)<1e-9) {
        p.x+=v*dt*std::cos(p.yaw); p.y+=v*dt*std::sin(p.yaw);
    } else {
        p.x+=v/turn*(std::sin(p.yaw+turn*dt)-std::sin(p.yaw));
        p.y-=v/turn*(std::cos(p.yaw+turn*dt)-std::cos(p.yaw));
    }
    p.yaw=wrap(p.yaw+turn*dt);
    return p;
}
}
