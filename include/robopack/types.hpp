#pragma once
#include <cmath>
#include <string>
#include <vector>

namespace robopack {
constexpr double pi = 3.14159265358979323846;
struct Point { double x = 0, y = 0; };
struct Pose { double x = 0, y = 0, yaw = 0; };
inline double wrap(double a) { return std::remainder(a, 2*pi); }
inline double norm(Point p) { return std::hypot(p.x, p.y); }
inline Point operator+(Point a, Point b) { return {a.x+b.x, a.y+b.y}; }
inline Point operator-(Point a, Point b) { return {a.x-b.x, a.y-b.y}; }
inline Point operator*(Point a, double s) { return {a.x*s, a.y*s}; }
inline Point transform(Pose p, Point q) {
    return {p.x+std::cos(p.yaw)*q.x-std::sin(p.yaw)*q.y,
            p.y+std::sin(p.yaw)*q.x+std::cos(p.yaw)*q.y};
}
inline Point inverseTransform(Pose p, Point q) {
    return transform({0, 0, -p.yaw}, {q.x-p.x, q.y-p.y});
}
inline Pose compose(Pose a, Pose b) {
    const auto q = transform(a, {b.x, b.y});
    return {q.x, q.y, wrap(a.yaw+b.yaw)};
}
inline bool finite(Pose p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.yaw);
}
inline bool fresh(double stamp, double now, double age) {
    return std::isfinite(stamp) && std::isfinite(now) && stamp >= 0 &&
           now >= stamp && now-stamp <= age;
}
// SI units. Timestamps share one monotonic clock. x forward, y left, z up.
struct Scan {
    double stamp = 0, angle_min = -pi, angle_step = pi/180;
    double range_min = 0.05, range_max = 12;
    // 0 = invalid/unknown; range_max = a VERIFIED clear ray to that distance.
    // A C1 zero/low-quality return is invalid, never evidence of free space.
    std::vector<double> ranges;
};
struct Anchor { int id; double x, y, z; };
struct Range {
    int anchor_id = 0, tag_id = 0;
    double stamp = 0, distance = 0, stddev = 0;
    bool line_of_sight = false;
};
struct Odometry {
    double stamp = 0;
    Pose pose; // continuous odom -> rear axle, measured by encoders/IMU
    double speed = 0, steering = 0;
};
struct Frame {
    double now = 0;
    Scan scan;
    Odometry odom;
    std::vector<Range> ranges;
    bool estop = false;
};
struct Target {
    bool valid = false;
    Point position; // base at measurement time
    double stamp = 0, residual = 0, position_stddev = 0;
    std::string reason = "no_ranges";
};
struct Command {
    double speed = 0, steering = 0, score = 0, clearance = 0;
    std::string reason = "stopped";
};
} // namespace robopack
