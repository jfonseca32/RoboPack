#include "robopack/navigation.hpp"
namespace robopack {
Navigator::Navigator(Config c) : config_(c),ranging_(c),slam_(c),planner_(c) {}
Result Navigator::update(const Frame& f) {
    Result result;
    auto stop=[&](const std::string& reason) { result.command.reason=reason; return result; };
    if (!std::isfinite(f.now) || f.now<0 || f.now<=last_now_) return stop("nonmonotonic_time");
    const double dt=last_now_<0 ? 0.1 : f.now-last_now_;
    last_now_=f.now;
    if (f.estop) return stop("emergency_stop");
    if (!fresh(f.scan.stamp,f.now,config_.sensor_timeout) ||
        !fresh(f.odom.stamp,f.now,config_.sensor_timeout)) return stop("stale_sensor");
    if (!validScan(f.scan) || !finite(f.odom.pose) || !std::isfinite(f.odom.speed) ||
        !std::isfinite(f.odom.steering)) return stop("invalid_sensor");
    if (f.scan.stamp<=last_scan_ || f.odom.stamp<=last_odom_) return stop("repeated_sensor");
    if (std::abs(f.scan.stamp-f.odom.stamp)>config_.max_sensor_skew) return stop("sensor_time_skew");
    if (dt>config_.sensor_timeout) return stop("control_gap");
    last_scan_=f.scan.stamp; last_odom_=f.odom.stamp;
    result.localization=slam_.update(f.scan,f.odom);
    if (!result.localization.valid) return stop(result.localization.reason);
    result.target=ranging_.solve(f.ranges,f.now);
    if (!result.target.valid) return stop(result.target.reason);
    if (result.target.stamp<=last_range_) return stop("repeated_ranges");
    if (std::abs(result.target.stamp-f.scan.stamp)>config_.max_sensor_skew) return stop("sensor_time_skew");
    last_range_=result.target.stamp;
    result.command=planner_.plan(f.scan,result.target,f.odom,dt);
    return result;
}
}
