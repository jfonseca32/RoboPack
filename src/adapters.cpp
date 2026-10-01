#include "robopack/adapters.hpp"
#include <algorithm>
#include <sstream>
#include <stdexcept>
namespace robopack {
Scan normalizeC1(const std::vector<C1Sample>& samples,double stamp) {
    if (!std::isfinite(stamp) || stamp<0) throw std::runtime_error("invalid C1 timestamp");
    Scan scan;
    scan.stamp=stamp;
    scan.ranges.assign(360,0);
    // Match SLAMTEC ROS convention: angle = pi - SDK angle (CCW).
    // Nearest-bin assignment retains the nearest obstacle; never interpolate gaps.
    for (const auto& sample:samples) {
        const double range=sample.distance_q2_mm/4000.0;
        if ((sample.quality>>2)==0 || range<scan.range_min || range>=scan.range_max) continue;
        const double angle=wrap(pi-sample.angle_q14*pi/32768.0);
        const int bin=static_cast<int>(std::lround((angle-scan.angle_min)/scan.angle_step))%360;
        double& value=scan.ranges[static_cast<std::size_t>(bin)];
        if (value==0 || range<value) value=range;
    }
    return scan;
}
std::optional<Range> parseBu04Distance(const std::string& line,int anchor_id,int tag_id,
                                     double stamp,double stddev,bool los) {
    if (line.size()>128 || anchor_id<0 || tag_id<0 || !std::isfinite(stamp) || stamp<0 ||
        !std::isfinite(stddev) || stddev<=0 || stddev>1) return {};
    std::istringstream row(line);
    std::string prefix,extra;
    double distance=0;
    if (!(row>>prefix) || prefix!="distance:" || !(row>>distance) || row>>extra ||
        !std::isfinite(distance) || distance<=0 || distance>100) return {};
    return Range{anchor_id,tag_id,stamp,distance,stddev,los};
}
}
