#include "robopack/simulation.hpp"
#include <algorithm>
namespace robopack {
Scan simulateScan(const Config& c,Pose robot,double stamp,const std::vector<Circle>& obstacles) {
    Scan s; s.stamp=stamp; s.ranges.resize(360);
    const Pose lidar=compose(robot,c.lidar_pose);
    for (std::size_t i=0;i<s.ranges.size();++i) {
        const double angle=lidar.yaw+s.angle_min+i*s.angle_step;
        const Point d{std::cos(angle),std::sin(angle)};
        double r=s.range_max;
        for (double x : {-8.0,8.0}) {
            if (std::abs(d.x)>1e-9) {
                const double t=(x-lidar.x)/d.x, y=lidar.y+t*d.y;
                if (t>0 && std::abs(y)<=6) r=std::min(r,t);
            }
        }
        for (double y : {-6.0,6.0}) {
            if (std::abs(d.y)>1e-9) {
                const double t=(y-lidar.y)/d.y, x=lidar.x+t*d.x;
                if (t>0 && std::abs(x)<=8) r=std::min(r,t);
            }
        }
        for (const auto& circle:obstacles) {
            const Point q{lidar.x-circle.center.x,lidar.y-circle.center.y};
            const double b=q.x*d.x+q.y*d.y, disc=b*b-(q.x*q.x+q.y*q.y-circle.radius*circle.radius);
            if (disc>=0) {
                const double t=-b-std::sqrt(disc);
                if (t>0) r=std::min(r,t);
            }
        }
        s.ranges[i]=r>=s.range_min ? r : 0;
    }
    return s;
}
std::vector<Range> simulateRanges(const Config& c,Point target,double stamp) {
    std::vector<Range> ranges;
    for (const auto& a:c.anchors)
        ranges.push_back({a.id,c.tag_id,stamp,
            std::sqrt(std::pow(target.x-a.x,2)+std::pow(target.y-a.y,2)+std::pow(c.tag_height-a.z,2)),
            c.range_stddev_floor,true});
    return ranges;
}
}
