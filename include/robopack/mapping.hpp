#pragma once
#include "robopack/config.hpp"
#include <cstdint>

namespace robopack {
bool validScan(const Scan& scan);
std::vector<Point> scanHits(const Scan& scan, Pose lidar_pose);
class Grid {
public:
    explicit Grid(const Config& config);
    int value(Point p) const; // 0 unknown/outside, negative free, positive occupied
    bool contains(Point p, double margin = 0) const;
    void integrate(const Scan& scan, Pose base, Pose lidar_pose);
    void clearUnknownFootprint(double radius);
    bool freeFootprint(Point p, double radius) const;
    double matchScore(Point p) const;
    void savePgm(const std::string& path) const;
    double resolution() const { return resolution_; }
private:
    int index(Point p) const;
    int cells_;
    double resolution_;
    std::vector<std::int8_t> data_;
};
struct Localization {
    bool valid = false;
    Pose pose;
    double match_score = 0;
    std::string reason = "uninitialized";
};
class LocalSlam {
public:
    explicit LocalSlam(Config config);
    Localization update(const Scan& scan, const Odometry& odom);
    const Grid& map() const { return map_; }
private:
    Config config_;
    Grid map_;
    bool initialized_ = false;
    Pose pose_, last_odom_;
    double last_stamp_ = -1;
};
} // namespace robopack
