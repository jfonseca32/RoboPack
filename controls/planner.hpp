#pragma once
#include "robopack/mapping.hpp"
namespace robopack {
class Planner {
public:
    explicit Planner(Config config) : config_(std::move(config)) { config_.validate(); }
    Command plan(const Scan& scan, const Target& target, const Odometry& odom, double dt) const;
private:
    Config config_;
};
}
