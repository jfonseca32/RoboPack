#pragma once
#include "robopack/ranging.hpp"
#include "planner.hpp"
namespace robopack {
struct Result { Command command; Target target; Localization localization; };
class Navigator {
public:
    explicit Navigator(Config config);
    Result update(const Frame& frame);
    const Grid& map() const { return slam_.map(); }
private:
    Config config_;
    RangingSolver ranging_;
    LocalSlam slam_;
    Planner planner_;
    double last_now_=-1, last_scan_=-1, last_odom_=-1, last_range_=-1;
};
}
