#pragma once
#include "robopack/types.hpp"

namespace robopack {
struct Config {
    double wheelbase = 0.32, footprint_radius = 0.48, safety_margin = 0.12;
    double max_speed = 0.45, max_steering = 0.50, steering_rate = 0.9;
    double acceleration = 0.5, braking = 0.8, reaction_time = 0.25;
    double follow_distance = 1.5, follow_deadband = 0.15, person_radius = 0.3;
    double horizon = 2, step = 0.05;
    double weight_goal = 4, weight_heading = 0.6, weight_clearance = 0.3;
    double weight_speed = 0.4, weight_smooth = 0.15;
    double sensor_timeout = 0.25, max_sensor_skew = 0.05;
    double range_stddev_floor = 0.10, max_range_residual = 0.15;
    double max_target_stddev = 0.75, max_target_range = 6;
    double tag_height = 1;
    int tag_id = 1;
    // Three receivers PLUS a pocket tag.
    std::vector<Anchor> anchors = {
        {0, -0.10, -0.22, 0.35}, {1, -0.10, 0.22, 0.35},
        {2, 0.30, 0, 0.35}};
    Pose lidar_pose{0.15, 0, 0}; // rear axle -> C1 scan origin
    double map_resolution = 0.10;
    int map_cells = 401;
    double match_translation = 0.15, match_rotation = 0.08, min_match_score = 0.40;
    void validate() const;
    static Config load(const std::string& path);
};
} // namespace robopack
