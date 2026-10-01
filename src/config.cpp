#include "robopack/config.hpp"
#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace robopack {
void Config::validate() const {
    for (double v : {wheelbase, footprint_radius, safety_margin, max_speed,
                    max_steering, steering_rate, acceleration, braking, reaction_time,
                    follow_distance, follow_deadband, person_radius, horizon, step,
                    sensor_timeout, max_sensor_skew, range_stddev_floor,
                    max_range_residual, max_target_stddev, max_target_range,
                    map_resolution, match_translation, match_rotation}) {
        if (!std::isfinite(v) || v <= 0)
            throw std::runtime_error("configuration values must be finite and positive");
    }
    for (double w : {weight_goal, weight_heading, weight_clearance, weight_speed, weight_smooth})
        if (!std::isfinite(w) || w < 0) throw std::runtime_error("invalid score weight");
    if (!finite(lidar_pose) || !std::isfinite(tag_height) || tag_height < 0 ||
        tag_id < 0 || wheelbase<0.05 || wheelbase>3 || footprint_radius<0.1 || footprint_radius>2 ||
        acceleration<0.05 || acceleration>3 || braking<0.05 || braking>3 ||
        steering_rate<0.1 || steering_rate>5 || sensor_timeout>0.25 || reaction_time>1 ||
        follow_distance>max_target_range || tag_height>3 || max_target_range>12 ||
        map_cells < 51 || map_cells > 1001 || map_cells%2 == 0 ||
        map_resolution < 0.02 || map_resolution > 0.5 ||
        max_steering > 0.6 || max_speed > 0.6 || step > 0.1 || step < 0.01 ||
        horizon < 0.5 || horizon > 5 || max_sensor_skew > sensor_timeout ||
        reaction_time < sensor_timeout || match_translation > 0.5 || match_rotation > 0.3 ||
        follow_distance-follow_deadband <= footprint_radius+safety_margin+person_radius ||
        !std::isfinite(min_match_score) || min_match_score <= 0 || min_match_score > 1 ||
        anchors.size() < 3 || anchors.size() > 16 || weight_goal <= 0 ||
        map_cells*map_resolution/2 < 12+footprint_radius)
        throw std::runtime_error("configuration violates geometry, sampling, or firmware limits");
    std::set<int> ids;
    for (const auto& a : anchors)
        if (a.id < 0 || !ids.insert(a.id).second || !finite({a.x,a.y,a.z}) || norm({a.x,a.y})>footprint_radius || a.z<0 || a.z>3)
            throw std::runtime_error("anchor IDs must be unique and coordinates finite");
    double area = 0;
    for (const auto& a : anchors) for (const auto& b : anchors) for (const auto& c : anchors)
        area = std::max(area, std::abs((b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x)));
    if (area < 1e-4) throw std::runtime_error("anchor positions are collinear or coincident");
}

Config Config::load(const std::string& path) {
    Config c;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config: "+path);
    std::map<std::string, double*> fields = {
        {"wheelbase",&c.wheelbase}, {"footprint_radius",&c.footprint_radius},
        {"safety_margin",&c.safety_margin}, {"max_speed",&c.max_speed},
        {"max_steering",&c.max_steering}, {"steering_rate",&c.steering_rate},
        {"acceleration",&c.acceleration}, {"braking",&c.braking}, {"reaction_time",&c.reaction_time},
        {"follow_distance",&c.follow_distance}, {"follow_deadband",&c.follow_deadband},
        {"person_radius",&c.person_radius}, {"horizon",&c.horizon}, {"step",&c.step},
        {"weight_goal",&c.weight_goal}, {"weight_heading",&c.weight_heading},
        {"weight_clearance",&c.weight_clearance}, {"weight_speed",&c.weight_speed},
        {"weight_smooth",&c.weight_smooth}, {"sensor_timeout",&c.sensor_timeout},
        {"max_sensor_skew",&c.max_sensor_skew}, {"range_stddev_floor",&c.range_stddev_floor},
        {"max_range_residual",&c.max_range_residual}, {"max_target_stddev",&c.max_target_stddev},
        {"max_target_range",&c.max_target_range}, {"tag_height",&c.tag_height},
        {"lidar_x",&c.lidar_pose.x}, {"lidar_y",&c.lidar_pose.y}, {"lidar_yaw",&c.lidar_pose.yaw},
        {"map_resolution",&c.map_resolution}, {"match_translation",&c.match_translation},
        {"match_rotation",&c.match_rotation}, {"min_match_score",&c.min_match_score}};
    std::set<std::string> seen;
    std::string line;
    bool have_anchors = false;
    while (std::getline(in,line)) {
        line = line.substr(0,line.find('#'));
        std::istringstream row(line);
        std::string key, extra;
        if (!(row >> key)) continue;
        if (key == "anchor") {
            if (!have_anchors) { c.anchors.clear(); have_anchors = true; }
            Anchor a{};
            if (!(row >> a.id >> a.x >> a.y >> a.z)) throw std::runtime_error("invalid anchor row");
            c.anchors.push_back(a);
        } else {
            if (!seen.insert(key).second) throw std::runtime_error("duplicate config key: "+key);
            if (key == "tag_id") { if (!(row >> c.tag_id)) throw std::runtime_error("invalid tag_id"); }
            else if (key == "map_cells") { if (!(row >> c.map_cells)) throw std::runtime_error("invalid map_cells"); }
            else if (fields.count(key)) {
                if (!(row >> *fields.at(key))) throw std::runtime_error("invalid config value: "+key);
            } else throw std::runtime_error("unknown config key: "+key);
        }
        if (row >> extra) throw std::runtime_error("extra config token: "+extra);
    }
    c.validate();
    return c;
}
} // namespace robopack
