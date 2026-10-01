#include "planner.hpp"
#include "steering.hpp"
#include "throttle.hpp"
#include <algorithm>
#include <limits>

namespace robopack {
Command Planner::plan(const Scan& scan, const Target& target, const Odometry& odom, double dt) const {
    Command stop;
    stop.reason="no_safe_trajectory";
    if (!target.valid) { stop.reason=target.reason; return stop; }
    if (!std::isfinite(target.position.x) || !std::isfinite(target.position.y) ||
        !std::isfinite(target.position_stddev) || target.position_stddev<0 ||
        target.position_stddev>config_.max_target_stddev) {
        stop.reason="target_uncertain"; return stop;
    }
    if (!validScan(scan) || !std::isfinite(dt) || dt<=0 || dt>config_.sensor_timeout ||
        !std::isfinite(odom.speed) || !std::isfinite(odom.steering) ||
        odom.speed<0 || odom.speed>config_.max_speed ||
        std::abs(odom.steering)>config_.max_steering) {
        stop.reason="invalid_planner_input"; return stop;
    }
    const double distance=norm(target.position);
    if (distance<=config_.follow_distance+config_.follow_deadband) {
        stop.reason="following_distance"; return stop;
    }
    if (target.position.x<=0) { stop.reason="target_behind"; return stop; }
    Grid local(config_);
    local.integrate(scan,{},config_.lidar_pose);
    // Only fill unknown cells already under the chassis; keep all measured hits.
    local.clearUnknownFootprint(config_.footprint_radius);
    const auto hits=scanHits(scan,config_.lidar_pose);
    Point goal=target.position*(1-config_.follow_distance/distance);
    const double radius=config_.footprint_radius+config_.safety_margin+
                        config_.max_speed*config_.step;
    auto clearanceAt=[&](Pose p) {
        double clearance=std::numeric_limits<double>::infinity();
        for (Point hit : hits) clearance=std::min(clearance,norm(hit-Point{p.x,p.y})-radius);
        // The radio target is ALSO an obstacle, even if its legs aren't in the scan.
        clearance=std::min(clearance,norm(target.position-Point{p.x,p.y})-radius-config_.person_radius);
        return clearance;
    };
    if (clearanceAt({})<=0 || !local.freeFootprint({},radius)) {
        stop.reason="obstacle_or_unknown_at_robot"; return stop;
    }
    // If the straight corridor to the standoff goal is blocked, choose a
    // visible local detour before scoring car-like trajectories. Without this,
    // direct attraction can prefer stopping in front of a passable obstacle.
    const double lookahead=std::min(norm(goal),1.8);
    auto visible=[&](Point endpoint) {
        const double length=norm(endpoint);
        const int steps=std::max(1,static_cast<int>(std::ceil(length/(config_.map_resolution*0.5))));
        for (int i=1;i<=steps;++i)
            if (!local.freeFootprint(endpoint*(static_cast<double>(i)/steps),radius)) return false;
        return true;
    };
    const Point direct=goal*(lookahead/norm(goal));
    if (!visible(direct)) {
        double best_detour=std::numeric_limits<double>::infinity();
        Point detour;
        for (int i=-15;i<=15;++i) {
            const double heading=i*pi/36; // forward hemisphere, 5 degree samples
            const Point candidate{lookahead*std::cos(heading),lookahead*std::sin(heading)};
            if (!visible(candidate)) continue;
            const double cost=norm(candidate-goal)+0.05*std::abs(heading-odom.steering);
            if (cost<best_detour) { best_detour=cost; detour=candidate; }
        }
        if (!std::isfinite(best_detour)) { stop.reason="no_visible_detour"; return stop; }
        goal=detour;
    }
    const double confidence_speed=config_.max_speed*std::clamp(
        1-target.position_stddev/config_.max_target_stddev,0.2,1.0);
    const double desired=approachSpeed(distance-config_.follow_distance,confidence_speed,config_.braking);
    Command best=stop;
    best.score=std::numeric_limits<double>::infinity();
    for (int vi=0; vi<=10; ++vi) for (int si=-6; si<=6; ++si) {
        const double raw_speed=desired*vi/10;
        const double v=std::clamp(raw_speed,std::max(0.0,odom.speed-config_.braking*dt),
                                  std::min(config_.max_speed,odom.speed+config_.acceleration*dt));
        const double target_steering=si*config_.max_steering/6;
        const double steering=slew(target_steering,odom.steering,config_.steering_rate,dt);
        Pose pose,endpoint;
        double min_clearance=clearanceAt(pose),sim_speed=odom.speed,sim_steering=odom.steering;
        bool safe=true;
        // Roll out terminal targets, but emit only the first reachable command.
        // Hold measured motion during latency, then obey calibrated actuator rates.
        // Every candidate must also have a clear braking tail after the horizon.
        const double brake_start=config_.reaction_time+config_.horizon;
        const double duration=brake_start+config_.max_speed/config_.braking+config_.step;
        for (double t=0; t<duration; t+=config_.step) {
            if (t>=config_.reaction_time) {
                const double wanted=t<brake_start ? raw_speed : 0;
                sim_speed=slew(wanted,sim_speed,wanted>sim_speed ? config_.acceleration : config_.braking,
                               config_.step);
                sim_steering=slew(target_steering,sim_steering,config_.steering_rate,config_.step);
            }
            pose=advanceBicycle(pose,sim_speed,sim_steering,config_.wheelbase,config_.step);
            if (t<brake_start) endpoint=pose;
            const double clearance=clearanceAt(pose);
            min_clearance=std::min(min_clearance,clearance);
            if (clearance<=0 || !local.freeFootprint({pose.x,pose.y},radius)) { safe=false; break; }
        }
        if (!safe) continue;
        const double goal_error=norm(goal-Point{endpoint.x,endpoint.y});
        const double heading=std::abs(wrap(std::atan2(goal.y-endpoint.y,
                                                     goal.x-endpoint.x)-endpoint.yaw));
        const double score=config_.weight_goal*goal_error+config_.weight_heading*heading+
            config_.weight_clearance*(1-std::clamp(min_clearance,0.0,1.0))+
            config_.weight_speed*(1-v/config_.max_speed)+
            config_.weight_smooth*std::abs(steering-odom.steering)/config_.max_steering;
        if (score<best.score) best={v,steering,score,min_clearance,v>0 ? "following" : "blocked"};
    }
    if (!std::isfinite(best.score)) return stop;
    return best;
}
} // namespace robopack
