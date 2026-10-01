#include "robopack/mapping.hpp"
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace robopack {
bool validScan(const Scan& s) {
    if (!std::isfinite(s.angle_min) || !std::isfinite(s.angle_step) ||
        !std::isfinite(s.range_min) || !std::isfinite(s.range_max) ||
        s.angle_step <= 0 || s.angle_step > 0.06 || s.range_min <= 0 ||
        s.range_max <= s.range_min || s.range_max > 12 ||
        s.ranges.size() < 120 || s.ranges.size() > 4096 ||
        std::abs(s.angle_step*s.ranges.size()-2*pi) > 0.1) return false;
    std::size_t usable = 0;
    for (double r : s.ranges) {
        if (!std::isfinite(r) || r < 0 || r > s.range_max) return false;
        if (r >= s.range_min) ++usable;
    }
    return usable*4 >= s.ranges.size()*3;
}
std::vector<Point> scanHits(const Scan& s, Pose lidar) {
    std::vector<Point> hits;
    for (std::size_t i=0; i<s.ranges.size(); ++i) {
        const double r = s.ranges[i], angle = s.angle_min+i*s.angle_step;
        if (r >= s.range_min && r < s.range_max)
            hits.push_back(transform(lidar,{r*std::cos(angle),r*std::sin(angle)}));
    }
    return hits;
}
namespace {
int checkedCells(const Config& c) { c.validate(); return c.map_cells; }
}
Grid::Grid(const Config& c) : cells_(checkedCells(c)), resolution_(c.map_resolution),
    data_(static_cast<std::size_t>(cells_)*cells_,0) {}
int Grid::index(Point p) const {
    if (!contains(p)) return -1;
    const int x = static_cast<int>(std::floor(p.x/resolution_+cells_/2.0));
    const int y = static_cast<int>(std::floor(p.y/resolution_+cells_/2.0));
    return y*cells_+x;
}
bool Grid::contains(Point p, double margin) const {
    const double half = cells_*resolution_/2;
    return std::isfinite(p.x) && std::isfinite(p.y) &&
           std::abs(p.x)+margin < half && std::abs(p.y)+margin < half;
}
int Grid::value(Point p) const { const int i=index(p); return i < 0 ? 0 : data_[i]; }
void Grid::integrate(const Scan& s, Pose base, Pose lidar_pose) {
    const Pose lidar = compose(base,lidar_pose);
    std::vector<int> endpoints;
    for (std::size_t i=0; i<s.ranges.size(); ++i) {
        const double r=s.ranges[i], angle=lidar.yaw+s.angle_min+i*s.angle_step;
        if (!std::isfinite(r) || r<s.range_min || r>s.range_max) continue;
        const bool hit = r<s.range_max;
        const int end=index({lidar.x+r*std::cos(angle),lidar.y+r*std::sin(angle)});
        int previous = -1;
        for (double d=0; d<r; d+=resolution_*0.4) {
            const int cell=index({lidar.x+d*std::cos(angle),lidar.y+d*std::sin(angle)});
            if (cell<0) break;
            if (cell!=previous && (!hit || cell!=end))
                data_[cell]=static_cast<std::int8_t>(std::max(-8,static_cast<int>(data_[cell])-2));
            previous=cell;
        }
        if (hit && end>=0) endpoints.push_back(end);
    }
    // Apply endpoints after clearing rays so nearby rays cannot erase a new obstacle.
    for (int cell : endpoints)
        data_[cell]=static_cast<std::int8_t>(std::min(8,std::max(4,static_cast<int>(data_[cell])+6)));
}
void Grid::clearUnknownFootprint(double radius) {
    const int n=static_cast<int>(std::ceil(radius/resolution_));
    for (int y=-n; y<=n; ++y) for (int x=-n; x<=n; ++x) {
        const Point p{x*resolution_,y*resolution_};
        const int i=index(p);
        if (norm(p)<=radius && i>=0 && data_[i]==0) data_[i]=-1;
    }
}
bool Grid::freeFootprint(Point p, double radius) const {
    // The extra half diagonal covers the entire cell and discretization of the path.
    const double padded=radius+resolution_*0.71;
    if (!contains(p,padded)) return false;
    const int n=static_cast<int>(std::ceil(padded/resolution_));
    for (int y=-n; y<=n; ++y) for (int x=-n; x<=n; ++x) {
        const Point offset{x*resolution_,y*resolution_};
        if (norm(offset)<=padded && value(p+offset)>=0) return false;
    }
    return true;
}
double Grid::matchScore(Point p) const {
    double score=0;
    for (int y=-2; y<=2; ++y) for (int x=-2; x<=2; ++x)
        if (value({p.x+x*resolution_,p.y+y*resolution_})>=2)
            score=std::max(score,std::exp(-0.5*(x*x+y*y)));
    return score;
}
void Grid::savePgm(const std::string& path) const {
    std::ofstream out(path,std::ios::binary);
    if (!out) throw std::runtime_error("cannot write map: "+path);
    out << "P5\n# resolution_m " << resolution_ << "; centered on initial rear axle\n"
        << cells_ << ' ' << cells_ << "\n255\n";
    for (int y=cells_-1; y>=0; --y) for (int x=0; x<cells_; ++x) {
        const int v=data_[y*cells_+x];
        const unsigned char pixel=v>=2 ? 0 : (v<0 ? 254 : 205);
        out.write(reinterpret_cast<const char*>(&pixel),1);
    }
    if (!out) throw std::runtime_error("failed writing map");
}
LocalSlam::LocalSlam(Config c) : config_(std::move(c)),map_(config_) { config_.validate(); }
Localization LocalSlam::update(const Scan& scan, const Odometry& odom) {
    auto fail=[&](const std::string& reason) { return Localization{false,pose_,0,reason}; };
    if (!validScan(scan) || !finite(odom.pose)) return fail("invalid_slam_input");
    if (scan.stamp<=last_stamp_) return fail("repeated_scan");
    const auto hits=scanHits(scan,config_.lidar_pose);
    if (hits.size()<20) return fail("insufficient_slam_features");
    if (!initialized_) {
        pose_={};
        map_.integrate(scan,pose_,config_.lidar_pose);
        last_odom_=odom.pose; last_stamp_=scan.stamp; initialized_=true;
        return {true,pose_,1,"map_initialized"};
    }
    const Point delta=inverseTransform(last_odom_,{odom.pose.x,odom.pose.y});
    const Pose predicted=compose(pose_,{delta.x,delta.y,wrap(odom.pose.yaw-last_odom_.yaw)});
    if (norm(delta)>1 || std::abs(wrap(odom.pose.yaw-last_odom_.yaw))>0.6)
        return fail("odometry_jump");
    if (!map_.contains({predicted.x,predicted.y},config_.footprint_radius+1))
        return fail("map_boundary");
    Pose best=predicted;
    double best_objective=-1e9, best_match=0;
    // Bounded correlative scan-to-map search, with a wheel/IMU odometry prior.
    for (int iy=-3; iy<=3; ++iy) for (int ix=-3; ix<=3; ++ix) for (int ia=-2; ia<=2; ++ia) {
        const double dx=ix*config_.match_translation/3, dy=iy*config_.match_translation/3;
        const double da=ia*config_.match_rotation/2;
        Pose trial{predicted.x+dx,predicted.y+dy,wrap(predicted.yaw+da)};
        double match=0; int count=0;
        for (std::size_t i=0; i<hits.size(); i+=3) { match+=map_.matchScore(transform(trial,hits[i])); ++count; }
        match/=count;
        const double objective=match-0.08*((dx*dx+dy*dy)/std::pow(config_.match_translation,2)+
                                        da*da/std::pow(config_.match_rotation,2));
        if (objective>best_objective) { best_objective=objective; best=trial; best_match=match; }
    }
    if (best_match<config_.min_match_score) return fail("scan_match_lost");
    pose_=best; last_odom_=odom.pose; last_stamp_=scan.stamp;
    map_.integrate(scan,pose_,config_.lidar_pose);
    return {true,pose_,best_match,"localized"};
}
} // namespace robopack
