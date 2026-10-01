#include "robopack/ranging.hpp"
#include <algorithm>
#include <limits>
#include <set>

namespace robopack {
namespace {
struct Observation { Anchor a; Range r; };
bool solve2(double a, double b, double c, double x, double y, Point& p) {
    const double det = a*c-b*b;
    if (!std::isfinite(det) || det <= 1e-10*std::max(1.0,a*c)) return false;
    p = {(c*x-b*y)/det, (a*y-b*x)/det};
    return std::isfinite(p.x) && std::isfinite(p.y);
}
}
Target RangingSolver::solve(const std::vector<Range>& ranges, double now) const {
    Target out;
    std::vector<Observation> obs;
    std::set<int> seen;
    double oldest = std::numeric_limits<double>::infinity(), newest = 0;
    for (const auto& r : ranges) {
        if (r.tag_id != config_.tag_id) continue;
        const auto a = std::find_if(config_.anchors.begin(),config_.anchors.end(),
                                    [&](const Anchor& a) { return a.id == r.anchor_id; });
        if (a == config_.anchors.end()) { out.reason = "unknown_anchor"; return out; }
        if (!seen.insert(r.anchor_id).second) { out.reason = "duplicate_anchor"; return out; }
        if (!r.line_of_sight || !fresh(r.stamp,now,config_.sensor_timeout) ||
            !std::isfinite(r.distance) || r.distance <= 0 || r.distance > 100 ||
            !std::isfinite(r.stddev) || r.stddev <= 0 || r.stddev > 1) continue;
        if (r.distance <= std::abs(config_.tag_height-a->z)) {
            out.reason = "impossible_slant_range"; return out;
        }
        oldest = std::min(oldest,r.stamp); newest = std::max(newest,r.stamp);
        obs.push_back({*a,r});
    }
    if (obs.size() < 3) { out.reason = "insufficient_ranges"; return out; }
    if (newest-oldest > config_.max_sensor_skew) { out.reason = "range_time_skew"; return out; }
    out.stamp = oldest;
    auto horizontalSquared = [&](const Observation& o) {
        return o.r.distance*o.r.distance-std::pow(config_.tag_height-o.a.z,2);
    };
    const auto& first = obs.front();
    double aa = 0, ab = 0, bb = 0, ax = 0, bx = 0;
    for (std::size_t i=1; i<obs.size(); ++i) {
        const auto& o = obs[i];
        const double a = 2*(o.a.x-first.a.x), b = 2*(o.a.y-first.a.y);
        const double rhs = horizontalSquared(first)-horizontalSquared(o) +
            o.a.x*o.a.x+o.a.y*o.a.y-first.a.x*first.a.x-first.a.y*first.a.y;
        aa += a*a; ab += a*b; bb += b*b; ax += a*rhs; bx += b*rhs;
    }
    Point p;
    if (!solve2(aa,ab,bb,ax,bx,p)) { out.reason = "poor_anchor_geometry"; return out; }
    for (int iter=0; iter<25; ++iter) {
        aa = ab = bb = ax = bx = 0;
        for (const auto& o : obs) {
            const double dx = p.x-o.a.x, dy = p.y-o.a.y;
            const double d = std::sqrt(dx*dx+dy*dy+std::pow(config_.tag_height-o.a.z,2));
            if (d < 1e-8) { out.reason = "singular_range"; return out; }
            const double w = 1/std::pow(std::max(config_.range_stddev_floor,o.r.stddev),2);
            const double jx = dx/d, jy = dy/d, e = o.r.distance-d;
            aa += w*jx*jx; ab += w*jx*jy; bb += w*jy*jy; ax += w*jx*e; bx += w*jy*e;
        }
        Point delta;
        if (!solve2(aa,ab,bb,ax,bx,delta)) { out.reason = "poor_range_geometry"; return out; }
        if (norm(delta) > 2) delta = delta*(2/norm(delta));
        p = p+delta;
        if (norm(delta) < 1e-6) break;
        if (iter == 24) { out.reason = "range_solver_not_converged"; return out; }
    }
    aa = ab = bb = 0;
    double sum_error = 0, normalized_error = 0, max_error = 0;
    for (const auto& o : obs) {
        const double dx = p.x-o.a.x, dy = p.y-o.a.y;
        const double d = std::sqrt(dx*dx+dy*dy+std::pow(config_.tag_height-o.a.z,2));
        const double sigma = std::max(config_.range_stddev_floor,o.r.stddev), e = d-o.r.distance;
        sum_error += e*e; normalized_error += e*e/(sigma*sigma);
        max_error = std::max(max_error,std::abs(e));
        aa += dx*dx/(d*d*sigma*sigma); ab += dx*dy/(d*d*sigma*sigma); bb += dy*dy/(d*d*sigma*sigma);
    }
    out.residual = std::sqrt(sum_error/static_cast<double>(obs.size()));
    const double min_eigenvalue = 0.5*(aa+bb-std::hypot(aa-bb,2*ab));
    if (min_eigenvalue <= 1e-9) { out.reason = "poor_range_geometry"; return out; }
    out.position_stddev = std::sqrt(std::max(1.0,normalized_error/(obs.size()-2))/min_eigenvalue);
    if (max_error > config_.max_range_residual) { out.reason = "range_residual"; return out; }
    if (out.position_stddev > config_.max_target_stddev) { out.reason = "target_uncertain"; return out; }
    if (norm(p) > config_.max_target_range) { out.reason = "target_out_of_range"; return out; }
    out.position = p;
    out.valid = true;
    out.reason = "tracked";
    return out;
}
} // namespace robopack
