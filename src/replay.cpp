#include "robopack/replay.hpp"
#include <iomanip>
#include <istream>
#include <ostream>
#include <stdexcept>
namespace robopack {
namespace {
void expect(std::istream& in,const char* word) {
    std::string token;
    if (!(in>>token) || token!=word) throw std::runtime_error(std::string("expected ")+word);
}
}
bool readFrame(std::istream& in,Frame& frame) {
    in>>std::ws;
    if (in.eof()) return false;
    Frame f;
    expect(in,"FRAME");
    int estop=0, count=0;
    if (!(in>>f.now>>estop) || (estop!=0 && estop!=1)) throw std::runtime_error("invalid FRAME");
    f.estop=estop!=0;
    expect(in,"ODOM");
    if (!(in>>f.odom.stamp>>f.odom.pose.x>>f.odom.pose.y>>f.odom.pose.yaw>>f.odom.speed>>f.odom.steering))
        throw std::runtime_error("invalid ODOM");
    expect(in,"SCAN");
    if (!(in>>f.scan.stamp>>f.scan.angle_min>>f.scan.angle_step>>f.scan.range_min>>f.scan.range_max>>count) ||
        count<0 || count>4096) throw std::runtime_error("invalid SCAN");
    f.scan.ranges.resize(static_cast<std::size_t>(count));
    for (auto& r:f.scan.ranges) if (!(in>>r)) throw std::runtime_error("truncated SCAN");
    expect(in,"RANGES");
    if (!(in>>count) || count<0 || count>64) throw std::runtime_error("invalid RANGES");
    for (int i=0;i<count;++i) {
        expect(in,"RANGE");
        Range r; int los=0;
        if (!(in>>r.anchor_id>>r.tag_id>>r.stamp>>r.distance>>r.stddev>>los) || (los!=0 && los!=1))
            throw std::runtime_error("invalid RANGE");
        r.line_of_sight=los!=0; f.ranges.push_back(r);
    }
    expect(in,"END");
    frame=std::move(f);
    return true;
}
void writeFrame(std::ostream& out,const Frame& f) {
    out<<std::setprecision(17)<<"FRAME "<<f.now<<' '<<f.estop<<"\nODOM "
       <<f.odom.stamp<<' '<<f.odom.pose.x<<' '<<f.odom.pose.y<<' '<<f.odom.pose.yaw<<' '
       <<f.odom.speed<<' '<<f.odom.steering<<"\nSCAN "<<f.scan.stamp<<' '<<f.scan.angle_min<<' '
       <<f.scan.angle_step<<' '<<f.scan.range_min<<' '<<f.scan.range_max<<' '<<f.scan.ranges.size()<<'\n';
    for (double r:f.scan.ranges) out<<r<<' ';
    out<<"\nRANGES "<<f.ranges.size()<<'\n';
    for (const auto& r:f.ranges)
        out<<"RANGE "<<r.anchor_id<<' '<<r.tag_id<<' '<<r.stamp<<' '<<r.distance<<' '<<r.stddev<<' '
           <<r.line_of_sight<<'\n';
    out<<"END\n";
    if (!out) throw std::runtime_error("failed to write frame");
}
}
