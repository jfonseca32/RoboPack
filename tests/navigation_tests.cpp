#include "robopack/navigation.hpp"
#include "robopack/protocol.hpp"
#include "robopack/adapters.hpp"
#include "robopack/replay.hpp"
#include "robopack/simulation.hpp"
#include "steering.hpp"
#include "throttle.hpp"
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace robopack;
#define CHECK(expr) do { if (!(expr)) throw std::runtime_error(std::string(__func__)+": "+#expr); } while (false)
void near(double a,double b,double tolerance=1e-5) { CHECK(std::abs(a-b)<tolerance); }
Frame frame(const Config& c,double time=1,Point target={2,0.2}) {
    Frame f; f.now=time; f.scan=simulateScan(c,{},time); f.odom.stamp=time;
    f.ranges=simulateRanges(c,target,time); return f;
}
void rangingTests() {
    Config c; RangingSolver solver(c);
    for (Point p : {Point{2,0.2},Point{1,1},Point{-1,0.5},Point{0,-1}}) {
        const auto t=solver.solve(simulateRanges(c,p,1),1);
        CHECK(t.valid); near(t.position.x,p.x); near(t.position.y,p.y);
        CHECK(t.position_stddev>0);
    }
    auto r=simulateRanges(c,{2,0.2},1);
    r.pop_back(); CHECK(!solver.solve(r,1).valid);
    r=simulateRanges(c,{2,0.2},1); r[0].line_of_sight=false; CHECK(!solver.solve(r,1).valid);
    r=simulateRanges(c,{2,0.2},1); r.push_back(r[0]); CHECK(!solver.solve(r,1).valid);
    r=simulateRanges(c,{2,0.2},1); r[0].distance=std::numeric_limits<double>::quiet_NaN();
    CHECK(!solver.solve(r,1).valid);
    r=simulateRanges(c,{2,0.2},1); r[0].anchor_id=999; CHECK(!solver.solve(r,1).valid);
    r=simulateRanges(c,{2,0.2},1); CHECK(!solver.solve(r,2).valid); CHECK(!solver.solve(r,0.9).valid);
    r[0].stamp=0.9; CHECK(!solver.solve(r,1).valid);
    r=simulateRanges(c,{2,0.2},1); r[0].distance+=2; CHECK(!solver.solve(r,1).valid);
    r=simulateRanges(c,{2,0.2},1); for (auto& v:r) v.tag_id=8; CHECK(!solver.solve(r,1).valid);
    r=simulateRanges(c,{2,0.2},1); r[0].distance=0.1; CHECK(!solver.solve(r,1).valid);
    // Small robot baselines cause high lateral uncertainty even with exact ranges.
    CHECK(!solver.solve(simulateRanges(c,{5,0},1),1).valid);
    c.anchors={{0,0,0,0},{1,0.2,0,0},{2,0.4,0,0}};
    bool threw=false; try { c.validate(); } catch (...) { threw=true; } CHECK(threw);
}
void adapterTests() {
    auto r=parseBu04Distance("distance: 2.340000\r",2,1,5,0.1,false);
    CHECK(r.has_value()); near(r->distance,2.34); CHECK(!r->line_of_sight);
    CHECK(!parseBu04Distance("OK",2,1,5,0.1,true));
    CHECK(!parseBu04Distance("distance: nan",2,1,5,0.1,true));
    CHECK(!parseBu04Distance("distance: 2.0 garbage",2,1,5,0.1,true));
    CHECK(!parseBu04Distance("distance: -1",2,1,5,0.1,true));
    CHECK(!parseBu04Distance("distance: 2",2,1,5,0,true));
    auto s=normalizeC1({{0,8000,4},{0,4000,4},{16384,8000,4},{32768,0,4},{49152,12000,0}},1);
    near(s.ranges[0],1); near(s.ranges[270],2);
    CHECK(s.ranges[180]==0); CHECK(s.ranges[90]==0); CHECK(!validScan(s));
}
void mappingTests() {
    Config c; Grid grid(c);
    CHECK(grid.value({1,0})==0);
    auto scan=simulateScan(c,{},1,{{{2,0},0.3}});
    CHECK(validScan(scan)); grid.integrate(scan,{},c.lidar_pose);
    CHECK(grid.value({1,0})<0); CHECK(grid.value({1.7,0})>0);
    CHECK(!grid.freeFootprint({1.5,0},0.3));
    LocalSlam slam(c);
    CHECK(slam.update(scan,{1,{},0,0}).valid);
    CHECK(!slam.update(scan,{1,{},0,0}).valid);
    Pose moved{0.15,0.02,0.02};
    auto scan2=simulateScan(c,moved,1.1,{{{2,0},0.3}});
    const auto localized=slam.update(scan2,{1.1,{0.20,0.02,0.02},0.2,0});
    CHECK(localized.valid);
    CHECK(std::abs(localized.pose.x-moved.x)<0.11);
    CHECK(std::abs(localized.pose.y-moved.y)<0.11);
    CHECK(!slam.update(simulateScan(c,moved,1.2),{1.2,{5,0,0},0,0}).valid);
    scan.ranges.assign(360,0); CHECK(!validScan(scan));
    scan.ranges.assign(360,12);
    LocalSlam no_features(c); CHECK(!no_features.update(scan,{1,{},0,0}).valid);
}
void plannerTests() {
    Config c; Planner planner(c);
    auto f=frame(c);
    auto t=RangingSolver(c).solve(f.ranges,f.now);
    auto cmd=planner.plan(f.scan,t,f.odom,0.1);
    CHECK(cmd.speed>0); CHECK(cmd.speed<=c.acceleration*0.1+1e-8);
    CHECK(std::abs(cmd.steering)<=c.steering_rate*0.1+1e-8);
    t.position={1,0}; CHECK(planner.plan(f.scan,t,f.odom,0.1).speed==0);
    t.position={-2,0}; CHECK(planner.plan(f.scan,t,f.odom,0.1).reason=="target_behind");
    t.position={2,0.2};
    auto blocked=simulateScan(c,{},1,{{{0.9,0},0.4}});
    CHECK(planner.plan(blocked,t,f.odom,0.1).speed==0);
    auto missing=f.scan;
    // Missing front sector: never fill unobserved space with free-space assumptions.
    for (int i=165;i<195;++i) missing.ranges[i]=0;
    CHECK(planner.plan(missing,t,f.odom,0.1).speed==0);
    f.odom.speed=0.45;
    blocked=simulateScan(c,{},1,{{{1.0,0},0.25}});
    CHECK(planner.plan(blocked,t,f.odom,0.1).speed==0);
    f.odom.speed=c.max_speed+0.01;
    CHECK(planner.plan(f.scan,t,f.odom,0.01).reason=="invalid_planner_input");
    f.odom.speed=0; f.odom.steering=c.max_steering+0.01;
    CHECK(planner.plan(f.scan,t,f.odom,0.01).reason=="invalid_planner_input");
    f.odom.steering=0;
    f.odom.speed=std::numeric_limits<double>::quiet_NaN();
    CHECK(planner.plan(f.scan,t,f.odom,0.1).speed==0);
    t.position.x=std::numeric_limits<double>::infinity();
    CHECK(planner.plan(f.scan,t,f.odom,0.1).speed==0);
    near(stoppingDistance(0.4,0.8,0.25),0.2);
    const auto p=advanceBicycle({},0.3,0.3,c.wheelbase,1);
    CHECK(p.y>0); CHECK(p.yaw>0);
    const auto stationary=advanceBicycle({},0,0.5,c.wheelbase,1);
    CHECK(stationary.x==0 && stationary.yaw==0);
}
void detourTests() {
    Config c;
    for (double side : {-1.0,1.0}) {
        Planner planner(c); Pose robot; Command previous;
        const Circle obstacle{{1.4,side*0.3},0.25};
        for (int i=0;i<80;++i) {
            const double time=1+i*0.1;
            auto scan=simulateScan(c,robot,time,{obstacle});
            Target target; target.valid=true;
            target.position=inverseTransform(robot,{3,0}); target.position_stddev=0.1;
            Odometry odom{time,robot,previous.speed,previous.steering};
            const auto cmd=planner.plan(scan,target,odom,0.1);
            CHECK(std::isfinite(cmd.speed) && std::isfinite(cmd.steering));
            robot=advanceBicycle(robot,cmd.speed,cmd.steering,c.wheelbase,0.1);
            CHECK(norm(Point{robot.x,robot.y}-obstacle.center)>
                  c.footprint_radius+c.safety_margin+obstacle.radius);
            previous=cmd;
        }
        CHECK(robot.x>1.4); // pass the obstacle, not just turn once and stall
        CHECK(robot.y*side < -0.5); // take the open side, for both mirrored scenes
    }
}
void navigationTests() {
    Config c;
    { Navigator n(c); auto f=frame(c); CHECK(n.update(f).command.speed>0);
      f=frame(c,1.1); f.ranges.clear(); CHECK(n.update(f).command.speed==0);
      f=frame(c,1.2); CHECK(n.update(f).command.speed>0); }
    { Navigator n(c); auto f=frame(c); f.estop=true;
      CHECK(n.update(f).command.reason=="emergency_stop"); }
    { Navigator n(c); auto f=frame(c); f.scan.stamp=0.1;
      CHECK(n.update(f).command.reason=="stale_sensor"); }
    { Navigator n(c); auto f=frame(c); CHECK(n.update(f).command.speed>0);
      f.now=1.1; CHECK(n.update(f).command.reason=="repeated_sensor"); }
    { Navigator n(c); auto f=frame(c); CHECK(n.update(f).command.speed>0);
      CHECK(n.update(f).command.reason=="nonmonotonic_time"); }
    { Navigator n(c); auto f=frame(c); f.odom.stamp=0.9;
      CHECK(n.update(f).command.reason=="sensor_time_skew"); }
    { Navigator n(c); auto f=frame(c); CHECK(n.update(f).command.speed>0);
      f=frame(c,2); CHECK(n.update(f).command.reason=="control_gap"); }
    { Navigator n(c); auto f=frame(c); CHECK(n.update(f).command.speed>0);
      auto next=frame(c,1.05); next.ranges=f.ranges;
      CHECK(n.update(next).command.reason=="repeated_ranges"); }
}
void protocolTests() {
    Config c; Navigator nav(c);
    const auto result=nav.update(frame(c));
    auto packet=encodeCommand(result.command,1);
    rp_guard guard; rp_guard_init(&guard); CHECK(rp_guard_arm(&guard,0));
    rp_guard_feed(&guard,packet.data(),packet.size(),100);
    CHECK(guard.enabled); CHECK(guard.speed_mm_s>0);
    auto bad=result.command; bad.speed=std::numeric_limits<double>::quiet_NaN();
    packet=encodeCommand(bad,2);
    rp_guard_feed(&guard,packet.data(),packet.size(),110);
    CHECK(!guard.enabled && guard.speed_mm_s==0);
    bad=result.command; bad.reason="emergency_stop";
    packet=encodeCommand(bad,3);
    rp_command decoded; CHECK(rp_decode(packet.data(),&decoded)); CHECK(!decoded.enable);
    bool threw=false;
    try { encodeCommand(result.command,4,1000); } catch (...) { threw=true; } CHECK(threw);
}
void replayTests() {
    const auto original=frame(Config{});
    std::stringstream text; writeFrame(text,original);
    Frame f; CHECK(readFrame(text,f)); CHECK(f.scan.ranges==original.scan.ranges);
    CHECK(f.ranges.size()==3); CHECK(!readFrame(text,f));
    bool threw=false; std::stringstream truncated("FRAME 1 0 ODOM");
    try { readFrame(truncated,f); } catch (...) { threw=true; } CHECK(threw);
    std::stringstream oversized("FRAME 1 0 ODOM 1 0 0 0 0 0 SCAN 1 -3.14 0.01 0.05 12 999999");
    threw=false; try { readFrame(oversized,f); } catch (...) { threw=true; } CHECK(threw);
}
int main(int argc,char** argv) {
    const std::pair<const char*,std::function<void()>> groups[]={
        {"ranging",rangingTests},{"adapters",adapterTests},{"mapping",mappingTests},
        {"planner",plannerTests},{"detour",detourTests},{"navigation",navigationTests},{"protocol",protocolTests},{"replay",replayTests}};
    int failed=0;
    for (const auto& group:groups) {
        if (argc>1 && group.first!=std::string(argv[1])) continue;
        try { group.second(); std::cout<<"PASS "<<group.first<<'\n'; }
        catch (const std::exception& e) { ++failed; std::cerr<<"FAIL "<<group.first<<": "<<e.what()<<'\n'; }
    }
    return failed==0 ? 0 : 1;
}
