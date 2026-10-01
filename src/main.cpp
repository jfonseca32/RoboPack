#include "robopack/navigation.hpp"
#include "robopack/replay.hpp"
#include "robopack/simulation.hpp"
#include "steering.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace robopack;
int main(int argc,char** argv) {
    try {
        std::string config_path,replay_path,record_path,map_path;
        bool demo=false; int steps=100;
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto next=[&]() {
                if (++i>=argc) throw std::runtime_error("missing value for "+arg);
                return std::string(argv[i]);
            };
            if (arg=="--config") config_path=next();
            else if (arg=="--replay") replay_path=next();
            else if (arg=="--record") record_path=next();
            else if (arg=="--map") map_path=next();
            else if (arg=="--steps") {
                const auto value=next(); std::size_t used=0;
                steps=std::stoi(value,&used);
                if (used!=value.size() || steps<1 || steps>100000) throw std::runtime_error("invalid --steps");
            }
            else if (arg=="--demo") demo=true;
            else if (arg=="--help") {
                std::cout<<"Usage: robopack (--demo | --replay FILE) [--config FILE]\n"
                         <<"       [--map FILE.pgm] [--steps N] [--record FILE]\n"
                         <<"Offline simulation/replay only; CSV commands never drive hardware.\n";
                return 0;
            } else throw std::runtime_error("unknown argument: "+arg);
        }
        if (demo==!replay_path.empty()) throw std::runtime_error("choose exactly one of --demo or --replay");
        if (!demo && !record_path.empty()) throw std::runtime_error("--record requires --demo");
        Config config=config_path.empty() ? Config{} : Config::load(config_path);
        config.validate(); Navigator nav(config);
        std::ifstream input; std::ofstream record;
        if (!demo) {
            input.open(replay_path);
            if (!input) throw std::runtime_error("cannot open replay: "+replay_path);
        }
        if (!record_path.empty()) {
            record.open(record_path);
            if (!record) throw std::runtime_error("cannot open record: "+record_path);
        }
        std::cerr<<"RoboPack offline "<<(demo ? "simulation" : "replay")<<"; no motor connection\n";
        std::cout<<"time_s,speed_m_s,steering_rad,reason,map_x_m,map_y_m,target_x_m,target_y_m,target_std_m\n";
        Pose robot; Command previous; int count=0;
        while (true) {
            Frame f;
            if (demo) {
                if (count>=steps) break;
                f.now=1+count*0.1;
                const Point person{2.0+0.05*count*0.1,0.45};
                f.scan=simulateScan(config,robot,f.now,{{{1.8,-0.60},0.30},{person,0.20}});
                f.odom={f.now,robot,previous.speed,previous.steering};
                f.ranges=simulateRanges(config,inverseTransform(robot,person),f.now);
                if (count>=60 && count<65) f.ranges.clear();
                if (record.is_open()) writeFrame(record,f);
            } else if (!readFrame(input,f)) break;
            const auto result=nav.update(f); const auto& cmd=result.command;
            std::cout<<std::fixed<<std::setprecision(6)<<f.now<<','<<cmd.speed<<','<<cmd.steering<<','
                     <<cmd.reason<<','<<result.localization.pose.x<<','<<result.localization.pose.y<<','
                     <<result.target.position.x<<','<<result.target.position.y<<','
                     <<result.target.position_stddev<<'\n';
            if (demo) {
                robot=advanceBicycle(robot,cmd.speed,cmd.steering,config.wheelbase,0.1); previous=cmd;
            }
            ++count;
        }
        if (count==0) throw std::runtime_error("replay contains no frames");
        if (!map_path.empty()) nav.map().savePgm(map_path);
        std::cerr<<"Processed "<<count<<" frames; end of input, stopped\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<"RoboPack: "<<e.what()<<'\n'; return 1; }
}
