// Independent client of the official SLAMTEC SDK. No vendor source copied.
#include "robopack/adapters.hpp"
#include "sl_lidar.h"
#include "sl_lidar_driver.h"
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
volatile std::sig_atomic_t interrupted=0;
void handleSignal(int) { interrupted=1; }
double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct StopDriver {
    void operator()(sl::ILidarDriver* driver) const {
        if (driver) { driver->stop(); driver->setMotorSpeed(0); driver->disconnect(); delete driver; }
    }
};
}
int main(int argc,char** argv) {
    if (argc!=2 || std::string(argv[1])=="--help") {
        std::cerr<<"Usage: robopack_c1_capture /dev/serial/by-id/DEVICE\n"
                 <<"Stationary bench capture only. CSV: receipt_s,angle_rad,range_m\n";
        return argc==2 ? 0 : 1;
    }
    try {
        std::unique_ptr<sl::IChannel> channel(*sl::createSerialPortChannel(argv[1],460800));
        std::unique_ptr<sl::ILidarDriver,StopDriver> driver(*sl::createLidarDriver());
        if (!channel || !driver || SL_IS_FAIL(driver->connect(channel.get())))
            throw std::runtime_error("C1 connection failed");
        sl_lidar_response_device_health_t health{};
        if (SL_IS_FAIL(driver->getHealth(health)) || health.status!=SL_LIDAR_STATUS_OK)
            throw std::runtime_error("C1 health check failed");
        if (SL_IS_FAIL(driver->setMotorSpeed()) || SL_IS_FAIL(driver->startScan(false,true)))
            throw std::runtime_error("C1 scan start failed");
        std::signal(SIGINT,handleSignal); std::signal(SIGTERM,handleSignal);
        std::cout<<"receipt_s,angle_rad,range_m\n"<<std::setprecision(12);
        while (!interrupted) {
            sl_lidar_response_measurement_node_hq_t nodes[8192];
            std::size_t count=8192;
            if (SL_IS_FAIL(driver->grabScanDataHq(nodes,count,500)))
                throw std::runtime_error("C1 scan timeout");
            std::vector<robopack::C1Sample> samples;
            samples.reserve(count);
            for (std::size_t i=0;i<count;++i)
                samples.push_back({nodes[i].angle_z_q14,nodes[i].dist_mm_q2,nodes[i].quality});
            const auto scan=robopack::normalizeC1(samples,now());
            for (std::size_t i=0;i<scan.ranges.size();++i)
                std::cout<<scan.stamp<<','<<scan.angle_min+i*scan.angle_step<<','<<scan.ranges[i]<<'\n';
            if (!std::cout) throw std::runtime_error("C1 output stream closed");
        }
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
