#include "robopack/protocol.hpp"
#include <stdexcept>
namespace robopack {
std::array<std::uint8_t,RP_PACKET_SIZE> encodeCommand(const Command& command,
                                                    std::uint32_t sequence,std::uint16_t lease) {
    if (lease==0 || lease>RP_MAX_LEASE_MS) throw std::runtime_error("invalid command lease");
    rp_command wire{};
    wire.sequence=sequence; wire.lease_ms=lease;
    if (command.reason=="following" && std::isfinite(command.speed) &&
        std::isfinite(command.steering) && command.speed>0 &&
        command.speed<=RP_MAX_SPEED_MM_S/1000.0 &&
        std::abs(command.steering)<=RP_MAX_STEERING_MRAD/1000.0) {
        wire.enable=1;
        wire.speed_mm_s=static_cast<std::uint16_t>(std::lround(command.speed*1000));
        wire.steering_mrad=static_cast<std::int16_t>(std::lround(command.steering*1000));
    }
    std::array<std::uint8_t,RP_PACKET_SIZE> packet{};
    rp_encode(&wire,packet.data());
    return packet;
}
}
