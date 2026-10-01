#pragma once
#include "robopack/types.hpp"
#include "drive_guard.h"
#include <array>
#include <cstdint>
namespace robopack {
std::array<std::uint8_t,RP_PACKET_SIZE> encodeCommand(const Command& command,
                                                    std::uint32_t sequence,
                                                    std::uint16_t lease_ms=200);
}
