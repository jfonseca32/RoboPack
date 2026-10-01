#pragma once
#include "robopack/types.hpp"
#include <cstdint>
#include <optional>
namespace robopack {
// Vendor-independent copy of the SDK's HQ sample fields, not a packed wire struct.
struct C1Sample {
    std::uint16_t angle_q14;
    std::uint32_t distance_q2_mm;
    std::uint8_t quality;
};
Scan normalizeC1(const std::vector<C1Sample>& samples,double acquisition_stamp);
std::optional<Range> parseBu04Distance(const std::string& line,int anchor_id,int tag_id,
                                     double acquisition_stamp,double stddev,bool line_of_sight);
}
