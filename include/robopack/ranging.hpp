#pragma once
#include "robopack/config.hpp"
#include <utility>

namespace robopack {
class RangingSolver {
public:
    explicit RangingSolver(Config config) : config_(std::move(config)) { config_.validate(); }
    Target solve(const std::vector<Range>& ranges, double now) const;
private:
    Config config_;
};
} // namespace robopack
