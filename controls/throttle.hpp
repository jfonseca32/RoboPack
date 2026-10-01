#pragma once
namespace robopack {
double stoppingDistance(double speed, double braking, double reaction_time);
double approachSpeed(double distance_error, double max_speed, double braking);
}
