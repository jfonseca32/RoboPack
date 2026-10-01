# Existing RoboPack mechanical design

The STL and SLDPRT directory contents are preserved from the original robot. The LiDAR/UWB software pivot does not change the frame, steering rack, suspension, drive train or body parts.

The controller assumes the existing car-style steering mechanism. Measure wheelbase, maximum road-wheel angle and a conservative loaded footprint about the rear axle for config/robot.conf.

Add the C1 and three BU04 receivers using mounts appropriate to the physical assembly. This revision does not add or change CAD mounting geometry. Place receiver antennas at three non-collinear, measured locations and keep the LiDAR scan plane clear of bodywork. See [hardware setup](../docs/hardware.md).
