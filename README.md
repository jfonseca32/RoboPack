# RoboPack

![TrailMate CAD](public/trailmate_render.png)

RoboPack is a person following robot built around a **Raspberry Pi, STM32, a bogey suspension, a SLAMTEC RPLIDAR C1, and four BU04 UWB boards**:
three receivers on the robot and one tag in the person's pocket.

The Pi estimates the robot's pose and builds a 2D LiDAR map, locates the tag in the person's pocket
from radio ranges, and scores steering/speed trajectories. The STM32 does the steering, motor speed control, and the final stop/watchdog behavior.

## Architecture

```mermaid
flowchart LR
    C1[RPLIDAR C1] --> Scan[LiDAR scan]
    Scan --> SLAM[Occupancy map]
    SLAM --> Collision[Obstacle and unknown space checks]
    STM[STM32 encoders and IMU] --> SLAM
    Tag[Pocket BU04 tag] <--> Anchors[Three BU04 robot receivers]
    Anchors --> UWB[Trilateration and uncertainty gate]
    UWB ---> Person[Where is the user the in map?]
    Person --> Planner[Score feasible trajectories]
    Collision --> Obstacles[Where are obstacles in the map?]
    Obstacles --> Planner
    Planner --> Guard[STM32 command guard]
    Stop[Physical emergency stop] --> Guard
    Guard --> Motors[Steering servo and drive motor]
```

## Build and try it

On Raspberry Pi OS 64-bit, Debian, or Ubuntu:

```bash
bash scripts/install_deps.sh
bash scripts/build.sh
bash scripts/run_demo.sh
```

The core needs a C++17 compiler and CMake 3.20+. The demo prints CSV commands to show how the robot would work.

Record and replay a synthetic run / simulation, including a tag dropout:

```bash
mkdir -p output
./build/robopack --demo --config config/robot.conf \
  --record output/demo.frames --map output/map.pgm > output/commands.csv
./build/robopack --replay output/demo.frames --config config/robot.conf
ctest --test-dir build --output-on-failure
```

![Simulation Output](public/simulation_output.png)

The map is a PGM image: black occupied, white free, gray unknown.

## Repository layout

```text
include/robopack/      Sensor, map, tracking, navigation and protocol interfaces
src/                   C++ implementation and offline executable
controls/              Steering/throttle helpers and scored trajectory planner
adapters/              Optional C1 capture program using the official SDK
config/robot.conf      chassis, sensor geometry and control settings
firmware/stm32/        Portable C command guard; no board-specific HAL
tests/                 Navigation, adapter, replay and firmware regression tests
scripts/               Dependency installation, build/test and demo commands
CAD_Files/             Self-explanatory
public/                Images
```
