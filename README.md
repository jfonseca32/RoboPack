# RoboPack
Lightweight person-following + obstacle-avoidance pipeline for a Raspberry Pi robot “backpack.”
It uses NanoDet (COCO person class) for following and a class-agnostic blob stage for obstacles.

- Fast on RPi4/RPi5 (10–14 FPS @ 640×480 on RPi4, higher if you drop resolution)
- Runs the detector every N frames (default 4) and does blob filtering every frame
- Ships with everything you need: C++ sources, prebuilt binary, and NanoDet model files (.param/.bin)

⚠️ The current app opens a preview window for debugging. A headless target and motor control (steering/throttle) are planned next.

## Contents
```
robot-backpack/  
├─ README.md  
├─ bin/  
│   └─ hybrid_app                     # prebuilt demo binary (debug preview)  
├─ models/  
│   ├─ nanodet_m.param                # NanoDet-m 320x320 (COCO)  
│   └─ nanodet_m.bin  
├─ src/  
│   ├─ hybrid_follow_avoid.cpp        # main app (preview window; blobs + person)  
│   ├─ nanodet_core.cpp               # NanoDet inference wrapper (ncnn)  
│   └─ nanodet_core.h  
├─ controls/                          # (placeholders; coming soon)  
│   ├─ steering.hpp  
│   ├─ steering.cpp                   # TODO: read scores, output steering command  
│   ├─ throttle.hpp  
│   └─ throttle.cpp                   # TODO: read scores, output throttle command  
├─ scripts/  
│   ├─ install_deps.sh                # convenience install (optional)  
│   ├─ build.sh                       # local build into ./bin  
│   └─ run_demo.sh                    # runs the preview build  
├─ CMakeLists.txt                     # optional (you can build with g++)
├─ CAD_Files/
│   ├─ STL_Files/              
│   └─ SLDPRT_Files/                   
```

## What this does
- Person follow: runs NanoDet-m (320×320) every N frames (default 8), picks best person box and overlays it in red.
- Obstacle blobs: motion/edge-based, class-agnostic, stabilized over time; drawn in green. (The person region is excluded from blobs so following isn’t blocked by “self” detections.)

## Quick Start (RPi OS, one folder)
Use one folder (the repo root) so everything stays together.

### 1) Flash Pi OS & enable camera
- Raspberry Pi OS (64-bit recommended), enable camera (raspi-config).
- Update base packages. In cmd:  
```
sudo apt update && sudo apt upgrade -y
```

### 2) Install dependencies (distro packages)
We use the distro OpenCV and ncnn packages for simplicity. In cmd:
```
sudo apt install -y build-essential cmake pkg-config \
                    libopencv-dev \
                    libncnn-dev
```

### 3) Clone this repo into a single folder
In cmd:
```
cd ~
git clone https://github.com/jfonseca32/RoboPack/ robot-backpack
cd robot-backpack
```

### 4) Build (to ./bin/hybrid_app)
Using g++ directly:
```
mkdir -p bin
g++ -O3 -DNDEBUG -ffast-math -fopenmp \
  src/hybrid_follow_avoid.cpp src/nanodet_core.cpp \
  -Isrc `pkg-config --cflags --libs opencv4` \
  -lncnn -pthread -o bin/hybrid_app
```
> If your distro’s pkg name is opencv (not opencv4), swap the backticked part accordingly.

Or with CMake:
```
mkdir -p build && cd build
cmake ..
make -j4
cd ..
```

### 5) Run (preview window)
USB webcam:
```
./bin/hybrid_app --source usb0 --resolution 640x480 --det_every 4
```

Pi Camera (if exposed as /dev/video*):
```
./bin/hybrid_app --source picamera0 --resolution 640x480 --det_every 4
```

**Controls**:  
- `q` → quit
- `s` → pause
- Green = blobs (obstacles), Red = best person box, FPS in yellow

## Headless & Robot Control (coming next)
The current app is not headless and does not yet drive motors.  
Planned changes:
- `src/followbot_headless.cpp` — same pipeline, no window; periodically emits scores to the control layer:
  - `follow_error_x` (−1 … +1): person center vs image center
  - `follow_error_area` (0 … 1): how big the person box is (proxy for distance)
  - `obstacle_left`, `obstacle_right` (0 … 1): blob density/penalty
- `controls/steering.cpp` — converts follow/obstacle scores to steering command
- `controls/throttle.cpp` — converts distance/obstacle scores to throttle command
- Simple IPC (choose one):
  - **In-process**: call control functions directly (single binary)
  - **Named pipe (FIFO)**: headless emits JSON lines → controls read
  - **UDP**: headless broadcasts scores → controls subscribe  

**Proposed structure**:
```
controls/
  steering.hpp     # API: set/get steering command; configure gains
  steering.cpp     # TODO: PID or simple P controller for heading
  throttle.hpp     # API: set/get throttle command; configure gains
  throttle.cpp     # TODO: distance control + obstacle braking
include/
  followbot_msgs.hpp    # structs for scores/commands (shared header)
src/
  followbot_headless.cpp   # TODO: emit scores to controls (no GUI)
  hybrid_follow_avoid.cpp  # (this repo) GUI preview/debug
```
> When we publish the headless target, systemd unit files will be included so the service auto-starts on boot.

## Configuration & Tuning
- Detector cadence: --det_every N (lower = more responsive, slower FPS; default 8).
  - If you want steadier FPS even when person is lost, set the same cadence for both “hit” and “miss” inside the code (we left comments where to change).
- Resolution: --resolution WxH (e.g., 320x240 to boost FPS; the model still runs 320×320 internally).
- Blob stability: parameters in hybrid_follow_avoid.cpp (comments in code):
  - bg_learn_rate (background learning), acc_alpha (temporal smoothing),
  - ROI (default = bottom 65% of frame),
  - min_area (ignore small flicker),
  - Canny thresholds (edge fallback for static obstacles).
- Model: ships with NanoDet-m (320).

## Troubleshooting
- Camera opens but no window / black: try `--resolution 320x240`; ensure your PiCam is exposed as `/dev/video*`. If using `libcamera`, you may need a GStreamer pipeline string instead of `picamera0`.
- Linker errors on OpenCV: ensure `libopencv-dev` is installed; verify `pkg-config --modversion opencv4` prints a version.
- ncnn not found: `sudo apt install libncnn-dev`.
- FPS too low: reduce `--resolution`; increase `--det_every`; tighten ROI; raise `min_area`.

## Build Scripts (Optional)
`scripts/install_deps.sh`
```
#!/usr/bin/env bash
set -e
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev libncnn-dev
echo "Deps installed."
```

`scripts/build.sh`
```
#!/usr/bin/env bash
set -e
mkdir -p bin
g++ -O3 -DNDEBUG -ffast-math -fopenmp \
  src/hybrid_follow_avoid.cpp src/nanodet_core.cpp \
  -Isrc `pkg-config --cflags --libs opencv4` \
  -lncnn -pthread -o bin/hybrid_app
echo "Built ./bin/hybrid_app"
```

`scripts/run_demo.sh`
```
#!/usr/bin/env bash
set -e
./bin/hybrid_app --source usb0 --resolution 640x480 --det_every 8
```

Make them executable:
```
chmod +x scripts/*.sh
```

## Notes on Models
- The `models/nanodet_m.param` & `models/nanodet_m.bin` provided here match the code’s expected layer names:
  - input: `input.1`
  - outputs: `792, 795, 814, 817, 836, 839`
- If you swap models, keep layer names in sync with `src/nanodet_core.cpp`.

## Safety
This is research/demo code. When you hook up motors:
- Test wheels off the ground first.
- Add emergency stop binding and a max speed.
- Run in open spaces before indoor clutter.
