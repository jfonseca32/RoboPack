# RoboPack
Code for RoboPack: backpack that follows you. It uses YOLO-based person and object detection is to be run on a Raspberry Pi.

## Dependencies
You must first download the Nanodet Library onto the Pi. To do so, you must have:
- A raspberry Pi 4 with a 32 or 64-bit operating system. It can be the Raspberry 64-bit OS, or Ubuntu 18.04 / 20.04. Install 64-bit OS: https://qengineering.eu/install-raspberry-64-os.html  
- The Tencent ncnn framework installed. Install ncnn: https://qengineering.eu/install-ncnn-on-raspberry-pi-4.html
- OpenCV 64 bit installed. Install OpenCV 4.5: https://qengineering.eu/install-opencv-on-raspberry-64-os.html
- Code::Blocks installed. ($ sudo apt-get install codeblocks)  

Then, follow the instalation instructions of the following link:
- https://github.com/Qengineering/NanoDetPlus-ncnn-Raspberry-Pi-4  

Once that has finished installing, reboot the Raspberry Pi. If you want to be a power user, you can do so by typing into the shell:

reboot

## Setup

We have one more thing to do, and that is to set up Thonny to use the virtual environment we just created for Nanodet. Thonny is the program we will be running all of our code out of and we need to get it to work out of the same venv so that it has access to the libraries we installed.

The first time you open Thonny it may be in the simplified mode, and you will see a "switch to regular mode" in the top right. If this is present click it and restart Thonny by closing it. 

Now enter the interpreter options menu by selecting Run > Configure Interpreter.  Under the executable option, there is a button with 3 dots. Select it and navigate to the C++ executable in the virtual environment we just created.

This will be located under home/pi/nanodet/bin and in this file, you will need to select the file called "cpp". Hit okay and you will now be working in this venv.

Whenever you open Thonny, it will now automatically work out of this environment. You can change the environment you are working out of by selecting it from the drop-down menu under the executable in the same interpreter options menu. If you wish to exit the virtual environment, select the option bin/python3.

Finally, create new script in Thonny and paste in all the code in this repository in your desired folder. This can also be done by opening this repository in your Pi and downloading from there. 
