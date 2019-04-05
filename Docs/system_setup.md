# Things to Do
- (Optional)Prepare WIN10
- Create Ubuntu16.04 Installation USB Drive
- Install Ubuntu16.04
- Install ROS
- Install TensorFlow

# Prepare WIN10
Assume you are going to dual boot ***Ubuntu 16.04*** alonside your ***Windows 10***

## Step-1. Check BIOS Mode
Search "System Information" in the search bar and open it. Make sure in `System Summary`, for `Item` `BIOS Mode` its corresponding `Value` is `UEFI`.
![sys_info](https://github.com/linZHank/two_loggers/blob/master/Docs/images/sys_info.PNG)

> If `Value` is `Legacy`, please refer [this guide](https://docs.microsoft.com/zh-cn/windows/deployment/mbr-to-gpt) to switch the disk partition style from MBR to GPT.
> - Right click `Win` button on bottom left, select `Disk Manager`, make sure *Window* is installed on Disk 0. If on other disk, you'll have to change the disk index accordingly.
> - Fire up `Command Prompt` as administrator
```console
X:\> DiskPart

DISKPART> list volume

Volume ###  Ltr  Label        Fs     Type        Size     Status     Info
  ----------  ---  -----------  -----  ----------  -------  ---------  --------
  Volume 0     F   CENA_X64FRE  UDF    DVD-ROM     4027 MB  Healthy
  Volume 1     C   System Rese  NTFS   Partition    499 MB  Healthy
  Volume 2     D   Windows      NTFS   Partition     58 GB  Healthy
  Volume 3     E   Recovery     NTFS   Partition    612 MB  Healthy    Hidden
```

> Note which volume is labeled as *Windows*, in this case **Volume 2**
```console
DISKPART> exit

Leaving DiskPart...

X:\>mbr2gpt /convert /disk:0 /allowFullOS

MBR2GPT will now attempt to convert disk 0.
If conversion is successful the disk can only be booted in GPT mode.
These changes cannot be undone!

MBR2GPT: Attempting to convert disk 0
MBR2GPT: Retrieving layout of disk
MBR2GPT: Validating layout, disk sector size is: 512 bytes
MBR2GPT: Trying to shrink the system partition
MBR2GPT: Trying to shrink the OS partition
MBR2GPT: Creating the EFI system partition
MBR2GPT: Installing the new boot files
MBR2GPT: Performing the layout conversion
MBR2GPT: Migrating default boot entry
MBR2GPT: Adding recovery boot entry
MBR2GPT: Fixing drive letter mapping
MBR2GPT: Conversion completed successfully
MBR2GPT: Before the new system can boot properly you need to switch the firmware to boot to UEFI mode!
```
> - Reboot your computer. Now, your *Windows* is on a GPT disk, and your BIOS mode should be `UEFI`.

## Step-2 Shrink Disk
- Right click `WIN` button on bottom left, select `Disk Manager`, locate the block for drive **(C:)**, right click on that block to bring up `Shrink C:` dialog.
- At **"Enter the amount of space to shrink in MB"**, input the space you want allocate to ***Ubuntu***. Then **`Shrink`**
> recommend: at least 50000 (~50G)
![shrink_disk](https://github.com/linZHank/two_loggers/blob/master/Docs/images/shrink_disk.PNG)


# Install Ubuntu 16.04

## Step-1 Create Ubuntu16.04 Installation USB Drive
Refer to [Create a bootable USB stick on Windows](https://tutorials.ubuntu.com/tutorial/tutorial-create-a-usb-stick-on-windows#0). Note: make sure you download the [Ubuntu 16.04 desktop image](http://releases.ubuntu.com/16.04/)

## Step-2 BIOS Settings
Insert the just created Ubuntu bootable USB drive and reboot. A few things you'll have to make sure in your BIOS (press F-2 when computer startup).
    1. The storage option should be **AHCI** instead of ~~RAID~~.
    2. **Disable** the option of "fast boot".
    3. Set your system to boot up from **USB Drive**
    4. **Save and Exit**

## Step-3 Install Ubuntu from USB
- A (purplish) screen will show up if successfully boot up from USB. This interface is called **GRUB**, and in this grub you'll have 4 options (`Try Ubuntu without installing`; `Install Ubuntu`; `OEM install (for manufacturers)`; `Check disc for defects`)
- Select first option: **Try Ubuntu without installing**. This should brings you to the Ubuntu tryout interface (It looks exactly the same as the installed one).
![try_ubuntu](https://github.com/linZHank/two_loggers/blob/master/Docs/images/try_ubuntu.png)
- Double click the only icon on desktop to **Install Ubuntu 16.04LTS**
- "Continue" -\> "**don't** download update, **don't** install 3rd party software" -\> "Continue"
- Make sure the first option is exactly **Install Ubuntu alongside Windows Boot Manager**. If not, go back to beginning of this guide and make sure your boot disk has been switched to GPT format.
- "Continue" all the way till end of installation.
- Reboot and you are all set with a dual-boot(***Ubuntu*** and ***Windows***) machine.

# Install ROS
Go to official [ROS](http://www.ros.org/) site for a full [installation guide](http://wiki.ros.org/kinetic/Installation/Ubuntu). One thing you should've notice: ROS version is strictly associate with Ubuntu version. So, in our case under Ubuntu 16.04, we must install **ROS-Kinetic**. The following steps are tailored to be a shorter guide of installing ROS

## Step-1 Desktop-full Installtion
Open a terminal by `Ctrl` `Alt` `t`
1. Setup your sources.list
    ```bash
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    ```
2. Setup your keys
    ```bash
    sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
    ```
3. Installation
    ```bash
    sudo apt-get update
    sudo apt-get install ros-kinetic-desktop-full
    ```

## Step-2 After Installation
1. Initialize rosdep
    ```bash
    sudo rosdep init
    rosdep update
    ```
2. Environment setup
    ```bash
    echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
    ```
    **Note**: a substitution way of doing this is open `~/.bashrc` with your favorite text editor (gedit, atom, emacs, vim, etc.). Scroll down to the end of this file and add this line `source /opt/ros/kinetic/setup.bash`.

# Build `two_loggers`
## Step-1 (Optional) Install catkin-command-line-tools
```bash
sudo apt install python-catkin-tools
```
## Step-2 Create a ROS workspace
```bash
mkdir -p ~/ros_ws/src
cd ~/ros_ws
catkin init
```
## Step-3 Build packages in *this repo*
```bash
cd ~/ros_ws/src
git clone https://github.com/linZHank/two_loggers.git
cd ~/ros_ws
catkin build
echo "source ~/ros_ws/devel/setup.bash" >> ~/.bashrc
```
> You can substitute `catkin build` with `catkin_make` if you use original *catkin_make* tool to build ros packages.

## Step-4 Verify Your Build
1. Verify single logger control
    Open a new terminal
    ```bash
    roslaunch loggers_control single_logger_control.launch
    ```
    Open another terminal
    ```bash
    rosrun loggers_control single_logger_test.py
    ```
2. Verify two loggers control
    Open a new terminal
    ```bash
    roslaunch loggers_control two_loggers_control.launch
    ```
    Open another terminal
    ```bash
    rosrun loggers_control two_loggers_test.py
    ```
# Install TensorFlow
