# Things to Do
- (Optional)Prepare WIN10
- Create Ubuntu16.04 Installation USB Drive
- Install Ubuntu16.04
- Install ROS
- Install TensorFlow

# Prepare WIN10

## Step-1. Check BIOS Mode
Search "System Information" in the search bar and open it. Make sure in `System Summary`, for `Item` `BIOS Mode` its corresponding `Value` is `UEFI`. \\
![sys_info](https://github.com/linZHank/two_loggers/raw/master/Docs/images/sys_info.png)

> If `Value` is `Legacy`, please refer [this guide](https://docs.microsoft.com/zh-cn/windows/deployment/mbr-to-gpt) to switch the disk partition style from MBR to GPT.
- Right click `Win` button on bottom left, select `Disk Manager`, make sure *Window* is installed on Disk 0. If on other disk, you'll have to change the disk index accordingly.
- Fire up `Command Prompt` as administrator
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
Note which volume is labeled as *Windows*, in this case **Volume 2**
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
- Reboot your computer. Now, your *Windows* is on a GPT disk, and your BIOS mode should be `UEFI`.  
# Create Ubuntu16.04 Installation USB Drive

# Install Ubuntu16.04

# Install ROS

# Install TensorFlow
