#!/bin/bash

sudo apt-get install gdb libglfw3-dev libxinerama-dev libxcursor-dev libxi-dev unzip

# unitree cpp sdk
cd ~/workspace
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2/
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/unitree_robotics
sudo make install

# mujoco cpp (but you can also use python simulator)
# cd ~/workspace
# git clone --depth 1 --branch 3.2.7 https://github.com/google-deepmind/mujoco.git
# cd mujoco
# mkdir build && cd build
# cmake ..
# make -j4
# sudo make install

# cd ~/workspace/unitree_mujoco/build
# cmake ..
# make -j4

# libtorch
cd ~/workspace/
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip
unzip libtorch-shared-with-deps-2.8.0+cpu.zip
rm -rf libtorch-shared-with-deps-2.8.0+cpu.zip 

# # https://github.com/unitreerobotics/unitree_mujoco/issues/51
# if ! grep -q "export UNITREE_DDS_PATH=/opt/unitree_robotics/lib" ~/.bashrc; then
#     echo "export UNITREE_DDS_PATH=/opt/unitree_robotics/lib" >> ~/.bashrc
# fi
# if ! grep -q "export LD_LIBRARY_PATH=\$UNITREE_DDS_PATH:\$LD_LIBRARY_PATH" ~/.bashrc; then
#     echo "export LD_LIBRARY_PATH=\$UNITREE_DDS_PATH:\$LD_LIBRARY_PATH" >> ~/.bashrc
# fi
# 
# sed -i '/source \/opt\/ros\/humble\/setup.bash/d' ~/.bashrc
# sed -i '/source \/home\/atari\/dependencies\/install\/setup.bash/d' ~/.bashrc
# sed -i '/source \/home\/atari\/workspace\/DOOM\/install\/setup.bash/d' ~/.bashrc

# source ~/.bashrc
