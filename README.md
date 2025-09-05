Inference scripts to run a policy as fast as possible, eg for low level torque control. A cpp and a python example.

This code runs in the [unitree_mujoco container]([https://github.com/Atarilab/DOOM](https://github.com/Atarilab/unitree_mujoco_container/tree/7a1f16362ff653b45feb41273c97b212d5604c5b)) from ATARI Lab. I use this repo by cloning it into `src` from [DOOM(https://github.com/Atarilab/DOOM)

# NOTES	
- for convinience, setup your vscode to compile the cpp program: `Ctrl+Shift+P` → `"CMake: Build"`, and follow the process. (You need to enable debug flags and disable release flags in the `CMakeLists.txt`!)
- build manually
  - mkdir build && cd build
  - cmake ..
  - make -j4
  - cd ..
  - ./build/run_controller
- when running unitree_mujoco cpp simulator do not use the -r go2 flag
- consider adding this to the "runArgs" section of devcontainer.json to improve RT capabilities. Then you can run programs inside the container using `nice` .
  - "--cpuset-cpus=0,1,2,3",
  - "--cap-add=SYS_NICE"







