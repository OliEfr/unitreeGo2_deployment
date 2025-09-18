Inference scripts to run a policy as fast as possible, eg for low level torque control. A cpp and a python example.

This code runs in the [unitree_mujoco container]([https://github.com/Atarilab/DOOM](https://github.com/Atarilab/unitree_mujoco_container/tree/7a1f16362ff653b45feb41273c97b212d5604c5b)) from ATARI Lab. I use this repo by cloning it into `src` from [DOOM](https://github.com/Atarilab/DOOM)

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


## Deployment with DOOM
- `source setup.sh` in ~/workspaces/DOOM and you are ready to go
- TODO robot tends to fall to forward-right direction -> add more pushes to policy?
- TOOD try smooth actions
- TODO oscillations are still visible, especially when contact with the ground -> further penalize action changes; penalize foot contact forces; penalize non-flat orientation


## NOTES
- When I deploy in DOOM using some action delay, the robot also tends to fall to the side. maybe this can be mitigated by a) adding more action delay in sim, b) adding base_lin_vel as obs in policy


## Results 18.09
- TORQUE_deploy_15_SEED_42 better (much less vibrations on robot; still falls to front right in sim and real)
- TORQUE_deploy_16_SEED_42 also good
- TORQUE_deploy_17_SEED_42 felt like vibrations were a bit severe
- TORQUE_deploy_18_SEED_42 best! had much less drift than the other policies! This can be used for video for sure! Maybe a bit less vibrations than 15 though, but still fine.
- TORQUE_deploy_18_SEED_1 more drift than seed 42
- TORQUE_deploy_18_SEED_2 better than seed 2, but worse than seed 42
- TORQUE_deploy_19_SEED_42 not tried because seemed bad in sim
- Overall I feel like the hips could be add more of an angle for more stable stance, but this might change torques required for hip actuators - so change this carefully!






