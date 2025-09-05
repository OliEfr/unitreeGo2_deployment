import numpy as np
import torch
import time
import math
import sys
from typing import Dict, List, Optional
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

JOINT_UNITREE_TO_ISAAC_LAB_MAPPING = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
JOINT_ISAAC_LAB_TO_UNITREE_MAPPING = torch.tensor([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10], dtype=torch.int64)

ISAAC_LAB_DEFAULT_JOINT_POS = [
    0.1000, # 0
    -0.1000, # 1
    0.1000, # 2
    -0.1000, # 3
    0.8000, # 4
    0.8000, # 5
    1.0000, # 6
    1.0000, # 7
    -1.5000, # 8
    -1.5000, # 9
    -1.5000, # 10
    -1.5000, # 11
  ]

# ACTION_SCALE = 10.0
# CLIP_ACTION_MIN = -23.5
# CLIP_ACTION_MAX = 23.5

ACTION_SCALE = 0.25
CLIP_ACTION_MIN = -23.5
CLIP_ACTION_MAX = 23.5
STIFFNESS= 27.0
DAMPING= 1.5

STATE_ORDER = [
    "robot/base_lin_vel",
    "robot/ang_vel_b",
    "robot/command", 
    "robot/projected_gravity_b",
    "robot/joint_pos",
    "robot/joint_vel",
    "robot/last_action"
]

class ObservationHistoryStorage:
    def __init__(
        self, num_envs: int, num_obs: int, max_length: int, device: torch.device = "cpu"
    ):
        """
        Initialize a FIFO queue for state history, starting with zeros at initialization.

        Args:
            num_envs (int): Number of environments.
            num_obs (int): Number of observations per environment.
            max_length (int): Maximum length of the state history for each environment.
            device (torch.device): Device to store the buffer (e.g., "cuda" or "cpu").
        """
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.max_length = max_length
        self.device = device

        # Initialize the buffer with zeros of shape (num_envs, num_obs * max_length)
        self.buffer = torch.zeros((num_envs, num_obs * max_length), device=device)

    def add(self, observation: torch.Tensor):
        """
        Add a new observation to the buffer. Perform FIFO replacement.

        Args:
            observation (torch.Tensor): The new observation to add.
                                         Should have shape `(num_envs, num_obs)`.
        """
        if observation.shape != (self.num_envs, self.num_obs):
            raise ValueError(
                f"Observation shape must be ({self.num_envs}, {self.num_obs})"
            )

        # Shift the buffer to make space for the new observation
        self.buffer[:, : -self.num_obs] = self.buffer[:, self.num_obs :].clone()

        # Add the new observation at the end
        self.buffer[:, -self.num_obs :] = observation

    def get(self) -> torch.Tensor:
        """
        Get the current state history.

        Returns:
            torch.Tensor: A tensor of shape `(num_envs, num_obs * max_length)`.
        """
        return self.buffer.detach().clone()

    def reset(self, done: torch.Tensor):
        """Reset the buffer for environments that are done.

        Args:
            done (torch.Tensor): mask of dones.
        """

        done_indices = torch.nonzero(done == 1)
        self.buffer[done_indices] = 0.0

def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(value, max_val))

def quat_rotate_inverse(q: List[float], v: List[float]) -> List[float]:
    q_w, q_x, q_y, q_z = q
    v_x, v_y, v_z = v
    
    factor_a = 2.0 * q_w * q_w - 1.0
    a = [v_x * factor_a, v_y * factor_a, v_z * factor_a]
    
    cross_x = q_y * v_z - q_z * v_z
    cross_y = q_z * v_x - q_x * v_z
    cross_z = q_x * v_y - q_y * v_x
    
    factor_b = q_w * 2.0
    b = [cross_x * factor_b, cross_y * factor_b, cross_z * factor_b]
    
    dot_product = q_x * v_x + q_y * v_y + q_z * v_z
    factor_c = dot_product * 2.0
    c = [q_x * factor_c, q_y * factor_c, q_z * factor_c]
    
    return [a[0] - b[0] + c[0], a[1] - b[1] + c[1], a[2] - b[2] + c[2]]

def projected_gravity_b(base_quat: List[float]) -> List[float]:
    gravity_dir = [0.0, 0.0, -1.0]
    return quat_rotate_inverse(base_quat, gravity_dir)

def go2_low_state_handler(msg):
    """Extracts and filters the low-level state of the robot.

    Args:
        msg (Dict): Low-level state message
        logger (logging.Logger): Logger for debugging

    Returns:
        Dict: Filtered low-level state of the robot
    """

    # Extract motor states
    motor_states = msg.motor_state[:12]  # 12 joint for the legs, the remaining 8 are unactuated
    joint_positions = np.array([motor.q for motor in motor_states])[JOINT_UNITREE_TO_ISAAC_LAB_MAPPING]
    joint_velocities = np.array([motor.dq for motor in motor_states])[JOINT_UNITREE_TO_ISAAC_LAB_MAPPING]


    # Extract and filter IMU data
    imu_state = msg.imu_state
    gyroscope = np.array(imu_state.gyroscope)
    quaternion = np.array(imu_state.quaternion)

    # Filter joint states and IMU data
    alpha = 1.0  # Adjust this value based on your needs (higher = more responsive)

    if not hasattr(go2_low_state_handler, "filtered_joint_pos"):
        go2_low_state_handler.filtered_joint_pos = joint_positions
        go2_low_state_handler.filtered_joint_vel = joint_velocities
        go2_low_state_handler.filtered_gyro = gyroscope
        go2_low_state_handler.filtered_quat = quaternion
    else:
        go2_low_state_handler.filtered_joint_pos = (
            alpha * joint_positions + (1 - alpha) * go2_low_state_handler.filtered_joint_pos
        )
        go2_low_state_handler.filtered_joint_vel = (
            alpha * joint_velocities + (1 - alpha) * go2_low_state_handler.filtered_joint_vel
        )
        go2_low_state_handler.filtered_gyro = alpha * gyroscope + (1 - alpha) * go2_low_state_handler.filtered_gyro
        go2_low_state_handler.filtered_quat = alpha * quaternion + (1 - alpha) * go2_low_state_handler.filtered_quat

    # Normalize the filtered quaternion
    go2_low_state_handler.filtered_quat = go2_low_state_handler.filtered_quat / np.linalg.norm(
        go2_low_state_handler.filtered_quat
    )
    
    # Construct and return the parsed states dictionary
    states = {
        "robot/ang_vel_b": go2_low_state_handler.filtered_gyro,
        "robot/projected_gravity_b": projected_gravity_b(go2_low_state_handler.filtered_quat),
        "robot/joint_pos": go2_low_state_handler.filtered_joint_pos-np.array(ISAAC_LAB_DEFAULT_JOINT_POS),
        "robot/joint_vel": go2_low_state_handler.filtered_joint_vel,
    }

    return states

class Go2LowStateHandler:
    def __init__(self):
        self.initialized = False
        self.filtered_joint_pos = np.zeros(12)
        self.filtered_joint_vel = np.zeros(12)
        self.filtered_gyro = np.zeros(3)
        self.filtered_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.alpha = 1.0
    
    def normalize_quaternion(self, quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat)
        if norm > 1e-6:
            return quat / norm
        return quat
    
    def apply_filter(self, filtered: np.ndarray, new_data: np.ndarray) -> np.ndarray:
        return self.alpha * new_data + (1.0 - self.alpha) * filtered
    

class Custom:
    def __init__(self):
        self.stand_up_joint_pos = np.array([
            0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
            0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
        ])
        
        self.stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
        ])
        
        self.dt = 0.002
        self.running_time = 0.0
        self.phase = 0.0
        
        self.neural_network = None
        self.model_loaded = False
        self.last_action = [0.0] * 12
        
        self.obs_history_storage = None
        self.state_dim = 48
        self.max_history_length = 5
        
        self.state_handler = Go2LowStateHandler()
        
        self.lowcmd_publisher = None
        self.lowstate_subscriber = None
        self.crc = CRC()
        self.debug_counter = 0
        
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        
        self.last_call_time = time.time()
    
    def load_neural_network(self, model_path: str) -> bool:
        try:
            self.neural_network = torch.jit.load(model_path)
            self.neural_network.eval()
            
            self.obs_history_storage = ObservationHistoryStorage(
                1, self.state_dim, self.max_history_length, torch.device("cpu"))
            
            self.last_action = [0.0] * 12
            self.model_loaded = True
            
            return True
        except Exception as e:
            print(f"Error loading neural network: {e}")
            self.model_loaded = False
            return False
    
    def states_to_tensor(self, states: Dict[str, List[float]]) -> torch.Tensor:
        concatenated_states = []
        
        for key in STATE_ORDER:
            if key in states:
                concatenated_states.extend(states[key])
            else:
                print(f"Warning: State key '{key}' not found in states map")
        
        tensor = torch.tensor(concatenated_states, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def run_inference(self, input_tensor: torch.Tensor) -> List[float]:
        if not self.model_loaded:
            print("Neural network not loaded!")
            return []
        
        try:
            with torch.no_grad():
                output = self.neural_network(input_tensor)
                output = output.squeeze(0).cpu()
                
                actions = output.tolist()
                return actions
        except Exception as e:
            print(f"Error during neural network inference: {e}")
            return []
    
    def init(self):
        self.init_low_cmd()
        
        model_path = "policies/flatAmpVision/2025-08-12_14-23-30_flat_DR5_improved_minimal_motion_files_SEED_3/policy.pt"
        if not self.load_neural_network(model_path):
            print("Failed to load neural network. Continuing without neural network inference.")
        
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_message_handler, 1)
    
    def init_low_cmd(self):
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
    
    def low_state_message_handler(self, msg: LowState_):
        self.last_call_time = time.time()
        
        self.low_state = msg
        
        states = go2_low_state_handler(msg)
        states["robot/last_action"] = self.last_action.copy()
        states["robot/command"] = np.array([0.0,0.0,0.0])
        states["robot/base_lin_vel"] = np.array([0.0,0.0,0.0])
        
        current_obs = self.states_to_tensor(states)
        
        if self.model_loaded and self.obs_history_storage:
            self.obs_history_storage.add(current_obs)
            history_input = self.obs_history_storage.get()
            
            actions = self.run_inference(history_input)
            self.last_action = actions.copy()
            
            if actions:
                for i in range(len(actions)):
                    actions[i] = clamp(
                        ISAAC_LAB_DEFAULT_JOINT_POS[i] + actions[i] * ACTION_SCALE,  
                        CLIP_ACTION_MIN, 
                        CLIP_ACTION_MAX
                    )
                    
                reordered_actions = [0.0] * 12
                for i in range(12):
                    isaac_lab_idx = JOINT_ISAAC_LAB_TO_UNITREE_MAPPING[i].item()
                    reordered_actions[i] = actions[isaac_lab_idx]
                
                for i in range(12):
                    self.cmd.motor_cmd[i].q = reordered_actions[i]
                    self.cmd.motor_cmd[i].kp = STIFFNESS
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kd = DAMPING
                    self.cmd.motor_cmd[i].tau = 0.0
                
                self.cmd.crc = self.crc.Crc(self.cmd)
                end_time = time.time()

                self.lowcmd_publisher.Write(self.cmd)
                
    
    def low_cmd_write(self):
        step_start = time.perf_counter()
        self.running_time += self.dt
        
        if self.running_time < 3.0:
            self.phase = np.tanh(self.running_time / 1.2)
            for i in range(12):
                self.cmd.motor_cmd[i].q = (self.phase * self.stand_up_joint_pos[i] + 
                                          (1 - self.phase) * self.stand_down_joint_pos[i])
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kp = self.phase * 50.0 + (1 - self.phase) * 20.0
                self.cmd.motor_cmd[i].kd = 3.5
                self.cmd.motor_cmd[i].tau = 0.0
        else:
            self.phase = np.tanh((self.running_time - 3.0) / 1.2)
            for i in range(12):
                self.cmd.motor_cmd[i].q = (self.phase * self.stand_down_joint_pos[i] + 
                                          (1 - self.phase) * self.stand_up_joint_pos[i])
                self.cmd.motor_cmd[i].kp = 50.0
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kd = 3.5
                self.cmd.motor_cmd[i].tau = 0.0
        
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.lowcmd_publisher.Write(self.cmd)
        
        time_until_next_step = self.dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

def main():
    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])
    
    if torch.cuda.is_available():
        print("CUDA is available!")
    else:
        print("CUDA is not available.")
    
    custom = Custom()
    custom.init()
    
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()