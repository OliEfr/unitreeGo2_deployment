#include <iostream>
#include <atomic>
#include <thread>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <termios.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include <map>
#include <string>
#include <vector>
#include <chrono>
#include <array>
#include <cmath>
#include <memory>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

// PyTorch
#include <torch/torch.h>
#include <torch/script.h>

using namespace unitree::common;
using namespace unitree::robot;

/////// USER VARS
double action_scale = 10.0;
double clip_action_min = -20.0;
double clip_action_max = 20.0;
double phase_policy_freq = 2;

static constexpr int CALLBACK_MESSAGE_SKIP = 1; // Can be used to only process every n-th lowState message in the callback. Higher value reduces policy inference frequency

static constexpr double OBS_EMA_FILTER_ALPHA = 0.95; // Filter coefficient for ema-filtering the obs

static const int STATE_DIM = 47; // 2 for phase

enum class ControllerType : int
{
    DAMPING = 0,
    STAND_UP = 1,
    NEURAL_NETWORK = 2,
    STAND_DOWN = 3,
};

////// CONSTANTS
#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

const int JOINT_UNITREE_TO_ISAAC_LAB_MAPPING[12] = {3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8};
torch::Tensor JOINT_ISAAC_LAB_TO_UNITREE_MAPPING = torch::tensor({1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10}, torch::kInt64);

static const std::array<double, 12> ISAAC_LAB_DEFAULT_JOINT_POS = {
    0.1000, -0.1000, 0.1000, -0.1000, 0.8000, 0.8000,
    1.0000, 1.0000, -1.5000, -1.5000, -1.5000, -1.5000};

constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

////// CODE

template <typename T>
T clamp(T value, T min_val, T max_val)
{
    return std::max(min_val, std::min(value, max_val));
}

// Forward declaration
class Go2LowStateHandler;

class ObservationHistoryStorage
{
private:
    int num_envs;
    int num_obs;
    int max_length;
    torch::Tensor buffer;

public:
    ObservationHistoryStorage(int num_envs, int num_obs, int max_length, torch::Device device = torch::kCPU)
        : num_envs(num_envs), num_obs(num_obs), max_length(max_length)
    {
        // Initialize buffer with zeros of shape (num_envs, num_obs * max_length)
        buffer = torch::zeros({num_envs, num_obs * max_length}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
    }

    void add(const torch::Tensor &observation)
    {
        // Check observation shape
        if (observation.size(0) != num_envs || observation.size(1) != num_obs)
        {
            throw std::runtime_error("Observation shape must be (" + std::to_string(num_envs) +
                                     ", " + std::to_string(num_obs) + ")");
        }

        // Shift the buffer to make space for the new observation
        buffer.slice(1, 0, -num_obs) = buffer.slice(1, num_obs, buffer.size(1)).clone();

        // Add the new observation at the end
        buffer.slice(1, -num_obs, buffer.size(1)) = observation;
    }

    torch::Tensor get() const
    {
        return buffer.detach().clone();
    }

    void reset(const torch::Tensor &done)
    {
        torch::Tensor done_indices = torch::nonzero(done == 1);
        if (done_indices.numel() > 0)
        {
            buffer.index_fill_(0, done_indices.squeeze(-1), 0.0);
        }
    }
};

// Add this class definition before the Custom class
class PerformanceTimer
{
private:
    struct timespec last_time;
    std::vector<double> time_diffs;
    struct timespec summary_start_time;
    bool first_call;
    int over_5ms_count;

    // Get current time in nanoseconds using clock_gettime (more performant than chrono)
    inline double getCurrentTimeNs()
    {
        struct timespec current_time;
        clock_gettime(CLOCK_MONOTONIC_RAW, &current_time);
        return current_time.tv_sec * 1e9 + current_time.tv_nsec;
    }

public:
    PerformanceTimer() : first_call(true)
    {
        time_diffs.reserve(10000); // Pre-allocate to avoid reallocations
        clock_gettime(CLOCK_MONOTONIC_RAW, &summary_start_time);
    }

    void recordCall()
    {
        if (first_call)
        {
            clock_gettime(CLOCK_MONOTONIC_RAW, &last_time);
            first_call = false;
            return;
        }

        struct timespec current_time;
        clock_gettime(CLOCK_MONOTONIC_RAW, &current_time);

        // Calculate difference in milliseconds
        double time_diff_ms = ((current_time.tv_sec - last_time.tv_sec) * 1000.0) +
                              ((current_time.tv_nsec - last_time.tv_nsec) / 1e6);

        time_diffs.push_back(time_diff_ms);
        if (time_diff_ms > 5.0)
        {
            over_5ms_count++;
        }
        last_time = current_time;

        // Check if 5 seconds have passed for summary
        double elapsed_since_summary = ((current_time.tv_sec - summary_start_time.tv_sec) * 1000.0) +
                                       ((current_time.tv_nsec - summary_start_time.tv_nsec) / 1e6);

        if (elapsed_since_summary >= 5000.0)
        { // 5 seconds in milliseconds
            printSummary();
            resetSummary();
            summary_start_time = current_time;
        }
    }

private:
    void printSummary()
    {
        if (time_diffs.empty())
            return;

        double sum = 0.0;
        double min_val = time_diffs[0];
        double max_val = time_diffs[0];

        for (double diff : time_diffs)
        {
            sum += diff;
            if (diff < min_val)
                min_val = diff;
            if (diff > max_val)
                max_val = diff;
        }

        double mean = sum / time_diffs.size();

        std::cout << "Command timing summary (5s): Mean=" << std::fixed << std::setprecision(3)
                  << mean << "ms, Min=" << min_val << "ms, Max=" << max_val
                  << "ms, Count=" << time_diffs.size() << ", Over5ms=" << over_5ms_count << std::endl;
    }

    void resetSummary()
    {
        time_diffs.clear();
        over_5ms_count = 0;
    }
};

class Custom
{
public:
    Custom() {};
    ~Custom()
    {
        // Clean shutdown
        should_exit.store(true);
        if (input_thread.joinable())
        {
            input_thread.join();
        }
    };
    void Init();
    bool LoadNeuralNetwork(const std::string &model_path);

private:
    void InitLowCmd();
    void LowStateMessageHandler(const void *messages);
    void LowCmdWrite();
    torch::Tensor StatesToTensor(const std::map<std::string, std::vector<double>> &states);
    std::vector<double> RunInference(const torch::Tensor &input_tensor);

    std::unique_ptr<ObservationHistoryStorage> obs_history_storage;
    static constexpr int max_history_length = 5;

    PerformanceTimer cmd_timer;

    torch::Tensor reused_state_tensor;
    std::vector<double> reused_state_vector;
    bool tensors_initialized = false;

    // Controller switching variables
    std::atomic<int> current_controller{static_cast<int>(ControllerType::DAMPING)};
    std::atomic<bool> should_exit{false};

    // Controller state variables
    ControllerType active_controller = ControllerType::DAMPING;
    double controller_switch_time = 0.0;
    std::array<double, 12> switch_start_positions{};
    bool position_captured = false;

    // Input thread
    std::thread input_thread;

    // Controller switching methods
    void startInputThread();
    void inputThreadFunction();

    // phase
    double phase_policy = 0.0;
    struct timespec last_inference_time;
    bool first_inference_call = true;

private:
    double stand_up_joint_pos[12] = {0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763, 0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763};
    double stand_down_joint_pos[12] = {0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375};
    double dt = 0.002;
    double runing_time = 0.0;
    double phase = 0.0;

    // Neural network variables
    torch::jit::script::Module neural_network;
    bool model_loaded = false;
    std::vector<double> last_action; // Store last action for next iteration

    unitree_go::msg::dds_::LowCmd_ low_cmd{};     // default init
    unitree_go::msg::dds_::LowState_ low_state{}; // default init

    /*publisher*/
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    /*subscriber*/
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;

    std::atomic<double> cmd_x{0.0};
    std::atomic<double> cmd_y{0.0};
    std::atomic<double> cmd_yaw{0.0};

    /*LowCmd write thread*/
    // ThreadPtr lowCmdWriteThreadPtr;
};

// Function to rotate a vector by the inverse of a quaternion
std::array<double, 3> quat_rotate_inverse(const std::array<double, 4> &q, const std::array<double, 3> &v)
{
    // Assuming quaternion order is [w, x, y, z] (wxyz)
    double q_w = q[0];
    double q_x = q[1];
    double q_y = q[2];
    double q_z = q[3];

    double v_x = v[0];
    double v_y = v[1];
    double v_z = v[2];

    double factor_a = 2.0 * q_w * q_w - 1.0;
    std::array<double, 3> a = {v_x * factor_a, v_y * factor_a, v_z * factor_a};

    double cross_x = q_y * v_z - q_z * v_y;
    double cross_y = q_z * v_x - q_x * v_z;
    double cross_z = q_x * v_y - q_y * v_x;

    double factor_b = q_w * 2.0;
    std::array<double, 3> b = {cross_x * factor_b, cross_y * factor_b, cross_z * factor_b};

    double dot_product = q_x * v_x + q_y * v_y + q_z * v_z;
    double factor_c = dot_product * 2.0;
    std::array<double, 3> c = {q_x * factor_c, q_y * factor_c, q_z * factor_c};

    // Return a - b + c
    return {a[0] - b[0] + c[0], a[1] - b[1] + c[1], a[2] - b[2] + c[2]};
}

// Function to calculate projected gravity vector
std::array<double, 3> projected_gravity_b(const std::array<double, 4> &base_quat)
{
    // Gravity direction vector [0, 0, -1]
    std::array<double, 3> gravity_dir = {0.0, 0.0, -1.0};

    // Rotate gravity direction by inverse of base quaternion
    return quat_rotate_inverse(base_quat, gravity_dir);
}

class Go2LowStateHandler
{
private:
    // Static variables to store filtered states
    static bool initialized;
    static std::array<double, 12> filtered_joint_pos;
    static std::array<double, 12> filtered_joint_vel;
    static std::array<double, 12> filtered_joint_acc;
    static std::array<double, 12> filtered_joint_tau;
    static std::array<double, 3> filtered_gyro;
    static std::array<double, 3> filtered_acc;
    static std::array<double, 4> filtered_quat;

    // Helper function to normalize quaternion
    static void normalizeQuaternion(std::array<double, 4> &quat)
    {
        double norm = std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] +
                                quat[2] * quat[2] + quat[3] * quat[3]);
        if (norm > 1e-6)
        {
            for (int i = 0; i < 4; ++i)
            {
                quat[i] /= norm;
            }
        }
    }

    // Helper function for exponential moving average filtering
    template <size_t N>
    static void applyFilter(std::array<double, N> &filtered,
                            const std::array<double, N> &new_data)
    {
        for (size_t i = 0; i < N; ++i)
        {
            filtered[i] = OBS_EMA_FILTER_ALPHA * new_data[i] + (1.0 - OBS_EMA_FILTER_ALPHA) * filtered[i];
        }
    }

public:
    /**
     * Extracts and filters the low-level state of the robot.
     *
     * @param low_state The LowState message from the robot
     * @return Map containing filtered low-level state of the robot
     */
    static std::map<std::string, std::vector<double>>
    processLowState(const unitree_go::msg::dds_::LowState_ &low_state)
    {

        // Extract motor states (first 12 joints for legs)
        std::array<double, 12> joint_positions;
        std::array<double, 12> joint_velocities;
        std::array<double, 12> joint_accelerations;
        std::array<double, 12> joint_tau_est;

        for (int i = 0; i < 12; ++i)
        {
            int unitree_index = JOINT_UNITREE_TO_ISAAC_LAB_MAPPING[i];
            joint_positions[i] = low_state.motor_state()[unitree_index].q() - ISAAC_LAB_DEFAULT_JOINT_POS[i];
            joint_velocities[i] = low_state.motor_state()[unitree_index].dq();
            // joint_accelerations[i] = low_state.motor_state()[unitree_index].ddq();
            // joint_tau_est[i] = low_state.motor_state()[unitree_index].tau_est();
        }

        // Extract foot forces
        // std::array<double, 4> foot_forces;
        // std::array<double, 4> foot_forces_est;
        // for (int i = 0; i < 4; ++i)
        // {
        //     foot_forces[i] = low_state.foot_force()[i];
        //     foot_forces_est[i] = low_state.foot_force_est()[i];
        // }

        // Extract IMU data
        std::array<double, 3> gyroscope = {
            low_state.imu_state().gyroscope()[0],
            low_state.imu_state().gyroscope()[1],
            low_state.imu_state().gyroscope()[2]};

        std::array<double, 3> accelerometer = {
            low_state.imu_state().accelerometer()[0],
            low_state.imu_state().accelerometer()[1],
            low_state.imu_state().accelerometer()[2]};

        std::array<double, 4> quaternion = {
            low_state.imu_state().quaternion()[0],
            low_state.imu_state().quaternion()[1],
            low_state.imu_state().quaternion()[2],
            low_state.imu_state().quaternion()[3]};

        std::array<double, 3> proj_gravity = projected_gravity_b(quaternion);

        // Initialize or update filtered states
        if (!initialized)
        {
            filtered_joint_pos = joint_positions;
            filtered_joint_vel = joint_velocities;
            // filtered_joint_acc = joint_accelerations;
            // filtered_joint_tau = joint_tau_est;
            filtered_gyro = gyroscope;
            // filtered_acc = accelerometer;
            filtered_quat = quaternion;
            initialized = true;
        }
        else
        {
            applyFilter(filtered_joint_pos, joint_positions);
            applyFilter(filtered_joint_vel, joint_velocities);
            // applyFilter(filtered_joint_acc, joint_accelerations);
            // applyFilter(filtered_joint_tau, joint_tau_est);
            applyFilter(filtered_gyro, gyroscope);
            // applyFilter(filtered_acc, accelerometer);
            applyFilter(filtered_quat, quaternion);
        }

        // Normalize the filtered quaternion
        normalizeQuaternion(filtered_quat);

        // Construct and return the parsed states map
        std::map<std::string, std::vector<double>> states;

        states["robot/base_lin_vel"] = std::vector<double>{0.0, 0.0, 0.0};
        states["robot/ang_vel_b"] = std::vector<double>(filtered_gyro.begin(), filtered_gyro.end());
        states["robot/command"] = std::vector<double>{0.0, 0.0, 0.0};
        states["robot/projected_gravity_b"] = std::vector<double>(proj_gravity.begin(), proj_gravity.end());
        states["robot/joint_pos"] = std::vector<double>(filtered_joint_pos.begin(), filtered_joint_pos.end());
        states["robot/joint_vel"] = std::vector<double>(filtered_joint_vel.begin(), filtered_joint_vel.end());
        states["robot/last_action"] = std::vector<double>(12, 0.0);
        // states["robot/joint_acc"] = std::vector<double>(filtered_joint_acc.begin(), filtered_joint_acc.end());
        // states["robot/joint_tau_est"] = std::vector<double>(filtered_joint_tau.begin(), filtered_joint_tau.end());
        //  states["robot/foot_forces"] = std::vector<double>(foot_forces.begin(), foot_forces.end());
        // states["robot/foot_forces_est"] = std::vector<double>(foot_forces_est.begin(), foot_forces_est.end());
        // states["robot/accelerometer"] = std::vector<double>(filtered_acc.begin(), filtered_acc.end());
        //  states["robot/base_quat"] = std::vector<double>(filtered_quat.begin(), filtered_quat.end());

        return states;
    }

    // Optional: Reset the filter state
    static void resetFilter()
    {
        initialized = false;
    }
};

bool Go2LowStateHandler::initialized = false;
std::array<double, 12> Go2LowStateHandler::filtered_joint_pos = {};
std::array<double, 12> Go2LowStateHandler::filtered_joint_vel = {};
std::array<double, 12> Go2LowStateHandler::filtered_joint_acc = {};
std::array<double, 12> Go2LowStateHandler::filtered_joint_tau = {};
std::array<double, 3> Go2LowStateHandler::filtered_gyro = {};
std::array<double, 3> Go2LowStateHandler::filtered_acc = {};
std::array<double, 4> Go2LowStateHandler::filtered_quat = {};

uint32_t crc32_core(uint32_t *ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }

    return CRC32;
}

bool Custom::LoadNeuralNetwork(const std::string &model_path)
{
    try
    {
        // Load the TorchScript model
        neural_network = torch::jit::load(model_path);
        neural_network.eval(); // Set to evaluation mode

        // Apply optimizations
        torch::jit::optimize_for_inference(neural_network);

        // Initialize observation history storage (num_envs=1 for single robot)
        obs_history_storage = std::make_unique<ObservationHistoryStorage>(
            1, STATE_DIM, max_history_length, torch::kCPU);

        // Initialize last_action with appropriate size
        last_action = std::vector<double>(12, 0.0);

        reused_state_vector.reserve(STATE_DIM); // Pre-allocate vector
        reused_state_tensor = torch::zeros({1, STATE_DIM}, torch::dtype(torch::kFloat32));
        tensors_initialized = true;

        model_loaded = true;
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading neural network: " << e.what() << std::endl;
        model_loaded = false;
        return false;
    }
}

torch::Tensor Custom::StatesToTensor(const std::map<std::string, std::vector<double>> &states)
{
    reused_state_vector.clear();

    // ang_vel_b (3 values)
    const auto &ang_vel = states.at("robot/ang_vel_b");
    reused_state_vector.insert(reused_state_vector.end(), ang_vel.begin(), ang_vel.end());

    // command (3 values)
    const auto &command = states.at("robot/command");
    reused_state_vector.insert(reused_state_vector.end(), command.begin(), command.end());

    // projected_gravity_b (3 values)
    const auto &proj_grav = states.at("robot/projected_gravity_b");
    reused_state_vector.insert(reused_state_vector.end(), proj_grav.begin(), proj_grav.end());

    // joint_pos (12 values)
    const auto &joint_pos = states.at("robot/joint_pos");
    reused_state_vector.insert(reused_state_vector.end(), joint_pos.begin(), joint_pos.end());

    // joint_vel (12 values)
    const auto &joint_vel = states.at("robot/joint_vel");
    reused_state_vector.insert(reused_state_vector.end(), joint_vel.begin(), joint_vel.end());

    // last_action (12 values)
    const auto &last_act = states.at("robot/last_action");
    reused_state_vector.insert(reused_state_vector.end(), last_act.begin(), last_act.end());

    // phase (2 values)
    const auto &sin_cos_phase = states.at("robot/sin_cos_phase");
    reused_state_vector.insert(reused_state_vector.end(), sin_cos_phase.begin(), sin_cos_phase.end());

    // Copy data into pre-allocated tensor
    std::copy(reused_state_vector.begin(), reused_state_vector.end(),
              reused_state_tensor.data_ptr<float>());

    return reused_state_tensor;
}
std::vector<double> Custom::RunInference(const torch::Tensor &input_tensor)
{
    std::vector<double> actions;

    if (!model_loaded)
    {
        std::cerr << "Neural network not loaded!" << std::endl;
        return actions;
    }

    try
    {
        // Prepare input for the model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        // Run inference
        torch::NoGradGuard no_grad; // Disable gradient computation for inference

        at::Tensor output = neural_network.forward(inputs).toTensor();

        // Convert output tensor to vector
        output = output.squeeze(0); // Remove batch dimension
        output = output.cpu();      // Ensure tensor is on CPU

        actions.resize(output.size(0));

        float *float_data = output.data_ptr<float>();
        for (int i = 0; i < output.size(0); ++i)
        {
            actions[i] = static_cast<double>(float_data[i]);
        }

        return actions;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during neural network inference: " << e.what() << std::endl;
        return actions;
    }
}

void Custom::Init()
{
    InitLowCmd();

    // Load the neural network model
    std::string model_path = "policies/TORQUE_styleRew_25_SEED_42/policy.pt";
    if (!LoadNeuralNetwork(model_path))
    {
        std::cerr << "Failed to load neural network. Continuing without neural network inference." << std::endl;
    }

    /*create publisher*/
    lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    lowcmd_publisher->InitChannel();

    /*create subscriber*/
    lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(std::bind(&Custom::LowStateMessageHandler, this, std::placeholders::_1), 1);

    // Start input thread for controller switching
    startInputThread();

    /*loop publishing thread*/
    // lowCmdWriteThreadPtr = CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, int(dt * 1000000), &Custom::LowCmdWrite, this);
}

void Custom::InitLowCmd()
{
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    for (int i = 0; i < 20; i++)
    {
        low_cmd.motor_cmd()[i].mode() = (0x01); // motor switch to servo (PMSM) mode
        low_cmd.motor_cmd()[i].q() = (0);       // (PosStopF);
        low_cmd.motor_cmd()[i].kp() = (0);
        low_cmd.motor_cmd()[i].dq() = (0); //(VelStopF);
        low_cmd.motor_cmd()[i].kd() = (0);
        low_cmd.motor_cmd()[i].tau() = (0);
    }
}

void Custom::LowStateMessageHandler(const void *message)
{
    static int msg_loop_counter = 0;
    msg_loop_counter++;

    if (msg_loop_counter % CALLBACK_MESSAGE_SKIP != 0)
    {
        return;
    }

    low_state = *(unitree_go::msg::dds_::LowState_ *)message;

    // Check for controller switch and handle switching
    int requested_controller = current_controller.load();
    if (requested_controller != static_cast<int>(active_controller))
    {
        active_controller = static_cast<ControllerType>(requested_controller);
        controller_switch_time = 0.0;
        position_captured = false;

        if (active_controller == ControllerType::NEURAL_NETWORK)
        {
            phase_policy = 0.0;
            first_inference_call = true;
            std::cout << "Phase resetted." << std::endl;
        }
    }

    // Execute current controller
    switch (active_controller)
    {
    case ControllerType::DAMPING:
    {
        // Set position control with damping
        for (int i = 0; i < 12; i++)
        {
            low_cmd.motor_cmd()[i].q() = 0;
            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = 0.0;
            low_cmd.motor_cmd()[i].kd() = 2.;
            low_cmd.motor_cmd()[i].tau() = 0;
        }

        low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
        lowcmd_publisher->Write(low_cmd);
        break;
    }

    case ControllerType::STAND_DOWN:
    {
        // Capture current position on first call
        if (!position_captured)
        {
            for (int i = 0; i < 12; i++)
            {
                switch_start_positions[i] = low_state.motor_state()[i].q();
            }
            position_captured = true;
        }

        // Smooth transition to stand up position (1.2 second transition)
        controller_switch_time += dt;
        double phase = tanh(controller_switch_time / 1.2);

        for (int i = 0; i < 12; i++)
        {
            double target_pos = phase * stand_down_joint_pos[i] + (1.0 - phase) * switch_start_positions[i];

            low_cmd.motor_cmd()[i].q() = target_pos;
            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = 50;
            low_cmd.motor_cmd()[i].kd() = 3.5;
            low_cmd.motor_cmd()[i].tau() = 0;
        }

        low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
        lowcmd_publisher->Write(low_cmd);
        break;
    }

    case ControllerType::STAND_UP:
    {
        // Capture current position on first call
        if (!position_captured)
        {
            for (int i = 0; i < 12; i++)
            {
                switch_start_positions[i] = low_state.motor_state()[i].q();
            }
            position_captured = true;
        }

        // Smooth transition to stand up position (1.2 second transition)
        controller_switch_time += dt;
        double phase = tanh(controller_switch_time / 1.2);

        for (int i = 0; i < 12; i++)
        {
            double target_pos = phase * stand_up_joint_pos[i] + (1.0 - phase) * switch_start_positions[i];

            low_cmd.motor_cmd()[i].q() = target_pos;
            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = phase * 50.0 + (1 - phase) * 25.0;
            low_cmd.motor_cmd()[i].kd() = 3.5;
            low_cmd.motor_cmd()[i].tau() = 0;
        }

        low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
        lowcmd_publisher->Write(low_cmd);
        break;
    }

    case ControllerType::NEURAL_NETWORK:
    {

        auto states = Go2LowStateHandler::processLowState(low_state);

        struct timespec current_time;
        clock_gettime(CLOCK_MONOTONIC_RAW, &current_time);
        double dt_inference = 0.0;

        if (first_inference_call)
        {
            first_inference_call = false;
        }
        else
        {
            dt_inference = ((current_time.tv_sec - last_inference_time.tv_sec)) +
                           ((current_time.tv_nsec - last_inference_time.tv_nsec) / 1e9);
        }
        last_inference_time = current_time;

        phase_policy += dt_inference * 2.0 * M_PI * phase_policy_freq;
        phase_policy = fmod(phase_policy, 2.0 * M_PI); // Keep phase within [0, 2*pi]
        states["robot/sin_cos_phase"] = {sin(phase_policy), cos(phase_policy)};

        states["robot/command"] = std::vector<double>{cmd_x.load(), cmd_y.load(), cmd_yaw.load()};
        states["robot/last_action"] = last_action;

        torch::Tensor current_obs = StatesToTensor(states);

        if (model_loaded && obs_history_storage)
        {
            obs_history_storage->add(current_obs);
            torch::Tensor history_input = obs_history_storage->get();

            std::vector<double> actions = RunInference(history_input);
            last_action = actions;

            if (!actions.empty())
            {
                std::vector<double> reordered_actions(12);
                for (int i = 0; i < 12; ++i)
                {
                    int isaac_lab_idx = JOINT_ISAAC_LAB_TO_UNITREE_MAPPING.data_ptr<int64_t>()[i];
                    reordered_actions[i] = actions[isaac_lab_idx];
                }

                for (double &action : reordered_actions)
                {
                    action = clamp(action * action_scale, clip_action_min, clip_action_max);
                }

                // Torque control mode
                for (int i = 0; i < 12; i++)
                {
                    low_cmd.motor_cmd()[i].q() = 0;
                    low_cmd.motor_cmd()[i].kp() = 0;
                    low_cmd.motor_cmd()[i].dq() = 0;
                    low_cmd.motor_cmd()[i].kd() = 0;
                    low_cmd.motor_cmd()[i].tau() = reordered_actions[i];
                }

                cmd_timer.recordCall();

                low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
                lowcmd_publisher->Write(low_cmd);
            }
        }
        break;
    }
    }
}

void Custom::startInputThread()
{
    input_thread = std::thread(&Custom::inputThreadFunction, this);
}

void Custom::inputThreadFunction()
{
    std::cout << "\nController Commands:\n";
    std::cout << "i - Idle Damping Controller\n";
    std::cout << "u - Stand Up Controller\n";
    std::cout << "l - Stand Down (Low) Controller\n";
    std::cout << "n - Neural Network Controller\n\n";
    std::cout << "Movement Commands (for Neural Network mode):\n";
    std::cout << "w/s - Forward/Backward (±0.1)\n";
    std::cout << "a/d - Left/Right (±0.1)\n";
    std::cout << "q/e - Rotate Left/Right (±0.1)\n";
    std::cout << "r - Reset commands to zero\n\n";

    // Set stdin to non-blocking AND non-canonical mode
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

    struct termios old_termios, new_termios;
    tcgetattr(STDIN_FILENO, &old_termios);
    new_termios = old_termios;
    new_termios.c_lflag &= ~(ICANON | ECHO); // Disable canonical mode and echo
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);

    char cmd;
    while (!should_exit.load())
    {
        // Non-blocking read
        if (read(STDIN_FILENO, &cmd, 1) > 0)
        {
            switch (cmd)
            {
            // Controller switching
            case 'i':
            case 'I':
                current_controller.store(static_cast<int>(ControllerType::DAMPING));
                std::cout << "Switched to Idle Damping Controller\n";
                break;
            case 'u':
            case 'U':
                current_controller.store(static_cast<int>(ControllerType::STAND_UP));
                std::cout << "Switched to Stand Up Controller\n";
                break;
            case 'l':
            case 'L':
                current_controller.store(static_cast<int>(ControllerType::STAND_DOWN));
                std::cout << "Switched to Stand Down Controller\n";
                break;
            case 'n':
            case 'N':
                current_controller.store(static_cast<int>(ControllerType::NEURAL_NETWORK));
                // Reset commands when switching to Neural Network controller
                cmd_x.store(0.0);
                cmd_y.store(0.0);
                cmd_yaw.store(0.0);
                std::cout << "Switched to Neural Network Controller\n";
                std::cout << "Commands reset to [0.0, 0.0, 0.0]\n";
                break;

            // Movement commands - ONLY work in Neural Network mode
            case 'w':
            case 'W': // Forward
                if (current_controller.load() == static_cast<int>(ControllerType::NEURAL_NETWORK))
                {
                    cmd_x.store(std::min(cmd_x.load() + 0.1, 0.9));
                    std::cout << "Command: [" << std::fixed << std::setprecision(1)
                              << cmd_x.load() << ", " << cmd_y.load() << ", " << cmd_yaw.load() << "]\n";
                }
                break;
            case 's':
            case 'S': // Backward
                if (current_controller.load() == static_cast<int>(ControllerType::NEURAL_NETWORK))
                {
                    cmd_x.store(std::max(cmd_x.load() - 0.1, -0.9));
                    std::cout << "Command: [" << std::fixed << std::setprecision(1)
                              << cmd_x.load() << ", " << cmd_y.load() << ", " << cmd_yaw.load() << "]\n";
                }
                break;
            case 'a':
            case 'A': // Left
                if (current_controller.load() == static_cast<int>(ControllerType::NEURAL_NETWORK))
                {
                    cmd_y.store(std::min(cmd_y.load() + 0.1, 0.2));
                    std::cout << "Command: [" << std::fixed << std::setprecision(1)
                              << cmd_x.load() << ", " << cmd_y.load() << ", " << cmd_yaw.load() << "]\n";
                }
                break;
            case 'd':
            case 'D': // Right
                if (current_controller.load() == static_cast<int>(ControllerType::NEURAL_NETWORK))
                {
                    cmd_y.store(std::max(cmd_y.load() - 0.1, -0.2));
                    std::cout << "Command: [" << std::fixed << std::setprecision(1)
                              << cmd_x.load() << ", " << cmd_y.load() << ", " << cmd_yaw.load() << "]\n";
                }
                break;
            case 'q':
            case 'Q': // Rotate left
                if (current_controller.load() == static_cast<int>(ControllerType::NEURAL_NETWORK))
                {
                    cmd_yaw.store(std::min(cmd_yaw.load() + 0.1, 1.4));
                    std::cout << "Command: [" << std::fixed << std::setprecision(1)
                              << cmd_x.load() << ", " << cmd_y.load() << ", " << cmd_yaw.load() << "]\n";
                }
                break;
            case 'e':
            case 'E': // Rotate right
                if (current_controller.load() == static_cast<int>(ControllerType::NEURAL_NETWORK))
                {
                    cmd_yaw.store(std::max(cmd_yaw.load() - 0.1, -1.4));
                    std::cout << "Command: [" << std::fixed << std::setprecision(1)
                              << cmd_x.load() << ", " << cmd_y.load() << ", " << cmd_yaw.load() << "]\n";
                }
                break;
            case 'r':
            case 'R': // Reset commands
                if (current_controller.load() == static_cast<int>(ControllerType::NEURAL_NETWORK))
                {
                    cmd_x.store(0.0);
                    cmd_y.store(0.0);
                    cmd_yaw.store(0.0);
                    std::cout << "Commands reset to [0.0, 0.0, 0.0]\n";
                }
                break;
            }
        }

        // Sleep for 100ms to reduce CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Restore terminal settings
    tcsetattr(STDIN_FILENO, TCSANOW, &old_termios);
    fcntl(STDIN_FILENO, F_SETFL, flags);
}

void Custom::LowCmdWrite()
{

    runing_time += dt;
    if (runing_time < 3.0)
    {
        // Stand up in first 3 second

        // Total time for standing up or standing down is about 1.2s
        phase = tanh(runing_time / 1.2);
        for (int i = 0; i < 12; i++)
        {
            low_cmd.motor_cmd()[i].q() = phase * stand_up_joint_pos[i] + (1 - phase) * stand_down_joint_pos[i];
            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = phase * 50.0 + (1 - phase) * 20.0;
            low_cmd.motor_cmd()[i].kd() = 3.5;
            low_cmd.motor_cmd()[i].tau() = 0;
        }
    }
    else
    {
        // Then stand down
        phase = tanh((runing_time - 3.0) / 1.2);
        for (int i = 0; i < 12; i++)
        {
            low_cmd.motor_cmd()[i].q() = phase * stand_down_joint_pos[i] + (1 - phase) * stand_up_joint_pos[i];
            low_cmd.motor_cmd()[i].dq() = 0;
            low_cmd.motor_cmd()[i].kp() = 50;
            low_cmd.motor_cmd()[i].kd() = 3.5;
            low_cmd.motor_cmd()[i].tau() = 0;
        }
    }

    low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
    lowcmd_publisher->Write(low_cmd);
}

int main(int argc, const char **argv)
{
    if (argc < 2)
    {
        ChannelFactory::Instance()->Init(1, "lo");
    }
    else
    {
        ChannelFactory::Instance()->Init(0, argv[1]);
    }
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available!" << std::endl;
    }
    else
    {
        std::cout << "CUDA is not available." << std::endl;
    }
    Custom custom;
    custom.Init();

    std::cout << "System initialized. Robot controller is running...\n";
    std::cout << "Starting with Damping Controller (robot will hold current position)\n";

    while (1)
    {
        sleep(10);
    }

    return 0;
}