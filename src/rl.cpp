#include "rl.hpp"
#include "fsm.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cerrno>
#include <cstdlib>
#include <poll.h>
#include <system_error>
#include <termios.h>
#include <unistd.h>

namespace
{

bool terminal_ready = false;
bool terminal_cleanup_registered = false;
termios saved_terminal{};

void RestoreTerminal()
{
    if (terminal_ready)
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &saved_terminal);
        terminal_ready = false;
    }
}

void PrepareTerminal()
{
    if (terminal_ready) return;
    Require(isatty(STDIN_FILENO) == 1,
            "Terminal required: use docker exec -it <container> bash");
    if (!terminal_cleanup_registered)
    {
        Require(std::atexit(RestoreTerminal) == 0, "Cannot register terminal cleanup");
        terminal_cleanup_registered = true;
    }
    if (tcgetattr(STDIN_FILENO, &saved_terminal) != 0)
        throw std::system_error(errno, std::generic_category(), "tcgetattr");
    termios mode = saved_terminal;
    mode.c_lflag &= ~(ICANON | ECHO | ECHONL);
    mode.c_cc[VMIN] = 0;
    mode.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &mode) != 0)
        throw std::system_error(errno, std::generic_category(), "tcsetattr");
    terminal_ready = true;
}

// 与参考框架一样，终端底层读取放在 RL 实现文件内。
int kbhit()
{
    PrepareTerminal();
    pollfd event{STDIN_FILENO, POLLIN, 0};
    const int result = poll(&event, 1, 0);
    if (result < 0)
    {
        if (errno == EINTR) return -1;
        throw std::system_error(errno, std::generic_category(), "terminal poll");
    }
    Require(!(event.revents & (POLLERR | POLLHUP | POLLNVAL)), "Terminal disconnected");
    if (!(event.revents & POLLIN)) return -1;
    unsigned char c;
    const auto count = read(STDIN_FILENO, &c, 1);
    if (count == 1) return c;
    if (count < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
        throw std::system_error(errno, std::generic_category(), "terminal read");
    return -1;
}

float ReadFloat(const YAML::Node& node, const char* key)
{
    const float value = node[key].as<float>();

    Require(std::isfinite(value),
            std::string("Invalid parameter: ") + key);

    return value;
}

template<std::size_t N>
std::array<float, N> ReadArray(
    const YAML::Node& node,
    const char* key)
{
    const auto values = node[key];

    Require(values.IsSequence() && values.size() == N,
            std::string("Invalid array: ") + key);

    std::array<float, N> result{};

    for (std::size_t i = 0; i < N; ++i)
    {
        result[i] = values[i].as<float>();
    }

    RequireFinite(result, key);

    return result;
}

} // namespace

RL::RL(const std::filesystem::path& project_root)
    : project_root_(project_root)
{
    const auto path =
        project_root_ / "policy/bpx/base.yaml";

    const auto node =
        YAML::LoadFile(path.string())["bpx"];

    Require(node.IsMap(), "Missing bpx configuration");

    params.dt = ReadFloat(node, "dt");
    params.decimation = node["decimation"].as<int>();

    params.initial_height = ReadFloat(node, "initial_height");
    params.armature = ReadFloat(node, "armature");

    params.torque_limit = ReadFloat(node, "torque_limit");
    params.fixed_kp = ReadFloat(node, "fixed_kp");
    params.fixed_kd = ReadFloat(node, "fixed_kd");
    params.passive_kd = ReadFloat(node, "passive_kd");

    params.default_pos =
        ReadArray<12>(node, "default_dof_pos");

    Require(
        std::abs(params.dt - 0.005f) < 1e-7f &&
        params.decimation == 4,
        "Current BPX policy requires dt=0.005 and decimation=4");

    Require(params.initial_height > 0,
            "initial_height must be positive");

    Require(params.armature >= 0,
            "armature must be nonnegative");

    Require(
        params.torque_limit > 0 &&
        params.torque_limit <= 30,
        "torque_limit must be in (0,30]");

    Require(
        params.fixed_kp >= 0 &&
        params.fixed_kd >= 0 &&
        params.passive_kd >= 0,
        "Controller gains must be nonnegative");

    robot_state.q = params.default_pos;
    target_pos_ = robot_state.q;

    fsm_ = std::make_unique<FSM>(*this);
}

RL::~RL() { RestoreKeyboard(); }

void RL::InitKeyboard()
{
    PrepareTerminal();
    keyboard_escape_ = 0;
}

void RL::RestoreKeyboard()
{
    RestoreTerminal();
    keyboard_escape_ = 0;
}

void RL::KeyboardInterface()
{
    if (control.current_keyboard != Input::Keyboard::None) return;
    for (int count = 0; count < 32; ++count)
    {
        int c = kbhit();
        if (c < 0) return;
        if (c == 27) { keyboard_escape_ = 1; continue; }
        if (keyboard_escape_ == 1)
        {
            keyboard_escape_ = c == '[' ? 2 : (c == 'O' ? 3 : 0);
            continue;
        }
        if (keyboard_escape_ == 2)
        {
            if (c >= 0x40 && c <= 0x7e) keyboard_escape_ = 0;
            continue;
        }
        if (keyboard_escape_ == 3) { keyboard_escape_ = 0; continue; }
        if (c >= 'a' && c <= 'z') c += 'A' - 'a';
        using K = Input::Keyboard;
        K key = K::None;
        switch (c)
        {
        case '0': key = K::Num0; break;
        case '1': key = K::Num1; break;
        case '9': key = K::Num9; break;
        case 'W': key = K::W; break;
        case 'S': key = K::S; break;
        case 'A': key = K::A; break;
        case 'D': key = K::D; break;
        case 'Q': key = K::Q; break;
        case 'E': key = K::E; break;
        case ' ': key = K::Space; break;
        case 'P': key = K::P; break;
        case 'R': key = K::R; break;
        case 'H': key = K::H; break;
        case '\n': case '\r': key = K::Enter; break;
        default: continue;
        }
        control.SetKeyboard(key);
        return;
    }
}

FSM& RL::GetFSM()
{
    return *fsm_;
}

void RL::InitRL()
{
    BeforeLog();
    const auto directory =
        project_root_ / "policy/bpx/mirrorme";

    const auto node =
        YAML::LoadFile(
            (directory / "config.yaml").string())["bpx/mirrorme"];

    Require(node.IsMap(), "Missing bpx/mirrorme configuration");

    PolicyParameters next;

    next.kp = ReadFloat(node, "kp");
    next.kd = ReadFloat(node, "kd");

    next.action_scale = ReadFloat(node, "action_scale");

    next.clip_obs = ReadFloat(node, "clip_obs");
    next.clip_actions = ReadFloat(node, "clip_actions");

    next.ang_vel_scale = ReadFloat(node, "ang_vel_scale");
    next.dof_vel_scale = ReadFloat(node, "dof_vel_scale");

    next.commands_scale =
        ReadArray<3>(node, "commands_scale");

    Require(next.kp >= 0 && next.kd >= 0,
            "Policy gains must be nonnegative");

    Require(
        next.action_scale > 0 &&
        next.clip_obs > 0 &&
        next.clip_actions > 0,
        "Invalid policy scale or clipping limit");

    const auto model_path =
        directory / node["model_name"].as<std::string>();

    // 先完成加载，再替换当前模型。
    auto next_model =
        InferenceRuntime::ModelFactory::load_model(
            model_path.string());

    policy = next;
    model_ = std::move(next_model);

    previous_action_.fill(0.0f);
    target_pos_ = robot_state.q;

    control.velocity.fill(0.0f);

    policy_steps = 0;
    rl_init_done = true;

    std::cout << "Policy loaded: " << model_path << '\n';
}

void RL::StateController()
{
    using K = Input::Keyboard;
    const auto input = control.current_keyboard;
    if (input == K::P)
    {
        ResetController();
        return;
    }
    Key request = Key::None;
    if (input == K::Num0) request = Key::GetUp;
    if (input == K::Num1) request = Key::Locomotion;
    if (input == K::Num9) request = Key::GetDown;
    fsm_->Run(request);

    // 在公共控制层限制速度输入，sim 和后续 real 共用。
    if (fsm_->Current() != StateID::Locomotion || !rl_init_done)
    {
        control.velocity.fill(0.0f);
        return;
    }

    switch (input)
    {
    case K::W: control.velocity[0] += 0.1f; break;
    case K::S: control.velocity[0] -= 0.1f; break;
    case K::A: control.velocity[1] += 0.1f; break;
    case K::D: control.velocity[1] -= 0.1f; break;
    case K::Q: control.velocity[2] += 0.1f; break;
    case K::E: control.velocity[2] -= 0.1f; break;
    case K::Space: control.velocity.fill(0.0f); break;
    default: break;
    }
    control.velocity[0] = std::clamp(control.velocity[0], -1.5f, 1.5f);
    control.velocity[1] = std::clamp(control.velocity[1], -1.0f, 1.0f);
    control.velocity[2] = std::clamp(control.velocity[2], -2.0f, 2.0f);
    // 命令按 0.1 的网格保存，消除连续加减产生的浮点残差。
    for (float& value : control.velocity)
    {
        value = std::round(value * 10.0f) / 10.0f;
        if (std::abs(value) < 1e-6f) value = 0.0f;
    }
    // 与参考框架一样，由 RobotControl 在本周期末 ClearInput。
}

Observation RL::ComputeObservation() const
{
    RequireFinite(robot_state.q, "Joint position");
    RequireFinite(robot_state.dq, "Joint velocity");
    RequireFinite(robot_state.gyro, "Gyroscope");
    RequireFinite(robot_state.quaternion, "Quaternion");
    RequireFinite(control.velocity, "Command");
    RequireFinite(previous_action_, "Previous action");

    const auto& quat = robot_state.quaternion;

    double norm_squared = 0.0;

    for (float value : quat)
    {
        norm_squared += static_cast<double>(value) * value;
    }

    Require(norm_squared > 1e-12,
            "Invalid quaternion norm");

    const double norm = std::sqrt(norm_squared);

    const double w = quat[0] / norm;
    const double x = quat[1] / norm;
    const double y = quat[2] / norm;
    const double z = quat[3] / norm;

    // R(q)^T * [0,0,-1]，q 为 wxyz。
    const std::array<float, 3> gravity = {
        static_cast<float>(2.0 * (w * y - x * z)),
        static_cast<float>(-2.0 * (w * x + y * z)),
        static_cast<float>(2.0 * (x * x + y * y) - 1.0)
    };

    Observation observation{};
    std::size_t index = 0;

    const auto append = [&](float raw, float scale)
    {
        // 与训练一致：先裁剪原始项，再缩放。
        observation.at(index++) =
            std::clamp(
                raw,
                -policy.clip_obs,
                policy.clip_obs) * scale;
    };

    for (float value : robot_state.gyro)
    {
        append(value, policy.ang_vel_scale);
    }

    for (float value : gravity)
    {
        append(value, 1.0f);
    }

    for (std::size_t i = 0; i < 3; ++i)
    {
        append(
            control.velocity[i],
            policy.commands_scale[i]);
    }

    for (std::size_t i = 0; i < kNumJoints; ++i)
    {
        append(
            robot_state.q[i] - params.default_pos[i],
            1.0f);
    }

    for (float value : robot_state.dq)
    {
        append(value, policy.dof_vel_scale);
    }

    for (float value : previous_action_)
    {
        append(value, 1.0f);
    }

    Require(index == observation.size(),
            "Incorrect observation dimension");

    RequireFinite(observation, "Observation");

    return observation;
}

Joints RL::Forward()
{
    Require(model_ != nullptr, "Policy is not loaded");

    auto actions = model_->forward(ComputeObservation());

    for (float& value : actions)
    {
        value = std::clamp(
            value,
            -policy.clip_actions,
            policy.clip_actions);
    }

    return actions;
}

void RL::ComputeOutput(const Joints& actions)
{
    for (std::size_t i = 0; i < kNumJoints; ++i)
    {
        target_pos_[i] = std::clamp(
            params.default_pos[i] +
                policy.action_scale * actions[i],
            -100.0f,
            100.0f);
    }

    RequireFinite(target_pos_, "Policy target");
}

void RL::RunModel()
{
    Require(rl_init_done, "Policy is not initialized");

    const auto actions = Forward();

    ComputeOutput(actions);

    // 保存裁剪后的原始动作，不保存缩放后的目标角度。
    previous_action_ = actions;

    ++policy_steps;
}

void RL::RLControl()
{
    robot_command.q = target_pos_;
    robot_command.dq.fill(0.0f);
    robot_command.tau.fill(0.0f);

    robot_command.kp = policy.kp;
    robot_command.kd = policy.kd;
}

void RL::ResetController()
{
    rl_init_done = false;

    previous_action_.fill(0.0f);
    target_pos_ = robot_state.q;

    control = Control{};
    policy_steps = 0;

    fsm_->Reset();
}