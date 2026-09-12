#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

inline constexpr std::size_t kNumJoints = 12;
inline constexpr std::size_t kNumObservations = 45;

using Joints = std::array<float, kNumJoints>;
using Observation = std::array<float, kNumObservations>;

inline constexpr std::array<const char*, kNumJoints> kJointNames = {
    "fl_hip_roll_joint",
    "fr_hip_roll_joint",
    "hl_hip_roll_joint",
    "hr_hip_roll_joint",

    "fl_hip_pitch_joint",
    "fr_hip_pitch_joint",
    "hl_hip_pitch_joint",
    "hr_hip_pitch_joint",

    "fl_knee_joint",
    "fr_knee_joint",
    "hl_knee_joint",
    "hr_knee_joint"
};

inline void Require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

template<std::size_t N>
inline void RequireFinite(
    const std::array<float, N>& values,
    const std::string& name)
{
    for (float value : values)
    {
        Require(std::isfinite(value), name + " contains NaN/Inf");
    }
}

struct RobotState
{
    Joints q{};
    Joints dq{};

    // wxyz，机体相对于世界的姿态。
    std::array<float, 4> quaternion{1.0f, 0.0f, 0.0f, 0.0f};

    // 机体坐标系角速度。
    std::array<float, 3> gyro{};
};

struct RobotCommand
{
    Joints q{};
    Joints dq{};
    Joints tau{};

    float kp = 0.0f;
    float kd = 0.0f;
};

enum class Key
{
    None,
    GetUp,
    Locomotion,
    GetDown,
    Passive
};

namespace Input
{
enum class Keyboard
{
    None, Num0, Num1, Num9,
    W, S, A, D, Q, E, Space,
    P, R, Enter, H
};
}

struct Control
{
    Input::Keyboard current_keyboard = Input::Keyboard::None;
    std::array<float, 3> velocity{};

    void SetKeyboard(Input::Keyboard key) { current_keyboard = key; }
    void ClearInput() { current_keyboard = Input::Keyboard::None; }
};

struct BaseParameters
{
    float dt = 0.005f;
    int decimation = 4;

    float initial_height = 0.48f;
    float armature = 0.003f;

    float torque_limit = 30.0f;
    float fixed_kp = 30.0f;
    float fixed_kd = 1.0f;
    float passive_kd = 1.0f;

    Joints default_pos{};
};

struct PolicyParameters
{
    float kp = 30.0f;
    float kd = 1.0f;

    float action_scale = 0.25f;

    float clip_obs = 100.0f;
    float clip_actions = 100.0f;

    float ang_vel_scale = 0.25f;
    float dof_vel_scale = 0.05f;

    std::array<float, 3> commands_scale{2.0f, 2.0f, 0.25f};
};