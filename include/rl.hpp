#pragma once

#include "types.hpp"
#include "inference_runtime.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>

class FSM;

class RL
{
public:
    explicit RL(const std::filesystem::path& project_root);
    virtual ~RL();

    virtual void GetState() = 0;
    virtual void SetCommand() = 0;

    void InitKeyboard();
    void RestoreKeyboard();
    void KeyboardInterface();

    void InitRL();
    void StateController();

    Observation ComputeObservation() const;
    Joints Forward();

    void ComputeOutput(const Joints& actions);
    void RunModel();
    void RLControl();

    void ResetController();

    FSM& GetFSM();

    BaseParameters params;
    PolicyParameters policy;

    RobotState robot_state;
    RobotCommand robot_command;
    Control control;

    Joints start_pos{};

    bool rl_init_done = false;
    std::uint64_t policy_steps = 0;

protected:
    std::filesystem::path project_root_;

private:
    std::unique_ptr<FSM> fsm_;
    std::unique_ptr<InferenceRuntime::Model> model_;

    int keyboard_escape_ = 0;
    Joints previous_action_{};
    Joints target_pos_{};
};