#include "fsm.hpp"
#include "rl.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>

namespace
{

std::size_t Index(StateID id)
{
    return static_cast<std::size_t>(id);
}

class Passive : public RLFSMState
{
public:
    explicit Passive(RL& rl)
        : RLFSMState(rl, StateID::Passive)
    {
    }

    void Enter() override
    {
        rl.rl_init_done = false;
        Run();
    }

    void Run() override
    {
        SetTarget(
            rl.robot_state.q,
            0.0f,
            rl.params.passive_kd);
    }

    StateID CheckChange(Key key) override
    {
        return key == Key::GetUp
            ? StateID::GetUp
            : id_;
    }
};

class GetUp : public RLFSMState
{
public:
    explicit GetUp(RL& rl)
        : RLFSMState(rl, StateID::GetUp)
    {
    }

    void Enter() override
    {
        rl.rl_init_done = false;

        from_ = rl.robot_state.q;
        rl.start_pos = from_;

        elapsed_ = 0.0f;
    }

    void Run() override
    {
        elapsed_ += rl.params.dt;

        Interpolate(
            from_,
            rl.params.default_pos,
            elapsed_,
            2.0f);
    }

    StateID CheckChange(Key key) override
    {
        if (elapsed_ >= 2.0f)
        {
            if (key == Key::Locomotion)
            {
                return StateID::Locomotion;
            }

            if (key == Key::GetDown)
            {
                return StateID::GetDown;
            }
        }

        return id_;
    }

private:
    Joints from_{};
    float elapsed_ = 0.0f;
};

class GetDown : public RLFSMState
{
public:
    explicit GetDown(RL& rl)
        : RLFSMState(rl, StateID::GetDown)
    {
    }

    void Enter() override
    {
        rl.rl_init_done = false;
        from_ = rl.robot_state.q;
        elapsed_ = 0.0f;
    }

    void Run() override
    {
        elapsed_ += rl.params.dt;

        Interpolate(
            from_,
            rl.start_pos,
            elapsed_,
            2.0f);
    }

    StateID CheckChange(Key key) override
    {
        if (elapsed_ >= 2.0f)
        {
            return StateID::Passive;
        }

        return key == Key::GetUp
            ? StateID::GetUp
            : id_;
    }

private:
    Joints from_{};
    float elapsed_ = 0.0f;
};

class RLLocomotion : public RLFSMState
{
public:
    explicit RLLocomotion(RL& rl)
        : RLFSMState(rl, StateID::Locomotion)
    {
    }

    void Enter() override
    {
        rl.InitRL();
    }

    void Run() override
    {
        Require(rl.rl_init_done, "RL state is not initialized");

        rl.RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    StateID CheckChange(Key key) override
    {
        if (key == Key::GetUp)
        {
            return StateID::GetUp;
        }

        if (key == Key::GetDown)
        {
            return StateID::GetDown;
        }

        return id_;
    }
};

} // namespace

void RLFSMState::SetTarget(
    const Joints& target,
    float kp,
    float kd)
{
    rl.robot_command.q = target;
    rl.robot_command.dq.fill(0.0f);
    rl.robot_command.tau.fill(0.0f);

    rl.robot_command.kp = kp;
    rl.robot_command.kd = kd;
}

void RLFSMState::Interpolate(
    const Joints& from,
    const Joints& to,
    float elapsed,
    float duration)
{
    const float ratio =
        std::clamp(elapsed / duration, 0.0f, 1.0f);

    Joints target{};

    for (std::size_t i = 0; i < kNumJoints; ++i)
    {
        target[i] =
            (1.0f - ratio) * from[i] + ratio * to[i];
    }

    SetTarget(
        target,
        rl.params.fixed_kp,
        rl.params.fixed_kd);
}

FSM::FSM(RL& context)
{
    states_[Index(StateID::Passive)] =
        std::make_unique<Passive>(context);

    states_[Index(StateID::GetUp)] =
        std::make_unique<GetUp>(context);

    states_[Index(StateID::GetDown)] =
        std::make_unique<GetDown>(context);

    states_[Index(StateID::Locomotion)] =
        std::make_unique<RLLocomotion>(context);

    Reset();
}

const char* FSM::Name() const
{
    static constexpr std::array<const char*, 4> names = {
        "Passive",
        "GetUp",
        "GetDown",
        "RLLocomotion"
    };

    return names[Index(current_)];
}

void FSM::Reset()
{
    states_[Index(current_)]->Exit();

    current_ = StateID::Passive;
    states_[Index(current_)]->Enter();
}

void FSM::Change(StateID next)
{
    states_[Index(current_)]->Exit();
    current_ = next;

    try
    {
        states_[Index(current_)]->Enter();
        std::cout << "FSM -> " << Name() << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "FSM Enter failed: "
            << error.what() << '\n';

        Reset();
    }
}

void FSM::Run(Key key)
{
    if (key == Key::Passive)
    {
        Reset();
        return;
    }

    try
    {
        auto& state = states_[Index(current_)];

        state->Run();

        const StateID next = state->CheckChange(key);

        if (next != current_)
        {
            Change(next);
        }
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "FSM Run failed: "
            << error.what() << '\n';

        Reset();
    }
}