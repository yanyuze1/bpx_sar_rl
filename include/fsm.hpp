#pragma once

#include "types.hpp"

#include <array>
#include <memory>

class RL;

enum class StateID
{
    Passive = 0,
    GetUp = 1,
    GetDown = 2,
    Locomotion = 3
};

class RLFSMState
{
public:
    RLFSMState(RL& context, StateID id)
        : rl(context), id_(id)
    {
    }

    virtual ~RLFSMState() = default;

    virtual void Enter() = 0;
    virtual void Run() = 0;
    virtual void Exit() {}

    virtual StateID CheckChange(Key)
    {
        return id_;
    }

protected:
    void SetTarget(const Joints& target, float kp, float kd);

    void Interpolate(
        const Joints& from,
        const Joints& to,
        float elapsed,
        float duration);

    RL& rl;
    StateID id_;
};

class FSM
{
public:
    explicit FSM(RL& context);

    void Run(Key key);
    void Reset();

    const char* Name() const;

private:
    void Change(StateID next);

    std::array<std::unique_ptr<RLFSMState>, 4> states_;

    StateID current_ = StateID::Passive;
};