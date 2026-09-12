#include "rl.hpp"
#include "fsm.hpp"
#include "loop.hpp"
#include "glfw_adapter.h"
#include "simulate.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace mj = mujoco;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

namespace {
volatile std::sig_atomic_t stop_requested = 0;
void SignalHandler(int) { stop_requested = 1; }

// 图形窗口只处理 MuJoCo 的原版界面和鼠标事件。
class BpxAdapter final : public mj::GlfwAdapter {
public:
    std::function<void()> after_events;
    std::function<void(const std::string&)> error;
    void PollEvents() override {
        mj::GlfwAdapter::PollEvents();
        try { if (after_events) after_events(); }
        catch (const std::exception& e) { if (error) error(e.what()); }
    }
};
}  // namespace

class RL_Sim final : public RL {
    using ModelPtr = std::unique_ptr<mjModel, decltype(&mj_deleteModel)>;
    using DataPtr = std::unique_ptr<mjData, decltype(&mj_deleteData)>;
    struct Scene {
        ModelPtr model{nullptr, mj_deleteModel};
        DataPtr data{nullptr, mj_deleteData};
        std::array<int, 12> qpos{}, dof{}, actuator{};
        int quat = -1;
        int gyro = -1;
    };

public:
    RL_Sim(const fs::path& root, const std::string& scene_name)
        : RL(root), file_(root / "models/bpx" / (scene_name + ".xml")) {
        Require(mjVERSION_HEADER == mj_version(), "MuJoCo header/library mismatch");
        scene_ = LoadScene(file_);
        ResetPose();
        std::cout << "MuJoCo " << mj_versionString()
                  << " | physics " << 1.0 / params.dt
                  << " Hz | policy " << 1.0 / (params.dt * params.decimation)
                  << " Hz (nominal at 100% simulation speed)\n";
    }

    void GetState() override {
        const auto* d = scene_.data.get();
        for (std::size_t i = 0; i < kNumJoints; ++i) {
            robot_state.q[i] = static_cast<float>(d->qpos[scene_.qpos[i]]);
            robot_state.dq[i] = static_cast<float>(d->qvel[scene_.dof[i]]);
        }
        for (int i = 0; i < 4; ++i)
            robot_state.quaternion[i] = static_cast<float>(d->sensordata[scene_.quat + i]);
        for (int i = 0; i < 3; ++i)
            robot_state.gyro[i] = static_cast<float>(d->sensordata[scene_.gyro + i]);
        RequireFinite(robot_state.q, "Joint position");
        RequireFinite(robot_state.dq, "Joint velocity");
        RequireFinite(robot_state.quaternion, "Quaternion");
        RequireFinite(robot_state.gyro, "Gyroscope");
    }

    void SetCommand() override {
        RequireFinite(robot_command.q, "Target position");
        RequireFinite(robot_command.dq, "Target velocity");
        RequireFinite(robot_command.tau, "Feedforward torque");
        auto* d = scene_.data.get();
        for (std::size_t i = 0; i < kNumJoints; ++i) {
            const double torque = robot_command.tau[i]
                + robot_command.kp * (robot_command.q[i] - d->qpos[scene_.qpos[i]])
                + robot_command.kd * (robot_command.dq[i] - d->qvel[scene_.dof[i]]);
            Require(std::isfinite(torque), "Invalid commanded torque");
            const double limit = params.torque_limit;
            d->ctrl[scene_.actuator[i]] = std::clamp(torque, -limit, limit);
        }
    }

    void Check() {
        InitRL();
        const auto observation = ComputeObservation();
        const auto actions = Forward();
        Require(std::abs(observation[3]) < 1e-5f &&
                std::abs(observation[4]) < 1e-5f &&
                std::abs(observation[5] + 1.0f) < 1e-5f,
                "Home projected gravity must be [0,0,-1]");
        std::cout << "Check passed: " << observation.size()
                  << " -> " << actions.size() << "\nActions:";
        for (float action : actions) std::cout << ' ' << action;
        std::cout << '\n';
        ResetController();
    }

    void Run() {
        InitKeyboard();
        mjv_defaultCamera(&camera_);
        mjv_defaultOption(&option_);
        mjv_defaultPerturb(&perturb_);
        auto adapter = std::make_unique<BpxAdapter>();
        auto* input = adapter.get();
        sim_ = std::make_unique<mj::Simulate>(
            std::move(adapter), &camera_, &option_, &perturb_, false);
        sim_->run = 0;
        input->after_events = [this] { OnEvents(); };
        input->error = [this](const std::string& message) { Fail(message); };
        PrintHelp();

        std::exception_ptr physics_error, main_error;
        std::thread physics;
        try {
            loop_control_ = MakeLoop("loop_control", params.dt,
                                     [this] { RobotControl(); });
            loop_rl_ = MakeLoop("loop_rl", params.dt * params.decimation,
                                [this] {
                                    if (Ready() && sim_->run && rl_init_done) {
                                        GetState();
                                        RunModel();
                                    }
                                });
            loop_keyboard_ = MakeLoop("loop_keyboard", 0.05f,
                                      [this] { if (Ready()) KeyboardInterface(); });
            physics = std::thread([this, &physics_error] {
                try { PhysicsLoop(); }
                catch (...) {
                    physics_error = std::current_exception();
                    sim_->exitrequest.store(1);
                }
            });
            loop_control_->start();
            loop_rl_->start();
            loop_keyboard_->start();
            sim_->RenderLoop();
        } catch (...) { main_error = std::current_exception(); }
        {
            const std::lock_guard<std::recursive_mutex> lock(sim_->mtx);
            sim_->exitrequest.store(1);
            sim_->loadrequest = 0;
            sim_->cond_loadrequest.notify_all();
        }
        // 不能持有 sim_->mtx 等待工作线程退出。
        if (loop_keyboard_) loop_keyboard_->shutdown();
        if (loop_rl_) loop_rl_->shutdown();
        if (loop_control_) loop_control_->shutdown();
        if (physics.joinable()) physics.join();
        RestoreKeyboard();
        sim_.reset();
        if (main_error) std::rethrow_exception(main_error);
        if (physics_error) std::rethrow_exception(physics_error);
        if (loop_error_) std::rethrow_exception(loop_error_);
    }

private:
    Scene LoadScene(const fs::path& file) {
        Scene result;
        char error[2048] = {};
        if (file.extension() == ".mjb")
            result.model.reset(mj_loadModel(file.string().c_str(), nullptr));
        else
            result.model.reset(mj_loadXML(file.string().c_str(), nullptr, error, sizeof(error)));
        Require(result.model != nullptr, std::string("Model load failed: ") + error);
        if (error[0]) std::cerr << "MuJoCo: " << error << '\n';
        auto* m = result.model.get();
        Require(m->nq == 19 && m->nv == 18 && m->nu == 12,
                "Expected BPX nq=19, nv=18, nu=12");
        const int base = mj_name2id(m, mjOBJ_JOINT, "base_freejoint");
        Require(base >= 0 && m->jnt_type[base] == mjJNT_FREE &&
                m->jnt_qposadr[base] == 0 && m->jnt_dofadr[base] == 0,
                "Unexpected BPX floating base");
        m->opt.timestep = params.dt;
        m->opt.disableflags |= mjDSBL_AUTORESET;
        for (std::size_t i = 0; i < kNumJoints; ++i) {
            const int joint = mj_name2id(m, mjOBJ_JOINT, kJointNames[i]);
            Require(joint >= 0 && m->jnt_type[joint] == mjJNT_HINGE,
                    std::string("Invalid joint: ") + kJointNames[i]);
            result.qpos[i] = m->jnt_qposadr[joint];
            result.dof[i] = m->jnt_dofadr[joint];
            const auto motor = std::string(kJointNames[i]) + "_motor";
            const int a = mj_name2id(m, mjOBJ_ACTUATOR, motor.c_str());
            Require(a >= 0, "Missing actuator: " + motor);
            Require(m->actuator_trntype[a] == mjTRN_JOINT &&
                    m->actuator_trnid[2*a] == joint, "Incorrect transmission");
            Require(m->actuator_dyntype[a] == mjDYN_NONE &&
                    m->actuator_gaintype[a] == mjGAIN_FIXED &&
                    m->actuator_biastype[a] == mjBIAS_NONE,
                    "Expected direct motor actuator");
            Require(std::abs(m->actuator_gear[6*a] - 1.0) < 1e-9 &&
                    std::abs(m->actuator_gainprm[mjNGAIN*a] - 1.0) < 1e-9,
                    "Expected motor gear=1 and gain=1");
            Require(m->actuator_ctrllimited[a] &&
                    std::abs(m->actuator_ctrlrange[2*a] + 30.0) < 1e-9 &&
                    std::abs(m->actuator_ctrlrange[2*a+1] - 30.0) < 1e-9,
                    "Expected motor ctrlrange=[-30,30]");
            Require(std::abs(m->dof_damping[result.dof[i]]) < 1e-9,
                    "Unexpected passive joint damping");
            m->dof_armature[result.dof[i]] = params.armature;
            result.actuator[i] = a;
        }
        const auto sensor = [m](const char* name, int type, int dimension) {
            const int id = mj_name2id(m, mjOBJ_SENSOR, name);
            Require(id >= 0 && m->sensor_type[id] == type && m->sensor_dim[id] == dimension,
                    std::string("Invalid sensor: ") + name);
            return m->sensor_adr[id];
        };
        result.quat = sensor("body_quat", mjSENS_FRAMEQUAT, 4);
        result.gyro = sensor("body_gyro", mjSENS_GYRO, 3);
        result.data.reset(mj_makeData(m));
        Require(result.data != nullptr, "mj_makeData failed");
        Home(result);
        return result;
    }

    void Home(Scene& scene) {
        auto* m = scene.model.get();
        auto* d = scene.data.get();
        mj_resetData(m, d);
        d->qpos[0] = d->qpos[1] = 0;
        d->qpos[2] = params.initial_height;
        d->qpos[3] = 1;
        d->qpos[4] = d->qpos[5] = d->qpos[6] = 0;
        for (std::size_t i = 0; i < kNumJoints; ++i)
            d->qpos[scene.qpos[i]] = params.default_pos[i];
        mj_forward(m, d);
    }

    void ResetPose() {
        Home(scene_);
        ResetFromCurrentPose();
    }

    void ResetFromCurrentPose() {
        mj_forward(scene_.model.get(), scene_.data.get());
        GetState();
        ResetController();
        SetCommand();
        if (sim_) {
            sim_->run = 0;
            sim_->speed_changed = true;
            sim_->pert.active = 0;
            sim_->pending_.ui_update_simulation = true;
        }
    }

    bool Ready() const {
        return sim_ && sim_->m_ == scene_.model.get() && sim_->loadrequest == 0;
    }

    void Fail(const std::string& message) {
        const std::lock_guard<std::recursive_mutex> lock(sim_->mtx);
        sim_->run = 0;
        sim_->speed_changed = true;
        sim_->pending_.ui_update_simulation = true;
        std::snprintf(sim_->load_error, sizeof(sim_->load_error), "%s", message.c_str());
        mju_zero(scene_.data->ctrl, scene_.model->nu);
        ResetController();
        std::cerr << "Simulation paused: " << message << '\n';
    }

    void OnEvents() {
        // RenderLoop 调用 PollEvents 时已经持有 sim_->mtx。
        if (stop_requested) sim_->exitrequest.store(1);
        if (!Ready() || sim_->exitrequest.load()) return;
        if (sim_->pending_.reset) {
            sim_->Sync();  // 保留原版 Reset 的 UI/历史游标更新。
            ResetPose();
        } else if (sim_->pending_.load_key || sim_->pending_.load_from_history) {
            sim_->Sync();
            // 历史回放只恢复 MuJoCo 状态，不包含策略 previous_action。
            ResetFromCurrentPose();
        }
        const std::string title = "MuJoCo : bpx " + file_.stem().string()
            + " | " + GetFSM().Name()
            + " | policy=" + std::to_string(policy_steps)
            + " | cmd=" + std::to_string(control.velocity[0])
            + "," + std::to_string(control.velocity[1])
            + "," + std::to_string(control.velocity[2]);
        sim_->platform_ui->SetWindowTitle(title.c_str());
    }

    std::unique_ptr<LoopFunc> MakeLoop(
        const std::string& name, float period, std::function<void()> job) {
        return std::make_unique<LoopFunc>(name, period, [this, job] {
            try {
                const std::lock_guard<std::recursive_mutex> lock(sim_->mtx);
                if (stop_requested) sim_->exitrequest.store(1);
                if (!sim_->exitrequest.load()) job();
            } catch (...) {
                const std::lock_guard<std::recursive_mutex> lock(sim_->mtx);
                if (!loop_error_) loop_error_ = std::current_exception();
                sim_->exitrequest.store(1);
            }
        });
    }

    static void PrintHelp() {
        std::cout << "\nTerminal: Enter:run/pause  R:reset  P:Passive\n"
                     "0:GetUp  1:RL  9:GetDown  W/S A/D Q/E:velocity\n"
                     "Space:zero velocity  H:status  Ctrl+C:exit\n"
                     "Focus the terminal for robot commands. Initial state: PAUSED.\n";
    }

    void PrintStatus() {
        std::cout << "[STATE] " << GetFSM().Name()
                  << " | " << (sim_->run ? "RUN" : "PAUSED")
                  << " | policy=" << policy_steps
                  << " | cmd=" << control.velocity[0] << ' '
                  << control.velocity[1] << ' ' << control.velocity[2] << '\n';
    }

    void RobotControl() {
        if (!Ready()) return;
        GetState();
        using K = Input::Keyboard;
        const auto key = control.current_keyboard;
        if (key == K::R) {
            ResetPose();
            sim_->scrub_index = 0;
            sim_->load_error[0] = '\0';
        } else if (key == K::Enter) {
            sim_->run = !sim_->run;
            sim_->pert.active = 0;
            if (sim_->run) sim_->scrub_index = 0;
            sim_->speed_changed = true;
            sim_->pending_.ui_update_simulation = true;
        } else if (key == K::H) {
            PrintHelp();
        } else if (sim_->run || key == K::P) {
            StateController();
        } else if (key != K::None) {
            std::cout << "[NOTE] Press Enter to run before sending motion commands.\n";
        }
        control.ClearInput();
        SetCommand();
        if (key != K::None) PrintStatus();
    }

    void Step() {
        auto* m = scene_.model.get();
        auto* d = scene_.data.get();
        // 物理线程只推进 MuJoCo；控制和策略由各自的 LoopFunc 负责。
        m->opt.timestep = params.dt;
        m->opt.disableflags |= mjDSBL_AUTORESET;
        if (sim_) sim_->InjectNoise();
        for (std::size_t i = 0; i < kNumJoints; ++i) {
            const double limit = params.torque_limit;
            auto& torque = d->ctrl[scene_.actuator[i]];
            torque = std::clamp(torque, -limit, limit);
        }
        mj_step(m, d);
        for (mjtWarning w : {mjWARN_BADQPOS, mjWARN_BADQVEL, mjWARN_BADQACC})
            Require(d->warning[w].number == 0, "MuJoCo numerical instability; reset model");
    }

    void Reload(const fs::path& file) {
        try {
            // 先验证新模型；失败时保留旧模型及其映射。
            sim_->LoadMessage(file.string().c_str());
            Scene next = LoadScene(file);
            if (!Publish(next, file)) return;
            const std::lock_guard<std::recursive_mutex> lock(sim_->mtx);
            if (sim_->exitrequest.load()) return;
            scene_ = std::move(next);
            file_ = file;
            ResetFromCurrentPose();
        } catch (const std::exception& e) {
            sim_->LoadMessageClear();
            Fail(e.what());
        }
    }

    // 与 Simulate::Load 相同的模型发布协议，补上退出条件和完整加锁。
    bool Publish(Scene& next, const fs::path& file) {
        const std::string name = file.string();
        Require(name.size() < sizeof(sim_->filename), "Model path too long");
        std::unique_lock<std::recursive_mutex> lock(sim_->mtx);
        if (sim_->exitrequest.load()) return false;
        sim_->mnew_ = next.model.get();
        sim_->dnew_ = next.data.get();
        std::snprintf(sim_->filename, sizeof(sim_->filename), "%s", name.c_str());
        sim_->loadrequest = 2;
        sim_->cond_loadrequest.wait(lock, [this] {
            return sim_->loadrequest == 0 || sim_->exitrequest.load();
        });
        return !sim_->exitrequest.load();
    }

    void PhysicsLoop() {
        if (!Publish(scene_, file_)) return;
        auto sync_cpu = Clock::now();
        double sync_sim = 0;
        while (!sim_->exitrequest.load()) {
            bool busy;
            {
                const std::lock_guard<std::recursive_mutex> lock(sim_->mtx);
                busy = sim_->run && sim_->busywait;
            }
            if (busy) std::this_thread::yield();
            else std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::string reload;
            {
                const std::lock_guard<std::recursive_mutex> lock(sim_->mtx);
                if (sim_->exitrequest.load()) break;
                if (sim_->droploadrequest.exchange(0)) reload = sim_->dropfilename;
                else if (sim_->uiloadrequest.exchange(0)) reload = file_.string();
                if (!reload.empty()) sim_->run = 0;
            }
            if (!reload.empty()) { Reload(reload); continue; }
            const std::lock_guard<std::recursive_mutex> lock(sim_->mtx);
            if (sim_->exitrequest.load()) break;
            auto* m = scene_.model.get();
            auto* d = scene_.data.get();
            if (!sim_->run) {
                mj_forward(m, d);
                sim_->speed_changed = true;
                continue;
            }
            const auto start = Clock::now();
            const double cpu_elapsed = Seconds(start - sync_cpu).count();
            const double sim_elapsed = d->time - sync_sim;
            const double slowdown = 100.0 / mj::Simulate::percentRealTime[sim_->real_time_index];
            bool stepped = false;
            try {
                if (sim_elapsed < 0 || sim_->speed_changed ||
                    std::abs(cpu_elapsed / slowdown - sim_elapsed) > 0.1) {
                    sync_cpu = start;
                    sync_sim = d->time;
                    sim_->speed_changed = false;
                    Step();
                    stepped = true;
                } else {
                    if (sim_elapsed > 0)
                        sim_->measured_slowdown = static_cast<float>(cpu_elapsed / sim_elapsed);
                    const double budget = 0.7 / std::max(1, sim_->refresh_rate);
                    while ((d->time - sync_sim) * slowdown < Seconds(Clock::now() - sync_cpu).count()
                           && Seconds(Clock::now() - start).count() < budget) {
                        Step();
                        stepped = true;
                    }
                }
                if (stepped) sim_->AddToHistory();
            } catch (const std::exception& e) { Fail(e.what()); }
        }
    }

    Scene scene_;
    fs::path file_;
    mjvCamera camera_{};
    mjvOption option_{};
    mjvPerturb perturb_{};
    std::unique_ptr<mj::Simulate> sim_;
    std::unique_ptr<LoopFunc> loop_control_, loop_rl_, loop_keyboard_;
    std::exception_ptr loop_error_;
};

int main(int argc, char** argv) {
    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);
    try {
        Require((argc == 3 || argc == 4) && std::string(argv[1]) == "bpx",
                "Usage: rl_sim_mujoco bpx scene [--check]");
        const bool check = argc == 4;
        if (check) Require(std::string(argv[3]) == "--check", "Unknown option");
        RL_Sim simulation(BPX_PROJECT_ROOT, argv[2]);
        if (check) simulation.Check();
        else simulation.Run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }
}