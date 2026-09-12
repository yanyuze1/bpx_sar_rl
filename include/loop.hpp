#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

// 保留参考框架的 LoopFunc/start/shutdown 接口。
// 不 detach，保证 shutdown 返回后回调已经结束。
class LoopFunc
{
public:
    LoopFunc(const std::string& name, float period, std::function<void()> func)
        : name_(name), period_(period), func_(std::move(func))
    {
        if (!std::isfinite(period) || period <= 0 || !func_)
            throw std::invalid_argument("Invalid loop period or callback");
    }

    ~LoopFunc() { shutdown(); }
    LoopFunc(const LoopFunc&) = delete;
    LoopFunc& operator=(const LoopFunc&) = delete;

    void start()
    {
        if (thread_.joinable()) throw std::logic_error("Loop already started");
        running_ = true;
        try { thread_ = std::thread(&LoopFunc::loop, this); }
        catch (...) { running_ = false; throw; }
        std::cout << "[INFO] [Loop] Loop start - name: " << name_
                  << ", period: " << period_.count() * 1000 << "ms\n";
    }

    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
        }
        cv_.notify_all();
        if (thread_.joinable())
        {
            thread_.join();
            std::cout << "[INFO] [Loop] Loop end - name: " << name_ << '\n';
        }
    }

private:
    void loop()
    {
        while (running_)
        {
            const auto deadline = std::chrono::steady_clock::now() + period_;
            func_();  // 应用回调负责捕获异常并通知主线程退出。
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_until(lock, deadline, [this] { return !running_; });
        }
    }

    std::string name_;
    std::chrono::duration<double> period_;
    std::function<void()> func_;
    std::atomic<bool> running_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread thread_;
};