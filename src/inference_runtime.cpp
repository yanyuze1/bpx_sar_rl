#include "inference_runtime.hpp"

#include <onnxruntime_cxx_api.h>

#include <torch/script.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <vector>

namespace InferenceRuntime
{

namespace
{

class ONNXModel final : public Model
{
public:
    explicit ONNXModel(const std::string& path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "bpx"),
          memory_(Ort::MemoryInfo::CreateCpu(
              OrtArenaAllocator,
              OrtMemTypeDefault))
    {
        Ort::SessionOptions options;

        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session_ = Ort::Session(env_, path.c_str(), options);

        Require(
            session_.GetInputCount() == 1 &&
            session_.GetOutputCount() == 1,
            "ONNX policy must have one input and one output");

        const auto input_type =
            session_.GetInputTypeInfo(0);

        const auto output_type =
            session_.GetOutputTypeInfo(0);

        const auto input_info =
            input_type.GetTensorTypeAndShapeInfo();

        const auto output_info =
            output_type.GetTensorTypeAndShapeInfo();

        Require(
            input_info.GetElementType() ==
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            output_info.GetElementType() ==
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            "ONNX input/output must use float32");

        Require(
            input_info.GetShape() ==
                std::vector<int64_t>{1, 45},
            "Expected ONNX input shape [1,45]");

        Require(
            output_info.GetShape() ==
                std::vector<int64_t>{1, 12},
            "Expected ONNX output shape [1,12]");

        Ort::AllocatorWithDefaultOptions allocator;

        const auto input_name =
            session_.GetInputNameAllocated(0, allocator);

        const auto output_name =
            session_.GetOutputNameAllocated(0, allocator);

        input_name_ = input_name.get();
        output_name_ = output_name.get();
    }

    Joints forward(
        const Observation& observation) override
    {
        RequireFinite(observation, "ONNX input");

        const std::array<int64_t, 2> shape{1, 45};

        auto tensor = Ort::Value::CreateTensor<float>(
            memory_,
            const_cast<float*>(observation.data()),
            observation.size(),
            shape.data(),
            shape.size());

        const char* input_names[] = {
            input_name_.c_str()
        };

        const char* output_names[] = {
            output_name_.c_str()
        };

        auto outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &tensor,
            1,
            output_names,
            1);

        Require(
            outputs.size() == 1 && outputs[0].IsTensor(),
            "Invalid ONNX output");

        const auto info =
            outputs[0].GetTensorTypeAndShapeInfo();

        Require(
            info.GetElementType() ==
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            info.GetShape() ==
                std::vector<int64_t>{1, 12},
            "Invalid ONNX output type or shape");

        Joints actions{};

        std::copy_n(
            outputs[0].GetTensorData<float>(),
            actions.size(),
            actions.begin());

        RequireFinite(actions, "ONNX output");

        return actions;
    }

private:
    // 声明顺序保证 session 比 env 更早析构。
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_;

    std::string input_name_;
    std::string output_name_;
};

class TorchModel final : public Model
{
public:
    explicit TorchModel(const std::string& path)
    {
        static std::once_flag thread_configuration;

        std::call_once(thread_configuration, []
        {
            at::set_num_threads(1);
            at::set_num_interop_threads(1);
        });

        module_ = torch::jit::load(
            path,
            torch::Device(torch::kCPU));

        module_.eval();

        // 当前 BPX 策略没有循环状态。
        // 用标准姿态观测检查输入输出接口。
        Observation probe{};
        probe[5] = -1.0f;

        (void)forward(probe);
    }

    Joints forward(
        const Observation& observation) override
    {
        RequireFinite(observation, "TorchScript input");

        c10::InferenceMode inference_guard;

        const auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU);

        // clone 后由 Tensor 独立拥有输入内存。
        auto input = torch::from_blob(
            const_cast<float*>(observation.data()),
            {1, 45},
            options).clone();

        const auto value = module_.forward({input});

        Require(
            value.isTensor(),
            "TorchScript must return a Tensor");

        auto output = value.toTensor();

        Require(
            output.device().is_cpu() &&
            output.scalar_type() == torch::kFloat32 &&
            output.dim() == 2 &&
            output.size(0) == 1 &&
            output.size(1) == 12,
            "Expected TorchScript CPU float32 output [1,12]");

        output = output.contiguous();

        Joints actions{};

        std::copy_n(
            output.data_ptr<float>(),
            actions.size(),
            actions.begin());

        RequireFinite(actions, "TorchScript output");

        return actions;
    }

private:
    torch::jit::script::Module module_;
};

} // namespace

std::unique_ptr<Model> ModelFactory::load_model(
    const std::string& path)
{
    Require(
        std::filesystem::is_regular_file(path),
        "Policy file does not exist: " + path);

    std::string extension =
        std::filesystem::path(path).extension().string();

    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(
                std::tolower(character));
        });

    std::unique_ptr<Model> model;

    if (extension == ".onnx")
    {
        model = std::make_unique<ONNXModel>(path);

        std::cout
            << "Backend: ONNX Runtime (CPU)\n";
    }
    else if (extension == ".pt")
    {
        model = std::make_unique<TorchModel>(path);

        std::cout
            << "Backend: LibTorch / TorchScript (CPU)\n";
    }
    else
    {
        throw std::runtime_error(
            "Supported policy extensions: .pt, .onnx");
    }

    return model;
}

} // namespace InferenceRuntime