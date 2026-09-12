#pragma once

#include "types.hpp"

#include <memory>
#include <string>

namespace InferenceRuntime
{

class Model
{
public:
    virtual ~Model() = default;

    virtual Joints forward(
        const Observation& observation) = 0;
};

class ModelFactory
{
public:
    static std::unique_ptr<Model> load_model(
        const std::string& path);
};

} // namespace InferenceRuntime