<h1 align="center">BPX_SAR_RL</h1>

<p align="center">
  <a href="readme_CN.md">中文</a> | <a href="readme.md">English</a>
</p>

<p align="center">
  <img src="./images/videoplayback.gif" alt="alt text" />
</p>

# 1 项目描述
  本项目提供BPX强化学习仿真验证与实机部署框架。BPX官方强化学习框架:[mirrorme_rl_train](https://github.com/mirrormerobotics/mirrorme_rl_train)。支持使用mujoco完成sim2sim仿真验证与真机部署验证。

# 2 项目结构图

# 3 快速开始
## 3.1 获取项目
```bash
git clone https://github.com/yanyuze1/bpx_sar_rl.git
```  
## 3.2 依赖
- 系统依赖
```bash
sudo apt install cmake g++ build-essential libyaml-cpp-dev libeigen3-dev libboost-all-dev libspdlog-dev libfmt-dev libtbb-dev liblcm-dev libglfw3-dev
```
- 项目依赖
```bash
cd bpx_sar_rl/library
mkdir inference_runtime 
cd inference_runtime
curl -L --progress-bar -o libtorch-cxx11-abi-shared-with-deps-2.3.0+cpu.zip \
  "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcpu.zip"
unzip -q libtorch-cxx11-abi-shared-with-deps-2.3.0+cpu.zip
curl -L --progress-bar -o onnxruntime-linux-x64-1.22.0.tgz \
  "https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz"
tar -xzf onnxruntime-linux-x64-1.22.0.tgz
rm -rf libtorch-cxx11-abi-shared-with-deps-2.3.0+cpu.zip onnxruntime-linux-x64-1.22.0.tgz 
mv onnxruntime-linux-x64-1.22.0 onnxruntime
cd bpx_sar_rl/library
wget https://github.com/google-deepmind/mujoco/releases/download/3.2.7/mujoco-3.2.7-linux-x86_64.tar.gz
tar -xzf mujoco-3.2.7-linux-x86_64.tar.gz
mv mujoco-3.2.7 mujoco
rm -rf mujoco-3.2.7-linux-x86_64.tar.gz 
```
## 3.2 编译
```bash
cd /bpx_sar_rl
cmake -S . -B cmake_build -DCMAKE_BUILD_TYPE=Release
cmake --build cmake_build --parallel 2
```
## 3.3 docker
```bash
docker compose build 
docker compose up -d  
docker compose exec bpx_rl_sar bash
```
## 3.4 运行
```bash
./cmake_build/bin/rl_sim_mujoco bpx scene
```
![alt text](<images/2026-09-12 20-41-07.gif>)