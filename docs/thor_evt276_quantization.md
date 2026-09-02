# Jetson Thor 上 π₀.₅ EVT276 量化试验笔记

本文记录在 NVIDIA Jetson AGX Thor 上，对自有微调模型 `pi05_tienkung_evt276_full_hand_abs` 做 TensorRT 量化与对比的完整过程。目标是复习：**怎么进容器、怎么导、怎么编、测了什么、数字怎么读**。

参考教程：[OpenPi π₀.₅ on Jetson Thor](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)（官方示例是 `pi05_libero`、horizon=10）。本试验用的是 TienKung EVT276，**action horizon=40**，单相机，配置名与路径都不同。

---

## 1. 环境与模型

| 项 | 本机 |
|---|---|
| 硬件 | Jetson AGX Thor |
| L4T | R38.2.1（教程写的是 R39 / JP 7.2，本机偏低但链路可跑通） |
| CUDA | 13.0 |
| Docker 镜像（10.16 试验） | `openpi-pi05:thor-r38` ← `pytorch:26.05-py3`，TensorRT **10.16.1**，容器 `jovial_goodall` |
| Docker 镜像（对齐真机） | `openpi-pi05:thor-r38-trt1013` ← `pytorch:25.09-py3`，TensorRT **10.13.3.9**，容器 `openpi-trt1013` |
| 配置 | `pi05_tienkung_evt276_full_hand_abs` |
| 结构 | PaliGemma `gemma_2b` + action expert `gemma_300m`，`pi05=True` |
| 动作 | horizon=40，model dim=32，TienKung 有效维=24 |
| JAX ckpt | `/home/nvidia/ckpt/evt276_move_box_0813_continuous_full_hand_h40_abs_bs128/29999` |
| PyTorch ckpt | `/home/nvidia/ckpt/evt276_move_box_0813_pytorch/`（`model.safetensors` + `config.json` + `assets/`） |
| 校准/评测数据 | `/home/nvidia/datasets/EVT276_MOVE_BOX_0813/move_box_0813` |

功耗模式当时是 MAXN。`trtexec`、PyTorch、TensorRT、ModelOpt 都在镜像里，**宿主机不要跑导出/编译**。

---

## 2. 流水线在干什么

| 步骤 | 输入 | 输出 |
|---|---|---|
| 1 | JAX ckpt | PyTorch SafeTensors（BF16） |
| 2 | PyTorch | ONNX（可带 FP8 / NVFP4；denoise 圈数写进图里） |
| 3 | ONNX | TensorRT engine（`trtexec --stronglyTyped`） |
| 4 | engine + 真实数据 | 和 **同圈数** 的 PyTorch BF16 比速度、MAE |

一次推理分两段：先看图读指令，再循环出动作。

| 段 | 干什么 | 里面有什么 | 每帧跑几次 |
|---|---|---|---|
| 前半（prefix） | 看图 + 读文字指令，写出 KV cache | 视觉编码器 + 语言模型 Gemma 2B | **1 次** |
| 后半（suffix） | 用上面的 cache，把噪声一步步变成动作 | 动作专家 Gemma 300M | **循环 N 圈**（本试验 N=10 或 5） |

N 在导出 ONNX 时就写死了，编好的 engine **不能**在运行时改圈数。5 步的 engine 只能跟 5 步的 PyTorch 比。

### 各试验压的是哪一块

网络可以拆成三块，分别用不同精度。压得越狠通常越快，和 BF16 的误差也越大。

| 试验 | 视觉（看图） | 语言（2B，最大头） | 动作专家（300M，出动作） | 循环圈数 N |
|---|---|---|---|---|
| TensorRT FP16 | FP16 | FP16 | FP16 | 10 |
| TensorRT FP8 | FP8 | FP8 | FP8 | 10 |
| TensorRT FP8+NVFP4 | FP8 | **NVFP4**（比 FP8 再压一档） | FP8 | 10 |
| Hybrid（`--keep_ae_fp16`） | FP8 | **NVFP4** | **全程 FP16**（进出投影 + `gemma_expert`） | **5** |
| Hybrid 无 NVFP4 | FP8 | FP8 | **全程 FP16**（`--keep_ae_fp16`，不开 `--enable_llm_nvfp4`） | **5** |
| AE 进出投影 FP16 | FP8 | FP8 | 进出投影 FP16，**中间 block FP8**（`--keep_ae_proj_fp16`） | **5** |

对照基准始终是没进 TensorRT 的 **PyTorch BF16**，而且必须同圈数。

`--keep_ae_fp16` 保住整颗动作专家（进出投影 + `gemma_expert`）。`--keep_ae_proj_fp16` 只保住进出投影，中间 block 仍 FP8。两者都不管语言模型。

---

## 3. Docker：为什么要容器、怎么启动、命令在哪跑

可以把容器想成一台已经装好 PyTorch + TensorRT + ModelOpt 的 Linux。代码、权重、数据集还是放在 Thor 的硬盘上，用 `-v` **挂进**容器，这样容器里改文件，宿主机 `/home/nvidia/...` 立刻能看到。

### 3.1 确认镜像在

在 **Thor 宿主机**（普通终端，提示符一般是 `nvidia@...`）执行：

```bash
docker images -a
```

应能看到 `openpi-pi05:thor-r38`（10.16）和/或 `openpi-pi05:thor-r38-trt1013`（10.13）。对齐真机部署用后者，重建步骤见 **§9.1**。本节下面的 `jovial_goodall` 是 10.16 试验当时的容器。

### 3.2 启动容器（与当时实际配置一致）

当时容器 `jovial_goodall` 的挂载和参数如下。在宿主机执行一次即可：

```bash
docker run --runtime nvidia -it --name jovial_goodall \
  --network host \
  --ipc host \
  -v /home/nvidia/openpi:/workspace \
  -v /home/nvidia/ckpt:/home/nvidia/ckpt \
  -v /home/nvidia/datasets:/home/nvidia/datasets:ro \
  -v /home/nvidia/.cache/openpi:/root/.cache/openpi \
  -v /home/nvidia/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  openpi-pi05:thor-r38
```

各参数含义：

| 参数 | 作用 |
|---|---|
| `--runtime nvidia` | 让容器能用 Thor GPU（没有这个，里面 `nvidia-smi` / `trtexec` 会失败） |
| `-it` | 交互终端。跑完后你会停在容器里的 shell（提示符常变成 `root@...`） |
| `--name jovial_goodall` | 容器名字，后面 `docker exec` / `docker start` 用这个名字 |
| `--network host` | 和宿主机共用网络（以后起 policy server 时端口不用再映射） |
| `--ipc host` | 共享 IPC，大模型多进程时少踩共享内存限制 |
| `-w /workspace` | 进去后当前目录就是 `/workspace` |
| 镜像名 | 最后一项 `openpi-pi05:thor-r38`，不要漏 |

挂载（左边宿主机路径 → 右边容器里路径）：

| 宿主机 | 容器内 | 用途 |
|---|---|---|
| `/home/nvidia/openpi` | `/workspace` | 代码。改 `deployment_scripts/` 两边是同一份 |
| `/home/nvidia/ckpt` | `/home/nvidia/ckpt` | 权重、ONNX、engine。路径在容器里和宿主机一样 |
| `/home/nvidia/datasets` | `/home/nvidia/datasets`（只读 `:ro`） | 校准和评测用的 LeRobot 数据 |
| `~/.cache/openpi` | `/root/.cache/openpi` | tokenizer 等缓存 |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | HF 缓存 |

因此：导出脚本里写 `--checkpoint_dir /home/nvidia/ckpt/...` 在容器里是对的，文件会出现在宿主机同一个目录。

这条 `docker run` 成功后，提示符已经在容器里。先设一次：

```bash
export PYTHONPATH=packages/openpi-client/src:src:.:$PYTHONPATH
```

后面第 4、5 节的 `python` / `bash deployment_scripts/build_engine.sh` **都是在这个提示符下敲**，不要在宿主机直接敲（宿主机没有 `trtexec`）。

退出容器但不要删：`exit`。容器还在，只是 shell 没了。

### 3.3 以后再进来

```bash
# 宿主机：看容器是否在跑
docker ps -a | grep jovial_goodall

# 若 Status 是 Exited，先启动
docker start jovial_goodall

# 再进交互 shell（推荐，和当时用法一样）
docker exec -it jovial_goodall bash
```

进去后再 `cd /workspace`，再 `export PYTHONPATH=...`（每次新 shell 都要 export）。

也可以人留在宿主机，把命令塞进容器执行（本试验后期常用这种）。注意整段要在引号里，且先设 PYTHONPATH：

```bash
docker exec -it jovial_goodall bash -lc '
  cd /workspace
  export PYTHONPATH=packages/openpi-client/src:src:.:$PYTHONPATH
  python deployment_scripts/pytorch_to_onnx.py --help
'
```

编 engine 不依赖 PYTHONPATH，但仍要在容器里：

```bash
docker exec -it jovial_goodall bash -lc '
  cd /workspace
  ACTION_HORIZON=40 MAX_BATCH=1 bash deployment_scripts/build_engine.sh \
    /home/nvidia/ckpt/evt276_move_box_0813_pytorch/onnx/model_fp16.onnx \
    /home/nvidia/ckpt/evt276_move_box_0813_pytorch/engine/model_fp16.engine
'
```

下面第 4、5 节只写 **容器内** 命令，假设你已经 `docker exec -it jovial_goodall bash` 并且设好了 `PYTHONPATH`。

### 3.4 编 engine 的固定参数

EVT276 不是 libero 的 horizon=10。每次编译都要：

```bash
ACTION_HORIZON=40 MAX_BATCH=1 bash deployment_scripts/build_engine.sh \
  <onnx路径> <engine路径>
```

脚本还会固定：`lang_tokens` 长度 208、`state` 32、`action_dim` 32，以及 `trtexec --stronglyTyped --useCudaGraph`。`MAX_BATCH=1` 与当时实际编译一致（脚本默认 max batch 是 4，不显式写会不一致）。

---

## 4. 导出与编译命令（已按脚本核对）

PyTorch 权重已经在 `evt276_move_box_0813_pytorch/`，从 ONNX 开始。

文件落在 `{output_path}/onnx/`。默认文件名：

| 导出参数 | 默认 ONNX 名 |
|---|---|
| `--precision fp16` | `model_fp16.onnx` |
| `--precision fp8` | `model_fp8.onnx` |
| `--precision fp8 --enable_llm_nvfp4` | `model_fp8_nvfp4.onnx` |
| 再加 `--onnx_name xxx.onnx` | 用你指定的名字（hybrid 必须这样做，以免覆盖上一行） |

`--quantize_attention_matmul` 默认就是开的（只对 fp8 有效），写出来是为了和当时命令一致。

### 4.1 TensorRT FP16（10 步）

不做 ModelOpt 量化，`model.to(float16)` 后导出。官方教程说纯 FP16 可能溢出；本 EVT276 微调上没有，和 BF16 几乎对齐。

```bash
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir /home/nvidia/ckpt/evt276_move_box_0813_pytorch \
  --output_path /home/nvidia/ckpt/evt276_move_box_0813_pytorch \
  --config_name pi05_tienkung_evt276_full_hand_abs \
  --precision fp16

ACTION_HORIZON=40 MAX_BATCH=1 bash deployment_scripts/build_engine.sh \
  /home/nvidia/ckpt/evt276_move_box_0813_pytorch/onnx/model_fp16.onnx \
  /home/nvidia/ckpt/evt276_move_box_0813_pytorch/engine/model_fp16.engine
```

### 4.2 TensorRT FP8（10 步）

```bash
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir /home/nvidia/ckpt/evt276_move_box_0813_pytorch \
  --output_path /home/nvidia/ckpt/evt276_move_box_0813_pytorch \
  --config_name pi05_tienkung_evt276_full_hand_abs \
  --precision fp8 \
  --quantize_attention_matmul

ACTION_HORIZON=40 MAX_BATCH=1 bash deployment_scripts/build_engine.sh \
  /home/nvidia/ckpt/evt276_move_box_0813_pytorch/onnx/model_fp8.onnx \
  /home/nvidia/ckpt/evt276_move_box_0813_pytorch/engine/model_fp8.engine
```

校准走 `deployment_scripts/calibration_data.py`：本地 LeRobot + Decord 读视频（容器里 TorchCodec 和 FFmpeg 版本不匹配），32 个样本。

### 4.3 TensorRT FP8 + NVFP4（10 步，action expert 仍是 FP8）

NVFP4 **只打在 LLM**：`paligemma_with_expert.paligemma.model.language_model.layers.*`。Vision 和 action expert 仍走 FP8。

```bash
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir /home/nvidia/ckpt/evt276_move_box_0813_pytorch \
  --output_path /home/nvidia/ckpt/evt276_move_box_0813_pytorch \
  --config_name pi05_tienkung_evt276_full_hand_abs \
  --precision fp8 \
  --enable_llm_nvfp4 \
  --quantize_attention_matmul

ACTION_HORIZON=40 MAX_BATCH=1 bash deployment_scripts/build_engine.sh \
  /home/nvidia/ckpt/evt276_move_box_0813_pytorch/onnx/model_fp8_nvfp4.onnx \
  /home/nvidia/ckpt/evt276_move_box_0813_pytorch/engine/model_fp8_nvfp4.engine
```

### 4.4 Hybrid：VLM NVFP4 + AE FP16 + 5 步（最终试验）

必须加 `--keep_ae_fp16`，并用 `--onnx_name`、`--num_steps 5`，避免覆盖 4.3 的 10 步文件。

```bash
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir /home/nvidia/ckpt/evt276_move_box_0813_pytorch \
  --output_path /home/nvidia/ckpt/evt276_move_box_0813_pytorch \
  --config_name pi05_tienkung_evt276_full_hand_abs \
  --precision fp8 \
  --enable_llm_nvfp4 \
  --quantize_attention_matmul \
  --keep_ae_fp16 \
  --num_steps 5 \
  --onnx_name model_fp8_nvfp4_ae_fp16_s5.onnx

ACTION_HORIZON=40 MAX_BATCH=1 bash deployment_scripts/build_engine.sh \
  /home/nvidia/ckpt/evt276_move_box_0813_pytorch/onnx/model_fp8_nvfp4_ae_fp16_s5.onnx \
  /home/nvidia/ckpt/evt276_move_box_0813_pytorch/engine/model_fp8_nvfp4_ae_fp16_s5.engine
```

`--keep_ae_fp16` 同时做两件事（缺一则 AE 不是全程 FP16）：

1. ModelOpt 不量化 action expert 的 Linear（`action_in_proj` / `action_out_proj` / `gemma_expert`）。
2. AE denoise 的 `Q@Kᵀ`、`attn@V` 不用 `QuantizedMatMul`，走 FP16 `torch.matmul`。

`--quantize_attention_matmul` 仍然打开，只作用于 **PaliGemma 前缀** attention。不加 `--keep_ae_fp16` 时，整条命令等价于 4.3（再加 `--num_steps 5` 才会变成 5 步图）。

---

## 5. 怎么评测

脚本：`deployment_scripts/compare_backends.py`。同样在容器内、先 `export PYTHONPATH`。

```bash
python deployment_scripts/compare_backends.py \
  --config-name pi05_tienkung_evt276_full_hand_abs \
  --checkpoint-dir /home/nvidia/ckpt/evt276_move_box_0813_pytorch \
  --num-samples 10 \
  --num-warmup 3 \
  --num-test-runs 10
```

| 项 | 设定 |
|---|---|
| 数据 | 训练集均匀抽 10 帧（Decord 解 `observation.images.camera_head`） |
| 输入 | `image` / `state` / `prompt` |
| 噪声 | 同一份 `golden_noise.npy`，shape `(40, 32)` |
| 时延 | sample 0 上 warmup 3 次 + 测 10 次，报均值 |
| 精度对照 | 相对 **同圈数** 的 PyTorch BF16（10 步对 10 步，5 步对 5 步） |
| 动作形状 | 有效维 `(40, 24)`，不是 engine 的 32 维 |
| cosine | 每帧拉平 960 维再算，然后 10 帧取平均 |
| MAE | 每帧全部元素的平均绝对误差，再对 10 帧平均 |

数字在 `/home/nvidia/ckpt/evt276_move_box_0813_pytorch/compare_backends.json`（`inference_stats.mean` / `model_stats.mean` / `accuracy.cosine_mean` / `accuracy.mae_mean`）。

---

## 6. 结果

下面两张表来自 `compare_backends.json`。同一条 engine 再跑，Total 会差约 1 ms。

### 6.1 十步四路（对 10 步 BF16）

来源：`compare_backends.json` 里 `num_steps=10` 的四条。

| 后端（JSON `name`） | Total (ms) | Model (ms) | vs BF16 | cosine | MAE |
|---|---|---|---|---|---|
| `pytorch_bf16` | 130.48 | 125.45 | 1.00× | 1.000000 | 0 |
| `tensorrt_fp16` | 77.92 | 75.76 | 1.67× | 1.000000 | 0.000347 |
| `tensorrt_fp8` | 44.88 | 42.86 | 2.91× | 0.999899 | 0.006371 |
| `tensorrt_fp8_nvfp4` | 39.20 | 37.06 | 3.33× | 0.987908 | 0.061482 |

### 6.2 五步 hybrid（对 5 步 BF16）

来源：同文件里 `pytorch_bf16_s5`、`tensorrt_hybrid_s5`。

| 后端（JSON `name`） | Total (ms) | Model (ms) | vs 5 步 BF16 | cosine | MAE |
|---|---|---|---|---|---|
| `pytorch_bf16_s5` | 101.38 | 96.27 | 1.00× | 1.000000 | 0 |
| `tensorrt_hybrid_s5`（VLM NVFP4 + AE 全程 FP16） | **34.45** | **32.21** | **2.94×** | **0.988135** | **0.060993** |

`trtexec` 自测（不含前后处理）：最终 hybrid GPU compute ≈ 32.0 ms。

### 6.3 产物体积

| 文件 | 大约 |
|---|---|
| `onnx/model_fp16.data` | 6.1 GB |
| `onnx/model_fp8.data` | 11 GB |
| `onnx/model_fp8_nvfp4.data` | 4.9 GB |
| `onnx/model_fp8_nvfp4_ae_fp16_s5.data` | 13 GB（5 步展开 + AE 未压） |
| `engine/model_fp16.engine` | 5.9 GB |
| `engine/model_fp8.engine` | 3.5 GB |
| `engine/model_fp8_nvfp4.engine` | 2.7 GB |
| `engine/model_fp8_nvfp4_ae_fp16_s5.engine` | 3.0 GB |

---

## 7. 关键细节

### 7.1 动作专家里其实有两套精度

关「权重量化」不等于 attention 也变成 FP16。

| 部分 | 谁在管 | 不加 `--keep_ae_fp16` | 加了 `--keep_ae_fp16` |
|---|---|---|---|
| 动作专家 Linear 权重（q/k/v/o、MLP、进出投影） | ModelOpt | FP8 | FP16 |
| 每圈 denoise 里的 `Q@Kᵀ`、`attn@V` | 脚本自己插的 MatMul | 前半段仍可能是 FP8 | 全程 FP16 `torch.matmul` |
| 语言模型前半段 attention | 另一条补丁 | 可以继续 FP8 | 仍然可以 FP8（不受这个旗标影响） |

### 7.2 为什么关掉 AE attention FP8 反而比第一次 hybrid 快

两次都是 5 步、VLM NVFP4、AE 权重 FP16，只差 prefix attention 要不要 QDQ。

这两次乘法很小：Q 大约 40 token，expert 是 `gemma_300m`。FP8 省不下多少算力，却要每层每步 Quantize → MatMul → Dequantize（约 180 组 QDQ）。q/k/v 已经是 FP16，再 round-trip 一趟 FP8 更亏。所以 36.31 ms → 34.45 ms。MAE 几乎不变，误差在 NVFP4 的 prefix KV 上。

### 7.3 不要拿 hybrid 去跟 10 步 NVFP4 比「谁量化更快」

10 步整网 NVFP4 ≈ 39 ms，5 步 hybrid ≈ 34 ms，差的主要是 **少 5 次 denoise**。Prefix 两边都只跑一次，所以不会快一倍。要对齐必须同一步数。

### 7.4 步数写进图

`--num_steps` 会折叠对应份数的 timestep 常数，并把 Euler 循环 unroll 进 ONNX。engine 推理时改 `num_steps` 无效。对比 5 步 TRT 必须用 5 步 PyTorch。

### 7.5 官方教程 vs 本试验

| | 教程 libero | 本试验 EVT276 |
|---|---|---|
| horizon | 10 | 40 |
| 相机 | 头+腕 | 仅 head |
| 官方 FP16 | 认为会溢出 | 本 ckpt 上 cosine≈1 |
| 评测输入 | 常合成 example | 真实训练集 10 帧 |

### 7.6 复现最终 hybrid

`deployment_scripts/pytorch_to_onnx.py` 就是最终试验结果对应的转换逻辑。出数之后只改过日志字符串，不改 ONNX 图。用 §4.4 的命令可以复现。

---

## 8. 结论

| 你更在意 | 选哪条 | 大约多快 | 大约差多少（MAE） |
|---|---|---|---|
| 贴着 BF16 | TensorRT FP16 | 1.67×（~78 ms） | ≈ 3e-4 |
| 较快、误差还小 | TensorRT FP8 | 2.91×（~45 ms） | ≈ 6e-3 |
| 要速度，能接受 MAE≈0.06 | 10 步 FP8+NVFP4，或 5 步 hybrid | ~39 ms / ~34 ms | ≈ 0.061 |

本 ckpt 上纯 FP16 **没有**溢出。Hybrid 的误差主要来自语言模型的 NVFP4，不是动作专家；动作专家 attention 再套一层 FP8 只会更慢。评测必须：真实数据、固定噪声、同圈数、MAE + cosine。

---

## 9. 真机部署需要 TensorRT 10.13.3.9（近期）

部署机是 L4T R38.2.1 + 系统 TensorRT **10.13.3.9**。§1–8 的 engine 是在 `pytorch:26.05-py3` 里用 **10.16.1** 编的：

- `.engine` 和 TensorRT **四段版本号**绑定。10.16 编的文件在 10.13 上 `deserializeCudaEngine` 会直接拒绝。
- 10.16 / ModelOpt 0.43 导出的 NVFP4 ONNX（`NVFP4QuantExporter`）在 10.13 上往往 **编不过**，不是缺 `trtexec`。只把旧 ONNX 拿到真机重编不够，要在 10.13 栈里 **重新导出**。

没有现成的 OpenPI + JP 7.0 镜像。能对上真机 TRT 版本号的官方基础镜像是 NGC `nvcr.io/nvidia/pytorch:25.09-py3`（NVIDIA 给 Thor / SBSA 点过名），自带 TensorRT **10.13.3.9** + ModelOpt **0.33**。不要用：

| 基础镜像 | TensorRT | 为什么不用 |
|---|---|---|
| `pytorch:25.08-py3` | 10.13.2.2 | 差一个小版本，engine 仍可能 deserialize 失败 |
| `pytorch:26.05-py3` | 10.16.1 | 就是现在编不过的原因 |
| `pytorch:25.11-py3` 及更新 | 10.14+ | 又偏新 |

旧镜像 `openpi-pi05:thor-r38`（10.16）**保留**，另打一份 tag。

### 9.1 重新构建 10.13 镜像（宿主机）

在仓库根目录 `/home/nvidia/openpi`。Dockerfile 是 [`deployment_scripts/thor.Dockerfile`](../deployment_scripts/thor.Dockerfile)，默认 `BASE_IMAGE` 仍是 `26.05-py3`，用 `--build-arg` 换掉即可，**不要改文件里的默认值**（避免下次误打 10.16）。

```bash
cd /home/nvidia/openpi

# 需要能拉 nvcr.io。第一次大约 30 GB，网络不稳会中断，层一般还在缓存里。
# 本机 BuildKit 抽完大层后 apt 步曾报 UtimesNanoAt / snapshot 错误，改用旧 builder：
DOCKER_BUILDKIT=0 docker build \
  -f deployment_scripts/thor.Dockerfile \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3 \
  -t openpi-pi05:thor-r38-trt1013 .
```

Dockerfile 在基础镜像之上做的事：

1. `apt-get` 装 ffmpeg、OpenCV 依赖、编译工具等。
2. `COPY deployment_scripts/pyproject.toml` 后分三次 `pip`：清华源装 PyYAML；Jetson `sbsa/cu130` 源装 `torchcodec` / `diffusers` / `decord2`；再 `-e '.[thor]'` 和 `onnxslim` `lief`。
3. 基础镜像里的 TensorRT / `trtexec` / ModelOpt **不会被降级**，所以 TRT 版本完全由 `BASE_IMAGE` 决定。

打完核对：

```bash
docker images | grep openpi
# 应同时有 openpi-pi05:thor-r38 和 openpi-pi05:thor-r38-trt1013

docker run --rm openpi-pi05:thor-r38-trt1013 \
  python3 -c "import tensorrt as t; print(t.__version__)"
# 必须打印 10.13.3.9
```

拉基础镜像失败时：`docker login nvcr.io` 后再 build。BuildKit 报 snapshot 错就保持 `DOCKER_BUILDKIT=0`。

### 9.2 启动 10.13 容器并打 transformers 补丁

挂载和当初 `jovial_goodall` 一样，只换镜像和容器名。后台跑即可（后面用 `docker exec`）：

```bash
docker run -d --runtime nvidia --name openpi-trt1013 \
  --network host \
  --ipc host \
  -v /home/nvidia/openpi:/workspace \
  -v /home/nvidia/ckpt:/home/nvidia/ckpt \
  -v /home/nvidia/datasets:/home/nvidia/datasets:ro \
  -v /home/nvidia/.cache/openpi:/root/.cache/openpi \
  -v /home/nvidia/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  openpi-pi05:thor-r38-trt1013 \
  sleep infinity
```

每个新容器都要拷 OpenPI 的 transformers 补丁（AdaRMS、attention reshape 等），否则导出/量化会报 `transformers_replace is not installed correctly`：

```bash
docker exec openpi-trt1013 bash -lc '
  SITE=$(python3 -c "import transformers,os; print(os.path.dirname(transformers.__file__))")
  cp -r /workspace/src/openpi/models_pytorch/transformers_replace/* "$SITE/"
  python3 -c "from transformers.models.siglip import check; print(check.check_whether_transformers_replace_is_installed_correctly())"
'
# 应打印 True
```

以后进容器：

```bash
docker exec -it openpi-trt1013 bash
cd /workspace
export PYTHONPATH=packages/openpi-client/src:src:.:$PYTHONPATH
```

或人留在宿主机：

```bash
docker exec openpi-trt1013 bash -lc '
  cd /workspace
  export PYTHONPATH=packages/openpi-client/src:src:.:$PYTHONPATH
  python deployment_scripts/pytorch_to_onnx.py --help
'
```

导出、编 engine **都在这个容器里**。日志第一行必须是 `TensorRT.trtexec [TensorRT v101303]`。

### 9.3 新增量化开关

写在 [`deployment_scripts/pytorch_to_onnx.py`](../deployment_scripts/pytorch_to_onnx.py)。两旗同时开时以 `--keep_ae_fp16` 为准（整颗 AE 仍 FP16）。

| 开关 | 作用 |
|---|---|
| `--keep_ae_fp16` | ModelOpt 不量化 `action_in_proj*`、`action_out_proj*`、`*gemma_expert*`；AE denoise 的 `Q@K` / `attn@V` 走 FP16 matmul |
| `--keep_ae_proj_fp16` | **只**不量化 `action_in_proj*`、`action_out_proj*`；中间 `gemma_expert` block 仍 FP8；AE attention 可走 FP8 QDQ |
| `--enable_llm_nvfp4` | 只把 PaliGemma LLM 压到 NVFP4。10.13 真机上这份图更容易溢出 / 编不过，近期默认 **不要开** |
| `--num_steps N` | denoise 圈数写进 ONNX，编完不能改 |
| `--onnx_name xxx.onnx` | 写到 `{output_path}/onnx/`，避免覆盖默认文件名 |

换 `--output_path` 就不会盖掉 §4 的 10.16 产物。

### 9.4 在 10.13 容器里导出并编译

共用（每条命令前都要）：

```bash
cd /workspace
export PYTHONPATH=packages/openpi-client/src:src:.:$PYTHONPATH
CKPT=/home/nvidia/ckpt/evt276_move_box_0813_pytorch
CFG=pi05_tienkung_evt276_full_hand_abs
```

编 engine 一律：

```bash
ACTION_HORIZON=40 MAX_BATCH=1 bash deployment_scripts/build_engine.sh \
  <onnx路径> <engine路径>
```

#### 四路重导（10 步 FP16 / FP8 / FP8+NVFP4，以及 hybrid s5 + NVFP4）

根目录：`/home/nvidia/ckpt/evt276_move_box_0813_pytorch_trt1013/`

```bash
# FP16，10 步
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir $CKPT --output_path ${CKPT}_trt1013 \
  --config_name $CFG --precision fp16

# FP8，10 步
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir $CKPT --output_path ${CKPT}_trt1013 \
  --config_name $CFG --precision fp8 --quantize_attention_matmul

# FP8+NVFP4，10 步
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir $CKPT --output_path ${CKPT}_trt1013 \
  --config_name $CFG --precision fp8 --enable_llm_nvfp4 --quantize_attention_matmul

# Hybrid：VLM NVFP4 + AE 全程 FP16 + 5 步
python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir $CKPT --output_path ${CKPT}_trt1013 \
  --config_name $CFG --precision fp8 --enable_llm_nvfp4 --quantize_attention_matmul \
  --keep_ae_fp16 --num_steps 5 \
  --onnx_name model_fp8_nvfp4_ae_fp16_s5.onnx
```

对应 engine：`model_fp16` / `model_fp8` / `model_fp8_nvfp4` / `model_fp8_nvfp4_ae_fp16_s5`，都 `PASSED`，`trtexec` GPU mean 大约 79.8 / 45.5 / 40.6 / 35.3 ms。

真机用带 NVFP4 的 hybrid 时出现过 **NaN**（训练帧上没有）。官方教程也写过 FP16 / 低精度在 Gemma attention 上会溢出；真机画面和校准分布不同，NVFP4 更容易炸。后面两版都 **不开 NVFP4**。

#### Hybrid 无 NVFP4：VLM FP8 + AE 全程 FP16 + 5 步

以第二次重导为准（`_r2`）。第一次在 `..._fp8_ae_s5/`，配置相同。

```bash
OUT=/home/nvidia/ckpt/evt276_move_box_0813_pytorch_trt1013_fp8_ae_s5_r2

python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir $CKPT --output_path $OUT \
  --config_name $CFG \
  --precision fp8 --quantize_attention_matmul \
  --keep_ae_fp16 --num_steps 5 \
  --onnx_name model_fp8_ae_fp16_s5.onnx

ACTION_HORIZON=40 MAX_BATCH=1 bash deployment_scripts/build_engine.sh \
  $OUT/onnx/model_fp8_ae_fp16_s5.onnx \
  $OUT/engine/model_fp8_ae_fp16_s5.engine
```

`v101303` + `PASSED`。engine 3.8 GB，`trtexec` GPU mean ≈ 40.6 ms。

#### AE 进出投影 FP16 + 中间 block / VLM FP8 + 5 步（无 NVFP4）

```bash
OUT=/home/nvidia/ckpt/evt276_move_box_0813_pytorch_trt1013_fp8_aeproj_s5

python deployment_scripts/pytorch_to_onnx.py \
  --checkpoint_dir $CKPT --output_path $OUT \
  --config_name $CFG \
  --precision fp8 --quantize_attention_matmul \
  --keep_ae_proj_fp16 --num_steps 5 \
  --onnx_name model_fp8_aeproj_fp16_s5.onnx

ACTION_HORIZON=40 MAX_BATCH=1 bash deployment_scripts/build_engine.sh \
  $OUT/onnx/model_fp8_aeproj_fp16_s5.onnx \
  $OUT/engine/model_fp8_aeproj_fp16_s5.engine
```

`v101303` + `PASSED`。engine 3.5 GB，`trtexec` GPU mean ≈ 32.8 ms。

### 9.5 10.13 产物目录

| 目录 | 内容 |
|---|---|
| `.../evt276_move_box_0813_pytorch/onnx/`、`engine/` | 10.16 旧产物，不要覆盖 |
| `.../evt276_move_box_0813_pytorch_trt1013/` | 四路 10.13 重导 |
| `.../evt276_move_box_0813_pytorch_trt1013_fp8_ae_s5_r2/` | hybrid 无 NVFP4（推荐拷这个 hybrid） |
| `.../evt276_move_box_0813_pytorch_trt1013_fp8_aeproj_s5/` | 只保住 AE 进出投影 |

真机只拷对应的 `.engine`。Python `tensorrt` 必须是 **10.13.3.9**。checkpoint 仍要带原来的 `assets/`（norm stats）。

