# openpi05 quantization on Thor

本仓库基于 [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)，在 **NVIDIA Jetson AGX Thor** 上把微调后的 π₀.₅（pi0.5）策略量化并编译成 TensorRT engine，用来加速真机推理。

官方 Thor 教程是 [OpenPi π₀.₅ on Jetson Thor](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)（示例模型 `pi05_libero`，horizon=10）。这里在同一条链路上做了这些事：

- 适配自有 TienKung EVT276 模型（`pi05_tienkung_evt276_full_hand_abs`，**action horizon=40**）
- 支持 FP16 / FP8 / FP8+NVFP4，以及动作专家混合精度（整颗 AE 保持 FP16，或只保持action_in/out_proj投影 FP16）
- 用真实训练帧对比 PyTorch BF16 与 TensorRT 的延迟、cosine 和 MAE

## 流水线

```text
JAX checkpoint  →  PyTorch (BF16)  →  ONNX (FP16 / FP8 / NVFP4)  →  TensorRT engine
                                                                      ↓
                                              与同 denoise 步数的 PyTorch BF16 对比
```

Denoise 步数在导出 ONNX 时写死，编好的 engine 不能在运行时改去噪步数。

## 主要文件

| 路径 | 作用 |
|---|---|
| `deployment_scripts/thor.Dockerfile` | Thor 部署镜像（PyTorch + TensorRT + ModelOpt） |
| `deployment_scripts/pytorch_to_onnx.py` | 导出 ONNX：`--precision`、`--enable_llm_nvfp4`、`--keep_ae_fp16`、`--keep_ae_proj_fp16` |
| `deployment_scripts/build_engine.sh` | `trtexec --stronglyTyped` 编译 engine |
| `deployment_scripts/compare_backends.py` | PyTorch BF16 与 TensorRT 后端对比 |
| `deployment_scripts/calibration_data.py` | FP8 / NVFP4 校准数据 |
| `docs/thor_evt276_quantization.md` | 试验笔记：环境、命令、结果怎么读 |

## 使用注意

- **Engine 和 TensorRT 版本绑定。** 在 TensorRT 10.16 里导出的 ONNX / engine，不能直接放到 10.13 上用。部署机若是 TensorRT 10.13.3.9，请拉取对应版本的基础镜像，在对应容器里重新导出并编译。
- 本仓库只含代码和文档，**不含** checkpoint、ONNX、engine 和数据集。
- 更细的步骤和数字见 [`docs/thor_evt276_quantization.md`](docs/thor_evt276_quantization.md)。

## 上游

训练、数据格式和通用推理请看 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)。本仓库基于提交 `15a9616`，再叠了 Jetson 部署补丁和上述量化改动。
