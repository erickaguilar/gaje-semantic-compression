# 🧬 GAJE 协议：语义自适应与基因组压缩 (v1.7.4-alpha)

[![Version](https://img.shields.io/badge/version-1.7.4--alpha_Helix_Ecosystem-purple)](docs/meta/EMPIRICAL_TRUTH_STATE.md)
[![Engine](https://img.shields.io/badge/Engine-Pure_Rust_PyO3_WASM-orange.svg)](src/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Format](https://img.shields.io/badge/Format-Zero--Copy_GAJE_mmap-brightgreen.svg)](docs/plans/UNIFIED_GAJE_ADAPTIVE_FORMAT_PLAN.md)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20Hub-yellow)](https://huggingface.co/eaguilar/gaje-models)
[![Language: Español](https://img.shields.io/badge/Language-Espa%C3%B1ol-yellow.svg)](README.md)
[![Language: English](https://img.shields.io/badge/Language-English-blue.svg)](README.en.md)

**GAJE (Genomic Adaptive Joint Embedding / 基因自适应联合嵌入)** 是一款面向超大语言模型 (LLM) 的极高密度压缩与纯原生 Rust 独立推理引擎。在生产环境中，GAJE 将 Transformer 主体压缩为 **4-bit 离散权重（Q4_0，16 个自适应质心）**，并将关键的嵌入表 (`token_embd` 与 `lm_head`) 保护在 **FP32** 精度中，封装于统一的零拷贝内存映射格式 **`.gaje` v2**。系统集成了主权生产级 HTTP 服务器 **`gaje-server` (Zero-Python Runtime)**，具备实时逐 Token SSE 流式传输与并发热插拔、**Island Model `.gmem`** 亚毫秒级关联持久化内存、动态自描述架构头 (**`ArchitectureDescriptor`**) 以及无需服务端的浏览器端 **WebAssembly** 运行时。

---

## 📦 认证模型库 (Hugging Face Hub)

官方预制模型可在 [Hugging Face 官方仓库 (`eaguilar/gaje-models`)](https://huggingface.co/eaguilar/gaje-models) 直接下载：

| 模型 / 架构 | 格式 | 体积 | 最佳运行环境 | 特性与定位 |
| :--- | :---: | :---: | :---: | :--- |
| **`gaje_nano_1.5b.gaje`** | `.gaje` v2 | **1.23 GB** | WebAssembly (移动端 / 浏览器) | 极速响应、极低内存占用，适合手机与嵌入式设备。 |
| **`gaje_prime_3b.gaje`** | `.gaje` v2 | **2.24 GB** | WASM 桌面端 / 云端服务 | 均衡旗舰，高通用推理能力与深层上下文连贯性。 |
| **`gaje_ultra_7b.gaje`** | `.gaje` v2 | **4.88 GB** | 原生服务器 / 云计算节点 | 深度多轮推理、复杂代码编写与长篇分析。 |

---

## 🔬 经验真理与生产环境基准认证 (AMD Ryzen 7 5800H)

根据 GAJE **经验真理标准** ([`docs/meta/EMPIRICAL_TRUTH_STATE.md`](docs/meta/EMPIRICAL_TRUTH_STATE.md))，引擎性能已获得全面验证：

### 🏆 1. A/B 对照对照实验 (GAJE Q4_0 对比 HuggingFace PyTorch FP32)

| 推理环境 | 格式 / 精度 | 生成输出 | 端到端生成吞吐量 | 物理内存占用 (RAM RSS) |
| :--- | :---: | :--- | :---: | :---: |
| **HuggingFace PyTorch** | **FP32 官方基准** | *"El planeta más grande del Sistema Solar es la Tierra, con una"* | **`1.38 tok/s`** | $1,980\text{ MB}$ |
| **GAJE 原生引擎 (`.gaje`)** | **4-bit 基因组零拷贝** | *"El planeta más grande del Sistema Solar es la Tierra."* | **`19.2 - 23.0 tok/s`** | **`448 MB` (节省 77%)** |

### ⚡ 2. 跨模型多语言吞吐量认证

| 模型架构 | 格式 | 事实生成认证 | CPU 吞吐量 | 冷启动加载耗时 | 实时运行内存 |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Qwen2.5 1.5B Instruct** | **`.gaje` 混合 v2** | 西班牙语: *"La capital de Francia es París."* | **`11.31 - 12.13 tok/s`** | **`< 0.75 ms` (mmap)** | **`2.6 GB` (虚拟内存)** |
| **Qwen2 0.5B Instruct** | **`.gaje` 混合 v2** | 中文: *"木星"* / 西班牙语: *"París"* | **`19.20 - 23.00 tok/s`** | **`< 0.75 ms` (mmap)** | **`~498 MiB` (节省 74%)** |
| **SmolLM2 135M Instruct** | **`.gaje` 零拷贝** | 英语: *"Berlin."* / *"100°C"* | **`28.28 - 32.10 tok/s`** | **`< 0.75 ms` (mmap)** | **`~472 MB`** |

---

### 🏝️ 3. Island Model (.gmem)：亚毫秒级海马 RAG 记忆检索

系统通过 64 字节内存对齐的平面二进制索引 (`.gmem`) 实现实时上下文关联检索与持久化：

* **RAG 向量检索延迟**：多生态位每次查询 **`< 0.5 ms`**，采用针对输入词向量表 ($W_E$) 的原生*加权平均池化 (Weighted Mean Pooling)*。
* **熵隙门控机制 ($\Delta_{top} \ge 0.12$)**：结合 K-WTA 侧向竞争抑制，在数学层面直接剔除模糊或竞争陷阱对。
* **卫星白化标定**：各向异性居中向量 ($\boldsymbol{\mu} \in \mathbb{R}^D$) 持久化于 `data/calibration/`，支持平滑降级（方案 C）。
* **规范化 6 状态实时遥测**：在 SSE 与服务端指标中实时监控 (`memory_injected`, `rejected_low_similarity`, `rejected_entropy_gap` 等)。
* **安全系统级注入**：在 ChatML 与 Llama3 对话模板中严格注入于系统提示词区块，杜绝轮次错乱与提示词污染。

---

## ⚡ 快速入门 — 独立原生二进制单文件 (`gaje-cli`)

GAJE Helix 可编译为 **100% 独立的单文件 Rust 原生二进制程序**，生产环境中无需 Python、C++ 依赖或外部运行时环境。

### 1. 编译优化版原生二进制
```bash
cargo build --release --bin gaje-cli
```

### 2. `gaje-cli` 核心指令集

```bash
# 启动生产级主权 HTTP 服务器 (Zero-Python，内嵌 Web UI 与实时 SSE 流式传输)
./target/release/gaje-cli serve --port 8080

# 终端交互式 Chat REPL 会话
./target/release/gaje-cli chat --model models/production/gaje_pico_135m.gaje

# 直接从 Hugging Face 快速下载官方模型
./target/release/gaje-cli pull pico

# 本地模型目录与结构检测
./target/release/gaje-cli models list
./target/release/gaje-cli models inspect models/production/gaje_pico_135m.gaje

# 将任何模型 (.gguf, .gaje, .flat) 转换为统一零拷贝 .gaje v2 格式
./target/release/gaje-cli export-flat models/source/model.gguf -o models/production/model.gaje

# 吞吐量 (TPS)、首字延迟 (TTFT) 与困惑度 (PPL) 综合基准测试
./target/release/gaje-cli benchmark --model models/production/gaje_pico_135m.gaje --tokens 64

# 权重数学审计 (严格 0 NaN / 0 Inf 检验与质心熵分析)
./target/release/gaje-cli audit models/production/gaje_pico_135m.gaje

# 硬件指令集诊断 (AVX2, AVX512, NEON, FMA)
./target/release/gaje-cli doctor
```

### 3. 原生测试套件
```bash
# 执行 Rust 原生单元测试与集成测试
cargo test --lib
cargo test --test cli_standalone_test
```

---

## 🎯 GAJE 的定位与非目标 (What GAJE Is Not)

为了在评估或部署前建立清晰的技术预期：

* **并非企业级多租户服务器：** 它是专为主权边缘计算、个人本地部署设计的单流/单用户推理与海马记忆引擎。
* **并非高维深层句向量嵌入运行时：** 其轻量提取器优先追求亚毫秒级延迟 ($< 0.5\text{ ms}$) 和零额外内存开销，而非 Transformer 深层的复杂上下文推断。
* **不与 llama.cpp 竞争多插槽服务器的纯并发吞吐量：** 它竞争的是极低常驻物理内存占用 (`mmap` 零拷贝)、极速冷启动，以及在单一独立原生二进制中原生集成的关联持久化记忆。
* **在无词汇重叠时不支持纯抽象语义同义重述：** 检索依赖共享词汇或局部语义重叠；若要在无共同词汇的情况下解析抽象同义表述，需要外接专用的密集向量编码器。
* **不建议在消费级硬件上运行 > 3B 规模的模型：** 认证模型库严格划定在 0.1B 至 3B 参数区间。

---

## ⚠️ 已知局限与系统边界 (Known Limitations)

本项目严格遵循**经验真理标准**。以下限制并非实现疏漏，而是深思熟虑的工程决策以及经过严格形式化测量的数学现实：

### 1. 忠实压缩下限：Q4_0
低于 4-bit (Q3_0, Q2_0) 的量化会导致自回归上下文中的语义连贯性坍缩。在中间注意力层与输出投影层中经经验证实语义相似度 $S_c < 0.40$ (`tests/test_attention_ablation.rs`, `tests/test_ffn_bitdepth_isolation.rs`)。Softmax 中的角度误差累积放大使 2-bit 压缩在标准生产级 Transformer 中在数学上不可行。**官方认证的生产路径为 Q4_0 主体 + FP32 嵌入表。**

### 2. 语义检索与同义改写边界
海马提取器 (`embed_text`) 直接在输入嵌入表 ($W_E \in \mathbb{R}^{V \times D}$) 上执行*加权平均池化 (Weighted Mean Pooling)*，无需经过 Transformer 层前向计算，实现 $< 0.5\text{ ms}$ 延迟与 $0\text{ MB}$ 额外内存。
* **设计权衡与决策：** 检索依赖词汇或局部词义重叠。无法召回完全没有共同词汇的纯同义改写（标定测试集中的典型案例：$P_6$, $P_{11}$, $P_{13}$ 见 `tests/test_threshold_calibration.rs`）。
* **替代方案：** 引入专用的外部重型编码器 (如 MiniLM, BGE, E5-small)，但这超出了当前超轻量单二进制架构的范围。

### 3. 序列化并发
内存中的活跃模型由单读写锁保护 (`active_model.write()`)。传入的对话请求与 SSE 流式生成逐一串行执行。
* **设计权衡与决策：** 专为单节点、移动端与消费级边缘硬件 (Android/Termux, 笔记本) 设计，确保绝对线程安全、零指针竞争与低热功耗。
* **替代方案：** 并行多租户并发需要分页 KV 缓存架构 (*PagedAttention* / 独立 KV 插槽)，留待未来的多用户服务器版本实现。

### 4. 卫星居中向量 $\boldsymbol{\mu}$ 与受控降级
高维模型 (SmolLM2 576维, Qwen2.5 896维) 存在各向异性锥体坍缩。引擎通过存储于 `data/calibration/<model>.mu.bin` 的卫星居中向量进行校准。
* **受控平滑降级 (方案 C Fallback)：** 若 `.mu.bin` 缺失或损坏，系统自动无损降级为未白化阈值 $\tau^* = 0.50$，并在 SSE 遥测中标记 `whitening_missing: true`。
* **实测影响：** 检索时的拒绝率或误报率略有上升，但系统保持 100% 可用性与稳定性，绝不中断推理。

### 5. 组件生命周期与状态划分
* ✅ **生产环境认证：** 纯 Q4_0 推理、`FlatHeaderV2` 规范、Rust 原生主权 SSE 服务器 (Zero-Python)、带熵隙门控 ($\Delta_{top} \ge 0.12$) 的 Island Model RAG。
* 📁 **严谨闭环封存：** 注意力与 FFN 权重的 Q2_0 / Q3_0 极限压缩（因数学不可行性已正式封存）。
* 🔬 **独立前沿探索：** 自适应质心微调 (`fit_lm_head` 在干净语料上可用；长上下文全主体优化数值仍不稳定)。

### 6. 硬件与模型适用范围
性能与稳定性已针对消费级 CPU (AMD Ryzen 7 5800H, Intel Core i7-8550U, ARM Cortex on Android) 上的 0.1B 至 3B 模型完成形式化验证。超过 3B 的模型未认证在此类硬件上直接流畅运行。

---

## ⚖️ 开源协议
本项目采用 **GNU Affero General Public License v3.0 (AGPL-3.0)** 协议开源。详情请参阅 [LICENSE](LICENSE)。

---
*GAJE 协议生态系统 v1.7.4-alpha — 迈向主权超高密度神经推理与基因记忆。*
