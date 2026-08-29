# 🧬 GAJE 语义压缩 — 全球开发协议与智能体开发指南

[![Language: Español](https://img.shields.io/badge/Language-Espa%C3%B1ol-yellow.svg)](AGENTS.md) [![Language: English](https://img.shields.io/badge/Language-English-blue.svg)](AGENTS.en.md)

本文档定义了 **GAJE 项目全局架构**、代码仓库结构以及所有开发者和人工智能智能体（Antigravity、Claude、Gemini、Copilot 等）必须严格遵守的**最高操作与技术规范**。

---

## 1. 项目全局概述

**GAJE (Genetic Adaptive Joint Embedding / 基因自适应联合嵌入)** 是一个结合神经基因记忆与高密度语义压缩的混合框架。其目标是以极低延迟压缩并检索基因组与神经网络稠密表示，融合遗传算法、零拷贝内存映射 (mmap) 与侧向抑制 K-WTA。

### 核心组件：
* **原生内核 (Rust):** 提供极致效率的压缩/解压、向量检索、mmap 与 SIMD 数学内核。
* **单二进制 CLI (`gaje-cli`):** 包含完整推理、内嵌 Web UI、模型拉取、转换与审计的独立原生工具。
* **研究与实验层 (Python):** 提供 PyTorch / HuggingFace 集成及 PyO3 绑定接口。
* **Web 用户界面 (Web UI):** 交互式聊天面板、遥测 HUD 与双主题系统。

---

## 2. 开发者与 AI 智能体黄金法则

### A. 核心主权与内存管理 (Rust & Python)
1. **高效内存:** 严禁在关键循环中进行无意义的张量大规模预分配。优先采用指针、引用与零拷贝共享内存 (`Arc<Vec<u8>>`, `mmap`)。
2. **原生主权:** 高性能功能与管理 CLI 工具必须作为 `gaje-cli` (Rust) 的子命令实现，杜绝一次性杂乱脚本。

### B. 生产环境单二进制与 Web UI
1. **Web UI 内嵌:** `gaje-cli serve` 内置了纯 Rust `rust-embed` 打包的聊天界面，可在无需 Python 或外部磁盘文件的情况下独立运行。
2. **轻量化原则:** 辅助页面 (`docs.html`, `architecture.html`) 严格排除于独立二进制包之外，确保体积保持在 $< 500\text{ KB}$。

### C. 严禁主动轮询与长任务 Token 守恒
1. **禁止 Active Polling:** 禁止在 AI 交互中频繁调用 `manage_task status` 轮询后台编译或训练进程。
2. **响应式唤醒 (Reactive Wakeup):** 使用 `run_command` 发起耗时任务后，智能体应立即暂停工具调用并交出轮次，等待系统在任务完成时自动唤醒。

---

## 3. 常用开发命令

```bash
# 编译发布版原生 CLI
cargo build --release --bin gaje-cli

# 运行 Rust 测试套件
cargo test --lib
cargo test --test cli_standalone_test

# 启动内嵌独立 Web UI
./target/release/gaje-cli serve --port 8080
```
