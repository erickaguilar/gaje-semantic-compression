# 📋 更新日志 (CHANGELOG) — GAJE 协议

[![Language: English](https://img.shields.io/badge/Language-English-blue.svg)](CHANGELOG.en.md) [![Language: Español](https://img.shields.io/badge/Language-Espa%C3%B1ol-yellow.svg)](CHANGELOG.es.md)

## [1.7.1-alpha] - 2026-08-29
### 新增 (Added)
- **Rust 主权单二进制文件 (`gaje-cli`) 与内嵌 Web UI**:
  - 集成 `rust-embed`，将精简版 `index.html` (7.2 KB) 及静态资源直接编译打包进二进制 `.rodata` 内存，彻底脱离外部磁盘文件依赖。
  - 原生 HTTP 服务器 (`gaje-cli serve`) 具备混合分发机制：生产独立模式下实现 RAM 零磁盘延迟分发，开发模式下支持热重载。
- **全新 CLI 运维与生产级子命令**:
  - `gaje-cli export-flat`: 基于 Rayon 多线程与 64 字节 SIMD 对齐的零拷贝导出器，生成带内嵌 GTOK 的 `.flat` v2 模型。
  - `gaje-cli benchmark` (别名: `bench`): 性能评测套件，支持测量 mmap 冷启动、TTFT、生成吞吐量 (tokens/s) 以及在 JSONL/文本语料上的困惑度 (PPL) / 交叉熵评估。
  - `gaje-cli dataset-build`: 多格式对话与指令数据集标准化构建工具，输出干净标准的 JSONL 格式。
  - `gaje-cli audit`: 权重深度数学审计，确保 `0 NaN / 0 Inf` 及质心熵平衡。
- **原生验证套件与工作流迁移指南**:
  - 新增 Rust 原生集成测试 `tests/cli_standalone_test.rs`，无需 Python 环境即可全面测试 CLI。
  - 发布 `scripts/README.md`，提供历史 Python 脚本与 `gaje-cli` 子命令的完整对照迁移矩阵。
  - 更新 `README.md`、`README.en.md` 与 `README.zh.md` 中的 `gaje-cli` 快速入门指南。

### 变动 (Changed)
- 从嵌入包中排除重量级与辅助页面 (`docs.html`, `architecture.html`)，将内嵌资源体积控制在 500 KB 以内。
- 模块化拆分 `index.html`，通过遥测弹窗懒加载机制将其体积由 22.5 KB 缩减至 7.2 KB。

## [1.7.0-alpha] - 2026-08-24
### 新增 (Added)
- **Flat GAJE 模型标准化与 Hugging Face Hub 官方发布**:
  - 推出统一转换管道 `scripts/transmute_qwen_models.py`，构建 `.flat` v2 混合格式模型。
  - 在 Hugging Face Hub (`eaguilar/gaje-models`) 发布 3 款标准模型：`gaje_nano_1.5b.flat`、`gaje_prime_3b.flat` 与 `gaje_ultra_7b.flat`。
- **Web UI 与双主题架构 (Y2K Dark / Scandinavian Light)**:
  - 北欧浅色模式 (`y2k-light` / 0px radius) 设为全局默认主题。
  - 模块化两层聊天工具栏与 macOS 风格交互控制按钮。
- **浏览器端 WebAssembly (Zero-Server) 与 Vercel 部署支持**:
  - 自动检测静态托管环境并平滑降级至浏览器内 WASM 本地流式推理。
