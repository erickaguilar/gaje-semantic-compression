# 📋 更新日志 (CHANGELOG) — GAJE 协议

[![Language: English](https://img.shields.io/badge/Language-English-blue.svg)](CHANGELOG.en.md) [![Language: Español](https://img.shields.io/badge/Language-Espa%C3%B1ol-yellow.svg)](CHANGELOG.es.md)

## [1.7.3-alpha] - 2026-09-11
### 新增 (Added)
- **生产级原生主权 HTTP 服务器 (`gaje-server` / `gaje-cli serve`)**:
  - 生产环境彻底移除 Python 运行时依赖：采用 Rust 原生 HTTP 服务器 (`tiny_http` + `rust-embed`)，直接从 `.rodata` 内存分发完整的 Web UI 与静态资源。
  - 真正的实时逐 Token SSE (*Server-Sent Events*) 流式传输，包含分块传输与 HUD 实时遥测 (`__gaje_metrics__`、TPS、延迟、压缩比及 DNA 审计链)。
  - 基于 `GTOK` 词表自省与 `FlatHeaderV2` 的规范化对话模板解析 (`ChatML`, `Llama3`, `Llama2`, `Gemma`, `Classic`) 与停止词检测，彻底根除基于文件名猜词的启发式逻辑 (`is_born`)。
  - 安全的动态热插拔 (Hot-Swap) 与序列化并发机制，通过 `active_model.write()` 并在加载新模型前强制显式解映射 (`munmap`) 旧模型内存。
  - 全面经验真理认证：`strace` 系统调用审计证实运行时零 Python 子进程调用 (`python_version: "None (Native Single-Binary)"`)，并通过多流并发压力测试。
  - 模型按质量/优先级智能排序 (旗舰 `max.gaje` 置顶) 及通过文件元数据与 `chrono` 呈现真实磁盘修改日期。

### 变动 (Changed)
- **统一模型命名法与规范扩展名 (`.gaje`)**:
  - 术语整合：正式确立 `.flat` 与 `.gaje` 为相同原生零拷贝平面二进制格式的同义词 (4096 字节 `FlatHeaderV2`，魔数 `b"GAJE"`)。
  - 所有导出与原生诞生模型规范扩展名正式统一为 **`.gaje`**。
  - 明确模型 (`.gaje`) 与 Island Model 持久化关联记忆索引 (`.gmem`，存放于 `<model>_memory/` 卫星目录) 的语义边界。
  - Web UI (`model_manager.py` 与 `toolbar.js`) 无缝兼容并展示 `.gaje` 和 `.flat` 扩展名。
  - 转换脚本与 CLI 工具默认输出扩展名切换为 `.gaje`。

## [1.7.2-alpha] - 2026-09-04
### 新增 (Added)
- **主权二进制格式统一 (`.gaje` v2) 与原地自适应**:
  - 统一 `.flat` 与 `.gaje` 为单一原生零拷贝二进制标准 (`mmap`，魔数 `b"GAJE"`)。
  - 固定 4096 字节标头 (`FlatHeaderV2` / `FlatHeaderV3`)，支持持续自适应、基因谱系与密码哈希追踪 (`lineage_parent_hash`, `lineage_current_hash`, `num_mutations`, `num_overrides`)。
  - 原生质心原地突变引擎 (`src/io/adaptive.rs`)，避免基础权重冗余复制和块错位。
  - 新增 CLI 子命令：`gaje-cli mutate` 与 `gaje-cli history`。
- **独立终端工具模块 (`src/cli/`)**:
  - 创建原生命名空间 `crate::cli` (`src/cli/models.rs`, `src/cli/tools.rs`)，将终端交互与底层纯 I/O 分离。
  - 清理 `src/bin/` 中的探索性脚本，确立 `gaje-cli.rs` 为唯一主权二进制。

### 变动 (Changed)
- **彻底剔除 `redb` 及过时依赖**:
  - 从 `Cargo.toml` 中移除 `redb` 和 `lz4_flex`，原生加载器全面基于 `GajeFlatFileReader` 实现零拷贝直接映射。
- **架构重构与语义澄清**:
  - `src/compute/sampler.rs` 重命名为 `src/compute/sintergic.rs`，聚焦 Grinberg 辛特吉理论与拉格朗日相空间。
  - 标准自回归采样器 (`sample_min_p`, `sample_top_p_core`) 统合于 `src/compute/sampling.rs`。
  - `src/nn/linear/database.rs` 重命名为 `src/nn/linear/storage.rs`，使用 `WeightStorage` 取代数据库称谓。
  - GGUF 子系统集中整合至 `src/io/gguf/`。
- **移动端与 Termux 环境全绿测试**:
  - 消除 100% 编译告警 (`cargo check --bins --tests`)。
  - 集成测试动态基于 `std::env::temp_dir()` 执行，Android/Termux 环境通过率 100%。

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
