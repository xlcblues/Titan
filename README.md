# Titan

Titan 是面向 Windows 桌面端的三维重建软件，采用自研的 Facet Radiance Field 深度学习方法，定位与 3DGS、NeRF 同类，并与 Mars、Ceres、Neptune 构成同系列产品体系。

## 特性

- 上传图片进行三维重建，支持 8GB 显存显卡
- 查看重建后的渲染图像与三维场景
- 深浅主题切换，跟随系统主题
- 极简 Titan Logo 与统一 UI 设计规范
- Tauri 2 + React + Rust 桌面架构
- 应用日志、崩溃报告与环境诊断
- 自动更新与可分发安装包

## 里程碑

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M0 | Logo、主题令牌、设计规范、预览页 | 已完成 |
| M1 | Tauri 桌面骨架、侧边栏/顶栏、主题切换、路由 | 已完成 |
| M2 | 项目创建、图片拖拽导入、缩略图、SQLite 持久化 | 已完成 |
| M3 | 重建管线与渲染场景接入 | 已完成 |
| M4 | 任务引擎、指标回传、渲染导出与真实场景预览 | 已完成 |
| M5 | 自动更新、崩溃上报、日志、性能优化、打包签名 | 已完成 |

## 工程化与分发

- 日志：`tauri-plugin-log` 写入应用日志目录，单文件 2MB、保留 5 份、按天轮换，并转发 Webview 日志。
- 崩溃上报：Rust panic hook 与前端未捕获异常都会写入 `app_data_dir/crash/`。
- 环境诊断：设置页可检测 GPU、Python、CPU、日志与崩溃目录。
- 自动更新：使用 `tauri-plugin-updater`，签名公钥位于 `src-tauri/.updater/titan.key.pub`，私钥不提交。
- 打包：`bundle.createUpdaterArtifacts` 已开启，构建时会同时生成安装包与更新签名产物。

发布流程：先运行带签名环境变量的 `npm run tauri build`，再把生成的 `Titan_<version>_x64-setup.exe` 与 `.sig` 上传为 GitHub Release，并同步更新 `update/latest.json` 后推送。`plugins.updater.endpoints` 指向 GitHub Release 的 `latest.json` 资源地址，更新包使用同一把私钥签名。

## 本地开发

前置依赖：Node.js 20+、Rust stable、Tauri 2、Windows 10/11。

```bash
npm install
npm run tauri dev
```

前端与后端检查：

```bash
npm run build
cargo check
```

## 目录结构

```text
src/          前端 React 应用
src-tauri/    Rust / Tauri 后端
python/       Python 重建引擎（TFRF）
design/       Logo、UI 规范与设计预览
docs/         技术选型、开发流程与算法方案
assets/logo/  Logo 源文件
```

## 文档

- [Logo 设计方案](design/logo-design.md)
- [UI 设计系统](design/ui-design-system.md)
- [技术选型与开发流程](docs/技术选型与开发流程.md)
- [算法方案](docs/算法方案.md)
