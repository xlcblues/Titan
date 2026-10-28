# Titan

Titan 是面向 Windows 桌面端的三维重建软件，采用自研的 Facet Radiance Field 深度学习方法，定位与 3DGS、NeRF 同类，并与 Mars、Ceres、Neptune 构成同系列产品体系。

## 特性

- 上传图片进行三维重建，支持 8GB 显存显卡
- 查看重建后的渲染图像与三维场景
- 深浅主题切换，跟随系统主题
- 极简 Titan Logo 与统一 UI 设计规范
- Tauri 2 + React + Rust 桌面架构

## 里程碑

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M0 | Logo、主题令牌、设计规范、预览页 | 已完成 |
| M1 | Tauri 桌面骨架、侧边栏/顶栏、主题切换、路由 | 已完成 |
| M2 | 项目创建、图片拖拽导入、缩略图、SQLite 持久化 | 待开始 |
| M3 | 重建管线与渲染场景接入 | 待开始 |

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
design/       Logo、UI 规范与设计预览
docs/         技术选型、开发流程与算法方案
assets/logo/  Logo 源文件
```

## 文档

- [Logo 设计方案](design/logo-design.md)
- [UI 设计系统](design/ui-design-system.md)
- [技术选型与开发流程](docs/技术选型与开发流程.md)
- [算法方案](docs/算法方案.md)
