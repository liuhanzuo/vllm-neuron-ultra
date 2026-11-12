# Package r2 root prepare/accept helpers — source-only delivery

这两个新 helper 从冻结 r1 派生，使用已经冻结的 runtime v4 和 preparer v3，保留原完整 History、全部 source/identity 验证与 874 个 LFM child 核对。没有修改 r1 或 v1–v4 运行器，没有实际执行 prepare、History、锁、服务、编译或模型。本交付不等于启动许可；结构审阅通过后仍须实际准备、完整接受成功及 root binding，才可依据原程序继续 runtime 准入。

`prepare_qwen35_package_full_root_r2_20261112_v1.py` 要求显式 CLI path/SHA refs：catalog、idle、runtime-review、preparer-review、stat-inputs、budget。catalog v112 和 idle v584 的路径/SHA 冻结；另四项路径固定为 root 指定 r2/v4/v3 文件，SHA 必须显式传入，随后用原稳定带哈希 reader 验证。runtime/preparer delivery 仍绑定确切 SHA。r1 stat 中 `pkg-r1-20261112` 推导为 `pkg-r2-20261112`；output/control 使用短 r2 名字，四个尚未创建的 model/build 路径由 root r2 stat/budget 保留 r1 名字。running handoff 固定为 v585 package-full-r2。view/artifact/cache/workdir 是否仍新鲜由原六树 `fresh_paths` 检查裁决。

prepare 的 `prep.prepare` 调用返回之后，所有观察身份复查、资源 reservation、exclusive publication 与 false candidate 回执体保持原完整 AST。source preparation 来源保留现有全量 pins，包含两个 r2 helper；普通 8 MiB 和指定 64 MiB metadata 边界保持。

`accept_qwen35_package_full_root_r2_20261112_v1.py` 还要求实际 freshly published candidate SHA，不能自行按文件名选未知 candidate。它核对 candidate 内的 refs 与 CLI 一致、r2 output/run/review/spec/running 路径一致。原锁内 `try/finally` 整体与 r1 AST 相同，只有成功 binding 输出文件名改为 r2：完整 `Bundle.resolve()`（未传 `verify_sources=False`）、`unchanged`、freshness、874 children、全部 loaded source 校验、资源重查、实际最新 handoff、最后成功发布及 `finally lock.release(True)` 全部保留。原 2 GiB AS、600 秒 CPU/guard/alarm 限制保持。失败不会新增 fallback 或省略历史检查。

执行体置于 `main`，CLI 解析在 runtime/History 模块导入之前。因此 import 和 `--help` 不执行准备、资源变更、History 或锁；只有显式完整 CLI 才进入原操作流程。

14 项 SOURCE/STDLIB MOCK 通过，实际工具 chunk `02e9dc` / exit 0：CLI 缺参/错误路径/错误 SHA、冻结 catalog/idle SHA、无副作用导入/help、完整 AST 保留；只抽取原 `try/finally` 并以合成 Bundle/lock/io 执行，验证正确 874/source/History 路径，以及 History 失败、873 children、identity 变化、source 不匹配、freshness 变化均不发布成功 binding，并执行模拟锁释放。没有运行真实 History、prepare 或锁。

`qwen35-package-full-r2-root-helpers-invocation-20261112-v1.json` 保存 root 已提供且本次只读验 SHA 的精确 prepare argv；accept argv 刻意缺少须由未来实际 candidate 返回的 SHA，并明确要求追加该参数，未伪造 candidate digest。此 JSON 是可审阅的命令参数，不是已执行回执。

最终启动仍取决于成功的 root binding。root 另写的 ACK helper 是外部真实 session/owner 观察工具，独立审阅且不强行加入 worker source 闭包；本交付没有代签它或假定已存在真实 runtime session。
