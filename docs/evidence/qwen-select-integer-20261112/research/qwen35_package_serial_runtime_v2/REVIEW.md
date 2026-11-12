# Package serial runtime v2 — delayed startup log readiness fix

本v2仅修复独立审阅确认的一项启动竞态：HTTP健康检查可能先于supervisor PIPE转发完成日志而成功。`completed_engine_line` 在尚无完整EngineCore完成行时返回None；plain与capture两条路径都在发布startup proof前把它传播成api_ready pending，继续原Phase.__enter__的14100秒等待，不重置任何deadline。完整行存在时仍要求唯一、正确PID、DP0；错误PID/rank、重复完成行和文件异常立即拒绝。

冻结v1交付 SHA256 `0cfa28a77ea72d946dc200faf85e343f36b8b79f324c316eb80dd9c28bdc1988` 及其19成员全部保持原样。除readiness.py外，生产代码仅run.py的import namespace从v1指向v2；contract/io/history/service/adapter/budget/wiring/final_audit等生产文件逐字节相同。原预算示例仍passed=false，没有native/binding资格。

本次只运行受变动影响的30项SOURCE/MOCK：Readiness、DelayedLogReadiness、LifecyclePreservation、FullSequenceMock；session82929，start1fb02d，实际exit e86618/0。没有机械重跑v1的45项全量检查。新增覆盖零行/延迟完整行、分段日志、错误PID/rank/重复行、文件错误、capture/plain pending在proof之前返回、原等待成功无cleanup以及原deadline超时仍cleanup。首次测试失败来自新mock遗漏phase.log字段，补齐fixture后30项全部通过，生产实现未因此改变。

独立原发现：`qwen35-package-serial-runtime-v1-independent-source-review-20261112-v1.json` SHA256 `22d81bb143729f864eef80c719b969d3996b3b8f18b8cffe11de1709b657134d`；其stdlib delayed-log mock实际85bbc9/0。详细v1架构/预算说明保留如下；“缺少日志会拒绝”在v2意味着原startup deadline到期仍未取得完整证明时拒绝，不是首次health200时提前cleanup。

本目录是原 sampler runtime v12 的独立有限后继，接入冻结 full_validation_v2 的25阶段。没有修改冻结来源、仓库 package、原 History 或当前 binding。未创建 native run、未取得锁、未启动服务、未读取真实 model/tensor/packet 内容、未编译。导入与 `run.py --help` 不执行工作。

## 实现边界

- `history.py` 完整保留 `Bundle`、`Authority`、`expected_running`、`check_running` AST。完整 recovered/current catalog、LFM supplement、原失败模型退出1、source witness、两次 physical pass、root running ACK 全部保留。只有模块导入改为本后继。
- `service.py` 的 `Service`、`supervise`、bounded Popen.wait、direct-child owner/release ACK、实际 InvocationID/systemd 终止、递归双稳定 drain、失败 latch、原清理窗口保持原 AST。只改变 CLI 路径、phase properties 的有限 spec 参数。
- `io.py` 的 `RetainedLock`、原 inode/descriptor proof、atomic publish/read、metadata lineage、已知 publication race、terminal 双扫描保持原 AST。存储分类和持续 reserve 扩展到六树。
- `adapter.py` 保留 `LeaseObserver`、`ActiveScopeMonitor`、完整 declared_services、emergency retained-lock cleanup 和 server_audit。删除 research graph reuse、旧 worker/driver dispatch，仅用固定 package wiring。prepare/build 在同一原 Phase 生命周期下各自运行。
- build 在真实 native 子进程 wait、supervisor 实际成功终止后、进入任何 cleanup 前，额外要求 cgroup 已经双稳定排空。仍有 compiler descendant 就失败；原 cleanup 继续执行，但不能把杀掉未完成编译者后的空组判成 build 成功。
- `wiring.py` 是固定顺序的有限调用表和三个 metadata gate，没有新调度器、队列、重试、resume、fallback compilation 或 fast qualification。

## 25阶段

1 prepare view；2 fresh build/load 12图；3–7 baseline state_copy/numeric/cold/long/vectors203；8–12 candidate 同五项；13 baseline actual capture；14 baseline replay；15 candidate actual capture；16 candidate replay；17 shared13 controls；18 sampler pair gate；19 baseline quality；20 candidate quality；21 quality pair gate；22 baseline HTTP；23 candidate HTTP；24 full pair gate；25 外部独立 root final audit。

所有 native/model/replay/control 与 API 阶段均经原真实 supervisor/direct-child ACK。上一阶段成功退出、真实 systemd terminal、递归 drain 后才发布 completed stage，下一阶段再核对完整 prefix。完整科学与客户端直接调用冻结 `qwen35_package_full_validation_v2`，不继承 research 图、旧结果或资格。203向量返回值明确保存。两臂 sampler replay、所有实际 packets、13 controls 均先于 quality/HTTP。

stage25 不由 coordinator 自签。协调器只产生24 completed receipts 和 pending stage25。外部 root 在独立执行 guard 下调用 `final_audit.audit(spec_ref, review_ref, external_tool_exit_ref, execution_guard)`；它要求原 outer 实际已经退出、不同 observer、root 真实工具 session/退出0回执，然后独立复核所有服务与子进程退出、完整 History physical pair、来源/input/封存 artifact identities、全部 saved science gates 和稳定六树存储。输出依然 `full_model_qualified=False`；root 对新 package 作最终独立资格决定，禁止自动 best promotion。

外部真实工具退出回执的精确键为：`schema=qwen35_package_root_actual_tool_exit_v1`、`reviewer=/root`、`spec`、`admission`、`outer_owner`、`tool={kind:tool_session,session_id:真实ID}`、`exit_code=0`、`completed=true`、`observation_source=actual_external_tool_result`、`observed_unix_ns`。工具退出无法由子 Python 进程独立观测，必须来自 root 的实际工具结果，不能填预测值。

## 环境、readiness、IPC

coordinator/supervisor 只使用不含空格的 clean environment。capture JSON bindings 仅由原直接 Popen/exec environment 传给真实子进程，绝不塞进 systemd command 的简单环境解析器。build 子进程唯一 `NEURON_QWEN35_ALLOW_COMPILE=1`，其余为0。两臂 FAST 分别0/1；无旧 TRN1_QWEN35/QWEN35_SAMPLER hooks。

质量/HTTP 是 package 原生 `serve` 命令，没有自定义 worker 或 attachment。readiness 独立检查：原 child PID/start/parent/cgroup 的 exec 转换、精确 package argv 与实际 `/proc/PID/environ`、实际 owned API listening socket、健康响应、唯一同 cgroup 的默认 `VLLM::EngineCore` 子进程、该 PID 的已完成 load/warmup 日志。日志名 `EngineCore_DP0`、title、UniProcExecutor 同步 `load_model` 到 EngineCore 的完成日志顺序均由 STARTUP_FILES 的源码 pin 约束。缺少所有权证据立即拒绝；完成日志尚未转发时pending，原startup deadline耗尽仍拒绝。health 本身不够。完整实际 native readiness 仍需 future run 验证，SOURCE/MOCK 不伪装为实测。

capture 还要求真实 capture worker attachment 与 case、manifest、operation、source closure、API env/argv 一致；第一次 ready 时 packets 目录只能含真实 writer-owner.json，诊断前零 packets/receipt。实际B1/B4各16 quota仍由原 native capture/replay验证。

六个 server 路径固定为 `output_root/arm/{capture,quality,http}/rpc/UUID4`。output_root 只能是 `/mnt/nvme/neuron-work/qwp-<run_name>`；四个 package case 路径仍保留 full-v2 长前缀。所有六条 socket 路径在任何锁/native之前检查 UTF-8 字节数≤107。TMPDIR 与 RPC 分开。测量两臂环境只允许 `NEURON_CPU_SAMPLER_FAST`、`TMPDIR`、`VLLM_RPC_BASE_PATH` 三项不同；candidate HTTP 请求前再次比较实际 live hosts。没有 per-arm HF/XDG/Python cache 差异。

## 六树与有限资源接口

spec 的 `resource_budget` 必须指向 root-reviewed JSON，并同时出现在原 JSON-only `input_extension` 中，保留其原 metadata stat identity/unchanged 检查。`binary_inputs` 单独承载六个原科学 `.pt` 的 typed stat identity，不送进 JSON loader。没有默认 budget/current spec/launch binding。预算缺失、未 passed、case/root不一致、cap无效、父目录或原 SOURCE_FILES 身份不符、空间不够，均在原 lock.acquire 之前拒绝。

监控树为 output、control、model_path/view、artifact、compiler_cache、compiler_workdir。六树不同、不嵌套，父目录真实 identity 与 NVMe filesystem device必须匹配。首次 admission 全部不存在；prepare只创建view，build要求artifact/cache/workdir仍不存在。原输出2GiB预算完全保留：model832MiB/capture512MiB/logs512MiB/metadata128MiB/tmp+dirs64MiB。

新增独立 class 为 view、artifact、compiler_cache、compiler_workdir、build_auxiliary。每项均需 root 给 `allocated_bytes`、`single_file_bytes`、`entries` 正有限上限。build_auxiliary覆盖新增 prepare/build 的整个control子树（phase metadata、两个8MiB日志和TMPDIR），不挤占原科学日志预算。目录和 symlink 本身也计 `st_blocks*512`；普通文件同时检查 logical size/allocated size；每class/总量/entry均检查。其余原日志、模型、server、CPU caps 不提高。build MemoryMax26GiB/High25GiB/Swap12GiB、14400秒；build LimitFSIZE等于显式新增class单文件caps的最大值。完整全局cap为原2GiB加新增五class。

唯一 symlink 例外：view 根内的精确 SOURCE_FILES（除新 config.json），只能链接到 `/mnt/nvme/models/Qwen3.5-2B` 对应 pinned 原文件。link与target身份都记录；既有权重目标字节不作为新复制payload计费。artifact/cache/workdir任何symlink或外部hardlink拒绝。源码固定 `save_sharded_checkpoint=False`，没有两个持久TP shards预算假设。RPC只接受六个已声明目录直属UUID4 socket。编译器未知删除/重命名不被猜测性放行；现版只保留原已知 atomic publication/declared TMPDIR有限重试，其他不一致严格失败。

原 outer 与每个 supervisor 的 guard 在运行中持续触发资源检查（目标间隔0.5秒，受每次真实扫描/工具观察耗时影响），不是只在step末核对。每次都保留：root floor28GiB + reviewed root headroom；NVMe floor30GiB + **所有预算未使用额** + reviewed NVMe headroom。build成功后封存view/artifact全部stat identities和12个graph.neff身份；每轮检查与stage25复核，不读取其payload。terminal再作两次稳定六树扫描。

周期扫描并非硬文件系统quota；单文件LimitFSIZE为硬限制，总量越限则通过原失败latch/owned cleanup停止。预算必须留监控和清理期间headroom，本交付没有证据能证明编译峰值或最小“足够容量”。

## 未绑定的有限试验建议

`budget.example.json` 故意 `passed=false`、缺identity、带blockers，不能启动。它仅提供root可以审阅的可失败试验ceiling，不表示新12图能完成：

| class | allocated cap | single file cap | entries |
|---|---:|---:|---:|
| view | 32MiB | 16MiB | 64 |
| artifact | 1GiB | 1GiB | 256 |
| compiler_cache | 2GiB | 1GiB | 4096 |
| compiler_workdir | 3GiB | 1GiB | 8192 |
| build_auxiliary | 256MiB | 128MiB | 2048 |

新增共6.28125GiB，另原science2GiB及NVMe headroom1GiB，初始NVMe至少39.28125GiB free（含30GiB floor）。root headroom1GiB对应root至少29GiB free。实际admission用字节级statvfs，不采用约数。更小/更大的cap只能由root重新review；不自动扩容、不续跑、更不删旧结果腾空间。历史旧artifact实际294494208B≈280.8515625MiB只是已完成输出观测，不是新12图artifact/cache/workdir峰值，不是可删额度，也不保证此建议能跑完。

## SOURCE/MOCK 验证与交付

使用固定3.10 Python、`-B`和native import blocker运行 `test_source.py`。测试覆盖完整24阶段mock与外部25pending、build/sampler/quality失败停止、预算/IPC/env、实际readiness约束的mock、六树allocation/entry/file caps、已知publication race、生命周期AST保留、真实build完成前强制drain、封存artifact身份变化拒绝。不会启动Popen/systemd或取得真实锁。

`source_diff.patch` 是本v2相对冻结v1的最小差异；v1到原v12的完整差异仍保留于冻结v1。`ast_preservation.json` 记录逐字节保留的生产文件和未改变的等待/生命周期AST。`input_pins.json` 记录本次核对的冻结文本来源与startup文本来源；`source_delivery.json` 冻结本后继文件。所有mock结果仅 SOURCE/MOCK；仍需独立root源码review、reviewed实际预算/六树/所有input身份、最新完整History与真实tool handoff、native full25执行和外部资格决定。
