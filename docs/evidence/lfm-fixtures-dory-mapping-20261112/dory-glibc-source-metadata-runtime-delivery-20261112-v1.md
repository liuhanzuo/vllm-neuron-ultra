# Dory glibc 源码元数据：原串行封装接入 v1

已将 root 接受的固定元数据 API 接到原 Dory inventory 父子进程封装。仅新增 4 个运行源码和 2 个检查文件，没有生成当前执行 plan/review、修改 handoff/catalog 或启动实验。

## 新文件与复用范围

- `run_dory_glibc_source_metadata_root_v1.py`：从已执行的 Ubuntu metadata root v5 继任，仅改工作绑定、worker 路径和结果 schema。
- `dory_glibc_source_metadata_worker_v1.py`：原 worker v5 的专用继任，真实父进程 ACK 后重新认证 7 份固定输入并运行新 adapter。
- `dory_glibc_source_metadata_binding_v1.py`：固定输入、请求政策、已接受源码摘要、父子 admission 绑定，以及下载结果的完整保存文件重放。
- `dory_glibc_source_metadata_runtime_v1.py`：固定稳定读取、有限 HTTPS、独占原子输出发布及原 collector API 调用。
- `check_dory_glibc_source_metadata_runtime_v1.py`：18 项运输与输出检查。
- `check_dory_glibc_source_metadata_integration_v1.py`：12 项绑定、重放、实际 worker dispatch 和原生命周期保持检查。

沿用 descriptor_inventory_runtime_v2 与 run_dory_serial_stage_root_v5 原语。原锁身份、父进程保留且不继承的 FD、完整 ACK、实际子进程退出、进程组排空后释放、失败时保留租约的规则均不变。两个 guard 与原 parent/worker 的 AST 一致；资源/启动/清理相关调用 AST 一致。

原 inventory 预算完整保留：2 GiB 地址空间、600 CPU 秒、900 秒阶段总期限、128 MiB 输出、64 MiB 单文件、8 MiB 日志、CPU 5 / nice 19、根盘 28 GiB 与 NVMe 30 GiB 下限。稳定读、输出基元和完整存储统计复用原代码。

复用 adapter_v6 的稳定读/输出/NoRedirect 会静态加载原 GPG 状态桥及其 adapter_v2/package_sources_v1 依赖，因此完整源码闭包明确包括这两个已有文件；本接入不调用任何 GPG 程序或新的签名验证。交付清单包含完整 20 个本地源码文件及摘要，供未来 root 的 source_sha256 使用。

## 有限请求与认证

保存数据只能是已接受 core.INPUTS 中的 7 份固定文件。父进程、worker ACK 后、collector 入口分别重新认证这一固定集合；不允许从 plan 加入其他数据文件。

网络顺序固定为 jammy-updates 和 jammy-security 两个 by-hash Sources.xz，压缩总量 981020 B。每份都检查完整压缩摘要、解压摘要和精确 source stanza。只有两份索引完整认证且文件表一致之后，才允许从其中导出的唯一 DSC URL；最多 3 次请求。没有重定向、代理、重试、别名/镜像回退或源码归档下载。两索引加最大 DSC 共 2029596 B，处于原 API 的 4 MiB 新传输上限内。

HTTPS 保留系统默认 TLS 验证，明确禁用代理与重定向。每次要求唯一且等于固定预期字节数的 Content-Length、identity 内容编码、完整 body，无 Transfer-Encoding 或 Content-Range；读取多出一字节或提前结束均拒绝。随后验证完整摘要。这里 EOF 明确指完整 HTTP message-body EOF，**不证明 raw socket EOF，也不证明协议消息体之外不存在其他字节**。

socket 超时为 5 秒，每次 open/read 前后检查 60 秒请求期限；DNS 或其他阻塞调用不能被 Python 检查立即中断，仍由原 900 秒父进程阶段监控及整个子进程组清理兜底。这不是对 DNS 独立 60 秒硬截止的证明。

元数据文件先用原持久 writer 写入固定临时文件，再独占链接到最终名字、删除临时名字、fsync 目录并稳定重读。不会覆盖已有最终文件；中断留下的临时证据计入原整体输出预算。

worker 在保存完成后使用同一已固定 core 对原压缩索引/DSC 文件完整重放：重读、哈希、解压、源包匹配、DSC 文件表与最终 JSON 一致性全部复查。保存的 HTTP 收据、请求数、字节总量及所有结论字段也要一致，之后才写 metadata-result。该 worker 内重放不能替代 root 独立确认实际工具退出。

## 实际检查结果

新增两入口合计 30 项通过，工具 `45a991`、exit 0，unittest 报告 0.211 秒。此前单独 runtime 18 项为 `71b8ac/0`，integration 12 项为 `ff0f6f/0`。原 core 25 项已由 root 独立验收，本次没有重复执行，也不计入新 30 项。

检查使用合成索引/DSC、HTTP/proc/锁/进程 mock 与临时输出文件；没有真实网络、模型/native、ELF、RECORD、原模型 raw、进程/服务/锁观测，没有读取那 7 份实际保存数据。完整 worker 业务 dispatch AST 在注入端口下实际执行；原 guard/资源/生命周期 AST 与原语身份另行核对。

只复跑新增检查，在 research 目录执行：

```sh
python3 -B -m unittest check_dory_glibc_source_metadata_runtime_v1 check_dory_glibc_source_metadata_integration_v1
```

## 当前边界

历史 by-hash 可能已不再提供；缺少 Content-Length 或采用 chunked 的响应会被拒绝；实际主机网络、可用性和时长均未验证。缺少精确 glibc 版本时只保留 `exact_source_version_missing`，不再尝试 DSC。原 API 的历史已签名快照声明保持，不声明最新仓库状态，也不单独验证 DSC 的 PGP 签名。

真实 parent/worker 尚未执行。root 仍需在现有串行工作完成后独立审核并绑定最新前置退出、完整物理历史、原 source_sha256 和 plan/review；再取得实际外部工具退出与完整排空审计。此源码交付不提供任何自动启动。

Dory 2 loops、56 份独立 attention history、TP1 不变。原 15 条 loader 边仍未解决，没有源归档内容检查、loader 语义、native ABI、全模型正确性或新增提速结论。
