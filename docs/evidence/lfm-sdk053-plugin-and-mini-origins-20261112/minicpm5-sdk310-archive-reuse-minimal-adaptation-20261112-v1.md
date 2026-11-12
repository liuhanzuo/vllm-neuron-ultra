# MiniCPM5 SDK 3.10 归档核验：离线输入与最小复用方案

本次仅准备文本输入，没有下载、安装、读取 ELF／deb／模型数据、导入模型或 ctypes、取得锁、创建运行 handoff/spec/admission。所有准备命令均受 CPU 7、nice 19、AS 512 MiB、CPU 60 秒限制。

## 可以复用什么

已核对 `dory-six-deb-source-root-review-20261111-v3.json` 的通过结论及源码摘要；所选 13 份源码均与已执行 Dory r3 的固定来源一致。Dory 六包下载 r3 和保存归档重放 r1 的实际退出均为 0，分别保存了 31 和 25 条事件。

可原样复用：

- `dory_deb_transport_v3.py`：`download`（第 26 行）、`file_ref`（第 10 行）、`TarPipe`（第 53 行）、`inspect`（第 86 行）。提供单次固定 HTTPS 地址、禁止重定向、全归档长度/SHA256、前后身份核对、固定 dpkg-deb、完整管道 EOF/退出和失败证据。
- `dory_deb_tar_members_v2.py`：完整有界 tar 流、全部普通成员哈希、限定目标与内部链接验证；无需新建 ar/XZ/Zstandard 解压框架。
- `dory_descriptor_inventory_runtime_v2.py`、`dory_serial_stage_contract_v2.py`、`run_dory_serial_stage_root_v5.py`：现有串行生命周期、父进程保留锁、子进程 ACK、历史核对、资源检查、整进程组结束与释放。这里只复用源码方案，不调用。
- `dory_ubuntu_package_sources_v4.py` 的有界 `stanzas` / `release_indexes` 解析逻辑可以用于新绑定。其 `discover` 和 `bind_inputs` 固定旧六包，不能直接用于 Mini。

## 八个精确包

具体 URL、SHA256、签名 Release／索引引用、完整索引内匹配段的字节偏移与摘要，保存在 `minicpm5-sdk310-archive-member-next-inputs-20261112-v1.json`。首选地址均为 `https://archive.ubuntu.com/ubuntu/pool/`，符合现有 transport；security 地址仅保留为已签名来源，不自动重试。

| 包 | 版本 | 类型 | 完整归档字节数 |
|---|---|---|---:|
| python3.10-minimal | 3.10.12-1~22.04.18 | 必需 | 2,277,402 |
| libpython3.10-stdlib | 3.10.12-1~22.04.18 | 必需 | 1,850,286 |
| libffi8 | 3.4.2-4 | 必需 | 22,326 |
| libpython3.10-minimal | 3.10.12-1~22.04.18 | 候选依赖 | 816,508 |
| libssl3 | 3.0.2-0ubuntu1.30 | 候选依赖 | 1,907,006 |
| libexpat1 | 2.4.7-1ubuntu0.9 | 候选依赖 | 96,926 |
| zlib1g | 1:1.2.11.dfsg-2ubuntu9.2 | 候选依赖 | 58,406 |
| libcrypt1 | 1:4.4.27-1 | 候选依赖 | 82,004 |

三个必需包共 4,150,014 字节，五个候选包共 2,960,850 字节；八包合计 7,110,864 字节。候选来自 bootstrap 的 hashlib/json/gzip 使用及已安装包 Depends；这些都不是实际 ELF DT_NEEDED 或进程映射证据。

12 个拟核验成员中，五个必需目标是 `/usr/bin/python3.10`、动态 `_ctypes.cpython-310-x86_64-linux-gnu.so`、`ctypes/__init__.py`、`ctypes/_endian.py` 和规范路径 `libffi.so.8.1.0`。另七个候选目标是 `_json`、`_hashlib`、`libcrypto.so.3`、`libssl.so.3`、`libexpat.so.1.8.7`、`libz.so.1.2.11` 和 `libcrypt.so.1.1.0`。`_json` 位于必需的 stdlib 包中，但其实际加载仍未知。

所有目标都只有本次 dpkg list 文本及 lstat 身份依据；归档成员路径尚待真实归档核验，成员 SHA256 尚未获得。`original_observed_reference=null`，旧全内容摘要数量为 0，不能填入 Dory 的旧 11 个摘要。所有实际加载状态显式为 null。12 个目标不覆盖完整 Python bootstrap 的全部纯 Python 源码或全部动态依赖；将来完整 SDK source-map 验收仍需补齐。

## 为什么旧 runner 不能直接使用

旧 `dory_deb_input_binding_v3.py` 第 7–9 行固定旧 NEXT/TOOL/TRUST；`authenticated`（第 22 行起）固定六包、4,234,134 字节、12 目标，并逐条对照旧 discovery 中的六包列表。Mini 的八包地址虽然来自同一已签名索引，但不在旧结果的六包条目内。不能只把六改成八，或把 SDK v2 缺口报告冒充新的根审核决定。

旧 adapter 的 `run` 以及 binding 的 `save_collected`还固定 `originals=11`、`current=12`、六包完成事件。需要明确采用本离线集合的八包／12 目标／0 个旧摘要，使用独立 Mini 工作类型和结果 schema，保留原有所有比较和失败处理。必要/候选标签必须原样带入结果，核验候选文件内容并不等于证明该库被加载。

旧外部 recorder 第 53 行最多接受 32 条事件：八包下载需要 **41 条**，八包重放需要 **33 条**，两者都会被拒绝。最小后继必须绑定具体顺序和数量：下载每包 5 条、重放每包 4 条，再加 1 条完整结束事件；最大值恰为 41，不能使用无界收集。第 58 和 64 行的固定结果计数也需一起改动。

## 最小源码差异清单（尚未实现）

1. 新建一个 Mini 专用输入绑定，固定本 next-inputs 的完整摘要、八个 package/version/architecture、12 目标、0 个旧摘要及原 dpkg-deb authority。复核已完成 Dory 签名元数据实际退出和 recorder 退出；重新核对完整 Release／明文 Packages 哈希，再逐条核对八包完整 stanza。旧 signed result 只用于 Release/索引信任链，不能借用旧六包的包级资格。状态文件和已安装版本若变化则停止。此步骤无需重新下载索引。
2. 从现有 adapter 作最小后继：仅替换输入绑定、确切计数及 Mini 结果/事件标识；下载、解压、完整归档前后核对、当前目标全哈希和保存重放均复用原实现。没有已验成员摘要时不能先宣称成功。
3. runner/worker/recorder 只更换上述绑定、adapter 和工作标识的模块连接；recorder 补齐明确的 41/33 事件协议和 0 个旧摘要要求。父进程保留锁、所有历史条目、守卫、ACK、结束排空及释放逻辑保持原实现。新来源需要后续独立源码审核；本次不生成运行计划。
4. 本输入的 `saved_package_names` 只列选定的规范成员名，因此可以原样调用 transport.inspect；它目前仅特别认识旧 libc/tcmalloc 别名。libffi、expat、zlib 和 crypt 的 SONAME 别名观测另列于 `observed_aliases_pending_archive_verification`，没有混入已验证别名。若下一阶段要求验证这些归档别名，需给 inspect 增加四个明确映射并针对性检查；不能静默宣称已覆盖。
5. 新运行使用新鲜完整历史、来源绑定和审核。保存数据重放应在生产者结束后使用同一精确集合，保留完整归档和事件；不要改写任何旧 Dory／Mini 结果。

尚没有可直接执行的 Mini 归档命令。原 runner 的 CLI 形状为 `python3 -B -s <runner> --plan <path> --plan-sha256 <digest> --review <path> --review-sha256 <digest>`，但它固定旧六包，不能把本离线输入传给旧 plan 或旧 review。最小源码后继审核完成并由主任务绑定新鲜串行运行后，才具备实际执行命令。

## 资源与验收边界

原 Dory 归档准备阶段为 AS 2 GiB、CPU 600/605 秒、900 秒、单文件 64 MiB、总输出 128 MiB；这些只是已审历史参数，并非本次新增准入。Python 解释器归档 2,277,402 字节大于 Mini 科学诊断的原单文件 2 MiB，因此不能把下载塞进 Mini worker 然后暗中抬高 FSIZE。后续归档准备须使用经过主任务绑定的独立串行阶段，Mini 科学阶段保持原限额。

Mini 原有 **9 项 native 检查、1,284 行科学描述符、900 秒和 AS 2 GiB** 全部保留；原 Mini 单文件 2 MiB 也不改变。本准备不证明完整原生依赖、ABI、实际库选择或原映射失败根因，不产生模型资格或提速倍数。

本次已完成三个明文 Packages 全长摘要及对应 Release 条目复核，14 处精确包 stanza 重查，八份 dpkg list 核对，12 个规范目标 lstat 核对，以及 13 份复用源码摘要检查。没有执行现有 runner、decompressor 或任何下载函数。具体验证、来源引用和函数行号见 `minicpm5-sdk310-archive-reuse-source-preparation-20261112-v1.json`。
