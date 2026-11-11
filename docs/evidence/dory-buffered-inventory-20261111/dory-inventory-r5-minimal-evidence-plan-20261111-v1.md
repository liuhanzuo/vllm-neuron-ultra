# r5 清单完成后的 30 项补证方案

本次只分类已保存的小型元数据和已冻结源码。r5 的清单与退出审计成功，
48,986 个文件的清单生成完成；`complete_authenticated_inventory=false`、
`native_ABI_qualified=false` 仍然成立。30 项均未在本次关闭。

| 类别 | 数量 | 已有证据 | 仍缺什么 |
| --- | ---: | --- | --- |
| 系统库来源 | 11 个文件 | 完整观测哈希，总计 6,064,048 B；部分包名/版本 | 独立发行包与精确成员内容绑定 |
| vLLM 加载来源 | 13 条依赖边、5 个扩展 | DT_NEEDED；3 个 Torch 目标已有认证清单引用 | 原始导入顺序下实际采用哪个对象 |
| libgomp 多候选 | 2 条依赖边、2 个候选文件 | 原始 readelf 输出保留 RUNPATH=$ORIGIN | 候选内容来源与真实选择；不能按候选顺序选 |
| CPython 缓存 | 4 个文件 | 每个只读了 16 B 头部，时间/长度匹配 | 完整缓存内容与认证源码编译结果一致 |

分类 JSON 保存每一项的准确路径、原始引用与待补条件；不会复制 native-files
大清单。它由纯内存 `classify_dory_inventory_r5_gaps_v1.py` 生成，输入是经过
实际退出审计哈希认证的 result/external-closure 与既有研究元数据。

## 已有证据能说明什么

系统库中，仅 tcmalloc、libstdc++、libnuma 的 canonical 路径包归属查询成功。
其余 8 个查询返回 1，包括 loader、6 个 libc 相关库和 libgcc。不能把这个
结果当成包不存在，也不能直接把 `/usr/lib` 改写为 `/lib` 就认定归属。
原 census 保留的版本只有：libc6 `2.35-0ubuntu3.15`、libgcc-s1/libstdc++6
`12.3.0-1ubuntu1~22.04.3`、libtcmalloc-minimal4 `2.9.1-0ubuntu3`，均 amd64。
libnuma1 的版本缺失。若最终可能选系统 libgomp，其包版本和内容证据也缺失。

此前 tcmalloc 的 98,202 B 安装包缓存已回收；保存的哈希
`f0953cb215ed943429951920f5b88b4c29a6ef0503207ddc4c724772538f06af`
仅是当时本地包的观测，不是独立发行方认证，也不能直接作为 source_evidence。
既有 wheel 认证覆盖 Torch、vLLM、Intel OpenMP，不覆盖这些 Ubuntu 系统库。

13 条缺候选分布为 `_C.abi3.so`、`_C_AVX2.abi3.so`、`_C_AVX512.abi3.so`
各 3 条，`fs_io_C.abi3.so`、`spinloop.abi3.so` 各 2 条。其 readelf 原文均没有
RPATH/RUNPATH。`libc10.so`、`libtorch_cpu.so`、`libtorch.so` 已在同一份
external-closure 中有 `authenticated_SITE` 引用；这排除了“清单中没有该库”
的解释，但未证明 loader 会将这些引用用于相应依赖边。

现有 resolver 的 MODULES 顺序先导入 Torch，再导入 vLLM；这提供有限的导入
顺序依据，但没有读取本次安装的 Torch 初始化源码，也未证明其 dlopen 参数、
对象可见性、命名空间或 vLLM 三种 ISA 扩展的选择。不能以“先导入 Torch”
直接关闭 13 条问题，也不能强制加载所有 ISA 版本来制造成功。

两个 libgomp 使用者是 Torch 的 libshm 和 libtorch_cpu。原始 readelf 明确是
RUNPATH=$ORIGIN；当前 `parse_elf` 把 RPATH 与 RUNPATH 合并到一个字段，
`dependency_candidates` 又把直接候选与 loader cache 合并排序。这个保守清单
算法没有表达真实优先级，不是“系统 loader 同时选择了两个库”的证据。
两个 libgomp 候选本身都没有被加入这 24 个 external 文件行；本次不能虚构
它们的内容引用。

四个 pyc 总计 84,155 B，四份源码总计 73,822 B。三个 encodings 缓存曾在
CPython archive 比较中各比上游大 43 B；这只是差异线索，尚未证明是文件名
重写。`_sysconfigdata` 的源码变换已有独立确认，但该源码确认不认证其 pyc。
`-B`/PYTHONDONTWRITEBYTECODE 防止写缓存，不阻止读取已有缓存。resolver
的 audit hook 在解释器启动之后才启用，无法保护此前的 encodings 启动读取。

## 最小下一步：先补独立内容，再证明加载选择

1. **一次有限包元数据补查。** 使用原串行锁/资源/历史协议，只查询这 11 个
   既知系统路径的 canonical 和明确别名，以及已发现的系统 libgomp 候选。
   记录两端路径/链接关系、dpkg 文件清单归属、确切包版本/架构。候选包最多
   6 个：上述 4 包、已观察到归属的 libnuma1，以及待证实的 libgomp 包。
   这一步不执行安装、升级、ldd 或目标解释器。任何新路径都先返回给 root
   审核，不能自动扩张扫描。

2. **绑定最多 6 个精确发行包。** 优先复用已独立认证的本地发行元数据；当前
   已读材料没有足够认证链。若之后需要联网，只获取确切版本的发行索引和包，
   不换成“最新版本”。必须保留签名发行元数据 → Packages 哈希 → 指定 .deb
   SHA256 → 对应 data 成员 → 安装文件字节的链条，并验证架构和路径映射。
   包压缩/解压大小应先由 metadata 确定并让 root 绑定具体预算，不能在此猜测
   数值。只比较这 11 个系统成员和必要的第 12 个系统 libgomp；仍执行完整包
   摘要和安全归档成员解析，禁止安装。现有精确双硬链接系统库读取策略保留，
   不把单链接检查整体放宽。完成后生成现有 `external_authorities` 接口要求的
   小型 reference/source_evidence，而不是重跑全部 wheel 清单。

3. **有限缓存内容验证。** 完整读取这 4 个 pyc 与 4 个已固定源码，逐个上限
   128 KiB、总输入上限 1 MiB 即足够覆盖已观测大小；变化则拒绝并重新审核。
   保留原缓存，不删缓存、不改目标解释器参数。必须用独立认证、同版本同构建
   CPython 编译器在隔离的验证进程中生成预期缓存，显式固定优化级别、源文件名、
   header 时间/长度和 marshal 格式；只比较字节，不执行可疑缓存 code object。
   验证进程的启动不能先消费这 4 个待认证缓存：在系统库已认证后使用经过审核
   的独立启动缓存路径/隔离副本；启动所需额外源码仍需来自既有 CPython 认证
   证据。该隔离只用于补证工具，不能悄然改变最终实验环境。若不能构造无循环
   信任的编译环境，4 项保持未解决。

4. **复用 wheel 内容证据，补确切加载路线。** 下一次授权读取只抽取现有
   Torch/vLLM wheel 中有限的初始化与扩展选择源码，连同已认证 archive/member
   引用：首先 Torch `__init__.py` 和 vLLM `__init__.py`、`_custom_ops.py`、CPU
   platform 选择入口；若源码指向另一个入口，先列出名字交 root 审核。对 Torch
   libgomp 从已保存 wheel 比较记录提取精确 member 证据，不能仅用文件名推断。
   保存 RPATH/RUNPATH 标签、SONAME、NEEDED 的不同语义；对 13+2 条边建立有限
   判定表，考虑原始预加载库、既有对象与导入顺序、实际 CPU 扩展选择。原始
   LD_PRELOAD 中的 Intel libiomp5 不能被忽略，也不能视为 libgomp 的替代。

5. **最后才做一次经过独立准入的加载确认。** 前述独立来源/启动缓存缺口关闭后，
   root 用既有 ABI probe/新进程协议绑定最新退出和完整历史，只加载原始路线
   实际选择的扩展，不构建模型、不跑推理。必须记录实际 loader 对象与 13+2 条
   依赖的绑定，并对加载对象回连已认证内容引用。`/proc/maps` 单独只能说明
   映射存在，不能在多候选场景证明具体依赖绑定；只读 loader tracing/source
   解释属于诊断证据，若改变环境还需原始环境下的确认。意外路径、缺证据、
   未采用的 ISA 分支都明确保持未资格化，不能拿候选路径冒充已选路径。

不需要为了这 30 项重做 48,986 文件全清单或全 wheel 下载，也不需要新增
一套诊断队列。以上是有限设计，未生成可执行准入、未读取安装 payload、未启动
探针。原 r4 CPU 失败仍为失败，r5 清单完成也不等于 ABI/模型资格或性能提升。
