# Dory M1 最小后继：三个 ELF 对象与 27 项内容候选

已准备确切输入表、纯 ELF 元数据解析器和窄采集入口；**21 项纯检查通过。本轮没有读取本地库、启动实验、查询当前进程/锁/历史或发布。**

未来获单独准入后，新采集只读以下三个文件的数据并验证完整 SHA256，共 **2,422,457 字节**，总上限 **4 MiB**：

| 对象 | 字节数 | 来源 |
|---|---:|---|
| torch/lib/libgomp.so.1 | 2,115,417 | 已认证 wheel 成员与 native-files / site-provenance |
| torch/lib/libtorch_global_deps.so | 8,264 | 同上 |
| 系统 libgomp.so.1.0.0 | 298,776 | 已完成的签名 deb 成员比对；仍是新内容观察 |

完整路径、预期哈希及全部 27 项内容来源保存在[固定输入表](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-loader-candidate-elf-inputs-20261111-v1.json)。另外 24 个对象的 ELF 信息直接复用认证的保存 JSON，不重复读取其约 882 MB 的本地库载荷。

## 实现与复用

- [纯解析器与候选递归](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory_loader_candidate_elf_v1.py)：只解析 ELF64 / little-endian / x86-64 程序头及动态表；保存 NEEDED、SONAME、RPATH、RUNPATH、INTERP 和 flags，额外 audit/filter 等指令保留为未覆盖。错误、截断、越界或歧义格式直接失败。
- [窄采集入口](/home/ubuntu/experiments/trn1-latest-models-20261031/research/collect_dory_loader_candidate_elf_v1.py)：从原源码采集入口继承 retained lock、完整源码校验、当前历史/物理条件、退出清理结构。guard 与 current_handoff 的 AST 保持一致；没有子进程，沿用原空子进程集合的 release-after-drain。目标模型/包不会被导入。
- [纯检查](/home/ubuntu/experiments/trn1-latest-models-20261031/research/check_dory_loader_candidate_elf_v1.py)：21 项通过，包含合成 ELF 错误输入、双候选/单候选都不变成已选库、递归循环、表外候选、未知搜索路径、输入表完整性，以及原生命周期片段一致性。

新入口没有调用 readelf 或 ldconfig，避免为新工具可执行文件引入额外内容依据。它复用已有 `finish_v4.file_bytes`，先以独立来源确认三个预期内容引用，再完整哈希读取，最后在内存中提取元数据。无需任何运行时绑定预先通过。

## 能推进到哪里

递归只在固定 27 个内容候选内进行。保存的静态候选、SONAME 相同及 ORIGIN 字面路径可以关联可能的内容来源；这些不证明搜索可见性、实际加载或提供者选择。单候选也保持 `actual_selected_path=null`。表外名称或路径只记为未覆盖，不自动 stat、扫描或读取。

当前[仅保存记录的预览](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-loader-candidate-elf-saved-preview-20261111-v1.json)有 **105 条声明依赖/解释器边、39 项未覆盖**，其中包括 3 个尚未采集 ELF 信息的对象。原 **15 条运行时绑定缺口全部保留**。未来即使 27 项元数据齐全，也不能直接声称整个原 11 项 import 闭包或 pre-exec 原生执行依据已经完成。

## 有限资源及后续接入

建议该元数据采集沿用原 CPU 5 / nice 19、根盘 28 GiB / NVMe 30 GiB 底线；地址空间 2 GiB、CPU 120 秒、墙钟 180 秒、输出 16 MiB。新原生输入限定上述三项，保存 JSON 单项 16 MiB、累计 32 MiB；实际历史输入若超限，应另行审阅有限修订，不能自动扩大。

父任务后续可使用 `dory_loader_candidate_ELF_plan_v1` / `dory_loader_candidate_ELF_review_v1`，绑定固定 input_table、content_scope、内容来源接受、完整源码、最新真实串行历史和物理证据。本轮没有生成当前 plan、review、catalog、handoff 或 admission，也未触碰正在运行的 MiniPV。

M1 仍保留表外依赖、其他本地扩展和动态 dlopen、当前搜索配置/hwcaps/链接以及 CPython 启动路径等未覆盖项。M2 的确切 glibc 请求到返回对象语义仍无可引用证据；已有保存元数据不等于该源码语义，本轮未下载或实现猜测式 provider mapper。

保持 **2 loops / 总计 56 个独立 histories / TP1**，后续仍需冷缓存、长解码和请求槽位复用检查。完整哈希、准入接口与未覆盖范围见[交付记录](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-loader-candidate-elf-source-delivery-20261111-v1.json)。
