# Dory M2：已定位加载器版本，实际请求到返回库的语义仍待补证

本轮从 10 份摘要已核对的保存元数据，确认原加载器对应 **Ubuntu libc6 2.35-0ubuntu3.15 amd64**。其 `/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2` 为 240,936 字节，SHA256 为 `41daf7836f7fce21a40ca63a00c5caf21b05a35a76ccc075d86f1716212cdf01`。原记录与后来的包成员比对均匹配这一内容，包归档的摘要来自已验证的 Ubuntu 索引。

这确认了二进制的来源。现有 `readelf -lW -dW` 保存文本只有程序头和动态表；出现 `.note.gnu.build-id` 节名称不等于记录了实际 Build ID 值。已检查的元数据没有精确 glibc 源码、Ubuntu 补丁、私有符号地址或匹配调试信息。对 `/usr/src`、`/home/ubuntu/src` 和实验 research 的有限名称／C 源码符号检索未找到实现；这是有界负面搜索，不是整个文件系统不存在源码的证明。

**本轮未读取 native、RECORD、原始张量或归档载荷，未下载、导入目标框架、查询服务／锁／当前运行历史，未启动实验。15 条依赖边全部保持未解决。**

## 为什么单次 11 项导入不能自动关闭全部 15 条边

| 请求者 | 未解决的 DT_NEEDED 数量 |
|---|---:|
| `vllm/_C.abi3.so` | 3 |
| `vllm/_C_AVX2.abi3.so` | 3 |
| `vllm/_C_AVX512.abi3.so` | 3 |
| `vllm/fs_io_C.abi3.so` | 2 |
| `vllm/spinloop.abi3.so` | 2 |
| `torch/lib/libshm.so` | 1 |
| `torch/lib/libtorch_cpu.so` | 1 |

已认证的 Python 源码预测在保存的 CPU 条件下会选择 AVX512，但不是原生导入成功证明。其他 CPU 变体可能没有被触发；fs 注册是惰性的，spinloop 又受开关控制。TP1 本身也不能排除构造器在并行度判断前导入相关模块。原 11 项导入必须保持原序列，未触发的请求者单独标为“未观察”，不能强行多导入几个扩展来制造原范围内的完成结论。

## 下一步需要取得的有限来源

1. 重新取得精确 `libc6 2.35-0ubuntu3.15 amd64` 的完整认证二进制条目或包控制字段，确认 `Source` 对应关系。保存的精简条目只保留了 `pool/main/g/glibc` 路径，不能用目录名代替完整来源字段。
2. 从已识别的 jammy-updates/main 或 jammy-security/main 的认证 Release、Sources 索引中选择这个精确源包版本。先生成带长度和 SHA256 的 DSC／全部源包成员有限表，再审阅归档读取与提取；不存在该版本时保留缺失，不能换用上游通用 2.35 或较新补丁。
3. 审阅精确 Ubuntu 补丁序列和构建设置，追踪加载请求、成功复用／新映射返回、失败、命名空间及卸载路径。`elf/dl-load.c`、`elf/dl-deps.c`、`elf/dl-open.c`、`elf/rtld.c`、`elf/dl-close.c`、`elf/dl-audit.c` 和对应 link-map 定义只是明确的检索目标；本轮未认证这些文件的实际内容或函数 ABI。
4. 若日志没有完整的成功返回记录，再单独取得 loader Build ID 和匹配的调试／地址依据，审阅直接观测方案。包版本字符串不足以证明某个私有函数偏移。新增工具本身、观测开销和内容依据也要绑定；本轮没有选择或启动探针。

源包的实际大小尚无认证输入，因此本设计没有填写猜测摘要或自动放大预算。后续先完成 metadata-only 的精确文件表，再绑定有限载荷、文件数和存储预算。

## 什么才算一条成功关系

需要记录同一进程及线程／嵌套请求下，**具体 requester 实例、namespace、DT_NEEDED 项、成功返回对象实例以及该对象独立认可的内容身份**。对象实例还必须有生命周期代数，防止卸载重载后复用地址或命名空间编号。

已加载对象的 SONAME 或 inode 复用可能没有新的搜索／打开事件，仍必须能记录真实返回关系。顶层 `dlopen` 的调用者和递归 DT_NEEDED 的请求者不能混用；符号 interposition 的提供者也不能代替 DT_NEEDED 对象。解析返回成功与整个 `dlopen`／导入／重定位成功须分别记录，防止映射后回滚被当成 ABI 成功。

优先审阅精确 `LD_DEBUG` 输出点是否真的覆盖这些条件。如果没有，就保留原始日志作为有限证据，另行审阅在源码确认的依赖解析返回位置进行直接观测。外部观测与 `LD_AUDIT` 都有额外前置条件；后者还会改变目标环境并加载审计库。本轮没有实现“按日志相邻顺序猜库”的解析器。

验证范围必须包含：首次加载、已加载对象复用、同 inode 别名、跨 namespace 同 SONAME、嵌套／并发 dlopen、失败候选、映射后版本或重定位失败、符号插入、卸载重载、未触发分支、丢失／截断记录，以及 Python hook 安装前的启动和 preload。任何不完整事件都保留“不足以证明”，不能靠后来的 maps 或单候选补全。

该步骤不替代 M1 启动／候选内容闭包，也不改变原串行运行机制。Dory 仍保持 **2 loops、共 56 个独立 attention histories、TP1**；原生 ABI、完整模型、冷缓存、长解码和请求槽位复用仍待实际通过。完整输入、15 条原边、来源取得顺序与 13 类后续验证场景见 [JSON 审查记录](/home/ubuntu/experiments/trn1-latest-models-20261031/research/dory-loader-M2-exact-source-prerequisite-review-20261111-v1.json)。
