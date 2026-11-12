# Package serial runtime v4 — exact inherited host envelope

Root 报告实际 v3 准入前检查在 `Bundle.resolve` 的 4 MiB host-source 读取处失败：唯一大文件为 `research/qwen35-sampler-r10-20261112-spec-v12.json`，8,804,235 字节，SHA256 `86a4c46dafcd3c16c74101caef26604af8e4b4041bcc3c6406faeab60d8d433f`。报告的真实工具 session 为 14595、terminal chunk `05496f`、exit 1。本作者没有重跑该 History、实际准备或准入检查。

v4 只为这个精确路径、SHA 和字节数增加 `io.host_source_bytes` 读取分支，使用原 `LIMITS['spec_envelope_bytes']` 64 MiB。传入 SHA 必须先匹配冻结常量；原 canonical/singleton identity 必须显示精确大小，之后才调用原 `raw` 读取完整内容。原 `O_NOFOLLOW`、open/fstat、完整读取及读前后 identity 校验保留；helper 再检查返回长度和最终 identity，并对全部字节核对 SHA256。不读取前缀、不截断、不跳过历史内容、不投影来源集合。

仅三个完整 host-source 入口改为这个 helper：`history.py:75` 的 `Bundle.resolve`、`adapter.py:149` 的 `worker_bootstrap`、`final_audit.py:106` 的独立终审。三处外层全 SHA 比较也保留。普通 host source 仍是 4 MiB；普通 metadata 的 `raw/read/ref` 默认仍是 8 MiB。`loaded_local_sources` 的真实 Python 文件和 Torch namespace 读取仍严格保持原 4 MiB，未扩大。旧文件的路径别名、其他大 JSON 或相同名字的其他路径均不会获得 envelope 例外。

生产差异为：新增一个 IO helper/常量、上述三处单调用替换及对应 import、`run.py` 的 v4 namespace。AST 控制将三处调用反向替换后验证整份 `history.py`、`adapter.py`、`final_audit.py` 与 v3 完整相同；旧 `io.py` 全部函数/类 AST 保持原样。History 解析、source 分类、完整 scope、运行 ACK、wait/drain、锁、科学阶段、v3 存储竞态处理、所有资源策略均未改动。

SOURCE/STDLIB MOCK 验证：

- `test_host_envelope.py` 15 项，实际 chunk `d24222` / exit 0：完整 8.8 MB 合成字节 stable read、精确 SHA/size/path 边界、同大小内容变化、截断、读取中及返回后 identity 变化、symlink/hardlink 拒绝、普通 4 MiB/8 MiB 上限、三处完整 AST 替换证明、其余生产字节保留。
- `test_source.py LifecyclePreservation FullSequenceMock` 17 项，实际 chunk `cc998e` / exit 0：原服务、锁、History authority/running ACK、wait/drain、完整阶段顺序和失败门禁。
- 首次 envelope 测试 chunk `8c4db8` / exit 1 的唯一问题是测试预期的硬链接错误文案过窄；原代码实际已正确拒绝，调整断言后重跑 15 项通过。生产代码未为此变化，初始失败日志保留。
- `ast_preservation.json` 和 `predecessor_pins.json` 记录 v1/v2/v3 共 64 个冻结文件未改，核对 chunk `c680e9` / exit 0；8 个生产/契约文件与 v3 字节相同。

所有测试只读源码与写入/读取临时合成字节，没有导入 torch、Neuron、vLLM、Transformers 或 NumPy。没有读取真实模型/tensor/NEFF payload，也没有运行实际 prepare、History、服务、编译、推理或锁。本交付 `launch_authorized=false`、`runtime_admitted=false`，需 root 对冻结交付独立接受并另行绑定；不能把这次 source-only 修复写成 package 已完成或新提速结果。

v3 的预算与路径限制仍成立：有限 trial 和持续 floors/reservation 监控不是硬配额，可选 layout compiler 继承 ROOT cwd 的额外相对写入仍未证实覆盖，不新增编译峰值承诺，不删除共享缓存。
