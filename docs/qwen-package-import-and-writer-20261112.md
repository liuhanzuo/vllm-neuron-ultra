# Qwen3.5 独立包导入修复与缓存写入候选

Qwen3.5 独立包第三轮在准备阶段退出 1，尚未创建模型目录或启动编译。第四轮已通过模型准备并生成全部 12 张 HLO，随后因运行器拒绝编译目录的符号链接而退出 1；完整模型验收仍待完成，没有新增整模型提速倍数。[第四轮实际结果与独立退出检查](qwen-package-compiler-link-failure-20261112.md)保留完整失败证据。

## 导入失败与修复

准备入口报错 `ModuleNotFoundError: No module named 'vllm_neuron.models'`。运行环境的 `PYTHONPATH` 只有研究目录和实验根目录，已有 editable 安装把 `vllm_neuron` 指向旧 checkout；那个目录没有本次新增的 `models` 包。

使用真实解释器、阻止原生库加载的 A/B 检查确认：原环境解析到旧目录；把优化 checkout 放在 `PYTHONPATH` 首位后，可以正确导入 `prepare_view`。检查没有调用 prepare、加载权重或执行设备任务。修复只调整这条环境路径及运行器版本引用，保留模型实现、权重、精度和全部验收要求。环境路径 9 项检查、准备程序 5 项针对性检查由主任务独立复跑通过；作者另外执行了原生命周期 32 项及准备程序全套 45 项检查。

这次问题属于我们的实验环境配置，不能据此认定 Qwen 模型或上游 Neuron 编译器存在故障。第三轮实际退出后的独立检查确认全部 22 个声明服务均已排空、记录的进程已结束、原串行锁可重新取得并释放。已完成模型阶段数为 0；原失败证据保留。

第四轮使用修复后的运行器，保留 48,518 个源码条目、全部 161 条历史运行记录和原 874 项子进程记录。目标仍是独立包的 25 阶段验证，包括数值对照、六个冷缓存桶、1,541 步长解码、请求槽位复用、203 个真实向量、采样重放、quality96 和匹配 HTTP。

按不扩容约束，已将可重建的两份 apt 二进制缓存完整归档并校验后移除，另回收 143,208,448 字节。连同此前已归档的临时测试目录和旧日志，共回收 244,674,560 字节（约 233.3 MiB）。包状态、索引、SDK、权重和已有实验数据均保留。

## 缓存写入候选已推送

独立分支 [`trn1-qwen35-decode-writer-experimental-20261112`](https://github.com/liuhanzuo/vllm-neuron-ultra/tree/trn1-qwen35-decode-writer-experimental-20261112)，提交 `e44408483a6880baa775d236d393812dedb2a443`，已核对远端提交和全部 15 份证据。

候选把 NKI 条件复制接到 decode 阶段的卷积状态、FP32 recurrent 状态及 Attention K/V 写入。默认仍为原 writer，显式开关才启用候选；prefill 和 CPU 参考使用原实现。编译产物身份包含 writer 模式、阶段、形状及源码哈希，防止错误复用另一模式的图。

20 项源码/mock 检查由作者与主任务分别执行通过。该分支尚未进行设备编译或整模型验收，真实 int32 槽位、KV 切片布局与别名、TP2 多状态图仍需验证。未来两臂固定相同 sampler，并分别使用独立 artifact、编译缓存和工作目录，才能衡量 writer 的增量收益。详见[候选说明](https://github.com/liuhanzuo/vllm-neuron-ultra/blob/e44408483a6880baa775d236d393812dedb2a443/docs/qwen35-decode-writer-candidate-20261112.md)。

## 已测倍数的适用范围

[全部模型提速总表](model-speedups-20261112.md)保留 LFM3、SmolLM3、Qwen3.5 的配对结果，以及早期四模型 INT8 和主机采样测量。Qwen3.5 研究适配器 r10 的随机采样为单请求 1.208×、C4 1.840×，混合长度 1.680×；贪心为 0.986–0.998×。独立包还须完成本轮验证，才能报告它自己的结果。

缓存算子矩阵的 1.671×～3.965× 比较的是 NKI 与整数 bitcast 诊断参考，仅是局部算子计时。它不是相对模型原 writer 的整模型倍数，也不能与 sampler 收益相乘。

[失败记录、导入诊断、修复源码和新一轮配置证据](evidence/qwen-package-import-20261112/manifest.json)。
