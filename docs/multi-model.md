最新运行：[Qwen3.5 主机配置记录修复与 r10 启动](qwen-host-receipts-r10-20261112.md)。主任务独立复跑17项运行接线检查、9项终审模拟检查通过；实际会话38957已完成运行前检查和握手，完整模型与HTTP验证继续，没有新增合格提速。

最新准备：[MiniCPM5 正式入口修复](minicpm5-formal-entry-20261112.md)。82 项独立源码／模拟检查与 3 项来源负控通过，修复导入路径、标准库来源接线和完整清单读取上限；原 9 项原生检查及 1,284 行科学诊断仍待串行执行。

最新进展：[Qwen3.5 采样实测与 MiniCPM5 双解释器启动结果](qwen-r9-results-mini-bootstrap-20261112.md)。Qwen 随机采样测得单请求 1.23×、C4 1.85×，混合长度 1.68×；完整数据已复核，但整轮因自有汇总问题退出 1，且主机配置回执缺项，暂不计入合格结果。MiniCPM5 两路启动观察实际退出 0、六份映射复核通过，正式模型验证待继续。

最新进展：[LFM3 新版插件发布与 MiniCPM5 环境准备](lfm-sdk053-plugin-and-mini-origins-20261112.md)。独立 0.5.3 分支已推送，25 项源码/mock 测试和 7 步完整 runner 生命周期模拟通过；设备与整模型仍待验证。Mini 八包核验后继已完成，24 项独立 mock 检查与实际文本来源核对通过，真实下载仍待串行绑定。LFM 设备阶段的空间预算暂未满足，保留原门槛，先推进较小输出的 Mini/Qwen 工作。LFM CPU r2 扩展测试继续，没有新增合格提速。

Qwen3.5-2B 的独立实验包源码已接入；入口、固定配置、源码检查和待完成的设备验收见 [Qwen3.5 包接入](qwen35-package-20261112.md)。该新包尚无合格性能结果。

此前进展：[LFM3 的 4096 验证补充与新栈接入](lfm4096-and-spark-preflight-20261112.md)。追加实际 LFM KV 形状后，完整设备方案为 105 个任务／127 张图／4,304 项检查，16 项独立源码检查通过；独立 0.5.3 分支中的模型插件迁移已开始，尚未取得模型资格。Spark 原生预检44项独立检查通过，未实际读取原生载荷；15处 OpenCV 冲突文件大小与选用版本一致，内容及加载关系仍待核验。LFM CPU 扩展回归继续，没有新增合格提速。

此前进展：[vLLM-Neuron 采样导入修复与三路运行准备](sampler-and-runtime-reviews-20261112.md)。LFM 候选首阶段完整 123 批／1,648 例已通过原验证器、输入不变与实际退出的独立审计，2,068 例扩展阶段继续。0.5.3 特定采样开关的模块调用错误已复现，单行补丁与六项回归测试已保存。LFM 设备执行代码 51 项、Mini 计时修复与完整入口 77 项独立检查通过；Spark 依赖清单重建与 31 项测试通过，记录 OpenCV 安装声明冲突。尚无新增合格提速。

此前进展：[LFM 新环境回归、vLLM 源码更正与诊断入口](editable-source-and-runtime-entry-20261112.md)。新环境完整 291 批／3,716 案例与原环境数据完全一致，实际退出与原验证器均通过；三路控制独立复核通过，保留原模拟器问题的复现，现进入候选全量测试。更正此前“新环境无 vLLM 源码”的判断：已有完整 editable 源码，静态依赖未见问题，真实导入与服务接入待验证。Mini 无旧 pyc 入口 35 项、Spark 串行控制器 26 项独立模拟检查通过，实际运行待接入。没有新增合格提速。

此前进展：[LFM 全套基准数据完成与三路原生接入](lfm-full-fixture-and-device-integration-20261112.md)。LFM 原环境 291 批／3,716 案例、全部真实零退出及进程结束已独立核验；r2 继续新环境与候选对照。真实 JIT／完整设备源码 47 项、Mini 描述符与认证器 45 项、Spark provider／worker 26 项独立源码／模拟检查通过，实际设备和新运行绑定仍待完成。确认 Mini 原环境为 CUDA wheel；映射失败根因尚未证实。没有新增提速。

最新进展：[LFM 首组 CPU 回归完成与后续设备准备](lfm-regression-and-native-preparation-20261112.md)。LFM r2 原环境首组 123 批／1,648 案例及真实零退出已独立核验，扩展阶段继续运行；完整 CPU、设备和整模型门槛仍待完成。Qwen 后继准备工具 30 项、Mini 历史入口 24 项、Spark 排序与排列诊断 13 项独立源码／模拟检查通过，均未取得新运行资格。context4096 目标继续保留为未完成项。没有新增提速。

最新进展：[Qwen 配对模型检查、LFM 重跑与诊断工具修正](qwen-r6-model-checks-lfm-r2-20261112.md)。Qwen r6 两侧通过原数值、冷缓存、长解码及各 203 个向量对照，随后 API 启动失败，整轮实际退出 1；根异常日志被自有监督代码提前截断，底层原因仍未知。后继日志收集通过 128 项独立检查。LFM r2 已通过实际准入并确认 worker 运行，完整 CPU 对照进行中。Mini 超时交接修正通过 63 项模拟检查，Spark 原生诊断通过 13 项源码检查；实际原生接入仍待完成。本轮没有新增合格提速。

[各模型提速总表](model-speedups-20261112.md)集中列出已测 BF16、历史 INT8 与主机开销优化，并标明基线、工作负载和验证范围。

前一版：[LFM 完整测试数据、Mini 映射诊断与 Dory 接入](lfm-fixtures-dory-mapping-20261112.md)；[MiniCPM5 SDK 导入实测与 Qwen 监控修正](mini-sdk-import-qwen-monitor-20261112.md)。

此前记录：[Qwen 冷缓存修复与 Dory 依赖检查实测](qwen-cold-fix-dory-elf-20261112.md)；[Qwen 原生预检与 MiniCPM5 固定输入对照](qwen-native-preflight-and-mini-pv-20261111.md)。

# Unified Trn1 experimental branch

[MiniCPM5 保存算子报告全量独立复核通过，LFM3 显式控制后继已启动](mini-independent-audit-lfm-selector-20261111.md)：1,284 份采集数据重新读取，六份报告和汇总精确重现；47 组相同 P/V 输入仍有 context 差异，固定输入重放待执行。LFM3 r4 三项通过后遇到槽位类型错误，r5 修正通过 12 项独立检查并已进入串行运行。没有新增模型提速。

[MiniCPM5 六组保存算子比较实际完成，退出后独立复核待执行](mini-comparison-terminal-20261111.md)：完整 1,284 份 capture 已处理，六份报告及汇总已写出、工具实际退出 0；LFM3 检查器修正后重跑已启动。没有新增模型提速。

[MiniCPM5 完整原生审计通过，LFM3 首例 NaN 位模式变化复现](mini-native-audit-lfm-writeback-20261111.md)：936 份原始文件及全部日志核验完成；逐算子比较入口 11 项独立检查通过并已启动真实比较。LFM3 候选组在运行前检查失败，三组对照未完成；没有新增模型提速。

[MiniCPM5 原生 r3 实际退出 0，Qwen3.5 运行接口 33 项独立测试通过](mini-r3-terminal-qwen-runtime-20261111.md)：24 个检查点／72 次局部调用完成，正在执行独立结束记录，完整原始数据审计仍待完成；保存数据读取工具 9 项独立测试通过。没有新增模型提速。

最新审阅：[MiniCPM5 保存算子比较与 Qwen3.5 采样补充](mini-comparison-qwen-review-20261111.md)。比较工具 9 项、后续准备工具 29 项独立检查通过；说明采集布局／启动结论边界，并将采样控制数量更正为原清单的 13 项。真实比较待原生退出和审计，无新增提速。

最新准备记录：[Qwen3.5-2B 采样优化与 MiniCPM5 后续核验修正](qwen35-sampler-preparation-20261111.md)。源码检查已通过，实机结果与提速仍待验收。

[2026-11-11 MiniCPM5 修正后重跑准入与 LFM3 写回对照准备](mini-r3-and-lfm-writeback-preparation-20261111.md)：新 MiniCPM5 任务已通过实际准入；LFM3 三组对照通过 17 项独立源码检查，Dory 有限补充采集通过 34 项检查，后两项实际运行待串行槽位。没有新增模型提速。

[2026-11-11 LFM3 输入写回差异与 Dory 加载路径分析](lfm-input-writeback-and-dory-loader-20261111.md)：LFM3 真实模拟器首例返回的 4,096 字节完全一致，但调用后主机输入改变，因此退出 1；剩余 31 例未执行。Dory 完成五份源码认证和 15 项条件路径映射，实际库选择与 ABI 仍待验证。本轮没有新增推理提速。

[MiniCPM5 原生依赖清单与重复校验修正：28 项独立检查通过，真实运行待完成](minicpm5-native-source-fix-20261111.md)

[2026-11-11 MiniCPM5 收尾失败与 LFM3、DeepSeek 诊断接入](model-diagnostics-20261111.md)：MiniCPM5 写出 24 个检查点的 72 份结果后，因已加载源码漏项退出 1，原进程已排空，严格数据审核与比较尚未通过。LFM3 串行诊断入口及 DeepSeek 独立观测、原始数据一致性检查已完成源码验证；实际运行仍待执行，没有新增推理提速。

[2026-11-11 小模型候选复查](recent-small-models-20261111.md)：19 个官方发布者、293 个仓库的有限复查完成。保留全部 15 个模型家族；新增观察 MiniCPM-OCR 1.0 的 OCR 专项与低热度 ZoomGain 视觉变体，区分独立生成模型、草稿头、评分模型和机器人策略。本轮没有新增本机推理提速。

[2026-11-11 MiniCPM5 完整历史准备成功](minicpm5-complete-history-20261111.md)：修正准备及运行校验的 64 条历史上限，17 项检查通过，全部记录保留。实际准备与配置绑定均退出 0，新运行已通过实际准入并启动协调进程；原生采集、完整数据复核和模型对照仍待完成，没有新增提速。

[2026-11-11 Dory 四个 Python 缓存完整复现](dory-python-caches-20261111.md)：4/4 缓存由已核对源码和解释器完整重新编译得到，独立保存数据复核通过；原 30 项缺口已补齐 15 项内容证据，剩余 15 条动态库加载关系。没有新增模型提速。

[2026-11-11 Dory 系统库内容核对与离线重放完成](dory-package-members-20261111.md)：六个安装包、12 个当前库文件及 11 个历史摘要全部匹配，真实离线重放和外部退出核对完成。修正自有工具的旧 GNU tar 头与两个宿主硬链接兼容问题；加载选择、原生 ABI 和整模型验收仍待完成，没有新增提速。

[2026-11-11 MiniCPM5 原生采集失败与后继修正](minicpm5-native-r1-storage-and-successor-20261111.md)：原生 r1 因根盘跌破原 28 GiB 保留线退出 1，未产生张量；已确认进程清理，独立失败记录正在执行。删除未使用旧工具版本和可再生成缓存释放 1.18 GB。后继避免重复复制完整历史，12 项主任务与 4 项独立测试通过；准备程序切换新版运行器并保留已执行审计来源，10 项测试独立复跑通过。保存数据比较入口的 12 项测试通过，尚未运行。没有新增模型提速。

`trn1-multi-model-experimental-20261107` combines the LFM3, SmolLM3 and
Nemotron Nano 9B v2 packages in one vLLM-Neuron checkout. All three general
plugins are installed, and each remains disabled until explicitly selected.
Their artifact loaders run before generic artifact deletion or recompilation,
so an invalid saved artifact continues to fail instead of being silently rebuilt.

[2026-11-11 MiniCPM5 批大小与布局实测](minicpm5-sdk-batch-layout-20261111.md)：144 次局部 SDK 调用、72 份完整输出及退出后独立复核完成。连续布局不改变输出；B1 仅改变 2/24 个检查点的 233 个 BF16 元素，局部相对官方误差由 0.44636% 变为 0.44650%，未改善。历史服务扫描漏计已修正，实际两遍完整核对 1,316 项；此前锁检查异常根因仍未知，失败记录保留。MiniCPM5 内部边界仍为源码准备；Dory 依赖清点已完成并独立复核 79,789 个普通文件，下一步补齐 Torch／解释器来源和真实 ABI 验证。没有新增合格模型提速。此前 [Nemotron 失败释放与存储恢复](recovery-20261111.md)保留。

[MiniCPM5 完整来源实测与审核修正](minicpm5-source-verification-20261110.md)：44,007 条真实来源记录完整保留，文本认证 71.38 秒、峰值约 304 MiB；没有读取模型或历史张量。独立审阅发现历史退出证据遗漏，后继严格审核与记录程序通过 99 项针对性检查，实际准入仍待完成。Nemotron 已独立核实 37/52 阶段，完整运行继续，无新增模型提速。

[MiniCPM5 新版运行器与独立审核程序](minicpm5-runtime-v2-20261110.md)：完整来源分类、worker/runner v2、退出后完整输出复核已接入源码并通过针对性检查；正式配置和实际准入仍待 Nemotron 终止。已独立核实 Nemotron 27/52 阶段，无新增模型提速。

[MiniCPM5 后继运行接入及审查发现](minicpm5-runtime-lineage-20261110.md)：新增结束记录、运行器与进程检查通过；全量历史来源的分类接入仍待修正，未启动新实验。Dory 完成固定 Torch 接口的静态审阅；Nemotron 已独立核实 16/52 阶段，完整运行继续，无新增提速。

[MiniCPM5 当前输入×历史的 180 次完整调用已完成](minicpm5-current-history-20261110.md)：原对照同时使用官方当前归一化输入和历史 K/V，注意力输出汇总相对 L2 从 1.2720% 降至 0.4464%，24 帧全部改善，但仍无一帧完全复现官方输出。108 次上一轮真实输出对照、双 rank 一致性和退出后独立复核通过；40 项检查通过。三个路径在两项均来自官方时输出完全相同。既有整模型通过率仍为对照 29/32、候选 28/32，未新增提速；这是 CPU 参考局部诊断。此前[仅换官方历史的结果](minicpm5-official-history-20261110.md)完整保留。

[MiniCPM5 下一轮诊断源码接入与 Dory 适配器准备](mini-dory-next-source-20261110.md)：MiniCPM5 四组共 67 项合成检查通过，补齐完整输出复核；Dory 原生张量适配与稳定文件审核已准备。Nemotron 已实际完成前 8/52 阶段，完整运行继续；没有新增模型资格或提速。

[Nemotron r12 已通过实际运行准入，完整 52 阶段验证重新启动](nemotron-r12-20261110.md)：106 项源码与启动检查通过，原科学、资源和性能门槛全部保留；整模型结果仍待运行及独立复核。MiniCPM5 下一步批大小／布局诊断和 Dory 目标层观测均仅完成源码准备。

[Nemotron 存储维护已完成](nemotron-storage-recovery-20261110.md)，六份旧日志完整压缩保留后释放约 768 MiB 根盘空间；整模型重试仍需检查实时空间。[Dory 下一步诊断设计](dory-next-diagnostic-20261110.md)保留全部 87 个检查点与 7 个数值失败，计划区分注意力与残差归一化影响，尚未执行。

The exact `e0a01d3` wheel passed [nine native installation/plugin/spawn/configuration checks](unified-native-checks-20261108.md) on 2026-11-08, after fixing a JSON-input bug in the probe. Fresh device and end-to-end qualification
of the merged branch is pending. Use the recorded source commits for reproducing
the existing measured comparisons; new merged source identities need new
artifacts and a new full comparison.

[2026-11-11 MiniCPM5 启动开销与逐层比较准备](minicpm5-startup-and-comparison-20261111.md)：启动校验存在重复完整计划摘要与目录构建；独立只读快照原型通过 9 项复核，后继运行器源码接入又通过 7 项独立集成复核，尚未实际启动。逐层比较器修复了长度 128 与头维度 128 的歧义，11 项合成测试及 6 项独立控制通过；完整合成报告已无损封装到 627,301 字节，7 项测试及 4 项独立控制通过，保持原 2 MiB 上限；真实数据接入仍待完成。当前原生采集继续，未增加模型提速结论。

## Install and select a model

Use a separate environment matching Neuron SDK 2.27, vLLM 0.11.0 and the exact
dependencies in the selected model's guide. Do not install into a running
benchmark environment.

```bash
git clone --branch trn1-multi-model-experimental-20261107 \
  https://github.com/liuhanzuo/vllm-neuron-ultra.git
cd vllm-neuron-ultra
pip install --no-deps -e .
```

Then follow the model-specific preparation and serving instructions. Choose one
model per server process. If restricting `VLLM_PLUGINS`, include the `neuron`
platform plugin and the appropriate general plugin below.

| Model and guide | General plugin | Model opt-in | Optimized path |
|---|---|---|---|
| [LFM3-1.7B](lfm3.md) | `lfm3` | `NEURON_LFM3=1` | `NEURON_LFM3_NKI_WRITER=1`; paired reference and candidate both use `NEURON_CPU_SAMPLER_FAST=1` |
| [SmolLM3-3B](smollm3.md) | `neuron_smollm3` | `NEURON_SMOLLM3=1` | Combined `NEURON_SMOLLM3_NKI_CACHE=1`; standard sampler required |
| [Nemotron Nano 9B v2](nemotron9b.md) | `nemotron9b` | `NEURON_NEMOTRON9B=1` | Packaged cache path remains unqualified; standard sampler required |

Model views and compiled artifact directories are model- and variant-specific.
Follow each guide's explicit compile opt-in, fresh destination and cached-load
rules. The unified package does not convert old artifacts or enable unsupported
batch sizes, contexts, precision, prefix caching or other serving features.

## Existing results and provenance

These gains use matched reference adapters with the same weights and settings,
not unmodified upstream vLLM-Neuron. C4 means aggregate throughput for four
concurrent requests. These are warm HTTP workload results, not compile times.

| Model / recorded implementation | Short single / C4 | Long single / C4 | Mixed lengths | Status |
|---|---:|---:|---:|---|
| LFM3 `822e1bd826321dc078226c2da3f199e1d385146f` | 2.02× / 1.80× | 8.51× / 6.70× | 5.92× | [Paired full-model gates passed in the stated scope](lfm3-results-20261107.md) |
| SmolLM3 combined `e2d1e28b83411863acef5f12ff3db1f887866a76` | 8.16× / 8.05× | 8.00× / 7.51× | 5.50× | [Qualified at decode floor2048 against cached native reference](smollm3-results-20261105.md); natural-bucket and full-causal-reference failures remain |
| Nemotron `60ab7f54a2c54ef0bc9b498621a45719ee7f0e84` | — | — | — | [Input/environment capture passed](nemotron-input-capture-20261108.md); no matched packaged speedup; historical 13.18 / 51.06 tok/s are absolute rates |

The model package trees are preserved from these source branches:

- LFM3: `lfm3-paired-results-20261107`, commit `fe398c2a93b89f18328d0db7f2a6f655d8a6936b`.
- SmolLM3: `smollm3-ablation-results-20261107`, commit `e3dc962`.
- Nemotron: `nemotron9b-experimental-v1`, commit `60ab7f54a2c54ef0bc9b498621a45719ee7f0e84`.

The repository also preserves the [historical four-model INT8 comparison](dense-int8-results-20261030.md)
for Qwen3-8B, Qwen2.5-7B, DeepSeek-R1-Distill-Qwen-7B and Mistral-7B: C4 gains
of 1.53×, 2.02×, 2.08× and 1.62× respectively. These used device greedy sampling
and an earlier small validation suite. They are not new merged-branch results
or evidence of lossless quantization.

The included SmolLM3 successor adds experimental ablation variants. Its
[write-only candidate failed deterministic output equality](smollm3-ablation-20261107.md),
so its measured speed is not a qualified improvement. The older combined
result does not certify these successor sources. The ongoing observer on/off
diagnostic measures whether instrumentation changes outputs; it supplies no
new speedup result.

## Integration validation and remaining work

On 2026-11-07, all **100 standard-library tests passed**: 33 LFM3, 26 SmolLM3,
37 Nemotron and 4 shared integration tests. A PEP 517 isolated wheel build
succeeded with the declared build dependencies. All 49 model-package files,
including six JSON/license resources, matched the checkout byte for byte in
the wheel, and all three plugin entry points were present. Setuptools warned
about the two license directories' namespace discovery; their license files
were explicitly checked in the built wheel.

The integration checks cover all three installed entry points, disabled-plugin
imports, required package resources, model-specific loader selection and
propagation of artifact rejection before generic recompilation. Existing model
tests exercise tokenizer/config identity, source contracts, registration,
cache/state contracts and artifact handling using small inert fixtures.

```bash
python3 -B -m unittest discover -s test/lfm3 -p 'test_*.py'
python3 -B -m unittest discover -s test/smollm3 -p 'test_*.py'
python3 -B -m unittest discover -s test/nemotron9b -p 'test_*.py'
python3 -B -m unittest discover -s test/multi_model -p 'test_*.py'
```

These checks do not load native model libraries or execute device inference.
The CPU sampler's native tests are a separate serial task, with historical
results in [Trn1 optimizations](trn1-optimizations.md). Native plugin/spawn/configuration checks for the exact wheel are recorded above. Before promoting the
merged branch, complete compile/load and corruption
rejection, original numeric gates, cold cache, long decode, request-slot reuse,
quality and matched serving measurements for each model. No model family is
declared fully optimized.

最新实测进展： [MiniCPM5 第 1 帧完成 15 个张量重放和独立数据复核](minicpm5-frame1-diagnostic-20261108.md)，固定输入的 down 路径差异与实际 MLP 输入差异已分别量化；整模型仍未合格。Nemotron r5 的失败与回收见 [结束记录](nemotron-r5-compile-interruption-20261108.md)；[v14 编译资源链接修复](nemotron-v14-compiler-links-20261108.md)已通过 94 项集成检查和 13 项配置检查，[r6 已因遗漏 DVE 资源链接而在编译阶段失败并释放；Dory 的 192 次位置编码比较及运行后完整数据复核通过](dory-r1-nemotron-r6-terminal-20261108.md)。整模型验收仍未完成。模型提速倍数仍以已合格的配对性能结果为准。

[2026-11-08 小模型候选复查](recent-small-models-20261108.md)记录官方仓库更新、草稿模型排除依据与原精度候选的准备状态，保留全部 15 个模型家族。

最新局部实测：[MiniCPM5 完整有效前缀的注意力输入一致，差异从注意力输出出现；固定 CPU 布局替换后恢复原生残差/归一化，独立字节复核通过](minicpm5-upstream-result-20261108.md)。没有新增整模型候选或提速倍数。

最新运行器进展：[Nemotron v15 核验完整 PWP/DVE 资源，113 项检查通过，已发起 r7 复验](nemotron-v15-compiler-resources-20261108.md)。原模型实现、52 阶段和提速表保持原有结论，尚无新整模型合格结果。

最新诊断准备：[MiniCPM5 注意力内部采集、完整结果复核、串行运行、配置生成、结束核验及退出后独立复核源码通过 144 项检查](minicpm5-attention-source-20261108.md)，当时尚未执行新的模型采集，实际后继结果见下文。Nemotron r7 已在编译阶段因运行器清单遗漏而失败释放，提速表没有新增结果。

[Nemotron r7 投影与卷积拆分对照](nemotron-r7-split-progress-20261108.md)：32 个局部对照阶段全部完成，累计核对 42/52 个工作阶段。16 组候选的 96 次完整设备输出比较全部逐字节一致；原写法 14 组多行输入失败、2 组单行对照通过，CPU 两种写法均正确。随后在整模型编译阶段因 `sgLnk` 链接清单遗漏而中断并失败释放，见[实际终止记录](nemotron-r7-terminal-20261108.md)。报告保留具体地址差异、环境与范围；整轮独立原始数据复核和完整模型验收仍待完成，没有新提速倍数。

[Dory 原权重参考加载准备](dory-metadata-and-optimization-20261108.md)：原加载器已接到完整参考候选，15 项源码/合成存储检查通过；新增存储共享与缓冲寿命探针，22 项源码/标准库检查通过，实际 PyTorch 探针及退出后独立复核现已通过，见[存储结果](dory-storage-result-20261108.md)；18 份完整张量、786,864 字节均核验。保留原 507 个 BF16 物理张量、两次循环与 56 份独立历史。原权重实际加载现已完成，见[加载结果](dory-original-weight-load-20261108.md)；整模型验证仍待完成。位置编码 192 次局部比较的既有结果已同步修正到概览。

[MiniCPM5 注意力 r1 实际失败与修正](minicpm5-attention-r1-failure-20261108.md)：原生加载后，K 缓存边界的元数据相等断言失败；CPU 与重放未启动，23 个部分文件已保留并确认资源释放。后继观测器分别保留生产/消费布局，仍要求全部 K 值一致，并按实际消费布局重放；54 项源码与合成记录检查通过，尚无新一轮合格采集或提速结果。

[MiniCPM5 注意力 r2/r3 实测与独立复核](minicpm5-attention-r2-r3-20261108.md)：r2 确认 K 缓存只改变布局，随后 CPU 被观测器缺失属性检查阻断。修正后的 r3 原生/CPU 采集均完成；原重放因固定计划读取限额错误失败，失败结果保留。独立复核直接读取全部 96 份原始输出、353,284,922 字节，21 项自身输入对照一致；Q 有 1 个元素差异、context 有 28 个、输出投影有 802,431 个。固定同一 context 的投影对照现已[完成](minicpm5-fixed-oproj-20261108.md)；当前没有新增整模型合格结果或提速倍数。

[Dory 存储探针实际通过](dory-storage-result-20261108.md)：六组共享/寿命/复制对照、18 份完整张量及退出后独立字节复核通过，原生与复核进程均实际退出 0。原权重加载已在后续完成；完整模型与 Neuron 基线仍待完成，未产生新提速。

[Dory 原权重加载与参考模型构造通过](dory-original-weight-load-20261108.md)：固定官方约 7.95 GB 文件完整哈希核验，507 个 BF16 张量实际加载并保持共享存储，峰值约 7.68 GiB；退出后逐张量和完整文件独立复核通过。首次运行器进程检查冲突已保留并修正。尚未执行模型前向或生成，未产生新的原生推理/Neuron 合格基线或提速倍数。

[Dory 官方分词与模板实际通过](dory-tokenizer-template-20261108.md)：13 组输入、1,000 个附加特殊标记通过，全部保存 token 经独立字节解码复核。确认当前 Transformers 对 Dory 发出 Mistral 正则修复误告警，原正则已与建议替换值相同，官方文件保持不变。尚未执行模型前向或生成，已有提速表数值不变。

[MiniCPM5 固定完整 context 的输出投影实测](minicpm5-fixed-oproj-20261108.md)：三份完整输出控制通过，六种公式在原完整形状运行；仅加宽求和没有变化，CPU context 改用完整 BF16 投影后与原生输出的差异从 802,431 个降至 2,967 个。27 份原始数据、204,808,192 字节及全部统计/舍入关系独立复核通过。整模型候选尚未运行，原质量与性能门槛保持不变。

[MiniCPM5 完整缓存与保存数据实测复核](minicpm5-layer1-saved-data-result-20261110.md)：84 份完整 logits、168 份完整缓存与历史衔接核验完成；96 个数值判定和首选索引重现原结果。原对照 29/32、两个候选各 28/32。75 项 FP64/FP32 余弦超容差失败保留，并通过全部 84 次调用与旧精确重放的等价核对确认其既有来源。自身历史重放尚未启动，没有新增提速。

[MiniCPM5 第二层自身历史注意力实测复现](minicpm5-layer1-own-history-result-20261110.md)：真实双 rank 的 36 次完整四行调用全部逐字节一致；退出后独立复核全部 589,824 字节。新证据链接通原 FP32 精确重放、独立 FP64 判定和完整缓存，保留标量容差失败。交叉历史矩阵尚未执行，整模型和提速结论不变。

[Dory 原生环境来源核对](dory-runtime-provenance-20261111.md)：Torch 官方归档12,395/12,396文件相同，仅安装清单不同；Python解释器及三份环境副本匹配上游，配置源码和三份字节码差异保留。MiniCPM5内部观测驱动两侧32项源码检查通过，尚待运行接入。无新增整模型合格结果或提速倍数。

最新环境验证：[Dory 签名状态、Release 图标路径与空 XZ 尾流的实际兼容诊断](dory-signed-metadata-compatibility-20261111.md)。原失败及其独立复核完整保留，没有新增模型提速。MiniCPM5 原生 r1 的失败记录已完成，尚无原生张量。
