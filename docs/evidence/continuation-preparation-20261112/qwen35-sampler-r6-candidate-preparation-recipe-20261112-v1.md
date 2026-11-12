# Qwen r6 候选元数据准备：执行配方 v1

本工具只从固定 r5 历史和显式未来元数据生成待审核对象。当前没有选择任何未来退出记录、catalog、run_name 或输出路径，也没有保存执行候选。

## 输入与输出

`prepare_qwen35_sampler_r6_candidate_v1.prepare(future, read_metadata, guard)` 返回候选 overlay/spec/review 对象和精确编码摘要，不写文件，不运行 Bundle.resolve，不发现 latest，也不启动服务或取得锁。

固定历史为 r5 spec（47929 个完整 source pins）、r5 overlay（3397 个 source_extension、137 个 input_extension）、r5 catalog（117 条完整串行记录）、已接受的 v6 root source review、leaf v2 root source review。它们按固定 SHA256 读取。原 spec 的 36 个 input_extension 和 overlay 的 137 个 input_extension 各自保留原身份；不将 spec 独有输入强塞到 overlay，避免覆盖原 base 输入。

`future` 必须显式提供以下 10 个字段：

| 字段 | 含义 |
|---|---|
| `terminal` | 与未来 catalog.current_handoff 完全一致的 root 已观察 idle handoff 引用 |
| `prerequisite_exit` | 必须已列入 catalog 的实际完成工作退出记录引用，例如之后的 LFM actual exit |
| `catalog` | 完整未来串行 catalog 引用，必须逐项保留原 117 条记录 |
| `run_name` | root 显式选择的新运行名字 |
| `output_root` | `/mnt/nvme/neuron-work/qwen35-sampler-pair-` 加 run_name |
| `overlay_output_root` | 显式新的历史 overlay 命名空间；只有元数据引用，没有创建目录 |
| `running_handoff_path` | 显式未来 running handoff 路径；此工具不写它 |
| `overlay_path` | 待审核候选 overlay 文件名 |
| `spec_path` | 待审核候选 spec 文件名 |
| `review_candidate_path` | 不授权的 review 候选文件名 |

前三者使用 `{path, sha256}` 引用，其余为绝对路径或 run_name 字符串。没有任何默认未来值。新的 output/run/handoff 不能复用 r5 的值；候选文件名不能覆盖固定历史或未来输入。实际路径是否尚不存在，仍需 root 在写入/准入前独立确认。

## 在实际退出之后使用

先由 root 完成当前工作的实际退出与排空审计，将该记录加入完整 catalog，再形成 idle handoff。完成这些工作之后，才把对应精确引用放入 `future`。不要用仍在运行的 handoff 或用预期退出代替实际退出。

在隔离的元数据准备进程中使用：

```python
from prepare_qwen35_sampler_r6_candidate_v1 import JsonMetadata, prepare, encode

# future 的全部 10 个字段由 root 显式填入真实、已完成的引用/新路径。
# guard 采用 root 此次元数据准备工作的原有约束；这里不提供默认许可。
reader = JsonMetadata(guard)
candidate = prepare(future, reader.read, guard)
```

`JsonMetadata` 只接受显式 `.json` 路径；对每次读取检查稳定 stat、完整字节 SHA256、唯一 JSON 字段及再次 stat。同一路径再次出现时不能刷新 identity。helper 将新的观察与所有旧 identity 对照，冲突直接失败。

`candidate['artifacts']` 是 `path -> object` 映射。root 独立检查后，可用 `encode(object)` 的精确字节以独占新建方式写这三个候选文件；`candidate['references']` 给出相应摘要。不能修改格式后继续使用原摘要。此步骤仅保存候选，不写 running_handoff，不启动 runtime。

review 候选使用 `passed=False`、`launch_authorized=False`、`reviewer='/root/candidate-preparation'`，明确列出独立重建待完成项。原 `validate_spec` 会拒绝它。不要为了让 Bundle 构造器通过而把这些字段临时改为 true 或伪装成 root 已审核。

## root 后续独立复核

此 helper 使用原 Qwen contract 的引用校验与 source merge，复用既有 overlay 的结构和完整 recovered/source 分区。它没有重新运行 recovered completion，也没有推断新 recorder/auditor 的继承 source witnesses；因此不是完整执行 closure 认证。

root 应沿用既有独立源码审阅流程重建 recovered 与 native_overlay_v3.resolve，执行 v7 executed_extension 的完整 witness 核对，并使最终 Bundle.resolve 的完整 source、inputs 与候选一致。如果独立重建发现额外来源或输入，需要先补齐候选且重新审核，不能放宽旧行或旧 identity。

Bundle 的正常构造器本来就要求最终 root review；本配方不增加绕过入口。独立 review 和 Bundle 检查由 root 的既有 source-only 审核流程完成，确认之后才创建最终 root review。实际准入仍需最新历史、全部物理检查、原租约和新输出路径检查。

## 已验证范围

工具 session 73739、terminal `08c316`、exit 0：12 项检查通过，unittest 报告 10.544 秒。检查读了固定五份历史 JSON（包括旧 r5 的完整 117 行 catalog），其他 terminal 内容和未来新增 row 均为明确合成 fixture；真实稳定 reader 仅对临时 JSON 做正反检查。没有读当前 handoff/catalog、实际模型 raw/RECORD/native 内容或查询服务/proc/锁，没有创建 r6 当前候选文件。

合成成功例保留所有 47929 旧 pins，加 6 个审核过的 v6 源码与 1 个合成 catalog source，得到 47936 pins；保留旧 117 行并追加合成行得到 118 行。测试覆盖旧行丢失/变更、旧 pin 冲突、新行间 pin 冲突、旧 identity 改动、未来仍 live、缺 prerequisite、缺显式参数、候选路径覆盖历史、source review/leaf 变化以及原运行合同拒绝未授权候选。

只跑本准备工具检查：

```sh
python3 -B /home/ubuntu/experiments/trn1-latest-models-20261031/research/check_prepare_qwen35_sampler_r6_candidate_v1.py
```

没有新增运行授权、native 合格或提速结论。
