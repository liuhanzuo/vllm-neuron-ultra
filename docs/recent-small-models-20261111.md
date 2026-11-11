# 小模型候选复查：2026-11-11

本次沿用 11 月 8 日的 19 个官方发布者，每家查看最新创建的 16 个仓库，共 293 项；其中 13 项版本变化或新仓库、6 个新建仓库。随后核对 9 个目标的元数据、模型卡和配置，共 45 次成功请求、1,096,284 字节响应。该范围不覆盖所有发布者，也可能遗漏未进入最新 16 项的旧仓库更新。

本轮没有发现足以替换现有主线的新热门通用文本模型。原 15 个家族、固定实验版本和验收要求全部保留。较值得新增观察的是 MiniCPM-OCR 1.0 的专项 OCR 推理；新发布的 ZoomGain 4B/9B 则可作为现有 Qwen3.5 家族的视觉变体。

| 候选 | 参数和热度依据 | 判断 |
| --- | --- | --- |
| [MiniCPM-OCR 1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0/blob/883ca368d17335b9ce5aa6fef4d000ef0684fcaa/README.md) | Hub 全仓库 BF16 元数据 1,300,434,160；114,431 下载、455 赞 | 本轮最值得保留的额外专项候选。SigLIP 视觉编码器 + Qwen3.5 语言模型，定位文档/OCR；不能未经验证就代替通用文本基线。仓库创建于 10 月 21 日，本轮仅看到版本更新，不能称 11 月 8 日后新发布 |
| [MiniCPM-OCR DSpark](https://huggingface.co/openbmb/MiniCPM-OCR-1.0-DSpark/blob/cb24ab4f593389832024a688216e05bf3803950e/README.md) | 草稿头 296,837,889 个 BF16 参数，另有 114,688 个 I64 映射项 | 官方明确不是独立 OCR 模型。主模型与草稿的 BF16 元数据合计 1,597,272,049，规模仍低于 10B，可保留推测解码研究方向 |
| [ZoomGain-4B](https://huggingface.co/inclusionAI/ZoomGain-4B/blob/2ac9e28f1de993b4d7b6ebeda1368ff2e6982a71/README.md) / [9B](https://huggingface.co/inclusionAI/ZoomGain-9B/blob/b4550ef2e5a30b33c5133278036a682197489145/README.md) | 4,539,265,536 / 9,409,813,744 个 BF16 元素；13/6 次下载、2/3 赞 | 官方卡明确 11 月 11 日发布检查点。Qwen3.5 视觉感知微调模型，当前热度低；保留观察，原 Qwen3.5 的数值验证与运行版本保持 |
| [SingVL S1 4B](https://huggingface.co/inclusionAI/SingVL_S1_4B/blob/93f15cd61c1cf417d9cb2631f6be2df13510829e/README.md) | 4,539,265,536 个 BF16 元素；17 次下载、3 赞 | 官方用一次前向对给定选项评分，支持 choice/yes-no/score；尽管架构字段是 Qwen3.5，也不应直接纳入自回归长解码基线 |
| [Granite Speech 5.0 2B](https://huggingface.co/ibm-granite/granite-speech-5.0-2b/blob/ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7/README.md) | 2,313,207,132 个 BF16 元素；9,852 次下载、82 赞 | 官方发布时间 10 月 20 日，当前版本与上次一致。ASR/语音翻译专项模型，包含音频编码器和投影器；不替换现有 Granite 文本家族 |
| [Metatron v1](https://huggingface.co/nvidia/Metatron-v1/blob/b19cb13f1953ea4c1345faa82912b80e34679da6/README.md) | 官方描述 Unitree G1 机器人动作/控制策略，ONNX 与原生检查点 | 排除语言模型队列；没有词元生成接口 |

这里的参数量是 Hub safetensors 元数据筛选证据，尚未下载权重核验实际文件头或全部字节。上述完整 BF16 候选以全仓库张量总量判断，包含元数据中的视觉/音频组件；没有把 MoE 激活参数、量化打包后的元素数或只算语言主干当成完整规模。

继续排除 [DSpine for Qwen3.8-27B](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16/blob/5f54174d301dbca65d9aa4448dc47e56732d9a12/README.md) 的独立小模型资格：其 2.026B 草稿仍需同时加载 Qwen3.8-27B。新出现的 [Kimi-K3.1-Pro-NVFP4](https://huggingface.co/nvidia/Kimi-K3.1-Pro-NVFP4/blob/a0b0283f0f4f91f09b37a6e5fe45a62de269cbd0/README.md) 官方声明总量 2.8T、激活 104B，NVFP4 文件不能改变超出 10B 的判断。GGUF/AWQ/GPTQ 更新仅记录，没有用它们替代原 BF16 候选。

对于 MiniCPM-OCR，后续先确认官方原生参考、图像输入形状与处理流程，以及 Qwen3.5 混合注意力/循环状态在 Neuron 上的对应实现。若开展 DSpark 优化，还需要正确处理 248,100 输入词表与 114,688 草稿输出词表的映射、15 个草稿 token 的目标验证、拒绝后的缓存和状态回退，再以相同主模型及输入与不启用草稿的 Neuron 路径配对计时。这些尚未实施，不改变当前 15 家族的冷缓存、长解码、槽位复用、质量与 HTTP 验收要求。

MiniCPM-OCR 模型卡的“效率提高 55%”来自上游 OCR 配置；其他模型卡的 GPU/推测解码数据同样不是本机 Trn1 结果，本次没有新增基线、合格模型或提速倍数。

已保存全部公开响应、固定版本 URL、请求时间、长度与 SHA-256；45 份响应已逐项重新核对。没有下载模型权重、执行远程模型代码、导入原生框架或启动编译/设备工作，现有模型版本和验收要求保持不变。

证据：[筛选记录](evidence/recent-small-models-20261111/selection.json) · [目标元数据](evidence/recent-small-models-20261111/targeted.json) · [原始请求](evidence/recent-small-models-20261111/receipts.json) · [独立复核](evidence/recent-small-models-20261111/root-confirmation.json) · [完整清单](evidence/recent-small-models-20261111/manifest.json)。
