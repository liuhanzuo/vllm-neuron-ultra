# Spark 原权重与缓存张量诊断源码（待串行接入）

本次提供 `spark26_native_port_diagnostic_contract_v1.py` 与 `spark26_native_port_diagnostic_v2.py`。没有执行 native/Torch、权重读取、下载、编译、服务或锁操作。v1 已经运行源码测试，故保留原样；v2 仅补充真实调用前后原权重及输入 K/V 不变检查。

## 固定输入和真实调用

官方模型仍为 `XHToken/Spark-X2.6-4B`，revision `aa074aa851209a683aeaf5f99a8e25a642d94541`。使用 root 新取得的固定 revision 原始 API/index JSON 摘要，index 的 290 个独立键等于原 291 个 state keys 去掉独立 `lm_head.weight`；其原值来自 tied embedding。全部五个 safetensors 文件共 8,224,192,408 字节，逐文件使用官方 LFS SHA256，完整流式读完后才允许 safe_open 读取原 QKV tensor。每个 shard 的完整 keys 也必须匹配 index；不下载、不量化、不替换 dtype。reader 每次 tensor 前后复核原文件身份，模型 tensor 不落盘。

静态计划列出 36 层的 `model.layers.N.self_attn.q_k_v_proj.weight`，原 BF16 shape `[6144,2560]`。真实 SDK `create_local_weight_qkv` 完整函数按 rank 切分，并与独立 Q/K/V 行切片逐位核对。官方 `Spark2_5Attention` 的原 `q_k_v_proj` 和 v2 `SparkPackedProjection` 接受同一原权重，使用三种固定形状的合成 hidden inputs。meta 构造仅避免给未用参数随机分配；原权重随后直接绑定到实际 CPU projection。SDK TP2 rank 分片的投影分别计算，完整记录 BF16 误差和是否逐位相等，**没有真实 TP2 collective**。非逐位相等仅是诊断分类，不能因此增加或替换整模型门槛。

缓存使用原 v2 `prefill_cache/decode_cache` 实际 Torch 函数与官方 `DynamicCache` 对照，保留原 head dimension 256、每 rank 两个 KV heads、B4 slots、SWA511 与 full4096。两种缓存层 × 两个 rank 各执行 534 个步骤，共 2,136 步，包含连续 520 步 decode、跨 511 边界、乱序请求、B1/B2/B3/B4、dummy -1、复用三个槽位和最大 cache position4095。每一步完整比较所有物理槽位，验证未参与槽位、dummy 的写隔离及输入缓存/KV不变。每次 prefill 另以新分配零缓存比较复用行。这是 CPU 张量冷缓存对照，**不是新进程模型冷启动、设备缓存别名或整模型长解码合格**。

## 原科学门槛保持

原 `validate_real_logits.py` 摘要保留，logits 门槛仍为 cosine≥0.995、KL≤0.02、max_abs≤2、top1比例≥0.9、reference margin>0.5 必须匹配。它们没有套到非 logits 的 QKV 张量上。原 logits32、fresh-process cold-cache、长解码、槽位复用/隔离、quality96 和匹配 HTTP 全部明确列为待执行。缓存和权重分片要求逐字节相同；投影舍入差异完整保留。

## 准入缺口与预算

core 不能单独执行，脚本入口直接拒绝。未来外层必须绑定真实原串行锁 device66306/inode8294421、实际 worker、原 Python3.10、当前完整来源和原生库认证，再通过 `guard` 和 `load_native` 接入。core 的 authority 字段校验只是接口形状，**不能用一份声称已认证的 dict 代替真实准入**。它不会改 v2 adapter 的无条件 guard，也不自动安装服务或注册模型。

原始权重尚未下载。本轮只曾查过两处标准 HF cache 目录不存在，不能把这称为全机权重普查。完整 download 最低需 8,224,192,408 字节；分片重试、缓存 staging、存储保留线及未来编译额外预算必须由 root 在串行空闲时明确决定。此 CPU core 不复制持久 checkpoint，不保留临时完整模型；最大 full cache 本体为 `[4,2,4096,256]` BF16，每个 K/V 16MiB，但多份完整 clones、官方 cache、投影权重、Torch 加载内存仍需实际资源预算。没有声明已经适配某个运行上限。

## 本轮检查

v1：13 项 stdlib 单元检查，工具 c009fc/0。v2：13 项 stdlib 单元检查，工具 d2a5ef/0。覆盖固定 metadata/key集合、全 trace 历史、原门槛不可改、错权限先于native factory拒绝、权限变化拒绝、原shard完整hash/size/符号链接、未认证先读tensor拒绝、失去ownership错误传播。微小临时 byte fixtures 只测试 reader IO，不是模型权重。测试没有执行 QKV 或 cache tensor 科学函数，不能证明这些函数的 Torch shape/index/runtime 行为。
