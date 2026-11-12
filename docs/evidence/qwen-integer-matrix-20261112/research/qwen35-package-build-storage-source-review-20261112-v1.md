# Qwen3.5 package build storage source review

本次结论：主要显式编译输出路径已由真实 SDK 调用链证实映射到六棵受管目录；不能据此宣称所有闭源原生写入均受六树硬配额限制。审阅没有运行原生导入、编译、模型、服务或锁。

预算要求为 root 29 GiB、NVMe 39.28125 GiB。root 准备回执时分别余 144,257,024 与 1,193,492,480 字节；该旧快照不是实时准入。视图只复制配置并链接原文件，不应把链接目标权重重复计入新增空间。

1 GiB artifact、2 GiB cache、3 GiB workdir、256 MiB build auxiliary、32 MiB view，以及原科学 2 GiB 是有限试跑上限，没有峰值预测资格。特别是 wrapper 的临时 HLO/NEFF 副本位于 shared-build/tmp，受 auxiliary 单文件 128 MiB 限制，即使 build LimitFSIZE 是 1 GiB 也不能越过这个扫描上限。

当前 save_sharded_checkpoint=False，且 compile 传 initialize_model_weights=False；不会必然将原始大权重另存到 artifact 或 workdir。实际中间产物是否超限仍需原资源监控判定。

源码发现 v2 会把合法 cache 原子重命名、lock unlink 及 bucket 工作目录重建当成未知消失。新增 v3 只扩大这些已界定路径的有限重扫：放弃不完整统计，重新取得拓扑/身份收敛的完整六树扫描。日志及普通文件的原地增长可以继续，每次完整扫描仍检查全部文件、条目、分类和总量上限；返回最新完整实际计费以继续原 free-space/reserve 检查。

可选 layout transform 经 torch_neuronx 直接调用 compiler，继承实验 ROOT 作为 cwd；已知 output/logfile 和 TMPDIR 被路由，但闭源 compiler 的其他相对写入没有源码覆盖证明。HF_HOME 也在六树外。保留原连续 root/NVMe floor 检查，不新增硬配额承诺，不删除共享缓存。

完整证据 JSON：[qwen35-package-build-storage-source-review-20261112-v1.json](/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35-package-build-storage-source-review-20261112-v1.json)，SHA256 `f9231ba5dda376514a601a9fc928d62cecfc820dc3901a7a6bab3a6d07b5b040`。

| 证据 | 源码位置 | SHA256 |
|---|---|---|
| Qwen inherits NxDI application compile | [qwen35/modeling.py:201](/home/ubuntu/src/vllm-neuron-ultra-multi-model-20261107/vllm_neuron/models/qwen35/modeling.py:201) | `2cffd29a292a5b5115e8c4499f0ab1a24b3388abbf6d9787580138ff918234ad` |
| NxDI inheritance | [models/model_base.py:3074](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/model_base.py:3074) | `b130a8540211a31c4d8715139d2b2d81385497d01ca36a16affa8d8f03056b63` |
| Explicit builder workdir | [models/application_base.py:165](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/application_base.py:165) | `c894ac1f1aa3b6563051d6bfde230c468aa72d47ee6fc282b5a15b1c821b67ad` |
| No trace checkpoint serialization | [models/application_base.py:303](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/application_base.py:303) | `c894ac1f1aa3b6563051d6bfde230c468aa72d47ee6fc282b5a15b1c821b67ad` |
| Sharded checkpoints default false | [models/config.py:367](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed_inference/models/config.py:367) | `6380d0671130b519642509bf4de76412a7ab0f6d092c3676197cd1610185314e` |
| Actual cache API | [trace/model_builder.py:603](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:603) | `7323a755fbe8c9253cdb8977d8a21dca083c9b132d2b8a3664aa89e0da047e91` |
| Compiler working directory | [libneuronxla/neuron_cc_wrapper.py:114](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:114) | `33218bc87806d18277a47a2a7c5d9e7ddc78314d00f27c86816d2ec3731c961f` |
| Compiler temporary HLO/NEFF copies | [libneuronxla/neuron_cc_wrapper.py:251](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:251) | `33218bc87806d18277a47a2a7c5d9e7ddc78314d00f27c86816d2ec3731c961f` |
| Cache environment selection | [libneuronxla/neuron_cc_cache.py:78](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_cache.py:78) | `cdfa0d6f5c2481da7d4331b36e14985363b6b29c05d257c27c866b8a1e841f90` |
| Cache atomic publication | [libneuronxla/neuron_cc_cache.py:385](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_cache.py:385) | `cdfa0d6f5c2481da7d4331b36e14985363b6b29c05d257c27c866b8a1e841f90` |
| Cache unlock unlink | [libneuronxla/neuron_cc_cache.py:343](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_cache.py:343) | `cdfa0d6f5c2481da7d4331b36e14985363b6b29c05d257c27c866b8a1e841f90` |
| Compiler bucket workdir reset | [libneuronxla/neuron_cc_wrapper.py:142](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/libneuronxla/neuron_cc_wrapper.py:142) | `33218bc87806d18277a47a2a7c5d9e7ddc78314d00f27c86816d2ec3731c961f` |
| Fixed SDK cache version | [version/__init__.py:1](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronxcc/version/__init__.py:1) | `2f1b7813d08920761278643cd87449c287580260e23b28fb1a74cadbbf31a726` |
| Optional layout compiler route | [trace/model_builder.py:1273](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/neuronx_distributed/trace/model_builder.py:1273) | `7323a755fbe8c9253cdb8977d8a21dca083c9b132d2b8a3664aa89e0da047e91` |
| Optional layout inherited cwd | [xla_impl/trace.py:320](/home/ubuntu/venvs/vllm_neuron/lib/python3.10/site-packages/torch_neuronx/xla_impl/trace.py:320) | `5bab452aaae0a73883c5ae6c3e57ed86fbe121095c3a622ad6c3675d2d390786` |
| Leaf temporary routing | [qwen35_package_serial_runtime_v3/contract.py:232](/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_package_serial_runtime_v3/contract.py:232) | `217d8a4270ef88571d81587d8ab10f39df6a7bf1c13195f0212ae7b53b089024` |
| Fresh leaf environment | [qwen35_package_serial_runtime_v3/service.py:401](/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_package_serial_runtime_v3/service.py:401) | `686cb514e6af992632d4427d5ba2886a296c0e9ff9a079a1cd1640f4ec13a732` |
| Global free-space and reserve checks | [qwen35_package_serial_runtime_v3/io.py:380](/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_package_serial_runtime_v3/io.py:380) | `c87cbe6702ac25bf0a8be950dcb1367a35baee5f4167814b111e77f5e17022fb` |
| Scoped compiler transient recovery | [qwen35_package_serial_runtime_v3/io.py:212](/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_package_serial_runtime_v3/io.py:212) | `c87cbe6702ac25bf0a8be950dcb1367a35baee5f4167814b111e77f5e17022fb` |
| Complete topology convergence | [qwen35_package_serial_runtime_v3/io.py:259](/home/ubuntu/experiments/trn1-latest-models-20261031/research/qwen35_package_serial_runtime_v3/io.py:259) | `c87cbe6702ac25bf0a8be950dcb1367a35baee5f4167814b111e77f5e17022fb` |
