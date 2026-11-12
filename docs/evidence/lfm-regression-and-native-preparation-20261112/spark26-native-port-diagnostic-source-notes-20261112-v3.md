# Spark 原生诊断后继：真实 host 方法与物理输入行排列

本次新增 `spark26_native_port_diagnostic_contract_v2.py`、`spark26_native_port_diagnostic_v3.py` 及 `check_spark26_native_port_diagnostic_source_v3.py`。原 contract v1、core v1/v2、原 port v2 和此前交付全部保留原摘要。没有执行 Torch/native/ctypes、权重读写或下载、服务和锁操作。

## 原计划与新增范围

新计划严格包含原 534 步缓存 trace、两种 cache × 两 rank 的原 2,136 步，以及全部 36 层 × 三种输入形状 × 两 rank 的 216 组原权重投影诊断。QKV 执行体和 OriginalWeights reader 直接复用冻结 core v2 的函数/类，不投影或重写原科学流程。原 logits 门槛、cold/long/slot/quality96/HTTP 待执行清单保持。

每个原 decode 事件先按原请求顺序创建真实 K/V、position、request markers，调用实际导入的 `vllm_neuron.worker.neuronx_distributed_model_loader.NeuronModelBase._reordered` 与其 `_sort_inputs`，检查四类输入按同一排列排序，再调用原 contextmanager 返回的 `restore` 逐项还原。返回的实际排序 K/V 和 position 被用于后续 B4 缓存输入构造。共 527 decode × 两 cache × 两 rank，拟调用 2,108 次 host 方法组。

调用使用最小 receiver（`is_reorder_needed=True` 和实际 `_sort_inputs`），只执行原目标路径的两个方法；**没有构造完整 vLLM 模型，没有调用完整 serving forward，也没有调用 SDK padding**。方法来源必须通过实际 class/module 名与原源码 `co_filename` 核对；`_reordered` 的 contextmanager 包装经 `inspect.unwrap` 定位原函数。外层仍须认证完整依赖。运行模块不使用 AST、exec、eval 或 compile 伪装原函数。host `_reordered`/`restore` 真实执行只有未来 native 测试才能证实，本轮测试里的 provider 明确是 list mock，并且不能通过 runtime 原方法来源检查。

在原 trace steps 4、5、6、7（分别 B2、B3、B1、B4）处，保留当步的原 prior-cache 快照。对四个物理输入行全部 24 种排列同步重排 K/V、seq_ids、position，再调用实际 v2 `decode_cache`。每次均从同一原 prior-cache 开始，不累积额外解码。返回 K/V、key positions、valid mask 用逆排列恢复，persistent K/V 不作行还原，直接按物理槽位与原未排列调用逐位比较；同时检查 prior-cache、原输入及排列后输入均未改变。

共有四类快照 ×24排列 ×两cache ×两rank，新增384次helper调用。这些排列包括dummy行的位置变化；dummy返回数据仅用于同一诊断调用的排列不变性，不能作为真实请求输出。所有排列在当前原步骤即时运行，完成后释放临时量，不持久保存张量。原534步和原state推进不增加decode token。

## 预算和真正执行仍待完成

新 core 仍无单独CLI执行入口。未来原串行worker必须持有原device66306/inode8294421锁，真实绑定当前owner和原Python，验证完整native/source闭包，并供给真实native providers和有界emit。新增固定loader源码pin已纳入authority检查，旧身份要求保持；一份声明已认证的dict不能代替真实认证。端到端model及芯片接口仍未获得资格。

每个 full KV cache `[4,2,4096,256]` BF16 为16MiB，K/V一对32MiB。排列函数额外保存一对 prior-cache clone（32MiB）用于输入不变检查；每个排列的结果、逆排列临时量和原trace自身的clones也占内存。SWA缓存相应更小。这里只给出直接张量尺寸，不声称实际峰值受控：Torch分配/模块/native库、QKV权重与reader映射都必须包含在外层预算。新增384次调用和2108次host排序/还原的实际时间未测量。

emit新增2108条host记录与384条物理排列记录，原QKV和cache记录保留；总条数原216+2136+新增2108+384=4844。实际序列化格式、日志、错误输出和总文件预算必须外层固定后审核，不能把有限条数当成已完成资源准入。没有权重下载或编译预算授权。

## 本轮检查

`python3 -B check_spark26_native_port_diagnostic_source_v3.py`：13项通过，实际工具186ae2/0。检查覆盖完整原plan保留、固定四类快照/24排列、删减plan拒绝、新host来源pin缺失时在native factory之前拒绝；64种请求slot顺序的list-mock调用和inverse restore；四种batch各24排列的list-mock调度；错误host排序/还原、写错persistent槽位、错误返回行序、修改prior-cache均失败；mock provider不能代替真实runtime函数；运行模块无AST执行机制、无native导入。本轮没有运行真实tensor科学函数或SDK方法。
