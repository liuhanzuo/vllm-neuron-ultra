LFM full CPU v3 仅修正已核实的重复日志，并保留全部科学门槛。

r1 保存的 runtime.log 恰为 262,144 字节。四条精确的 SyntaxWarning 两行记录共 929 次，占 261,963 字节；最后 181 字节是下一条同类警告的截断前缀。没有可见的 EFBIG、SIGXFSZ 或 flush 异常 traceback，因此“写满共享日志导致解释器退出 flush 失败”是与 exit 120 一致的原因假设，底层异常类型仍未直接取得。

原环境完成 291 个退出 0 的批次 / 3,716 cases。新环境写出 232 份报告 / 2,968 cases，其中只有 231 个批次 / 2,967 cases 以 0 退出。最后一批 extended_cpu 01319–01320 的报告 complete/passed 均为 true，但进程最终 exit 120；worker 的保存 service-exit 同样为 120。不得把报告完成计为该批次或全轮成功。实际 outer exit、物理 drain、ACK 全量核对及新 catalog 由 root 负责；本交付没有重新执行这些操作。

v3 在 runner 首个本地依赖导入之前安装小型 warnings.showwarning 转发器。只有类别恰为 SyntaxWarning、消息、绝对路径、行号和源码完整 SHA256 均匹配这四条已审定签名时才去重；每次匹配都重新进行有界、稳定、O_NOFOLLOW 的源码读取。显式其他输出流或变化的源码行也继续原样转发。无 PYTHONWARNINGS、无 ignore filter、无原模块修改。

worker 对每条已识别警告保留首次原始输出。batch 将已识别警告的完整签名、源码行、observed/forwarded/deduplicated 计数收入既有 batch JSON；worker 在既有 runtime-auth 和 diagnostic-result JSON 保存自己的快照。计数范围是快照之前送达 showwarning 的调用，不声称统计 Python 原有过滤器已经隐藏的警告、其他库自行捕获的警告或解释器最终关闭阶段。外部导入本模块无警告状态修改；outer 模式沿用原日志行为。未知警告、异常和 traceback 不经过额外抑制。未知输出仍可能触及原文件上限并使实验失败，不能把本修正声称为无限日志保障。

未增加输出文件、写权限、子进程、线程或队列。原 874 children、三轮 291 batches / 3,716 cases、单独三臂 controls、bitwise/input/shape/dtype/extraction 检查以及实际子进程退出检查均保持原样。原 1 GiB 总输出 / 2,048 files / 256 KiB per file、MemoryMax 10 GiB、AS 12 GiB、CPU 6–7、nice 19、30,000 秒 worker 预算保持原样。原 no-spawn/no-network/no-device/output-only audit 不变；源码读取按文件名进行，避免该审计禁止的任意 numeric-fd 包装。

纯集成 26 项 + 警告专项 12 项通过（最终工具终态分别为 3e491a/0、7ec149/0）。集成规范化 AST 检查证明：runner 删除新增 hook/receipt 后整体与 v2 相同；leaf 删除新增 warning receipt/validation 后整体与 v2 相同。专项覆盖精确/变化签名、源码 SHA、非 SyntaxWarning、显式输出流、原异常转发、收据造假、874 个独立 batch 的计数，以及 passed 报告不能掩盖最终 exit 120。以上为 source/mock 证明，未执行 native framework、模型、RECORD 验证或完整 CPU science。

新源码闭包 candidate 含 595 个 Python pins，保留 v2 全部 592 pins，仅增加 runner v3、leaf v3、warning helper。闭包只作为候选供 root 独立复核；未创建 spec/review/admission，没有触碰当前绑定或自动启动。
