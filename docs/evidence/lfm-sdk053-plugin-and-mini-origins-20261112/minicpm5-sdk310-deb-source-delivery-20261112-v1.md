# MiniCPM5 八包归档核验后继源码交付

已新增六份运行源码和一份标准库/mock 检查文件，旧 Dory 和已有执行源码全部保持原摘要。输入固定为 `minicpm5-sdk310-archive-member-next-inputs-20261112-v1.json`，SHA256 `e91bca576461d9d1d4b206e7f8b508c20a103a3231bef9b29535e97d3e8da266`。

## 具体实现

| 文件 | 职责 |
|---|---|
| minicpm5_sdk310_deb_input_binding_v1.py | 固定八包、12 目标、零旧摘要；复核已审签名来源、完整明文索引、14 个精确包条目、当前 dpkg 文本及原工具 authority；不将候选文件写成实际加载 |
| minicpm5_sdk310_deb_members_adapter_v1.py | 调用原 transport v3/tar v2；逐包全归档及目标核对、保持原 lstat 身份，独立重放逐项相等，保留 3 必需／5 候选标签 |
| minicpm5_sdk310_deb_events_v1.py | 精确验证下载 41 条／重放 33 条完整有序事件，绑定每包 URL、归档、解压命令/退出、成员流和最终结果 |
| run_minicpm5_sdk310_deb_members_root_v1.py | 原父运行器仅替换模块和 worker 路径 |
| minicpm5_sdk310_deb_members_worker_v1.py | 原 worker 仅替换绑定及 adapter 模块 |
| record_minicpm5_sdk310_deb_stage_exit_v1.py | 原独立退出核验连接新绑定、结果和 41/33 事件协议 |

运行器和 worker 的 AST 已验证仅存在接线替换。原 transport、tar、串行锁、子进程 ACK、完整历史、结束排空与释放实现直接复用。完整保留的运行来源清单为 88 份：旧固定来源 82 份，加本次六份后继；清单不包含测试代码，也没有构造运行 plan/review/admission。

保留旧合同要求的输出前缀：`/mnt/nvme/dory-serial-inventory-minicpm5-sdk310-<主任务选择的新后缀>`。这是独立包来源准备阶段的命名约束，不表示执行了 Dory 模型。

## 已执行验证

`check_minicpm5_sdk310_deb_sources_v1.py` 的 24 项标准库/mock 检查通过，工具 `7e3e52` 退出 0。覆盖每一个下载事件位置的遗漏/重复（各 41 个）、40 个相邻交换、跨包来源、错误 Release／索引／URL、错误成员及主机身份、下载与解压失败不成功、旧入口拒绝新输入，以及纯接线 AST 和旧来源摘要不变。

主任务明确允许后，又执行了真实 `authenticated(guard)` 文本绑定检查，工具 `2b87f3` 退出 0。读取范围仅为固定 JSON、Release、Packages 和 dpkg 文本；八包、12 目标、14 个完整索引条目全部通过，1.994 秒、1,227 次 guard 调用。没有运行 transport、dpkg-deb、ctypes 或模型，也没有读取原生二进制、归档或张量。

测试入口（只做标准库/mock 检查）：

```bash
cd /home/ubuntu/experiments/trn1-latest-models-20261031/research
taskset -c 7 nice -n 19 prlimit --as=536870912 --cpu=60 -- python3 -S -B check_minicpm5_sdk310_deb_sources_v1.py
```

源码清单及所有摘要位于 `minicpm5-sdk310-deb-source-manifest-20261112-v1.json`；原运行器接线差异单独保存于 `minicpm5-sdk310-deb-lifecycle-dispatch-diff-20261112-v1.patch`。

## 后续仍需主任务绑定

先独立审核上述完整来源，再绑定新鲜的 inventory 阶段计划、输入摘要、当前完整历史和真实前驱退出。原始锁证明、空闲窗口、磁盘预留、子进程 ACK 及结束排空由原生命周期执行。本次没有查询或改变仍在运行的 42892。禁止复用旧 Dory 六包运行计划或把源码清单当成实际准入。

新运行器沿用原 CLI：`/usr/bin/python3 -B -s <新运行器> --plan <新计划> --plan-sha256 <完整摘要> --review <主任务审核> --review-sha256 <完整摘要>`。目前没有对应的新计划、审核或实际命令实例。独立 recorder 还必须使用工具实际返回的 `--session`、`--chunk`、`--exit-code` 以及新输出位置；不能从保存的成功字段推断真实退出。

包准备阶段沿用历史 Dory INVENTORY 限额（AS 2 GiB、CPU 600/605 秒、900 秒、单文件 64 MiB、总输出 128 MiB），不改 Mini 科学阶段的原 **9 项检查、1,284 行、900 秒、AS 2 GiB、单文件 2 MiB**。选定成员的来源完成后，完整 Python bootstrap 纯源码/原生依赖、SONAME 别名、实际 ELF／FFI／映射和科学诊断仍须分别闭合。三个必需包和五个候选包不代表完整原生闭包；所有 `actual_loaded` 仍为 null，没有模型资格或提速结论。
