# Qwen slot r2 独立全量 raw 审阅

审阅者：`/root/qwen_slot_diagnostic_source_review_20261112`。本记录只独立重算保存的 raw/u32 数据；实际运行、退出和 drain 引用 root 回执，未重新观察进程或锁，不是 root acceptance。

| Arm | Profile | 比较字节 | 差异字节 | 差异 BF16 字 |
|---|---|---:|---:|---:|
| reference | original_raw | 98,304 | 139 | 139 |
| reference | negative_nan_raw | 98,304 | 155 | 147 |
| candidate | original_raw | 98,304 | 0 | 0 |
| candidate | negative_nan_raw | 98,304 | 0 | 0 |
| identity | original_raw | 98,304 | 0 | 0 |
| identity | negative_nan_raw | 98,304 | 0 | 0 |

两个 reference 输出均**逐字完全符合**：`0x8000 → 0x0000`，全部 BF16 NaN `→ 0x7fc0`，其余字不变。原 profile 有5个负零、135个正 NaN；其中1个 NaN原本就是 `0x7fc0`，故134个 NaN发生变化。负 NaN profile 额外包含4个 `0xffc1` 和4个 `0xffa1`，全部变成**正** `0x7fc0`：负号及原 payload 均不保留，8个字各改变2字节。

candidate 与真实 NKI identity 的两个 profile 均保存全部49,152个 BF16 字，包括负零、NaN符号/payload、无穷及次正规数。六个 u32 文件与独立重算的**完整有序差异位置**逐项相同，无缺失、额外、重复或越界；空差异文件也已验证。六份输入 raw 通过独立固定公式重建；report 中输入后验SHA与保存输入一致，未把不存在的独立输入后验raw快照当作证据。

这些结果排除了“所有臂共享的无条件传输规范化足以单独解释差异”。差异仍可能位于 reference 特定的图、运算、编译或运行路径；未定位具体compiler pass，也未排除所有传输交互。范围仅为 `[4,3072,4]` BF16、两组raw、全槽 `-1`、每臂每组一次调用；不是原8-layout、primitive、整模型或性能资格，timing为0。

全量逐字分类、每个变化字、全部偏移与文件SHA见[qwen35-slot-special-value-r2-independent-raw-review-20261112-v1.json](qwen35-slot-special-value-r2-independent-raw-review-20261112-v1.json)。JSON SHA256：`1cb3e38eb2d0dec4fad72f4f027d34c2e8437814fa15f0f7806bf5ec41094b23`。

原实际运行：session29233 / terminal `bc15e6` / exit0；root回执SHA：`24536a19e7fcef6a8d13e17ab1f10749916b2e4ca057076a729f94b4904e09fd`。只使用stdlib读取指定raw/u32/report及该回执，未读真实模型artifact payload、未运行native、未扫描锁/History/current binding。
