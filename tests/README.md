# 测试目录

已为每个 Solver 模块加入 unittest 测试，支持 `python -m unittest discover -s tests -t . -v`。为减少依赖采用标准库测试工具，后续按需要使用 pytest。

覆盖输入模型、编号映射、几何、刚度矩阵、坐标变换、整体组装、荷载、边界条件、节点位移、支座反力、杆端内力及基本失败情形。

参考值需有独立来源，比较时明确单位和绝对/相对误差容限。具体测试随理论和模块实现加入。

当前 42 个测试方法通过，证据与源码哈希在 `docs/validation/`。课件 C06/C09、三维解析/力法/能量积分验证保留；新增教材第五章矩阵逐项核对、正文/附录局部轴及符号转换、原题 IV-1/IV-2 位移和截面力验证。原题显式转为节点模型，IV-2 两项印刷差异保留并由独立方程核对，详见教材核对与 E04。

新增 test_parameters.py 的 14 项参数/Agent 离线集成测试，全套共 56 项通过。
日志、源码哈希与离线示例见 docs/validation/material-section.md。这些测试不调用模型 API。

再新增 test_member_loads.py 的 20 项跨内荷载/截面与 Agent 接口测试，当前全套 76 项通过。
详见 docs/validation/member-loads.md，教材通用原题、解析解和局部/总体平衡均核对。

新增 test_postprocessing.py 的 8 项杆内变形与可视化数据检查，当前全套 84 项通过。
实际日志与源文件哈希见 docs/validation/visualization.md；浏览器检查是人工验证，未新增自动 UI 测试框架。
