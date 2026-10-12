# 集中荷载位置纠正修复（2026-10-11）

真实Qwen调用中第一次补丁遗漏集中荷载的interval字段，被“Field required”拒绝；
第二次给出不符合两项区间要求的内容，被“interval应为起止位置两项”拒绝。
已保存诊断不含供应商完整参数值，因此不判断第二次究竟是哪种无效区间。
这是识别记录/补丁字段格式问题，不是API未联通、额度耗尽或用户纠正内容不合规。

修复：

- RecognizedLoad.position/interval可省略为None。没有实际位置的信息不会补成0或全跨。
- 无信息的空数组/空字符串位置字段转为None，然后按类型校验。
- 集中力、力矩无需interval；节点荷载无需任何杆内位置；均布区间仍必须明确。
- 非空冲突区间、非两项区间、错误荷载分量仍拒绝。
- load_changes支持按原label只修改position、interval、target_id、coordinate_system、distribution或components。
- 只更新显式提交的字段；原荷载大小、方向、坐标、单位等保持原样。仅写描述不算修改。
- 修改后重新验证整份识别记录与物理模型，继续要求用户确认后才可求解。

新增8项回归，全套138项通过。
证据：results/member_position_correction_test_run.txt。
当前用户记录的离线回放更新了节点2固定约束、①/③集中力距各自i端1m、节点4顺时针力矩5kN*m，原均布荷载逐字段保留。
只回放，没有修改用户当前会话、确认或求解；work/member_position_correction_replay.json保存验证摘要。
新提示与输入兼容的真实供应商提交效果，仍待重试。
