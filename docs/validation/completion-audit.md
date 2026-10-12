# 本次计算资料与程序要求验收

任务：以十份课件及阎军、杨春秋 2014 年《计算结构力学》的理论和算例为依据，推导/实现空间刚架计算，并满足课件程序样式。

| 要求 | 当前证据 | 结论 |
| --- | --- | --- |
| 记录十份课件 | course/manifest.json、source-index.md；10 份原件副本哈希已核对 | 已满足 |
| 参考指定教材而非仅书目 | 教材 PDF 原件/manifest；版权页、第五章及附录 IV 的逐页图像阅读记录 | 已满足 |
| 明确空间刚架理论依据 | theory.md、space-frame-derivation.md、textbook-audit.md 的页码/公式/假设映射 | 已满足 |
| 6 自由度、12 自由度单元及完整计算流程 | model、geometry、stiffness、assembly、equations、recovery、analysis | 已满足第一版范围 |
| 课件式主程序及子函数 | main.py；i0j0/ch/stif/dor/ldlt/slveq/doh/dofanli；四分块与数字化数组；course-program-style.md | 已满足 |
| 数值结构计算交给确定性程序 | Solver 无 LLM/API 依赖；公共 solve_space_frame；结果含模型哈希、编号、单位 | 已满足 |
| 模块及独立结果测试 | test-run.txt、verification.json 的 42 项测试及源码哈希 | 已满足当前版本 |
| 教材原题核对 | IV-1/IV-2 原输入/原答案、显式适配模型、验证 CLI、位移和截面内力比较 | 已满足注明容限，IV-2明确使用 E04 |
| 不伪造缺失或错误答案 | E01～E04；IV-2 原打印值保留，独立方程验证并标记原样不通过 | 已满足 |
| 第一版范围与来源边界 | StructuralModel 拒绝跨内荷载字段；原题由验证适配器处理；剪切/温度/非线性等范围外内容未冒充支持 | 已满足 |

上述验收针对本次计算理论、核心程序和课程样式要求。项目最终的自然语言 Agent、图片理解和课程交付仍按 docs/requirements.md 的后续阶段推进，不在本次计算资料核对完成的声明中。

验收必须以当前 verification.json 中的源码哈希和实际测试输出为准；代码修改后应重新验证。教材原值不通过的两项已明确保留，不可将“注明勘误后的对照通过”改写为“整本教材所有数值原样通过”。
