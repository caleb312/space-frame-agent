# 材料与截面参数验证记录

2026-10-07 完成参数工具与 Agent 离线集成核对。

## 实际执行

- 新增 tests/test_parameters.py：14 项参数/集成测试。
- 全套 unittest.discover：56 项，0 失败，0 错误，0 跳过。
- parameters_main.py examples/material_section_iv1.json：成功，保存实际计算 JSON。
- 将用户保存的真实识别记录与明确教材原参数送入 FrameAgent：六个字段由程序填入；未请求 API，未调用结构求解。

完整日志见 material-section-tests.txt；计数、源码 SHA-256 和离线状态见 material-section-verification.json。
早期 verification.json 与 42 项日志是原 Solver 阶段的历史记录，本次文件作为新增阶段证据。

## 独立依据与覆盖

解析常数按预先列示的值比较，rtol=1e−14、atol=0；另与教材程序录入精度比较。
来源、单位与绝对容差详见 [材料截面理论](../theory/material-section.md)。
覆盖六种 E 单位、三种 R 单位、N/kN 缩放、非法/缺失值、溢出/下溢、修改实例后重校验、
计算标识，以及 Agent 的参数填入、来源保留、数值冲突、单位冲突、未知状态、不支持荷载和缓存失效。

## 产物与限制

- results/material_section_result.json：教材 E/ν/R → 全精度参数。
- results/agent_parameters_iv1_demo.json：真实旧识别记录的离线参数填入示例。
- examples/material_section_iv1.json：原始参数、单位、形状假定与教材出处。

自动参数工具只支持各向同性线弹性材料与实心圆截面。
本参数阶段当时未接入跨内荷载/截面；后续已扩展，当前状态见 [新验证](member-loads.md)。
离线运行本身不能作为模型 API 联调证据。

## 用户完成的 DeepSeek 真实调用

用户随后重新运行图片输入与参数工具指令，保存的记录已核对：

- calculate_material_section 返回 parameters_calculated，原始输入 E=210GPa、ν=0.3、R=0.05m，source=image。
- 六个 computed_fields 均为 python_tool，并关联同一个 calculation_id。
- record_observations 首次 invalid 后重新提交为 needs_information，保留未解决的结构信息与不支持项。
- 未出现 solve_space_frame 调用，未进行结构求解。

本次实际证据快照保存为 results/deepseek_material_section_verified.json。
这证明 DeepSeek 图片 → 参数工具 → 候选模型参数填入的流程已成功运行，
不表示整张图的支座、坐标、力矩和局部轴已全部识别准确，也不表示完整结构已求解。
