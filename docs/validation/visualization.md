# 三维可视化验证记录

日期：2026-10-08。接续前一天已保存的代码与实际例 IV-1 结果。

## 数值验证

新增 8 项后处理测试，全套 84 项通过：0 失败、0 错误、0 跳过。
日志为 visualization-tests.txt，计数、源码 SHA-256 和实际采样误差为 visualization-verification.json。
验证采用独立梁解析曲线、轴向/扭转解和节点端点约束，不只检查能生成图。

真实 IV-1 模型的重构端点最大平动误差约 8.67e−18 m，转角误差约 3.47e−18 rad。
可视化不重新求解，不修改模型或 Solver 数值；错配哈希和端点不一致均拒绝绘图。
数据面板与 CSV 的转角/力矩由 Python 转为教材附录右手约定，原始 JSON 保持课件约定。

## 实际运行与浏览器检查

- visualize_result.py 成功读取 results/agent_result.json，生成 agent_result.3d.html。
- 结构预览成功生成 textbook_iv1.model3d.html，明确未求解，不显示位移/内力数值。
- FrameAgent.dispatch 的 visualize_space_frame 离线调用返回 visualized。
- 本机浏览器实际显示结构、放大中心线变形、支座及原始荷载。
- My 色带与截面1侧栏值一致；XZ 等比例投影已显示；PNG 导出返回成功并保存文件。
- 已观察的浏览器控制台 error/warn 列表为空。Cloud Share 按钮已移除。

浏览器检查为人工验证，不宣称完整自动 UI 测试或所有浏览器都已覆盖。
新工具的真实模型 API 调用尚未核对；/view 和独立 CLI 不依赖模型 API。

## 交付文件

- results/agent_result.3d.html：真实已确认、已求解的 IV-1 模型。
- results/textbook_iv1.model3d.html：教材模型的未求解结构预览。
- results/frame_3d_preview.png：从实际浏览器导出的三维预览。
- docs/visualization.md：入口、交互、结果约定与范围。

图形库和数据嵌入 HTML，可直接双击或通过 --open 查看；临时检查服务器不属于项目部署设施。
本次未加入云部署、实体截面渲染、应力/设计判定或连续极值搜索。
