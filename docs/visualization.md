# 三维结构与结果可视化

## 已实现

使用固定版本 Plotly.js 4.1.1，将图形库和当前模型/结果嵌入一个 HTML 文件，可离线打开。
不依赖 Plotly Python 包、不请求模型 API、不重新求解，不增加云部署设施。

- 可旋转/缩放的三维结构，节点/构件编号、支座、原始节点及杆上荷载、力矩方向。
- 原构形与放大变形叠加，可调放大系数、开关编号/荷载/支座/局部轴/截面标记。
- 位移模、全局位移/转角分量及局部 N/Qy/Qz/T/My/Mz 色带。
- XY、XZ、YZ 等比例二维投影；投影隐藏一个方向，不同节点可能重合，不改变三维模型或结果。
- 点击节点、构件和截面，或用面板下拉框查看位移、反力、真实杆端力和指定截面内力。
- 材料与截面参数面板；位移/反力/截面 CSV 和当前图 PNG 导出。
- 模型预览明确标为尚未求解，不能显示或编造位移/内力。

节点位移和支座反力沿全局坐标，杆端/截面内力沿各杆局部坐标。
展示转角及力矩复用 Python 的右手转换；原始 JSON 保持 course_clockwise，
截面仍采用 textbook_section 正号。箭头大小是示意；所有具体数值看悬浮提示或右侧面板。

## 现在查看已完成的例 IV-1

在项目目录运行：

```powershell
& 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' visualize_result.py --open
```

默认读取 results/agent_result.json，生成 results/agent_result.3d.html，然后在默认浏览器打开。
也可直接双击已生成的 HTML。此流程不需要 DeepSeek 密钥。

其他已保存结果：

```powershell
python visualize_result.py --input results/agent_result.json --output results/my_frame.html --open
```

只预览完整模型：

```powershell
python visualize_result.py --model examples/textbook_iv1_native.json --open
```

程序拒绝模型与结果哈希不一致的输入。图中反力、位移和内力不能与另一个模型的结果混用。

## Agent 接入

- 新命令 /view：完整模型未求解时打开结构预览；已有当前结果时打开结果视图。
- /confirm 求解成功后自动生成 .3d.html，再进入独立文字解释。
- 新工具 visualize_space_frame(mode=structure/results)：只使用程序持有的模型/结果，不接收 AI 绘图数值或任意文件路径。
- mode=results 没有结果时明确拒绝；可视化不会偷偷调用 Solver 或撤销确认。
- 修改程序后需要重启交互进程。浏览器打开动作只在用户 /view 或命令行 --open 时执行。

目前自然语言修改/追问仍按原 Agent 的保守策略撤销旧模型授权；仅查看当前三维图建议使用 /view。
新可视化工具的真实 DeepSeek 调用尚未联调，命令行生成与实际浏览器显示已检查。

## 变形重构

采用 Euler–Bernoulli 和 Saint-Venant 关系，从已求解真实杆端力与原荷载恢复杆内中心线位移/转角，
不只是用直线连接移动后的节点。具体公式见 theory/deformation.md。
集中位置及分布区间端点划分积分区间，三点 Gauss 精确积分当前常值荷载下的分段多项式。
两端固定杆虽节点位移为零，杆内受载弯曲仍能显示。放大仅用于显示，不改变计算结果。
只画中心线，不画实体截面扭转/应力分布，不进行设计安全或屈曲判断。
内力色带为有限采样，包括集中荷载处左右值；不是解析极值搜索。

## 文件

- src/solver/postprocessing.py：确定性杆内变形与内力采样。
- src/visualization/frame.py：数据校验、荷载图示、色带与离线 HTML 生成。
- src/visualization/viewer.html：交互界面与 Plotly 图形控制。
- src/visualization/assets：官方 Plotly 库、MIT 许可证、来源与 SHA-256。
- visualize_result.py：独立入口。
- results/agent_result.3d.html：真实例 IV-1 图形；frame_3d_preview.png 为浏览器实际导出的预览。

## 验证状态

新增 8 项后处理检查通过，全套 84 项通过，涵盖固定/悬臂均布梁解析曲线、
集中力/力矩曲线、轴向与扭转、斜杆/闭合刚架/参考资料题端点一致性、内力跳跃、哈希与数据不可变性。
浏览器已检查三维显示、My 色带及截面面板、平面投影和 PNG 导出，没有观察到控制台警告或错误。
UI 检查为人工操作记录，不宣称存在完整自动 UI 回归测试。

官方资源：[Plotly.js](https://github.com/plotly/plotly.js)、[三维线图接口](https://plotly.com/javascript/3d-line-plots/)。
