# Space Frame Agent · 结构分析工作台

个人计算结构力学课程项目：从图片和文字整理可核对的结构模型，经用户确认后调用确定性 Python 求解器，展示位移、反力、内力及三维图。

## 支持的结构

| 类型 | 每节点自由度 | 参数 | 荷载 |
| --- | --- | --- | --- |
| 平面桁架 | ux、uy | E、A 或题给 EA | 节点力 |
| 空间桁架 | ux、uy、uz | E、A 或题给 EA | 节点力 |
| 平面刚架 | ux、uy、rz | E、A、I | 节点力/力矩、杆上集中力/力矩、区间均布力 |
| 空间刚架 | ux、uy、uz、rx、ry、rz | E、G、A、Iy、Iz、J | 节点力/力矩、杆上集中力/力矩、区间均布力 |

采用线弹性直接刚度法。各类结构使用对应自由度和求解器。结构计算、材料换算和数值统计由 Python 完成，语言模型负责识读、整理和解释。

不支持几何/材料非线性、动力分析、屈曲、杆端释放、刚臂、剪切变形、非零支座位移及温度荷载。图片识别尚需人工核对，不能直接作为工程设计依据。

## Windows 一键启动

1. 安装 Python 3.10 或更新版本，安装时勾选 **Add Python to PATH**。
2. 下载仓库 ZIP 并解压，或克隆仓库。
3. 双击 **启动网页.cmd**。首次运行自动创建 `.venv` 和安装依赖，需要联网。
4. 工作台默认打开 http://127.0.0.1:8765/ 。演示期间保留启动窗口。

重复双击会打开已有工作台。依赖完整后无需重复下载；不要求安装 Codex。详见 [启动说明](启动说明.md)。

图片识别、自然语言建模与追问需要在“模型设置”配置 Qwen、DeepSeek 或 OpenAI 兼容接口。API 费用及模型访问权限由使用者账号提供。Qwen/DeepSeek 页面密钥仅存服务内存，真正重启服务后需重新输入；密钥不会随本项目发布。

## 工作流程

1. 导入图片或 Markdown 题目，明确结构类型。
2. 核对节点、构件、支座、材料截面、荷载及单位；通过自然语言纠正。
3. 模型完整且校验通过后，点击确认计算。
4. 查看数值结果、三维图和计算概要。修改模型后需要重新确认。

识别不确定的信息会阻止求解。计算概要直接读取 Solver 结果，不等待模型 API。已有计算与会话保存在本机 `results`。

## 无 API 的计算与验证

在已安装依赖的环境中：

```powershell
python -m pip install -r requirements.txt
python main.py examples/plane_truss_triangle.json --output results/triangle.json
python -m examples.prepare_midterm_examples
python -m unittest discover -s tests -t . -v
```

使用一键启动创建的环境时，可将 `python` 换成 `.venv\Scripts\python.exe`。`prepare_midterm_examples` 生成三个自定义解析算例的演示结果，可在工作台打开。

求解器可以独立运行，不依赖 API。测试包含解析解、独立方程、课程参考结果、坐标转换、平衡关系、错误输入和工作台状态校验。界面状态测试使用的保存样本不是独立数值参考答案。

## 计算约定与资料

依据阎军、杨春秋《计算结构力学》（2014）及项目理论记录。原始转角和广义力矩采用 `course_clockwise`；界面转角和节点/杆端力矩转换为右手显示，截面内力保持教材正号。桁架轴力受拉为正。不同坐标系与符号约定在结果中注明。

- [理论与约定](theory.md)
- [原生桁架及平面刚架依据](docs/theory/planar-truss-frame-2014.md)
- [项目核对与勘误](docs/theory/errata.md)
- [示例说明](examples/README.md)
- [网页说明](docs/web-interface.md)
- [三维说明](docs/visualization.md)

已通过具体算例不代表图片全自动识别或所有结构条件均已验收。文档中的历史测试数量对应当时版本。

公开仓库不附教材 PDF、课件原件和教材截图，理论文档中引用的原件需自行合法获取。

## 项目结构

- `src/model`：结构模型和输入校验。
- `src/solver`：确定性结构计算。
- `src/agent`：模型接口、识读与纠正。
- `src/web`：本地工作台。
- `src/visualization`：计算结果与三维后处理。
- `examples`、`tests`：示例及验证。
- `docs`：理论来源、限制和使用说明。
- `tools/package_release.py`：重新生成公开源代码目录和 ZIP。

## 开源与更新

项目代码采用 [MIT](LICENSE)。第三方组件和引用资料见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

密钥、个人会话、运行环境和原始教材不纳入版本管理。后续修改应先完成相应验证，再提交到仓库。开源仓库提供源码；完整工作台需要本地 Python 服务运行。
