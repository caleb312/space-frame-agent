# 可复现算例

## E01：三维悬臂

`cantilever_3d.json`：节点 10 完全固定，节点 90 位于 (2,0,0)，局部 y=(0,1,0)。E=2e8 kN/m²，G=8e7 kN/m²，A=0.01 m²，Iy=2e−5 m⁴，Iz=4e−5 m⁴，J=1e−5 m⁴。端荷载为 [3,−5,7,2,−4,6]，力 kN、力矩 kN·m，力矩与转角遵循项目正号。

独立参考：积分轴向、弯曲曲率与扭转关系，悬臂自由端满足：

```text
ux = Fx L/(EA)
uy = Fy L³/(3EIz) − Mz L²/(2EIz)
uz = Fz L³/(3EIy) + My L²/(2EIy)
rx = Mx L/(GJ)
ry = Fz L²/(2EIy) + My L/(EIy)
rz = −Fy L²/(2EIz) + Mz L/(EIz)
```

基端反力由静力平衡得 `[−Fx,−Fy,−Fz,−Mx,−My−LFz,−Mz+LFy]`。数值参考保存于 `cantilever_3d_reference.json`，由上述解析公式生成，未采用 Solver 输出作为参考答案。此为标准理论补充算例，不标作参考资料原题。

## E02：三维折杆

`bent_frame_3d.json`：在 E01 后连接节点 300=(2,1,1)，尖端承受相同六分量荷载。两单元是连续刚接空间梁，均输入局部 y 参考向量。

独立参考由 `tests/helpers.py` 的力法积分计算：对六个单位尖端荷载求各杆截面的 N、T、My、Mz，积分截面内力乘积得到 6×6 柔度，再乘荷载。该参考不调用 Solver 的单刚、坐标转换或组装函数。测试还使用独立静力平衡检查各单元两端全部 12 项内力。

测试中的附加案例将尖端 uy 约束，用力法的位移协调方程求冗余反力，从而核对超静定情形。此为独立补充算例，不标作参考资料原题。

## C06：课件四杆平桁参考

`course_four_bar_reference.json` 保存 C06 p15 的约化总刚、p4 的荷载及勘误后的位移，作为 LDLT 分解和回代的课程回归依据。它不是将四杆平桁当作空间刚架计算，也不意味着完整桁架 Solver 已实现。

C09 的平面单刚在 `tests/test_geometry_stiffness.py` 中用逐项课程矩阵核对；空间刚架通过约束面外自由度核对平面悬臂退化。

## 运行

在项目根目录运行：

```powershell
python main.py examples/cantilever_3d.json --output results/cantilever_3d_result.json --diagnostics
python main.py examples/bent_frame_3d.json --output results/bent_frame_3d_result.json --diagnostics
python -m unittest discover -s tests -t . -v
```

比较容限：解析位移/转角采用 rtol=1e−10、atol=1e−12（m 或 rad）；力/力矩采用 atol=1e−8（kN 或 kN·m）。每项指标按自己的单位比较，不把力与位移混在同一个误差指标中。

参考资料原件已归档。两道附录空间算例的显式适配验证见下文，不能把补充解析案例改标为参考资料原题。

## E03：闭合空间框架

`closed_frame_3d.json`：8 节点、10 杆件，4 个完全固定基端、4 个非共面上部节点；包含柱、上部闭合环与交叉杆，以及两种材料和两种主轴不等刚度截面。交叉杆中间没有定义相交节点，因此不在交叉处自动连接。所有局部轴方向均显式输入。

独立参考为 `tests/energy_reference.py`：从 Hermite 形函数二阶导数构造应变矩阵，在每杆上用三点 Gauss 积分形成能量方程，再用 NumPy 求解。验证工具不调用生产 Solver 的单刚、组装、转换或 LDLT 函数；结果恢复也直接积分虚应变功。它是同一梁理论的独立算法核对，不代表外部软件或参考资料原题验证。

`tests/test_closed_frame.py` 比较全部节点位移、约束反力和每根杆的 12 项局部端力；同时检查总体旋转/平移、节点重编号/输入顺序、杆端反转及 N/kN 单位一致性。该算例为明确标记的理论补充。

```powershell
python main.py examples/closed_frame_3d.json --output results/closed_frame_3d_result.json --diagnostics
```

## B01：参考资料例 IV-1 与 IV-2

原题位于印刷 p186～189（PDF p195～198）。`textbook_iv1_original.json`、`textbook_iv2_original.json` 保存原始坐标、连接、参数、荷载及印刷答案。`*_equivalent_nodes.json` 是明确转换后的第一版节点模型。

```powershell
python -m examples.validate_textbook --output results/textbook_verification.json
```

该命令显式处理附录旋转约定和跨内荷载的节点等效，调用核心 Solver，再恢复原题指定截面力。不是通用跨内荷载功能。直接对 equivalent_nodes.json 运行 main.py，只能得到该节点模型的杆端位移项；真实原题端力另由验证适配器补入固定端项。

IV-1 原表对照通过；IV-2 两项印刷 x 位移不满足原题方程，原值保留，采用单独记录的 E04 对照值。详见 `docs/theory/textbook-audit.md`。书中表值比较采用 m/rad/kN/kN·m 各自注明的绝对容限，独立解析测试仍使用更严格容限。

## 中期新增演示模型

- plane_truss_triangle.json / .png：三杆三角桁架，原生2自由度；10kN向下，支座10铰、20竖向约束。解析轴力：底杆+5kN、两斜杆−5√2kN。
- plane_frame_cantilever.json / .png：2m平面悬臂，原生3自由度；E=200GPa、A=.01m²、I=.0001m⁴，10kN向下。解析自由端uy=−PL³/(3EI)、顺时针rz=PL²/(2EI)。
- space_truss_three_bar.json：三根正交杆的自由节点，原生3平动自由度；解析U=P/(EA)，轴力由端序投影确定。

均为自定义简单验证算例，不是原书编号题。理论矩阵按项目计算规则；独立解析参考见tests/test_specialized_solver.py。运行python -m examples.prepare_midterm_examples生成结果表及离线图；工作台保存列表可直接打开查看，不消耗API。
