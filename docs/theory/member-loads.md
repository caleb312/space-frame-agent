# 跨内荷载、固定端力与指定截面内力

## 本次扩展范围

用户授权继续实现杆上集中荷载、分段均布荷载和指定截面内力。
新增通用接口处理直杆上的三分量集中力、三分量集中力矩，以及任意区间的常值三分量分布力。
力矩仍为课件负右手分量，荷载可用全局或局部坐标给出，均布力按单位实际杆长计。
保留节点荷载接口。分布力矩、三角形/其他变强度荷载、温度、释放、支座非零位移和非线性尚未支持。

## 来源与适配

- 教材第五章和原课件：直杆线弹性刚度、节点六自由度、局部/总体转换与整体组装。
- 教材附录 IV，印刷 p196～197 / PDF p205～206：KGReactionForce、集中力、均布力、轴力/扭矩和集中弯矩的固定端反力。
- 印刷 p198～199 / PDF p207～208：KGInternalForce、KGCtlInternalForce，将杆端位移项、荷载固定端项及截面静力叠加。
- 印刷 p186～189 / PDF p195～198：例 IV-1、IV-2 原题及印刷答案，用于完整对照；IV-2 原印刷差异仍保留 E04。
- 标准理论补充：[TU Delft 有限元教学材料的梁弱式与形函数](https://interactivetextbooks.citg.tudelft.nl/computational-modelling/structural_linear/rod.html)。
  本模块使用 Hermite 虚功积分统一生成荷载向量，而不逐字复制教材所有荷载类型的分支。
  教材前缀区间 [0,b] 的均布力推广到任意 [a,b]，由线性叠加/积分得到，明确为实现推广。

教材附录用右手旋转分量，核心沿用已验证的课件负右手分量。
原验证适配器继续作为独立交叉检查；新的主求解器直接接收原始杆上荷载，不再要求用户提供等效节点力。

## 一致节点荷载

令 s=x/L，三次弯曲形函数为：

```text
Ni = 1−3s²+2s³
Hi = L(s−2s²+s³)
Nj = 3s²−2s³
Hj = L(−s²+s³)
```

轴向平动和扭转用 [1−s,s] 插值。
在项目课件转角下，局部横向位移：

```text
v = [Ni,−Hi,Nj,−Hj] · [uyi,rzi,uyj,rzj]
w = [Ni, Hi,Nj, Hj] · [uzi,ryi,uzj,ryj]
rz = −dv/dx
ry =  dw/dx
```

集中力按 F·δu 的虚功得到 p_eq，集中力矩按 M_course·δr_course 的虚功得到 p_eq。
任意常值分布力区间 [a,b] 的 p_eq 为其形函数积分。
形函数最高三次，三点 Gauss 对该常值荷载积分精确，无网格细分或人为补节点。
全局输入向量先用 Qᵀ 转到局部，随后 T12·p_eq 累加到完整总体 P，包含受约束节点的等效量。

## 固定端力与求解流程

```text
K U = P_node + Σ(T12 p_eq)
S_local = k_local (T12ᵀ U_element) − p_eq
R_global = K U − P_full
```

−p_eq 是零位移下的固定端作用力。
只使用 k_local·u_local 会丢失原跨内荷载的固定端项，不能当作真实杆端力。
LDLT、四分块总刚、自由度对号、零位移约束等原课件流程保留。
P 的整体合力/力矩等于原始荷载合力/力矩，测试分别核对局部和总体平衡。

## 截面静力及正号

截面位置 x 从构件 i 端起算。取 i 端到截面的左侧部分：

```text
F_left = S_i_force + 左侧原荷载合力
M_left_RH = −S_i_moment_course − [x,0,0]×S_i_force
            + Σ([荷载位置−x,0,0]×F_load − M_load_course)
```

均布荷载仅取区间与 [0,x] 的交集，合力作用于该交集的形心。
正面截面物理右手作用力为 [−F_left,−M_left_RH]。
教材附录截面输出需再乘对角符号 [1,−1,−1,1,−1,1]，得到：

```text
[N, Qy, Qz, T, My, Mz]
```

该截面正号与节点/杆端的课件广义力不同，结果明确标记 section_force_convention=textbook_section。
结果同时保存构件两端编号与局部基 Q，避免不同局部轴的分量被直接比较。
集中力位置的 shear/axial 存在跳跃，集中力矩位置的 moment/torsion 存在跳跃；
side=left 不计恰好位于 x 的杆上集中荷载，side=right 计入，默认 right 与教材截面表约定一致。
side 是对注册杆上荷载的不连续位置取极限；节点荷载作为节点作用，不重复加入杆内荷载。

## 输入格式

```json
{
  "member_loads": [
    {"element_id": 1, "kind": "uniform_force", "coordinate_system": "global",
     "components": [0,0,-0.8], "interval": [0,3]},
    {"element_id": 1, "kind": "point_force", "coordinate_system": "global",
     "components": [4,0,0], "position": 3}
  ],
  "section_queries": [
    {"id": 1, "element_id": 1, "location": "midspan", "side": "right"},
    {"id": 2, "element_id": 1, "location": "position", "position": 3, "side": "left"}
  ]
}
```

上例只是附加字段，仍须提供完整节点、构件、材料、截面、支座及局部轴。
point_force 的三分量为 F，point_moment 的三分量为课件 M；uniform_force 为力/实际杆长。
position 或 interval 都是实际杆长距离，不是照片像素、全局坐标分量或投影长度。
集中位置须 0≤x≤L，均布区间须 0≤a<b≤L。几何、引用、维度和有限值由 StructuralModel 校验。
midspan 请求的真实位置由程序按 L/2 计算，不让 AI 计算。

## 验证

独立检查全轴向/弯曲/扭转荷载合力矩、虚功多项式积分、两端固定梁反力、
悬臂全跨/部分均布荷载位移、跨内集中力/力矩的位移和截面跳跃、
全局/局部转换、端点反向、均布区间分段叠加、N/kN 缩放及每杆静力平衡。
两道教材题直接调用通用 Solver，与旧适配器的 U/R/S/截面力及教材表值分别对照。
教材误差容限沿用原记录；原始 IV-2 的错误印刷值不被改成“已匹配”。
