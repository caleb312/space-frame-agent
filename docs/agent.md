# 第一版 Agent：三家 API 的课程演示

范围限定为 DeepSeek、Qwen、GPT（OpenAI API）。使用现有模型，不训练模型。
三个服务商共享一个单 Agent 流程和一个确定性 Solver，HTTP 调用使用 Python 标准库。
不引入 Agent 框架。服务商标识和具体模型名称分开，模型名由账号持有人配置。

## 运行状态

### 2026-10-08 实际 IV-2 联调

以下是最新实际状态；后续段落保留此前各阶段记录。
开启 thinking 时用户遇到 32768 token 截断；同一模型关闭 thinking 后成功提交记录与参数工具。
首轮漏节点、误读连接和荷载作用点，第一轮人工纠正后仍保留错误坐标；
第二轮按参考资料输入表补充完整坐标、支座、局部轴和全局荷载分量后，模型通过核对。
用户 /confirm 完成 Solver、程序结果表、新版 DeepSeek 文字解释与三维 HTML 生成，并执行 /view 返回文件路径。
保存结果的位移/转角/五个跨中内力与参考资料在原容差内一致（显式保留 E04 原印刷及项目修正）；
原始荷载及实际作用位置独立力/力矩平衡检查通过，后处理杆端相容性检查完成。
本轮没有重新检查浏览器图形，不声称本例通过全自动识图或模型主动调用图形工具。
新版文字解释已实际引用程序统计，并区分全局节点/反力与局部杆端/截面；文字过长仍需改善。
证据见 [IV-2 实际核对](../results/deepseek_iv2_live_report.md) 与 [数值记录](../results/deepseek_iv2_live_comparison.json)。

已编写三家 Chat Completions 适配、工具调用循环、识别记录、候选模型编译、
逐项来源校验、未知信息与不支持荷载阻塞、用户确认入口、独立 Solver 调用及命令行交互。
支持为已配置视觉能力的模型构造本地图片输入。
用户已通过 DeepSeek 完成首轮真实图片请求；初次回答存在杆件方向和荷载位置误识别，不能视为识图验收通过。
Python 材料/实心圆截面工具已接入 Agent；原 42 项与新增 14 项参数/集成测试合计 56 项通过。
参数集成通过离线检查，不代表真实模型必然正确调用工具。详见 [参数验证](validation/material-section.md)。
用户已完成多轮 DeepSeek 图片核对，方向归一化后识别改善，部分支座、坐标和力矩仍需澄清。
用户已完成 DeepSeek 参数工具真实调用，Qwen/GPT 未联调。当前进一步扩展跨内荷载/截面，全套 76 项测试通过；
新接口的 Agent 编译与确认/求解通过离线检查，真实 DeepSeek 荷载流程联调待核对，见 [验证](validation/member-loads.md)。
三维可视化已接入，见 [使用说明](visualization.md)。新的图形工具与命令行生成已离线核对；
网页聊天界面和图片全自动建模质量验收仍未完成。

## 配置与启动

在项目目录运行。读取 PowerShell 环境变量；GPT 额外支持项目 `.env.local` 中的 `OPENAI_API_KEY/MODEL/BASE_URL/VISION/TIMEOUT/EXPLAIN_TIMEOUT`，环境变量优先，默认模型 `gpt-4.1`、视觉开启。其他供应商不自动读取 `.env`；
`.env.example` 仅为字段模板。不要把真实密钥提交到仓库或发给聊天助手。

例如 DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY = '在本机填写密钥'
$env:DEEPSEEK_MODEL = '在本机填写可调用的模型名称'
python agent_main.py --provider deepseek
```

Qwen 使用 `QWEN_API_KEY`、`QWEN_MODEL`，运行 `python agent_main.py --provider qwen`。
GPT 使用 `OPENAI_API_KEY`、`OPENAI_MODEL`，运行 `python agent_main.py --provider gpt`。
服务商默认地址可用对应 `*_BASE_URL` 改写，Qwen 尤其要与账号所属区域相符。
网络等待默认 180 秒，可通过 `DEEPSEEK_TIMEOUT`、`QWEN_TIMEOUT` 或 `OPENAI_TIMEOUT` 设置秒数并重启程序。
这是单次网络操作等待时间，不是整个多轮 Agent 的总时间限制。
DeepSeek 显式设置 `DEEPSEEK_MAX_TOKENS=32768`、`DEEPSEEK_THINKING=enabled`、
`DEEPSEEK_REASONING_EFFORT=low`，避免依赖服务默认高推理强度。
仍遇到长度截断时，可增大 MAX_TOKENS，或为排查设置 THINKING=disabled；均需重启。
程序分别显示 length、content_filter 与服务中断原因，截断内容不会当作完整工具参数执行。
IV-2 的实际请求在 enabled 时发生 32768 token 截断，改为 THINKING=disabled 后成功调用工具。
这是本例实际结果，不保证其他输入不会截断或保持相同识图质量；不用自动提高输出预算或重新请求。
只允许 HTTPS。选用支持 Chat Completions 工具调用的型号；并非每个型号均支持该协议。

图片演示：先确认型号同时支持图片输入与工具调用，再启用对应 `*_VISION=1`：

```powershell
$env:QWEN_VISION = '1'
python agent_main.py --provider qwen --image 'C:\本机图片目录\结构.png'
```

图片与输入文字会发送到所选 API。支持 PNG、JPEG、WebP，限制 10 MiB。
上传前用 Pillow 的 exif_transpose 将 EXIF 方向标记落实到像素，重新编码首帧后发送；不覆盖原图片。
手机照片即使在看图软件里正向显示，原始像素仍可能横着或竖着，不能直接发送原始字节后依赖 API 自动旋转。
启动时显示图片绝对路径，识别记录的 image_input 保存原始/上传尺寸、方向标记及两份哈希，方便排查实际输入。
本次用户的照片原始尺寸为 1756×4096、EXIF Orientation=8，方向校正后的尺寸应为 4096×1756。
方向元数据已读出，修改通过语法检查；校正后真实 API 识别效果尚待用户重新运行验证。
算法依据：[Pillow exif_transpose](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.exif_transpose)。
VISION 是人工确认的能力开关，不是程序自动识别供应商型号能力。
不支持图片的配置会在发送前明确拒绝，不自动切换到其他收费服务商。
第一张图片随首轮自然语言发送；当前 CLI 后续图片需要重新启动传入。

## 交互

1. 用文字描述结构，或说明首轮图片中的问题。
2. Agent 可询问坐标、材料、截面、荷载、支座或局部轴等缺失信息。
3. 工具 `record_observations` 提交识别记录；Python 从逐项事实与独立荷载清单生成候选模型。
4. 默认显示节点、约束、构件连接示意、每项荷载及阻塞原因；输入 `/model` 查看完整事实来源与候选模型。
5. 信息正确后输入 `/confirm`，程序确认当前模型哈希并直接调用 Solver。
6. Agent 解释确定性结果；原始结果保存在 `results/agent_result.json`。
7. `/quit` 退出。

识别记录自动保存为 `results/agent_result.recognition.json`，供用户发送给开发者核对。
任何新的文字消息或识别记录更新都会撤销计算授权并清除当前结果。
当前 CLI 为简化实现，结果追问也需再次确认；后续可区分解释追问与修改指令。
模型校验通过只说明格式与基本连接合法，不证明图片理解和物理建模正确。
若 Solver 失败则返回诊断，不添加支座，不使用 LLM 补出数值。

## 代码分工

- `src/agent/providers.py`：仅三家配置、HTTP 请求、图片编码；保留 DeepSeek 工具续调用所需 reasoning_content，不显示该字段。
- `src/agent/recognition.py`：识别记录字段、来源与未知值规则、候选模型编译、荷载分类及摘要。
- `src/agent/runtime.py`：识别、参数、求解与三维四个工具，识别/确认/结果状态、最多八轮工具调用。
- `src/solver/parameters.py`：参考资料 E/ν/R 参数公式与单位换算，独立于 API。
- `agent_main.py`：终端用户操作和结果保存。
- `src/solver/`：沿用已验证的确定性计算实现。

`confirm()` 没有暴露给 AI，只有应用侧用户操作可以调用。
`solve_space_frame` 工具不接受模型参数，只能使用程序里已经确认的模型。
理论提示覆盖适用范围、单位、六自由度、课件转角/力矩正号以及 J 的含义；
不把十份课件全文塞入每次请求，不让模型复算公式。

## 识别与建模规则（本次改进）

1. 原始 observations 保留题面材料、尺寸等文字及原单位；不能自动改写成计算参数。
2. facts 每项使用路径、值、状态、来源和证据。例：`/nodes/0/coordinates` 表示模型数组中第一个节点的坐标，不是节点编号0。
3. `uncertain` 和 `needs_calculation` 的 value 必须为 null，否则程序拒绝。未知量不允许临时数值、全 false 支座等占位。
4. 项目约定只能填 `/convention`，不能拿“惯例”补齐材料、截面、几何或荷载。
5. 原图的每项荷载独立记录：节点力、节点力矩、杆上集中力、杆上集中力矩、杆上分布荷载或待确定荷载。
6. 荷载必须注明目标、位置或区间、原始文字和信息来源。无法确定方向/位置保持 null；分布荷载不默认全跨。
7. facts 不可直接写 loads/member_loads；程序从荷载清单分类生成。杆上荷载需明确全局/局部坐标系和位置，均布力需完整起止区间及 uniform 类型，否则阻止求解。
8. 程序检查所有必填字段、向量内的 null、荷载盘点完成状态、单位、类型/分量一致性及结果需求，之后才进行已有 StructuralModel 校验。
9. 跨中/指定截面内力保留为 section_forces，不能以杆端力替换需求。
10. 用户澄清后需要提交完整更新记录。没有本轮结构化记录时，文字回答不能作为可计算模型。

用户回传的改进后首轮记录仍有照片方向与 4kN 作用点误识别；本轮尚未通过识图质量验收。
后续进一步加入提交前箭头尖端、分布区间、支座剖线、尺寸界线与阅读方向二次核对规则。
同一字段的“待确定/仍有未知项/缺失”合并为一条；终端再按节点坐标、约束、局部轴及待程序计算参数分组，
详细记录与问题保留在 `/model` 和 recognition.json。摘要把 None 显示为“待确定”。
这次只执行修改文件的语法检查；新提示的真实识图效果仍需再次核对。

已实现 calculate_material_section：由 Python 换算并计算 E、G、A、Iy、Iz、J。
原始事实的派生字段保持 null/needs_calculation；候选模型填入工具结果，computed_fields 单独记录来源。
跨内荷载、固定端力、指定截面内力与三维图现已接入。
节点连接示意为文本拓扑图，不是按坐标绘制的三维结构。

限制：程序可以校验记录格式和记录中声明的未知项、不支持项，但不能独立证明 AI 已识别全部荷载，
也无法识别一个被 AI 错误标为 observed 的猜测值。因此仍必须由用户对照原图审核，不能宣称已消除误识别。
改进提示没有包含 IV-1 的参考资料答案，避免把已知答案伪装为通用图片识别结果。

方向校正后的用户回传记录已能正确描述水平/竖直构件及部分荷载区间，但出现 node_i/node_j 使用数组下标、
漏填构件 id 和材料/截面管理编号的问题。现增加不完整模型的编号唯一性、连接引用、荷载目标和证据编号一致性检查。
即使坐标/截面仍缺失，也不会跳过这些检查。程序不会静默把下标映射成节点 id；
同一轮最多要求模型修正两次编号/引用，仍失败则保留记录并报告阻塞，不会求解。
材料/截面的管理编号可用 assigned/agent_label 显式分配及引用，仅允许编号字段，不允许借此补物理参数。
新的编号修正流程只通过语法检查，尚无真实 API 核对证据。

## 材料与截面参数工具

### 求解后文字解释

求解成功后程序立即保存原始 JSON，并显示/保存程序结果表，再发送一次独立文本请求解释结果。
解释不带图片、长识别历史或工具，不能重新建模、撤销确认或再求解。
DeepSeek 解释请求关闭 thinking，输出预算 4096，默认总等待上限 60 秒，
可用 DEEPSEEK_EXPLAIN_TIMEOUT 设置。识图/建模的原配置保留。
本地超时只结束等待，不能保证服务端同步取消；同一客户端不启动第二条尚未结束的解释请求。

若解释超时、空白或失败，已保存结果仍有效；当前会话可用 /explain 重试。
也可以退出交互后，只解释已有文件，不重新识图或求解：

```powershell
python explain_result.py --provider deepseek
```

只生成结果表、不请求 API：`python explain_result.py --local`。
默认读 results/agent_result.json，输出 agent_result.summary.md 和成功时的 agent_result.answer.md。
结果表/解释输入的节点转角及节点/杆端力矩由 Python 换为右手正号；
原始 JSON 仍为 course_clockwise，截面保持 textbook_section，不把两类符号混用。
本次改进已做语法检查和实际保存结果的本地表格生成；新的解释 API 调用尚未核对。

用户随后成功运行独立解释请求，生成 agent_result.answer.md；数值引用与结果一致。
校阅发现了位移/转角混比、全局节点与局部杆端坐标混写、由轴力大小推断受力主导模式的问题。
已在解释指令中明确禁止这些推断，新增 Python 分组统计供模型引用；新版在 2026-10-08 的 IV-2 实际 API 回答中已核对。
原始解释保留，校阅版为 results/agent_result.answer.reviewed.md，统计为 agent_result.explanation_statistics.json。

多行补充信息请先输入 /paste，再粘贴全文，最后单独输入 /end。
这些行合并为一条请求，避免 input() 将每行当新一轮对话而反复调用参数工具。

支持各向同性线弹性材料与实心圆截面，输入原始 E、ν、R 及单位，不套用于圆管或一般截面。
E 输入 Pa/kPa/MPa/GPa/N/m^2/kN/m^2，R 输入 m/cm/mm，输出沿用 N_m 或 kN_m。
公式与参考资料 Jy/Jz/Jp → Iy/Iz/J 的映射见 [材料截面理论](theory/material-section.md)。

独立运行，不需要 API 密钥：

```powershell
python parameters_main.py examples/material_section_iv1.json
```

结果保存到 results/material_section_result.json。自动工具保持全精度计算，不模拟参考资料录入值的舍入。

Agent 指令示例：

> 识别结构图，调用 calculate_material_section 根据题面 E、ν、R 计算材料与实心圆截面参数，然后提交完整识别记录，不进行结构求解。

用户明确说“不要计算”时只识别；“不做结构求解”允许仅计算参数。
工具需 material_id、section_id、material_type、section_type、E、E_unit、nu、R、R_unit、units、source、evidence。
程序保存原值和计算哈希，按编号填入，不依赖 AI 抄写派生值；用户新消息撤销旧缓存，更新后重新计算。
参数非法、单位不一致或识别值冲突会阻止求解；uncertain 不因有一次工具计算而被覆盖。

results/agent_parameters_iv1_demo.json 是本次离线示例：读取用户的真实旧识别记录，
使用已核对的参考资料 E/ν/R，未发送 API 请求，不能称为新一轮 DeepSeek 识图结果。

解释阶段通过指令要求数值引用当前工具结果，完整原始结果同时保存供核对。
当前尚无自动校验每个自然语言数值引用的机制，不能宣称彻底消除模型幻觉。
材料单位转换、解释用分组统计及三维生成已由 Python 接入；其他派生计算不能由 LLM 自行完成。
后续三维图应由程序根据已确认的结构模型和 Solver 结果绘制。

## 给老师的三个演示案例

### 新荷载与截面接口的核对指令

重启 Agent 后输入：

> 识别结构图，调用材料参数工具，保留杆上集中荷载和分段均布荷载，明确它们使用全局还是局部分量，并登记各杆跨中截面请求；提交模型让我确认，先不要结构求解。

当前识别记录新增可选字段：

- 杆上荷载 coordinate_system=global/local；分布荷载 distribution=uniform/other。other 仍保留，但不可求解。
- section_queries：截面 id、element_id、location=midspan/position、position、side=left/right、source、evidence。
- midspan 位置由 Python 计算；position 是距构件 i 端的实际距离。未知位置或方向保持 null。

信息足够时模型可确认，输入 /confirm 计算。结果包含真正杆端力及 sections，不用 AI 等效荷载或恢复内力。
未解决的节点坐标、支座、局部轴或力矩方向仍需用户/参考资料明确补充；本次功能扩展不等于图片所有信息都已正确识别。
需要独立检查 Solver 可运行 main.py examples/textbook_iv1_native.json；其输入明确来自参考资料，不是图片提取结果。

1. DeepSeek：明确的文字结构输入 → 草稿 → 确认 → Solver → 解释。
2. Qwen：选用具备视觉及工具能力的型号，图片 + 文字 → 提取 → 追问 → 确认 → 求解。
3. GPT：使用同一结构或另一张图片，展示切换供应商后仍调用同一 Solver。

演示记录建议保留服务商、具体型号、日期、结构模型、工具状态和原始结果。
三家都能支持文字演示，不要求三家都完成图片演示。
真实联调时需核对正常建模、缺失参数、无法识别图片、校验失败、结构不稳定和工具输出解释。

## 官方接口依据

核对日期：2026-10-07。运行时能力以账号可调用的具体型号为准。

- [OpenAI Chat Completions](https://developers.openai.com/api/reference/chat-completions/overview)：消息、图片输入及函数调用入口。
- [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)：tools/tool_calls、工具结果与 reasoning_content。
- [Qwen Function Calling](https://help.aliyun.com/zh/model-studio/qwen-function-calling)：兼容接口与工具调用。
- [Qwen VL 兼容接口](https://help.aliyun.com/en/model-studio/qwen-vl-compatible-with-openai)：视觉模型接口。
