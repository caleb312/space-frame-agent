# GitHub 首次发布与后续更新

## 首次发布

1. 登录 GitHub，打开 https://github.com/new 。
2. Repository name 填 `space-frame-agent`，Owner 选择自己的账号。
3. Description 可填：`基于确定性 Python 求解器的结构分析工作台，支持平面/空间桁架和刚架。`
4. Visibility 选择 Public。
5. 本项目已经提供 README、.gitignore 和 MIT LICENSE，因此创建时不勾选这些初始化选项。
6. 点击 Create repository，保留生成的仓库地址。

上传前，运行 `python tools/package_release.py` 获取公开文件目录和 ZIP。脚本排除密钥、个人会话、教材原件、教材截图及缓存，保留第三方组件许可证，并执行可识别凭据格式的扫描。扫描不是对任意秘密的完整识别，发布前仍需核对文件清单。

不要直接上传开发目录中的所有文件，也不要只上传 ZIP 代替仓库源码。源文件应位于仓库根目录，便于查看、下载及后续修改。

通过 Git 或 GitHub Desktop 上传需要本人的 GitHub 登录授权。不要把密码、API 密钥或 GitHub 访问令牌发送到聊天中。

## 后续更新

保持一个明确的开发目录。每次改动后运行对应测试，核对提交文件，再提交有说明的版本并上传。密钥、个人会话及运行环境继续保持忽略。

`tools/package_release.py` 可重新生成发行包，每次输出一个新目录，不删除旧版本。GitHub 仓库维护源码，ZIP 可作为版本附件。

## 本轮发布检查

公开版本使用 MIT；Plotly 原始许可证另行保留。历史显示/会话测试改为依赖 `tests/fixtures/iv2_display_saved.json`，不依赖个人 results 文件。干净目录的三维测试会自行创建临时工作目录。计算算法和识读交互保持原有实现。
