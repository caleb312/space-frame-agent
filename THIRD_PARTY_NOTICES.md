# 第三方组件与资料来源

项目代码按根目录 MIT LICENSE 发布。第三方组件保留各自许可证：

- 离线三维查看使用 plotly.js；原始版权声明及许可证位于 `src/visualization/assets/LICENSE.plotly`，版本和来源见同目录 `manifest.json`。打包的 JavaScript 内部声明一并保留。
- NumPy、Pydantic 和 Pillow 通过 `requirements.txt` 安装，遵循其各自许可证；本发布包不包含其安装文件。

计算依据为阎军、杨春秋《计算结构力学》（2014）及项目理论说明。来源、页码、符号转换和项目核对记录见 `theory.md`、`docs/theory/`。开源包不附教材 PDF、课程课件或教材截图；文档提及的这些原件需由读者自行合法获取。项目许可证不授予这些原始资料的使用权。

自定义演示题与教材算例在示例文档中注明。已验证算例不意味着所有结构条件均受支持，具体限制见 README。
