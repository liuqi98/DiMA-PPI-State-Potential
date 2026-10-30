DiMA-PPI 完整会议论文初稿（含结果）

主文件
  dima_iscas2026_full.pdf：完整英文会议论文，5 页 IEEE 双栏；正文在前 4 页，第 5 页仅参考文献。
  dima_iscas2026_full.tex：主入口；Overleaf 中将其设为 Main document。
  dima_method_conference.tex：会议篇幅的方法正文。
  dima_results.tex：实验设计、结果、讨论与结论。
  dima_iscas2026_full_source.zip：完整源文件、图、数据汇总与来源清单。

模型图
  图 1：规范化四状态输入、共享多模态图编码器、势能差输出与推理复用。
  fig_model.pdf：用于论文的矢量图；fig_model.svg：可编辑矢量源图。
  fig_model.png：450 dpi 预览；draw_model.py：可重复生成三种格式。
  模型图已加入第 2 页，结果图顺延为图 2。

新增结果
  表 I：DSSA-PPI、DiMA-PPI 与 5 组匹配控制/消融；
        S1129 的 PCC、Spearman、RMSE，5 次重复/种子均值与样本 SD。
  表 II：反向、路径、三角环和四状态环的一致性残差。
  图 2a：9,904 条端点推导未见边的原生预测与 WT 端点组合对照。
  图 2b：固定 StaB-ddG 权重的 Monte Carlo 状态身份实验。
  正文：状态描述符消融与复合物 bootstrap 区间；跨模型结果；
        规范状态分数复用的调用量与单次固定硬件推理耗时。

数据集命名
  S1129 保留公开基准名称。
  自建派生评测集命名为 PPI-Path92（Protein–Protein Interaction Mutation
  Path Benchmark），来自 S1129 同位点多突变记录。
  ppi_path92/PPI-Path92_edges.csv：9,904 条有向突变边，含端点实测值、
  推导标签、外层 fold 与原始 source_row，已去除模型预测列。
  ppi_path92/PPI-Path92_panels.csv：92 个面板、27 个复合物、793 个状态。
  ppi_path92/dataset_card.json：构建规则、标签定义、来源与文件校验值。
  名称已同步至摘要、正文、表 II、结果图与图注。

数据来源与解释
  此次使用已经保存的实验汇总，没有新训练模型或重新跑推理实验。
  evidence/ 存放实际用于制表绘图和核对正文的汇总文件。
  evidence/manifest.json 记录原始项目相对路径与 SHA-256。
  build_results.py 直接读取 evidence/ 重建两张 LaTeX 表和结果图。
  比较结果按数值直接报告：soft 的直接 PCC 略高，
  端点组合后的 soft RMSE 低于 DiMA-PPI。
  未见边目标由已测 WT 端点相减得到，不是独立测量的中间突变标签。
  所报置信区间针对五种子集成预测；不与种子均值差混为同一估计量。
  29.1 倍只指既有 seed-42 CPU 模型推理计时，不包括预处理和 I/O。
  状态标量复用的精确零与五种子确定性重算的浮点残差分开报告。

编译
  在本目录执行：
  python draw_model.py
  python build_results.py
  pdflatex -interaction=nonstopmode -halt-on-error dima_iscas2026_full.tex
  bibtex dima_iscas2026_full
  pdflatex -interaction=nonstopmode -halt-on-error dima_iscas2026_full.tex
  pdflatex -interaction=nonstopmode -halt-on-error dima_iscas2026_full.tex
  Python 制图需要 matplotlib 和 numpy；仅编译现成图表不需要 Python。

稿件状态
  完整写作初稿，不是已提交或已通过 IEEE PDF eXpress 的终稿。
  使用官方指向的 IEEEtran conference 模式，未缩小字号或页边距。
  作者顺序沿用项目现稿，单位简写为学校；最终署名和单位需由作者核定。
  先前仅方法部分的文件及 ZIP 保留；请以带 full 的文件阅读完整版本。
  会议名称暂按 IEEE ISCAS 2026，尚未收到用户的会议/年份确认。
  官网标注 ISCAS 2026 投稿截止为 2025-10-12，当前已过期。
  https://2026.ieee-iscas.org/authors/author-instructions.html
  https://epapers2.org/iscas2026/ESR/samples.php

本地检查：5 页 US Letter，约 342 KB；第 5 页仅参考文献；无未定义引用与 Overfull；
全部字体嵌入，无 Type 3；逐页检查无裁切，末页双栏已平衡。

2026-10-13 全文科研语言润色
  已润色摘要、引言、方法、结果、结论和图注，统一状态与路径术语。
  数据表、证据文件和 PPI-Path92 导出数据与润色前版本逐字节一致。
  原稿保存在 revisions/pre_polish_20261013/，不收入交付源码包。
  重编译后仍为 4 页，无未定义引用和排版溢出，已逐页检查。

2026-10-14：表 I 加入全部 6 个模型的 Spearman 均值与 SD，
同步补充指标定义、DiMA-PPI 和状态描述符消融的排序相关性结果。
表格数值直接读取既有实验 CSV；论文仍为 4 页。

2026-10-14：加入已发表模型对比
  表 I 分组展示 DSSA-PPI、StaB-ddG、DiMA-PPI，以及匹配控制/消融。
  新增数值直接读取 evidence/comparator_model_summary.csv。
  DSSA-PPI 使用 5 组已发布折模型；StaB-ddG 从官方稳定性微调初始化
  在同一 S1129 外层划分上重新拟合，推理使用 20 次 Monte Carlo 采样。
  正文补充比较协议、三项指标差异，并明确 StaB-ddG 直接准确性更高。
  本次没有新训练模型；重编译后正文在前 4 页，第 5 页仅参考文献。

2026-10-14：按要求暂不比较 StaB-ddG
  从表 I 和直接预测准确性分析中移除 StaB-ddG，保留 DSSA-PPI 对比。
  固定 StaB-ddG 权重的状态复用实验、相关工作引用和原始证据继续保留。

参考文献更新：按用户提供的完整 BibTeX 替换为 12 个条目。
  graphppi 改为 GeoPPI 的 PLoS Computational Biology 期刊版本。
  esm2、proteinmpnn 等引用键与正文一致；新增 5 条存入文献库，
  尚未在正文引用的条目不会自动出现在 PDF 参考文献列表中。
  已重新运行 BibTeX 和 LaTeX，无未定义引用。

新增文献已引用：引言加入 FoldX、MutaBind2、mCSM-PPI2 和 DDMut-PPI；
方法中的局部侧链重排描述加入 PyRosetta 引用。12 条文献均已在正文
引用并出现在 PDF 中；FoldX 的 suppl_2 下划线已作 LaTeX 转义。
重新编译后仍为 5 页，第 5 页仅参考文献，无未定义引用或排版溢出。

Word 编辑版
  dima_iscas2026_full.docx：单栏可编辑文档，保留当前全文、作者、
  两幅图、两张原生 Word 表格、编号公式及全部 12 条参考文献。
  使用 Word 原生 OMML 公式；引用编号与当前 LaTeX PDF 一致。
  编辑版采用单栏排版，LibreOffice 检查为 7 页；IEEE 双栏 PDF 另行保留。
  build_word.py 可重新生成，需 pypandoc_binary 与 python-docx。

模型图重绘：按方法正文拆分为规范状态、共享编码器、势能差和状态复用
四个面板。图中补全特征维度、图邻域、消息传递与原子描述符融合关系；
用一致骨架图示展示 WT/突变体及结合/游离状态。同步更新 Word、
LaTeX PDF、图注、矢量 PDF/SVG 和 450 dpi PNG。

内部网络展开版模型图
  核对 train_residue_graph.py 的 ResidueGraphModel 后重绘：门控融合、
  四头 TransformerConv、SiLU/残差/LN、双路掩码注意力池化、
  141→96 描述符支路及 288→96→1 预测头。节点数量为层级示意，
  实际宽度按数字标注；图注同步说明。Word/PDF 和全部图形格式已更新。

Word 模型图替换：图 1 使用用户提供的 model figure.jpg，按原比例嵌入；
Word 图注同步对应 (a) 总体架构、(b) 门控融合、(c) 图注意力模块。
build_word.py 已同步，LaTeX 版仍使用 fig_model.pdf。
