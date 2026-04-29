> **稿本说明**：本文为 Markdown 初稿，便于版本管理与离线编辑。若学校终稿需 Word，可在本机安装 [Pandoc](https://pandoc.org/) 后执行  
> `pandoc thesis_draft_edge_ai_ricecooker.md -o thesis_draft.docx` 生成 `.docx`，再在模板中调整字体与页眉页脚。

---

# 基于边缘 AI 与异常检测的电饭煲外观质检系统研究与实现（初稿）

**摘要**

工业外观检验在消费电子产线中占用大量人力，且结果受光照、视角与操作习惯影响。本文围绕电饭煲多面外观检查场景，设计并实现一条可在树莓派类边缘设备上运行的视觉流水线：首先对单张输入图像做质量门控与几何归一（清晰度、曝光、覆膜干扰、姿态等），再在标准化后的产品 ROI 上采用基于记忆库的 PatchCore 类方法完成无监督异常评分，用于缺件、错装等整体性异常的初筛；在需要时，在同一 ROI 内引入 YOLOv8 导出的 ONNX 模型，对细划痕与裂纹类目标做可选的第二阶段筛查。全过程结果通过 SQLite 持久化，便于追溯与后续统计。实践表明，该架构把“规则可控的前处理”和“数据驱动的异常检测”分层衔接，有利于在算力受限条件下保持可部署性与可解释性。受样本规模与产线保密性限制，指标性实验仍待按产线数据补齐，文中对实验安排与评价协议给出明确占位，便于后续成稿。

**关键词**：边缘计算；工业视觉质检；无监督异常检测；PatchCore；YOLOv8；ONNX Runtime

---

## 第 1 章 绪论

### 1.1 研究背景

家电外壳类零件的出厂检验通常包含螺钉孔、按键、接口盖、标签等细小组件的完整性，以及面壳划痕、顶盖裂纹等表面缺陷。传统人工目检强度高、一致性与追溯性不足；而产线换型频繁、品类多，又为“逐类训练全监督检测器”带来标注成本。近年来，以 MVTec AD 为代表的数据集与评测协议推动了仅依赖正常样本训练的异常检测方法发展^[1]^，其中 PatchCore 通过特征记忆库与最近邻评分在精度与速度之间取得了较好折中^[2]^。同时，边缘侧 NPU 与 ONNX Runtime 等通用推理接口的普及，使在树莓派或配套 AI 加速卡上部署此类模型成为可行方案^[3][4]^。

### 1.2 问题界定

本文关注以下约束：（1）输入为**单张**彩色图像（线阵上传或单帧抓拍），不再假设连拍选帧；（2）检测需在**有限算力**下完成，并允许关闭部分阶段以缩短时延；（3）缺件类问题与细划痕类问题特点不同——前者更适合整图级或 patch 级异常汇总，后者更接近小目标检测，因而需要**分工**而非单一网络“包打天下”。

### 1.3 本文工作与章节安排

第 2 章综述异常检测、代表性深度学习检测器及边缘部署相关研究，其中目标检测部分仅讨论本文采用的 **YOLOv8**。第 3 章给出系统总体结构，分阶段说明 Stage1 质量与几何处理、Anomalib/PatchCore 推理、可选的 YOLOv8 细筛及 Stage3 数据落库。第 4 章描述实验与部署方案及待填指标。第 5 章总结并讨论局限与改进方向。

---

## 第 2 章 相关工作

### 2.1 工业图像异常检测

工业场景中异常形态多样、样本稀缺，一类主流思路是在仅含正常图像的训练集上学习“正常流形”，对测试图提取嵌入并在特征空间度量偏离程度。PatchCore 从预训练 CNN 的多层特征中采样 patch 描述子，构建核心集近似的记忆库，并以最近邻距离作为异常分数^[2]^。开源库 Anomalib 集成了 PatchCore 等多种算法与训练/评测脚本，降低了工程复现成本^[5]^。MVTec AD 提供了多类工业产品的高分辨率影像及像素级标注，是算法比较与消融的常用基准^[1]^。

### 2.2 可选的细粒度缺陷：YOLOv8

对于边界相对清晰、尺度较小的划痕与裂纹，仅依赖异常热图有时不够直观，可采用专用目标检测器在 ROI 内输出框与类别。本文工程选用 **YOLOv8**：在其常规检测头与训练管线基础上，将导出模型以 **ONNX** 形式交给 onnxruntime 或硬件配套运行时执行；若设备集成 Hailo-8L 等加速器，可将同一网络转换后的 `.hef` 等格式接入^[6]^。检测分支的选型与实验均仅针对 YOLOv8，不涉及其它目标检测器的横向综述。

### 2.3 经典视觉算子与边缘推理

ROI 对齐与前处理大量依赖传统算子：ORB 特征用于与标准图库的粗略配准与参考选择^[7]^；GrabCut 一类交互式分割思想常见于前景提取流程^[8]^。部署端采用 ONNX Runtime 可在 x86、ARM 等平台上获得相对统一的 C++/Python 接口^[4]^，有利于同一套代码在开发与树莓派环境之间迁移。

---

## 第 3 章 系统设计与实现

### 3.1 总体流程

系统按顺序执行以下阶段，任一前置阶段判定失败时可提前终止并返回原因：

1. **单图输入**：来自上传或 `trigger` 模块触发的单帧。
2. **Stage1（规则与几何）**：质量门控（Laplacian 方差衡量清晰度、亮度过曝/欠曝比例、最小分辨率）；塑料保护膜高亮覆盖检测；ORB 与标准库匹配选参考；模板/多尺度搜索定位 `product_bbox`；GrabCut 或相关策略抠图并保留标签区域；根据与参考的差异图产生缺件候选框，并与数量阈值比较给出 PASS/FAIL/RETAKE。
3. **Anomalib PatchCore**：在 Stage1 给出的产品裁剪图上计算异常分数，与配置阈值比较；可选地结合模板一致性或孔洞类几何约束（具体策略以 `stage_anomalib.py` 实现为准）。
4. **可选 Stage2（YOLOv8）**：在产品 ROI 内 resize 至网络输入尺寸，解析输出并经 NMS 与产品 mask 过滤，返回划痕/裂纹框。
5. **Stage3**：将视角、时间、各阶段结果写入 SQLite，字段设计兼容 `inspected_idx` 等历史列名。

上述编排由 `InspectorPipeline` 实现，支持 `enable_stage2`、`enable_anomalib`、`enable_sql_record` 等开关，便于在 Pi 上关闭细筛或记录以节约资源（参见工程内 `src/inspector_pipeline.py` 说明）。

### 3.2 Stage1 要点

-configurable 阈值集中在 `src/config.py`，例如 `MIN_SHARPNESS`、`OVEREXPOSE_RATIO_THRESHOLD`、`FILM_COVERAGE_THRESHOLD`、`MAX_MISSING_COUNT` 等。参数应根据相机、工位距离与补光重新标定，而不是照搬默认值。

### 3.3 PatchCore / Anomalib 集成

训练与导出流程由 `scripts/train_anomalib.py`、`scripts/eval_anomalib.py` 等脚本支持；运行时优先加载 ONNX，其次回退到 Lightning 权重，以便与边缘推理栈一致。阈值 `ANOMALIB_SCORE_THRESHOLD` 应结合验证集与产线可接受误报率标定。

### 3.4 YOLOv8 细筛模块

`Stage2Detector` 读取 YOLOv8 导出的 ONNX，输入张量形状为 `[1, 3, 640, 640]`；对 `[1, 6, 8400]` 形式输出手工实现 NMS，并按 mask 过滤背景误检。工程上同时探测 Hailo 与 onnxruntime，以适配开发机与树莓派+AI HAT（参见 `src/stage2_scratch.py`）。

### 3.5 数据持久化与图形界面

`Stage3SQLRecorder` 将结果写入 `data/inspection_records.db`。Streamlit 界面（如 `gui/app1.py`）用于演示单图上传与结果可视化，便于非研发人员试用。

---

## 第 4 章 实验与部署

### 4.1 硬件与软件环境

- 开发/验证：x86 PC，Python 3.10+，OpenCV、onnxruntime、Anomalib 依赖按 `requirements` 安装。
- 边缘部署：树莓派；可选 Hailo-8L + 对应 HEF；顺序执行以降低内存峰值。

### 4.2 评价协议（待填）

建议分别报告：

1. **Stage1**：RETAKE 比例、与人工一致的 PASS/FAIL 一致性（需定义金标准）。
2. **PatchCore**：在自建正常/异常验证集上的图像级 AUROC、或在固定阈值下的检出率与误报率。
3. **YOLOv8**：划痕/裂纹两类 mAP 或 F1；推理时延（纯 CPU / NPU）。
4. **端到端**：单张流水线总时延与失败原因分布。

当前文稿未填入具体数字，留待采集脱敏样本后补表。

### 4.3 消融与讨论（提纲）

- 关闭 Stage2 对总时延与漏检结构的影响。
- PatchCore 阈值变化对误报/漏报折中的敏感性。
- 覆膜、强反光工况下 Stage1 门控的必要性。

---

## 第 5 章 总结与展望

本文给出一套面向电饭煲多面质检的边缘视觉方案：单图进入后先由 Stage1 保证数据可用性与几何对齐，再用 PatchCore 类方法承担缺失性异常的主检，必要时用 YOLOv8 在 ROI 内补充细划痕信息，最后入库。该分工符合“前处理可解释、异常检测可数据驱动、细缺陷可用检测框输出”的产线需求。后续工作包括：完善验证集与定量指标；对阈值做系统化标定；在更多视角与 SKU 上验证泛化；以及对 Streamlit 演示与产线 MES 对接的接口规范化。

---

## 参考文献

[1] Bergmann P, Fauser M S, Sattlegger D, et al. MVTec AD — A comprehensive real-world dataset for unsupervised anomaly detection\[C\]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. （亦可查阅数据集官方页面及后续引用的标准描述。）

[2] Roth K, Pemula L, Zepeda J, et al. Towards Total Recall in Industrial Anomaly Detection\[C\]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022: 14318-14328.

[3] Jocher G, Chaurasia A, Qiu J. Ultralytics YOLOv8\[EB/OL\]. https://github.com/ultralytics/ultralytics , 2023. （工程实现与文档；学校格式允许时可改为机构技术报告条目。）

[4] ONNX Runtime documentation\[EB/OL\]. https://onnxruntime.ai/ , Microsoft et al.

[5] Anomalib Contributors. Anomalib: A Deep Learning Library for Anomaly Detection\[EB/OL\]. https://github.com/openvinotoolkit/anomalib

[6] Hailo Technologies. Hailo Developer Zone / 产品文档\[EB/OL\]. https://hailo.ai/ （用于说明 HEF 与 VDevice 推理链路；具体版本以实际 SDK 为准。）

[7] Rublee E, Rabaud V, Konolige K, et al. ORB: An efficient alternative to SIFT or SURF\[C\]//2011 International Conference on Computer Vision. IEEE, 2011: 2564-2571.

[8] Rother C, Kolmogorov V, Blake A. GrabCut: Interactive foreground extraction using iterated graph cuts\[J\]. ACM Transactions on Graphics (TOG), 2004, 23(3): 309-314.

（Additional contextual references: OpenCV library documentation; SQLite project documentation — 可在终稿按 GB/T 7714 补全出版项。）

---

## 附录 A：流水线字段说明（与实现对齐）

- `stage1_result`：单帧 Stage1 结构化输出；质量不合格时含 `RETAKE` 或失败原因。
- `anomaly_score`、`anomaly_map`（若有）：PatchCore 或融合逻辑输出。
- `stage2_rects`：YOLOv8 检测框列表（可选）。
- 数据库路径默认 `data/inspection_records.db`，可通过配置覆写。

---

*初稿完稿日期：2026-04-29；与代码库 RiceCooker_Project 单图版流程一致。*
