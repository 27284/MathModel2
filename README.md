# C题第二问

这是整理后的唯一主版本。项目位置：`D:\pycharm\Math_Model2`；环境：已有的 **EEGModel**。第一问程序和结果保持不变。

## 运行

在PyCharm打开本项目，解释器选择：

`D:\Anaconda\annconda\envs\EEGModel\python.exe`

运行 **main.py**，或选择预设配置“第二问”，也可以双击run_model2.bat。无需安装新库。

- 唯一结果目录：`outputs/results`
- 先看：`outputs/results/第二问结果报告.md`
- 拟合图：`outputs/results/figures`，每条记录上排为训练拟合，下排为留出预测。
- 再次运行核对代码、数据、库版本及核心结果指纹；默认200次置换，已完成的次数会复用。
- 代码修改后应使用新输出目录，不能把新代码与旧结果混合。

```powershell
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' main.py
# 相同方案追加置换
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' main.py --permutations 999
# 需要从头重新计算时指定新的输出目录
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' main.py --output outputs/recompute
# 代码和结果验收
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' -m unittest discover -s tests -v
```

## 项目结构

```text
main.py                 主流程：建模、验证、绘图、报告、续跑
predict.py              用保存的模型预测未知Trial
eeg_model2/
  config.py             固定参数
  data.py               原始数据与相同轻处理
  preprocess.py         训练内质量权重与稳健ERP
  shape.py              DoG/Gabor空间形状编码
  neural.py             稳定Wilson–Cowan神经群
  response.py           共同响应、左右差异、因果记忆状态
  features.py           固定逐记录24维LR主方案及分类对照
  diagnostics.py        训练内ECG、QC和试次顺序回归诊断
  evaluation.py         固定分类验证、嵌套波形选型与完整置换
  report.py             训练拟合/留出预测分开的结果展示
data/                   四份原始MAT，未修改
outputs/results/        当前版本唯一正式结果
references/             原建模思路、预设分析规则和历史对照表
tests/                  数据隔离、机制性质、结果一致性测试
```

## 如何理解结果

曲线拟合、未见Trial的平均响应预测、单次左右识别是三个不同问题。报告同时列出三者，不用训练R²代替分类准确率，也不隐藏拟合较差的通道。模型固定神经时间常数，仅估计受约束的等效观测系数；三电极不用于唯一脑源定位。

主分类器固定为逐记录24维时间窗特征 + LR(C=0.1)，不使用QC样本权重或机理差异分；LDA、池化、加权和差异分作为预设对照。混杂回归只用训练块估计参数，不改变原始主输入。完整0–800ms用于生成模型；离线零相位滤波不能当作500ms实时决策。

本轮合并BA从49.55%到55.75%；任务1为60.67%，任务2为50.74%。留出ERP R²为0.2688，仍低于直接训练ERP模板的0.2817，不宣称机理模型已优于模板。

详解见 `第二问实现方法与代码导读.md`，本轮采用和暂缓的评审建议见 `references/本轮评审修订规则.md`。当前最终模型格式为schema_version=2，共四个逐记录分类器；旧模型须重新生成。

## 未知Trial预测

将相同轻处理后的脑电保存为NPZ的`epochs`键，形状N×3×257，通道Fz/F3/F4，时间网格与final_model.json一致，刺激前均值为零。指定该模型已校准的记录标识：

```powershell
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' predict.py new_trials.npz --record VisualCogA_Task-1 --output predictions.csv
```

不传入未知Trial的真实标签。完全新的记录需要独立校准，不能冒用A/B标识宣称跨记录泛化。
