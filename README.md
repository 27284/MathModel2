# C题第二问

这是整理后的唯一主版本。项目位置：`D:\pycharm\Math_Model2`；环境：已有的 **EEGModel**。第一问程序和结果保持不变。

## 运行

在PyCharm打开本项目，解释器选择：

`D:\Anaconda\annconda\envs\EEGModel\python.exe`

运行 **main.py**，或选择预设配置“第二问”，也可以双击run_model2.bat。无需安装新库。

- 唯一结果目录：`outputs/results`
- 先看：`outputs/results/第二问结果报告.md`
- 拟合图：`outputs/results/figures`，每条记录上排为训练拟合，下排为留出预测。
- 再次运行会核对代码、数据与库版本，自动复用已完成结果；不会每次重新做19次置换。
- 代码修改后应使用新输出目录，不能把新代码与旧结果混合。

```powershell
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' main.py
# 相同方案追加置换
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' main.py --permutations 99
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
  features.py           时空特征与噪声加权差异匹配
  evaluation.py         嵌套时间块验证与完整流程置换
  report.py             训练拟合/留出预测分开的结果展示
data/                   四份原始MAT，未修改
outputs/results/        当前版本唯一正式结果
references/             原建模思路、预设分析规则和历史对照表
tests/                  数据隔离、机制性质、结果一致性测试
```

## 如何理解结果

曲线拟合、未见Trial的平均响应预测、单次左右识别是三个不同问题。报告同时列出三者，不用训练R²代替分类准确率，也不隐藏拟合较差的通道。模型固定神经时间常数，仅估计受约束的等效观测系数；三电极不用于唯一脑源定位。

分类特征是100–500ms时空均值和共同/差异匹配，保留左右模板幅值信息。全部监督步骤在训练块内完成。完整0–800ms用于生成模型；离线零相位滤波不能当作500ms实时决策。

## 未知Trial预测

将相同轻处理后的脑电保存为NPZ的`epochs`键，形状N×3×257，通道Fz/F3/F4，时间网格与final_model.json一致，刺激前均值为零。指定该模型已校准的记录标识：

```powershell
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' predict.py new_trials.npz --record VisualCogA_Task-1 --output predictions.csv
```

不传入未知Trial的真实标签。完全新的记录需要独立校准，不能冒用A/B标识宣称跨记录泛化。

旧版程序及结果已从项目中清理，恢复备份存放在项目外的`D:\数模真题\中文题目\C题\第二问\history`，不会混入当前运行。
