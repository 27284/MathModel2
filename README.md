# C 题第二问：MathModel2 V3

V3 按 [完整优化方案](references/MathModel2_第二问_V3完整优化方案.md) 实现绝对时间慢趋势、独立 Common/Contrast、latency/stretch、pooling 消融和两等效源空间前向模型。第一问处理、333 个有效 Trial、5 个原始时间块、逐记录 temporal24 + LR(C=0.1) 主分类保持冻结。

## 运行

使用现有 EEGModel 环境，无需新增依赖。PyCharm 运行 `main.py` 或双击 `run_model2.bat`。

```powershell
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' main.py
# 完整流程快速检查，省去置换（不产生显著性结论）
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' main.py --output outputs/v3_check --permutations 0 --ecg-permutations 0
# 相同代码/数据/环境下续跑置换
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' main.py --permutations 500 --ecg-permutations 500
# 独立重算历史 V2 算法
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' main.py --version v2
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' -m unittest discover -s tests -v
```

V3 默认结果目录为 `outputs/results_v3`，先看 `第二问V3结果报告.md` 和 `figures/`。默认主分类、ECG 敏感性各 200 次独立置换；ECG 未通过训练核验时不进行对应回归/置换。`outputs/results` 保留全部 V2 历史结果；V2 重算默认写入 `outputs/results_v2_recompute`。代码变更后应指定新的输出目录，避免混合版本。

## 模型和验证

|模型|内容|
|---|---|
|M0|训练稳健 ERP 模板|
|M1|绝对时间 Slow + 普通时间基底|
|M2|Slow + WC Common/Contrast|
|M3|M2 + 内层选择 latency/stretch 组|
|M4|M3 + 独立 λC/λD|
|M5|M4 + 两等效源、对称空间前向 G|

采用预设的分阶段内层选择，主要损失为 250–500 ms joint 与 Contrast 误差。M1/M2 选择 slow×ridge，M3 选择形变组，M4 选择独立正则，M5 选择前向正则；主阶梯 pool=0，另做 pool=0/0.1/1、内层 pool 选择、LP750、geometry-prior 和旧响应加慢趋势的消融。不会根据外层成绩追加候选。

慢趋势从每个记录训练 Trial 的 -1 至 -0.2 秒原始样本按绝对时间拟合，再对 S(T+t) 做与 EEG 相同的基线校正。基线校正已消除的原始偏置不能再次加回 ERP；这意味着低自由度实验漂移不一定能解释刺激锁定的慢坡。慢成分不自动等同于伪影。

前向模型将训练 WC 自由读出系数转换到固定示意几何的两个等效源坐标，再用训练 ERP 拟合对称 G 或带 G0 惩罚的修正。它不是个体头模型，不做真实脑源定位。自由读出与受约束模型的预测表现均报告。

## 输出

|文件|用途|
|---|---|
|ablation_v3.csv|pooled、每记录、每折；joint/Common/Contrast；四窗口；R²、RMSE、SSE、ΔR²/ΔSSE|
|component_metrics_v3.csv|record×fold 的完整分量指标|
|foldwise_delta_sse_v3.csv|与相同折、相同目标 ERP template 配对比较|
|slow_decomposition_v3.csv|Observed / Slow / Residual ERP / Predicted neural|
|forward_metrics_v3.csv、forward_parameters_v3.csv|free/symmetric/geometry 对照和逐折 G|
|pooling_metrics_v3.csv|独立 pooling 消融|
|response_parameters_v3.json|内层候选损失、选择、训练编号、slow 与响应参数|
|heldout_erp_v3.npz|M4/M5 逐折条件 ERP 与预测，供重绘|
|classification_metrics.csv、confound_metrics.csv|冻结分类和混杂敏感性；包含完整覆盖标记|
|nuisance_status_v3.csv、confound_audit_v3.json|ECG-like 门槛、QRS-like/周期性、triggered EEG、训练回归参数|
|permutation_test.json、ecg_permutation_test.json|主分类及独立 ECG 敏感性置换|
|manifest.json、data_audit.json|代码、环境、数据和核心输出哈希；Trial/fold 审计|

图包括每记录慢趋势分解、Common/Contrast 预测、250–500 ms 放大、空间结构、M0–M5 消融和训练 heartbeat-triggered EEG。指标按逐折目标计分，展示曲线是等权折平均，两者不可混用。

## 代码

新增 `slow.py`、`basis.py`、`forward.py`、`nuisance.py`；V3 响应、验证、报告分别在 `response_v3.py`、`evaluation_v3.py`、`report_v3.py`，入口编排为 `pipeline_v3.py`。原 `response.py`、`evaluation.py`、`report.py` 保留用于历史 V2 复现，避免将全部逻辑堆入原文件。

[V3 实施说明](references/V3实施说明.md) 解释数值约定和方案映射。`模型说明.md`、`第二问实现方法与代码导读.md` 保留为 V2 历史说明，不代表 V3 新算法。

## 未知 Trial 分类

`final_model.json` 保持 schema_version=2 分类接口、response_version=3 元数据。全数据分类器可用于已校准记录；外层响应模型保存在 `response_parameters_v3.json`，不把留出模型冒充全数据响应模型。

```powershell
& 'D:\Anaconda\annconda\envs\EEGModel\python.exe' predict.py new_trials.npz --record VisualCogA_Task-1 --output predictions.csv
```

NPZ 的 `epochs` 为相同轻处理、基线校正后的 N×3×257 数组，通道顺序 Fz/F3/F4。无需真实标签；完全新记录须独立校准。
