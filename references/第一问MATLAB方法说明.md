# 第一问：QC×Huber 分层方案

当前实现是 **MATLAB**，主流程为：原始刺激前 QC → 保护 SSP → EMD → 8维单 Trial 重建 → QC×Huber 稳健 ERP → 参数拟合。

## 运行顺序

在 MATLAB 中将当前目录切换到本文件所在的 matlab 文件夹，然后执行：

```matlab
run_problem1_complete
```

这一入口会重新计算、检验、绘图，并生成 `ProccesdData/layered_qc_huber/第一问交付结果/实验项目1` 和 `实验项目2`，保存 PNG/PDF/FIG 图片、逐时间点CSV、窗口指标与拟合系数。若已有最新计算结果，只需运行 `export_problem1_deliverables` 重新整理交付文件。

`run_problem1_layered` 一次处理四份原始数据。不要逐个运行 `+layered` 内函数。需要 Signal Processing Toolbox。绘图在数值计算结束后运行。

当前数值结果：`ProccesdData/layered_qc_huber`。新增加权对比图也在该目录的 `Figures` 中；原总览图仍在 `Figures/04_overview`。

## 方法与边界

- QC 使用未经零相位滤波的原始刺激前片段，防止滤波把刺激后响应带入质量评分。四项指标为基线 MAD、基线漂移、刺激前跳变和斜率。尺度由其他时间折的 MAD/IQR 估计。
- 原始削顶或分段越界的试次权重为零；一般质量偏差使用连续权重。保留试次数量与有效样本量不同。
- 保护 SSP 使用背景估计共同方向，其余四折训练模板限定投影幅度。10% 约束针对训练模板范数，不代表真实神经信号损伤保证，也不代表最终 Huber 曲线的改变量保证。
- EMD 使用扩展窗口和频率分组，联合训练一致性、空间方向和 ECG 相干决定是否衰减；证据不足就保留。
- 单试次在8维高斯/导数基底中岭重建。该先验可能损伤窄峰，不能证明恢复了干净脑电。
- 最终每个时刻的权重为 QC 权重乘 Huber 权重，QC 只乘一次；Huber 常数1.345，固定加权 MAD 尺度迭代求位置。非线性聚合后再作刺激前基线校正。
- 同时保留第8阶段 `basis8_QC_mean_reference`，与第5阶段 `QC_Huber_ERP` 比较，隔离 Huber 的影响。`huber_diagnostics.csv` 报告随时间变化的联合有效样本量。
- 拟合为3个高斯加3个尺度导数，不强制800 ms回零。高拟合优度只是曲线表达准确，不等于去噪准确。
- 300次块重采样在固定上游处理结果上重估 QC×Huber，只是条件区间，未包括整个处理流程的不确定性，不作人群推断。

主要文件：配置 `layered_config.m`；质量权重 `+layered/qc_weights.m`；稳健估计 `+layered/qc_huber.m`；流程 `+layered/process_file.m`；曲线调用 `evaluate_problem1_curve.m`。

```matlab
y = evaluate_problem1_curve('VisualCogA_Task-1', 2, -1, 0:800);
plot(0:800,y)
```

通道1/2/3对应Fz/F3/F4；方向-1/+1为左/右；输入时间单位毫秒。仅在原分析窗口内解释曲线。

结果说明文档由辅助脚本 `build_layered_report.py` 生成；它只整理 MATLAB 结果，不参与信号处理。
