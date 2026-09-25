# MathModel2 第二问完整优化方案（V3）

> 适用项目：`Math_Model2`  
> 目标：在**不继续追分类分数**的前提下，重点提升第二问的**响应生成模型、慢趋势分离、Common/Contrast 解耦与空间前向解释**。  
> 本文档面向后续人工修改和 Codex 辅助修改，所有“当前结果”均基于现有 V2 输出；所有“V3 建议”属于下一轮方案，不等同于已经实现。

---

## 0. 一句话总判断

当前 V2 已经形成了非常明确的结果分层：

1. **分类部分已有实质进步，可以基本冻结。**
2. **波形响应模型几乎没有实质提升。**
3. **0–800 ms 的正 \(R^2\) 很大程度上受到慢趋势影响，而 250–500 ms 的局部 ERP 预测仍然很差。**
4. **Common/Contrast 的“加减分解”本身没有问题，问题在于 Common、Contrast 各自的生成结构仍然偏粗糙。**
5. **真正需要投入精力的是：慢趋势与 ERP 显式拆分、Common/Contrast 独立建模、时间形变、最小空间前向模型。**

---

# 1. 当前 V2 的实测状态

## 1.1 分类主结果

当前正式主方案：

```text
逐记录 temporal24 + Logistic Regression
C = 0.1
5-fold blocked CV
不使用 QC 样本权重
不使用机理差异分
```

当前结果：

| 分组 | n | BA | Accuracy | permutation p |
|---|---:|---:|---:|---:|
| all | 333 | **0.5575** | 0.5586 | **0.0448** |
| Task1 | 168 | **0.6067** | 0.6071 | **0.0100** |
| Task2 | 165 | **0.5074** | 0.5091 | 0.4975 |
| A_Task1 | 83 | 0.6015 | 0.6024 | 0.0597 |
| A_Task2 | 86 | 0.5019 | 0.5116 | 0.5423 |
| B_Task1 | 85 | 0.6124 | 0.6118 | 0.0398 |
| B_Task2 | 79 | 0.5068 | 0.5063 | 0.4726 |

### 当前可以支持的结论

- Task1 中存在**弱但可检测的左右条件可分信息**。
- Task2 当前三导额区 EEG 中**没有可靠的单 Trial 左右可分证据**。
- 不能写成“Task2 理论上就应该随机”；只能说当前数据与当前特征下未得到可靠可分性。
- 不能把 \(d'=0.27\) 推出的 BA≈0.554 称为“严格理论上限”。

更准确的说法：

\[
\boxed{\text{BA}\approx0.55\text{ 是当前特征体系下观察到的经验可分水平，而不是数学不可突破上限。}}
\]

---

## 1.2 分类消融结果

| 方案 | all BA | Task1 BA | Task2 BA |
|---|---:|---:|---:|
| 主方案：逐记录 LR | **0.5575** | **0.6067** | 0.5074 |
| 逐记录 LDA | 0.5544 | 0.6068 | 0.5006 |
| 同任务池化 | 0.4802 | 0.4765 | 0.4826 |
| QC 权重 | 0.5465 | 0.5420 | 0.5502 |
| 机理差异分 | 0.5426 | 0.5712 | 0.5127 |
| ECG-adjusted | **0.5726** | 0.6127 | 0.5317 |
| QC-adjusted | 0.5411 | 0.5714 | 0.5110 |
| trial-order adjusted | 0.5456 | 0.5534 | 0.5370 |
| all-adjusted | 0.5557 | 0.5589 | 0.5543 |
| prestim-only | 0.4732 | 0.4220 | 0.5254 |

### 对分类部分的 V3 决策

**主分类器冻结。**

不要再以“把 BA 从 0.56 提高到 0.60+”为主要目标。

原因：

- 跨记录池化是明显负收益。
- QC 权重没有改善主结果。
- 机理差异分没有改善主结果。
- ECG-adjusted 虽然达到 0.5726，但它是探索性对照，不能因为看到它最高以后事后改成主方案。

### ECG-adjusted 如何处理

可以追加：

- 200–500 次独立置换；
- 但保持“敏感性分析”身份；
- 除非有新的独立数据验证，否则不要替换当前预先固定的主方案。

---

# 2. 当前响应模型是真正的主要问题

当前公平留出 ERP 对照：

| 模型 | 0–800 ms \(R^2\) | 0–500 ms \(R^2\) | 250–500 ms \(R^2\) | 600–800 ms \(R^2\) |
|---|---:|---:|---:|---:|
| V2 内层选型响应 | **0.2688** | -1.0172 | **-2.3188** | -1.3635 |
| V1 原结构响应 | 0.2671 | -1.0238 | -2.3349 | -1.3702 |
| Common-only | 0.2464 | -0.9540 | -2.2037 | -1.5181 |
| ERP template | **0.2817** | **-0.9639** | **-2.2347** | **-1.3193** |

因此：

\[
0.2671\rightarrow0.2688
\]

几乎没有实质提升。

而且：

\[
R^2_{\text{mechanism},0-800}
<
R^2_{\text{template},0-800}
\]

即：

\[
0.2688<0.2817
\]

所以目前不能声称机制模型在预测上优于简单 ERP 模板。

---

## 2.1 更严重的是 Common / Contrast 的留出表现

按当前 `erp_validation.csv` 汇总：

### 0–800 ms

\[
R^2_{\text{common}}\approx0.321
\]

\[
R^2_{\text{contrast}}\approx-1.077
\]

### 250–500 ms

\[
R^2_{\text{common}}\approx-10.425
\]

\[
R^2_{\text{contrast}}\approx-9.854
\]

说明整体 joint \(R^2\) 看起来没那么极端，是因为不同分量、不同幅度在总 SSE/SST 中发生了加权效果。

真正从机制解释角度看：

\[
\boxed{\text{Common 与 Contrast 在核心 250–500 ms 窗口都没有得到可靠泛化预测。}}
\]

---

# 3. 为什么总体 \(R^2\) 为正，但 250–500 ms 很差

当前数据中存在很强的慢趋势。

例如 Task2 的部分条件，在 250–500 ms 内简单线性趋势本身就可以解释非常高比例的方差：

- A_Task2 左条件多个通道约 0.81–0.90；
- A_Task2 右条件 F3 可达约 0.96。

因此当前响应模型容易优先解释：

\[
\boxed{\text{大幅度、低频、缓慢变化的成分}}
\]

而不是：

\[
\boxed{\text{250–500 ms 内更局部的事件锁定 ERP 形态}}
\]

注意：

> **慢趋势不等于已经证明是伪影。**

它可能包含：

- 电极漂移；
- 状态变化；
- 疲劳；
- 眼动；
- 心搏相关影响；
- 真实慢神经电位；
- 条件相关慢认知成分。

所以 V3 不应简单“把低频全部删掉”。

---

# 4. V3 的总模型结构

建议把当前响应模型升级为：

\[
\boxed{
Y_{i,e}(t)
=
S_e(T_i+t)
+
C_e(t)
+
z_iD_e(t)
+
\varepsilon_{i,e}(t)
}
\]

其中：

- \(i\)：Trial；
- \(e\)：Fz/F3/F4；
- \(T_i\)：该 Trial 在整个实验中的绝对起始时间；
- \(t\)：刺激锁定后的时间；
- \(S_e\)：慢趋势子模型；
- \(C_e\)：共同 ERP；
- \(D_e\)：左右差异 ERP；
- \(z_i=-1/+1\)：左右标签；
- \(\varepsilon\)：剩余噪声。

核心思想：

\[
\boxed{
\text{实验时间趋势}
\neq
\text{共同ERP}
\neq
\text{左右差异ERP}
}
\]

---

# 5. 第一项核心改造：慢趋势子模型与神经 ERP 子模型真正拆开

## 5.1 当前问题

当前 `response.py` 的 Common 中仍包含：

- 原始 WC source；
- 80 ms 低通；
- 250 ms 低通；
- 750 ms 低通；
- 300 ms persistent step；
- 800 ms persistent step。

这些 basis 中尤其：

- 750 ms；
- 300/800 ms persistent step；

很容易拟合大慢坡。

因此现在的模型实际更接近：

\[
C(t)=\sum_j\beta_jB_j(t)
\]

而没有明确回答：

> 哪部分是慢趋势，哪部分才是 ERP。

---

## 5.2 V3 主方案

显式拆成：

\[
\boxed{
C_e(t)=S_e(T)+N_{C,e}(t)
}
\]

其中：

### 慢趋势

\[
S_e(T)
\]

只使用**实验绝对时间**。

### 神经 ERP

\[
N_{C,e}(t)
\]

只使用**刺激锁定时间**。

这是最关键的可辨识来源：

- 漂移跟实验进程走；
- ERP 每个 Trial 都在刺激时刻 \(t=0\) 重新对齐。

---

## 5.3 推荐的慢趋势候选

不要一上来使用高自由度 spline。

建议训练折内比较：

### S0：无慢趋势

\[
S(T)=0
\]

### S1：线性趋势

\[
S(T)=a+bT
\]

### S2：二次趋势

\[
S(T)=a+bT+cT^2
\]

### S3：低自由度自然样条

例如 3–4 个自由度。

选择完全在内层训练折完成。

### 不建议

- 高自由度 spline；
- 每个 Trial 单独 detrend；
- 直接把 <0.5 Hz 全删掉；
- 只靠 800 ms epoch 内部判断“漂移”。

---

# 6. 第二项核心改造：Common / Contrast 保留，但完全解耦

Common/Contrast 分解本身是严格可逆的：

\[
C(t)=\frac{R(t)+L(t)}2
\]

\[
D(t)=\frac{R(t)-L(t)}2
\]

恢复：

\[
L=C-D
\]

\[
R=C+D
\]

因此问题不在“相加相减太粗糙”，而在于：

\[
\boxed{\text{当前 }C\text{ 与 }D\text{ 的生成结构仍然太相似。}}
\]

---

## 6.1 V3 设计

### Common

\[
C_e(t)
=
S_e(T)
+
B_C(t)\beta_C
\]

允许：

- 慢趋势；
- 共同视觉响应；
- 比较平滑的 WC 相关动态。

### Contrast

\[
D_e(t)
=
B_D(t)\beta_D
\]

重点表达：

- 左右短暂差异；
- 潜伏期差异；
- 峰宽差异；
- 形状偏好源差异。

---

## 6.2 独立正则

当前：

```python
lambda_contrast = 10 * lambda_common
```

V3 改成训练内独立选择：

\[
\lambda_C\in\Lambda_C
\]

\[
\lambda_D\in\Lambda_D
\]

不要再固定：

\[
\lambda_D=10\lambda_C
\]

候选可以保持很小：

```text
lambda_C ∈ {0.1, 0.01, 0.001}
lambda_D ∈ {0.1, 0.01, 0.001}
```

避免搜索空间爆炸。

---

# 7. 第三项核心改造：真正加入 latency + stretch

V2 只给 Contrast 增加：

\[
-40\text{ ms},\quad +40\text{ ms}
\]

两个平移版本。

这仍然不足以表达不同记录、不同条件中 ERP 的：

- 潜伏期变化；
- 峰宽变化；
- 时间压缩/拉伸。

---

## 7.1 建议的时间形变

定义：

\[
B_{\delta,s}(t)
=
B\left(\frac{t-\delta}{s}\right)
\]

其中：

### latency

\[
\delta\in\{-80,-40,0,+40,+80\}\text{ ms}
\]

### stretch

\[
s\in\{0.7,1.0,1.4\}
\]

---

## 7.2 不要把全部组合无约束地塞进去

如果所有组合全部自由：

\[
5\times3\times6
\]

会导致 basis 数迅速膨胀并产生共线性。

推荐两种实现：

### 实现 A：小型候选组 + 内层选择

每次只启用一组：

```text
Base
Base + ±40 ms
Base + ±80 ms
Base + stretch(0.7,1.4)
Base + shift + stretch
```

由 inner CV 选组。

### 实现 B：完整 basis bank + 强 Ridge / group Ridge

如果实现方便，可以产生较大的 basis bank，但必须：

- 标准化；
- 强正则；
- 完全训练内选择；
- 报告 effective df。

---

# 8. 第四项核心改造：重新处理长时间尺度 basis

V2 Common 中：

```text
q
LP80
LP250
LP750
step300
step800
```

建议 V3 做分层消融。

## 主模型

神经 ERP basis：

```text
q
LP80
LP250
latency variants
stretch variants
```

慢趋势：

```text
单独的 S(T)
```

## 扩展模型

再测试：

```text
+ LP750
```

如果 LP750 在 held-out 250–500 ms 有稳定改善，可以保留。

## 不再建议

让 300/800 ms persistent step 与 WC 神经 basis 混在同一个神经子模型里。

它们如果保留，应归到：

\[
\boxed{S_{\text{slow}}}
\]

或作为单独消融。

---

# 9. 第五项核心改造：补最小空间前向模型

当前项目最大的机制缺口之一：

\[
\text{形状偏好源}
\rightarrow
\text{Fz/F3/F4}
\]

仍主要通过自由读出系数实现。

`dipole_demo()` 有几何示意，但没有进入主拟合。

---

## 9.1 不做真实脑源定位

只有 Fz/F3/F4 三个额区通道，因此：

\[
\boxed{\text{不能声称恢复真实三维脑源。}}
\]

V3 只建立：

\[
\boxed{\text{受几何约束的等效前向模型}}
\]

---

## 9.2 最小对称前向矩阵

定义两个等效源：

\[
\mathbf q(t)=
\begin{bmatrix}
q_L(t)\\
q_R(t)
\end{bmatrix}
\]

三个电极：

\[
\mathbf y(t)=
\begin{bmatrix}
Fz(t)\\
F3(t)\\
F4(t)
\end{bmatrix}
\]

建立：

\[
\mathbf y(t)=G\mathbf q(t)+\varepsilon(t)
\]

最小对称结构：

\[
G=
\begin{bmatrix}
a_0&a_0\\
a_1&a_2\\
a_2&a_1
\end{bmatrix}
\]

含义：

- Fz 对左右源近似对称；
- F3 更敏感于一侧源；
- F4 为镜像结构。

这比 6 个完全自由的 channel-source 系数更有解释性。

---

## 9.3 几何先验版本

如果使用 `dipole_demo()` 得到先验：

\[
G_0
\]

则：

\[
G=G_0+\Delta G
\]

并加入：

\[
\lambda_G\|\Delta G\|_F^2
\]

即：

> 几何给出主结构，数据只允许做小修正。

---

## 9.4 空间模型评价

必须同时报告：

- constrained forward；
- free readout。

如果 constrained forward：

- \(R^2\) 略下降；
- 但参数稳定性和左右侧化解释更清楚；

可以写成：

> 增强了机制可解释性，但没有带来预测性能优势。

不要为了“机制好看”隐藏预测下降。

---

# 10. 第六项：ECG / QC / Trial-order nuisance controls

## 10.1 ECG

当前第 7 通道标签存在“ECG/参考相关”描述歧义。

因此先检查：

- 是否有周期性心搏波形；
- 是否能检测稳定 QRS；
- ECG-triggered EEG average 是否存在同步污染。

只有确认后，才把它作为标准 ECG nuisance regressor。

### ECG-adjusted

当前：

\[
BA=0.5726
\]

值得继续做：

- 200–500 次置换；
- 作为敏感性分析；
- 不事后替换主模型。

---

## 10.2 QC nuisance

刺激前已有 QC 指标。

原则：

\[
\boxed{\text{所有 nuisance 回归参数必须只在训练折估计。}}
\]

不能用测试折确定回归系数。

---

## 10.3 Trial order

显式加入：

\[
trial\_index
\]

因为如果：

- 左右标签与试次顺序略相关；
- EEG 又随实验时间漂移；

就可能形成伪左右差异。

V3 必须报告：

```text
raw
ECG-adjusted
QC-adjusted
order-adjusted
all-adjusted
```

但这些仍然是**敏感性结果**，不要从中挑最高值当主方案。

---

# 11. 第七项：response pooling 必须做独立消融

分类已经明确显示：

\[
\text{同任务跨记录池化}
\]

是明显负收益。

因此响应模型中的：

```python
pool ∈ {1.0, 0.1, 0}
```

也必须单独评估。

建议 V3 重点比较：

\[
pool=0
\]

与：

\[
pool>0
\]

如果：

\[
pool=0
\]

在 held-out ERP 上更稳定，则停止对 A/B 做同任务系数收缩。

结论可以是：

> 个体间观测结构差异较大，共享神经机制不等于共享观测系数。

---

# 12. V3 的模型消融阶梯

不要一次性把所有结构同时加进去。

建议固定如下阶梯：

| 模型 | 内容 | 目的 |
|---|---|---|
| M0 | 训练 ERP template | 最简单经验基准 |
| M1 | Slow + 普通时间 basis | 判断慢趋势显式分离是否有效 |
| M2 | Slow + WC Common/Contrast | 判断 WC 是否带来增益 |
| M3 | M2 + latency/stretch | 判断时间形变是否改善 |
| M4 | M3 + \(\lambda_C,\lambda_D\) 独立 | 判断 C/D 解耦是否有效 |
| M5 | M4 + constrained forward | 判断空间机制是否增加价值 |

每增加一层只回答两个问题：

\[
\boxed{\text{留出预测有没有改善？}}
\]

以及：

\[
\boxed{\text{机制解释是否更清楚？}}
\]

---

# 13. V3 最重要的评价指标

以后不要首先看训练 \(R^2\)。

建议优先级：

## 第一优先级

\[
\boxed{R^2_{\text{heldout},250-500}}
\]

这是当前最差、最值得改善的核心指标。

---

## 第二优先级

\[
\boxed{R^2_{\text{contrast},250-500}}
\]

因为第二问本质要求解释左右差异。

---

## 第三优先级

\[
\boxed{
\Delta R^2
=
R^2_{\text{mechanism}}
-
R^2_{\text{template}}
}
\]

当前：

\[
\Delta R^2<0
\]

V3 希望至少达到：

\[
\Delta R^2\ge0
\]

但不要为了强行转正而不断扩大模型复杂度。

---

## 第四优先级

\[
RMSE
\]

以及：

\[
\Delta SSE
=
SSE_{\text{template}}
-
SSE_{\text{mechanism}}
\]

对每个 record×fold 分别计算。

---

## 第五优先级

分类：

\[
BA,\quad p_{\text{perm}}
\]

分类主流程冻结，只用于确认 V3 的响应模型修改没有破坏原有分类结论。

---

# 14. 验证协议必须保持严格

## 14.1 第一问冻结

保持：

- 333 个有效 Trial；
- 现有 QC；
- 原始 5 个时间块；
- 数据哈希；
- 原始预处理规则。

不要为了 V3 的结果回头修改第一问。

---

## 14.2 训练 / 测试边界

所有会从数据学习的量都只能在训练块估计，包括：

- slow trend；
- nuisance regression；
- basis 选择；
- ridge；
- pooling；
- latency/stretch；
- forward correction；
- standardization。

测试块只能：

\[
\boxed{\text{应用训练阶段已经确定的规则}}
\]

---

## 14.3 不再只看 pooled \(R^2\)

必须同时报告：

- pooled；
- 每个 record；
- 每个 fold；
- Common；
- Contrast；
- 0–800；
- 0–500；
- 250–500；
- 600–800。

避免一个大慢趋势把局部失败掩盖掉。

---

# 15. 推荐代码结构

建议 V3 不直接把所有逻辑继续堆进 `response.py`。

推荐：

```text
eeg_model2/
├── data.py
├── preprocess.py
├── shape.py
├── neural.py
├── slow.py          # 新增：慢趋势模型
├── forward.py       # 新增：最小空间前向模型
├── response.py      # Common/Contrast + WC basis
├── nuisance.py      # 新增或整理 ECG/QC/order controls
├── features.py
├── evaluation.py
├── diagnostics.py
└── report.py
```

---

# 16. 具体模块修改建议

## 16.1 `slow.py`

负责：

```python
fit_slow(train_epochs, onsets, setting)
apply_slow(model, epochs, onsets)
```

候选：

```text
none
linear
quadratic
natural_spline_df4
```

要求：

- per-record；
- per-channel；
- training-only；
- 输出可单独保存、绘图。

---

## 16.2 `response.py`

改造目标：

```text
Common:
slow + WC_common

Contrast:
WC_contrast
```

并实现：

```python
lambda_common
lambda_contrast
basis_common
basis_contrast
```

不要再把 C/D 的正则比例写死。

---

## 16.3 `basis.py`（可选新增）

负责：

```python
shift_basis()
stretch_basis()
build_common_basis()
build_contrast_basis()
```

避免 `response.py` 继续膨胀。

---

## 16.4 `forward.py`

实现：

```python
geometry_prior()
fit_constrained_forward()
apply_forward()
```

至少支持：

```text
free_readout
symmetric_forward
geometry_prior_forward
```

三种消融。

---

## 16.5 `nuisance.py`

统一：

```text
ECG
QC
trial order
all combined
```

所有系数只在训练块拟合。

---

## 16.6 `evaluation.py`

新增主要输出：

```text
ablation_v3.csv
component_metrics_v3.csv
foldwise_delta_sse_v3.csv
forward_metrics_v3.csv
slow_decomposition_v3.csv
```

---

# 17. V3 必须新增的图

建议至少输出：

## 图 1：慢趋势与 ERP 分解图

每个记录画：

```text
Observed ERP
Estimated slow component
Residual/event-locked ERP
Predicted neural ERP
```

这是 V3 最关键的解释图。

---

## 图 2：Common / Contrast 分开预测

分别画：

\[
C_{\text{obs}},\ C_{\text{pred}}
\]

\[
D_{\text{obs}},\ D_{\text{pred}}
\]

不要只画最终左/右。

---

## 图 3：250–500 ms 放大图

必须单独放大：

\[
250-500\text{ ms}
\]

因为当前总窗图会掩盖局部失败。

---

## 图 4：空间前向模型

画：

```text
左形状偏好源 ─┬→ Fz
              ├→ F3
              └→ F4

右形状偏好源 ─┬→ Fz
              ├→ F3
              └→ F4
```

并标注 \(G\) 的对称约束。

---

## 图 5：M0–M5 消融

横轴：

```text
M0 M1 M2 M3 M4 M5
```

纵轴分别画：

- held-out 0–800 \(R^2\)；
- held-out 250–500 \(R^2\)；
- contrast \(R^2\)。

---

# 18. 推荐实施顺序

## P0：保持不动

- 第一问；
- Trial 集合；
- 分类主模型；
- 5-fold blocked CV；
- 当前结果目录留档。

---

## P1：先拆慢趋势

实现：

\[
Y=S_{\text{slow}}+ERP+\epsilon
\]

这是最优先的改动。

先不要改空间模型。

比较：

```text
旧 response
vs
slow + old ERP response
```

---

## P2：Common / Contrast 独立正则

实现：

```text
lambda_C
lambda_D
```

并报告 C/D 分开结果。

---

## P3：latency/stretch

加入：

\[
\delta,\ s
\]

但用 inner CV 控制复杂度。

---

## P4：pooling 消融

重点比较：

```text
pool = 0
pool = 0.1
pool = 1
```

---

## P5：最小空间 forward

最后再接：

\[
G
\]

避免同时改太多结构后无法判断收益来源。

---

## P6：最终统一消融

跑：

\[
M0\rightarrow M5
\]

形成正式结果。

---

# 19. 推荐的停止条件

不要无限优化。

出现以下情况之一，应停止继续加复杂度。

## 条件 1

M5 后：

\[
R^2_{250-500}<0
\]

且仍不优于 ERP template。

结论：

> 当前三导额区 EEG 不足以验证具有预测优势的详细生成机制。

---

## 条件 2

空间 forward 提升解释性，但预测略下降。

保留为机制消融，但不声称提高预测能力。

---

## 条件 3

某个改动只提高训练 \(R^2\)，held-out 没提高。

判定为：

\[
\boxed{\text{无泛化收益}}
\]

---

## 条件 4

需要不断根据外层测试结果修改窗口、basis、参数才能提升。

立即停止。

否则会产生明显测试集选择偏差。

---

# 20. V3 论文结论应该如何定位

如果最终分类结果基本保持，而响应模型仍较弱，可写：

> 在三导额区 EEG 条件下，Task1 的左右视觉条件表现出弱但可重复的记录内可分信息，而 Task2 未获得可靠单 Trial 解码证据。通过形状编码、受约束 Wilson–Cowan 动力学及等效观测模型，可构造从视觉形状到神经群响应再到头皮 EEG 的机制框架；然而其留出 ERP 预测尚未稳定超过训练 ERP 模板，尤其 250–500 ms 时窗的泛化能力有限。因此，本模型当前的主要价值在于提供结构化机制解释，而非证明其具有优于经验模板的预测优势。

如果 V3 成功超过 template，则可以改为：

> 在显式分离慢趋势、允许 Common/Contrast 独立动力学并加入受约束空间前向关系后，机制模型在留出数据上获得稳定的 SSE 降低/ \(R^2\) 提升，说明所加入的结构不仅具有解释意义，也提供了额外预测信息。

---

# 21. 不能写的结论

无论 V3 结果如何，目前都不要写：

1. “BA=0.5575 已经达到严格理论极限。”
2. “所有慢趋势都是伪影。”
3. “已经定位出真实左右脑源。”
4. “Wilson–Cowan 参数就是受试者真实突触参数。”
5. “Task2 本来就应该随机。”
6. “ECG-adjusted 更高，因此 ECG 回归后模型一定更正确。”
7. “训练 \(R^2\) 很高，所以神经机制得到了验证。”

---

# 22. V3 最终目标不是“把所有数字变漂亮”

真正需要回答三个问题：

### 问题 A：预测

\[
\boxed{\text{机制模型是否比 ERP template 更能预测未见数据？}}
\]

### 问题 B：差异

\[
\boxed{\text{模型是否能预测真正的左右 Contrast，而不是只拟合共同慢趋势？}}
\]

### 问题 C：机制

\[
\boxed{\text{形状 → 神经群 → 空间投影 → EEG 的每一层是否有明确数学作用？}}
\]

只要这三点能回答清楚，即使最终 BA 仍在 0.55 左右，项目质量也会明显高于单纯追分类分数。

---

# 23. 给 Codex 的直接执行清单

可以把下面内容直接作为下一轮代码修改要求：

```text
目标：MathModel2 V3，不修改第一问，不修改333个有效Trial，不改变5个原始blocked folds。

1. 冻结当前主分类：
   - per-record temporal24
   - LogisticRegression C=0.1
   - no QC sample weighting
   - no mechanism contrast feature

2. 新增 slow.py：
   - slow trend uses absolute experiment time, not within-epoch time
   - candidate: none / linear / quadratic / low-df natural spline
   - fit training folds only
   - apply to validation/test without refitting

3. response.py：
   - model Common = slow + neural_common
   - model Contrast = neural_contrast
   - remove persistent step from neural contrast
   - persistent/very-slow terms belong to slow submodel
   - common and contrast use separate basis families
   - lambda_common and lambda_contrast selected independently

4. Add latency/stretch:
   - latency candidates: -80, -40, 0, +40, +80 ms
   - stretch candidates: 0.7, 1.0, 1.4
   - avoid unrestricted full basis explosion
   - use nested inner CV / strong ridge

5. Add response pooling ablation:
   - pool = 0, 0.1, 1.0
   - report held-out metrics separately

6. Add forward.py:
   - minimal 2-source → 3-electrode forward model
   - symmetric structure:
       [[a0,a0],
        [a1,a2],
        [a2,a1]]
   - optional geometry-prior G0 + penalized DeltaG
   - do not claim true source localization

7. Nuisance controls:
   - ECG only after verifying channel-7 is usable ECG-like waveform
   - QC covariates
   - trial index
   - all fitted on training folds only
   - keep as sensitivity analyses

8. Evaluation:
   - M0 ERP template
   - M1 slow + temporal basis
   - M2 slow + WC
   - M3 + latency/stretch
   - M4 + independent C/D regularization
   - M5 + constrained forward

9. Required metrics:
   - held-out R2: 0-800, 0-500, 250-500, 600-800
   - Common R2 and Contrast R2 separately
   - RMSE, SSE
   - mechanism minus template delta-R2
   - foldwise delta-SSE
   - per-record results

10. Stop rule:
   - if M5 still does not beat ERP template and 250-500 R2 remains negative,
     stop adding complexity and report data/model limitation honestly.

11. Keep all old outputs; write new outputs to outputs/results_v3.
```

---

# 24. 最终推荐

V3 不要再围绕“分类精度还能不能高一点”设计。

推荐主线：

\[
\boxed{
\text{冻结分类}
\rightarrow
\text{慢趋势/ERP拆分}
\rightarrow
\text{C/D解耦}
\rightarrow
\text{latency/stretch}
\rightarrow
\text{pooling消融}
\rightarrow
\text{最小空间forward}
\rightarrow
\text{M0–M5公平比较}
}
\]

真正希望看到的不是：

\[
BA:0.5575\rightarrow0.60
\]

而是：

\[
\boxed{
R^2_{\text{mechanism,heldout}}
>
R^2_{\text{template,heldout}}
}
\]

以及：

\[
\boxed{
R^2_{\text{contrast},250-500}
\text{得到实质改善}
}
\]

如果做不到，也应明确报告：

> 当前三导额区 EEG 支持弱记录内判别，但不足以验证一个具有预测优势的详细视觉神经动力学生成模型。

这将是比“继续堆参数追高分”更稳健、更符合数学建模论文逻辑的结论。
