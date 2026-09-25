"""V3 heldout plots and conservative report, generated from saved scores."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .forward import geometry_prior


def figures_v3(out):
    directory = out/'figures'; directory.mkdir(exist_ok=True)
    table = pd.read_csv(out/'slow_decomposition_v3.csv')
    for record, values in table.groupby('record'):
        avg = values.groupby(['direction','channel','time_s']).mean(numeric_only=True).reset_index()
        fig, axes = plt.subplots(2,3,figsize=(13,7),sharex=True)
        for row, sign in enumerate((-1,1)):
            for col, ch in enumerate(('Fz','F3','F4')):
                a = avg[(avg.direction==sign)&(avg.channel==ch)]
                ax = axes[row,col]
                for field, label, style in [('observed','Observed ERP','k-'),('slow','Estimated slow','C2--'),
                                           ('residual_erp','Residual ERP','C0-'),('neural_prediction','Predicted neural','C3-')]:
                    ax.plot(a.time_s*1000,a[field],style,label=label,alpha=.8)
                ax.axvspan(250,500,color='grey',alpha=.08); ax.set_title(f'{ch} / {"L" if sign<0 else "R"}')
                ax.set_xlabel('Time (ms)'); ax.set_ylabel('Original signal units')
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'{record}: M4 heldout decomposition (equal fold average)')
        fig.tight_layout();fig.savefig(directory/f'{record}_slow.png',dpi=140);plt.close(fig)
    archive = np.load(out/'heldout_erp_v3.npz')
    t = archive['time_s']*1000
    for record in table.record.unique():
        for zoom in (False,True):
            fig,axes = plt.subplots(2,3,figsize=(13,6),sharex=True)
            for name,color in [('M4','C0'),('M5','C3')]:
                actual=np.mean([archive[f'{f}__{record}__{name}__actual'] for f in range(5)],axis=0)
                pred=np.mean([archive[f'{f}__{record}__{name}__predicted'] for f in range(5)],axis=0)
                for row in (0,1):
                    a=actual.mean(axis=0) if row==0 else (actual[1]-actual[0])/2
                    p=pred.mean(axis=0) if row==0 else (pred[1]-pred[0])/2
                    for col,ch in enumerate(('Fz','F3','F4')):
                        ax=axes[row,col]
                        if name=='M4':ax.plot(t,a[col],'k-',label='Observed')
                        ax.plot(t,p[col],color=color,label=name)
                        ax.set_title(f'{ch} / {"Common" if row==0 else "Contrast"}')
                        ax.set_xlabel('Time (ms)');ax.set_ylabel('Original signal units')
                        ax.set_xlim((250,500) if zoom else (0,800))
                        if zoom:
                            mask=(t>=250)&(t<=500)
                            plotted=np.concatenate([line.get_ydata()[mask] for line in ax.lines])
                            span=max(float(np.ptp(plotted)),1e-6)
                            ax.set_ylim(plotted.min()-.08*span,plotted.max()+.08*span)
            axes[0,0].legend();fig.suptitle(f'{record}: heldout components (equal fold average)')
            fig.tight_layout();fig.savefig(directory/f'{record}_{"zoom250_500" if zoom else "components"}.png',dpi=140);plt.close(fig)
    s=pd.read_csv(out/'ablation_v3.csv');s=s[(s.scope=='pooled')&s.model.isin([f'M{i}' for i in range(6)])]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(component,window) in zip(axes,[('joint','0_800'),('joint','250_500'),('contrast','250_500')]):
        a=s[(s.component==component)&(s.window==window)].sort_values('model')
        ax.plot(a.model,a.r2,'o-');ax.axhline(0,color='grey',lw=.8)
        ax.set_title(f'Heldout {component}: {window} ms');ax.set_ylabel('Pooled R² (summed SSE/SST)')
    fig.tight_layout();fig.savefig(directory/'ablation_M0_M5.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));ax.axis('off')
    for j,label in enumerate(('Equivalent left source','Equivalent right source')):
        y=.75-.5*j;ax.text(.05,y,label,ha='left',va='center',bbox=dict(facecolor='#e1ecfa',edgecolor='none',pad=8))
        for i,electrode in enumerate(('Fz','F3','F4')):
            ey=.8-.3*i
            ax.annotate('',xy=(.76,ey),xytext=(.35,y),arrowprops=dict(arrowstyle='->',color=f'C{j}',alpha=.7))
    for i,electrode in enumerate(('Fz','F3','F4')):ax.text(.8,.8-.3*i,electrode,va='center',fontsize=14)
    ax.set_title('Symmetric forward: G = [[a0,a0], [a1,a2], [a2,a1]]\nEquivalent geometry only; no anatomical source localization')
    fig.tight_layout();fig.savefig(directory/'forward_model.png',dpi=150);plt.close(fig)
    audits=json.loads((out/'confound_audit_v3.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,record in zip(axes.ravel(),table.record.unique()):
        entries=[a['ecg_verification'] for a in audits if a['record']==record]
        eeg=np.mean([a['triggered_eeg_average'] for a in entries],axis=0)
        for i,ch in enumerate(('Fz','F3','F4')):ax.plot(np.array(entries[0]['triggered_time_s'])*1000,eeg[i],label=ch)
        ax.set_title(record);ax.set_xlabel('Time from candidate QRS (ms)');ax.legend()
    fig.suptitle('Training-only candidate heartbeat-triggered EEG (descriptive; gate may fail)')
    fig.tight_layout();fig.savefig(directory/'ecg_triggered_eeg.png',dpi=140);plt.close(fig)


def report_v3(out, permutations):
    scores=pd.read_csv(out/'ablation_v3.csv')
    pooled=scores[(scores.scope=='pooled')&scores.model.isin([f'M{i}' for i in range(6)])]
    lines=['# 第二问 V3 留出验证报告','',
           '第一问、333 个有效 Trial、原始 5 个时间块、temporal24 + 逐记录 LR(C=0.1) 均冻结。所有响应选择只在外层训练块内进行。', '',
           '## M0–M5 公平消融','', '|模型|0–800 joint R²|250–500 joint R²|250–500 Contrast R²|250–500 ΔR² vs template|',
           '|---|---:|---:|---:|---:|']
    def row(model,component,window):
        return pooled[(pooled.model==model)&(pooled.component==component)&(pooled.window==window)].iloc[0]
    for model in [f'M{i}' for i in range(6)]:
        a,b,c=row(model,'joint','0_800'),row(model,'joint','250_500'),row(model,'contrast','250_500')
        lines.append(f'|{model}|{a.r2:.4f}|{b.r2:.4f}|{c.r2:.4f}|{b.delta_r2:.4f}|')
    final=row('M5','joint','250_500')
    if final.r2<0 and final.delta_r2<=0:
        conclusion='达到预设停止条件：M5 核心窗 R² 仍为负且不优于 ERP template。停止增加复杂度；当前三导额区 EEG 不足以验证具有预测优势的详细生成机制。'
    elif final.delta_r2>0:
        conclusion='M5 核心窗相对模板的合并指标有改善；须结合逐记录、逐折 ΔSSE 判断稳定性，不能仅凭 pooled 指标宣称机制已验证。'
    else:
        conclusion='M5 未获得相对模板的核心窗预测优势。保留空间模型作为机制消融，不声称提高预测能力。'
    lines += ['',conclusion,'','## 方法与解释边界','',
        '- M0：训练稳健 ERP；M1：绝对慢趋势 + 普通高斯时间基底；M2：慢趋势 + WC；M3：小组 latency/stretch；M4：C/D 独立正则；M5：两等效源 + 对称 G。',
        '- 分阶段内层选择：M1/M2 搜索 slow×ridge；M3 在 M2 上选时间形变；M4 在 M3 上选独立 λ；M5 在 M4 上选前向正则。不是所有组合穷举。pool 主阶梯固定为 0，单独报告 0/0.1/1 及内层选择的 pool。',
        '- 内层损失预设为 250–500 ms 的 joint 和 Contrast 均方误差之和，按训练通道尺度归一化；全窗损失仅用于平手。没有根据外层成绩调整候选。',
        '- 慢趋势从训练 Trial 的 -1 至 -0.2 s 原始样本、按绝对实验秒数拟合。自然样条为 4 自由度，节点与尺度只来自训练。输出 S(T+t) 再做原基线校正；这使低自由度全记录漂移对已基线校正 ERP 的贡献可能很小。不能把刺激锁定慢神经成分自动归为漂移或伪影。',
        '- 神经 Common 使用 q/LP80/LP250，Contrast 使用 q/LP80；不含 persistent step。LP750 单独消融。所有形变在 padded 时间轴生成后沿用原滤波及基线校正。',
        '- 空间模型先将训练 WC 自由读出映射到固定几何坐标的两等效源，再用训练 ERP 拟合 G；避免 G 与源振幅同时任意缩放。G0 是示意几何，不是受试者头模型。effective_linear_df_free_stage 仅指第一阶段线性自由度，不能作为两阶段总自由度。',
        '- 测试慢趋势仅由训练系数与测试已知 onset 得到。测试标签只用于条件 ERP 评估；不能将条件波形生成当成未知标签解码。所有模型使用相同原始稳健 heldout ERP 目标，预测 cue duration 使用训练中位数。',
        '- pooled R² = 1 − ΣSSE/ΣSST，SST 为每个 record×fold 的同一局部目标；不是逐折 R² 的平均。RMSE 按样本元素数加权。图是等权折平均，仅用于展示，计分仍按逐折目标。',
        '- ECG 使用训练数据核验周期性、QRS-like 波形一致性及 triggered EEG。未通过时 ECG/all-adjusted 明确记为未执行，不能把零改善伪装成回归结果。自动门槛不是临床通道认证；即使通过也仅作敏感性分析。',
        '', '## 分类冻结验证','', '|分组|n|BA|置换 p|','|---|---:|---:|---:|']
    cls=pd.read_csv(out/'classification_metrics.csv');lookup={p['group']:p for p in permutations}
    for _,r in cls[cls.variant=='main'].iterrows():
        p=lookup.get(r.group,{}).get('p_value')
        lines.append(f'|{r.group}|{r.n}|{r.ba:.4f}|{p:.4f}|' if p is not None else f'|{r.group}|{r.n}|{r.ba:.4f}|未运行|')
    status=pd.read_csv(out/'nuisance_status_v3.csv')
    failed=int(((status.variant=='ecg_adjusted')&(status.status!='computed')).sum())
    lines += ['', f'ECG-adjusted 在 {failed}/20 个记录×折未通过训练核验；见 nuisance_status_v3.csv。敏感性指标包含 coverage_complete，缺少折时不能与完整主结果直接比较。', '',
        '## 文件','',
        '- ablation_v3.csv：pooled、逐记录、逐折、四时间窗、joint/Common/Contrast、ΔR²/ΔSSE。',
        '- component_metrics_v3.csv、foldwise_delta_sse_v3.csv：record×fold 原始计分及配对模板差值。',
        '- forward_metrics_v3.csv、forward_parameters_v3.csv：free/symmetric/geometry 对照与各折 G、侧化系数。',
        '- slow_decomposition_v3.csv、heldout_erp_v3.npz：慢趋势、残余 ERP、神经预测，可重绘。',
        '- response_parameters_v3.json：所有候选内层损失、外层训练编号、slow 参数、响应系数。',
        '- figures/：慢趋势分解、C/D、250–500 ms 放大、空间图、M0–M5 消融和 triggered EEG。',
        '', '不能将当前 BA 称为严格理论极限，不能声称 Task2 理应随机、定位真实脑源或恢复真实突触参数。']
    ecg=json.loads((out/'ecg_permutation_test.json').read_text(encoding='utf-8'))
    if isinstance(ecg,list):
        lines += ['', '## ECG 敏感性独立置换','', '|分组|BA|次数|p|','|---|---:|---:|---:|']
        for r in ecg:
            p='未运行' if r['p_value'] is None else f"{r['p_value']:.4f}"
            lines.append(f"|{r['group']}|{r['ba']:.4f}|{r['permutations']}|{p}|")
        lines += ['', '使用与主分类不同的独立种子，逐记录×时间块内置换。每次重训分类标准化与 LR；标签无关、只在训练块拟合的混杂回归可复用。结果保持敏感性分析身份，不替换主方案。']
    (out/'第二问V3结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
