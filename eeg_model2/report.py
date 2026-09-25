"""训练拟合、留出ERP、单Trial预测分开出图和计分。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .shape import energy_maps


def fitted_curves(records,bank,model,cells):
    rows=[];archive={'time_s':bank.t}
    for j,r in enumerate(records):
        actual=cells[0][j]
        predicted=model.predict(bank,j,[-1,1],[cells[2][j]]*2)
        archive[r.name+'__target']=actual
        archive[r.name+'__fitted']=predicted
        mask=bank.t>=0
        for sign in (0,1):
            for ch in range(3):
                x=actual[sign,ch,mask];p=predicted[sign,ch,mask]
                sse=np.sum((x-p)**2);sst=np.sum((x-x.mean())**2)
                rows.append(dict(record=r.name,task=r.task,direction=-1 if sign==0 else 1,
                                 channel=['Fz','F3','F4'][ch],rmse=float(np.sqrt(np.mean((x-p)**2))),
                                 r2=float(1-sse/max(sst,1e-12)),sse=float(sse),sst=float(sst)))
    return pd.DataFrame(rows),archive


def figures(records,encoder,classification,pred,curves,out):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'figure.dpi':120,'savefig.dpi':160})
    dest=out/'figures';dest.mkdir(exist_ok=True)
    erp=np.load(out/'heldout_erp.npz');t=curves['time_s']
    colors=['#237a9b','#c44e52']
    for r in records:
        fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
        for ch in range(3):
            for sign in (0,1):
                label='左' if sign==0 else '右'
                axes[0,ch].plot(t,curves[r.name+'__target'][sign,ch],color=colors[sign],alpha=.45,label=label+'：训练稳健ERP')
                axes[0,ch].plot(t,curves[r.name+'__fitted'][sign,ch],color=colors[sign],ls='--',label=label+'：模型拟合')
                actual=np.mean([erp[f'{f}__{r.name}__heldout__{sign}__actual'][ch] for f in range(5)],axis=0)
                estimated=np.mean([erp[f'{f}__{r.name}__heldout__{sign}__predicted'][ch] for f in range(5)],axis=0)
                axes[1,ch].plot(t,actual,color=colors[sign],alpha=.45,label=label+'：留出稳健ERP')
                axes[1,ch].plot(t,estimated,color=colors[sign],ls='--',label=label+'：留出预测')
            for row in range(2):
                axes[row,ch].axvline(0,color='gray',lw=.6)
                axes[row,ch].axhline(0,color='gray',lw=.6)
                axes[row,ch].set_title(['Fz','F3','F4'][ch]);axes[row,ch].set_xlabel('时间（秒）')
                axes[row,ch].legend(fontsize=7)
        axes[0,0].set_ylabel('训练拟合\n原始文件幅值单位')
        axes[1,0].set_ylabel('留出预测\n原始文件幅值单位')
        fig.suptitle(r.name+'｜上：全数据再拟合；下：五个留出块的稳健ERP等权平均')
        fig.savefig(dest/(r.name+'_拟合与验证.png'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,4),layout='constrained')
    for task,ax in zip((1,2),axes):
        s=pred[(pred.task==task)&(pred.variant=='main')]
        cm=confusion_matrix(s.truth,s.prediction,labels=[-1,1]);ax.imshow(cm,cmap='Blues',vmin=0)
        for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black')
        ax.set_xticks([0,1],['左','右']);ax.set_yticks([0,1],['左','右'])
        ax.set_xlabel('预测');ax.set_ylabel('真实');ax.set_title(f'任务{task}：单Trial留出分类')
    fig.savefig(dest/'分类混淆矩阵.png');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
    for row,im in enumerate(encoder.images):
        lgn,energy=energy_maps(im)
        for ax,x,title in zip(axes[row],[im,lgn,energy.sum(axis=0)],['几何刺激','DoG局部对比','Gabor空间方向能量']):
            ax.imshow(x,cmap='magma' if title.startswith('Gabor') else 'gray');ax.set_title(title);ax.axis('off')
    fig.suptitle('视觉编码示意；两类群体表示形状偏好，非已定位的左右半球')
    fig.savefig(dest/'视觉编码.png');plt.close(fig)


def make_report(classification,wave,fit_metrics,erp,permutation,model,out,reference):
    train_r2=1-fit_metrics.sse.sum()/fit_metrics.sst.sum()
    hold=erp[erp.split=='heldout'];held_r2=1-hold.sse.sum()/hold.sst.sum()
    old=pd.read_csv(reference/'classification_metrics.csv')
    old_wave=pd.read_csv(reference/'waveform_metrics.csv')
    lines=['# 第二问结果报告','',
           '本项目仅保留一个主入口main.py。第一问程序和结果未修改；输入沿用30Hz低通、原始刺激前QC和基线规则，共333个有效Trial。',
           '模型：空间方向编码 → 稳定Wilson–Cowan群体 → 因果记忆/持续状态 → 共同与形状差异观测 → 噪声加权时空特征。','',
           '## 三类效果必须分开理解','',
           f'- 全数据训练稳健ERP的24条曲线汇总R²：**{train_r2:.4f}**；这是曲线拟合指标。',
           f'- 五折未见Trial的稳健ERP预测汇总R²：**{held_r2:.4f}**；汇总各折/记录SSE和SST，不把折当独立被试。',
           '- 单Trial分类如下；曲线拟合良好不代表能可靠识别每个Trial。','',
           '|任务|原主版BA|当前主流程BA|95%时间块条件区间|','|---|---:|---:|---|']
    for task in (1,2):
        row=classification[(classification.task==task)&(classification.variant=='main')].iloc[0]
        previous=old[(old.group==f'task{task}')&(old.variant=='mechanistic5')].balanced_accuracy.iloc[0]
        lines.append(f'|{task}|{previous:.4f}|{row.ba:.4f}|[{row.conditional_low:.4f}, {row.conditional_high:.4f}]|')
    lines+=['','条件区间仅对固定OOF预测做记录内时间块bootstrap1000次，不包括全流程重新训练。主流程由内层选择；不从外层消融中挑最高值替换主成绩。','',
            '|任务|完整分类流程置换次数|单侧探索性p值|','|---|---:|---:|']
    for p in permutation:lines.append(f"|{p['task']}|{p['permutations']}|{p['p_value'] if p['p_value'] is not None else '尚未运行'}|")
    lines+=['','置换在同记录同时间块内交换有效Trial标签，重新计算ERP、差异模板、类内噪声协方差、内层分类选择和最终分类器。波形专用参数不进入分类，因此不在分类置换中重复无关的波形选参。19次p值分辨率0.05，块内可交换性未由实验元数据独立确认。',
            '![混淆矩阵](figures/分类混淆矩阵.png)','', '## 曲线拟合明细','',
            '|记录|方向|电极|训练R²|训练RMSE|','|---|---|---|---:|---:|']
    for r in fit_metrics.itertuples():lines.append(f'|{r.record}|{"左" if r.direction<0 else "右"}|{r.channel}|{r.r2:.4f}|{r.rmse:.3f}|')
    lines+=['','## 留出单Trial波形对照','', '|记录|原模型0–500ms R²|当前模型0–500ms R²|当前模型0–800ms R²|','|---|---:|---:|---:|']
    for rec in model['records']:
        prev=old_wave[(old_wave.record==rec)&(old_wave.model=='task_gain')&(old_wave.window=='primary_0_500')].r2.iloc[0]
        short=wave[(wave.record==rec)&(wave.model=='response')&(wave.window=='0_500')].r2.iloc[0]
        full=wave[(wave.record==rec)&(wave.model=='response')&(wave.window=='0_800')].r2.iloc[0]
        lines.append(f'|{rec}|{prev:.4f}|{short:.4f}|{full:.4f}|')
    lines+=['','当前模型用完整0–800ms训练响应，原模型只拟合0–500ms，因此本表比较的是整个流程改动，不是只更换一项参数的因果消融。保留所有负R²；单Trial评分未降权困难测试样本。',
            '留出ERP的目标用留出组内QC×Huber估计，只用于评分；其QC尺度来自训练块。它与原始等权单Trial指标不同。','']
    for rec in model['records']:lines.append(f'![{rec}](figures/{rec}_拟合与验证.png)')
    response=model['response']
    lines+=['','## 参数与可辨识性','',
            '原时间常数拟合反复触界，本版固定τE=60ms、τI=120ms、延迟70ms、任务增益1；不再声称估计出真实突触时间常数或任务2生理增益。每种共同/差异响应使用6个固定因果基底。',
            f"全数据展示模型的Ridge={response['ridge']}，记录收缩={response['pool']}；名义系数{response['nominal_coefficients']}个，条件线性有效自由度约{response['effective_linear_df']:.2f}。有效自由度未包含ERP估计和调参复杂度。",
            '每条记录的左右响应共用共同系数和差异系数，通过同一形状符号正负变化产生。记录系数向同任务均值收缩；差异部分的Ridge是共同部分的10倍，防止把微弱差波过拟合。',
            '完整波形正则由内层共同/差异误差选择。分类特征使用固定Ridge=0.01、收缩=0.1的模型，避免波形调参越过分类内层边界。',
            '先从训练类内残差估计Ledoit–Wolf协方差，再计算差异模板的噪声加权投影；未分别把左右模板压成单位范数。候选为4幅值、24时空均值、24时空均值+差异分数，三种LR正则和一个收缩LDA对照。',
            '![编码](figures/视觉编码.png)','', '## 文件与适用边界','',
            '- oof_predictions.csv：每个有效Trial只在外层测试一次；classifier_selection.json记录内层选择。',
            '- curve_fit_metrics.csv和fitted_curves.npz：全数据训练拟合；erp_validation.csv：留出稳健ERP评分。',
            '- waveform_metrics.csv和heldout_waveforms.npz：单Trial原幅值预测，含共同响应和训练ERP消融。',
            '- final_model.json供predict.py使用；预测输入不接受真实标签，但需要已校准的记录标识。新记录需另行校准。',
            '- 本轮在先前已查看的数据上探索，不能把这333个Trial当新增独立测试集。不同记录不等于已确认的不同被试。',
            '- 只有三个额区通道，无独立眼电和头模型；形状选择性群体不表示左右半球，不作真实脑源定位。',
            '- 零相位滤波使用完整扩展窗，分类特征截止500ms不等于500ms在线实时决策。',
            '- 第一问处理结果不参与本次模型重写；分类接近机会水平时照实报告，不能由训练曲线的高R²替代证据。']
    (out/'第二问结果报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
