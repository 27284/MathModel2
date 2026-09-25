"""先报告留出与对照，再展示训练拟合；不以外层对照选择主方案。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from .shape import energy_maps
from .evaluation import WAVE_MODELS, WINDOWS, groups

NAMES = dict(main='逐记录LR（主方案）', record_lda='逐记录LDA', pooled='同任务池化LR',
             qc_weighted='逐记录LR+QC权重', mechanistic='逐记录LR+机理差异分',
             ecg_adjusted='训练内ECG回归', qc_adjusted='训练内QC回归',
             order_adjusted='训练内试次顺序回归', all_adjusted='联合回归', prestim_only='仅刺激前特征')
WAVE_NAMES = dict(response='内层选型响应', legacy_response='原结构响应',
                  common_only='删除差异项', train_erp='直接训练ERP模板')


def aggregate_r2(frame):
    return float(1 - frame.sse.sum() / max(frame.sst.sum(), 1e-12))


def fitted_curves(records, bank, model, cells):
    rows, archive = [], {'time_s': bank.t}
    for j, record in enumerate(records):
        actual = cells[0][j]
        predicted = model.predict(bank, j, [-1, 1], [cells[2][j]] * 2)
        archive[record.name + '__target'] = actual
        archive[record.name + '__fitted'] = predicted
        mask = bank.t >= 0
        for sign in (0, 1):
            for ch, name in enumerate(('Fz', 'F3', 'F4')):
                x, p = actual[sign, ch, mask], predicted[sign, ch, mask]
                sse, sst = np.sum((x - p) ** 2), np.sum((x - x.mean()) ** 2)
                rows.append(dict(record=record.name, task=record.task, direction=-1 if sign == 0 else 1,
                                 channel=name, rmse=float(np.sqrt(np.mean((x - p) ** 2))),
                                 r2=float(1 - sse / max(sst, 1e-12)), sse=float(sse), sst=float(sst)))
    return pd.DataFrame(rows), archive


def figures(records, encoder, classification, pred, curves, out):
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'figure.dpi': 120, 'savefig.dpi': 160})
    dest = out / 'figures'
    dest.mkdir(exist_ok=True)
    erp = np.load(out / 'heldout_erp.npz')
    t, colors = curves['time_s'], ['#237a9b', '#c44e52']
    for record in records:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), layout='constrained')
        for ch in range(3):
            for sign in (0, 1):
                label = '左' if sign == 0 else '右'
                axes[0, ch].plot(t, curves[record.name + '__target'][sign, ch], color=colors[sign], alpha=.5, label=label + '：训练ERP')
                axes[0, ch].plot(t, curves[record.name + '__fitted'][sign, ch], color=colors[sign], ls='--', label=label + '：训练拟合')
                actual = np.mean([erp[f'{f}__{record.name}__heldout__response__{sign}__actual'][ch] for f in range(5)], axis=0)
                axes[1, ch].plot(t, actual, color=colors[sign], alpha=.5, label=label + '：留出ERP')
                for model, style, title in [('response', '--', '模型预测'), ('train_erp', ':', '模板预测')]:
                    estimate = np.mean([erp[f'{f}__{record.name}__heldout__{model}__{sign}__predicted'][ch] for f in range(5)], axis=0)
                    axes[1, ch].plot(t, estimate, color=colors[sign], ls=style, label=label + '：' + title)
            for row in range(2):
                axes[row, ch].axvline(0, color='gray', lw=.6)
                axes[row, ch].axhline(0, color='gray', lw=.6)
                axes[row, ch].axvspan(.25, .5, color='#e9c46a', alpha=.08)
                axes[row, ch].set_title(('Fz', 'F3', 'F4')[ch])
                axes[row, ch].set_xlabel('时间（秒）'); axes[row, ch].legend(fontsize=7, ncols=2)
        axes[0, 0].set_ylabel('训练拟合\n原始文件幅值单位')
        axes[1, 0].set_ylabel('留出预测\n原始文件幅值单位')
        fig.suptitle(record.name + '｜上：全数据再拟合；下：五个留出块等权平均（含模板对照）')
        fig.savefig(dest / (record.name + '_拟合与验证.png')); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout='constrained')
    main = pred[pred.variant == 'main']
    displays = [(r.name, main[main.record == r.name]) for r in records]
    displays += [(f'任务{task}', main[main.task == task]) for task in (1, 2)]
    for ax, (name, s) in zip(axes.ravel(), displays):
        cm = confusion_matrix(s.truth, s.prediction, labels=[-1, 1])
        ax.imshow(cm, cmap='Blues', vmin=0)
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha='center', va='center', color='white' if value > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1], ['左', '右']); ax.set_yticks([0, 1], ['左', '右'])
        ax.set_xlabel('预测'); ax.set_ylabel('真实')
        ax.set_title(f'{name}\nBA={balanced_accuracy_score(s.truth, s.prediction):.3f}')
    fig.savefig(dest / '分类混淆矩阵.png'); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')
    for row, im in enumerate(encoder.images):
        lgn, energy = energy_maps(im)
        for ax, x, title in zip(axes[row], [im, lgn, energy.sum(axis=0)], ['几何刺激', 'DoG局部对比', 'Gabor空间方向能量']):
            ax.imshow(x, cmap='magma' if title.startswith('Gabor') else 'gray'); ax.set_title(title); ax.axis('off')
    fig.suptitle('形状编码示意；非真实皮层空间定位')
    fig.savefig(dest / '视觉编码.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, names, title in zip(axes, [list(NAMES)[:5], ['main'] + list(NAMES)[5:]],
                               ['预设分类对照（不按此图择优）', '训练内混杂敏感性（不判定生理来源）']):
        select = classification[classification.group == 'all'].set_index('variant').loc[names]
        y = np.arange(len(names))
        # 区间未必包住点估计，直接画区间端点，避免错误的负误差条。
        ax.hlines(y, select.conditional_low, select.conditional_high, color='#537b8a')
        ax.scatter(select.ba, y, color='#237a9b', zorder=3)
        for i, score in enumerate(select.ba): ax.annotate(f'{score:.3f}', (score, i), xytext=(4, 7), textcoords='offset points', fontsize=8)
        ax.axvline(.5, color='gray', ls='--'); ax.set_yticks(y, [NAMES[n] for n in names]); ax.invert_yaxis()
        ax.set_xlim(0.25, .8); ax.set_xlabel('全部333试次合并BA及条件95%区间'); ax.set_title(title)
    fig.savefig(dest / '对照与混杂敏感性.png'); plt.close(fig)
    table = pd.read_csv(out / 'erp_validation.csv')
    table = table[(table.split == 'heldout') & (table.component == 'joint')]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), layout='constrained')
    for ax, (window, _, _) in zip(axes, WINDOWS):
        values = [aggregate_r2(table[(table.model == name) & (table.window == window)]) for name in WAVE_MODELS]
        ax.bar(range(4), values, color=['#237a9b', '#819ca9', '#b8c4c8', '#c88742'])
        ax.axhline(0, color='gray', lw=.6)
        ax.set_xticks(range(4), ['主响应', '原结构', '仅共同', 'ERP模板'], rotation=35)
        for i, value in enumerate(values): ax.annotate(f'{value:.3f}', (i, value), ha='center', xytext=(0, 4 if value >= 0 else -12), textcoords='offset points', fontsize=8)
        ax.margins(y=.18); ax.set_title(window.replace('_', '–') + 'ms'); ax.set_ylabel('留出稳健ERP汇总R²')
    fig.savefig(dest / '分时窗ERP对照.png'); plt.close(fig)


def make_report(classification, wave, fit_metrics, erp, permutation, model, out, reference):
    main = classification[classification.variant == 'main']
    held = erp[(erp.split == 'heldout') & (erp.component == 'joint')]
    old_file = reference / 'review_baseline_oof.csv'
    previous = {}
    if old_file.exists():
        old = pd.read_csv(old_file); old = old[old.variant == 'main']
        previous = {name: balanced_accuracy_score(s.truth, s.prediction) for name, _, _, s in groups(old)}
    null = {p['group']: p for p in permutation}
    lines = ['# 第二问结果报告', '',
             '本轮根据评审修订：固定逐记录24维LR，加入公平ERP模板对照、混杂敏感性、分时窗评价及200次默认置换。第一问、原始数据、轻处理、333个有效试次和五个原始时间块不变。', '',
             '## 先看留出分类', '',
             '主方案固定为逐记录 temporal24 + LR(C=0.1)，不使用QC样本权重或机理差异分，不根据外层对照成绩改选LDA。每个试次由其他四个时间块训练的模型预测。', '',
             '|分组|有效试次|修订前主流程BA|当前BA|普通准确率|95%条件区间|置换p值|',
             '|---|---:|---:|---:|---:|---|---:|']
    for row in main.itertuples():
        p = null.get(row.group, {})
        prior = f'{previous[row.group]:.4f}' if row.group in previous else '—'
        pv = str(round(p['p_value'], 6)) if p.get('p_value') is not None else '待计算'
        lines.append(f'|{row.group}|{row.n}|{prior}|{row.ba:.4f}|{row.accuracy:.4f}|[{row.conditional_low:.4f}, {row.conditional_high:.4f}]|{pv}|')
    count = max([p['permutations'] for p in permutation], default=0)
    lines += ['', f'已完成{count}次主分类流程置换；最小可达p值为1/({count}+1)' + (f'={1/(count+1):.6f}。' if count else '；当前尚无置换结论。'),
              '置换只在同记录同时间块的有效试次间交换标签，每次重新训练标准化及固定分类器。主方案不依赖监督模板或波形选参，因此无需置换独立波形流程。总合并结果为主描述，其余分组p值未经多重比较校正，作为探索性结果。',
              '区间对固定OOF预测做记录内整块bootstrap1000次，不包含模型重训不确定性。块内可交换性未经实验元数据独立确认；这也不是跨记录或跨被试泛化。修订前后同时改变了分类器选择、池化和权重，整体前后比较不等于单因素因果对照。', '',
              '![分类混淆矩阵](figures/分类混淆矩阵.png)', '', '## 预设对照与混杂敏感性', '',
              '|方案|全部BA|任务1 BA|任务2 BA|', '|---|---:|---:|---:|']
    for variant, name in NAMES.items():
        s = classification[classification.variant == variant].set_index('group')
        lines.append(f'|{name}|{s.loc["all", "ba"]:.4f}|{s.loc["task1", "ba"]:.4f}|{s.loc["task2", "ba"]:.4f}|')
    lines += ['', 'ECG、QC、试次顺序及联合回归的系数与标准化尺度只用训练块估计。ECG对照使用既有1–25Hz辅助心电与±约40/80ms滞后项；QC回归使用三通道共12项刺激前指标；试次顺序保留原始100试次位置。仅刺激前特征取自未零相位滤波的原始基线。',
              '这些是探索性敏感性对照，未分别做置换检验，也不根据其外层高低选择主模型。回归后成绩下降可能来自混杂或共同变异被移除，成绩不变也不能证明成分具有神经来源。未把诊断结果用于改写第一问处理。', '',
              '![对照与混杂敏感性](figures/对照与混杂敏感性.png)', '', '## 公平的留出ERP预测对照', '',
              '|模型|0–800ms R²|0–500ms R²|250–500ms R²|600–800ms R²|', '|---|---:|---:|---:|---:|']
    for name in WAVE_MODELS:
        values = [aggregate_r2(held[(held.model == name) & (held.window == w)]) for w, _, _ in WINDOWS]
        lines.append('|' + WAVE_NAMES[name] + '|' + '|'.join(f'{v:.4f}' for v in values) + '|')
    main_r2 = aggregate_r2(held[(held.model == 'response') & (held.window == '0_800')])
    template_r2 = aggregate_r2(held[(held.model == 'train_erp') & (held.window == '0_800')])
    comparison = '高于' if main_r2 > template_r2 else '未超过'
    lines += ['', f'本轮主响应的0–800ms留出ERP汇总R²为{main_r2:.4f}，{comparison}直接训练ERP模板的{template_r2:.4f}。这只是本数据上的预测比较，不等于证明神经机制正确。',
              '所有行使用同一留出目标、质量权重规则、时间窗与SST定义。分时窗保留负R²；汇总按1−总SSE/总SST计算，不平均R²。erp_validation.csv另含共同/差异分量评分，可检查整体成绩是否掩盖微弱差波。',
              '主响应在原结构和受限差异结构的24组候选中做内层选择；原结构对照只在自身12组正则中选择。共同项消融是在主拟合上删除差异项，没有另行重拟合。训练ERP模板只读取训练条件ERP，留出ERP只用于评分。', '',
              '![分时窗ERP对照](figures/分时窗ERP对照.png)', '', '## 单试次波形预测', '',
              '|记录|主响应0–500ms R²|主响应0–800ms R²|训练ERP模板0–800ms R²|', '|---|---:|---:|---:|']
    for record in model['records']:
        def value(name, window):
            return float(wave[(wave.record == record) & (wave.model == name) & (wave.window == window)].r2.iloc[0])
        lines.append(f'|{record}|{value("response", "0_500"):.4f}|{value("response", "0_800"):.4f}|{value("train_erp", "0_800"):.4f}|')
    lines += ['', '单试次波形生成以已知视觉刺激方向为条件；左右分类则不读取待预测试次标签。两者不可混同。单试次指标按原幅值等权评分，没有降低困难测试试次的权重。', '',
              '## 训练曲线与参数（不是识别率）', '',
              f'全数据24条训练稳健ERP曲线汇总R²={aggregate_r2(fit_metrics):.4f}，其中{int((fit_metrics.r2 < 0).sum())}条为负。训练拟合、留出ERP、单试次波形的评价对象不同，不能把不同层次的差距全部归因于过拟合。', '',
              '|记录|方向|电极|训练R²|训练RMSE|', '|---|---|---|---:|---:|']
    for row in fit_metrics.itertuples():
        lines.append(f'|{row.record}|{"左" if row.direction < 0 else "右"}|{row.channel}|{row.r2:.4f}|{row.rmse:.3f}|')
    response = model['response']
    lines += ['', f'全数据展示模型：结构={response["structure"]}，ridge={response["ridge"]}，pool={response["pool"]}；名义系数{response["nominal_coefficients"]}，条件线性有效自由度{response["effective_linear_df"]:.2f}（不含ERP估计和模型选择复杂度）。',
              '固定τE=60ms、τI=120ms、驱动延迟70ms和任务增益1。共同/差异分别拟合读出系数，本来就不存在差异波形只能是共同波形缩放副本的硬约束。受限候选用±40ms神经源平移替代差异项直接持续阶跃，不代表彻底消除了慢漂移。',
              'trend_metrics.csv保存全数据条件ERP的描述性线性趋势及斜率；趋势解释率高只说明曲线形态，不能单凭它判定心源、眼源或神经来源。', '']
    for record in model['records']: lines.append(f'![{record}](figures/{record}_拟合与验证.png)')
    lines += ['', '## 适用边界与文件', '',
              '- 已查看过这些数据，本轮属于探索性修订，333个试次不是新独立测试集；不把55%称为已证明的准确率上限。',
              '- 当前只有三个额区通道；形状偏好群体仍未保留真实皮层布局，无解剖依据的空间前向模型。dipole_demo未接入主拟合，不能声称完成真实脑源定位。',
              '- 原始幅值单位未确认为微伏；零相位滤波使用完整扩展窗，因此不是500ms实时分类。',
              '- final_model.json为schema_version=2的四个逐记录分类器；predict.py仅接受同处理、已校准记录的新试次，不能冒用记录标识宣称跨记录泛化。',
              '- classifier_audit.json保存每折预设方案、标准化、系数和训练试次；confound_audit.json保存训练内混杂回归。',
              '- oof_predictions.csv、classification_metrics.csv和confound_metrics.csv保存逐试次与分组成绩；erp_validation.csv及waveform_metrics.csv保存公平波形对照。',
              '- manifest.json绑定代码、数据、环境和核心结果指纹，原处理文件保持冻结。只保留一个正式入口与结果目录。', '',
              '![视觉编码](figures/视觉编码.png)', '']
    (out / '第二问结果报告.md').write_text('\n'.join(lines), encoding='utf-8')
