"""V3 nuisance controls: gate ECG interpretation using training-only diagnostics."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from .features import time_features
from .diagnostics import residualize, ecg_adjust


def verify_ecg(record, train, fs):
    """Conservative ECG-like gate, not a clinical QRS detector or channel identity proof."""
    intervals, counts, averages, templates = [], [], [], []
    mask = (record.padded_t >= -1.) & (record.padded_t <= 1.5)
    radius = round(.12*fs)
    for trial in np.flatnonzero(train):
        signal = record.ecg[trial]
        z = sosfiltfilt(butter(2, [5., 20.], btype='bandpass', fs=fs, output='sos'), signal)
        scale = max(float(1.4826*np.median(np.abs(z-np.median(z)))), 1e-8)
        peaks, _ = find_peaks(np.abs(z), distance=round(.35*fs), prominence=3*scale)
        peaks = peaks[mask[peaks]]
        counts.append(len(peaks))
        if len(peaks)>1:
            intervals.extend(np.diff(peaks)/fs)
        for peak in peaks:
            if peak-radius < 0 or peak+radius >= len(z):
                continue
            templates.append(z[peak-radius:peak+radius+1]*np.sign(z[peak]))
            eeg = record.raw_epochs[trial, :, peak-radius:peak+radius+1].copy()
            averages.append(eeg-eeg.mean(axis=-1, keepdims=True))
    rr = np.asarray(intervals)
    median = float(np.median(rr)) if len(rr) else 0.
    cv = float(np.std(rr)/max(np.mean(rr), 1e-8)) if len(rr) else float('inf')
    coverage = float(np.mean(np.array(counts)>=2)) if counts else 0.
    morphology = 0.
    if templates:
        waves = np.array(templates)
        waves /= np.maximum(np.linalg.norm(waves, axis=1, keepdims=True), 1e-12)
        morphology = float(np.linalg.norm(waves.mean(axis=0)))
    usable = len(rr)>=30 and .4<=median<=1.5 and cv<.3 and coverage>=.6 and morphology>=.65
    triggered = np.mean(averages, axis=0) if averages else np.zeros((3, 2*radius+1))
    return dict(usable_ecg_like=bool(usable), interval_count=len(rr), median_rr_s=median,
                rr_cv=cv if np.isfinite(cv) else None, trial_peak_coverage=coverage,
                morphology_consistency=morphology, triggered_eeg_average=triggered.tolist(),
                triggered_time_s=(np.arange(-radius, radius+1)/fs).tolist(),
                triggered_peak_to_peak=np.ptp(triggered, axis=-1).tolist(),
                training_trials=np.flatnonzero(train).tolist(),
                note='Training-only heuristic checks periodicity, QRS-like morphology and heartbeat-triggered EEG; does not prove ECG identity or causality.')


def nuisance_features(record, train, fs):
    raw = time_features(record.light, record.t)
    qc = record.metrics.reshape(len(record.labels), -1)
    order = np.arange(len(record.labels), dtype=float)[:, None]
    gate = verify_ecg(record, train, fs)
    quality, qa = residualize(raw, qc, train)
    ordinal, oa = residualize(raw, order, train)
    features = dict(raw=raw, qc_adjusted=quality, order_adjusted=ordinal)
    audit = dict(ecg_verification=gate, qc=qa, order=oa)
    if gate['usable_ecg_like']:
        adjusted, ea = ecg_adjust(record, train)
        ecg = time_features(adjusted, record.t)
        combined, ca = residualize(ecg, np.column_stack([qc, order]), train)
        features.update(ecg_adjusted=ecg, all_adjusted=combined)
        audit.update(ecg=ea, combined=ca)
    else:
        audit['ecg_status'] = 'not_run_channel7_not_verified'
        audit['all_status'] = 'not_run_requires_verified_ecg'
    return features, audit
