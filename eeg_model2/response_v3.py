"""V3 response fits; historical response.py remains reproducible for V2."""
from dataclasses import dataclass, asdict, replace
import numpy as np
from .forward import geometry_prior, fit_constrained_forward


@dataclass(frozen=True)
class Setting:
    slow: str = 'none'
    family: str = 'wc'
    warp: str = 'base'
    lambda_common: float = .1
    lambda_contrast: float = 1.
    pool: float = 0.
    forward: str = 'free_readout'
    lambda_forward: float = .1
    lp750: bool = False


@dataclass
class ResponseV3:
    setting: Setting
    coefficients: list
    effective_df: float
    forward_matrices: list
    source_coefficients: list

    def neural(self, bank, record, duration):
        b = bank.get(duration, self.setting.family, self.setting.warp, self.setting.lp750)
        return np.array([self.coefficients[c][record].T @ b[c] for c in (0, 1)])

    def to_dict(self):
        return dict(setting=asdict(self.setting), coefficients=[c.tolist() for c in self.coefficients],
                    effective_linear_df_free_stage=self.effective_df,
                    forward_matrices=[g.tolist() for g in self.forward_matrices],
                    source_coefficients=[[c.tolist() for c in pair] for pair in self.source_coefficients],
                    note='Forward: training free-stage coefficients calibrated to fixed geometry source coordinates, then training G fit. df describes free stage, not total nonlinear df.')


def fit_response_v3(records, bank, curves, neff, durations, setting):
    n = len(records)
    lap = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if records[i].task == records[j].task:
                lap[i, i] += .5; lap[j, j] += .5
                lap[i, j] -= .5; lap[j, i] -= .5
    mask = bank.t >= 0
    weight = 1/(1/neff[:, 0]+1/neff[:, 1]); weight /= weight.mean()
    targets = (curves.mean(axis=1), (curves[:, 1]-curves[:, 0])/2)
    coefficients, df = [], 0.
    bases = [bank.get(d, setting.family, setting.warp, setting.lp750) for d in durations]
    for c, ridge in enumerate((setting.lambda_common, setting.lambda_contrast)):
        p = bases[0][c].shape[0]
        gram, rhs = np.zeros((n*p, n*p)), np.zeros((n*p, 3))
        for j in range(n):
            x = bases[j][c][:, mask].T
            sl = slice(j*p, (j+1)*p)
            gram[sl, sl] = weight[j] * x.T@x/len(x)
            rhs[sl] = weight[j] * x.T@targets[c][j][:, mask].T/len(x)
        penalty = ridge*np.eye(n*p) + setting.pool*np.kron(lap, np.eye(p))
        coefficients.append(np.linalg.solve(gram+penalty, rhs).reshape(n, p, 3))
        df += 3*float(np.trace(np.linalg.solve(gram+penalty, gram)))
    matrices, source_coefficients = [], []
    if setting.forward != 'free_readout':
        inverse = np.linalg.pinv(geometry_prior().T)
        for j in range(n):
            # Identified source coordinate gauge is fixed, so G and q cannot freely rescale.
            source_coef = [coefficients[c][j] @ inverse for c in (0, 1)]
            latent = [bases[j][c][:, mask].T @ source_coef[c] for c in (0, 1)]
            x = np.concatenate([latent[0]-latent[1], latent[0]+latent[1]])
            y = curves[j][..., mask].transpose(0, 2, 1).reshape(-1, 3)
            g = fit_constrained_forward(x, y, setting.forward, setting.lambda_forward)
            for c in (0, 1):
                coefficients[c][j] = source_coef[c] @ g.T
            matrices.append(g); source_coefficients.append(source_coef)
    return ResponseV3(setting, coefficients, df, matrices, source_coefficients)
