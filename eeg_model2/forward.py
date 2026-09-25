"""Two equivalent sources -> Fz/F3/F4. Not anatomical source localization."""
import numpy as np

FORWARD_MODES = ('free_readout', 'symmetric_forward', 'geometry_prior_forward')


def geometry_prior():
    # Mirrored, dimensionless illustrative dipoles, not measured head geometry.
    electrodes = np.array([[0., .8, .6], [-.5, .7, .5], [.5, .7, .5]])
    sources = np.array([[-.3, .2, .2], [.3, .2, .2]])
    displacement = electrodes[:, None, :] - sources[None, :, :]
    g = displacement[..., 2] / np.linalg.norm(displacement, axis=-1)**3
    return g / np.linalg.norm(g, axis=0, keepdims=True)


def apply_forward(g, sources):
    return np.asarray(sources) @ np.asarray(g).T


def fit_constrained_forward(sources, target, mode='symmetric_forward', strength=.1):
    """Fit G using training-only (..., 2) source and (..., 3) target arrays."""
    x, y = np.asarray(sources).reshape(-1, 2), np.asarray(target).reshape(-1, 3)
    if mode not in FORWARD_MODES:
        raise ValueError(mode)
    if mode == 'symmetric_forward':
        design = np.zeros((len(x), 3, 3))
        design[:, 0, 0] = x.sum(axis=1)
        design[:, 1, 1:] = x
        design[:, 2, 1:] = x[:, ::-1]
        a = design.reshape(-1, 3)
        coef = np.linalg.solve(a.T@a/len(x) + strength*np.eye(3), a.T@y.ravel()/len(x))
        a0, a1, a2 = coef
        g = np.array([[a0, a0], [a1, a2], [a2, a1]])
    else:
        prior = geometry_prior().T if mode == 'geometry_prior_forward' else np.zeros((2, 3))
        g = np.linalg.solve(x.T@x/len(x) + strength*np.eye(2), x.T@y/len(x) + strength*prior).T
    return g
