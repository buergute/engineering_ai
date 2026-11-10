"""Train quantile models using explicit train/validation/calibration inputs.

The caller owns split construction and persists model artifacts. Test data must
never be supplied to this module. LightGBM is imported lazily so feature and
metric tests work without it installed.
"""
from __future__ import annotations

import numpy as np
from models.calibration import cqr_offset


def fit_quantiles(x_train, y_train, x_val, y_val, x_calib, y_calib, *,
                  seed=42, coverage=0.8, n_estimators=1000,
                  learning_rate=0.03, num_leaves=15, early_stopping_rounds=50):
    """Fit three quantiles, using val only for early stopping and calib for CQR.

    Inputs must already be constructed from disjoint chronological splits.
    Returns models and one global calibration offset; group calibration is not
    implemented here. The caller must record split provenance.
    """
    import lightgbm as lgb

    features = list(x_train.columns)
    if not features or len(set(features)) != len(features):
        raise ValueError("training features must be unique and nonempty")
    for x, y in ((x_train, y_train), (x_val, y_val), (x_calib, y_calib)):
        if list(x.columns) != features or len(x) != len(y) or not len(x):
            raise ValueError("split columns or lengths do not match")
        if not np.isfinite(np.asarray(y, dtype=float)).all():
            raise ValueError("targets must be finite")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between zero and one")
    alpha = (1 - coverage) / 2
    models = {}
    for label, quantile in (("lower", alpha), ("median", .5), ("upper", 1 - alpha)):
        model = lgb.LGBMRegressor(objective="quantile", alpha=quantile,
                                  n_estimators=n_estimators, learning_rate=learning_rate,
                                  num_leaves=num_leaves, random_state=seed, verbosity=-1,
                                  deterministic=True, force_col_wise=True, n_jobs=1)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
                  callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
        models[label] = model
    predictions = np.column_stack([models[k].predict(x_calib) for k in ("lower", "upper")])
    lower, upper = predictions.min(axis=1), predictions.max(axis=1)
    offset = cqr_offset(y_calib, lower, upper, coverage)
    return {"models": models, "features": features, "offset": offset, "coverage": coverage}
