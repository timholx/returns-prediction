import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

X_MOM_HALFLIFE_GRID = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]  # in 10 s bars
SPREAD_Z_WIN = 60
GAP_THRESHOLD_SEC = 60  # a longer gap between bars starts a new session
VOL_WIN = 360
ALPHA_GRID = [1e3, 1e4, 3e4, 1e5, 3e5, 1e6]
N_CV_FOLDS = 6
CV_FIRST_TRAIN_FRAC = 0.3  # the first fold trains on the first 30% of rows

PREDICTION_SCALE = 0.8  # shrink towards zero: the signal is weak and its strength drifts


def _session_momentum(df, halflife):
    gap_sec = pd.to_datetime(df["timestamp"], unit="ms").diff().dt.total_seconds().fillna(0.0).values
    is_open = gap_sec > GAP_THRESHOLD_SEC
    is_open[0] = True
    opens = np.flatnonzero(is_open)
    session = np.cumsum(is_open) - 1
    steps = np.arange(len(df)) - opens[session]

    dx = df["xprice"].diff().fillna(0.0).to_numpy(dtype=float, copy=True)
    dx[is_open] = 0.0  # the first change of a session spans the overnight break
    a = 1.0 - 0.5 ** (1.0 / halflife)
    m = pd.Series(dx).ewm(alpha=a, adjust=False).mean().values
    carried = np.r_[0.0, m][opens]  # value just before each open
    momentum = m - (1.0 - a) ** (steps + 1) * carried[session]  # same as restarting the average at each open

    # divide by X's recent volatility so the feature measures how unusual the move is
    vol = np.sqrt(pd.Series(dx ** 2).rolling(VOL_WIN, min_periods=10).mean().values)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vol > 1e-4, momentum / vol, 0.0)


def _build_features(df, x_mom_halflife):
    spread = df["xprice"] - df["yprice"]
    feats = pd.DataFrame({
        "x_mom": _session_momentum(df, x_mom_halflife),
        "spread_z": (spread - spread.rolling(SPREAD_Z_WIN).mean()) / spread.rolling(SPREAD_Z_WIN).std(),
    })
    return feats.fillna(0.0).replace([np.inf, -np.inf], 0.0).values


def _fit(X, y, alpha):
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=alpha).fit(scaler.transform(X), y)
    return scaler, model


def _walk_forward_mse(X, y, alpha):
    # expanding window: fit on all rows before each test block, then score on that block
    edges = np.linspace(int(len(y) * CV_FIRST_TRAIN_FRAC), len(y), N_CV_FOLDS + 1).astype(int)
    sq_errors = []
    for start, end in zip(edges[:-1], edges[1:]):
        scaler, model = _fit(X[:start], y[:start], alpha)
        pred = model.predict(scaler.transform(X[start:end]))
        sq_errors.append((y[start:end] - pred) ** 2)
    return np.concatenate(sq_errors).mean()


def _select_hyperparams(df, y, valid):
    cv_mse = {}
    for halflife in X_MOM_HALFLIFE_GRID:
        X = _build_features(df, halflife)[valid]
        for alpha in ALPHA_GRID:
            cv_mse[(halflife, alpha)] = _walk_forward_mse(X, y, alpha)
    return min(cv_mse, key=cv_mse.get)


def modelEstimate(trainingFilename):
    df = pd.read_csv(trainingFilename)
    valid = np.isfinite(df["returns"].values)
    y = df["returns"].values[valid]

    halflife, alpha = _select_hyperparams(df, y, valid)
    X = _build_features(df, halflife)[valid]
    scaler, model = _fit(X, y, alpha)

    return {"model": model, "scaler": scaler, "alpha": alpha, "x_mom_halflife": halflife}


def modelForecast(testFilename, parameters):
    df = pd.read_csv(testFilename)
    X = _build_features(df, parameters["x_mom_halflife"])
    X_scaled = parameters["scaler"].transform(X)
    return PREDICTION_SCALE * parameters["model"].predict(X_scaled)
