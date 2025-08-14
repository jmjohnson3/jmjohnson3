# !/usr/bin/env python3
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Any, Tuple, Union
import math
from scipy.stats import poisson
from aiohttp import TCPConnector, BasicAuth
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score
import difflib
import re
import asyncpg
import urllib.parse
from pybaseball import statcast
import lightgbm as lgb
import pandas as pd
from jinja2 import Template
import smtplib
from email.message import EmailMessage
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
# at the top of your script
from tqdm import tqdm
import logging, lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

import warnings, logging, numpy as np

# 1) Lift log level so you don’t see every HTTP/asyncio debug message
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 2) Globally suppress only the known warnings you’re handling explicitly
warnings.filterwarnings("ignore", message="errors='ignore' is deprecated")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=FutureWarning)  # if you want to suppress any leftover FutureWarnings
warnings.filterwarnings("ignore", category=UserWarning)  # careful—this hides _all_ UserWarnings; or be more specific
# 1) Lift log level so you don’t see every HTTP/asyncio debug message
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# 2) Globally suppress only the known warnings you’re handling explicitly
warnings.filterwarnings("ignore", message="errors='ignore' is deprecated")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 3) Optionally silence numpy runtime warnings too
np.seterr(all='ignore')

# 3) Optionally silence numpy runtime warnings too
np.seterr(all='ignore')

# --- Configuration ---
API_PREFIX = "https://api.mysportsfeeds.com/v2.1/pull/mlb"
API_USER = "4359aa1b-cc29-4647-a3e5-7314e2"
API_PASS = "MYSPORTSFEEDS"
SEASONS = ["2024-regular", "2025-regular"]
BACKOFF = {
    "player_gamelogs": 5,
    "team_gamelogs": 5,
    "team_stats_totals": 5,
    "player_stats_totals": 5,
    "standings": 5,
    "lineups": 5,
}
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
ODDS_EVENTS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
ODDS_API_KEY = "5b6f0290e265c3329b3ed27897d79eaf"
ODDS_REGIONS = ["us"]
ODDS_FORMAT = "decimal"
PROP_MARKETS = {
    "batter_total_bases": "TB",
    "batter_home_runs": "HR",
    "pitcher_strikeouts": "K",
}
PROP_TYPE_KEYWORDS = {
    "Home Runs": "HR",
    "Total Bases": "TB",
}

# Logging setup
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# --- Database caching helpers ---
async def init_db_pool() -> asyncpg.Pool:
    user = os.getenv("PGUSER", "josh")
    password = os.getenv("PGPASSWORD", "password")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    database = os.getenv("PGDATABASE", "mlb")
    dsn = f"postgresql://{user}:{urllib.parse.quote_plus(password)}@{host}:{port}/{database}?sslmode=disable"
    return await asyncpg.create_pool(dsn)


async def ensure_table(pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS json_cache (
                feed TEXT NOT NULL,
                season TEXT NOT NULL,
                key TEXT NOT NULL,
                data JSONB NOT NULL,
                PRIMARY KEY(feed, season, key)
            );
        """)


async def ensure_feedback_table(pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS prop_feedback (
            feedback_date DATE      NOT NULL,
            prop         TEXT      NOT NULL,
            entity_id    INTEGER   NOT NULL,
            actual       REAL      NOT NULL,
            predicted    REAL      NOT NULL,
            error        REAL      NOT NULL,
            features     JSONB     NOT NULL,
            PRIMARY KEY(feedback_date, prop, entity_id)
        );
        """)


async def save_prop_feedback(
        pool: asyncpg.Pool,
        date: date,
        prop: str,
        df_feedback: pd.DataFrame,
        feature_cols: List[str]
):
    # df_feedback must have columns: entity_id, actual, predicted
    # plus all feature_cols
    records = []
    for _, row in df_feedback.iterrows():
        feat_json = row[feature_cols].to_dict()
        records.append({
            'feedback_date': date,
            'prop': prop,
            'entity_id': int(row['entity_id']),
            'actual': float(row['actual']),
            'predicted': float(row['predicted']),
            'error': float(row['actual'] - row['predicted']),
            'features': json.dumps(feat_json)
        })

    insert_sql = """
    INSERT INTO prop_feedback
      (feedback_date, prop, entity_id, actual, predicted, error, features)
    VALUES ($1,$2,$3,$4,$5,$6,$7)
    ON CONFLICT (feedback_date, prop, entity_id) DO NOTHING
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            for r in records:
                await conn.execute(insert_sql,
                                   r['feedback_date'], r['prop'], r['entity_id'],
                                   r['actual'], r['predicted'], r['error'], r['features']
                                   )


async def load_prop_feedback(
        pool: asyncpg.Pool,
        prop: str,
        days: int = 30
) -> pd.DataFrame:
    cutoff = date.today() - timedelta(days=days)
    sql = """
    SELECT feedback_date
         , entity_id
         , actual
         , predicted
         , error
         , features
    FROM prop_feedback
    WHERE prop = $1
      AND feedback_date >= $2
    ORDER BY feedback_date
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, prop, cutoff)

    if not rows:
        return pd.DataFrame()

    # turn each asyncpg.Record into a real dict
    records = [dict(r) for r in rows]
    df = pd.DataFrame.from_records(records)

    # asyncpg already gives you JSONB as a Python dict,
    # so you can normalize it directly
    # (if you ever store it as a string, you can do
    #  df['features'] = df['features'].apply(json.loads) first)
    feats = pd.json_normalize(df['features'])

    # drop the raw JSON column and glue in your flattened features
    df = pd.concat([df.drop(columns=['features']), feats], axis=1)
    return df


async def load_json(pool, feed, season, key):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT data FROM json_cache WHERE feed=$1 AND season=$2 AND key=$3",
            feed, season, key
        )
    if not row:
        return None

    val = row["data"]
    # if it comes back as a string, parse it:
    if isinstance(val, str):
        return json.loads(val)
    return val


async def save_json(pool: asyncpg.Pool, feed: str, season: str, key: str, data: Any):
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO json_cache(feed, season, key, data)
            VALUES($1,$2,$3,$4)
            ON CONFLICT (feed, season, key) DO UPDATE
              SET data = EXCLUDED.data;
            """,
            feed, season, key, json.dumps(data)
        )


async def get_or_fetch(
        session: aiohttp.ClientSession,
        pool: asyncpg.Pool,
        url: str,
        feed: str,
        season: str,
        key: str
) -> Any:
    # try cache
    cached = await load_json(pool, feed, season, key)
    if cached is not None:
        return cached

    # fetch from API (note: only session, url, feed!)
    data = await fetch_json(session, url, feed)

    # save and return
    await save_json(pool, feed, season, key, data)
    return data


# --- Original network fetch with backoff ---
async def fetch_json(session: aiohttp.ClientSession, url: str, feed: str) -> Dict[str, Any]:
    backoff = BACKOFF.get(feed, 0)
    while True:
        try:
            async with session.get(url, auth=BasicAuth(API_USER, API_PASS)) as resp:
                if resp.status == 429:
                    await asyncio.sleep(backoff)
                    continue
                if resp.status == 204:
                    return {}
                resp.raise_for_status()
                return await resp.json()
        except Exception:
            raise


# --- Data gathering, now using DB cache ---
async def gather_seasonal(session: aiohttp.ClientSession, pool: asyncpg.Pool) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for season in SEASONS:
        base = f"{API_PREFIX}/{season}"
        # games
        j_games = await get_or_fetch(session, pool, f"{base}/games.json", "games", season, "games")
        dfs[f"games_{season}"] = pd.json_normalize(j_games.get("games", []))
        # venues
        j_venues = await get_or_fetch(session, pool, f"{base}/venues.json", "venues", season, "venues")
        dfs[f"venues_{season}"] = pd.json_normalize(j_venues.get("venues", []))
        # team stats totals
        j_team = await get_or_fetch(session, pool, f"{base}/team_stats_totals.json", "team_stats_totals", season,
                                    "team_stats_totals")
        team_df = pd.json_normalize(j_team.get("teamStatsTotals", []))
        gp_col = next(c for c in team_df.columns if c.lower().endswith("gamesplayed"))
        team_df["runs_per_game"] = team_df["stats.batting.runs"] / team_df[gp_col]
        team_df["runs_allowed_per_game"] = team_df["stats.standings.runsAgainst"] / team_df[gp_col]
        dfs[f"team_stats_totals_{season}"] = team_df
        # player stats totals
        j_player = await get_or_fetch(session, pool, f"{base}/player_stats_totals.json", "player_stats_totals", season,
                                      "player_stats_totals")
        df_player = pd.json_normalize(j_player.get("playerStatsTotals", []))
        # derive vs L/R splits
        # in gather_seasonal, after you compute df_player:
        for side in ['vsLeft', 'vsRight']:
            # batting average
            avg_col = next((c for c in df_player.columns
                            if side in c and 'battingaverage' in c.lower()), None)
            df_player[f"avg_{side}"] = df_player[avg_col].fillna(0) if avg_col else 0

            # home runs
            hr_col = next((c for c in df_player.columns
                           if side in c and 'homeruns' in c.lower()), None)
            df_player[f"hr_{side}"] = df_player[hr_col].fillna(0) if hr_col else 0

            # total bases
            tb_col = next((c for c in df_player.columns
                           if side in c and 'totalbases' in c.lower()), None)
            df_player[f"tb_{side}"] = df_player[tb_col].fillna(0) if tb_col else 0

        # in gather_seasonal(), after your existing for side in ['vsLeft','vsRight'] block:
        tb_col = next((c for c in df_player.columns if side in c and 'totalBases' in c), None)
        df_player[f"tb_{side}"] = df_player[tb_col].fillna(0) if tb_col else 0

        # also grab the pitcher’s throws‐hand so you can look it up later:
        throws_col = next((c for c in df_player.columns if 'throws' in c.lower()), None)
        if throws_col:
            df_player = df_player.rename(columns={throws_col: 'throwsSide'})

        # pitching stats
        for colname in ['earnedrunaverage', 'whip', 'strikeoutsper9']:
            col = next((c for c in df_player.columns if colname in c.lower()), None)
            if col:
                df_player[col.split('.')[-1]] = df_player[col]
        dfs[f"player_stats_totals_{season}"] = df_player
        # standings
        j_stand = await get_or_fetch(session, pool, f"{base}/standings.json", "standings", season, "standings")
        data = j_stand.get("standings") or next((v for v in j_stand.values() if isinstance(v, list)), [])
        dfs[f"standings_{season}"] = pd.json_normalize(data)
    return dfs


async def gather_all_gamelogs(
        session: aiohttp.ClientSession,
        seasonal: Dict[str, pd.DataFrame],
        pool: asyncpg.Pool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_player, all_team = [], []
    yesterday = date.today() - timedelta(days=1)
    current_season = SEASONS[-1]
    for season in SEASONS:
        games_df = seasonal.get(f"games_{season}", pd.DataFrame())
        if games_df.empty:
            continue
        dates = sorted(pd.to_datetime(games_df["schedule.startTime"]).dt.date.unique())
        for d in tqdm(dates, desc=f"Fetching {season}", unit="day"):
            if season == current_season and d > yesterday:
                break
            ds = d.strftime("%Y%m%d")
            base = f"{API_PREFIX}/{season}/date/{ds}"
            for feed, col in [("player_gamelogs", "playerGamelogs"), ("team_gamelogs", "teamGamelogs")]:
                url = f"{base}/{feed}.json"
                try:
                    j = await get_or_fetch(session, pool, url, feed, season, ds)
                    arr = j.get(col) or next((v for v in j.values() if isinstance(v, list)), [])
                    df = pd.json_normalize(arr)
                    if df.empty:
                        continue
                    df["season"] = season
                    if feed == "player_gamelogs":
                        all_player.append(df)
                    else:
                        all_team.append(df)
                except Exception:
                    logger.warning("Skipping %s on %s (%s)", feed, ds, season)
    player_logs = pd.concat(all_player, ignore_index=True) if all_player else pd.DataFrame()
    team_logs = pd.concat(all_team, ignore_index=True) if all_team else pd.DataFrame()
    return player_logs, team_logs


async def gather_daily(
        session: aiohttp.ClientSession,
        pool: asyncpg.Pool,
        log_date: Union[str, date],
        sched_date: Union[str, date]
) -> Dict[str, Any]:
    season = SEASONS[-1]
    out: Dict[str, Any] = {}

    # 1) normalize inputs to "YYYYMMDD" strings
    if isinstance(log_date, date):
        log_date_str = log_date.strftime("%Y%m%d")
    else:
        log_date_str = log_date
    if isinstance(sched_date, date):
        sched_date_str = sched_date.strftime("%Y%m%d")
    else:
        sched_date_str = sched_date

    # 2) yesterday's games
    url_yest = f"{API_PREFIX}/{season}/games.json?date={log_date_str}"
    j_yest = await get_or_fetch(session, pool, url_yest, "games", season, f"games_{log_date_str}")
    out["yesterday_games"] = pd.json_normalize(j_yest.get("games", []))

    # 3) today's games
    url_today = f"{API_PREFIX}/{season}/games.json?date={sched_date_str}"
    j_today = await get_or_fetch(session, pool, url_today, "games", season, f"games_{sched_date_str}")
    raw_games = j_today.get("games", [])
    out["today_games_json"] = raw_games
    df_today = pd.json_normalize(raw_games)

    # 4) find away/home abbreviation columns robustly
    away_cols = [c for c in df_today.columns if "awayteam.abbreviation" in c.lower()]
    home_cols = [c for c in df_today.columns if "hometeam.abbreviation" in c.lower()]
    if not away_cols or not home_cols:
        raise KeyError(
            f"Could not find away/home abbreviation columns in daily_sched. "
            f"Available columns: {df_today.columns.tolist()}"
        )
    away_col, home_col = away_cols[0], home_cols[0]

    # 5) rename and return
    df_today = df_today.rename(columns={
        away_col: "away_team_abbr",
        home_col: "home_team_abbr"
    })
    out["today_games"] = df_today

    # 6) fetch lineups, etc. (unchanged)
    lineups = []
    for game in raw_games:
        gid = game.get("schedule", {}).get("id")
        if not gid:
            continue
        try:
            url = f"{API_PREFIX}/{season}/games/{gid}/lineup.json"
            raw = await get_or_fetch(session, pool, url, "lineups", season, str(gid))
            j = raw.get("lineup", raw)
            lineups.append(j)
        except Exception:
            logger.warning("Failed to fetch lineup for game %s", gid)
    out["today_lineups"] = lineups

    return out


# --- All remaining get_*, build_*, train_* functions remain unchanged ---
def get_park_factor_map(seasonal: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    # ... (unchanged) ...
    season = SEASONS[-1]
    df_all_games = seasonal[f"games_{season}"].copy()
    venue_col = next(c for c in df_all_games.columns if c.lower().endswith("venue.id"))
    away_sc = next(c for c in df_all_games.columns if "awayscore" in c.lower())
    home_sc = next(c for c in df_all_games.columns if "homescore" in c.lower())
    df_all_games["total_runs"] = df_all_games[away_sc].fillna(0) + df_all_games[home_sc].fillna(0)
    venue_stats = (
        df_all_games.groupby(venue_col, as_index=False)
        .agg(runs_per_game=pd.NamedAgg(column="total_runs", aggfunc="mean"),
             games_played=pd.NamedAgg(column="total_runs", aggfunc="count"))
    )
    league_avg = df_all_games["total_runs"].mean()
    venue_stats["park_factor"] = venue_stats["runs_per_game"] / league_avg
    return venue_stats.set_index(venue_col)["park_factor"].to_dict()


import logging
import numpy as np
from scipy.stats import poisson
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold, cross_val_score
import lightgbm as lgb

logger = logging.getLogger(__name__)

# --- Game Ensemble Fix ---
def train_and_predict_games_ensemble(
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_today: pd.DataFrame,
        games_today: pd.DataFrame
) -> pd.DataFrame:
    """
    Train PoissonGLM and GBM on game scores, ensemble by deviance-based weights,
    then calibrate using the **training-set** lambdas and compute win-probabilities
    from the **final** ensemble lambdas.
    """
    X_today = X_today[X.columns]

    # 1) Fit GLM pipelines
    glm_away = Pipeline([('scaler', StandardScaler()), ('model', PoissonRegressor(alpha=0.0, max_iter=2000))])
    glm_home = Pipeline([('scaler', StandardScaler()), ('model', PoissonRegressor(alpha=0.0, max_iter=2000))])
    glm_away.fit(X, y['away_score']); glm_home.fit(X, y['home_score'])

    # 2) Fit GBMs
    gbm_away = lgb.LGBMRegressor(objective='poisson', n_estimators=300, learning_rate=0.05)
    gbm_home = lgb.LGBMRegressor(objective='poisson', n_estimators=300, learning_rate=0.05)
    gbm_away.fit(X, y['away_score']); gbm_home.fit(X, y['home_score'])

    # 3) Cross-validate
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_glm_away = -cross_val_score(glm_away.named_steps['model'], X, y['away_score'], cv=kf, scoring='neg_mean_poisson_deviance')
    cv_gbm_away = -cross_val_score(gbm_away, X, y['away_score'], cv=kf, scoring='neg_mean_poisson_deviance')
    cv_glm_home = -cross_val_score(glm_home.named_steps['model'], X, y['home_score'], cv=kf, scoring='neg_mean_poisson_deviance')
    cv_gbm_home = -cross_val_score(gbm_home, X, y['home_score'], cv=kf, scoring='neg_mean_poisson_deviance')

    # 4) Compute weights and raw lambdas
    w_glm_away = cv_gbm_away.mean() / (cv_glm_away.mean() + cv_gbm_away.mean())
    w_gbm_away = cv_glm_away.mean() / (cv_glm_away.mean() + cv_gbm_away.mean())
    w_glm_home = cv_gbm_home.mean() / (cv_glm_home.mean() + cv_gbm_home.mean())
    w_gbm_home = cv_glm_home.mean() / (cv_glm_home.mean() + cv_gbm_home.mean())

    lam_away_glm   = glm_away.predict(X_today)
    lam_away_gbm   = gbm_away.predict(X_today)
    lam_home_glm   = glm_home.predict(X_today)
    lam_home_gbm   = gbm_home.predict(X_today)
    lam_away_glm_tr = glm_away.predict(X)
    lam_away_gbm_tr = gbm_away.predict(X)
    lam_home_glm_tr = glm_home.predict(X)
    lam_home_gbm_tr = gbm_home.predict(X)

    # 5) Ensemble
    lam_away_tr = w_glm_away * lam_away_glm_tr + w_gbm_away * lam_away_gbm_tr
    lam_home_tr = w_glm_home * lam_home_glm_tr + w_gbm_home * lam_home_gbm_tr
    lam_away   = w_glm_away * lam_away_glm   + w_gbm_away * lam_away_gbm
    lam_home   = w_glm_home * lam_home_glm   + w_gbm_home * lam_home_gbm

    # 6) Calibration via training lambdas
    mean_away = y['away_score'].mean(); mean_home = y['home_score'].mean()
    scale_away = mean_away / np.mean(lam_away_tr) if np.mean(lam_away_tr) > 0 else 1.0
    scale_home = mean_home / np.mean(lam_home_tr) if np.mean(lam_home_tr) > 0 else 1.0
    logger.info(f"Calibrating game λ: away ×{scale_away:.3f}, home ×{scale_home:.3f}")
    lam_away *= scale_away; lam_home *= scale_home

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # 1) build a small DataFrame of raw-poisson win-probs + true labels
    p_pois_tr = batch_poisson_win_probs(lam_away_tr, lam_home_tr)
    cal_df = pd.DataFrame({
        "p_pois": p_pois_tr,
        "away_lambda": lam_away_tr,
        "home_lambda": lam_home_tr,
        "away_won": (y["away_score"] > y["home_score"]).astype(int)
    })

    # 2) train/test split (you could even use all of it, but hold out 20% is safest)
    Xc, Xv, yc, yv = train_test_split(
            cal_df[["p_pois", "away_lambda", "home_lambda"]],
            cal_df["away_won"],
            test_size = 0.2,
            random_state = 42
       )

    # 3) fit your calibrator
    cal = LogisticRegression().fit(Xc, yc)

    # 7) Build output
    df = games_today.copy()
    df['away_score_pred'] = np.round(lam_away, 1)
    df['home_score_pred'] = np.round(lam_home, 1)

    p_raw = batch_poisson_win_probs(lam_away, lam_home)
    df_pred = pd.DataFrame({
        "p_pois": p_raw,
        "away_lambda": lam_away,
        "home_lambda": lam_home
    })
    df['away_win_prob'] = cal.predict_proba(df_pred)[:, 1]

    return df, lam_away_tr, lam_home_tr


# --- Prop Ensemble Fix ---
def train_and_predict_props_gbm(
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_today: pd.DataFrame
) -> pd.DataFrame:
    """
    Use a Poisson GLM baseline and a LightGBM Poisson regressor,
    ensemble 50/50, then calibrate against training-set lambdas for each prop.
    """
    X_today = X_today[X.columns]
    preds = pd.DataFrame(index=X_today.index)

    for target in ['HR', 'TB', 'K']:
        # Baseline GLM
        base_pipe = Pipeline([('scaler', StandardScaler()), ('model', PoissonRegressor(alpha=0.0, max_iter=2000))])
        base_pipe.fit(X, y[target])
        base_pred = base_pipe.predict(X_today)
        base_tr   = base_pipe.predict(X)

        # GBM
        gbm = lgb.LGBMRegressor(objective='poisson', n_estimators=200, learning_rate=0.05, force_row_wise=True)
        gbm.fit(X, y[target], callbacks=[lgb.log_evaluation(period=0)])
        gbm_pred = gbm.predict(X_today)
        gbm_tr   = gbm.predict(X)

        # Ensemble raw
        ens_tr = 0.5 * base_tr + 0.5 * gbm_tr
        ens   = 0.5 * base_pred + 0.5 * gbm_pred

        # Calibrate so mean λ matches training mean
        mean_train = y[target].mean()
        scale = mean_train / np.mean(ens_tr) if np.mean(ens_tr) > 0 else 1.0
        logger.info(f"Calibrating {target} λ: ×{scale:.3f}")
        ens *= scale

        preds[target] = ens
        if target == 'HR':
            # convert to probability of ≥1 HR
            preds['HR_prob'] = 1 - np.exp(-ens)
    return preds




def get_pitcher_stats(seasonal: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # ... (unchanged) ...
    df = seasonal[f"player_stats_totals_{SEASONS[-1]}"].copy()
    pid_col = next(c for c in df.columns if c.lower().endswith("player.id"))
    era_col = next((c for c in df.columns if "earnedrunaverage" in c.lower() or c.lower().endswith("era")), None)
    whip_col = next((c for c in df.columns if "whip" in c.lower()), None)
    k9_col = next((c for c in df.columns if "strikeoutsper9" in c.lower() or "k9" in c.lower()), None)
    ip_col = next((c for c in df.columns if "inningspitched" in c.lower()), None)
    pitcher_stats = pd.DataFrame(index=df[pid_col])
    if era_col: pitcher_stats["era"] = df[era_col]
    if whip_col: pitcher_stats["whip"] = df[whip_col]
    if k9_col: pitcher_stats["k9"] = df[k9_col]
    if ip_col: pitcher_stats["ip"] = df[ip_col]
    return pitcher_stats.fillna({"era": 4.50, "whip": 1.30, "k9": 7.5, "ip": 150.0})


def build_player_matrix(
        daily_logs: Dict[str, pd.DataFrame],
        daily_sched: Dict[str, Any],
        seasonal: Dict[str, pd.DataFrame],
        park_factor_map: Dict[str, float],
        pitcher_stats: pd.DataFrame,
        recent_n: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --- 1) Base batter-season features ---
    df_tot = seasonal["player_stats_totals_2025-regular"]
    id_col = next(c for c in df_tot.columns if c.lower().endswith("player.id"))

    # raw stat columns
    obp_col = next((c for c in df_tot.columns if "onbasepct" in c.lower()), None)
    slg_col = next((c for c in df_tot.columns if "sluggingpct" in c.lower()), None)
    ops_col = next((c for c in df_tot.columns if "onbaseplusslugging" in c.lower()), None)

    avg_col = next(c for c in df_tot.columns if "battingavg" in c.lower())
    hr_col = next(c for c in df_tot.columns if "homeruns" in c.lower())
    tb_col = next(c for c in df_tot.columns if "totalbases" in c.lower())
    k_col = next(c for c in df_tot.columns if "strikeouts" in c.lower())

    # now pull *all* four stats
    feats = df_tot.set_index(id_col)[[avg_col, hr_col, tb_col, k_col]].fillna(0)
    feats.columns = ["avg_bat", "hr_season", "tb_season", "k_season"]

    pa_col = "stats.batting.plateAppearances"
    pa = df_tot.set_index(id_col)[pa_col].replace(0, np.nan)  # avoid division by zero
    feats["hr_rate"] = feats["hr_season"] / pa
    feats["tb_rate"] = feats["tb_season"] / pa

    feats.drop(columns=["hr_season", "tb_season"], inplace=True)

    if obp_col: feats["obp"] = df_tot.set_index(id_col)[obp_col]
    if slg_col: feats["slg"] = df_tot.set_index(id_col)[slg_col]
    if ops_col: feats["ops"] = df_tot.set_index(id_col)[ops_col]

    # --- 2) Raw daily logs + dates ---
    daily_plogs = daily_logs["player_gamelogs"].copy()
    daily_plogs["game_id"] = daily_plogs["game.id"].astype(str)
    daily_plogs["game_date"] = pd.to_datetime(daily_plogs["game.startTime"])

    # --- 3) Pull in true venue_id from today's schedule ---
    venue_map = {
        str(g["schedule"]["id"]): g["schedule"]["venue"]["id"]
        for g in daily_sched["today_games_json"]
        if g.get("schedule", {}).get("id") and g.get("schedule", {}).get("venue")
    }
    daily_plogs["venue_id"] = daily_plogs["game_id"].map(venue_map)
    daily_plogs["park_factor"] = daily_plogs["venue_id"].map(park_factor_map).fillna(1.0)

    # --- 4) Pull in opposing starter from today's lineups ---
    starter_map = {}
    throws_map = {}
    for lu in daily_sched["today_lineups"]:
        # 1) pull out the 'game' object if it exists, otherwise try top‐level
        game_obj = lu.get("game", lu)
        gid = game_obj.get("id")
        if not gid:
            # nothing we can do here
            continue
        gid = str(gid)

        # 2) each teamLineups entry may be under 'teamLineups' or 'lineups'
        teams = lu.get("teamLineups") or lu.get("lineups") or []
        for team_block in teams:
            abbr = team_block.get("team", {}).get("abbreviation")
            if not abbr:
                continue
            lineup = team_block.get("actual") or team_block.get("expected") or {}
            for pos in lineup.get("lineupPositions", []):
                if pos.get("position") == "P" and pos.get("player"):
                    pid = pos["player"]["id"]
                    # 3) throwsSide might live under the player object, so .get safely
                    side = pos["player"].get("throwsSide") or pos["player"].get("throws") or None
                    starter_map[(gid, abbr)] = pid
                    throws_map[(gid, abbr)] = side
                    break

    daily_plogs["team_abbr"] = daily_plogs["team.abbreviation"]
    # … after df_tot, daily_plogs etc. …

    daily_plogs["opp_pitcher_id"] = daily_plogs.apply(
        lambda r: starter_map.get((r["game_id"], r["team.abbreviation"])), axis=1
    ).fillna(0).astype(int)

    # bring in the pitcher’s handedness (will be "L" or "R" or None)
    daily_plogs["throwsSide_opp"] = daily_plogs.apply(
        lambda r: throws_map.get((r["game_id"], r["team.abbreviation"])), axis=1
    )

    # now merge in the batter’s seasonal splits
    # merge seasonal splits onto daily_plogs
    splits = (
        seasonal[f"player_stats_totals_{SEASONS[-1]}"]
        .loc[:, ["player.id", "avg_vsLeft", "hr_vsLeft", "avg_vsRight", "hr_vsRight", "tb_vsLeft", "tb_vsRight"]]
        .rename(columns={"player.id": "player.id"})
    )
    daily_plogs = daily_plogs.merge(
        splits,
        left_on="player.id",
        right_on="player.id",
        how="left"
    )

    # compute vs‐handedness
    daily_plogs["avg_vs_opp"] = np.where(
        daily_plogs["throwsSide_opp"] == "L",
        daily_plogs["avg_vsLeft"],
        daily_plogs["avg_vsRight"]
    )
    daily_plogs["hr_vs_opp"] = np.where(
        daily_plogs["throwsSide_opp"] == "L",
        daily_plogs["hr_vsLeft"],
        daily_plogs["hr_vsRight"]
    )

    # --- 5) Merge in opp-pitcher stats (or default) ---
    if not daily_plogs["opp_pitcher_id"].eq(0).all():
        daily_plogs = daily_plogs.merge(
            pitcher_stats,
            left_on="opp_pitcher_id",
            right_index=True,
            how="left",
            suffixes=("", "_opp")
        ).fillna({
            "era_opp": 4.50,
            "whip_opp": 1.30,
            "k9_opp": 7.5,
            "ip_opp": 150.0
        })
    else:
        daily_plogs["era_opp"] = 4.50
        daily_plogs["whip_opp"] = 1.30
        daily_plogs["k9_opp"] = 7.5
        daily_plogs["ip_opp"] = 150.0

    # --- 6) Rolling and head-to-head batter metrics ---
    tb_col = next(c for c in daily_plogs.columns if "totalbases" in c.lower())
    hr2 = next(c for c in daily_plogs.columns if "homeruns" in c.lower())
    k2 = next(c for c in daily_plogs.columns if "strikeouts" in c.lower())

    # rolling EWMAs with multiple spans
    daily_plogs = add_ewma_features(
        daily_plogs,
        player_id_col="player.id",
        tb_col=tb_col, hr_col=hr2, k_col=k2,
        spans=[5, 10, 20]
    )

    tail_n = (
        daily_plogs
        .sort_values("game_date")
        .groupby("player.id")
        .tail(recent_n)
    )
    recent = (
        tail_n.groupby("player.id")[[tb_col, hr2, k2]]
        .mean()
        .rename(columns={tb_col: "tb_recent", hr2: "hr_recent", k2: "k_recent"})
    )

    feats = feats.join(recent, how="left").fillna(0)

    h2h = (
        daily_plogs
        .groupby(["player.id", "opp_pitcher_id"])[[tb_col, hr2, k2]]
        .agg({
            tb_col: "mean",
            hr2: lambda s: s.gt(0).mean(),
            k2: "count"
        })
        .rename(columns={
            tb_col: "h2h_TB_mean",
            hr2: "h2h_HR_rate",
            k2: "h2h_PA_count"
        })
    )

    last5 = (daily_plogs
             .sort_values("game_date")
             .groupby(["player.id", "opp_pitcher_id"])
             .tail(5))
    recent_h2h = (
        last5
        .groupby(["player.id", "opp_pitcher_id"])[[tb_col, hr2, k2]]
        .agg({
            tb_col: "mean",
            hr2: lambda s: s.gt(0).mean(),
            k2: "count"
        })
        .rename(columns={
            tb_col: "r5_TB_mean",
            hr2: "r5_HR_rate",
            k2: "r5_PA_count"
        })
    )
    # pick the right side based on the pitcher’s handedness
    daily_plogs["avg_vs_opp"] = np.where(
        daily_plogs["throwsSide_opp"] == "L",
        daily_plogs["avg_vsLeft"],
        daily_plogs["avg_vsRight"]
    )
    daily_plogs["hr_vs_opp"] = np.where(
        daily_plogs["throwsSide_opp"] == "L",
        daily_plogs["hr_vsLeft"],
        daily_plogs["hr_vsRight"]
    )
    daily_plogs["tb_vs_opp"] = np.where(
        daily_plogs["throwsSide_opp"] == "L",
        daily_plogs["tb_vsLeft"],
        daily_plogs["tb_vsRight"]
    )
    # --- 7) Assemble final X/Y ---
    merged = daily_plogs.set_index("player.id").join(feats, how="inner")
    merged = merged.join(h2h, on=["player.id", "opp_pitcher_id"])
    merged = merged.join(recent_h2h, on=["player.id", "opp_pitcher_id"])
    merged.fillna(0, inplace=True)

    y = merged[[tb_col, hr2, k2]].rename(
        columns={tb_col: "TB", hr2: "HR", k2: "K"}
    ).fillna(0)
    # ——— DEBUG: check that HR is a per‐game rate, not season total ———
    print("🔍 [build_player_matrix] y['HR'] summary:")
    print(y["HR"].describe())
    print("First 10 y['HR'] values:\n", y["HR"].head(10).values)

    feature_cols = [
        "avg_bat", "k_season", "hr_rate", "tb_rate",
        "tb_recent", "hr_recent", "k_recent",
        "park_factor",
        "avg_vs_opp", "hr_vs_opp", "tb_vs_opp",
        "era_opp", "whip_opp", "k9_opp", "ip_opp",
        "h2h_TB_mean", "h2h_HR_rate", "h2h_PA_count",
        "r5_TB_mean", "r5_HR_rate", "r5_PA_count", "tb_ewm_5", "hr_ewm_5", "k_ewm_5"
    ]
    for extra in ("obp", "slg", "ops"):
        if extra in feats.columns:
            feature_cols.append(extra)

    X = merged[feature_cols].fillna(0)

    # --- 8) Add weather / wind features (fixed) ---
    # --- 8) Add weather / wind features (fixed) ---
    sched = daily_sched["today_games"]

    # 1) detect whatever your weather columns are called
    temp_col = next((c for c in sched.columns if "weather.temp" in c.lower()), None)
    hum_col = next((c for c in sched.columns if "weather.humidity" in c.lower()), None)
    ws_col = next((c for c in sched.columns if "weather.wind.speed" in c.lower()
                   or "weather.wind_speed" in c.lower()), None)
    wd_col = next((c for c in sched.columns if "weather.wind.direction" in c.lower()), None)

    # 2) build up a rename map only for the ones you found
    col_map = {}
    if temp_col: col_map[temp_col] = "temp"
    if hum_col:  col_map[hum_col] = "humidity"
    if ws_col:   col_map[ws_col] = "wind_speed"
    if wd_col:   col_map[wd_col] = "wind_direction"

    weather_df = sched[list(col_map)].rename(columns=col_map)

    # fill missing simple weather cols
    for c in ("temp", "humidity", "wind_speed"):
        weather_df[c] = weather_df.get(c, 0)

    # ensure get returns a Series, not a bare string
    wind_dir = weather_df.get("wind_direction",
                              pd.Series("", index=weather_df.index))
    weather_df["wind_advantage"] = wind_dir.isin(["OUT_TO_RF", "OUT_TO_LF"]).astype(int)

    # 5) average into a single‐row Series
    avg_w = weather_df[["temp", "humidity", "wind_speed", "wind_advantage"]] \
        .mean() \
        .fillna(0)

    X = merged[feature_cols].fillna(0)
    X_today = X.copy()
    for df in (X, X_today):
        for feat, val in avg_w.items():
            df[feat] = val
    return X, y, X_today


def build_pitcher_matrix(
        daily_logs: Dict[str, pd.DataFrame],
        daily_sched: Dict[str, Any],
        seasonal: Dict[str, pd.DataFrame],
        park_factor_map: Dict[str, float],
        statcast_agg: pd.DataFrame,
        recent_n: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # --- 1) Season‐to‐date pitcher stats ---
    df_tot = seasonal["player_stats_totals_2025-regular"]
    season_id_col = next(c for c in df_tot.columns if c.lower().endswith("player.id"))

    # Identify your key columns
    k_season_col = next(c for c in df_tot.columns if "strikeouts" in c.lower() and "pitching" in c.lower())
    k9_candidates = [c for c in df_tot.columns if "strikeoutsper9" in c.lower() or "k9" in c.lower()]
    ip_candidates = [c for c in df_tot.columns if "inningspitched" in c.lower()]
    bb_col = next((c for c in df_tot.columns if "walks" in c.lower() and "pitching" in c.lower()), None)
    hr_col = next((c for c in df_tot.columns if "homeruns" in c.lower() and "pitching" in c.lower()), None)

    k9_col = k9_candidates[0] if k9_candidates else None
    ip_col = ip_candidates[0] if ip_candidates else None

    feats = df_tot.set_index(season_id_col)
    feats_dict = {
        "k_season": feats[k_season_col] if k_season_col in feats.columns else 0,
        "k_per9": feats[k9_col] if k9_col and k9_col in feats.columns else pd.Series(7.5, index=feats.index),
        "ip_season": feats[ip_col] if ip_col and ip_col in feats.columns else pd.Series(150.0, index=feats.index)
    }
    if bb_col and bb_col in feats.columns:
        feats_dict["bb_season"] = feats[bb_col]
    if hr_col and hr_col in feats.columns:
        feats_dict["hr_season"] = feats[hr_col]

    pitcher_feats = pd.DataFrame(feats_dict).fillna(0)

    # --- 2) Daily logs for pitchers only ---
    daily_plogs = daily_logs["player_gamelogs"].copy()
    cols = daily_plogs.columns.tolist()
    date_col = next(c for c in cols if "date" in c.lower() or "starttime" in c.lower())
    daily_plogs["game_date"] = pd.to_datetime(daily_plogs[date_col])

    venue_col = next((c for c in cols if "venue.id" in c.lower()), None)
    daily_plogs["park_factor"] = (
        daily_plogs[venue_col].map(park_factor_map).fillna(1.0)
        if venue_col else 1.0
    )

    id_cands = [c for c in cols if "player.id" in c.lower()]
    if not id_cands:
        id_cands = [c for c in cols if "player" in c.lower() and "id" in c.lower()]
    daily_id = id_cands[0]

    pos_col = next(c for c in cols if "position" in c.lower())
    pit_logs = daily_plogs[daily_plogs[pos_col] == "P"].copy()

    # --- 3) Rest-day features ---
    pit_logs = pit_logs.sort_values([daily_id, "game_date"])
    pit_logs["rest_days"] = pit_logs.groupby(daily_id)["game_date"].diff().dt.days.fillna(0)
    pit_logs["rest_days_sq"] = pit_logs["rest_days"] ** 2
    pit_logs["rest_bin"] = pd.cut(
        pit_logs["rest_days"], bins=[-1, 3, 6, 999], labels=[0, 1, 2]
    ).astype(int)

    # --- 4) Pull in TODAY’s weather for each game ---
    # --- robustly pull today's weather columns ---
    sched = daily_sched["today_games"]

    # 1) detect whatever your weather columns are called
    temp_col = next((c for c in sched.columns if "temp" in c.lower()), None)
    hum_col = next((c for c in sched.columns if "humidity" in c.lower()), None)
    ws_col = next((c for c in sched.columns if "wind.speed" in c.lower()
                   or "wind_speed" in c.lower()), None)
    wd_col = next((c for c in sched.columns if "wind.direction" in c.lower()), None)

    # 2) build up a rename map only for the ones you found
    col_map = {}
    if temp_col: col_map[temp_col] = "temp"
    if hum_col:  col_map[hum_col] = "humidity"
    if ws_col:   col_map[ws_col] = "wind_speed"
    if wd_col:   col_map[wd_col] = "wind_direction"

    # 3) pull those out & rename
    weather_df = sched[list(col_map)].rename(columns=col_map)
    # remember: sched is your daily_sched["today_games"] DataFrame
    weather_df["game_id"] = sched["schedule.id"].astype(str)

    # fill missing simple weather cols
    for c in ("temp", "humidity", "wind_speed"):
        weather_df[c] = weather_df.get(c, 0)

    # ensure get returns a Series, not a bare string
    wind_dir = weather_df.get("wind_direction",
                              pd.Series("", index=weather_df.index))
    weather_df["wind_advantage"] = wind_dir.isin(["OUT_TO_RF", "OUT_TO_LF"]).astype(int)

    # 5) average into a single‐row Series
    avg_w = weather_df[["temp", "humidity", "wind_speed", "wind_advantage"]] \
        .mean() \
        .fillna(0)

    # 6) broadcast those four features into both X and X_today

    weather_df["game_id"] = weather_df["game_id"].astype(str)
    pit_logs["game_id"] = pit_logs["game.id"].astype(str)

    # --- 5) Assemble merged with full feats + weather + rest ---
    merged = pit_logs.set_index(daily_id).join(pitcher_feats, how="inner")

    # pick up the target strikeout column
    k2_candidates = [c for c in merged.columns if "strikeouts" in c.lower() and "pitching" in c.lower()]
    k2 = k2_candidates[0] if k2_candidates else None

    if k2:
        y = merged[[k2]].rename(columns={k2: "K"}).fillna(0)
    else:
        merged["K"] = 0
        y = merged[["K"]]

    # --- 6) Define feature list & build X ---
    base_feats = [
        "k_season", "k_per9", "ip_season", "rest_days", "rest_days_sq", "rest_bin",
        "park_factor", "bb_season", "hr_season",
        "temp", "humidity", "wind_speed", "wind_advantage"
    ]
    stat_feats = [c for c in statcast_agg.columns if c in merged.columns]
    feature_cols = [c for c in base_feats if c in merged.columns] + stat_feats
    missing = set(base_feats) - set(merged.columns)
    print("today_games columns:", sched.columns.tolist())

    if missing:
        print("⚠️ build_pitcher_matrix: missing columns:", missing)

    X = merged[feature_cols].fillna(0)
    X_today = X.copy()
    for df in (X, X_today):
        for feat, val in avg_w.items():
            df[feat] = val
    return X, y, X_today


def build_game_matrix(
        daily: Dict[str, Any],
        seasonal: Dict[str, pd.DataFrame],
        team_logs: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    # --- 1) Historical games for training ---
    historical = [seasonal[f"games_{s}"] for s in SEASONS if not seasonal[f"games_{s}"].empty]
    df_train = pd.concat(historical, ignore_index=True)
    cols_train = df_train.columns

    away_id = next(c for c in cols_train if c.lower().endswith("awayteam.id"))
    home_id = next(c for c in cols_train if c.lower().endswith("hometeam.id"))
    away_sc = next(c for c in cols_train if "awayscore" in c.lower())
    home_sc = next(c for c in cols_train if "homescore" in c.lower())

    status_col = (next((c for c in cols_train if "playedstatus" in c.lower()), None)
                  or next((c for c in cols_train if "schedulestatus" in c.lower()), None))

    # parse dates & compute rest days
    df_train["game_date"] = pd.to_datetime(df_train["schedule.startTime"])
    df_train = df_train.sort_values("game_date")
    df_train["away_rest"] = df_train.groupby(away_id)["game_date"].diff().dt.days.fillna(0)
    df_train["home_rest"] = df_train.groupby(home_id)["game_date"].diff().dt.days.fillna(0)

    # --- NEW: defensive & momentum features ---

    # 1) season‐to‐date runs allowed per game
    team_tot = seasonal[f"team_stats_totals_{SEASONS[-1]}"]
    allowed_map = team_tot.set_index("team.id")["runs_allowed_per_game"]
    # build a mapping from team.id → team.abbreviation for bullpen lookups
    abbr_map = team_tot.set_index("team.id")["team.abbreviation"].to_dict()

    # 2) last‐5 games rolling average of runs scored
    df_train["away_runs"] = df_train[away_sc].fillna(0)
    df_train["home_runs"] = df_train[home_sc].fillna(0)
    df_train["away_last5_rpg"] = (
        df_train.groupby(away_id)["away_runs"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    df_train["home_last5_rpg"] = (
        df_train.groupby(home_id)["home_runs"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )

    # 3) bullpen ERA over last 30 days
    bp_df = bullpen_era_last30(team_logs)
    bp_map = bp_df.set_index("team.abbreviation")["bullpen_ERA_last30"]

    # --- 2) H2H win‐pct ---
    h2h = {}
    for _, row in df_train.iterrows():
        pair = (row[away_id], row[home_id])
        rec = h2h.setdefault(pair, {"games": 0, "away_wins": 0})
        rec["games"] += 1
        if row[away_sc] > row[home_sc]:
            rec["away_wins"] += 1
    df_train["h2h_pct"] = df_train.apply(
        lambda r: h2h[(r[away_id], r[home_id])]["away_wins"] / h2h[(r[away_id], r[home_id])]["games"],
        axis=1
    )

    # --- 3) Venue & weather for training ---
    venues = seasonal["venues_2025-regular"][[
        "venue.id", "venue.hasRetractableRoof", "venue.hasRoof", "venue.playingSurface"
    ]].rename(columns={"venue.id": "venue_id"})
    venue_col = next(c for c in cols_train if c.lower().endswith("venue.id"))
    weather_col = next((c for c in cols_train if "weather.type" in c.lower()), None)

    df_train = df_train.merge(venues, left_on=venue_col, right_on="venue_id", how="left")
    df_train["is_rain"] = df_train[weather_col].str.contains("rain", na=False).astype(int)
    df_train["is_clear"] = df_train[weather_col].str.contains("clear", na=False).astype(int)
    df_train["temp"] = df_train.get("schedule.weather.temp", pd.Series(70, index=df_train.index))
    df_train["wind"] = df_train.get("schedule.weather.wind.speed", pd.Series(0, index=df_train.index))
    df_train["humidity"] = df_train.get("schedule.weather.humidity", pd.Series(50, index=df_train.index))
    wind_dir_col = next((c for c in cols_train if "wind.direction" in c.lower()), None)
    df_train["wind_advantage"] = (
        df_train[wind_dir_col].isin(["OUT_TO_RF", "OUT_TO_LF"]).astype(int)
        if wind_dir_col else 0
    )

    # --- 4) Filter to completed games & park factor ---
    df_t = df_train[df_train[status_col].str.lower()
    .isin(["final", "completed", "postgame-reviewing"])].copy()
    # carry forward the game’s ID so we can join odds later
    df_t["schedule.id"] = df_t["schedule.id"].astype(str)
    df_t["event_id"] = df_t["schedule.id"]

    df_all = seasonal["games_2025-regular"].copy()
    df_all["total_runs"] = df_all[away_sc].fillna(0) + df_all[home_sc].fillna(0)
    venue_stats = (df_all.groupby(venue_col, as_index=False)
                   .agg(runs_per_game=("total_runs", "mean"),
                        games_played=("total_runs", "count")))
    league_avg = df_all["total_runs"].mean()
    venue_stats["park_factor"] = venue_stats["runs_per_game"] / league_avg
    park_factor_map = venue_stats.set_index(venue_col)["park_factor"].to_dict()
    df_t["park_factor"] = df_t[venue_col].map(park_factor_map).fillna(1.0)

    # --- 5) Standings & team‐offense maps ---
    standings = seasonal["standings_2025-regular"]
    wpct_map = standings.set_index(
        next(c for c in standings.columns if c.lower().endswith("team.id")))
    wpct_map = wpct_map[next(c for c in standings.columns if "winpct" in c.lower())]
    runs_map = seasonal[f"team_stats_totals_{SEASONS[-1]}"] \
        .set_index("team.id")["runs_per_game"]

    # --- 6) Build X_train & y_train with new + new features ---
    X_train = pd.DataFrame({
        "event_id": df_t["event_id"],
        "away_wp": df_t[away_id].map(wpct_map),
        "home_wp": df_t[home_id].map(wpct_map),
        "runs_pg_away": df_t[away_id].map(runs_map),
        "runs_pg_home": df_t[home_id].map(runs_map),
        "away_rest": df_t["away_rest"],
        "home_rest": df_t["home_rest"],
        "h2h_pct": df_t["h2h_pct"],
        "roof": df_t["venue.hasRoof"].fillna(0).astype(int),
        "rain": df_t["is_rain"],
        "clear": df_t["is_clear"],
        "temp": df_t["temp"],
        "wind": df_t["wind"],
        "humidity": df_t["humidity"],
        "wind_advantage": df_t["wind_advantage"],
        "park_factor": df_t["park_factor"],

        # ——— NEW FEATURES ———
        "away_allowed_rpg": df_t[away_id].map(allowed_map),
        "home_allowed_rpg": df_t[home_id].map(allowed_map),
        "away_last5_rpg": df_t["away_last5_rpg"],
        "home_last5_rpg": df_t["home_last5_rpg"],
        "away_bp_era": df_t[away_id].map(lambda tid: bp_map.get(abbr_map.get(tid), np.nan)),
        "home_bp_era": df_t[home_id].map(lambda tid: bp_map.get(abbr_map.get(tid), np.nan)),

    }).fillna(0)

    y_train = df_t[[away_sc, home_sc]].rename(columns={away_sc: "away_score", home_sc: "home_score"})

    # --- 7) Build today's feature matrix (unchanged) ---
    df_pred = daily["today_games"]
    raw = pd.json_normalize(daily["today_games_json"])

    status_pred = (next((c for c in df_pred.columns if "playedstatus" in c.lower()), None)
                   or next((c for c in df_pred.columns if "schedulestatus" in c.lower()), None))
    mask2 = raw[status_pred].str.lower().isin(["scheduled", "unplayed", "in-progress"])
    df_p = df_pred[mask2].copy()

    temp_col = next((c for c in df_p.columns if "weather.temp" in c.lower()), None)
    hum_col = next((c for c in df_p.columns if "weather.humidity" in c.lower()), None)
    ws_col = next((c for c in df_p.columns
                   if "weather.wind.speed" in c.lower()
                   or "wind_speed" in c.lower()), None)
    wd_col = next((c for c in df_p.columns if "weather.wind.direction" in c.lower()), None)
    if temp_col and hum_col and ws_col:

        df_p["temp"] = pd.to_numeric(df_p[temp_col], errors="coerce").fillna(70)
        df_p["humidity"] = pd.to_numeric(df_p[hum_col], errors="coerce").fillna(50)
        df_p["wind_speed"] = pd.to_numeric(df_p[ws_col], errors="coerce").fillna(0)
        df_p["wind_advantage"] = (
            df_p[wd_col].isin(["OUT_TO_RF", "OUT_TO_LF"]).astype(int)
            if wd_col else 0
        )
    # … right after your `if temp_col and hum_col and ws_col:` branch …
    else:
        # fallback to the raw schedule.weather column
        wcol = next((c for c in df_p.columns if "schedule.weather" in c.lower()), None)
        temps, hums, wspeeds, wdirs = [], [], [], []

        for w in df_p[wcol]:
            # if it's a dict, pull out each field, otherwise use defaults
            if isinstance(w, dict):
                temp = w.get("temp", 70)
                hum = w.get("humidity", 50)
                wind = w.get("wind") or {}
                sp = wind.get("speed", 0)

                # speed can itself be a dict
                if isinstance(sp, dict):
                    wspeed = (
                            sp.get("milesPerHour")
                            or sp.get("mph")
                            or sp.get("kph")
                            or sp.get("speed")
                            or 0
                    )
                else:
                    wspeed = sp or 0

                wdir = wind.get("direction")
            else:
                # no weather object at all
                temp, hum, wspeed, wdir = 70, 50, 0, None

            temps.append(temp)
            hums.append(hum)
            wspeeds.append(wspeed)
            wdirs.append(wdir)

        df_p["temp"] = temps
        df_p["humidity"] = hums
        df_p["wind_speed"] = wspeeds
        df_p["wind_advantage"] = [1 if d in ("OUT_TO_RF", "OUT_TO_LF") else 0
                                  for d in wdirs]

    df_p["schedule.id"] = df_p["schedule.id"].astype(str)
    df_p[away_id] = df_p[away_id].astype(int)
    df_p[home_id] = df_p[home_id].astype(int)

    past = df_train[["game_date", away_id, home_id]].copy()
    past["date_only"] = past["game_date"].dt.date
    restful = {
        team: (date.today() - g.max()).days
        for team, g in past.groupby(lambda r: (past.iloc[r][away_id], past.iloc[r][home_id]))["date_only"]
    }
    df_p["away_rest"] = df_p[away_id].map(restful).fillna(0)
    df_p["home_rest"] = df_p[home_id].map(restful).fillna(0)
    df_p["h2h_pct"] = df_p.apply(
        lambda r: h2h.get((r[away_id], r[home_id]), {"away_wins": 0, "games": 1})["away_wins"]
                  / h2h.get((r[away_id], r[home_id]), {"away_wins": 0, "games": 1})["games"],
        axis=1
    )

    df_p = df_p.merge(venues, left_on=venue_col, right_on="venue_id", how="left")
    # --- robust weather‐type detection for today’s games ---
    weather_type_cols = [c for c in df_p.columns if "weather.type" in c.lower()]
    if weather_type_cols:
        wc = weather_type_cols[0]  # e.g. "schedule.weather.type" or "schedule.venue.weather.type"
        df_p["is_rain"] = df_p[wc].str.contains("rain", na=False).astype(int)
        df_p["is_clear"] = df_p[wc].str.contains("clear", na=False).astype(int)
    else:
        # no weather.type column, default to no rain/clear
        df_p["is_rain"] = 0
        df_p["is_clear"] = 0

    df_p["temp"] = df_p.get("schedule.weather.temp", pd.Series(70, index=df_p.index))
    df_p["wind"] = df_p.get("schedule.weather.wind.speed", pd.Series(0, index=df_p.index))
    df_p["humidity"] = df_p.get("schedule.weather.humidity", pd.Series(50, index=df_p.index))
    df_p["wind_advantage"] = (
        df_p[wind_dir_col].isin(["OUT_TO_RF", "OUT_TO_LF"]).astype(int)
        if wind_dir_col in df_p.columns else 0
    )
    df_p["park_factor"] = df_p["venue_id"].map(park_factor_map).fillna(1.0)

    # today's bullpen & allowed
    df_p["away_allowed_rpg"] = df_p[away_id].map(allowed_map).fillna(0)
    df_p["home_allowed_rpg"] = df_p[home_id].map(allowed_map).fillna(0)
    df_p["away_last5_rpg"] = df_p[away_id].map(lambda tid: df_train.loc[
        df_train[away_id] == tid, "away_last5_rpg"].iat[-1] if tid in df_train[away_id].values else 0)
    df_p["home_last5_rpg"] = df_p[home_id].map(lambda tid: df_train.loc[
        df_train[home_id] == tid, "home_last5_rpg"].iat[-1] if tid in df_train[home_id].values else 0)
    df_p["away_bp_era"] = df_p[away_id].map(lambda tid: bp_map.get(
        seasonal["standings_2025-regular"]
        .set_index("team.id").loc[tid, "team.abbreviation"], 0))
    df_p["home_bp_era"] = df_p[home_id].map(lambda tid: bp_map.get(
        seasonal["standings_2025-regular"]
        .set_index("team.id").loc[tid, "team.abbreviation"], 0))
    df_p["schedule.id"] = df_p["schedule.id"].astype(str)
    df_p["event_id"] = df_p["schedule.id"]

    # assemble X_today
    X_today = pd.DataFrame({
        "event_id": df_p["event_id"],
        "away_wp": df_p[away_id].map(wpct_map),
        "home_wp": df_p[home_id].map(wpct_map),
        "runs_pg_away": df_p[away_id].map(runs_map),
        "runs_pg_home": df_p[home_id].map(runs_map),
        "away_rest": df_p["away_rest"],
        "home_rest": df_p["home_rest"],
        "h2h_pct": df_p["h2h_pct"],
        "roof": df_p["venue.hasRoof"].fillna(0).astype(int),
        "rain": df_p["is_rain"],
        "clear": df_p["is_clear"],
        "temp": df_p["temp"],
        "wind": df_p["wind_speed"],
        "humidity": df_p["humidity"],
        "wind_advantage": df_p["wind_advantage"],
        "park_factor": df_p["park_factor"],
        "away_allowed_rpg": df_p["away_allowed_rpg"],
        "home_allowed_rpg": df_p["home_allowed_rpg"],
        "away_last5_rpg": df_p["away_last5_rpg"],
        "home_last5_rpg": df_p["home_last5_rpg"],
        "away_bp_era": df_p["away_bp_era"],
        "home_bp_era": df_p["home_bp_era"],
    }).fillna(0)

    games_today = df_p[["away_team_abbr", "home_team_abbr"]].copy()
    games_today["event_id"] = df_p.get("event_id")
    games_today["game_id"] = df_p["schedule.id"].astype(str)

    return X_train, y_train, X_today, games_today, park_factor_map


def backtest_model(X: pd.DataFrame, y: pd.DataFrame, model, multioutput=False) -> np.ndarray:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    if multioutput:
        return np.stack(cross_val_score(
            model, X, y, cv=kf, scoring="neg_mean_squared_error",
            n_jobs=-1, verbose=0
        ).reshape(5, -1)) * -1
    else:
        return cross_val_score(
            model, X, y.values.ravel(), cv=kf,
            scoring="neg_mean_squared_error", n_jobs=-1
        ).reshape(-1, 1) * -1


import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import cross_val_score


def train_and_predict_props(X, y, X_today):
    """
    For each target in y, build a pipeline that imputes missing values,
    scales features, fits a Poisson regression, and predicts on X_today.
    Returns a dict of predictions.
    """
    props_preds = {}
    for target in y.columns:
        # 1) Median-impute missing data, then standardize, then Poisson GLM
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', PoissonRegressor(alpha=0.0, max_iter=2000))
        ])

        # 2) Evaluate with cross-validated deviance
        neg_dev = cross_val_score(
            pipe,
            X,
            y[target],
            cv=5,
            scoring='neg_mean_poisson_deviance',
            error_score='raise'
        )
        deviance = -neg_dev.mean()
        print(f"[{target}] CV poisson deviance: {deviance:.4f}")

        # 3) Refit on full data and predict
        pipe.fit(X, y[target])
        props_preds[target] = pipe.predict(X_today)
        if target == "HR":
            lam = props_preds["HR"]
            print("🔍 [props] HR λ’s preview:", lam[:10])
            print("🔍 [props] HR λ distribution mean,std:", lam.mean(), lam.std())

    return props_preds


def train_and_predict_games(
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_today: pd.DataFrame,
        games_today: pd.DataFrame
) -> pd.DataFrame:
    logger.info("Starting training & prediction for game scores")
    # build pipelines
    away_pipe = Pipeline([("scaler", StandardScaler()), ("model", PoissonRegressor(max_iter=2000, alpha=0.0))])
    home_pipe = Pipeline([("scaler", StandardScaler()), ("model", PoissonRegressor(max_iter=2000, alpha=0.0))])

    logger.debug("Fitting away score model")
    away_pipe.fit(X, y["away_score"])
    logger.debug("Fitting home score model")
    home_pipe.fit(X, y["home_score"])

    pois_away = away_pipe.named_steps["model"]
    pois_home = home_pipe.named_steps["model"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    logger.debug("Cross-validating away score model")
    cv_away = -cross_val_score(pois_away, X, y["away_score"], cv=kf,
                               scoring="neg_mean_poisson_deviance", n_jobs=-1)
    logger.debug("Cross-validating home score model")
    cv_home = -cross_val_score(pois_home, X, y["home_score"], cv=kf,
                               scoring="neg_mean_poisson_deviance", n_jobs=-1)
    logger.info(f"Away Poisson Deviance: {cv_away.mean():.4f} (std: {cv_away.std():.4f})")
    logger.info(f"Home Poisson Deviance: {cv_home.mean():.4f} (std: {cv_home.std():.4f})")

    logger.debug("Predicting today's lambdas")
    lam_away = away_pipe.predict(X_today)
    lam_home = home_pipe.predict(X_today)

    records = []
    for la, lh in zip(lam_away, lam_home):
        max_run = 20
        probs = np.array([sum(poisson.pmf(r1, la) * poisson.pmf(r - r1, lh) for r1 in range(r + 1))
                          for r in range(max_run + 1)])
        P_tot_ge_10 = 1 - probs[:10].sum()
        records.append({"P_total_ge_10": P_tot_ge_10})

    df_lams = pd.DataFrame(records, index=X_today.index)

    df = games_today.copy()
    df["away_score_pred"] = np.round(lam_away, 1)
    df["home_score_pred"] = np.round(lam_home, 1)
    df["away_win_prob"] = pois_away.predict(X_today) / (
            pois_away.predict(X_today) + pois_home.predict(X_today)
    )
    df["P_total_ge_10"] = df_lams["P_total_ge_10"]

    logger.info("Finished predictions for game scores")
    return df


def train_and_predict_pitchers(
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_today: pd.DataFrame
) -> pd.DataFrame:
    logger.info("Starting training & prediction for pitcher strikeouts")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }
    logger.debug("Grid search over parameters: %s", param_grid)
    grid = GridSearchCV(RandomForestRegressor(n_jobs=-1), param_grid,
                        cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X, y.values.ravel())
    best = grid.best_estimator_
    logger.info(f"Best RandomForest params: {grid.best_params_}")

    mse_scores = backtest_model(X, y, best)
    logger.info(f"Pitchers backtest MSE:\n{mse_scores}")

    importances = np.stack([est.feature_importances_ for est in best.estimators_])
    avg_imp = importances.mean(axis=0)
    for col, imp in zip(X.columns, avg_imp):
        logger.debug(f"Feature importance {col}: {imp:.3f}")

    preds = best.predict(X_today)
    logger.info("Finished predictions for pitcher strikeouts")
    return pd.DataFrame(preds, index=X_today.index, columns=["K_p"])


async def fetch_moneyline_odds(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    key = ODDS_API_KEY or os.getenv("ODDS_API_KEY")
    if not key:
        return []
    params = {
        "apiKey": key,
        "regions": ",".join(ODDS_REGIONS),
        "markets": "h2h",
        "oddsFormat": ODDS_FORMAT,
    }
    async with session.get(ODDS_API_URL, params=params) as resp:
        try:
            resp.raise_for_status()
            return await resp.json()
        except Exception:
            return []


def implied_prob(odds: float) -> float:
    return 1 / odds if odds else 0


def find_value_moneyline(game_preds: pd.DataFrame,
                         odds_data: List[Dict[str, Any]],
                         ev_thresh: float = 0) -> pd.DataFrame:
    # 1) pull out every book’s head-to-head lines into a flat table
    rows = []
    for ev in odds_data:
        eid = ev.get("id")
        for b in ev.get("bookmakers", []):
            for m in b.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                for o in m["outcomes"]:
                    rows.append({
                        "event_id": eid,
                        "team": o["name"],
                        "odds": float(o["price"]),
                        "imp": implied_prob(o["price"])
                    })
    odds_df = pd.DataFrame(rows)

    # 2) if there's nothing, bail out with the right columns
    if odds_df.empty:
        return pd.DataFrame(columns=[
            "event_id", "team", "odds", "imp",
            "away_win_prob", "ev"
        ])

    # 3) ensure the merge‐keys are the same dtype
    odds_df["team"] = odds_df["team"].astype(str)
    game_preds = game_preds.copy()
    game_preds["away_full"] = game_preds["away_full"].astype(str)
    game_preds["home_full"] = game_preds["home_full"].astype(str)

    # 4) merge on event_id + away_full vs team, compute EV, filter & pick one per game

    df = (
        game_preds
        .merge(
            odds_df,
            left_on=["event_id", "away_full"],
            right_on=["event_id", "team"],
            how="inner"
        )
        .assign(ev=lambda d: d["away_win_prob"] - d["imp"])
        # only keep lines with EV above threshold *and* projected win probability > 0.5
        .query("ev > @ev_thresh and away_win_prob > 0.5")
        .sort_values("ev", ascending=False)
        .drop_duplicates("event_id")
        .reset_index(drop=True)
    )

    return df


async def fetch_prop_odds(session: aiohttp.ClientSession, event_ids: List[str]) -> List[Dict[str, Any]]:
    key = ODDS_API_KEY or os.getenv("ODDS_API_KEY")
    if not key or not event_ids:
        logger.debug("No API key or no event IDs; skipping prop-odds fetch.")
        return []

    results = []
    for eid in event_ids:
        url = f"{ODDS_EVENTS_URL}/{eid}/odds"
        params = {
            "apiKey": key,
            "regions": "us",
            "markets": "batter_home_runs,batter_total_bases,pitcher_strikeouts",
            "oddsFormat": "decimal",
        }
        try:
            async with session.get(url, params=params) as resp:
                text = await resp.text()
                if resp.status != 200:
                    logger.warning(f"Prop-odds [{resp.status}] for event {eid}")
                    continue
                data = json.loads(text)
                if isinstance(data, list):
                    data = data[0]
                results.append(data)
        except Exception:
            logger.exception(f"Error fetching prop odds for event {eid}")

    return results


def normalize_name(name: str) -> str:
    return re.sub(r'[^a-z]', '', name.lower())


def find_value_props(player_preds: pd.DataFrame,
                     odds_data: List[Dict[str, Any]],
                     ev_thresh: float = 0) -> pd.DataFrame:
    rows = []
    unmatched_names = set()
    if "player_name" not in player_preds.columns and "player" in player_preds.columns:
        player_preds = player_preds.rename(columns={"player": "player_name"})
    all_player_names = player_preds["player_name"].unique().tolist()
    normalized_name_map = {normalize_name(p): p for p in all_player_names}

    for game in odds_data:
        eid = game.get("id")
        for b in game.get("bookmakers", []):
            for mkt in b.get("markets", []):
                mk = mkt.get("key")
                if mk not in PROP_MARKETS:
                    continue
                target_col = PROP_MARKETS[mk]

                if target_col not in player_preds.columns:
                    candidates = [c for c in player_preds.columns if target_col.lower() in c.lower()]
                    if candidates:
                        target_col = candidates[0]
                    else:
                        continue

                for o in mkt.get("outcomes", []):
                    name = o.get("participant") or o.get("description") or o.get("name")
                    label = " ".join([
                        str(o.get("type", "")),
                        str(o.get("name", "")),
                        str(o.get("description", ""))
                    ]).lower()
                    if "over" not in label:
                        continue

                    if target_col == "HR" and o.get("point") and float(o.get("point")) > 0.5:
                        continue

                    name_clean = normalize_name(name)
                    matched_name = None

                    if name_clean in normalized_name_map:
                        matched_name = normalized_name_map[name_clean]
                    else:
                        match = difflib.get_close_matches(name, all_player_names, n=1, cutoff=0.8)
                        if match:
                            matched_name = match[0]
                        else:
                            last = name.split()[-1].lower()
                            candidates = [p for p in all_player_names if last in p.lower().split()[-1]]
                            if candidates:
                                matched_name = candidates[0]

                    if not matched_name:
                        unmatched_names.add(name)
                        continue

                    if target_col not in player_preds.columns:
                        continue

                    mask = player_preds["player_name"] == matched_name
                    pred_series = player_preds.loc[mask, target_col]
                    if pred_series.empty:
                        continue
                    pred = float(pred_series.iloc[0])

                    # Calculate probability based on Poisson distribution
                    line_val = float(o.get("point", 0))
                    threshold = math.ceil(line_val)

                    if target_col == "HR" and "HR_prob" in player_preds.columns:
                        # Use pre-calculated HR probability
                        prob = float(player_preds.loc[mask, "HR_prob"].iloc[0])
                    else:
                        # Calculate using Poisson CDF
                        prob = 1 - poisson.cdf(threshold - 1, pred)

                    price = float(o.get("price", 0))
                    imp = implied_prob(price)
                    ev = prob - imp

                    prop_type = o.get("name", "").strip().lower()

                    rows.append({
                        "event_id": eid,
                        "player": matched_name,
                        "prop": target_col,
                        "line": line_val,
                        "type": prop_type,
                        "odds": price,
                        "imp": imp,
                        "model_pred": pred,
                        "model_prob": prob,
                        "ev": ev,
                    })

    df = pd.DataFrame(rows)

    if df.empty or "ev" not in df.columns:
        return pd.DataFrame([], columns=[
            "event_id", "player", "prop", "line", "type",
            "odds", "imp", "model_pred", "model_prob", "ev"
        ])

    df.sort_values("ev", ascending=False, inplace=True)

    hr_df = df[df["prop"] == "HR"].drop_duplicates(subset=["player", "prop"], keep="first")
    non_hr_df = df[df["prop"] != "HR"].drop_duplicates(subset=["player", "prop", "line"], keep="first")
    df = pd.concat([non_hr_df, hr_df], ignore_index=True)

    df["ev_tier"] = pd.cut(df["ev"],
                           bins=[-float("inf"), 0.02, 0.05, 0.1, float("inf")],
                           labels=["Marginal", "Fair", "Good", "Great"]
                           )

    df = df[~((df["prop"] == "HR") & (df["line"] > 0.5))]

    mask_hr = df["prop"] == "HR"
    mask_pred_over = df["model_pred"] > df["line"]

    df_filtered = df[(df["ev"] > ev_thresh) & (mask_pred_over | mask_hr)].copy()
    df_filtered["line"] = df_filtered["line"].astype(float)
    df_filtered.sort_values("ev", ascending=False, inplace=True)

    df_filtered = df_filtered.drop_duplicates(subset=["player", "prop"], keep="first")

    return df_filtered.sort_values("ev", ascending=False)


def bottom7_teams_last14(player_gamelogs: pd.DataFrame) -> pd.DataFrame:
    if "game_date" not in player_gamelogs.columns:
        player_gamelogs["game_date"] = (
            pd.to_datetime(player_gamelogs["game.startTime"])
            .dt.floor("D")
            .dt.tz_convert(None)
        )

    today = pd.to_datetime(datetime.now(timezone.utc).date()).floor("D")
    cutoff = (today - pd.Timedelta(days=14)).date()

    recent = player_gamelogs.loc[player_gamelogs["game_date"] > cutoff].copy()
    if recent.empty:
        return pd.DataFrame(columns=["team", "wRC+", "PA"])

    lw = {
        "single": 0.90,
        "double": 1.24,
        "triple": 1.56,
        "home_run": 1.95,
        "walk": 0.72,
        "hbp": 0.73,
        "sb": 0.20,
        "cs": -0.40,
    }

    recent["singles"] = (
            recent["stats.batting.hits"]
            - recent["stats.batting.secondBaseHits"]
            - recent["stats.batting.thirdBaseHits"]
            - recent["stats.batting.homeruns"]
    )

    recent["RC"] = (
            recent["singles"] * lw["single"]
            + recent["stats.batting.secondBaseHits"] * lw["double"]
            + recent["stats.batting.thirdBaseHits"] * lw["triple"]
            + recent["stats.batting.homeruns"] * lw["home_run"]
            + recent["stats.batting.batterWalks"] * lw["walk"]
            + recent["stats.batting.hitByPitch"] * lw["hbp"]
            + recent["stats.batting.stolenBases"] * lw["sb"]
            + recent["stats.batting.caughtBaseSteals"] * lw["cs"]
    )

    league_R_per_PA = 0.11
    park_factor = 1.00
    lg_wRC_per_PA = league_R_per_PA * 1.15

    recent["wRAA"] = (
                             recent["RC"]
                             - recent["stats.batting.plateAppearances"] * league_R_per_PA
                     ) / 1.15

    recent["wRC+"] = (
                             (
                                     (recent["wRAA"] / recent["stats.batting.plateAppearances"])
                                     + league_R_per_PA
                                     + (league_R_per_PA - park_factor * league_R_per_PA)
                             )
                             / lg_wRC_per_PA
                     ) * 100

    recent["wRC+_×PA"] = recent["wRC+"] * recent["stats.batting.plateAppearances"]
    team_agg = recent.groupby("team.abbreviation").agg(
        total_wRCp_x_PA=pd.NamedAgg(column="wRC+_×PA", aggfunc="sum"),
        total_PA=pd.NamedAgg(column="stats.batting.plateAppearances", aggfunc="sum"),
    )

    team_agg = team_agg.loc[team_agg["total_PA"] > 0].copy()
    team_agg["team_wRC+"] = team_agg["total_wRCp_x_PA"] / team_agg["total_PA"]

    bottom7 = team_agg.nsmallest(7, "team_wRC+").reset_index()
    bottom7 = bottom7.rename(columns={
        "team.abbreviation": "team",
        "team_wRC+": "wRC+",
        "total_PA": "PA"
    })[["team", "wRC+", "PA"]]

    return bottom7


def bullpen_era_last30(team_gamelogs: pd.DataFrame) -> pd.DataFrame:
    df = team_gamelogs.copy()

    if "game.startTime" not in df.columns:
        raise KeyError(f"Column 'game.startTime' not found. Available columns: {list(df.columns)}")

    df["game.startTime"] = pd.to_datetime(df["game.startTime"], utc=True)
    df["game_date"] = df["game.startTime"].dt.date

    cutoff_dt = (datetime.now(timezone.utc) - timedelta(days=30)).date()

    recent = df.loc[df["game_date"] > cutoff_dt].copy()

    er_col = "stats.pitching.earnedRunsAllowed"
    ip_col = "stats.pitching.inningsPitched"

    for c in (er_col, ip_col, "team.abbreviation"):
        if c not in recent.columns:
            raise KeyError(
                f"Column '{c}' not found in filtered DataFrame. "
                f"Available columns: {list(recent.columns)}"
            )

    agg = (
        recent
        .groupby("team.abbreviation", as_index=False)
        .agg({
            er_col: "sum",
            ip_col: "sum",
        })
        .rename(columns={
            er_col: "earnedRuns",
            ip_col: "inningsPitched"
        })
    )

    agg["bullpen_ERA_last30"] = agg.apply(
        lambda r: (r["earnedRuns"] / r["inningsPitched"] * 9)
        if r["inningsPitched"] > 0 else float("nan"),
        axis=1
    )

    return agg


def good_pitching_matchups(daily_sched: pd.DataFrame,
                           starter_stats: pd.DataFrame,
                           bullpen_stats: pd.DataFrame) -> pd.DataFrame:
    AWAY_COL = "away_team_abbr"
    HOME_COL = "home_team_abbr"

    ST_COL_TEAM = "team_abbreviation"
    ST_COL_FIP = "starter_FIP"

    BP_COL_TEAM = "team.abbreviation"
    BP_COL_ERA = "bullpen_ERA_last30"

    missing = [c for c in (AWAY_COL, HOME_COL) if c not in daily_sched.columns]
    if missing:
        raise KeyError(
            f"In daily_sched: missing column(s) {missing}. "
            f"Available columns are: {list(daily_sched.columns)}"
        )

    for col in (ST_COL_TEAM, ST_COL_FIP):
        if col not in starter_stats.columns:
            raise KeyError(
                f"In starter_stats: missing column '{col}'. Available: {list(starter_stats.columns)}"
            )

    for col in (BP_COL_TEAM, BP_COL_ERA):
        if col not in bullpen_stats.columns:
            raise KeyError(
                f"In bullpen_stats: missing column '{col}'. Available: {list(bullpen_stats.columns)}"
            )

    fip_lookup = starter_stats.set_index(ST_COL_TEAM)[ST_COL_FIP].to_dict()
    era_lookup = bullpen_stats.set_index(BP_COL_TEAM)[BP_COL_ERA].to_dict()

    results = []
    for _, game in daily_sched.iterrows():
        away_abbr = game[AWAY_COL]
        home_abbr = game[HOME_COL]

        away_fip = fip_lookup.get(away_abbr, float("nan"))
        home_fip = fip_lookup.get(home_abbr, float("nan"))
        away_era = era_lookup.get(away_abbr, float("nan"))
        home_era = era_lookup.get(home_abbr, float("nan"))

        away_score = away_fip + away_era
        home_score = home_fip + home_era

        results.append({
            "away_team": away_abbr,
            "home_team": home_abbr,
            "away_starter_FIP": away_fip,
            "home_starter_FIP": home_fip,
            "away_bullpen_ERA": away_era,
            "home_bullpen_ERA": home_era,
            "away_total_pitching_score": away_score,
            "home_total_pitching_score": home_score,
        })

    return pd.DataFrame(results)


def six_hitter_under_candidates(daily_logs, daily_sched, seasonal,
                                bad_team_abbrs, good_pitch_abbrs,
                                ba_threshold, park_df):
    candidates = []

    seasonal_stats = seasonal["player_stats_totals_2025-regular"]
    ba_col = next((c for c in seasonal_stats.columns if "battingAverage" in c.lower()), None)
    pid_col = next((c for c in seasonal_stats.columns if c.lower().endswith("player.id")), "player.id")

    games_df = daily_sched["today_games"]

    for lineup_json in daily_sched["today_lineups"]:
        game_info = lineup_json.get("game", {})
        game_id = game_info.get("id")
        if not game_id:
            continue

        mask = games_df["schedule.id"] == game_id
        if not mask.any():
            continue
        game_row = games_df.loc[mask].iloc[0]

        away_abbr = game_row["away_team_abbr"]
        home_abbr = game_row["home_team_abbr"]
        venue_id = game_row.get("schedule.venue.id")
        park_factor = park_df.get(venue_id, 1.0)

        if park_factor >= 1.0:
            continue

        # …
        weather = lineup_json.get("game", {}).get("weather") or {}
        # now weather is {} if original was None
        is_rain = weather.get("type") == "RAIN" or \
                  ((weather.get("precipitation") or {}).get("type") == "RAIN")
        wind = weather.get("wind") or {}
        wind_mph = 0.0
        if isinstance(wind, dict):
            # e.g. wind = {"speed": {"milesPerHour": 12}, ...}
            sp = wind.get("speed", {})
            if isinstance(sp, dict):
                wind_mph = sp.get("milesPerHour", 0.0)

        if "speed" in wind and isinstance(wind["speed"], dict):
            wind_mph = wind["speed"].get("milesPerHour", 0.0)
        rain_or_windy = is_rain or (wind_mph >= 10)

        for idx, team_block in enumerate(lineup_json.get("teamLineups", []) or []):
            # Skip if this entry is None
            if team_block is None:
                continue

            team_abbr = team_block.get("team", {}).get("abbreviation")
            if not team_abbr:
                continue

            if team_abbr not in bad_team_abbrs:
                continue

            opponent_abbr = home_abbr if (team_abbr == away_abbr) else away_abbr
            if opponent_abbr not in good_pitch_abbrs:
                continue

            actual_positions = (team_block.get("actual") or {}).get("lineupPositions", [])
            expected_positions = (team_block.get("expected") or {}).get("lineupPositions", [])

            lineup_positions = actual_positions if actual_positions else expected_positions

            sixth_slot = None
            for slot in lineup_positions:
                if slot.get("position") == "BO6" and slot.get("player") is not None:
                    sixth_slot = slot
                    break

            if sixth_slot is None:
                continue

            player = sixth_slot["player"]
            player_id = player.get("id")
            player_name = f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()

            ba_col = next((c for c in seasonal_stats.columns
                           if "battingavg" in c.lower()), None)
            if ba_col is None:
                continue
            ba_row = seasonal_stats.loc[seasonal_stats[pid_col] == player_id]
            if ba_row.empty:
                continue
            player_BA = float(ba_row.iloc[0][ba_col] or 0.0)

            if player_BA >= ba_threshold:
                continue

            candidates.append({
                "event_id": game_id,
                "venue_id": venue_id,
                "park_factor": park_factor,
                "team_abbr": team_abbr,
                "opponent_abbr": opponent_abbr,
                "player_id": player_id,
                "player_name": player_name,
                "player_BA": player_BA,
                "rain_or_windy": rain_or_windy,
                "wind_mph": wind_mph,
                "is_rain": is_rain
            })

    if not candidates:
        return pd.DataFrame(columns=[
            "event_id", "venue_id", "park_factor", "team_abbr", "opponent_abbr",
            "player_id", "player_name", "player_BA",
            "rain_or_windy", "wind_mph", "is_rain"
        ])

    df_cand = pd.DataFrame(candidates)
    df_cand.sort_values(by=["rain_or_windy", "player_BA"], ascending=[False, True], inplace=True)
    df_cand.reset_index(drop=True, inplace=True)
    return df_cand


import pandas as pd
import numpy as np
from pybaseball import statcast
import lightgbm as lgb

logging.getLogger("lightgbm").setLevel(logging.ERROR)

# --- 1. Fetch Statcast metrics ---
from datetime import date


async def fetch_statcast_metrics(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Statcast metrics between start_date and end_date,
    return a DataFrame indexed by player_name with whichever numeric columns are available.
    """
    loop = asyncio.get_event_loop()

    # 1) actually call statcast and capture its output
    try:
        sc: pd.DataFrame = await loop.run_in_executor(None, statcast, start_date, end_date)
    except Exception as e:
        logger.error("Statcast fetch failed: %s", e)
        return pd.DataFrame()  # bail out on error

    # 2) log shape & a few rows so we can see what came back
    logger.info("Statcast returned DataFrame with shape %s", sc.shape)
    # if not sc.empty:
    # logger.debug("Statcast columns: %s", list(sc.columns))
    # logger.debug("Statcast sample:\n%s", sc.head().to_dict(orient="list"))

    # 3) if we got nothing, warn and return empty
    if sc.empty:
        logger.warning(
            "No Statcast rows between %s and %s. "
            "Double‐check your date range or network connectivity.",
            start_date, end_date
        )
        return pd.DataFrame()

    # 4) ensure we have a player_name column
    sc['player_name'] = sc['player_name'].fillna(sc.get('batter', np.nan))

    # 5) pick only the numeric metrics you care about
    desired = [
        'exit_velocity',
        'launch_speed',
        'launch_angle',
        'estimated_woba_using_speedangle',
        'woba_value',
        'iso_value',
        'babip_value',
        'hit_distance_sc',
        'launch_speed_angle',
        'release_speed',
        'release_spin_rate',
        'pfx_x',
        'pfx_z',
        'release_extension',
        'n_thruorder_pitcher',
        'on_1b',
        'on_2b',
        'on_3b',
        'inning',
        'outs_when_up'
    ]
    available = [c for c in desired if c in sc.columns]

    statcast_df = (
        sc
        .loc[:, ['player_name'] + available]
        .dropna(subset=['player_name'])
    )

    # 6) log how many players we have
    statcast_df = statcast_df.select_dtypes(include='number')
    statcast_df = statcast_df.groupby(sc['player_name']).mean()
    logger.info("Statcast aggregated to %d players × %d metrics",
                statcast_df.shape[0], statcast_df.shape[1])

    return statcast_df


# --- 2. Add EWMA features in build_player_matrix ---
def add_ewma_features(
        daily_plogs: pd.DataFrame,
        player_id_col: str,
        tb_col: str, hr_col: str, k_col: str,
        spans: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Computes EWMA for TB, HR, K per player over multiple spans and adds to daily_plogs.
    Creates columns: tb_ewm_{span}, hr_ewm_{span}, k_ewm_{span} for each span in spans.
    """
    daily_plogs = daily_plogs.sort_values(['game_date'])
    for span in spans:
        daily_plogs[f'tb_ewm_{span}'] = (
            daily_plogs
            .groupby(player_id_col)[tb_col]
            .transform(lambda s: s.ewm(span=span, adjust=False).mean())
        )
        daily_plogs[f'hr_ewm_{span}'] = (
            daily_plogs
            .groupby(player_id_col)[hr_col]
            .transform(lambda s: s.ewm(span=span, adjust=False).mean())
        )
        daily_plogs[f'k_ewm_{span}'] = (
            daily_plogs
            .groupby(player_id_col)[k_col]
            .transform(lambda s: s.ewm(span=span, adjust=False).mean())
        )
    return daily_plogs


# --- 3. Merge Statcast & EWMA into X matrices ---
from typing import Dict, Tuple
import pandas as pd


def enhance_player_matrix(
        X: pd.DataFrame,
        X_today: pd.DataFrame,
        name_map: Dict[int, str],
        statcast_df: pd.DataFrame,
        ewma_span: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge Statcast metrics into training and today's feature sets.
    If statcast_df is empty (no rows), we skip entirely.
    """
    if statcast_df is None or statcast_df.empty:
        logger.info("No Statcast data available; skipping Statcast merge.")
        return X, X_today

    # --- normalize the index from "Last, First" → "First Last" ---
    normalized_idx = []
    for nm in statcast_df.index:
        if isinstance(nm, str) and "," in nm:
            last, first = nm.split(",", 1)
            normalized_idx.append(f"{first.strip()} {last.strip()}")
        else:
            normalized_idx.append(nm)
    statcast_df.index = normalized_idx

    # 1) Attach player_name for the join
    X_en = X.copy()
    X_en["player_name"] = X_en.index.map(name_map)
    X_today_en = X_today.copy()
    X_today_en["player_name"] = X_today_en.index.map(name_map)

    # 2) Merge on the normalized names
    X_en = X_en.merge(statcast_df, left_on="player_name", right_index=True, how="left")
    X_today_en = X_today_en.merge(statcast_df, left_on="player_name", right_index=True, how="left")

    # 3) Fill any remaining holes with the Statcast medians
    for col in statcast_df.columns:
        med = statcast_df[col].median()
        X_en[col].fillna(med, inplace=True)
        X_today_en[col].fillna(med, inplace=True)

    # 4) (Future) EWMA placeholder
    if ewma_span is not None:
        # assume you stored tb_ewm/hr_ewm/k_ewm in your X matrix already
        logger.debug(f"Applying EWMA span={ewma_span} in enhancer (no-op if not present)")
    # 5) Clean up
    X_en.drop(columns=["player_name"], inplace=True)
    X_today_en.drop(columns=["player_name"], inplace=True)

    return X_en, X_today_en


# --- 4. Stronger GBM ensemble for props ---
def train_and_predict_props_gbm(
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_today: pd.DataFrame
) -> pd.DataFrame:
    """
    Uses a LightGBM Poisson regressor and ensembles with baseline Poisson pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import PoissonRegressor
    from sklearn.model_selection import KFold, cross_val_score
    import numpy as np

    logger.info("Training stronger GBM ensemble for batter props")
    # baseline Poisson
    base_pipe = Pipeline([('scaler', StandardScaler()), ('model', PoissonRegressor(alpha=0.0, max_iter=2000))])
    # gbm regressor
    gbm = lgb.LGBMRegressor(objective='poisson', n_estimators=200, learning_rate=0.05, verbose=-1, verbosity=-1,
                            force_row_wise=True)

    X_today = X_today[X.columns]
    preds = pd.DataFrame(index=X_today.index)

    for target in ['HR', 'TB', 'K']:
        # fit baseline
        base_pipe.fit(X, y[target])
        base_pred = base_pipe.predict(X_today)
        # fit GBM
        gbm.fit(X, y[target], callbacks=[lgb.log_evaluation(period=0)])
        gbm_pred = gbm.predict(X_today)
        # ensemble: simple average
        ensemble = 0.5 * base_pred + 0.5 * gbm_pred
        preds[target] = ensemble
        if target == 'HR':
            preds['HR_prob'] = 1 - np.exp(-preds['HR'])
        if target == "HR":
            lam = preds['HR_prob']
            print("🔍 [props] HR λ’s preview:", lam[:10])
            print("🔍 [props] HR λ distribution mean,std:", lam.mean(), lam.std())


    logger.info("Completed GBM ensemble predictions")
    return preds


async def fetch_events(session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
    key = ODDS_API_KEY or os.getenv("ODDS_API_KEY")
    print(key)
    if not key:
        logger.warning("No ODDS_API_KEY; skipping fetch_events")
        return []
    logger.info("Fetching today's events from odds API")
    async with session.get(ODDS_EVENTS_URL, params={"apiKey": key}) as resp:
        resp.raise_for_status()
        events = await resp.json()
        logger.debug("Fetched %d events", len(events))
        return events


import pandas as pd

import smtplib
from email.message import EmailMessage


def generate_html_email_body(
        scores_table: str,
        details_table: str,
        value_table: str,
        top20_batters_table: str,
        hr_html: str,
        all_pitchers_table: str,
        under_run_table: str,
        hr_link: str,
        tb_link: str,
        k_link: str,
        ml_link: str
) -> str:
    return f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB Daily Report</title>
    <style>
      body {{margin:0;padding:0;background:#f7f7f7;font-family:Arial,sans-serif;color:#333}}
      .container{{max-width:700px;margin:auto;background:#fff;padding:20px}}
      h2{{text-align:center;color:#2a2a2a}}
      table{{width:100%;border-collapse:collapse;margin-bottom:30px}}
      th,td{{border:1px solid #ddd;padding:8px;vertical-align:top}}
      th{{background:#efefef}}
      .section{{margin-top:40px}}
      ul{{padding-left:16px;margin:4px 0}}
      li{{margin-bottom:2px}}
    </style>
  </head>
  <body>
    <div class="container">
      <h2>🏟 Today’s MLB Preview & Value Bets</h2>

      <div class="section">
        <h3>Game Predictions</h3>
        {scores_table}
      </div>

      <div class="section">
        <h3>Game Details</h3>
        {details_table}
      </div>

      <div class="section">
        <h3>💎 Value Moneyline Bets</h3>
        {value_table}
        <p>
          <a href="{ml_link}" target="_blank"  
             style="display:inline-block;margin-top:8px;
                    padding:6px 12px;background:#0073e6;color:#fff;
                    text-decoration:none;border-radius:4px;">
            ▶️ View & Share ML Slip
          </a>
        </p>
      </div>

      <div class="section">
        <h3>⭐ Top 20 Total-Bases Props</h3>
        {top20_batters_table}
        <p>
          <a href="{tb_link}" target="_blank"  
             style="display:inline-block;margin-top:8px;
                    padding:6px 12px;background:#0073e6;color:#fff;
                    text-decoration:none;border-radius:4px;">
            ▶️ View & Share TB Slip
          </a>
        </p>
      </div>

      <div class="section">
        <h3>💥 Top 10 Home-Run Candidates</h3>
        {hr_html}
        <p>
          <a href="{hr_link}" target="_blank"  
             style="display:inline-block;margin-top:8px;
                    padding:6px 12px;background:#0073e6;color:#fff;
                    text-decoration:none;border-radius:4px;">
            ▶️ View & Share Home-Run Slip
          </a>
        </p>
      </div>

      <div class="section">
        <h3>📊 Top Strikeout Props</h3>
        {all_pitchers_table}
        <p>
          <a href="{k_link}" target="_blank"  
             style="display:inline-block;margin-top:8px;
                    padding:6px 12px;background:#0073e6;color:#fff;
                    text-decoration:none;border-radius:4px;">
            ▶️ View & Share K-Props Slip
          </a>
        </p>
      </div>

      <div class="section">
        <h3>🚩 Under-Run Hitter Candidates</h3>
        {under_run_table}
      </div>
          </a>
        </p>
      </div>
    </div>
  </body>
</html>
"""


def optimize_props(trial, X: pd.DataFrame, y: pd.Series):
    model_type = trial.suggest_categorical("model", ["poisson", "gbm"])
    if model_type == "poisson":
        alpha = trial.suggest_loguniform("alpha", 1e-6, 1.0)
        model = PoissonRegressor(alpha=alpha, max_iter=2000)
    else:
        params = {
            "objective": "poisson",
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.2),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
        model = lgb.LGBMRegressor(**params)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # disable the per-iteration log
    if isinstance(model, lgb.LGBMRegressor):
        model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(period=0)])
    else:
        model.fit(X_tr, y_tr)

    preds = model.predict(X_val)
    return mean_poisson_deviance(y_val, preds)


def optimize_game_score(trial, X, y):
    # tune only the GBM component—keep PoissonGLM fixed or tune alpha similarly
    params = {
        "objective": "poisson",
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),

    }

    gbm = lgb.LGBMRegressor(**params)

    # split your data
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # disable any per‑iteration logging
    gbm.fit(
        X_tr, y_tr,
        callbacks=[lgb.log_evaluation(period=0)]
    )

    preds = gbm.predict(X_val)
    return mean_poisson_deviance(y_val, preds)


def optimize_pitcher(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        # <- no LightGBM flags here
    }
    rf = RandomForestRegressor(**params, n_jobs=-1)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.fit(X_tr, y_tr.values.ravel())
    preds = rf.predict(X_val)
    return mean_squared_error(y_val, preds)


def train_and_predict_pitchers_optuna(X, y, X_today, n_trials=1):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: optimize_pitcher(t, X, y), n_trials=n_trials, show_progress_bar=True)
    print("best RF params:", study.best_params)

    rf = RandomForestRegressor(**study.best_params, n_jobs=-1)
    rf.fit(X, y.values.ravel())
    preds = rf.predict(X_today)
    return pd.DataFrame(preds, index=X_today.index, columns=["K_p"])


def send_email(subject, plain_body, to_email, html_body=None):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = "jjohnson0636@gmail.com"
    msg['To'] = to_email
    # Fallback for clients that don't support HTML
    msg.set_content(plain_body or "Your email client does not support HTML.")
    if html_body:
        msg.add_alternative(html_body, subtype='html')
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login("jjohnson0636@gmail.com", "wodh yzme squo lmzc")
            smtp.send_message(msg)
        logging.info("Email sent successfully to %s", to_email)
    except smtplib.SMTPAuthenticationError as e:
        logging.error("SMTP Authentication Error: %s. Please use an application-specific password.", e)
    except Exception as e:
        logging.error("Failed to send email: %s", e)


import re
import logging
from typing import List, Optional
from playwright.async_api import async_playwright, TimeoutError


async def create_gambly_slip_link(
        bets: List[str] = None,
        *,
        raw_prompt: Optional[str] = None,
        debug: bool = False
) -> Optional[str]:
    """
    Build a parlay on Gambly from bets and return the share-link URL.
    If anything goes wrong (no card found, selector mismatch, timeout, etc.),
    the error is logged and None is returned so the script can continue.
    """
    try:
        # 1) decide what to type
        if raw_prompt:
            prompt = raw_prompt
        else:
            prompt = "parlay: " + "; ".join(bets[:10]) + " on FanDuel or DraftKings"

        providers = ["fan-duel.svg", "draft-kings.svg"]  # we'll click the <img> src

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=not debug)
            context = await browser.new_context(permissions=["clipboard-read", "clipboard-write"])
            page = await context.new_page()

            # Navigate and wait for the page to settle
            await page.goto("https://gambly.com", timeout=120_000)
            await page.wait_for_load_state("networkidle")
            logging.info("Gambly page loaded for prompt: %s", prompt)
            await page.wait_for_timeout(10)

            # Enter the parlay prompt
            await page.fill("textarea[placeholder*='bet']", prompt)
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(10)
            # Wait for the response cards to appear
            await page.wait_for_selector(".bet-response-card", timeout=120_000)
            cards = page.locator(".bet-response-card")
            await page.wait_for_timeout(10)
            # Find the card with the best odds
            best_idx = None
            best_odds = None
            best_site = None
            for i in range(await cards.count()):
                txt = await cards.nth(i).locator(".parlay-summary").inner_text()
                site = ("FanDuel" if "FanDuel" in txt else
                        "DraftKings" if "DraftKings" in txt else None)
                if not site:
                    continue
                m = re.search(r"([-+]\d+)", txt)
                if not m:
                    continue
                odds = int(m.group(1))
                await page.wait_for_timeout(10)
                if best_odds is None or odds > best_odds:
                    best_idx, best_odds, best_site = i, odds, site

            if best_idx is None:
                logging.error("No valid parlay card found for prompt: %s", prompt)
                await browser.close()
                return None

            logging.info("→ picking %s card (idx %d, odds %d)", best_site, best_idx, best_odds)

            # Expand the chosen card and click the sportsbook logo
            card = cards.nth(best_idx)
            await card.scroll_into_view_if_needed()
            await card.locator(".parlay-summary").click()

            for logo in providers:
                try:
                    await card.locator(f'img[src*="{logo}"]').click(timeout=5_000)
                    break
                except TimeoutError:
                    continue
            else:
                logging.error("Could not click any sportsbook logo for prompt: %s", prompt)
                await browser.close()
                return None

            logging.info("Expanded %s panel", best_site)

            # Click "Share Bet Slip" and copy the link
            await page.locator("#share-bet-slip").click()
            await page.locator("div.copy-link[title='Copy link']").click()
            link = await page.evaluate("navigator.clipboard.readText()")

            logging.info("🎉 Got share link: %s", link)

            if debug:
                # leave browser open for manual inspection
                import time;
                time.sleep(10)

            await browser.close()
            return link

    except Exception as e:
        logging.error("Gambly slip creation failed: %s", e, exc_info=True)
        return None


import pandas as pd
from datetime import date


def get_actual_tb_from_gamelog(player_logs: pd.DataFrame) -> pd.Series:
    """
    Given the full `player_logs` DataFrame, find the most recent game_date
    and return a Series mapping player_id -> actual total bases that day.
    """
    # ensure we have a date column
    date_col = next((c for c in player_logs.columns if "date" in c.lower() or "starttime" in c.lower()), None)
    if date_col is None:
        raise KeyError("No date/startTime column found in player_logs")

    df = player_logs.copy()
    df["game_date"] = pd.to_datetime(df[date_col]).dt.date
    latest = df["game_date"].max()
    today = df[df["game_date"] == latest]
    if today.empty:
        return pd.Series(dtype=float)

    id_col = next(c for c in today.columns if c.lower().endswith("player.id"))
    tb_col = next(c for c in today.columns if "totalbases" in c.lower())

    out = today.groupby(id_col)[tb_col].sum().astype(float)
    out.name = "actual_TB"
    return out


def get_actual_hr_from_gamelog(player_logs: pd.DataFrame) -> pd.Series:
    """
    Like TB above, but returns player_id -> number of home runs that day.
    """
    date_col = next((c for c in player_logs.columns if "date" in c.lower() or "starttime" in c.lower()), None)
    if date_col is None:
        raise KeyError("No date/startTime column found in player_logs")

    df = player_logs.copy()
    df["game_date"] = pd.to_datetime(df[date_col]).dt.date
    latest = df["game_date"].max()
    today = df[df["game_date"] == latest]
    if today.empty:
        return pd.Series(dtype=float)

    id_col = next(c for c in today.columns if c.lower().endswith("player.id"))
    hr_col = next(c for c in today.columns if "homeruns" in c.lower())

    out = today.groupby(id_col)[hr_col].sum().astype(float)
    out.name = "actual_HR"
    return out


def get_actual_k_from_gamelog(player_logs: pd.DataFrame) -> pd.Series:
    """
    Like TB above, but returns player_id -> number of strikeouts that day.
    """
    date_col = next((c for c in player_logs.columns if "date" in c.lower() or "starttime" in c.lower()), None)
    if date_col is None:
        raise KeyError("No date/startTime column found in player_logs")

    df = player_logs.copy()
    df["game_date"] = pd.to_datetime(df[date_col]).dt.date
    latest = df["game_date"].max()
    today = df[df["game_date"] == latest]
    if today.empty:
        return pd.Series(dtype=float)

    id_col = next(c for c in today.columns if c.lower().endswith("player.id"))
    k_col = next(c for c in today.columns if "strikeouts" in c.lower())

    out = today.groupby(id_col)[k_col].sum().astype(float)
    out.name = "actual_K"
    return out


def get_actual_game_winners_from_gamelog(games_df: pd.DataFrame) -> pd.Series:
    """
    Given a games DataFrame (e.g. your yesterday_games or team_logs),
    find the most recent game_date, and return a Series mapping game_id -> 1 if away won, 0 if home won.
    """
    # pick off a date column
    date_col = next((c for c in games_df.columns if "date" in c.lower() or "starttime" in c.lower()), None)
    if date_col is None:
        raise KeyError("No date/startTime column found in games_df")

    df = games_df.copy()
    df["game_date"] = pd.to_datetime(df[date_col]).dt.date
    latest = df["game_date"].max()
    today = df[df["game_date"] == latest]
    if today.empty:
        return pd.Series(dtype=float)

    game_id_col = next(c for c in today.columns if c.lower().endswith("id"))
    away_sc = next(c for c in today.columns if "awayscore" in c.lower())
    home_sc = next(c for c in today.columns if "homescore" in c.lower())

    # 1 if away_score > home_score, else 0
    winners = (today[away_sc] > today[home_sc]).astype(int)
    winners.index = today[game_id_col].astype(str)
    winners.name = "actual_away_win"
    return winners


import numpy as np
import pandas as pd

from tqdm import tqdm


def train_and_predict_props_optuna(
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_today: pd.DataFrame,
        n_trials: int = 1
) -> pd.DataFrame:
    """
    Tune each prop model with Optuna (one optimize() call per target),
    then refit the best model (poisson‐GLM or GBM) silently.
    """
    props_preds = {}

    for target in y.columns:
        study = optuna.create_study(direction="minimize")
        # run exactly n_trials trials, show Optuna's own progress bar
        study.optimize(
            lambda t: optimize_props(t, X, y[target]),
            n_trials=n_trials,
            show_progress_bar=True
        )

        best = study.best_params
        if best["model"] == "poisson":
            model = PoissonRegressor(
                alpha=best["alpha"],
                max_iter=2000
            )
        else:
            gbm_params = {
                "objective": "poisson",
                "force_row_wise": True,
                "verbose": -1,
                "verbosity": -1,
                "silent": True,
                **{k: best[k] for k in (
                    "n_estimators", "learning_rate",
                    "num_leaves", "min_child_samples"
                )}
            }
            model = lgb.LGBMRegressor(**gbm_params)

        model.fit(X, y[target])
        props_preds[target] = model.predict(X_today)

        if target == "HR":
            lam = props_preds["HR"]
            print("🔍 [props] HR λ’s preview:", lam[:10])
            print("🔍 [props] HR λ distribution mean,std:", lam.mean(), lam.std())


    return pd.DataFrame(props_preds, index=X_today.index)


def train_and_predict_games_optuna(
        X, y, X_today, games_today, n_trials=1
) -> pd.DataFrame:
    X_today_model = X_today[X.columns]

    # 1) Tune a GBM for away_score
    study_away = optuna.create_study(direction="minimize")
    study_away.optimize(lambda t: optimize_game_score(t, X, y["away_score"]),
                        n_trials=n_trials, show_progress_bar=True)
    gbm_away = lgb.LGBMRegressor(objective="poisson",
                                 force_row_wise=True,
                                 **study_away.best_params)
    gbm_away.fit(X, y["away_score"], callbacks=[lgb.log_evaluation(period=0)])

    # 2) Tune a GBM for home_score
    study_home = optuna.create_study(direction="minimize")
    study_home.optimize(lambda t: optimize_game_score(t, X, y["home_score"]),
                        n_trials=n_trials, show_progress_bar=True)
    gbm_home = lgb.LGBMRegressor(objective="poisson",
                                 force_row_wise=True,
                                 **study_home.best_params)
    gbm_home.fit(X, y["home_score"], callbacks=[lgb.log_evaluation(period=0)])

    # 3) Fit PoissonGLMs on full data
    glm_away = PoissonRegressor(alpha=0.0, max_iter=2000).fit(X, y["away_score"])
    glm_home = PoissonRegressor(alpha=0.0, max_iter=2000).fit(X, y["home_score"])

    # 4) Predict
    lam_away = 0.5 * gbm_away.predict(X_today_model) + 0.5 * glm_away.predict(X_today_model)
    lam_home = 0.5 * gbm_home.predict(X_today_model) + 0.5 * glm_home.predict(X_today_model)

    df = games_today.copy()
    df["away_score_pred"] = np.round(lam_away, 1)
    df["home_score_pred"] = np.round(lam_home, 1)

    df["away_win_prob"] = lam_away / (lam_away + lam_home)
    return df


def train_and_predict_pitchers_optuna(
        X: pd.DataFrame,
        y: pd.DataFrame,
        X_today: pd.DataFrame,
        n_trials: int = 1
) -> pd.DataFrame:
    """
    Tune a RandomForest for pitcher K’s with Optuna (one optimize() call),
    then fit the best forest and return predictions.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: optimize_pitcher(t, X, y),
        n_trials=n_trials,
        show_progress_bar=True
    )

    rf = RandomForestRegressor(
        **study.best_params,
        n_jobs=-1
    )
    rf.fit(X, y.values.ravel())
    preds = rf.predict(X_today)

    return pd.DataFrame(preds, index=X_today.index, columns=["K_p"])


from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List


def monte_carlo_game_totals(
        lam_away: np.ndarray,
        lam_home: np.ndarray,
        n_sims: int = 10
) -> pd.DataFrame:
    """
    Simulate full‐game run totals by drawing Poisson(lam_game) directly.
    """

    # draw an (n_games × n_sims) array in one go
    away_samps = np.random.poisson(lam=lam_away[:, None], size=(len(lam_away), n_sims))
    home_samps = np.random.poisson(lam=lam_home[:, None], size=(len(lam_home), n_sims))

    total = away_samps + home_samps

    df = pd.DataFrame({
        "mean_total": total.mean(axis=1),
        "std_total": total.std(axis=1)
    })

    # tail probabilities (e.g. P(total ≥ k))
    for k in range(5, 13):
        df[f"P_total_ge_{k}"] = (total >= k).mean(axis=1)

    return df


def monte_carlo_player_props(
        lambdas: pd.Series,
        thresholds: List[int],
        n_sims: int = 10
) -> pd.DataFrame:
    """
    Simulate each player’s prop distribution, with tqdm bars on both loops.
    """
    lam = lambdas.values
    n_players = len(lam)
    sims = np.empty((n_players, n_sims), dtype=int)

    # 1) simulation progress
    for i in tqdm(range(n_players), desc="Simulating props", unit="player"):
        sims[i] = np.random.poisson(lam[i], size=n_sims)

    out = pd.DataFrame(index=lambdas.index)

    # 2) threshold‐prob progress
    for t in tqdm(thresholds,
                  desc="Computing P(prop ≥ t)",
                  unit="threshold"):
        out[f"P_ge_{t}"] = (sims >= t).mean(axis=1)

    return out


import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from aiohttp import TCPConnector, BasicAuth
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import poisson

from scipy.stats import poisson
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd


def train_and_predict_win_probs(
        X_train: pd.DataFrame,
        y_train_scores: pd.DataFrame,
        X_today: pd.DataFrame,
        odds_df: pd.DataFrame,
        elo_train: pd.Series,
        elo_today: pd.Series
) -> pd.DataFrame:
    # 1) binary target
    y_win = (y_train_scores['away_score'] > y_train_scores['home_score']).astype(int)

    # 2) poisson‑based P(win)
    def pois_win(la, lh, N=20):
        p = 0.0
        for i in range(N + 1):
            for j in range(i):
                p += poisson.pmf(i, la) * poisson.pmf(j, lh)
        return p

    # pull off & drop any existing lambdas
    lam_away = X_train.pop('lam_away')
    lam_home = X_train.pop('lam_home')
    X_train['p_pois'] = [pois_win(a, h) for a, h in zip(lam_away, lam_home)]

    lam_away_t = X_today['lam_away']
    lam_home_t = X_today['lam_home']
    # now drop from features and build p_pois
    X_today = X_today.drop(columns=['lam_away', 'lam_home'])
    X_today['p_pois'] = [pois_win(a, h) for a, h in zip(lam_away_t, lam_home_t)]

    X_today['pred_away_runs'] = lam_away_t
    X_today['pred_home_runs'] = lam_home_t

    # 3) Elo
    X_train['p_elo'] = elo_train
    X_today['p_elo'] = elo_today

    # 4) odds — drop any existing imp columns first, then merge
    X_train = X_train.drop(columns=['away_ml_imp', 'home_ml_imp'], errors='ignore') \
        .merge(odds_df, on='event_id', how='left') \
        .fillna(0)
    X_today = X_today.drop(columns=['away_ml_imp', 'home_ml_imp'], errors='ignore') \
        .merge(odds_df, on='event_id', how='left') \
        .fillna(0)

    # 5) split off event_id
    ids = X_today['event_id'].copy()
    X_tr = X_train.drop(columns=['event_id'])
    X_td_full = X_today.copy()  # <-- keep full for later

    # 6) build & fit calibrated GBM pipeline
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('gbm', HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_depth=3,
            random_state=42
        ))
    ])
    clf = CalibratedClassifierCV(base_pipe, method='isotonic', cv=5)
    clf.fit(X_tr, y_win)
    X_td = X_td_full.drop(columns=['event_id', 'pred_away_runs', 'pred_home_runs'])

    # 7) predict & compute edge
    drop_cols = ['event_id', 'pred_away_runs', 'pred_home_runs', 'away_team_abbr', 'home_team_abbr']
    X_td = X_td.drop(columns=drop_cols, errors='ignore')
    p_away = clf.predict_proba(X_td)[:, 1]
    X_td_full['p_away'] = p_away
    X_td_full['edge'] = p_away - X_td_full['away_ml_imp']

    # 8) only return positive‑edge bets (including your nice columns)
    return (
        X_td_full
        .loc[X_td_full.edge > 0.02,
        ['event_id',
         'away_team_abbr', 'home_team_abbr',
         'pred_away_runs', 'pred_home_runs',
         'p_away', 'away_ml_imp', 'edge']]
        .reset_index(drop=True)
    )


def compute_elo_series(games_df, k=20, initial_elo=1500):
    """
    games_df must have columns: game_date (datetime), away_id, home_id, away_score, home_score
    Returns a DataFrame with columns [team_id, game_date, elo].
    """
    # initialize
    teams = set(games_df['away_id']).union(games_df['home_id'])
    elo = {t: initial_elo for t in teams}
    records = []

    for _, row in games_df.sort_values('game_date').iterrows():
        a, h = row['away_id'], row['home_id']
        Sa = 1 if row['away_score'] > row['home_score'] else 0
        Sh = 1 - Sa
        Ea = 1 / (1 + 10 ** ((elo[h] - elo[a]) / 400))
        Eh = 1 - Ea

        # update
        elo[a] += k * (Sa - Ea)
        elo[h] += k * (Sh - Eh)

        # record
        records.append({'team_id': a, 'game_date': row['game_date'], 'elo': elo[a]})
        records.append({'team_id': h, 'game_date': row['game_date'], 'elo': elo[h]})

    return pd.DataFrame(records)

def train_and_predict_props_tweedie(X, y, X_today):
    from sklearn.linear_model import TweedieRegressor
    TW_POWER = 1.5

    X_today = X_today[X.columns]
    preds = pd.DataFrame(index=X_today.index)

    # baseline Tweedie GLM
    base_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', TweedieRegressor(power=TW_POWER, alpha=0.0, max_iter=3000))
    ])

    # tweedie LightGBM
    gbm = lgb.LGBMRegressor(
        objective='tweedie',
        tweedie_variance_power=TW_POWER,
        n_estimators=200,
        learning_rate=0.05,
        force_row_wise=True
    )

    for target in ['HR','TB','K']:
        # 1) fit baseline
        base_pipe.fit(X, y[target])
        b_pred = base_pipe.predict(X_today)
        b_tr   = base_pipe.predict(X)

        # 2) fit GBM
        gbm.fit(X, y[target], callbacks=[lgb.log_evaluation(period=0)])
        g_pred = gbm.predict(X_today)
        g_tr   = gbm.predict(X)

        # 3) simple average
        ens_tr = 0.5*b_tr + 0.5*g_tr
        ens    = 0.5*b_pred + 0.5*g_pred

        # 4) calibrate so mean λ matches training mean
        scale = y[target].mean() / np.mean(ens_tr)
        ens  *= scale

        preds[target] = ens
        if target=='HR':
            preds['HR_prob'] = 1 - np.exp(-ens)

    return preds


import optuna
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import TweedieRegressor
import lightgbm as lgb

# Monte Carlo helper (from your code)
def monte_carlo_player_props(lambdas: pd.Series, thresholds: list, n_sims: int = 10):
    sims = np.random.poisson(lam=lambdas.values[:, None], size=(len(lambdas), n_sims))
    df = pd.DataFrame(index=lambdas.index)
    for t in thresholds:
        df[f"P_ge_{t}"] = (sims >= t).mean(axis=1)
    return df

# 1) Optuna objective to tune both Poisson/Tweedie and GBM for a single target
def prop_objective(trial, X, y):
    # 1a) Tweedie GLM hyperparams
    power = trial.suggest_float("power", 1.0, 2.0)
    alpha = trial.suggest_loguniform("alpha", 1e-6, 1e-1)
    glm = Pipeline([
        ("imputer", SimpleImputer("median")),
        ("scale", StandardScaler()),
        ("tweedie", TweedieRegressor(power=power, alpha=alpha, max_iter=3000))
    ])
    # 1b) Tweedie-GBM hyperparams
    n_est = trial.suggest_int("n_estimators", 50, 500)
    lr    = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
    leaves= trial.suggest_int("num_leaves", 16, 128)
    gbm = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=power,
        n_estimators=n_est,
        learning_rate=lr,
        num_leaves=leaves,
        force_row_wise=True
    )

    # 2) CV deviance for ensemble
    # here we simply average predictions from both
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_poisson_deviance
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    devs = []
    for train_idx, val_idx in kf.split(X):
        Xt, Xv = X.iloc[train_idx], X.iloc[val_idx]
        yt, yv = y.iloc[train_idx], y.iloc[val_idx]

        glm.fit(Xt, yt)
        gbm.fit(Xt, yt, callbacks=[lgb.log_evaluation(period=0)])

        pred = 0.5 * (glm.predict(Xv) + gbm.predict(Xv))
        devs.append(mean_poisson_deviance(yv, pred))

    return np.mean(devs)

# 3) Master function: tune, refit, predict, simulate
def train_predict_props_optuna_tweedie(
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_today: pd.DataFrame,
    n_trials: int = 1,
    mc_sims: int = 10
) -> pd.DataFrame:
    best_models = {}

    # For each prop (HR, TB, K) tune separately
    for prop in y.columns:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: prop_objective(t, X, y[prop]), n_trials=n_trials)

        # Rebuild best pipelines
        p = study.best_params["power"]
        glm = Pipeline([
            ("imputer", SimpleImputer("median")),
            ("scale", StandardScaler()),
            ("tweedie", TweedieRegressor(power=p,
                                         alpha=study.best_params["alpha"],
                                         max_iter=3000))
        ])
        gbm = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=p,
            n_estimators=study.best_params["n_estimators"],
            learning_rate=study.best_params["learning_rate"],
            num_leaves=study.best_params["num_leaves"],
            force_row_wise=True
        )
        glm.fit(X, y[prop])
        gbm.fit(X, y[prop], callbacks=[lgb.log_evaluation(period=0)])
        best_models[prop] = (glm, gbm)

    # Predict & ensemble on today’s data
    preds = pd.DataFrame(index=X_today.index)
    for prop, (glm, gbm) in best_models.items():
        p_glm = glm.predict(X_today)
        p_gbm = gbm.predict(X_today)
        lam   = 0.5 * (p_glm + p_gbm)
        # calibrate mean to training mean
        lam *= (y[prop].mean() / lam.mean())
        preds[prop] = lam
        if prop == "HR":
            preds["HR_prob"] = 1 - np.exp(-lam)

    # Monte Carlo sim for each threshold you care about
    # e.g. HR≥1, TB≥4, K≥5
    mc = monte_carlo_player_props(preds["HR"], thresholds=[1], n_sims=mc_sims)
    mc = mc.join(monte_carlo_player_props(preds["TB"], [4], n_sims=mc_sims))
    mc = mc.join(monte_carlo_player_props(preds["K"],  [5, 6, 7,8, 9], n_sims=mc_sims))

    return preds.join(mc)
from scipy.stats import poisson
def poisson_win_prob(la, lh, max_runs=20):
    # max_runs caps the sum—tweak as needed
    p = 0.0
    for i in range(max_runs+1):
        # P(X=i) * P(Y < i)
        p += poisson.pmf(i, la) * poisson.cdf(i-1, lh)
    return p

import numpy as np
from scipy.stats import poisson

def batch_poisson_win_probs(lam_a, lam_h, max_runs=20):
    # lam_a, lam_h are 1d arrays of the same length
    probs = []
    for la, lh in zip(lam_a, lam_h):
        # sum_{i=0..max_runs} P(X=i)*P(Y<i)
        terms = poisson.pmf(np.arange(max_runs+1), la) * poisson.cdf(np.arange(max_runs+1)-1, lh)
        probs.append(terms.sum())
    return np.array(probs)

async def main():
    import pickle
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # 1) Init DB and tables
    pool = await init_db_pool()
    await ensure_table(pool)
    await ensure_feedback_table(pool)

    # 2) Figure out dates
    today = datetime.today().date()
    yest = today - timedelta(days=1)
    yest_str = yest.strftime("%Y%m%d")
    today_str = today.strftime("%Y%m%d")

    # 3) Fetch & cache everything
    conn = TCPConnector(ssl=False)
    async with aiohttp.ClientSession(connector=conn) as session:
        # 1. Load / fetch & cache

        seasonal = await gather_seasonal(session, pool)
        player_stats_df = seasonal["player_stats_totals_2025-regular"]
        player_logs, team_logs = await gather_all_gamelogs(session, seasonal, pool)
        # right after gather_all_gamelogs(...)
        yest = datetime.today().date() - timedelta(days=1)
        player_logs['game_date'] = pd.to_datetime(player_logs['game.startTime']).dt.date

        daily_player_logs = player_logs[player_logs['game_date'] == yest]  # or == today
        # before your filtering, add:
        team_logs['game_date'] = pd.to_datetime(team_logs['game.startTime']).dt.date

        # then filter exactly like you did for the players:
        daily_team_logs = team_logs[team_logs['game_date'] == yest]

        daily_logs = {
            "player_gamelogs": daily_player_logs,
            "team_gamelogs": daily_team_logs,
        }

        daily_sched = await gather_daily(session, pool, yest, today)

        # 2. Prep data & feature matrices
        park_factor_map = get_park_factor_map(seasonal)
        pitcher_stats = get_pitcher_stats(seasonal)
        start_sc = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        end_sc = datetime.today().strftime("%Y-%m-%d")
        statcast_df = await fetch_statcast_metrics(start_sc, end_sc)

        # 1) Build your game‐level feature matrices
        Xg, yg, Xg_today, games_today, park_factor_map = build_game_matrix(
            daily_sched,
            seasonal,
            team_logs
        )
        # --- right after build_game_matrix(...) returns ---
        # df_train was the concatenated historical games inside that function;
        # so reassemble it here:
        # right after:

        # rebuild exactly the same df_t that build_game_matrix used internally:
        historical = [seasonal[f"games_{s}"] for s in SEASONS if not seasonal[f"games_{s}"].empty]
        df_train = pd.concat(historical, ignore_index=True)
        df_train['game_date'] = pd.to_datetime(df_train['schedule.startTime'])
        away_id_col = next(c for c in df_train.columns if c.lower().endswith('awayteam.id'))
        home_id_col = next(c for c in df_train.columns if c.lower().endswith('hometeam.id'))

        # filter to completed games exactly as in build_game_matrix…
        status_col = next((c for c in df_train.columns if 'playedstatus' in c.lower()), None)
        df_t = df_train[df_train[status_col].str.lower().isin(['final', 'completed', 'postgame-reviewing'])].copy()

        # now align the index and pull the IDs back in
        Xg['away_team_id'] = df_t[away_id_col]
        Xg['home_team_id'] = df_t[home_id_col]

        # and for today’s matrix you need the same from your df_p:
        df_p = pd.json_normalize(daily_sched["today_games_json"])
        df_p['event_id'] = df_p['schedule.id'].astype(str)

        df_p[away_id_col] = df_p[away_id_col].astype(int)
        df_p[home_id_col] = df_p[home_id_col].astype(int)
        # … after you’ve built df_p and Xg_today …
        # df_p: full raw schedule (16 rows)
        # Xg_today: filtered schedule used for prediction (15 rows)

        # 1) Ensure both sides use the same dtype
        df_p['event_id'] = df_p['event_id'].astype(str)
        Xg_today['event_id'] = Xg_today['event_id'].astype(str)

        # 2) Drop any duplicate event_ids in the schedule just in case
        df_p = df_p.drop_duplicates(subset=['event_id'])

        # 3) Re‐index df_p on event_id
        df_p_indexed = df_p.set_index('event_id')

        # 4) Grab the exact list of event_ids in the order of Xg_today
        event_ids = Xg_today['event_id'].tolist()

        # 5) Safely align; any missing event_id will show up as NaN (or raise if you prefer strict)
        try:
            df_p2 = df_p_indexed.loc[event_ids]
        except KeyError as e:
            logger.error("Could not align schedule rows to feature matrix: %s", e)
            raise

        # 6) Now you can pull in the team‐ID columns without error:
        Xg_today['away_team_id'] = df_p2[away_id_col].values
        Xg_today['home_team_id'] = df_p2[home_id_col].values

        away_sc = next(c for c in df_train.columns if 'awayscore' in c.lower())
        home_sc = next(c for c in df_train.columns if 'homescore' in c.lower())

        elo_df = compute_elo_series(
            df_train.rename(columns={
                away_id_col: 'away_id',
                home_id_col: 'home_id',
                away_sc: 'away_score',
                home_sc: 'home_score'
            })[['game_date', 'away_id', 'home_id', 'away_score', 'home_score']],
            k=20,
            initial_elo=1500
        )

        # pivot to get each team’s latest Elo
        last_elo = (elo_df.sort_values('game_date')
                    .groupby('team_id')['elo']
                    .last())

        # now map Elo into your feature matrices Xg and Xg_today
        # assuming Xg has columns away_team_id & home_team_id
        Xg['away_elo'] = Xg['away_team_id'].map(last_elo).fillna(1500)
        Xg['home_elo'] = Xg['home_team_id'].map(last_elo).fillna(1500)
        Xg_today['away_elo'] = Xg_today['away_team_id'].map(last_elo).fillna(1500)
        Xg_today['home_elo'] = Xg_today['home_team_id'].map(last_elo).fillna(1500)

        # grab both before dropping:
        train_event_ids = Xg["event_id"].copy()
        event_ids = Xg_today["event_id"].copy()

        # restore them correctly:
        Xg["event_id"] = train_event_ids.values
        Xg_today["event_id"] = event_ids.values

        print("y_train summary:", yg.describe())
        print("X_train first rows:\n", Xg.head())

        # Drop event_id only from Xg, but keep it in Xg_today for your mappings
        # just after
        Xg = Xg.drop(columns=["event_id"])
        # now your debug blocks and ensemble will all run without that dtype error

        # Baseline sanity check
        baseline = games_today.copy()
        baseline["away_score_pred"] = Xg_today["runs_pg_away"]
        baseline["home_score_pred"] = Xg_today["runs_pg_home"]

        # right before your Monte Carlo sim, etc.
        ens, lam_away_train, lam_home_train = train_and_predict_games_ensemble(
            Xg, yg, Xg_today, games_today
        )

        # put the train‐set lambdas back into Xg
        Xg['lam_away'] = lam_away_train
        Xg['lam_home'] = lam_home_train

        # put today’s lambdas (the predictions) into Xg_today
        Xg_today['lam_away'] = ens['away_score_pred'].values
        Xg_today['lam_home'] = ens['home_score_pred'].values

        # just copy it
        df = ens.copy()
        df["event_id"] = event_ids.values

        # if you really want to reconstruct it:
        pred_array = ens[["away_score_pred", "home_score_pred"]].to_numpy()
        away_preds = pred_array[:, 0]
        home_preds = pred_array[:, 1]
        df2 = pd.DataFrame({
            "event_id": event_ids.values,
            "away_team_abbr": games_today["away_team_abbr"].values,
            "home_team_abbr": games_today["home_team_abbr"].values,
            "away_score_pred": pred_array[:, 0],
            "home_score_pred": pred_array[:, 1],
        })

        print("\n=== Ensemble (GLM+GBM) predictions ===")
        print(ens[["away_team_abbr", "home_team_abbr", "away_score_pred", "home_score_pred"]])

        # 2c) Inspect the pure Poisson GLM
        from sklearn.linear_model import PoissonRegressor
        glm = PoissonRegressor(alpha=0.0, max_iter=2000)
        glm.fit(Xg, yg["away_score"])
        log_mean = np.log(yg.mean())
        intercept_adj = log_mean - glm.intercept_
        glm.intercept_ += intercept_adj
        print("\n=== Poisson GLM intercept & coeffs ===")
        print("Intercept:", glm.intercept_)

        for feat, coef in zip(Xg.columns, glm.coef_):
            print(f"  {feat:20s} → {coef:.4f}")

        # 2d) Minimal 2‑feature Poisson test
        X_simple = Xg[["runs_pg_away", "runs_pg_home"]]
        X_simple_today = Xg_today[["runs_pg_away", "runs_pg_home"]]
        glm_simple = PoissonRegressor(alpha=0.0, max_iter=2000)
        glm_simple.fit(X_simple, yg["away_score"])
        pred_simple = glm_simple.predict(X_simple_today)
        print("\n=== 2‑feature Poisson predictions ===")
        for (a, h), p in zip(games_today[["away_team_abbr", "home_team_abbr"]].values, pred_simple):
            print(f"  {a} @ {h} → {p:.1f}")

        # ────────────────────────────────────────────────────────────────────
        # 3) Now run your regular Optuna/GBM pipeline
        Xg_today_for_model = Xg_today.drop(columns=["event_id"])
        gpreds = train_and_predict_games_optuna(
            Xg, yg,
            Xg_today_for_model,  # now without event_id
            games_today,
            n_trials=1
        )
        print("y_train summary:", yg.describe())
        print("X_train first rows:\n", Xg.head())
        print("X_today first rows:\n", Xg_today.head())

        teams = seasonal["team_stats_totals_2025-regular"]
        abbr_map = (
            teams
            .set_index("team.abbreviation")
            .apply(lambda r: f"{r['team.city']} {r['team.name']}", axis=1)
            .to_dict()
        )

        # now games_today *does* have those columns for you to use immediately
        games_today["away_full"] = games_today["away_team_abbr"].map(abbr_map)
        games_today["home_full"] = games_today["home_team_abbr"].map(abbr_map)
        # --- grab pre‐game odds from the API ---
        money_odds = await fetch_moneyline_odds(session)  # h2h + totals markets
        # build lookup maps keyed by event_id
        away_imp_map, home_imp_map, total_line_map = {}, {}, {}

        for ev in money_odds:
            eid = str(ev["id"])
            for bm in ev.get("bookmakers", []):
                for m in bm.get("markets", []):
                    if m["key"] == "h2h":
                        for o in m["outcomes"]:
                            # match on the team’s full name
                            team = o["name"]
                            imp = implied_prob(o["price"])
                            # see whether this team is away or home in games_today
                            side = ("away" if team in games_today["away_full"].values
                                    else "home" if team in games_today["home_full"].values
                            else None)
                            if side == "away":
                                away_imp_map[eid] = imp
                            elif side == "home":
                                home_imp_map[eid] = imp

                    if m["key"] == "totals":
                        # assume the first outcome carries the line (e.g. "O 8.5")
                        point = m["outcomes"][0].get("point")
                        if point is not None:
                            total_line_map[eid] = float(point)

        # ─── MONTE CARLO GAME TOTALS ────────────────────────────────────────────────
        lam_away = ens["away_score_pred"].values
        lam_home = ens["home_score_pred"].values
        Xg['lam_away'] = lam_away_train
        Xg['lam_home'] = lam_home_train
        Xg_today['lam_away'] = lam_away
        Xg_today['lam_home'] = lam_home

        # 1) Simulate full‐game totals
        mc_games = monte_carlo_game_totals(
            lam_away=lam_away,
            lam_home=lam_home,
             n_sims=10  # bump this up if you want more precision
        )

        # 2) Merge the MC‐derived tail probabilities back onto your ensemble
        # 2) Merge the MC‐derived tail probabilities back onto your ensemble
        ens = pd.concat([ens, mc_games], axis=1)

        # 3) (Optional) pretty‐print a quick sanity check
        for idx, row in ens.iterrows():
            a, b = row["away_team_abbr"], row["home_team_abbr"]
            print(f"{a} @ {b} → P(total ≥ 10): {row['P_total_ge_10']:.3f}")

        # 1) restore the ID columns
        Xg["event_id"] = train_event_ids  # length == Xg
        Xg_today["event_id"] = event_ids  # length == Xg_today

        # 2) map in the odds
        for df in (Xg, Xg_today):
            df["away_ml_imp"] = df["event_id"].map(away_imp_map).fillna(0)
            df["home_ml_imp"] = df["event_id"].map(home_imp_map).fillna(0)
            df["total_line"] = df["event_id"].map(total_line_map).fillna(0)

        # 3) drop the temporary ID again
        Xg.drop(columns="event_id", inplace=True)
        Xg_today.drop(columns="event_id", inplace=True)

        Xp, yp, Xp_today = build_player_matrix(
            daily_logs,
            daily_sched,
            seasonal,
            park_factor_map,
            pitcher_stats
        )

        Xpk, ypk, Xpk_today = build_pitcher_matrix(
            daily_logs,
            daily_sched,
            seasonal,
            park_factor_map,
            statcast_df
        )

        # 3. Player models & predictions
        info = seasonal["player_stats_totals_2025-regular"][
            ["player.id", "player.firstName", "player.lastName"]].drop_duplicates()
        info["player_name"] = info["player.firstName"] + " " + info["player.lastName"]
        name_map = info.set_index("player.id")["player_name"].to_dict()

        Xp_en, Xp_today_en = enhance_player_matrix(
            Xp, Xp_today,
            name_map=name_map,
            statcast_df=statcast_df,
            ewma_span=10
        )
        # 4) Build features for props
        Xp, yp, Xp_today = build_player_matrix(
            daily_logs,
            daily_sched,
            seasonal,
            park_factor_map,
            pitcher_stats
        )
        # 5) Enhance with EWMA / Statcast
        Xp_en, Xp_today_en = enhance_player_matrix(
            Xp, Xp_today,
            name_map=name_map,
            statcast_df=statcast_df,
            ewma_span=10
        )

        # ─── DEBUG: inspect your Poisson λ’s on the *enhanced* matrix ─────────────
        lam_raw = train_and_predict_props_tweedie(Xp, yp, Xp_today)["HR"]
        lam_en = train_and_predict_props_tweedie(Xp_en, yp, Xp_today_en)["HR"]

        print("🔍 [debug] RAW HR λ mean,std:", lam_raw.mean(), lam_raw.std())
        print("🔍 [debug] ENHANCED HR λ mean,std:", lam_en.mean(), lam_en.std())


        # ─── DEBUG: inspect raw Poisson‐λ on the player matrix ────────────────────────
        # note: train_and_predict_props is your simpler, non‐Optuna Poisson + GLM pipeline
        player_X, player_y, X_today_player = Xp, yp, Xp_today

        props_preds = train_and_predict_props_tweedie(
            player_X,
            player_y,
            X_today_player
        )

        # — now inspect the HR props on the player‐level X_today_player —
        lam = props_preds["HR"]
        print("🔍 [props] HR λ’s preview:", lam[:10])
        print("🔍 [props] HR λ distribution mean,std:", lam.mean(), lam.std())

        # ───────────────────────────────────────────────────────────────────────────────

        feature_cols = Xp_en.columns.tolist()
        print("↪︎ runs per game (away):\n", Xg["runs_pg_away"].describe())
        print("↪︎ runs per game (home):\n", Xg["runs_pg_home"].describe())

        # 5) Load past feedback & train residuals
        resid_models = {}
        for prop in ["TB", "HR", "K"]:
            hist = await load_prop_feedback(pool, prop, days=30)
            if not hist.empty:
                Xh = hist.reindex(columns=feature_cols, fill_value=0)
                y_err = hist["error"]
                m = LinearRegression().fit(Xh, y_err)
                # pickle for reproducibility
                with open(f"resid_{prop}.pkl", "wb") as f:
                    pickle.dump(m, f)
                resid_models[prop] = m
            else:
                resid_models[prop] = None

        # 6) Raw GBM‐ensemble predictions
        ppreds = train_and_predict_props_optuna(Xp_en, yp, Xp_today_en, n_trials=1)
        # at the end of your prediction block:
        ppreds.to_pickle(f"./raw_props_{today_str}.pkl")

        ppreds["player_name"] = ppreds.index.map(name_map)

        # 7) Apply residual corrections
        for prop in ["TB", "HR", "K"]:
            m = resid_models[prop]
            if m is not None:
                corr = m.predict(Xp_today_en[feature_cols])
                ppreds[prop] += corr

        ppreds["HR_prob"] = 1 - np.exp(-ppreds["HR"])

        # 8) (Optional) Persist today’s raw preds for tomorrow’s feedback
        with open("raw_props_today.pkl", "wb") as f:
            pickle.dump(ppreds[["TB", "HR", "K"]], f)

        ppreds["HR_prob"] = 1 - np.exp(-ppreds["HR"])

        Xp_en = Xp_en.fillna(Xp_en.median())
        Xp_today_en = Xp_today_en.fillna(Xp_en.median())

        ppreds = train_and_predict_props_tweedie(
            Xp_en, yp, Xp_today_en
        )

        ppreds['player_name'] = ppreds.index.map(name_map)
        ppreds['HR_prob'] = 1 - np.exp(-ppreds['HR'])
        # ——— DEBUG: λ’s should be ≪ 1 for a single game ———
        lam = ppreds['HR']
        print("🔍 [main] final HR λ’s summary:\n", lam.describe())
        # check probability
        probs = 1 - np.exp(-lam)
        print("🔍 [main] P(HR≥1) preview:\n", probs.describe())

        # ——— MONTE‐CARLO PLAYER PROPS ———
        hr_mc = monte_carlo_player_props(
            lambdas=ppreds['HR'],
            thresholds=[1],  # whatever “≥ k” you care about
             n_sims= 10
        )
        # force its index to match ppreds
        hr_mc.index = ppreds.index

        tb_mc = monte_carlo_player_props(ppreds['TB'], thresholds=[2],  n_sims= 10)
        tb_mc.index = ppreds.index

        k_mc = monte_carlo_player_props(ppreds['K'], thresholds=[5, 6, 7, 8],  n_sims= 10)
        k_mc.index = ppreds.index

        ppreds = pd.concat([ppreds, hr_mc, tb_mc, k_mc], axis=1)

        # now ppreds has P_ge_2, P_ge_3 for HR/TB/K etc.

        # Pull out every non-pitcher ID from today’s lineups
        playing_ids = {
            pos['player']['id']
            for lu in daily_sched['today_lineups']
            for team in lu.get('teamLineups', [])
            for pos in (team.get('actual') or team.get('expected') or {}).get('lineupPositions', [])
            if pos.get('player') and pos['position'] != 'P'
        }
        ppreds_today = ppreds.loc[ppreds.index.isin(playing_ids)]
        raw = ppreds.copy()
        # 2) shrink your Xp_today_en (and Xp_today) to just those IDs:
        Xp_today_en = Xp_today_en.loc[Xp_today_en.index.isin(playing_ids)]
        Xp_today = Xp_today.loc[Xp_today.index.isin(playing_ids)]
        ppreds = train_and_predict_props_tweedie(Xp_en, yp, Xp_today_en)
        # ─── MONTE CARLO PLAYER PROPS ───────────────────────────────────────────────
        # thresholds = whatever “over X” you care about
        hr_mc = monte_carlo_player_props(ppreds['HR'], thresholds=[1],  n_sims=10)
        tb_mc = monte_carlo_player_props(ppreds['TB'], thresholds=[2],  n_sims=10)
        k_mc = monte_carlo_player_props(ppreds['K'], thresholds=[5, 6, 7],  n_sims=10)

        # merge ’em onto your raw ppreds
        ppreds = pd.concat([ppreds, hr_mc, tb_mc, k_mc], axis=1)

        print(ppreds[['HR', 'HR_prob']].sort_values('HR', ascending=False).head(10))
        print(ppreds[['TB']].sort_values('TB', ascending=False).head(10))

        ppitch = train_and_predict_pitchers_optuna(Xpk, ypk, Xpk_today, n_trials=1)
        ppitch['player_name'] = ppitch.index.map(name_map)

        print("Missing per column:")
        print(Xp_en.isna().sum().loc[lambda s: s > 0])

        # 4. Game models & predictions
        events = await fetch_events(session)
        evdf = pd.DataFrame(events)
        # if not evdf.empty:
        evdf["commence_time"] = pd.to_datetime(evdf["commence_time"])
        today_ev = evdf[evdf["commence_time"].dt.date == datetime.now(timezone.utc).date()]
        event_map = {}
        for ev in events:
            away = ev.get("away_team")
            home = ev.get("home_team")
            if away and home:
                event_map[(away, home)] = ev["id"]
                event_map[(home, away)] = ev["id"]

        teams = seasonal["team_stats_totals_2025-regular"]
        abbr_map = teams.set_index("team.abbreviation") \
            .apply(lambda r: f"{r['team.city']} {r['team.name']}", axis=1) \
            .to_dict()
        full_to_abbr = {full.upper(): abbr for abbr, full in abbr_map.items()}
        games_today = games_today.assign(
            away_full=games_today["away_team_abbr"].map(abbr_map),
            home_full=games_today["home_team_abbr"].map(abbr_map),
        )
        import numpy as np

        # after you build ppreds but before Monte Carlo:
        hr_lambdas = ppreds['HR']

        # 4) Now map to the odds event_id:
        def lookup_event_id(row):
            key = (row.away_full, row.home_full)
            return event_map.get(key)

        games_today["event_id"] = games_today.apply(lookup_event_id, axis=1)

        gpreds = train_and_predict_games_optuna(Xg, yg, Xg_today, games_today, n_trials=1)
        # ——— MONTE‐CARLO GAME TOTALS ———
        mc_games = monte_carlo_game_totals(
            lam_away=gpreds['away_score_pred'].values,
            lam_home=gpreds['home_score_pred'].values,
             n_sims= 10
        )
        # merge back onto your gpreds
        gpreds = pd.concat([gpreds, mc_games], axis=1)
        # now gpreds has columns like P_total_ge_10, P_total_ge_11, etc.

        Xp_today_en.index.name = "entity_id"

        # 1) Get actual TB from yesterday's logs
        # 1) compute actuals:
        actual_tb = get_actual_tb_from_gamelog(player_logs)
        actual_hr = get_actual_hr_from_gamelog(player_logs)
        actual_k = get_actual_k_from_gamelog(player_logs)
        actual_win = get_actual_game_winners_from_gamelog(daily_sched["yesterday_games"])
        # print("=== FEEDBACK TB HIST ===", hist.shape)
        # print("ACTUAL_TB:", actual_tb.head())
        # 2) build raw predictions DataFrame (you already have `raw` for props and `gpreds` for games)
        raw_games = gpreds  # contains away_win_prob, indexed by game_id
        raw_props = ppreds  # still used for your big “print everything” section
        ppreds_today = ppreds.loc[ppreds.index.isin(playing_ids)]

        # 3) prepare your feature‐matrix for today, but give it a proper entity_id column:
        feats = Xp_today_en.copy()
        feats["entity_id"] = feats.index.astype(int)
        feats = feats.set_index("entity_id")[feature_cols]

        # … inside your main(), after building fb …
        # … after you build ppreds_today …
        feedback_props = ppreds_today  # only today’s players
        # after you know which players you’re giving feedback for:
        feedback_ids = feedback_props.index.astype(int)  # ~20 IDs

        # right after you have Xp_today_en (features for today's players)
        feats_today = Xp_today_en.copy()
        feats_today["entity_id"] = feats_today.index.astype(int)
        feats_today = feats_today.set_index("entity_id")

        # drop any duplicate entity_id rows
        feats_today = feats_today[~feats_today.index.duplicated(keep='first')]

        # now reindex will work
        small_feats = feats_today.reindex(feedback_ids)

        # .loc will not error on duplicate labels

        # now you really will get only ~20 rows here:
        print("small_feats.shape:", small_feats.shape)

        # then, later in your feedback loop:
        for prop, actual in [("TB", actual_tb), ("HR", actual_hr), ("K", actual_k)]:
            df = (
                pd.DataFrame({
                    "entity_id": feedback_ids,
                    "predicted": feedback_props[prop].astype(float),
                    "actual": actual.reindex(feedback_ids).fillna(0).astype(float),
                })
                .set_index("entity_id")
            )
            # ← fix here:
            print("About to merge; left has", df.shape, "rows; right has", small_feats.shape, "rows")
            # de‑index and turn index into a column so we can do a true SQL‑style join
            # both df and small_feats have entity_id as their index,
            # so just inner‑join them on that index:
            # === 1) Make sure small_feats has its index named ===
            # … for each prop in ("TB","HR","K"):
            small_feats.index.name = 'entity_id'

            left = df.reset_index()  # ['entity_id','predicted','actual']
            right = small_feats.reset_index()  # ['entity_id', *features*]

            # get rid of duplicate rows now that entity_id is a real column
            left = left.drop_duplicates(subset=['entity_id'])
            right = right.drop_duplicates(subset=['entity_id'])

            fb = (
                left
                .merge(right,
                       on='entity_id',
                       how='inner',
                       validate='one_to_one')
                .loc[lambda d: ~d['predicted'].isna()]
            )
            await save_prop_feedback(pool, date.today(), prop, fb, feature_cols)

            # just in case: ensure one row per entity
            fb = fb.drop_duplicates(subset=['entity_id'])
            # print(f"\n--- Feedback preview for {prop} ---")
            # print(fb.loc[:, ["entity_id", "actual", "predicted"] + feature_cols[:3]].head())
            try:
                await save_prop_feedback(pool, date.today(), prop, fb, feature_cols)
            except Exception as e:
                logger.warning("Could not save feedback for %s: %s", prop, e)

        # first, keep only the first row for each game_id
        gpreds_unique = gpreds.drop_duplicates(subset=["game_id"], keep="first")
        pred = gpreds_unique.set_index("game_id")["away_win_prob"].rename("predicted")
        act = actual_win.rename("actual")  # <— rename to exactly 'actual'

        # drop duplicates, rename to entity_id
        pred = pred[~pred.index.duplicated()].rename('predicted')
        act = act[~act.index.duplicated()].rename('actual')

        fb = (
            pd.concat([pred, act], axis=1)
            .dropna(subset=['predicted'])
            .reset_index()
            .rename(columns={'index': 'entity_id'})
        )

        try:
            await save_prop_feedback(pool, date.today(), "game_winner", fb, [])
        except Exception as e:
            logger.warning("Could not save feedback for %s: %s", "game_winner", e)
        # then build your pred Series off the de-duplicated frame
        # after

        # 5. Odds & value bets
        # --- Moneyline ---
        money_odds = await fetch_moneyline_odds(session)

        # --- build a long-form version of gpreds with one row per team ---
        # for away side:
        away_df = (
            gpreds[['event_id', 'game_id', 'away_team_abbr', 'away_win_prob', 'away_score_pred', 'home_score_pred']]
            .rename(columns={
                'away_team_abbr': 'team_abbr',
                'away_score_pred': 'score_pred'
            })
            .assign(win_prob=gpreds['away_win_prob'])
        )

        # for home side: flip win‐prob & score‐pred
        home_df = (
            gpreds[['event_id', 'game_id', 'home_team_abbr', 'away_win_prob', 'away_score_pred', 'home_score_pred']]
            .rename(columns={
                'home_team_abbr': 'team_abbr',
                'home_score_pred': 'score_pred'
            })
            # home win prob is (1 - away_win_prob)
            .assign(win_prob=1 - gpreds['away_win_prob'])
        )

        gpreds_long = pd.concat([away_df, home_df], ignore_index=True)

        # now flatten your odds as before:
        # --- build a long-form version of gpreds with one row per team ---
        # … your away_df, home_df, gpreds_long …

        # now flatten the H2H markets exactly once:
        rows = []
        for ev in money_odds:
            eid = ev["id"]
            for b in ev["bookmakers"]:
                if b["key"] != "fanduel":
                    continue  # skip everybody but FanDuel
                for m in b["markets"]:
                    if m["key"] != "h2h":
                        continue
                    for o in m["outcomes"]:
                        # o["name"] is the full team name, e.g. "Tampa Bay Rays"
                        team_full = o["name"].upper()
                        team_abbr = full_to_abbr.get(team_full)
                        if not team_abbr:
                            # skip if we don't recognize this team name
                            continue
                        rows.append({
                            "event_id": eid,
                            "bookmaker": b["key"],
                            "team_abbr": team_abbr,  # now a 3-letter code
                            "odds": float(o["price"]),
                            "imp": 1 / float(o["price"])
                        })
        odds_df = pd.DataFrame(rows)

        gpreds_long["team_full"] = gpreds_long["team_abbr"].map(abbr_map)

        # 2) merge odds_df on event_id + full name
        ml_merge = (
            odds_df
            .merge(
                gpreds_long[["event_id", "team_abbr", "win_prob", "score_pred", "game_id"]],
                on=["event_id", "team_abbr"],  # ← both columns exist and are the same dtype
                how="inner"
            )
            .assign(ev=lambda df: df.win_prob * df.odds - (1 - df.win_prob))
        )
        odds_df = pd.DataFrame([
            {'event_id': gid, 'away_ml_imp': imp}
            for gid, imp in away_imp_map.items()
        ])
        odds_df['home_ml_imp'] = odds_df['event_id'].map(home_imp_map)

        p_elo_train = Xg['away_elo'] / (Xg['away_elo'] + Xg['home_elo'])
        p_elo_today = Xg_today['away_elo'] / (Xg_today['away_elo'] + Xg_today['home_elo'])

        # make a deep copy of your feature matrix for win‑prob modeling:
        X_win = Xg_today.copy()

        # re‑attach the event IDs
        X_win['event_id'] = event_ids.values

        # merge in the two abbreviation columns
        X_win = X_win.merge(
            games_today[['event_id', 'away_team_abbr', 'home_team_abbr']],
            on='event_id',
            how='left'
        )

        # attach your predicted runs
        X_win['pred_away_runs'] = ens['away_score_pred'].values
        X_win['pred_home_runs'] = ens['home_score_pred'].values

        # map in the moneyline implied probabilities
        X_win['away_ml_imp'] = X_win['event_id'].map(away_imp_map).fillna(0)
        X_win['home_ml_imp'] = X_win['event_id'].map(home_imp_map).fillna(0)

        value_bets = train_and_predict_win_probs(
            Xg.assign(event_id=train_event_ids),  # your training‐time X
            yg,  # your y_train_scores
            X_win,  # <— now contains the two abbv columns
            odds_df,
            p_elo_train,
            p_elo_today
        )
        for _, row in value_bets.iterrows():
            at = row['away_team_abbr']
            ht = row['home_team_abbr']
            ar = row['pred_away_runs']
            hr = row['pred_home_runs']
            wp = row['p_away']
            print(f"{at} @ {ht} → {ar:.1f} – {hr:.1f} (win% {wp:.3f})")

        # print("merged rows:", ml_merge.shape[0])
        # print(ml_merge[['team_abbr', 'win_prob', 'imp']].head())
        # 3) pick the columns you want
        # only keep positive EV bets
        vm = (
            ml_merge
            .query("ev > 1 and win_prob > 0.5")
            [[
                "event_id", "bookmaker", "team_abbr", "odds", "imp",
                "game_id", "win_prob", "score_pred", "ev"
            ]]
        )

        # print(vm)

        # --- Props ---
        prop_event_ids = [ev['id'] for ev in money_odds]
        prop_odds = await fetch_prop_odds(session, prop_event_ids)
        ppreds["player_name"] = ppreds.index.map(name_map)

        vp_batters = find_value_props(ppreds, prop_odds, ev_thresh=0)
        vp_pitchers = find_value_props(ppitch, prop_odds, ev_thresh=0)
        # let’s also pull in the offered line and your EV calc:
        # assuming you have `prop_odds` from fetch_prop_odds:
        # --- after fetching prop odds and building ppreds ---
        hr_props = find_value_props(ppreds, prop_odds, ev_thresh=0)


        # narrow to HR market *and* only highly probable guys
        hr_props = hr_props[
            (hr_props.prop == "HR") &
            (hr_props.ev > 0.01) &
            (hr_props.model_prob > 0.08)
            ]

        # choose your top N
        hr_candidates = (
            hr_props
            .drop_duplicates("player")
            .sort_values(["ev", "model_prob"], ascending=False)
            .head(12)
            .reset_index(drop=True)
        )

        # print(hr_candidates[['player', 'line', 'model_prob', 'ev']])



        # print("\n=== All HR Props with Lines & EV ===")
        # print(hr_props.to_string(index=False))
        inv_name_map = {v: k for k, v in name_map.items()}
        hr_candidates['player_id'] = hr_candidates['player'].map(inv_name_map)


        # 2a) Convert pandas → HTML (with simple styling classes)
        value_html = vm.to_html(index=False, classes="bets-table", border=0, justify="center")
        batters_html = vp_batters.to_html(index=False, classes="bets-table", border=0, justify="center")
        pitchers_html = vp_pitchers.to_html(index=False, classes="bets-table", border=0, justify="center")

        # build a map (event_id, player_name) → team_abbr
        pl_map = {}
        for lu in daily_sched["today_lineups"]:
            game_info = lu.get("game", {})
            gid = game_info.get("id")
            if not gid:
                # sometimes you get back an empty or unexpected JSON—just skip it
                logger.warning("Skipping lineup with no game id: %r", lu)
                continue
            eid = str(gid)
            for team_block in lu.get("teamLineups", []):
                abbr = team_block["team"]["abbreviation"]
                lineup = team_block.get("actual") or team_block.get("expected") or {}
                for pos in lineup.get("lineupPositions", []):
                    p = pos.get("player")
                    if not p: continue
                    name = f"{p['firstName']} {p['lastName']}"
                    pl_map[(eid, name)] = abbr

        vp_pitchers["team_abbr"] = vp_pitchers.apply(
            lambda r: pl_map.get((r["event_id"], r["player"])),
            axis=1
        )

        # Map event -> team abbreviations
        props_meta = games_today[['event_id', 'away_team_abbr', 'home_team_abbr']]
        vp_batters = vp_batters.merge(props_meta, on='event_id', how='left')
        vp_pitchers = vp_pitchers.merge(props_meta, on='event_id', how='left')

        # 6. Additional analytics
        bottom7_df = bottom7_teams_last14(player_logs)
        bullpen_stats = bullpen_era_last30(team_logs)
        # ←— this is the corrected block:
        starter_df: pd.DataFrame = player_stats_df.loc[
                                   :, ["player.currentTeam.abbreviation", "stats.pitching.earnedRunAvg"]
                                   ].rename(
            columns={
                "player.currentTeam.abbreviation": "team_abbreviation",
                "stats.pitching.earnedRunAvg": "starter_FIP",
            }
        )
        starter_stats = starter_df.groupby("team_abbreviation", as_index=False).mean()
        good_pitchers = good_pitching_matchups(daily_sched["today_games"], starter_stats, bullpen_stats)

        bad_team_abbrs = bottom7_df["team"].tolist()
        good_pitch_abbrs = good_pitchers["away_team"].tolist() + good_pitchers["home_team"].tolist()
        park_df_series = pd.Series(park_factor_map)
        candidates = six_hitter_under_candidates(
            daily_logs, daily_sched, seasonal,
            bad_team_abbrs, good_pitch_abbrs, 0.250, park_df_series
        )

        # 7. Output
        # after daily_sched = await gather_daily(...)
        lineup_map = {
            str(lu.get("game", {}).get("id") or lu.get("schedule", {}).get("id")): lu
            for lu in daily_sched["today_lineups"]
            if (lu.get("game") or lu.get("schedule") or {}).get("id") is not None
        }



        # --- Starting pitchers ---
        # print("  Starters:")
        for team_block in lu["teamLineups"]:
            abbr = team_block["team"]["abbreviation"]
            lineup = team_block.get("actual") or team_block.get("expected") or {}
            positions = lineup.get("lineupPositions", [])
            sp = next((p for p in positions if p["position"] == "P" and p.get("player")), None)
            if not sp:
                continue

            ply = sp["player"]
            pid = ply["id"]
            name = f"{ply['firstName']} {ply['lastName']}"

            # pull out all matching rows (might be 0, 1, or >1)
            if pid in ppitch.index:
                vals = ppitch.loc[pid, "K_p"]
                # if you get a Series (duplicates), take the first element
                k_val = float(vals.iloc[0]) if isinstance(vals, pd.Series) else float(vals)
            else:
                k_val = np.nan

            if np.isnan(k_val):
                print(f"    {abbr} SP: {name} → K_pred N/A")
            else:
                print(f"    {abbr} SP: {name} → K_pred {k_val:.1f}")

        # print()

        # --- Batting lineups ---
        # print("  Batting lineups:")

        for team_block in lu["teamLineups"]:
            abbr = team_block["team"]["abbreviation"]
            lineup = team_block.get("actual") or team_block.get("expected") or {}
            positions = lineup.get("lineupPositions", [])

            # ---- DEDUPE snippet starts here ----
            seen = set()
            unique_positions = []
            for pos in positions:
                p = pos.get("player")
                if not p: continue
                pid = p["id"]
                if pid not in seen:
                    seen.add(pid)
                    unique_positions.append(pos)
            positions = unique_positions

            # ---- DEDUPE snippet ends here ----

            batters = [p for p in positions if p["position"] != "P"]
            print(f"    {abbr}:")
            for slot in batters:

                ply = slot["player"]
                pid = ply["id"]
                name = f"{ply['firstName']} {ply['lastName']}"
                # grab the first element (or convert a scalar) so we get a bona fide float
                if pid in ppreds.index:
                    # pull the raw value (might be a Series if duplicates slipped through)
                    raw_tb = ppreds.loc[pid, 'TB']
                    raw_hr = ppreds.loc[pid, 'HR']
                    raw_k = ppreds.loc[pid, 'K']

                    # unwrap to a scalar if it’s a Series
                    tb_scalar = raw_tb.iloc[0] if isinstance(raw_tb, pd.Series) else raw_tb
                    hr_scalar = raw_hr.iloc[0] if isinstance(raw_hr, pd.Series) else raw_hr
                    k_scalar = raw_k.iloc[0] if isinstance(raw_k, pd.Series) else raw_k

                    # now safely format
                    tb_val = f"{tb_scalar:.1f}"
                    hr_val = f"{hr_scalar:.1f}"
                    k_val = f"{k_scalar:.1f}"
                else:
                    tb_val = hr_val = k_val = "N/A"

                print(f"      - {name}: TB {tb_val}, HR {hr_val}, K {k_val}")

        # print()
    # 3) Value Moneyline Bets
    value_html = vm.drop(columns=["event_id", "bookmaker"]) \
        .to_html(index=False, border=0, classes="value-table")

    # 4) Top 20 Batter Props (Value)
    # instead of vp_batters.head(20), do:
    # new
    top20_batters_df = vp_batters[
        (vp_batters["prop"] == "TB") &
        (vp_batters["model_pred"] >= 2)
        ].head(50)
    # top10_hr_df = hr_candidates.copy()
    top10_hr_df = vp_batters[vp_batters.prop == "HR"] \
        .sort_values(["model_pred", "model_prob"], ascending=False) \
        .head(10)

    # ── 7. Summary printouts ────────────────────────────────────────────────
    # gpreds: your game‐level predictions DataFrame
    # hr_candidates: top‐10 HR props DataFrame
    # vp_pitchers: all pitcher‐props with EV
    # candidates: under‐run hitter candidates DataFrame
    # top20_batters_df: top‐20 TB props DataFrame

    print("\n=== Away-Team vs Home-Team Game Predictions ===")
    print(
        gpreds[[
            "away_team_abbr",
            "home_team_abbr",
            "away_score_pred",
            "home_score_pred",
            "away_win_prob"
        ]].to_string(index=False)
    )
    print("\n=== Baseline predictions (runs_pg) ===")
    print(baseline[["away_team_abbr", "home_team_abbr", "away_score_pred", "home_score_pred"]])

    # 1) Invert your existing name_map (id → player_name) to get name → id
    id_map = {v: k for k, v in name_map.items()}

    # 2) From your seasonal DataFrame grab the team abbreviation
    roster = seasonal["player_stats_totals_2025-regular"]  # came from gather_seasonal
    # make a dict: player.id → team.abbreviation
    team_map = (
        roster
        .set_index("player.id")["team.abbreviation"]
        .to_dict()
    )

    # 3) Enrich each of your result‐tables

    # --- Home Run Candidates ---
    hr_candidates["player_id"] = hr_candidates["player"].map(id_map)
    hr_candidates["team"] = hr_candidates["player_id"].map(team_map)

    print("\n=== Top 10 Home-Run Candidates ===")
    print(
        hr_candidates[[
            "player",
            "team",
            "line",
            "model_prob",
            "ev"
        ]]
        .sort_values("ev", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    # --- Pitcher Props (K) ---
    vp_pitchers["player_id"] = vp_pitchers["player"].map(id_map)
    vp_pitchers["team"] = vp_pitchers["player_id"].map(team_map)

    print("\n=== Top Pitcher Props ===")
    print(
        vp_pitchers[[
            "player",
            "team",
            "prop",  # should be "K"
            "line",
            "model_pred",
            "model_prob",
            "ev"
        ]]
        .sort_values("ev", ascending=False)
        .to_string(index=False)
    )

    # --- Under-Run Hitter Candidates ---
    # this one already has a team_abbr column; just rename it for consistency
    candidates = candidates.rename(columns={"team_abbr": "team"})

    print("\n=== Under-Run Hitter Candidates ===")
    print(
        candidates[[
            "player_name",
            "team",
            "player_BA",
            "rain_or_windy"
        ]]
        .to_string(index=False)
    )

    # --- Top 20 Batter Props (TB) ---
    top20 = top20_batters_df.copy()
    top20["player_id"] = top20["player"].map(id_map)
    top20["team"] = top20["player_id"].map(team_map)

    print("\n=== Top 20 Batter Props ===")
    print(
        top20[[
            "player",
            "team",
            "prop",  # "TB"
            "line",
            "model_pred",
            "model_prob",
            "ev"
        ]]
        .sort_values("ev", ascending=False)
        .head(20)
        .to_string(index=False)
    )

    # ——————————————

    # 1) Game Predictions
    g = gpreds.copy()
    g["total_score"] = g["away_score_pred"] + g["home_score_pred"]
    scores_html = g[[
        "away_team_abbr", "home_team_abbr",
        "away_score_pred", "home_score_pred", "total_score"
    ]].to_html(index=False, border=0, classes="scores-table")

    # 2) Game Details
    # build a list of lines
    # build a flat list of all players (SPs + batters) exactly as before,
    # but *omit* Role and Game on every row
    details_rows = []
    for _, game in gpreds.iterrows():
        # label = f"{game.away_team_abbr}: {game.away_score_pred:.1f} @ {game.home_team_abbr}: {game.home_score_pred:.1f}"
        away_abbr = game.away_team_abbr
        home_abbr = game.home_team_abbr
        label = f"{away_abbr}: {game.away_score_pred:.1f} @ {home_abbr}: {game.home_score_pred:.1f}"
        gid = game.game_id
        lu = lineup_map.get(str(gid))
        if lu is None:
            logger.warning(f"No lineup data for game {gid}, skipping that game")
            continue
        # --- starters first (they'll have K_pred) ---
        for team_block in lu.get("teamLineups", []):
            starter_map: Dict[Tuple[str, str], int] = {}

            abbr = team_block.get("team", {}).get("abbreviation", "<unknown>")
            # safely grab either the actual or expected lineup, or empty dict
            lineup = team_block.get("actual") or team_block.get("expected") or {}
            positions = lineup.get("lineupPositions", [])

            if not positions:
                logger.warning("No lineupPositions found for game %s team %s", gid, abbr)
                continue

            # find the first pitcher in the positions list
            sp = next(
                (p for p in positions
                 if p.get("position") == "P" and p.get("player")),
                None
            )
            if not sp:
                logger.warning("No starting pitcher in lineupPositions for game %s team %s", gid, abbr)
                continue

            player = sp["player"]
            pid = player.get("id")
            name = f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()

            # map (game_id, team_abbr) → pitcher id
            starter_map[(gid, abbr)] = pid
            logger.debug("Starter for game %s / %s is %s (id=%s)", gid, abbr, name, pid)

            # lookup K_pred (or blank if missing)
            k_val = ""
            if pid in ppitch.index:
                ser = ppitch.loc[pid, "K_p"]
                val = float(ser.iloc[0]) if hasattr(ser, "iloc") else float(ser)
                k_val = f"{val:.1f}"

            details_rows.append({
                "GameLabel": label,
                "away_team_abbr": away_abbr,
                "home_team_abbr": home_abbr,
                "Team": abbr,
                "Player": name,
                "TB_pred": "",
                "HR_pred": "",
                "K_pred": k_val
            })

        # --- then every batter (they’ll have TB_pred & HR_pred) ---
        for team_block in lu.get("teamLineups", []):
            abbr = team_block.get("team", {}).get("abbreviation")
            lineup = team_block.get("actual") or team_block.get("expected") or {}
            raw_positions = lineup.get("lineupPositions", [])

            if not raw_positions:
                logger.warning("No lineupPositions found for game %s team %s", gid, abbr)
                continue

            # ---- DEDUPE snippet starts here ----
            seen = set()
            positions = []
            for pos in raw_positions:
                p = pos.get("player")
                if not p:
                    continue
                pid = p["id"]
                if pid not in seen:
                    seen.add(pid)
                    positions.append(pos)
            # ---- DEDUPE snippet ends here ----

            bats = [p for p in positions if p["position"] != "P"]
            for slot in bats:
                ply = slot["player"]
                pid = ply["id"]
                name = f"{ply['firstName']} {ply['lastName']}"

                if pid in ppreds.index:
                    raw_tb = ppreds.loc[pid, "TB"]
                    raw_hr = ppreds.loc[pid, "HR"]
                    raw_k = ppreds.loc[pid, "K"]

                    def unwrap(s):
                        return float(s.iloc[0]) if hasattr(s, "iloc") else float(s)

                    tb_val = f"{unwrap(raw_tb):.1f}"
                    hr_val = f"{unwrap(raw_hr):.1f}"
                    k_val = f"{unwrap(raw_k):.1f}"
                else:
                    tb_val = hr_val = k_val = ""

                details_rows.append({
                    "GameLabel": label,
                    "away_team_abbr": away_abbr,
                    "home_team_abbr": home_abbr,
                    "Team": abbr,
                    "Player": name,
                    "TB_pred": tb_val,
                    "HR_pred": hr_val,
                    "K_pred": k_val
                })

    # build DataFrame
    details_df = pd.DataFrame(details_rows)

    # drop any repeated (Game, Team, Player) combinations
    details_df = details_df.drop_duplicates(subset=["GameLabel", "Team", "Player"])
    # ——— guard .str.strip() ———
    # — after building and deduplicating details_df —

    if details_df.empty:
        logger.warning("details_df is empty – skipping details-table generation.")
        details_table = "<p>No game details available.</p>"
    else:
        # make sure those columns exist and are strings
        for c in ("away_team_abbr", "home_team_abbr"):
            if c not in details_df.columns:
                raise KeyError(f"{c} is missing from details_df!")
            details_df[c] = details_df[c].astype(str)

        # populate full team names, strip whitespace
        details_df["away_full"] = details_df["away_team_abbr"].str.strip()
        details_df["home_full"] = details_df["home_team_abbr"].str.strip()
        # build the GameLabel
        details_df["GameLabel"] = details_df["away_full"] + " @ " + details_df["home_full"]

        # now build your HTML
        html = ['<table border="0" cellpadding="4" cellspacing="0" class="details-table">']
        for game_label, grp in details_df.groupby("GameLabel"):
            html.append(f"""
              <tr>
                <th colspan="5" style="background:#efefef;text-align:left">
                  {game_label}
                </th>
              </tr>
            """)
            html.append("<tr><th>Team</th><th>Player</th><th>TB</th><th>HR</th><th>K</th></tr>")
            for _, row in grp.iterrows():
                html.append(
                    "<tr>"
                    f"<td>{row.Team}</td>"
                    f"<td>{row.Player}</td>"
                    f"<td>{row.TB_pred}</td>"
                    f"<td>{row.HR_pred}</td>"
                    f"<td>{row.K_pred}</td>"
                    "</tr>"
                )
        html.append("</table>")
        details_table = "\n".join(html)

    # 5) Top 10 Home-Run Candidates by model_pred then model_prob

    # drop any columns you don’t want rendered (e.g. event_id)

    top20_html = top20_batters_df.drop(columns=["event_id"]) \
        .to_html(index=False, border=0, classes="batters-table")

    # 6) All Pitcher Prop Predictions
    all_pitchers_html = vp_pitchers.drop(columns=["event_id"]) \
        .to_html(index=False, border=0, classes="pitchers-table")

    # 7) Under-Run Hitter Candidates
    under_run_html = candidates.to_html(index=False, border=0, classes="under-run-table")

    # 5) Top 10 Home-Run Candidates (we already computed these in `hr_candidates`)
    top10_hr_html = hr_candidates.to_html(index=False, border=0, classes="hr-table")

    # build our parlay off *those* names
    hr_bets = hr_candidates["player"].tolist()
    hr_prompt = f"homerun parlay for {', '.join(hr_bets)} on FanDuel or DraftKings"
    hr_link = await create_gambly_slip_link(raw_prompt=hr_prompt)

    # for TB >1.5
    tb_bets = [
        f"{row.player}"
        for _, row in top20_batters_df.iterrows()
    ]
    # 2) Total‐bases parlay
    tb_prompt = (
        f"over 1.5 total bases parlay for {', '.join(tb_bets)} "
        "on FanDuel or DraftKings"
    )
    tb_link = await create_gambly_slip_link(raw_prompt=tb_prompt)

    # for K props
    k_bets = [
        f"{row.player}"
        for _, row in vp_pitchers.iterrows()
    ]
    k_prompt = (
        f"over strikeout parlay for {', '.join(k_bets)} "
        "on FanDuel or DraftKings"
    )
    k_link = await create_gambly_slip_link(raw_prompt=k_prompt)

    # for moneyline bets (FanDuel only)
    # pick the team with the higher predicted score from gpreds
    import numpy as np

    # add a column to gpreds for which side is favoured
    gpreds["fav_team"] = np.where(
        gpreds["away_score_pred"] > gpreds["home_score_pred"],
        gpreds["away_team_abbr"],
        gpreds["home_team_abbr"]
    )

    ml_bets = gpreds["fav_team"].tolist()

    ml_prompt = (
        f"moneyline parlay for {', '.join(ml_bets)} "
        "on FanDuel or DraftKings"
    )
    ml_link = await create_gambly_slip_link(raw_prompt=ml_prompt)

    html_body = generate_html_email_body(
        scores_html,
        details_table,
        value_html,
        top20_html,
        top10_hr_html,
        all_pitchers_html,
        under_run_html,
        hr_link,
        tb_link,
        k_link,
        ml_link
    )

    send_email(
        subject="📧 MLB Daily Value & Props Report",
        plain_body="Please open in an HTML-capable client.",
        to_email="jjohnson0636@gmail.com",
        html_body=html_body
    )

    if pool:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())