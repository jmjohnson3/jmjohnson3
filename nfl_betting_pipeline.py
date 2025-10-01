"""NFL betting pipeline.

This module downloads NFL data from the MySportsFeeds API, persists it into a
PostgreSQL warehouse, fetches complementary odds from The Odds API, and builds
simple predictive models to surface potential betting edges.

The script is intentionally modular: each step (data ingestion, storage,
feature engineering, and modelling) can be run independently when imported as a
library, or orchestrated end-to-end from the command line.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from sqlalchemy import JSON, Column, Date, Integer, MetaData, String, Table, create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import OneHotEncoder


LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MySportsFeedsConfig:
    """Configuration for the MySportsFeeds API."""

    username: str
    password: str
    season: str
    league: str = "nfl"
    format: str = "json"

    def auth_tuple(self) -> Tuple[str, str]:
        return self.username, self.password


@dataclasses.dataclass
class OddsApiConfig:
    """Configuration for The Odds API."""

    api_key: str
    region: str = "us"
    market: str = "h2h"
    odds_format: str = "american"


@dataclasses.dataclass
class DatabaseConfig:
    """Configuration for the PostgreSQL connection."""

    uri: str


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------


class MySportsFeedsClient:
    BASE_URL = "https://api.mysportsfeeds.com/v2.1/pull"

    def __init__(self, config: MySportsFeedsConfig) -> None:
        self.config = config

    def _build_url(self, endpoint: str) -> str:
        return f"{self.BASE_URL}/{self.config.league}/{self.config.season}/{endpoint}.{self.config.format}"

    def fetch_games(self, week: int) -> List[Dict]:
        params = {"week": week}
        url = self._build_url("games")
        LOG.debug("Requesting MySportsFeeds games: %s %s", url, params)
        response = requests.get(url, params=params, auth=self.config.auth_tuple(), timeout=30)
        response.raise_for_status()
        payload = response.json()
        games = payload.get("games", [])
        LOG.info("Fetched %d games for week %d", len(games), week)
        return games

    def fetch_player_gamelogs(self, week: int) -> List[Dict]:
        params = {"week": week}
        url = self._build_url("player_gamelogs")
        LOG.debug("Requesting MySportsFeeds gamelogs: %s %s", url, params)
        response = requests.get(url, params=params, auth=self.config.auth_tuple(), timeout=30)
        response.raise_for_status()
        payload = response.json()
        logs = payload.get("gamelogs", [])
        LOG.info("Fetched %d player logs for week %d", len(logs), week)
        return logs


class OddsApiClient:
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, config: OddsApiConfig) -> None:
        self.config = config

    def fetch_game_odds(self, sport: str = "americanfootball_nfl") -> List[Dict]:
        url = f"{self.BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey": self.config.api_key,
            "regions": self.config.region,
            "markets": self.config.market,
            "oddsFormat": self.config.odds_format,
        }
        LOG.debug("Requesting odds: %s %s", url, params)
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        LOG.info("Fetched %d odds entries", len(data))
        return data


# ---------------------------------------------------------------------------
# Database utilities
# ---------------------------------------------------------------------------


def create_tables(engine: Engine) -> Dict[str, Table]:
    metadata = MetaData()

    games = Table(
        "games",
        metadata,
        Column("id", String, primary_key=True),
        Column("week", Integer, nullable=False),
        Column("start_time", Date, nullable=True),
        Column("venue", String, nullable=True),
        Column("home_team", String, nullable=False),
        Column("away_team", String, nullable=False),
        Column("home_score", Integer, nullable=True),
        Column("away_score", Integer, nullable=True),
        Column("weather", JSON, nullable=True),
        Column("raw", JSON, nullable=False),
    )

    player_stats = Table(
        "player_stats",
        metadata,
        Column("id", String, primary_key=True),
        Column("game_id", String, nullable=False),
        Column("team", String, nullable=False),
        Column("player_name", String, nullable=False),
        Column("position", String, nullable=True),
        Column("offense_stats", JSON, nullable=True),
        Column("defense_stats", JSON, nullable=True),
        Column("raw", JSON, nullable=False),
    )

    odds = Table(
        "odds",
        metadata,
        Column("id", String, primary_key=True),
        Column("game_id", String, nullable=True),
        Column("sport_key", String, nullable=False),
        Column("home_team", String, nullable=False),
        Column("away_team", String, nullable=False),
        Column("commence_time", Date, nullable=True),
        Column("bookmakers", JSON, nullable=False),
        Column("raw", JSON, nullable=False),
    )

    metadata.create_all(engine)
    return {"games": games, "player_stats": player_stats, "odds": odds}


class PostgresRepository:
    def __init__(self, config: DatabaseConfig) -> None:
        self.engine = create_engine(config.uri)
        self.tables = create_tables(self.engine)

    def upsert(self, table: Table, rows: Sequence[Dict]) -> None:
        if not rows:
            LOG.info("No rows to upsert for table %s", table.name)
            return
        try:
            with self.engine.begin() as conn:
                stmt = insert(table).values(rows)
                update_cols = {col.name: col for col in table.c if col.name not in ("id",)}
                stmt = stmt.on_conflict_do_update(index_elements=[table.c.id], set_=update_cols)
                conn.execute(stmt)
            LOG.info("Upserted %d rows into %s", len(rows), table.name)
        except SQLAlchemyError as exc:
            LOG.exception("Failed to upsert into %s", table.name)
            raise RuntimeError(f"Database upsert failed: {exc}") from exc

    def fetch_dataframe(self, query: str) -> pd.DataFrame:
        try:
            with self.engine.connect() as conn:
                return pd.read_sql_query(query, conn)
        except SQLAlchemyError as exc:
            LOG.exception("Query failed: %s", query)
            raise RuntimeError(f"Database query failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Feature engineering and modelling
# ---------------------------------------------------------------------------


def parse_game_row(game: Dict, week: int) -> Dict:
    game_id = game.get("schedule", {}).get("id") or game.get("game", {}).get("id")
    schedule = game.get("schedule", {})
    weather = game.get("weather", {}) or game.get("game", {}).get("weather", {})
    venue = schedule.get("venue", {}).get("name") if schedule.get("venue") else None
    start_time = schedule.get("startTime") or game.get("game", {}).get("startTime")
    scoreboard = game.get("score", {}) or game.get("scoreboard", {})
    home_score = (
        scoreboard.get("homeScoreTotal")
        or scoreboard.get("homeScore")
        or schedule.get("homeScoreTotal")
        or game.get("game", {}).get("homeScoreTotal")
    )
    away_score = (
        scoreboard.get("awayScoreTotal")
        or scoreboard.get("awayScore")
        or schedule.get("awayScoreTotal")
        or game.get("game", {}).get("awayScoreTotal")
    )
    parsed_time = None
    if start_time:
        try:
            parsed_time = datetime.fromisoformat(start_time.replace("Z", "+00:00")).date()
        except ValueError:
            parsed_time = None
    return {
        "id": str(game_id),
        "week": week,
        "start_time": parsed_time,
        "venue": venue,
        "home_team": schedule.get("homeTeam", {}).get("abbreviation") or game.get("homeTeam", {}).get("abbreviation"),
        "away_team": schedule.get("awayTeam", {}).get("abbreviation") or game.get("awayTeam", {}).get("abbreviation"),
        "home_score": home_score,
        "away_score": away_score,
        "weather": weather,
        "raw": game,
    }


def parse_player_row(log: Dict) -> Dict:
    stat = log.get("stats", {})
    player = log.get("player", {})
    team = log.get("team", {})
    game = log.get("game", {})
    return {
        "id": f"{player.get('id')}-{game.get('id')}",
        "game_id": str(game.get("id")),
        "team": team.get("abbreviation"),
        "player_name": player.get("firstName", "") + " " + player.get("lastName", ""),
        "position": player.get("position"),
        "offense_stats": stat.get("offense") or stat.get("rushing") or stat.get("passing"),
        "defense_stats": stat.get("defense"),
        "raw": log,
    }


def parse_odds_row(entry: Dict) -> Dict:
    commence = entry.get("commence_time")
    parsed_time = None
    if commence:
        try:
            parsed_time = datetime.fromisoformat(commence.replace("Z", "+00:00")).date()
        except ValueError:
            parsed_time = None
    return {
        "id": str(entry.get("id", entry.get("sport_key") + entry.get("commence_time", ""))),
        "game_id": entry.get("id"),
        "sport_key": entry.get("sport_key"),
        "home_team": entry.get("home_team"),
        "away_team": entry.get("away_team"),
        "commence_time": parsed_time,
        "bookmakers": entry.get("bookmakers", []),
        "raw": entry,
    }


def build_game_features(games: pd.DataFrame, player_stats: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        LOG.warning("No games available for feature engineering")
        return pd.DataFrame()

    features = games.copy()

    if not player_stats.empty:
        offense = player_stats[["game_id", "team", "offense_stats"]].copy()
        offense["offense_stats"] = offense["offense_stats"].apply(lambda stats: stats if isinstance(stats, dict) else {})
        offense_numeric = pd.json_normalize(offense["offense_stats"])
        offense = pd.concat([offense.drop(columns=["offense_stats"]), offense_numeric], axis=1)
        offense_grouped = offense.groupby(["game_id", "team"]).sum(numeric_only=True).reset_index()

        home_stats = offense_grouped.rename(columns={"team": "home_team"})
        home_stats = home_stats.add_prefix("home_")
        home_stats = home_stats.rename(columns={"home_game_id": "id", "home_home_team": "home_team"})
        features = features.merge(home_stats, on=["id", "home_team"], how="left")

        away_stats = offense_grouped.rename(columns={"team": "away_team"})
        away_stats = away_stats.add_prefix("away_")
        away_stats = away_stats.rename(columns={"away_game_id": "id", "away_away_team": "away_team"})
        features = features.merge(away_stats, on=["id", "away_team"], how="left")

        defense = player_stats[["game_id", "team", "defense_stats"]].copy()
        defense["defense_stats"] = defense["defense_stats"].apply(lambda stats: stats if isinstance(stats, dict) else {})
        defense_numeric = pd.json_normalize(defense["defense_stats"])
        defense = pd.concat([defense.drop(columns=["defense_stats"]), defense_numeric], axis=1)
        defense_grouped = defense.groupby(["game_id", "team"]).sum(numeric_only=True).reset_index()

        home_def = defense_grouped.rename(columns={"team": "home_team"}).add_prefix("home_def_")
        home_def = home_def.rename(columns={"home_def_game_id": "id", "home_def_home_team": "home_team"})
        features = features.merge(home_def, on=["id", "home_team"], how="left")

        away_def = defense_grouped.rename(columns={"team": "away_team"}).add_prefix("away_def_")
        away_def = away_def.rename(columns={"away_def_game_id": "id", "away_def_away_team": "away_team"})
        features = features.merge(away_def, on=["id", "away_team"], how="left")

    weather_series = features["weather"].apply(lambda w: w if isinstance(w, dict) else {})
    weather_df = pd.json_normalize(weather_series)
    weather_df.columns = [f"weather_{col}" for col in weather_df.columns]
    features = pd.concat([features.drop(columns=["weather", "raw"], errors="ignore"), weather_df], axis=1)

    venue_str = features["venue"].fillna("").astype(str)
    features["venue_type"] = np.where(venue_str.str.contains("Field", case=False), "outdoor", "indoor")

    if {"home_score", "away_score"}.issubset(features.columns):
        features["home_win"] = np.where(features["home_score"] > features["away_score"], 1, 0)
        features["points_scored"] = features[["home_score", "away_score"]].sum(axis=1)

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)
    features = features.fillna("")
    return features


def train_models(features: pd.DataFrame) -> Tuple[Optional[LinearRegression], Optional[LogisticRegression], Optional[OneHotEncoder]]:
    if features.empty:
        LOG.warning("No features available for model training")
        return None, None, None

    encoder = OneHotEncoder(handle_unknown="ignore")
    categorical = features[["home_team", "away_team", "venue_type"]]
    encoded = encoder.fit_transform(categorical).toarray()

    numeric_cols = [
        col
        for col in features.columns
        if col
        not in {"id", "start_time", "home_team", "away_team", "venue", "venue_type", "week", "home_win", "points_scored"}
        and not col.startswith("weather_")
    ]
    numeric = features[numeric_cols].to_numpy(dtype=float, copy=False)

    X = np.hstack([encoded, numeric]) if numeric.size else encoded

    y_points = features.get("points_scored")
    reg_model = None
    if y_points is not None and np.unique(y_points).size > 1:
        try:
            reg_model = LinearRegression().fit(X, y_points)
        except ValueError as exc:
            LOG.warning("Skipping points model: %s", exc)

    y_outcome = features.get("home_win")
    clf_model = None
    if y_outcome is not None and np.unique(y_outcome).size > 1:
        try:
            clf_model = LogisticRegression(max_iter=1000).fit(X, y_outcome)
        except ValueError as exc:
            LOG.warning("Skipping win model: %s", exc)

    return reg_model, clf_model, encoder


def predict_outcomes(
    features: pd.DataFrame,
    encoder: OneHotEncoder,
    reg_model: Optional[LinearRegression],
    clf_model: Optional[LogisticRegression],
) -> pd.DataFrame:
    if features.empty or encoder is None:
        LOG.warning("Cannot generate predictions without features and encoder")
        return pd.DataFrame()

    categorical = features[["home_team", "away_team", "venue_type"]]
    encoded = encoder.transform(categorical).toarray()

    numeric_cols = [
        col
        for col in features.columns
        if col
        not in {"id", "start_time", "home_team", "away_team", "venue", "venue_type", "week", "home_win", "points_scored"}
        and not col.startswith("weather_")
    ]
    numeric = features[numeric_cols].to_numpy(dtype=float, copy=False)
    X = np.hstack([encoded, numeric]) if numeric.size else encoded

    predictions = pd.DataFrame(
        {
            "game_id": features["id"],
            "home_team": features["home_team"],
            "away_team": features["away_team"],
        }
    )

    if reg_model is not None:
        predictions["predicted_points"] = reg_model.predict(X)

    if clf_model is not None:
        predictions["home_win_probability"] = clf_model.predict_proba(X)[:, 1]

    return predictions


def american_to_probability(odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    if odds <= -100:
        return -odds / (-odds + 100.0)
    return None


def _ensure_bookmakers(value) -> List[Dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def compute_betting_edges(predictions: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or odds_df.empty:
        LOG.warning("No odds available to compute betting edges")
        return predictions

    def best_offer(bookmakers: List[Dict], team: str) -> Tuple[Optional[float], Optional[str]]:
        best_price: Optional[float] = None
        best_source: Optional[str] = None
        for bookmaker in bookmakers:
            bm_name = bookmaker.get("title") or bookmaker.get("key")
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("name") != team:
                        continue
                    price = outcome.get("price") or outcome.get("odds")
                    if price is None:
                        continue
                    try:
                        price = float(price)
                    except (TypeError, ValueError):
                        continue
                    if best_price is None or price > best_price:
                        best_price = price
                        best_source = bm_name
        return best_price, best_source

    enriched = predictions.copy()
    enriched[[
        "home_best_odds",
        "home_best_book",
        "home_implied_prob",
        "home_edge",
        "away_best_odds",
        "away_best_book",
        "away_implied_prob",
        "away_edge",
    ]] = None

    for idx, row in enriched.iterrows():
        matching = odds_df[(odds_df["home_team"] == row["home_team"]) & (odds_df["away_team"] == row["away_team"])]
        if matching.empty:
            continue
        bookmakers: List[Dict] = []
        for _, odds_row in matching.iterrows():
            bookmakers.extend(_ensure_bookmakers(odds_row.get("bookmakers")))

        home_price, home_book = best_offer(bookmakers, row["home_team"])
        away_price, away_book = best_offer(bookmakers, row["away_team"])

        home_implied = american_to_probability(home_price)
        away_implied = american_to_probability(away_price)

        enriched.at[idx, "home_best_odds"] = home_price
        enriched.at[idx, "home_best_book"] = home_book
        enriched.at[idx, "home_implied_prob"] = home_implied
        if home_implied is not None and "home_win_probability" in enriched:
            enriched.at[idx, "home_edge"] = enriched.at[idx, "home_win_probability"] - home_implied

        enriched.at[idx, "away_best_odds"] = away_price
        enriched.at[idx, "away_best_book"] = away_book
        enriched.at[idx, "away_implied_prob"] = away_implied
        if away_implied is not None and "home_win_probability" in enriched:
            away_prob = 1 - enriched.at[idx, "home_win_probability"] if "home_win_probability" in enriched else None
            if away_prob is not None:
                enriched.at[idx, "away_edge"] = away_prob - away_implied

    return enriched


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def ingest_week(
    week: int,
    msf_client: MySportsFeedsClient,
    repo: PostgresRepository,
    odds_client: Optional[OddsApiClient] = None,
) -> None:
    games = msf_client.fetch_games(week)
    repo.upsert(repo.tables["games"], [parse_game_row(game, week) for game in games])

    gamelogs = msf_client.fetch_player_gamelogs(week)
    repo.upsert(repo.tables["player_stats"], [parse_player_row(log) for log in gamelogs])

    if odds_client is not None:
        odds_entries = odds_client.fetch_game_odds()
        repo.upsert(repo.tables["odds"], [parse_odds_row(entry) for entry in odds_entries])


def run_pipeline(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG)

    msf_config = MySportsFeedsConfig(
        username=args.msf_username,
        password=args.msf_password,
        season=args.season,
        league=args.league,
    )
    msf_client = MySportsFeedsClient(msf_config)

    odds_client = None
    if args.odds_api_key:
        odds_client = OddsApiClient(OddsApiConfig(api_key=args.odds_api_key, region=args.odds_region, market=args.odds_market))

    db_config = DatabaseConfig(uri=args.database_uri)
    repo = PostgresRepository(db_config)

    for week in args.weeks:
        LOG.info("Ingesting week %d", week)
        ingest_week(week, msf_client, repo, odds_client)

    games_df = repo.fetch_dataframe("SELECT * FROM games")
    player_df = repo.fetch_dataframe("SELECT * FROM player_stats")

    features = build_game_features(games_df, player_df)
    reg_model, clf_model, encoder = train_models(features)

    predictions = predict_outcomes(features, encoder, reg_model, clf_model)
    if odds_client is not None:
        odds_df = repo.fetch_dataframe("SELECT * FROM odds")
        predictions = compute_betting_edges(predictions, odds_df)
    if predictions.empty:
        LOG.warning("No predictions generated")
        return

    LOG.info("Generated %d predictions", len(predictions))
    print(predictions.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFL betting pipeline using MySportsFeeds and The Odds API")
    parser.add_argument("season", help="Season string understood by MySportsFeeds (e.g. 2024-regular)")
    parser.add_argument("weeks", nargs="+", type=int, help="Week numbers to ingest and model")
    parser.add_argument("--league", default="nfl", help="Sports league key for MySportsFeeds (default: nfl)")
    parser.add_argument("--msf-username", default=os.getenv("MSF_USERNAME"), help="MySportsFeeds username (env: MSF_USERNAME)")
    parser.add_argument("--msf-password", default=os.getenv("MSF_PASSWORD"), help="MySportsFeeds password (env: MSF_PASSWORD)")
    parser.add_argument("--database-uri", default=os.getenv("DATABASE_URL", "postgresql://localhost:5432/nfl"), help="SQLAlchemy URI for PostgreSQL")
    parser.add_argument("--odds-api-key", default=os.getenv("ODDS_API_KEY"), help="The Odds API key (env: ODDS_API_KEY)")
    parser.add_argument("--odds-region", default="us", help="Odds API region parameter")
    parser.add_argument("--odds-market", default="h2h", help="Odds API market parameter")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args(argv)

    if not args.msf_username or not args.msf_password:
        parser.error("MySportsFeeds credentials are required via flags or environment variables")

    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        run_pipeline(args)
    except (requests.RequestException, RuntimeError) as exc:
        LOG.error("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
