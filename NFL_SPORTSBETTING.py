#!/usr/bin/env python3
"""Comprehensive NFL betting analytics and prediction pipeline.

This script ingests game, player, and odds data from the MySportsFeeds and The Odds
API services, persists the information to PostgreSQL, and then builds machine
learning models to predict player performance and game outcomes.

The workflow is designed to run incrementally: the first execution ingests all
available data for the configured seasons, while subsequent runs only request new
games and odds. The machine learning models incorporate contextual features such
as venue effects, day-of-week trends, officiating crews, weather, and team unit
strengths (rush/pass offense & defense) to deliver rich predictive insights that
can be used to identify profitable betting opportunities.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests import HTTPError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_PREFIX_NFL = "https://api.mysportsfeeds.com/v2.1/pull/nfl"
NFL_SEASONS = ["2024-regular", "2025-regular"]

NFL_API_USER = "4359aa1b-cc29-4647-a3e5-7314e2"
NFL_API_PASS = "MYSPORTSFEEDS"

ODDS_API_KEY = "5b6f0290e265c3329b3ed27897d79eaf"
ODDS_BASE = "https://api.the-odds-api.com/v4"
NFL_SPORT_KEY = "americanfootball_nfl"
ODDS_REGIONS = ["us"]
ODDS_FORMAT = "american"

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


def default_now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_dt(value: str) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclasses.dataclass
class NFLConfig:
    pg_user: str = os.getenv("PGUSER", "josh")
    pg_password: str = os.getenv("PGPASSWORD", "password")
    pg_host: str = os.getenv("PGHOST", "localhost")
    pg_port: str = os.getenv("PGPORT", "5432")
    pg_database: str = os.getenv("PGDATABASE", "nfl")

    seasons: Tuple[str, ...] = tuple(NFL_SEASONS)
    log_level: str = DEFAULT_LOG_LEVEL

    @property
    def pg_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


# ---------------------------------------------------------------------------
# Database schema & helpers
# ---------------------------------------------------------------------------


class NFLDatabase:
    """Encapsulates PostgreSQL persistence for NFL data."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.meta = MetaData()
        self._define_tables()
        self.meta.create_all(self.engine)

    def _define_tables(self) -> None:
        self.games = Table(
            "nfl_games",
            self.meta,
            Column("game_id", String, primary_key=True),
            Column("season", String, nullable=False),
            Column("week", Integer),
            Column("start_time", DateTime(timezone=True)),
            Column("venue", String),
            Column("city", String),
            Column("state", String),
            Column("country", String),
            Column("surface", String),
            Column("day_of_week", String),
            Column("referee", String),
            Column("temperature_f", Float),
            Column("weather_conditions", String),
            Column("home_team", String),
            Column("away_team", String),
            Column("home_score", Integer),
            Column("away_score", Integer),
            Column("status", String),
            Column("home_moneyline", Float),
            Column("away_moneyline", Float),
            Column("home_implied_prob", Float),
            Column("away_implied_prob", Float),
            Column("odds_updated", DateTime(timezone=True)),
            Column("ingested_at", DateTime(timezone=True), default=default_now_utc),
        )

        self.player_stats = Table(
            "nfl_player_stats",
            self.meta,
            Column("game_id", String, nullable=False),
            Column("player_id", String, nullable=False),
            Column("player_name", String),
            Column("team", String),
            Column("position", String),
            Column("rushing_attempts", Float),
            Column("rushing_yards", Float),
            Column("rushing_tds", Float),
            Column("receiving_targets", Float),
            Column("receptions", Float),
            Column("receiving_yards", Float),
            Column("receiving_tds", Float),
            Column("passing_attempts", Float),
            Column("passing_completions", Float),
            Column("passing_yards", Float),
            Column("passing_tds", Float),
            Column("fantasy_points", Float),
            Column("snap_count", Float),
            Column("ingested_at", DateTime(timezone=True), default=default_now_utc),
            UniqueConstraint("game_id", "player_id", name="uq_player_game"),
        )

        self.team_unit_ratings = Table(
            "nfl_team_unit_ratings",
            self.meta,
            Column("season", String, nullable=False),
            Column("team", String, nullable=False),
            Column("week", Integer, nullable=False),
            Column("offense_pass_rating", Float),
            Column("offense_rush_rating", Float),
            Column("defense_pass_rating", Float),
            Column("defense_rush_rating", Float),
            Column("updated_at", DateTime(timezone=True), default=default_now_utc),
            UniqueConstraint("season", "team", "week", name="uq_team_week"),
        )

        self.model_predictions = Table(
            "nfl_predictions",
            self.meta,
            Column("prediction_id", String, primary_key=True),
            Column("game_id", String, nullable=False),
            Column("entity_type", String, nullable=False),
            Column("entity_id", String, nullable=False),
            Column("prediction_target", String, nullable=False),
            Column("prediction_value", Float),
            Column("model_version", String),
            Column("features", JSON),
            Column("created_at", DateTime(timezone=True), default=default_now_utc),
        )

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def upsert_rows(
        self,
        table: Table,
        rows: Iterable[Dict[str, Any]],
        conflict_cols: List[str],
        update_columns: Optional[Iterable[str]] = None,
    ) -> None:
        if not rows:
            return
        stmt = insert(table).values(list(rows))
        if update_columns is None:
            update_cols = {
                col.name: stmt.excluded[col.name]
                for col in table.columns
                if col.name not in conflict_cols
            }
        else:
            valid_columns = {
                col
                for col in update_columns
                if col in table.c.keys() and col not in conflict_cols
            }
            update_cols = {col: stmt.excluded[col] for col in valid_columns}

        if update_cols:
            stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
        try:
            with self.engine.begin() as conn:
                conn.execute(stmt)
        except SQLAlchemyError:
            logging.exception("Failed to upsert rows into %s", table.name)
            raise

    def fetch_existing_game_ids(self) -> set[str]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(self.games.c.game_id)).fetchall()
        return {row[0] for row in rows}

    def fetch_games_with_player_stats(self) -> set[str]:
        """Return the set of game IDs that already have player statistics stored."""

        with self.engine.begin() as conn:
            rows = conn.execute(select(self.player_stats.c.game_id).distinct()).fetchall()
        return {row[0] for row in rows}

    def latest_team_rating_week(self, season: str) -> Optional[int]:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(func.max(self.team_unit_ratings.c.week)).where(self.team_unit_ratings.c.season == season)
            ).scalar()
        return row


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------


class MySportsFeedsClient:
    def __init__(self, user: str, password: str, timeout: int = 30):
        self.auth = (user, password)
        self.timeout = timeout

    def _request(self, endpoint: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{API_PREFIX_NFL}/{endpoint}"
        logging.debug("Requesting MySportsFeeds endpoint %s", url)
        resp = requests.get(url, params=params, auth=self.auth, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_games(self, season: str) -> List[Dict[str, Any]]:
        """Fetch the schedule for a season, retrying with alternative filters."""

        base_params: Dict[str, Any] = {"limit": 500}
        attempts: Tuple[Optional[str], ...] = (
            "completed,upcoming",
            "final,inprogress,scheduled",
            None,
        )

        for status_filter in attempts:
            params = dict(base_params)
            if status_filter:
                params["status"] = status_filter

            data = self._request(f"{season}/games.json", params=params)
            games = data.get("games", [])
            if games:
                if status_filter and status_filter != attempts[0]:
                    logging.debug(
                        "Fetched %d games for %s after retrying with status filter '%s'",
                        len(games),
                        season,
                        status_filter,
                    )
                return games

        logging.debug(
            "No games returned for %s even after retrying with multiple status filters",
            season,
        )
        return []

    def fetch_game_boxscore(self, season: str, game_id: str) -> Dict[str, Any]:
        return self._request(f"{season}/games/{game_id}/boxscore.json")

    def fetch_player_gamelogs(self, season: str, game_id: str) -> List[Dict[str, Any]]:
        try:
            data = self._request(
                f"{season}/games/{game_id}/player_gamelogs.json",
                params={"stats": "Rushing,Receiving,Passing,Fumbles"},
            )
        except HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                logging.debug(
                    "No player gamelogs found for season %s game %s (HTTP 404)",
                    season,
                    game_id,
                )
                return []
            raise
        return data.get("gamelogs", [])


class OddsApiClient:
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        url = f"{ODDS_BASE}/{endpoint}"
        params = params or {}
        params.update({"apiKey": self.api_key})
        logging.debug("Requesting Odds API endpoint %s", url)
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            logging.debug("Odds API requests remaining: %s", remaining)
        return resp.json()

    def fetch_odds(self) -> List[Dict[str, Any]]:
        params = {
            "regions": ",".join(ODDS_REGIONS),
            "oddsFormat": ODDS_FORMAT,
            "markets": "h2h",
        }
        return self._request(f"sports/{NFL_SPORT_KEY}/odds", params=params)


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------


class NFLIngestor:
    def __init__(self, db: NFLDatabase, msf_client: MySportsFeedsClient, odds_client: OddsApiClient):
        self.db = db
        self.msf_client = msf_client
        self.odds_client = odds_client

    def ingest(self, seasons: Iterable[str]) -> None:
        existing_games = self.db.fetch_existing_game_ids()
        games_with_stats = self.db.fetch_games_with_player_stats()
        logging.info("Found %d games already in database", len(existing_games))

        for season in seasons:
            games = self.msf_client.fetch_games(season)
            logging.info("Fetched %d games for season %s", len(games), season)
            if not games:
                logging.warning(
                    "No games returned from MySportsFeeds for %s. "
                    "Verify your API credentials, plan access, and season configuration.",
                    season,
                )

            new_game_rows: List[Dict[str, Any]] = []
            player_rows: List[Dict[str, Any]] = []

            for game in games:
                schedule = game.get("schedule", {})
                game_id = schedule.get("id")
                if not game_id:
                    continue

                game_id_str = str(game_id)
                have_player_stats = game_id_str in games_with_stats

                score = game.get("score") or {}
                home_score, away_score = self._extract_score_totals(score)
                start_time = parse_dt(schedule.get("startTime"))
                venue = schedule.get("venue") or {}
                weather = schedule.get("weather") or {}
                officials = schedule.get("officials") or []

                referee_name: Optional[str] = None
                if officials:
                    lead_official = officials[0] or {}
                    first = lead_official.get("firstName", "")
                    last = lead_official.get("lastName", "")
                    referee_name = f"{first} {last}".strip()
                    if not referee_name:
                        referee_name = lead_official.get("fullName")

                new_game_rows.append(
                    {
                        "game_id": game_id_str,
                        "season": season,
                        "week": schedule.get("week"),
                        "start_time": start_time,
                        "venue": venue.get("name"),
                        "city": venue.get("city"),
                        "state": venue.get("state"),
                        "country": venue.get("country"),
                        "surface": venue.get("surface"),
                        "day_of_week": start_time.strftime("%A") if start_time else None,
                        "referee": referee_name,
                        "temperature_f": self._extract_temperature_fahrenheit(weather.get("temperature")),
                        "weather_conditions": weather.get("conditions"),
                        "home_team": schedule.get("homeTeam", {}).get("abbreviation"),
                        "away_team": schedule.get("awayTeam", {}).get("abbreviation"),
                        "home_score": home_score,
                        "away_score": away_score,
                        "status": schedule.get("status"),
                    }
                )

                status = (
                    schedule.get("status")
                    or schedule.get("playedStatus")
                    or (game.get("status") if isinstance(game, dict) else None)
                    or ""
                ).lower()
                is_completed = status.startswith("final") or status in {"completed", "postponed"}

                if have_player_stats:
                    logging.debug(
                        "Skipping player stats for already ingested game %s", game_id_str
                    )
                    continue

                if not is_completed:
                    logging.debug(
                        "Game %s in season %s has status '%s'; skipping player stats fetch until completion",
                        game_id_str,
                        season,
                        schedule.get("status"),
                    )
                    continue

                gamelog_entries = self.msf_client.fetch_player_gamelogs(season, game_id_str)
                player_entries = list(gamelog_entries)
                if not player_entries:
                    logging.debug(
                        "No player gamelog entries returned for season %s game %s", season, game_id_str
                    )
                    fallback_entries = self._fetch_boxscore_player_stats(season, game_id_str)
                    if fallback_entries:
                        logging.debug(
                            "Using boxscore fallback for season %s game %s player stats",
                            season,
                            game_id_str,
                        )
                        player_entries = fallback_entries
                if not player_entries:
                    continue
                for entry in player_entries:
                    player = entry.get("player", {})
                    team = entry.get("team", {})
                    stats = entry.get("stats", {})

                    def stat_value(stat_group: str, field: str) -> Optional[float]:
                        group = stats.get(stat_group, {})
                        value = group.get(field, {})
                        return value.get("#text") or value.get("value")

                    player_rows.append(
                        {
                            "game_id": game_id_str,
                            "player_id": str(player.get("id")),
                            "player_name": f"{player.get('firstName', '')} {player.get('lastName', '')}".strip(),
                            "team": team.get("abbreviation"),
                            "position": player.get("position"),
                            "rushing_attempts": self._safe_float(stat_value("Rushing", "RushingAttempts")),
                            "rushing_yards": self._safe_float(stat_value("Rushing", "RushingYards")),
                            "rushing_tds": self._safe_float(stat_value("Rushing", "RushingTD")),
                            "receiving_targets": self._safe_float(stat_value("Receiving", "Targets")),
                            "receptions": self._safe_float(stat_value("Receiving", "Receptions")),
                            "receiving_yards": self._safe_float(stat_value("Receiving", "ReceivingYards")),
                            "receiving_tds": self._safe_float(stat_value("Receiving", "ReceivingTD")),
                            "passing_attempts": self._safe_float(stat_value("Passing", "PassAttempts")),
                            "passing_completions": self._safe_float(stat_value("Passing", "PassCompletions")),
                            "passing_yards": self._safe_float(stat_value("Passing", "PassYards")),
                            "passing_tds": self._safe_float(stat_value("Passing", "PassTD")),
                            "fantasy_points": self._safe_float(stat_value("Fantasy", "FantasyPoints")),
                            "snap_count": self._safe_float(stat_value("Miscellaneous", "Snaps")),
                        }
                    )

            self.db.upsert_rows(self.db.games, new_game_rows, ["game_id"])
            self.db.upsert_rows(self.db.player_stats, player_rows, ["game_id", "player_id"])
            if len(new_game_rows) == 0 and len(player_rows) == 0:
                logging.warning(
                    "Ingested %d new games and %d player stat rows for %s. "
                    "If these counts are unexpectedly low, confirm that your MySportsFeeds subscription "
                    "includes detailed stats and that the targeted seasons contain completed games.",
                    len(new_game_rows),
                    len(player_rows),
                    season,
                )

        # Ingest odds separately as they change frequently (always upsert)
        self._ingest_odds()

    def _ingest_odds(self) -> None:
        odds_data = self.odds_client.fetch_odds()
        logging.info("Fetched %d odds entries", len(odds_data))

        odds_rows: List[Dict[str, Any]] = []
        for event in odds_data:
            commence_time = parse_dt(event.get("commence_time"))
            home_team = event.get("home_team")
            away_team = next((t for t in event.get("teams", []) if t != home_team), None)
            markets = event.get("bookmakers", [])
            if not markets:
                continue
            # Use the freshest bookmaker odds
            market = sorted(markets, key=lambda b: parse_dt(b.get("last_update")) or default_now_utc(), reverse=True)[0]
            last_update = parse_dt(market.get("last_update"))
            h2h = next((m for m in market.get("markets", []) if m.get("key") == "h2h"), None)
            if not h2h:
                continue

            prices = {outcome.get("name"): outcome.get("price") for outcome in h2h.get("outcomes", [])}
            home_price = prices.get(home_team)
            away_price = prices.get(away_team)

            def american_to_prob(odds: Optional[float]) -> Optional[float]:
                if odds is None:
                    return None
                odds = float(odds)
                if odds > 0:
                    return 100.0 / (odds + 100.0)
                return -odds / (-odds + 100.0)

            odds_rows.append(
                {
                    "game_id": event.get("id"),
                    "season": self._infer_season(commence_time),
                    "week": None,
                    "start_time": commence_time,
                    "venue": None,
                    "city": None,
                    "state": None,
                    "country": None,
                    "surface": None,
                    "day_of_week": commence_time.strftime("%A") if commence_time else None,
                    "referee": None,
                    "temperature_f": None,
                    "weather_conditions": None,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": None,
                    "away_score": None,
                    "status": "upcoming",
                    "home_moneyline": home_price,
                    "away_moneyline": away_price,
                    "home_implied_prob": american_to_prob(home_price),
                    "away_implied_prob": american_to_prob(away_price),
                    "odds_updated": last_update,
                }
            )

        self.db.upsert_rows(
            self.db.games,
            odds_rows,
            ["game_id"],
            update_columns=[
                "start_time",
                "home_moneyline",
                "away_moneyline",
                "home_implied_prob",
                "away_implied_prob",
                "odds_updated",
            ],
        )

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_temperature_fahrenheit(value: Any) -> Optional[float]:
        """Normalize temperature payloads into a Fahrenheit float."""

        if value is None:
            return None

        if isinstance(value, dict):
            candidates = [
                value.get("fahrenheit"),
                value.get("F"),
                value.get("tempF"),
                value.get("value"),
            ]
            for candidate in candidates:
                result = NFLIngestor._safe_float(candidate)
                if result is not None:
                    return result
            return None

        if isinstance(value, (list, tuple)):
            for item in value:
                result = NFLIngestor._extract_temperature_fahrenheit(item)
                if result is not None:
                    return result
            return None

        return NFLIngestor._safe_float(value)

    @staticmethod
    def _infer_season(start_time: Optional[dt.datetime]) -> Optional[str]:
        if not start_time:
            return None
        year = start_time.year
        if start_time.month < 3:
            year -= 1
        return f"{year}-regular"

    def _fetch_boxscore_player_stats(self, season: str, game_id: str) -> List[Dict[str, Any]]:
        """Fallback to boxscore endpoint when detailed gamelogs are unavailable."""

        try:
            boxscore = self.msf_client.fetch_game_boxscore(season, game_id)
        except HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                logging.debug(
                    "Boxscore not available for season %s game %s (HTTP 404)",
                    season,
                    game_id,
                )
                return []
            logging.debug(
                "Failed to fetch boxscore for season %s game %s: %s",
                season,
                game_id,
                exc,
            )
            return []

        game_info = boxscore.get("game", {}) or {}
        team_lookup = {
            "home": (game_info.get("homeTeam") or {}).get("abbreviation"),
            "away": (game_info.get("awayTeam") or {}).get("abbreviation"),
        }

        stats_root = boxscore.get("stats") or {}
        normalized: List[Dict[str, Any]] = []
        for side in ("home", "away"):
            side_payload = stats_root.get(side) or {}
            players = side_payload.get("players") or []
            team_abbr = team_lookup.get(side)
            for player_entry in players:
                if not isinstance(player_entry, dict):
                    continue
                player_stats = self._normalize_boxscore_stat_groups(player_entry.get("playerStats"))
                normalized.append(
                    {
                        "player": player_entry.get("player", {}) or {},
                        "team": {"abbreviation": team_abbr} if team_abbr else {},
                        "stats": player_stats,
                    }
                )

        return normalized

    @staticmethod
    def _normalize_boxscore_stat_groups(raw_groups: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Convert boxscore player stat groups to the gamelog-style schema."""

        if not isinstance(raw_groups, list):
            return {}

        normalized: Dict[str, Dict[str, Dict[str, Any]]] = {}

        def assign(group: str, target: str, value: Any) -> None:
            if value in (None, ""):
                return
            normalized.setdefault(group, {})[target] = {"value": value}

        for group_entry in raw_groups:
            if not isinstance(group_entry, dict):
                continue
            for group_name, metrics in group_entry.items():
                if not isinstance(metrics, dict):
                    continue
                key = group_name.lower()
                if key == "rushing":
                    assign("Rushing", "RushingAttempts", metrics.get("rushAttempts"))
                    assign("Rushing", "RushingYards", metrics.get("rushYards"))
                    assign("Rushing", "RushingTD", metrics.get("rushTD"))
                elif key == "receiving":
                    assign("Receiving", "Targets", metrics.get("targets"))
                    assign("Receiving", "Receptions", metrics.get("receptions"))
                    assign("Receiving", "ReceivingYards", metrics.get("recYards"))
                    assign("Receiving", "ReceivingTD", metrics.get("recTD"))
                elif key == "passing":
                    assign("Passing", "PassAttempts", metrics.get("passAttempts"))
                    assign("Passing", "PassCompletions", metrics.get("passCompletions"))
                    assign("Passing", "PassYards", metrics.get("passYards"))
                    assign("Passing", "PassTD", metrics.get("passTD"))
                elif key == "fumbles":
                    assign("Fumbles", "Fumbles", metrics.get("fumbles"))
                elif key == "snapcounts":
                    offense_snaps = metrics.get("offenseSnaps")
                    if offense_snaps is not None:
                        assign("Miscellaneous", "Snaps", offense_snaps)

        return normalized

    @staticmethod
    def _extract_score_totals(score_payload: Any) -> Tuple[Optional[float], Optional[float]]:
        """Extract final home and away scores from the flexible MSF schedule payload."""

        if not isinstance(score_payload, dict):
            return None, None

        def first_numeric(mapping: Dict[str, Any], candidates: Tuple[str, ...]) -> Optional[float]:
            for key in candidates:
                if key not in mapping or mapping[key] in (None, ""):
                    continue
                value = mapping[key]
                if isinstance(value, dict):
                    for inner_key in ("#text", "value", "total", "score", "amount"):
                        inner_val = value.get(inner_key)
                        parsed = NFLIngestor._safe_float(inner_val)
                        if parsed is not None:
                            return parsed
                    parsed = NFLIngestor._safe_float(value)
                    if parsed is not None:
                        return parsed
                else:
                    parsed = NFLIngestor._safe_float(value)
                    if parsed is not None:
                        return parsed
            return None

        home_candidates = (
            "homeScore",
            "homeScoreTotal",
            "homeScoreFinal",
            "homePoints",
            "homeScoreValue",
        )
        away_candidates = (
            "awayScore",
            "awayScoreTotal",
            "awayScoreFinal",
            "awayPoints",
            "awayScoreValue",
        )

        return first_numeric(score_payload, home_candidates), first_numeric(score_payload, away_candidates)


# ---------------------------------------------------------------------------
# Feature engineering & modeling
# ---------------------------------------------------------------------------


class FeatureBuilder:
    """Transforms raw database tables into model-ready feature sets."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def load_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        games = pd.read_sql_table("nfl_games", self.engine)
        player_stats = pd.read_sql_table("nfl_player_stats", self.engine)
        team_ratings = pd.read_sql_table("nfl_team_unit_ratings", self.engine)

        # Some SQLAlchemy engines (notably PostgreSQL) return column names as
        # ``quoted_name`` objects, which scikit-learn refuses to accept when
        # validating feature names. Normalize them to plain strings up-front so
        # downstream selectors and pipelines see consistent column labels.
        games = games.rename(columns=lambda col: str(col))
        player_stats = player_stats.rename(columns=lambda col: str(col))
        team_ratings = team_ratings.rename(columns=lambda col: str(col))
        return games, player_stats, team_ratings

    def build_features(self) -> Dict[str, pd.DataFrame]:
        games, player_stats, _ = self.load_dataframes()

        if games.empty:
            logging.warning("No games available in the database. Skipping model training.")
            return {}

        games = games.copy()
        player_stats = player_stats.copy()

        # Basic cleanup
        games["start_time"] = pd.to_datetime(games["start_time"])
        games["day_of_week"] = games["day_of_week"].fillna(
            games["start_time"].dt.day_name()
        )
        games["game_result"] = np.where(
            games["home_score"] > games["away_score"], "home",
            np.where(games["home_score"] < games["away_score"], "away", "push"),
        )

        # Derive rolling scoring, rest, and win-rate indicators from historical games.
        team_game_history = self._compute_team_game_rolling_stats(games)

        datasets: Dict[str, pd.DataFrame] = {}
        team_strength: pd.DataFrame

        if player_stats.empty:
            logging.warning(
                "Player statistics table is empty. Player-level models will not be trained."
            )
            team_strength = self._compute_team_unit_strength(player_stats)
        else:
            enrichment_columns = [
                "game_id",
                "season",
                "week",
                "start_time",
                "venue",
                "city",
                "state",
                "day_of_week",
                "referee",
                "weather_conditions",
                "temperature_f",
                "home_team",
                "away_team",
            ]
            player_stats = player_stats.merge(
                games[enrichment_columns],
                on="game_id",
                how="left",
            )

            team_strength = self._compute_team_unit_strength(player_stats)

            player_stats = player_stats.merge(
                team_strength,
                on=["team", "season", "week"],
                how="left",
                suffixes=("", "_team"),
            )

            opponent_strength = team_strength.rename(
                columns={
                    "team": "opponent",
                    "offense_pass_rating": "opp_offense_pass_rating",
                    "offense_rush_rating": "opp_offense_rush_rating",
                    "defense_pass_rating": "opp_defense_pass_rating",
                    "defense_rush_rating": "opp_defense_rush_rating",
                }
            )

            player_stats = player_stats.merge(
                games[["game_id", "home_team", "away_team"]],
                on="game_id",
                how="left",
                suffixes=("", "_game"),
            )

            player_stats["opponent"] = np.where(
                player_stats["team"] == player_stats["home_team"],
                player_stats["away_team"],
                player_stats["home_team"],
            )

            player_stats = player_stats.merge(
                opponent_strength,
                on=["opponent", "season", "week"],
                how="left",
            )

            context_features = self._compute_contextual_averages(player_stats)
            player_stats = player_stats.merge(
                context_features,
                on=["team", "venue", "day_of_week", "referee"],
                how="left",
            )

            def add_dataset(target: str, positions: Iterable[str]) -> None:
                subset = player_stats[player_stats["position"].isin(list(positions))].copy()
                subset = subset[subset[target].notna()]
                if subset.empty:
                    logging.debug(
                        "Skipping %s dataset because no rows remained after filtering", target
                    )
                    return
                datasets[target] = subset

            add_dataset("rushing_yards", ["RB", "HB", "FB", "QB"])
            add_dataset("receiving_yards", ["WR", "RB", "HB", "FB", "TE"])
            add_dataset("receptions", ["WR", "RB", "HB", "FB", "TE"])
            add_dataset("rushing_tds", ["RB", "HB", "FB", "QB"])
            add_dataset("receiving_tds", ["WR", "RB", "TE"])
            add_dataset("passing_tds", ["QB"])

        home_strength = team_strength.rename(
            columns={
                "team": "home_team",
                "offense_pass_rating": "home_offense_pass_rating",
                "offense_rush_rating": "home_offense_rush_rating",
                "defense_pass_rating": "home_defense_pass_rating",
                "defense_rush_rating": "home_defense_rush_rating",
            }
        )
        away_strength = team_strength.rename(
            columns={
                "team": "away_team",
                "offense_pass_rating": "away_offense_pass_rating",
                "offense_rush_rating": "away_offense_rush_rating",
                "defense_pass_rating": "away_defense_pass_rating",
                "defense_rush_rating": "away_defense_rush_rating",
            }
        )

        home_history = team_game_history[team_game_history["is_home"]].drop(
            columns=["team", "is_home"]
        )
        home_history = home_history.rename(
            columns={
                "game_id": "game_id",
                "rolling_points_for": "home_points_for_avg",
                "rolling_points_against": "home_points_against_avg",
                "rolling_point_diff": "home_point_diff_avg",
                "rolling_win_pct": "home_win_pct_recent",
                "prev_points_for": "home_prev_points_for",
                "prev_points_against": "home_prev_points_against",
                "prev_point_diff": "home_prev_point_diff",
                "rest_days": "home_rest_days",
            }
        )

        away_history = team_game_history[~team_game_history["is_home"]].drop(
            columns=["team", "is_home"]
        )
        away_history = away_history.rename(
            columns={
                "game_id": "game_id",
                "rolling_points_for": "away_points_for_avg",
                "rolling_points_against": "away_points_against_avg",
                "rolling_point_diff": "away_point_diff_avg",
                "rolling_win_pct": "away_win_pct_recent",
                "prev_points_for": "away_prev_points_for",
                "prev_points_against": "away_prev_points_against",
                "prev_point_diff": "away_prev_point_diff",
                "rest_days": "away_rest_days",
            }
        )

        games_context = (
            games.merge(
                home_strength,
                on=["home_team", "season", "week"],
                how="left",
            )
            .merge(
                away_strength,
                on=["away_team", "season", "week"],
                how="left",
            )
            .merge(home_history, on="game_id", how="left")
            .merge(away_history, on="game_id", how="left")
        )

        games_context["moneyline_diff"] = games_context["home_moneyline"] - games_context["away_moneyline"]
        games_context["implied_prob_diff"] = (
            games_context["home_implied_prob"] - games_context["away_implied_prob"]
        )
        games_context["implied_prob_sum"] = (
            games_context["home_implied_prob"] + games_context["away_implied_prob"]
        )

        games_context["point_diff"] = games_context["home_score"] - games_context["away_score"]
        games_labeled = games_context.dropna(subset=["home_score", "away_score"])
        if games_labeled.empty:
            logging.warning(
                "No completed games with scores available. Game outcome model will be skipped."
            )
        else:
            datasets["game_outcome"] = games_labeled

        return datasets

    def _compute_team_unit_strength(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        if player_stats.empty:
            return pd.DataFrame(
                columns=
                [
                    "season",
                    "week",
                    "team",
                    "offense_pass_rating",
                    "offense_rush_rating",
                    "defense_pass_rating",
                    "defense_rush_rating",
                ]
            )

        grouped = (
            player_stats.groupby(["season", "week", "team"])
            .agg(
                rush_yards=pd.NamedAgg(column="rushing_yards", aggfunc="sum"),
                rush_tds=pd.NamedAgg(column="rushing_tds", aggfunc="sum"),
                rec_yards=pd.NamedAgg(column="receiving_yards", aggfunc="sum"),
                rec_tds=pd.NamedAgg(column="receiving_tds", aggfunc="sum"),
                pass_yards=pd.NamedAgg(column="passing_yards", aggfunc="sum"),
                pass_tds=pd.NamedAgg(column="passing_tds", aggfunc="sum"),
            )
            .reset_index()
        )

        grouped = grouped.sort_values(["team", "season", "week"]).reset_index(drop=True)
        for col in ["rush_yards", "rush_tds", "rec_yards", "rec_tds", "pass_yards", "pass_tds"]:
            grouped[f"rolling_{col}"] = (
                grouped.groupby(["team", "season"])[col]
                .rolling(window=4, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )

        # Offense: use rushing and passing production. Defense approximated by opponent restriction.
        grouped["offense_rush_rating"] = grouped[["rolling_rush_yards", "rolling_rush_tds"]].mean(axis=1)
        grouped["offense_pass_rating"] = grouped[["rolling_pass_yards", "rolling_pass_tds", "rolling_rec_yards", "rolling_rec_tds"]].mean(axis=1)

        # Defense derived by comparing to league averages (placeholder). In practice, integrate opponent stats.
        grouped["defense_rush_rating"] = grouped.groupby(["season", "week"])["offense_rush_rating"].transform("mean") - grouped["offense_rush_rating"]
        grouped["defense_pass_rating"] = grouped.groupby(["season", "week"])["offense_pass_rating"].transform("mean") - grouped["offense_pass_rating"]

        cols = ["season", "week", "team", "offense_pass_rating", "offense_rush_rating", "defense_pass_rating", "defense_rush_rating"]

        return grouped[cols]

    def _compute_contextual_averages(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        if player_stats.empty:
            return pd.DataFrame(
                columns=
                [
                    "team",
                    "venue",
                    "day_of_week",
                    "referee",
                    "avg_rush_yards",
                    "avg_rec_yards",
                    "avg_receptions",
                    "avg_rush_tds",
                    "avg_rec_tds",
                ]
            )

        context = (
            player_stats.groupby(["team", "venue", "day_of_week", "referee"])
            .agg(
                avg_rush_yards=pd.NamedAgg(column="rushing_yards", aggfunc="mean"),
                avg_rec_yards=pd.NamedAgg(column="receiving_yards", aggfunc="mean"),
                avg_receptions=pd.NamedAgg(column="receptions", aggfunc="mean"),
                avg_rush_tds=pd.NamedAgg(column="rushing_tds", aggfunc="mean"),
                avg_rec_tds=pd.NamedAgg(column="receiving_tds", aggfunc="mean"),
            )
            .reset_index()
        )
        return context

    def _compute_team_game_rolling_stats(self, games: pd.DataFrame) -> pd.DataFrame:
        """Create rolling scoring, win-rate, and rest indicators for each team game."""

        if games.empty:
            return pd.DataFrame(
                columns=[
                    "game_id",
                    "team",
                    "is_home",
                    "rolling_points_for",
                    "rolling_points_against",
                    "rolling_point_diff",
                    "rolling_win_pct",
                    "prev_points_for",
                    "prev_points_against",
                    "prev_point_diff",
                    "rest_days",
                ]
            )

        games = games.copy()
        games["start_time"] = pd.to_datetime(games["start_time"])

        home = games[[
            "game_id",
            "season",
            "week",
            "start_time",
            "home_team",
            "home_score",
            "away_score",
        ]].rename(
            columns={
                "home_team": "team",
                "home_score": "points_for",
                "away_score": "points_against",
            }
        )
        home["is_home"] = True

        away = games[[
            "game_id",
            "season",
            "week",
            "start_time",
            "away_team",
            "away_score",
            "home_score",
        ]].rename(
            columns={
                "away_team": "team",
                "away_score": "points_for",
                "home_score": "points_against",
            }
        )
        away["is_home"] = False

        team_games = pd.concat([home, away], ignore_index=True)
        team_games = team_games.dropna(subset=["team"])  # handle null abbreviations

        team_games = team_games.sort_values([
            "team",
            "season",
            "start_time",
            "game_id",
        ]).reset_index(drop=True)

        def compute_group(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("start_time").copy()
            win_flag = np.where(
                group["points_for"].notna() & group["points_against"].notna(),
                (group["points_for"] > group["points_against"]).astype(float),
                np.nan,
            )

            group["prev_points_for"] = group["points_for"].shift(1)
            group["prev_points_against"] = group["points_against"].shift(1)
            group["prev_point_diff"] = (
                group["prev_points_for"] - group["prev_points_against"]
            )

            rolling_points_for = (
                group["points_for"].rolling(window=5, min_periods=1).mean()
            )
            rolling_points_against = (
                group["points_against"].rolling(window=5, min_periods=1).mean()
            )
            rolling_point_diff = (
                (group["points_for"] - group["points_against"]).rolling(window=5, min_periods=1).mean()
            )
            rolling_win_pct = (
                pd.Series(win_flag, index=group.index)
                .rolling(window=5, min_periods=1)
                .mean()
            )

            group["rolling_points_for"] = rolling_points_for.shift(1)
            group["rolling_points_against"] = rolling_points_against.shift(1)
            group["rolling_point_diff"] = rolling_point_diff.shift(1)
            group["rolling_win_pct"] = rolling_win_pct.shift(1)

            rest_days = group["start_time"].diff().dt.total_seconds() / 86400.0
            group["rest_days"] = rest_days

            return group

        grouped_frames: List[pd.DataFrame] = []
        for _, group in team_games.groupby(["team", "season"], sort=False):
            grouped_frames.append(compute_group(group))

        if grouped_frames:
            team_games = pd.concat(grouped_frames, ignore_index=True)
        else:
            team_games = team_games.iloc[0:0]

        return team_games[
            [
                "game_id",
                "team",
                "is_home",
                "rolling_points_for",
                "rolling_points_against",
                "rolling_point_diff",
                "rolling_win_pct",
                "prev_points_for",
                "prev_points_against",
                "prev_point_diff",
                "rest_days",
            ]
        ]


# ---------------------------------------------------------------------------
# Modeling pipeline
# ---------------------------------------------------------------------------


class ModelTrainer:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.feature_builder = FeatureBuilder(engine)

    # ------------------------------------------------------------------
    # Chronological splitting utilities
    # ------------------------------------------------------------------

    def _sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        if "start_time" in df.columns:
            return df.sort_values("start_time")
        if {"season", "week"}.issubset(df.columns):
            return df.sort_values(["season", "week"])
        if "week" in df.columns:
            return df.sort_values("week")
        return df.sort_index()

    def _chronological_split(
        self,
        df: pd.DataFrame,
        holdout_fraction: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_sorted = self._sort_by_time(df).reset_index(drop=True)
        if len(df_sorted) < 5:
            split_index = max(1, len(df_sorted) - 1)
        else:
            holdout_size = max(1, int(len(df_sorted) * holdout_fraction))
            if holdout_size >= len(df_sorted):
                holdout_size = max(1, len(df_sorted) - 1)
            split_index = len(df_sorted) - holdout_size

        if split_index <= 0 or split_index >= len(df_sorted):
            split_index = max(1, len(df_sorted) - 1)

        train_df = df_sorted.iloc[:split_index]
        test_df = df_sorted.iloc[split_index:]
        return train_df, test_df, df_sorted

    def _build_time_series_cv(self, n_samples: int) -> TimeSeriesSplit:
        if n_samples < 3:
            raise ValueError("At least 3 samples are required for time series CV.")

        n_splits = min(5, max(2, n_samples - 1))
        if n_splits >= n_samples:
            n_splits = n_samples - 1
        return TimeSeriesSplit(n_splits=n_splits)

    def train(self) -> Dict[str, Pipeline]:
        datasets = self.feature_builder.build_features()
        models: Dict[str, Pipeline] = {}

        for target, df in datasets.items():
            if target == "game_outcome":
                model = self._train_game_models(df)
                models.update(model)
                continue

            model = self._train_regression_model(df, target)
            if model is not None:
                models[target] = model
        return models

    def _train_regression_model(self, df: pd.DataFrame, target: str) -> Optional[Pipeline]:
        if len(df) < 20 or df[target].nunique() <= 1:
            logging.warning(
                "Not enough data to train %s model (rows=%d, unique targets=%d).", 
                target,
                len(df),
                df[target].nunique(),
            )
            return None

        numeric_features = [
            "week",
            "temperature_f",
            "offense_pass_rating",
            "offense_rush_rating",
            "defense_pass_rating",
            "defense_rush_rating",
            "opp_offense_pass_rating",
            "opp_offense_rush_rating",
            "opp_defense_pass_rating",
            "opp_defense_rush_rating",
            "avg_rush_yards",
            "avg_rec_yards",
            "avg_receptions",
            "avg_rush_tds",
            "avg_rec_tds",
            "snap_count",
            "receiving_targets",
        ]
        categorical_features = [
            "team",
            "opponent",
            "venue",
            "day_of_week",
            "referee",
            "position",
        ]

        available_numeric = [
            col for col in numeric_features if col in df.columns and df[col].notna().any()
        ]
        dropped_numeric = sorted(set(numeric_features) - set(available_numeric))
        if dropped_numeric:
            logging.debug(
                "Dropping numeric features with no observed values for %s model: %s",
                target,
                ", ".join(dropped_numeric),
            )

        available_categorical = [
            col for col in categorical_features if col in df.columns and df[col].notna().any()
        ]
        dropped_categorical = sorted(set(categorical_features) - set(available_categorical))
        if dropped_categorical:
            logging.debug(
                "Dropping categorical features with no observed values for %s model: %s",
                target,
                ", ".join(dropped_categorical),
            )

        if not available_numeric and not available_categorical:
            logging.warning(
                "No usable features with observed values available to train %s model; skipping.",
                target,
            )
            return None

        feature_columns = available_numeric + available_categorical

        train_df, test_df, sorted_df = self._chronological_split(df)
        X_train = train_df[feature_columns]
        y_train = train_df[target]
        X_test = test_df[feature_columns]
        y_test = test_df[target]

        transformers = []
        if available_numeric:
            transformers.append(
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]),
                    available_numeric,
                )
            )
        if available_categorical:
            transformers.append(
                (
                    "cat",
                    Pipeline([
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    available_categorical,
                )
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(random_state=42)),
        ])

        model.fit(X_train, y_train)
        baseline_pred = model.predict(X_test)
        baseline_r2 = model.score(X_test, y_test)
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
        logging.info(
            "Trained %s model (baseline GBM), R^2=%.3f on holdout (MAE=%.3f, RMSE=%.3f)",
            target,
            baseline_r2,
            baseline_mae,
            baseline_rmse,
        )

        try:
            cv = self._build_time_series_cv(len(X_train))
        except ValueError as exc:
            logging.warning(
                "Skipping hyperparameter tuning for %s due to insufficient data: %s",
                target,
                exc,
            )
            best_model = model
        else:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=self._gb_param_grid("regressor__"),
                n_iter=10,
                scoring="neg_mean_absolute_error",
                cv=cv,
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_model: Pipeline = search.best_estimator_
            logging.info(
                "Best parameters for %s model: %s (CV MAE=%.3f)",
                target,
                search.best_params_,
                -search.best_score_,
            )

        y_pred = best_model.predict(X_test)
        r2 = best_model.score(X_test, y_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        logging.info(
            "%s holdout metrics | R^2=%.3f | MAE=%.3f | RMSE=%.3f",
            target,
            r2,
            mae,
            rmse,
        )

        best_model.fit(sorted_df[feature_columns], sorted_df[target])
        return best_model

    def _train_game_models(self, df: pd.DataFrame) -> Dict[str, Pipeline]:
        if len(df) < 20 or df["game_result"].nunique() <= 1:
            logging.warning(
                "Not enough completed games (%d) with outcomes to train game-level models.",
                len(df),
            )
            return {}

        numeric_features = [
            "week",
            "temperature_f",
            "home_moneyline",
            "away_moneyline",
            "home_implied_prob",
            "away_implied_prob",
            "moneyline_diff",
            "implied_prob_diff",
            "implied_prob_sum",
            "home_offense_pass_rating",
            "home_offense_rush_rating",
            "home_defense_pass_rating",
            "home_defense_rush_rating",
            "away_offense_pass_rating",
            "away_offense_rush_rating",
            "away_defense_pass_rating",
            "away_defense_rush_rating",
            "home_points_for_avg",
            "home_points_against_avg",
            "home_point_diff_avg",
            "home_win_pct_recent",
            "home_prev_points_for",
            "home_prev_points_against",
            "home_prev_point_diff",
            "home_rest_days",
            "away_points_for_avg",
            "away_points_against_avg",
            "away_point_diff_avg",
            "away_win_pct_recent",
            "away_prev_points_for",
            "away_prev_points_against",
            "away_prev_point_diff",
            "away_rest_days",
        ]
        categorical_features = ["venue", "day_of_week", "referee", "home_team", "away_team"]

        available_numeric = [
            col for col in numeric_features if col in df.columns and df[col].notna().any()
        ]
        dropped_numeric = sorted(set(numeric_features) - set(available_numeric))
        if dropped_numeric:
            logging.debug(
                "Dropping numeric game features with no observed values: %s",
                ", ".join(dropped_numeric),
            )

        available_categorical = [
            col for col in categorical_features if col in df.columns and df[col].notna().any()
        ]
        dropped_categorical = sorted(set(categorical_features) - set(available_categorical))
        if dropped_categorical:
            logging.debug(
                "Dropping categorical game features with no observed values: %s",
                ", ".join(dropped_categorical),
            )

        if not available_numeric and not available_categorical:
            logging.warning(
                "No usable features with observed values available to train game-level models.",
            )
            return {}

        feature_columns = available_numeric + available_categorical

        train_df, test_df, sorted_df = self._chronological_split(df)
        X_train = train_df[feature_columns]
        X_test = test_df[feature_columns]

        y_winner_train = (train_df["game_result"] == "home").astype(int)
        y_winner_test = (test_df["game_result"] == "home").astype(int)

        y_home_train = train_df["home_score"]
        y_home_test = test_df["home_score"]

        y_away_train = train_df["away_score"]
        y_away_test = test_df["away_score"]

        transformers = []
        if available_numeric:
            transformers.append(
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]),
                    available_numeric,
                )
            )
        if available_categorical:
            transformers.append(
                (
                    "cat",
                    Pipeline([
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    available_categorical,
                )
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        clf = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(random_state=42)),
        ])
        reg_home = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(random_state=42)),
        ])
        reg_away = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(random_state=42)),
        ])

        clf.fit(X_train, y_winner_train)
        reg_home.fit(X_train, y_home_train)
        reg_away.fit(X_train, y_away_train)

        baseline_winner_acc = clf.score(X_test, y_winner_test)
        baseline_home_r2 = reg_home.score(X_test, y_home_test)
        baseline_away_r2 = reg_away.score(X_test, y_away_test)
        logging.info(
            "Trained game outcome classifier (baseline), accuracy=%.3f",
            baseline_winner_acc,
        )
        logging.info(
            "Trained home score regressor (baseline), R^2=%.3f",
            baseline_home_r2,
        )
        logging.info(
            "Trained away score regressor (baseline), R^2=%.3f",
            baseline_away_r2,
        )

        best_clf = clf
        best_reg_home = reg_home
        best_reg_away = reg_away

        try:
            cv = self._build_time_series_cv(len(X_train))
        except ValueError as exc:
            logging.warning(
                "Skipping hyperparameter tuning for game models due to insufficient data: %s",
                exc,
            )
        else:
            clf_search = RandomizedSearchCV(
                estimator=clf,
                param_distributions=self._gb_param_grid("classifier__"),
                n_iter=10,
                scoring="roc_auc",
                cv=cv,
                random_state=42,
                n_jobs=-1,
            )
            clf_search.fit(X_train, y_winner_train)
            best_clf: Pipeline = clf_search.best_estimator_
            logging.info(
                "Best parameters for game winner model: %s (CV ROC-AUC=%.3f)",
                clf_search.best_params_,
                clf_search.best_score_,
            )

            reg_home_search = RandomizedSearchCV(
                estimator=reg_home,
                param_distributions=self._gb_param_grid("regressor__"),
                n_iter=10,
                scoring="neg_mean_absolute_error",
                cv=cv,
                random_state=42,
                n_jobs=-1,
            )
            reg_home_search.fit(X_train, y_home_train)
            best_reg_home: Pipeline = reg_home_search.best_estimator_
            logging.info(
                "Best parameters for home score model: %s (CV MAE=%.3f)",
                reg_home_search.best_params_,
                -reg_home_search.best_score_,
            )

            reg_away_search = RandomizedSearchCV(
                estimator=reg_away,
                param_distributions=self._gb_param_grid("regressor__"),
                n_iter=10,
                scoring="neg_mean_absolute_error",
                cv=cv,
                random_state=42,
                n_jobs=-1,
            )
            reg_away_search.fit(X_train, y_away_train)
            best_reg_away: Pipeline = reg_away_search.best_estimator_
            logging.info(
                "Best parameters for away score model: %s (CV MAE=%.3f)",
                reg_away_search.best_params_,
                -reg_away_search.best_score_,
            )

        winner_pred = best_clf.predict(X_test)
        winner_proba = best_clf.predict_proba(X_test)[:, 1]
        winner_accuracy = accuracy_score(y_winner_test, winner_pred)
        try:
            winner_roc_auc = (
                roc_auc_score(y_winner_test, winner_proba)
                if len(np.unique(y_winner_test)) > 1
                else float("nan")
            )
        except ValueError:
            winner_roc_auc = float("nan")
        try:
            winner_log_loss = log_loss(y_winner_test, winner_proba, labels=[0, 1])
        except ValueError:
            winner_log_loss = float("nan")
        logging.info(
            "Game winner holdout metrics | accuracy=%.3f | ROC-AUC=%s | log_loss=%s",
            winner_accuracy,
            f"{winner_roc_auc:.3f}" if not np.isnan(winner_roc_auc) else "nan",
            f"{winner_log_loss:.3f}" if not np.isnan(winner_log_loss) else "nan",
        )

        home_pred = best_reg_home.predict(X_test)
        home_r2 = best_reg_home.score(X_test, y_home_test)
        home_mae = mean_absolute_error(y_home_test, home_pred)
        home_rmse = float(np.sqrt(mean_squared_error(y_home_test, home_pred)))
        logging.info(
            "Home score holdout metrics | R^2=%.3f | MAE=%.3f | RMSE=%.3f",
            home_r2,
            home_mae,
            home_rmse,
        )

        away_pred = best_reg_away.predict(X_test)
        away_r2 = best_reg_away.score(X_test, y_away_test)
        away_mae = mean_absolute_error(y_away_test, away_pred)
        away_rmse = float(np.sqrt(mean_squared_error(y_away_test, away_pred)))
        logging.info(
            "Away score holdout metrics | R^2=%.3f | MAE=%.3f | RMSE=%.3f",
            away_r2,
            away_mae,
            away_rmse,
        )

        X_full = sorted_df[feature_columns]
        best_clf.fit(X_full, (sorted_df["game_result"] == "home").astype(int))
        best_reg_home.fit(X_full, sorted_df["home_score"])
        best_reg_away.fit(X_full, sorted_df["away_score"])

        return {
            "game_winner": best_clf,
            "home_points": best_reg_home,
            "away_points": best_reg_away,
        }


# ---------------------------------------------------------------------------
# Prediction utilities
# ---------------------------------------------------------------------------


def predict_upcoming_games(models: Dict[str, Pipeline], engine: Engine, output_path: Path) -> pd.DataFrame:
    games = pd.read_sql_table("nfl_games", engine)
    upcoming = games[(games["status"].isin(["upcoming", "scheduled"])) | games["home_score"].isna()]
    if upcoming.empty:
        logging.warning("No upcoming games found for prediction")
        return pd.DataFrame()

    feature_builder = FeatureBuilder(engine)
    datasets = feature_builder.build_features()
    # Use latest features for upcoming games by merging with context features
    games_context = datasets["game_outcome"]
    future_games = upcoming.merge(
        games_context,
        on="game_id",
        how="left",
        suffixes=("", "_hist"),
    )

    # Predictions
    predictions: List[Dict[str, Any]] = []
    for _, row in future_games.iterrows():
        prediction_id_base = f"{row['game_id']}"
        features = row[[
            "week",
            "temperature_f",
            "home_moneyline",
            "away_moneyline",
            "home_implied_prob",
            "away_implied_prob",
            "moneyline_diff",
            "implied_prob_diff",
            "implied_prob_sum",
            "home_offense_pass_rating",
            "home_offense_rush_rating",
            "home_defense_pass_rating",
            "home_defense_rush_rating",
            "away_offense_pass_rating",
            "away_offense_rush_rating",
            "away_defense_pass_rating",
            "away_defense_rush_rating",
            "home_points_for_avg",
            "home_points_against_avg",
            "home_point_diff_avg",
            "home_win_pct_recent",
            "home_prev_points_for",
            "home_prev_points_against",
            "home_prev_point_diff",
            "home_rest_days",
            "away_points_for_avg",
            "away_points_against_avg",
            "away_point_diff_avg",
            "away_win_pct_recent",
            "away_prev_points_for",
            "away_prev_points_against",
            "away_prev_point_diff",
            "away_rest_days",
            "venue",
            "day_of_week",
            "referee",
            "home_team",
            "away_team",
        ]]

        winner_prob = models["game_winner"].predict_proba(pd.DataFrame([features]))[0, 1]
        home_points = models["home_points"].predict(pd.DataFrame([features]))[0]
        away_points = models["away_points"].predict(pd.DataFrame([features]))[0]

        predictions.extend(
            [
                {
                    "prediction_id": f"{prediction_id_base}_winner",
                    "game_id": row["game_id"],
                    "entity_type": "game",
                    "entity_id": row["game_id"],
                    "prediction_target": "home_win_probability",
                    "prediction_value": float(winner_prob),
                    "model_version": "gbm_v1",
                    "features": features.to_dict(),
                },
                {
                    "prediction_id": f"{prediction_id_base}_home_pts",
                    "game_id": row["game_id"],
                    "entity_type": "game",
                    "entity_id": row["home_team"],
                    "prediction_target": "projected_points",
                    "prediction_value": float(home_points),
                    "model_version": "gbm_v1",
                    "features": features.to_dict(),
                },
                {
                    "prediction_id": f"{prediction_id_base}_away_pts",
                    "game_id": row["game_id"],
                    "entity_type": "game",
                    "entity_id": row["away_team"],
                    "prediction_target": "projected_points",
                    "prediction_value": float(away_points),
                    "model_version": "gbm_v1",
                    "features": features.to_dict(),
                },
            ]
        )

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_json(output_path, orient="records", indent=2)
    logging.info("Saved %d game predictions to %s", len(predictions_df), output_path)
    return predictions_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFL betting analytics pipeline")
    parser.add_argument("--config", type=str, help="Optional path to JSON config file")
    parser.add_argument("--predict", action="store_true", help="Generate predictions for upcoming games")
    parser.add_argument("--output", type=Path, default=Path("predictions.json"), help="Where to save predictions")
    return parser.parse_args()


def load_config(path: Optional[str]) -> NFLConfig:
    config = NFLConfig()
    if not path:
        return config

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for field in dataclasses.fields(config):
        if field.name in payload:
            setattr(config, field.name, payload[field.name])
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.log_level)

    logging.info("Connecting to PostgreSQL at %s", config.pg_url)
    engine = create_engine(config.pg_url, future=True)
    db = NFLDatabase(engine)

    msf_client = MySportsFeedsClient(NFL_API_USER, NFL_API_PASS)
    odds_client = OddsApiClient(ODDS_API_KEY)

    ingestor = NFLIngestor(db, msf_client, odds_client)
    ingestor.ingest(config.seasons)

    trainer = ModelTrainer(engine)
    try:
        models = trainer.train()
    except RuntimeError as exc:
        logging.error("Unable to train models: %s", exc)
        logging.error(
            "Model training requires historical games and player statistics. "
            "Ensure ingestion succeeded (check API credentials, plan access, and season settings) before rerunning."
        )
        if args.predict:
            logging.error("Prediction generation skipped because models were not trained.")
        return

    if not models:
        logging.warning(
            "No models were trained. Verify that sufficient labeled data exists in the database before requesting predictions."
        )
        if args.predict:
            logging.error("Prediction generation skipped because no models were available.")
        return

    if args.predict:
        predict_upcoming_games(models, engine, args.output)


if __name__ == "__main__":
    main()
