"""Utility for fetching MySportsFeeds data with week-based ingestion controls."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import List, Optional, Sequence

import aiohttp
from aiohttp import BasicAuth

API_PREFIX = "https://api.mysportsfeeds.com/v2.1/pull/nfl"
DEFAULT_CONSECUTIVE_MISSES = 2


@dataclass(frozen=True)
class WeekIngestionPlan:
    season: str
    weeks: Sequence[int]
    upcoming_week: int


async def fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    feed: str,
) -> dict:
    async with session.get(url) as response:
        if response.status == 204:
            return {}
        response.raise_for_status()
        payload = await response.json()
        return payload.get(feed, payload)


async def fetch_games(
    session: aiohttp.ClientSession,
    season: str,
    week: int,
) -> List[dict]:
    url = f"{API_PREFIX}/{season}/week/{week}/games.json"
    data = await fetch_json(session, url, "games")
    if isinstance(data, dict):
        return data.get("games", [])
    return data or []


async def fetch_player_gamelogs(
    session: aiohttp.ClientSession,
    season: str,
    week: int,
) -> List[dict]:
    url = f"{API_PREFIX}/{season}/week/{week}/player_gamelogs.json"
    data = await fetch_json(session, url, "playerGamelogs")
    if isinstance(data, dict):
        return data.get("playerGamelogs", [])
    return data or []


async def enumerate_season_weeks(
    season: str,
    session: aiohttp.ClientSession,
    *,
    consecutive_misses: int = DEFAULT_CONSECUTIVE_MISSES,
) -> List[int]:
    discovered: List[int] = []
    misses = 0
    week = 1

    while misses < consecutive_misses:
        games = await fetch_games(session, season, week)
        if games:
            discovered.append(week)
            misses = 0
        else:
            misses += 1
        week += 1

    return discovered


def parse_week_list(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    weeks: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            weeks.extend(range(int(start), int(end) + 1))
        else:
            weeks.append(int(chunk))
    return sorted(set(weeks))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest weekly game and player gamelog data from MySportsFeeds. "
            "By default all completed weeks are fetched automatically; "
            "use --weeks to narrow the backfill or --no-all-weeks to reuse a "
            "manually provided list."
        )
    )
    parser.add_argument("season", help="Season string, e.g. 2023-regular")
    parser.add_argument(
        "--weeks",
        metavar="LIST",
        help=(
            "Comma-separated list or range (e.g. 1-3,5) of weeks to ingest. "
            "If omitted the tool will enumerate completed weeks automatically."
        ),
    )
    parser.add_argument(
        "--all-weeks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enumerate all completed weeks automatically (default: enabled). "
            "Disable with --no-all-weeks when providing an explicit --weeks list."
        ),
    )
    parser.add_argument(
        "--upcoming-week",
        type=int,
        help=(
            "Optional override for the upcoming (incomplete) week. By default the "
            "value is set to one greater than the final completed week discovered."
        ),
    )
    parser.add_argument(
        "--username",
        required=True,
        help="MySportsFeeds API username",
    )
    parser.add_argument(
        "--password",
        required=True,
        help="MySportsFeeds API password",
    )
    return parser


async def ingest_weeks(plan: WeekIngestionPlan, session: aiohttp.ClientSession) -> None:
    for week in plan.weeks:
        games = await fetch_games(session, plan.season, week)
        player_logs = await fetch_player_gamelogs(session, plan.season, week)
        print(f"Week {week}: {len(games)} games, {len(player_logs)} player logs")
    print(f"Upcoming week: {plan.upcoming_week}")


def build_plan(args: argparse.Namespace, weeks: Sequence[int]) -> WeekIngestionPlan:
    final_week = max(weeks) if weeks else 0
    upcoming_week = args.upcoming_week or final_week + 1
    return WeekIngestionPlan(season=args.season, weeks=weeks, upcoming_week=upcoming_week)


async def async_main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    explicit_weeks = parse_week_list(args.weeks)
    weeks: Sequence[int]

    auth = BasicAuth(args.username, args.password)
    async with aiohttp.ClientSession(auth=auth) as session:
        if args.all_weeks or not explicit_weeks:
            weeks = await enumerate_season_weeks(args.season, session)
        else:
            weeks = explicit_weeks

        plan = build_plan(args, weeks)
        await ingest_weeks(plan, session)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
