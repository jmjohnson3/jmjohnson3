## Usage

```bash
python ingest.py 2024-regular --username YOUR_USER --password YOUR_PASS
```

The CLI now enumerates completed weeks automatically. By default the tool
discovers every week with games (`--all-weeks` is enabled) and uses that list
for both game and player gamelog ingestion. The upcoming week value is
calculated as the next week after the final completed one.

To ingest a narrow range instead, provide an explicit list or range with
`--weeks` (for example `--weeks 3-5,8`). When you pass `--weeks`, you can keep
`--all-weeks` enabled (the default) to backfill missing weeks automatically, or
disable it with `--no-all-weeks` to ingest only the explicit list. Override the
computed upcoming week with `--upcoming-week WEEK_NUMBER` when needed.

## NFL betting pipeline

The repository also includes an end-to-end script that downloads NFL data from
MySportsFeeds, stores it in PostgreSQL, enriches it with live odds from The
Odds API, and trains lightweight models to surface potential betting edges.

```bash
python nfl_betting_pipeline.py 2024-regular 1 2 3 \
  --msf-username YOUR_USER --msf-password YOUR_PASS \
  --database-uri postgresql://USER:PASS@HOST:PORT/DB_NAME \
  --odds-api-key YOUR_ODDS_API_KEY
```

The script creates the required tables on first run (`games`, `player_stats`,
and `odds`), so the target database only needs to exist ahead of time. API
credentials can be supplied via flags or the `MSF_USERNAME`, `MSF_PASSWORD`,
`DATABASE_URL`, and `ODDS_API_KEY` environment variables.

Predictions are printed to stdout with win probabilities, projected points, and
the highest available moneyline prices for each side. The code uses `requests`,
`pandas`, `numpy`, `SQLAlchemy`, and `scikit-learn`; install them into your
environment (for example with `pip install requests pandas numpy SQLAlchemy scikit-learn`).
