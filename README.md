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
