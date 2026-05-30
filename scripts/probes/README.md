# probes

One-off operator scripts that POST to `/portfolio/orders` against
`demo-api.kalshi.co` for diagnosis or schema spelunking. Anything that hand-builds
an order body and signs it directly belongs here, never at the repo root.

Probe scripts instantiate `bot.kalshi_client.KalshiDemoClient` directly and POST a
hand-built body. They do not import `bot.main` or `bot.execution.order_placer`;
coupling them to the bot loop would defeat the isolation the directory exists for,
and would orphan rows in `demo_orders` because probes bypass `_place` and
`_commit_phase2`.
