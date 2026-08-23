# V11.2 free-tier deployment report

Generated 2026-08-23. Free-tier terms can change; verify them before deployment.

## Architecture

- Render Free: one Flask/Gunicorn web service only.
- Supabase Free: PostgreSQL operational state and prediction persistence.
- GitHub Actions: scheduled/manual current-season acquisition, exact feature
  construction, frozen inference, and health updates.
- GitHub Actions cache: resumable current-season files only.

No Render worker, persistent disk, or Render database is declared.

## Expected cost

Target: **$0/month**, subject to staying within third-party free quotas.

- Render documents 750 free instance-hours per workspace, idle spin-down after
  15 minutes, approximately one-minute cold starts, ephemeral filesystems, and
  bandwidth/build-minute limits. Source: https://render.com/docs/free
- Supabase currently lists a 500 MB Free database limit, 5 GB egress, and project
  pausing after one week of inactivity. Scheduled database writes should normally
  prevent inactivity, but this is not guaranteed. Sources:
  https://supabase.com/pricing and
  https://supabase.com/docs/guides/platform/database-size
- GitHub states standard hosted runners are free and unlimited for public
  repositories. Source:
  https://docs.github.com/en/actions/reference/runners/github-hosted-runners

Charges could occur if a service is upgraded, paid add-ons are enabled, Render
usage exceeds included quotas while billing is enabled, the repository becomes
private and exhausts included Actions minutes, or providers change their plans.

## Cache audit

The current 2026 cache selection is about 745.5 MB uncompressed locally across
roughly 3,963 files. Compression should reduce transfer/storage, but actual cache
size must be observed in Actions.

The cache includes:

- resumable seven-day Statcast chunks;
- current-season normalized Statcast files;
- MLB schedule/boxscore caches;
- current-season lineups and official pitching lines;
- processed files whose names contain `2026`;
- live prediction/marker state;
- live tracking outputs.

It excludes 2021–2025 research data, secrets, credentials, virtual environments,
and unrelated results.

GitHub caches are immutable. The workflow creates at most one cache generation
per UTC date and restores the newest prior generation. Runtime lineup markers and
snapshots are also persisted in Supabase, so later jobs do not require an updated
same-day cache to recognize an already-built forecast. Older daily generations
will be evicted under GitHub's repository cache quota. A cache miss is recoverable
but triggers a longer resumable bootstrap.

## Refresh schedule

- Every 30 minutes, 12:00–15:59 UTC, March–November.
- Every 15 minutes, 16:00–05:59 UTC, March–November.
- No scheduled runs from 06:00–11:59 UTC.
- Manual `workflow_dispatch` at any time, with an optional forced season refresh.

GitHub cron is UTC, can be delayed, and is not an SLA.

## Incremental behavior

Frequent executions first poll today's lightweight MLB schedule. The expensive,
resumable season source refresh runs only when:

- required cache files are absent;
- the UTC date changes; or
- a manual run explicitly forces it.

The frozen builders exclude the current game date from pregame history, so games
that finish earlier on the same date do not require a same-day season rebuild.
They enter the next UTC date's cached season advancement.

Open Statcast tail chunks are refreshed rather than accepting partial cached
weeks. Exact feature mathematics and builders remain unchanged. A lineup change
still invokes the exact current-day staging and rebuild path.

## Persistence and immutability

Supabase is accessed as PostgreSQL using a server-side connection URL. The schema
contains final snapshots, latest provisional snapshots, append-only provisional
history, owner lineups, per-player availability, append-only audit records,
rebuild requests, operational state, health, and compact tracking summaries.

Final snapshots retain `INSERT ... ON CONFLICT DO NOTHING` semantics. Owner edits
queue a rebuild; Actions marks it complete, incomplete, or locked. Official MLB
lineups remain authoritative and provisional predictions remain excluded from
official live tracking.

## Known constraints

- The acquisition bootstrap remains explicitly frozen to the 2026 season and
  must be advanced deliberately for 2027.
- A first-ever cache miss can take much longer than routine polling.
- Supabase Free has no included automatic backups/PITR.
- Direct PostgreSQL connections from transient Actions jobs should use the
  Supabase connection method/pooler recommended for the selected network setup.
- Owner edits are asynchronous; the website must not promise exact completion.
