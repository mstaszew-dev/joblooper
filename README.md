# JobLooper

A headless, single-file job-application looper built on the
[OpenRouter Agent SDK](https://www.npmjs.com/package/@openrouter/agent).

It drives an OpenRouter model in a continuous `callModel` loop with a custom
`stopWhen` that returns `false` until the submitted count reaches the target.
The model applies to jobs by calling local tools: file IO, a Playwright
browser (connected over CDP to your already-logged-in Chrome), and a local
vector DB (`vectra`) for augmented context.

## Requirements

- Node 18+ (developed on Node 20)
- Google Chrome with a profile at `~/.playwright-chrome`, launched with
  `--remote-debugging-port=9222` (the `login` command does this for you)
- An `OPENROUTER_API_KEY`

## Setup

```bash
npm install
cp .env.example .env   # then edit OPENROUTER_API_KEY
```

## Usage

```bash
# 1. Open Chrome with portal tabs and wait for you to log in, then start applying:
node index.mjs login

# 2. Run assuming Chrome is already open on :9222:
node index.mjs run

# 3. Safe test: search + fill but do NOT submit/record real applications:
DRY_RUN=1 JOBLOOPER_TEST_MAX_APPLIES=1 node index.mjs dry
```

## Tools the agent can call

- `read_campaign_file` / `write_campaign_file` (tracker.json, applicant.json, notes)
- `browser_navigate` / `browser_snapshot` / `browser_click` / `browser_fill` /
  `browser_upload` / `browser_evaluate` (Playwright over CDP)
- `search_portal` (open a board, extract listing links)
- `score_candidate` (CV-alignment gate)
- `dedupe` (one company once, against tracker.json)
- `record_submission` / `record_skip` (write tracker.json)
- `get_status` (progress)
- `vector_store` / `vector_query` (local vectra context memory)
- `log`

## Notes

- Campaign state is continuous with the existing
  `/Users/mst/Downloads/job-search/job-apply/tracker.json`. The looper reads
  `submitted` from it and writes new entries there.
- `DRY_RUN` makes `record_submission`/`record_skip` log-only (no file writes).
- The loop is wrapped in an outer `while` so it keeps going even if the model
  emits a turn with no tool call before the target is reached.
