# Plan: deploy the viewer automatically when `production` is pushed

Status: plan only, nothing set up yet (2026-11-24).

## Goal

A push to a `production` branch on GitHub updates the live viewer on this
machine (`ensemble.ucsd.edu/pmp/`) with no manual steps. Today the steps are
manual:

1. `git pull` in `~/production-pmp-webserver`;
2. rebuild the frontend (`frontend/dist`, which nginx serves);
3. `sudo systemctl restart pmp-backend`.

## GitHub Actions in five terms

- **Workflow**: a YAML file in `.github/workflows/` that says *when* to run
  (the trigger, e.g. "a push to `production`") and *what* to run.
- **Job**: a group of steps in a workflow that runs on one machine. Jobs can
  depend on each other (`needs:`).
- **Step**: one command, or one ready-made **action** (e.g.
  `actions/checkout`, which clones the repo).
- **Runner**: the machine that runs a job. GitHub lends a fresh VM for each job
  (a *GitHub-hosted* runner), or the job can go to your own machine (a
  *self-hosted* runner).
- **Self-hosted runner**: a small program installed on this server. It keeps
  an outbound HTTPS connection open to GitHub and asks for work. No port is
  opened, and GitHub never connects in, which suits a university server behind
  a firewall.

So the design is: test on a GitHub-hosted runner, then deploy on a
self-hosted runner on this machine.

```
push to production ──► GitHub ──► job "test"   (GitHub VM: pytest, npm build)
                                   │ passes
                                   ▼
                                  job "deploy" (runner on this server)
                                   └─► /usr/local/bin/pmp-deploy <commit>
                                        fetch → build → swap → restart → health check
                                        (rolls back to the previous commit on failure)
```

## The main risk: the repo is public

`dotng14/panMP_whales` is public. GitHub's guidance is to use self-hosted
runners only with private repositories: anyone can open a pull request, and a
workflow triggered by a pull request runs the *submitter's* code. On a
self-hosted runner, that means running a stranger's code on this server.

Mitigations, all of which are in this plan:

1. **The deploy workflow triggers only on `push` to `production`** (and on a
   manual run button). Never on `pull_request` or `pull_request_target`.
2. **Protect `production`** (Settings → Branches → rule): no direct pushes or
   force-pushes, changes only through a pull request from `main` that a
   maintainer approves, with the `test` job required to pass. Only people with
   write access can then trigger a deploy.
3. **Settings → Actions → General**: "Require approval for all outside
   collaborators" for fork PRs, and give `GITHUB_TOKEN` read-only permissions
   by default.
4. **The runner runs as its own unprivileged user**, `pmp-deploy`, not `rivo`
   and not root. Its only extra right is one exact `sudo` command (below).
5. **The deploy logic lives on the server** (`/usr/local/bin/pmp-deploy`, owned
   by root), not in the repo. The workflow only calls it with a commit id.
   Someone who edits the workflow still gets only what `pmp-deploy` may do.
6. **Pin third-party actions** to a full commit SHA, not a tag like `@v4`,
   since a tag can be moved.

If that's still too much exposure, the fallback needs no Actions at all: a
systemd timer that runs `git fetch` every minute and calls the same
`pmp-deploy` when `origin/production` moves. It's simpler and safer, but has
no test gate or deploy log on GitHub. Making the repo private would remove
most of the risk too.

## One-time setup on this server (needs sudo, about 30 minutes)

1. **Deploy user**: `sudo useradd --system --create-home --groups whale_processing pmp-deploy`.
2. **Checkout ownership**: the deploy user must write the checkout and
   `frontend/dist`. Today the checkout is owned by `rivo` and `dist` by root.
   Give both to `pmp-deploy:whale_processing`, group-writable:
   `sudo chown -R pmp-deploy:whale_processing ~rivo/production-pmp-webserver`.
   Moving the checkout to `/srv/pmp` would be cleaner, but then the unit's
   `PYTHONPATH` / `WorkingDirectory` and nginx's `root` must change with it.
3. **The one `sudo` right**, in `/etc/sudoers.d/pmp-deploy` (edit with `visudo`):
   ```
   pmp-deploy ALL=(root) NOPASSWD: /usr/bin/systemctl restart pmp-backend
   ```
4. **The deploy script** at `/usr/local/bin/pmp-deploy` (root-owned, mode
   755); see the sketch below.
5. **The runner**: on GitHub, Settings → Actions → Runners → New self-hosted
   runner → Linux. Follow its download and `./config.sh` steps as
   `pmp-deploy`, adding the label `pmp-prod`. Then
   `sudo ./svc.sh install pmp-deploy && sudo ./svc.sh start`, so the runner
   runs as a systemd service and survives reboots.

## The deploy script (sketch)

```bash
#!/usr/bin/env bash
# pmp-deploy <commit>: deploy one commit of origin/production; roll back on failure.
set -euo pipefail
SHA=$1
APP=/home/rivo/production-pmp-webserver
cd "$APP"
git fetch --quiet origin production
git merge-base --is-ancestor "$SHA" origin/production   # only commits on production
PREV=$(git rev-parse HEAD)
cp -p backend/data/catalog.sqlite "backend/data/catalog.sqlite.bak.$(date -u +%Y%m%dT%H%M%S)_deploy"

deploy() {
  git checkout --quiet --detach "$1"
  /home/rivo/.pyenv/versions/3.13.5/envs/dev/bin/pip install --quiet -e .  # new Python deps
  (cd frontend && npm ci --silent && npx vite build --outDir dist.new --emptyOutDir)
  rm -rf frontend/dist.old && mv frontend/dist frontend/dist.old && mv frontend/dist.new frontend/dist
  sudo /usr/bin/systemctl restart pmp-backend
  for i in $(seq 30); do curl -fsS http://127.0.0.1:8001/api/catalog/csvs >/dev/null && return 0; sleep 2; done
  return 1
}

if ! deploy "$SHA"; then
  echo "deploy of $SHA failed; rolling back to $PREV" >&2
  deploy "$PREV"
  exit 1
fi
echo "deployed $SHA"
```

What the script does, and why:

- **Deploys only commits on `production`**, so the commit id passed in can't
  point anywhere else.
- **Backs up the catalog** before each deploy, as we already do by hand
  before each write.
- **Installs Python dependencies**, since new ones do appear (`pyarrow`
  recently).
- **Builds the frontend into a new folder and swaps it in**, so nginx never
  serves a half-built `dist`.
- **Checks the backend is answering** after the restart; on failure it
  redeploys the previous commit and the job turns red on GitHub.

Two decisions for whoever sets it up:

- **The Python environment is shared.** `pip install` into the `dev` env
  changes the env that `rivo` uses too. A separate env for production would
  stop deploys and development from affecting each other.
- **Repeated deploys can't overlap.** The workflow's `concurrency` setting
  (below) makes sure of that.

## The workflow file (sketch): `.github/workflows/deploy.yml`

```yaml
name: deploy
on:
  push:
    branches: [production]
  workflow_dispatch: {}          # a "Run workflow" button, for redeploys
permissions:
  contents: read
concurrency:
  group: deploy-production
  cancel-in-progress: false      # never stop a deploy halfway

jobs:
  test:                          # GitHub-hosted: nothing runs on our server yet
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@<full-sha>   # pin, see "risk" 6
      - uses: actions/setup-python@<full-sha>
        with: { python-version: "3.13" }
      - run: pip install -e ".[test]"
      - run: GPL_COMPUTE=cpu pytest -q analysis/test analysis/GPL_Python/tests --no-cov
      - uses: actions/setup-node@<full-sha>
        with: { node-version: "22" }
      - run: cd frontend && npm ci && npm run build

  deploy:
    needs: test
    runs-on: [self-hosted, pmp-prod]
    environment: production      # deploy history in the GitHub UI; can require an approval click
    steps:
      - run: /usr/local/bin/pmp-deploy "$GITHUB_SHA"
```

The deploy job doesn't check out the repo on the server: `pmp-deploy` updates
the live checkout itself.

## Before the tests can gate deploys

The suite has known failures on `main`:

- `test_subspace_dictionary`;
- `test_siteh_auto_index_skipped`, which depends on whether `/mnt/sda` is
  mounted on the machine running it;
- `test_matlab_report_193500.py`, which needs a Windows path;
- the 75% integration coverage gate.

Either fix them, or mark them skipped or expected to fail on CI first,
otherwise every deploy is blocked. The MATLAB and GPU tests already skip
without MATLAB or a GPU.

## Day-to-day use

1. Work on feature branches and merge to `main` as now.
2. To release: open a pull request `main` → `production`, have it approved,
   and merge.
3. Watch the run under the repo's **Actions** tab. Green means live; red means
   rolled back, and the log says why.
4. To roll back on purpose: revert on `production` (a normal commit), or use
   "Run workflow" on an older commit.

## Rollout

1. Create `production` from the commit live now:
   `git push origin <live-sha>:refs/heads/production`. Add the branch rule.
2. Server setup (above). Run `sudo -u pmp-deploy /usr/local/bin/pmp-deploy <live-sha>`
   by hand once, which should be a no-op deploy.
3. Add `deploy.yml` on `main` with only `workflow_dispatch`, and run it once
   from the Actions tab.
4. Add the `push` trigger and merge `main` → `production` with a small change.
   Check the viewer, and check that a deliberately broken commit rolls back.

## Decisions

- **The repo is public: Actions runner as planned, the timer fallback, or
  make the repo private?** Recommended: the runner with every mitigation
  above, or making the repo private if nothing requires it to be public.
- **Keep the checkout in `~rivo/production-pmp-webserver` or move it to
  `/srv/pmp`?** Moving is cleaner, but means editing the systemd unit and
  nginx.
- **A separate Python env for production?** Recommended.
- **Require a manual approval click for each deploy** (`environment:
  production` with a required reviewer)? This is optional, and useful while the
  setup is new.
