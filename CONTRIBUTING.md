# Development workflow

`main` is always releasable. Nobody pushes to it directly: every change arrives
through a pull request from a short-lived development branch, and only after CI
passes. Every merged PR with a new version becomes a release, and the live viewer
is updated from it.

## Day to day

1. Branch from `main` with a short, descriptive name: `anatomy-convert`,
   `viewer-about`, `fix-l5-landmarks`.
2. Commit as you go. CI (`.github/workflows/ci.yml`) runs the tests, builds
   the viewer and runs the GPU check (shader vs NumPy) on Linux/SwiftShader,
   Windows/Direct3D 11 and macOS/Metal on every push.
3. Before opening or updating a PR, run the same checks locally:
   ```bash
   ci/check.sh
   ```
4. In the PR (template provided):
   - bump `__version__` in `implicit/__init__.py`. Use
     [semver](https://semver.org/): patch for fixes, minor for features (0.x for now);
   - add a dated section `## [X.Y.Z] - YYYY-MM-DD` to `CHANGELOG.md`, which
     becomes the release notes;
   - list any new third-party code or data in `NOTICE.md`.

   PRs that should not release (docs, CI chores) get the label `no-release`;
   `ci/check.sh --no-release` skips the version check locally.
5. Merge once CI is green. `.github/workflows/release.yml` then:
   - re-runs the tests and builds the viewer stamped with version and commit;
   - creates tag `vX.Y.Z` and a GitHub release with the CHANGELOG section as
     notes and the built viewer attached;
   - deploys the viewer to GitHub Pages: https://siril.me/implicit/

## Third-party dependencies

Every third-party dependency is pinned to one exact version, never a range
(`==`, not `>=`): Python packages in `requirements.txt`, the Python and Node
versions, Playwright and the GitHub Actions in `.github/workflows/`. A build then
always uses the same versions. Update a pin deliberately, in its own change with a
CHANGELOG entry, and only to a version whose licence allows commercial use
(`docs/LICENSING.md`).

## One-time repository settings

These are set in the GitHub web UI (repository **Settings**):

1. **Pages** → *Build and deployment* → Source: **GitHub Actions**.
2. **Rules → Rulesets → New branch ruleset** (or **Branches → Add classic branch protection rule**), target `main`:
   - Restrict deletions, Block force pushes;
   - Require a pull request before merging;
   - Require status checks to pass → add **test**, **gpu (linux-swiftshader)**,
     **gpu (windows-d3d11)** and **gpu (macos-metal)**. They appear in the list
     after CI has run once. Also tick "Require branches to be up to date before merging";
   - no bypass list, so the rules apply to admins too.
3. **Actions → General → Workflow permissions**: *Read and write permissions*,
   so the release workflow can create tags and releases.
