# Publishing setup (one time)

`.github/workflows/release.yml` publishes three packages without stored tokens, through OIDC trusted publishing:

| Tag | Package | Registry project | GitHub environment |
|---|---|---|---|
| `v<version>` | library (repository root) | PyPI `breadbowl-embed` | `pypi` |
| `sdk-python-v<version>` | Python SDK (`sdks/python`) | PyPI `breadbowl` | `pypi-sdk-python` |
| `sdk-typescript-v<version>` | TypeScript SDK (`sdks/typescript`) | npm `@breadbowl/embed` | `npm` |

A tag publishes only if `<version>` is exactly the version inside the built package (`pyproject.toml` and `breadbowl_embed/_version.py` for the library, `sdks/python/pyproject.toml`, `sdks/typescript/package.json`); otherwise the build job fails before anything is uploaded. Each publish job runs only after the package's build, `twine check` / `npm pack` and test jobs pass. A manual run (Actions → release → Run workflow → package) builds, checks and tests one package and never publishes; use it to rehearse a release.

Every tag that starts with `v` goes to the library job. A tag such as `v1.3` for a model version would therefore start a library release and fail at the version check; use another prefix for non-library tags.

The registries need to be told once which repository, workflow and environment may publish. Do this before pushing the first tag.

## 1. GitHub: environments

Repository → Settings → Environments → New environment, three times: `pypi`, `pypi-sdk-python`, `npm`. For each one:

- Deployment branches and tags → Selected branches and tags → add a tag rule: `v*` for `pypi`, `sdk-python-v*` for `pypi-sdk-python`, `sdk-typescript-v*` for `npm`.
- Optional: Required reviewers, so a person approves each publish job.
- Add no secrets.

Settings → Rules: consider a tag ruleset that limits who can create `v*`, `sdk-python-v*` and `sdk-typescript-v*` tags.

## 2. PyPI: pending trusted publishers

Neither PyPI project exists yet, so add *pending* publishers: pypi.org → Your account → Publishing → "Add a new pending publisher" → GitHub. Two entries:

| Field | Library | Python SDK |
|---|---|---|
| PyPI Project Name | `breadbowl-embed` | `breadbowl` |
| Owner | `BreadBowlAI` | `BreadBowlAI` |
| Repository name | `breadbowl-embed` | `breadbowl-embed` |
| Workflow name | `release.yml` | `release.yml` |
| Environment name | `pypi` | `pypi-sdk-python` |

A pending publisher does not reserve the name: if someone else registers the project first, it becomes invalid. Once used, it turns into a normal trusted publisher for the project. Repeat the same on test.pypi.org only if you add a TestPyPI job (the workflow has none).

After the first release, in each project's settings, check that the trusted publisher is listed and that no API tokens are needed.

## 3. npm: trusted publisher for `@breadbowl/embed`

1. The `@breadbowl` scope must exist and belong to you: create the npm organization `breadbowl` (npmjs.com → Add Organization) if it does not.
2. npm configures trusted publishers in a package's settings, so the package may need to exist first. The npm documentation does not describe the very first publish; if the package settings page is not available for a never-published name, publish `0.1.0` once by hand from a clean checkout with 2FA (`cd sdks/typescript && pnpm install --frozen-lockfile && pnpm run build && npm publish --access public`), then continue. Check whether npm has added a pending-publisher option before doing this.
3. npmjs.com → package `@breadbowl/embed` → Settings → Trusted Publisher → GitHub Actions:
   - Organization or user: `BreadBowlAI`
   - Repository: `breadbowl-embed`
   - Workflow filename: `release.yml`
   - Environment name: `npm`
   - Allowed actions: enable **`npm publish`**. Configurations created after 2026-09-03 allow only staged publishing (`npm stage publish`, approved by a maintainer with 2FA) by default; the workflow runs `npm publish`, which fails unless it is allowed here. If you prefer staged publishing, change the last step of `sdk-typescript-publish` to `npm stage publish` and approve each release on npmjs.com.
   - A new configuration must complete a publish within 2 days, or it expires and must be added again.
4. Package → Settings → Publishing access → "Require two-factor authentication and disallow tokens", so only the trusted publisher (and maintainers with 2FA) can publish.

Trusted publishing needs npm 11.5.1 or newer and Node 22.14 or newer; the workflow uses Node 24 and installs npm 11. Provenance is attested for public packages published this way; the workflow also passes `--provenance`.

## 4. Releasing

1. Update the version (and changelog) in the package, merge to `main`.
2. Optional rehearsal: Actions → release → Run workflow → choose the package.
3. Tag the release commit and push the tag, e.g. `git tag v0.2.0 && git push origin v0.2.0`.
4. Approve the publish job if the environment requires reviewers.

For the library, docs/releasing.md covers the model weights, which are published separately on Hugging Face.
