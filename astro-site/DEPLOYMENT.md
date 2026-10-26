# GitHub Pages deployment plan

## Current status

- The live `main` branch still builds and publishes the Hugo site from `kwanlung_blog/`.
- AstroPaper lives in `astro-site/` on the `astro-paper-migration` branch.
- `.github/workflows/astro-paper-check.yml` validates AstroPaper and does not deploy.
- `.github/workflows/deploy.yaml` has been updated on the migration branch. It deploys AstroPaper only after the PR is merged and a push reaches `main`.
- The first article currently has `draft: false`, so it will appear on the public site after the cutover. Set it to `true` if it is not ready.

## Before switching production

1. Ensure the latest AstroPaper build check succeeds.
2. Review the site locally with `pnpm dev`, then run `pnpm build` from `astro-site/`.
3. Review the homepage, theme switching, search, article page, tags, archives, RSS, sitemap, résumé link, and mobile layout.
4. Decide which old Hugo articles and project pages should be migrated. The existing Hugo source remains preserved in `kwanlung_blog/`.
5. Review the first article. Set `draft: true` if you do not want it published yet.
6. On GitHub, open the migration PR, click **Ready for review**, and merge it only when you are ready to replace the public homepage.
7. Confirm the repository's **Settings → Pages → Build and deployment → Source** is **GitHub Actions**.
8. After merging, watch the **Actions** tab until **Deploy AstroPaper site to GitHub Pages** succeeds, then verify `https://kwanlung.github.io/`.

## What happens after merging

The workflow on `main` runs when new commits are pushed. It installs pnpm dependencies, type-checks and builds AstroPaper, uploads `astro-site/dist/` as the Pages artifact, then deploys that artifact. The Hugo deployment command is no longer used by the production workflow after the merge.

## Rollback

The old Hugo source remains in `kwanlung_blog/`, and the Git history on `main` is retained. If the Astro deployment needs to be rolled back, restore the previous Hugo Pages workflow and redeploy the Hugo output. Do not delete the old source as part of the migration.

## Why the Astro config has no `base` path

This is a user site served at `https://kwanlung.github.io/`, because the repository is named `kwanlung.github.io`. Unlike a project site such as `https://username.github.io/project-name/`, it does not need a project-subdirectory `base` setting.
