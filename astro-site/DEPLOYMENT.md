# GitHub Pages deployment plan

## Current status

- The live website continues to be built from `kwanlung_blog/` by the existing Hugo workflow.
- AstroPaper lives in `astro-site/` on the `astro-paper-migration` branch.
- `.github/workflows/astro-paper-check.yml` builds AstroPaper for validation only. It does **not** deploy to GitHub Pages.
- The first article is a draft.

## Before switching production

1. Run the build check in the GitHub Actions tab and resolve any errors.
2. Run `pnpm install`, `pnpm build`, and `pnpm preview` locally from `astro-site/`.
3. Review the homepage, theme switching, search, article pages, tags, archives, RSS, sitemap, and mobile layout.
4. Decide which Hugo posts and project pages to migrate. Keep the old Hugo source and the `main` branch history intact.
5. Review the first article before production. It is currently enabled in the migration preview; set `draft: true` again if it should remain unpublished when the production workflow is switched.
6. Make a separate reviewed change to the production deployment workflow so it builds `astro-site/` and uploads `astro-site/dist/` as the Pages artifact.
7. In GitHub repository **Settings → Pages**, confirm the source is **GitHub Actions**.
8. Merge the production cutover only after the build passes and you are ready to replace the public homepage.

## Rollback

The previous Hugo source remains in `kwanlung_blog/` and the Git history on `main` is retained. If the Astro deployment needs to be rolled back, restore the previous Pages workflow and redeploy the Hugo output. Do not delete the old source as part of the migration.

## Why the Astro config has no `base` path

This is a user site served at `https://kwanlung.github.io/`, because the repository is named `kwanlung.github.io`. Unlike a project site such as `username.github.io/project-name/`, it does not need a `base: "/project-name"` setting.
