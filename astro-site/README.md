# Dean's Notebook — AstroPaper

This folder contains the AstroPaper-based successor to the existing Hugo website. It is being prepared on the `astro-paper-migration` branch so the live site remains unchanged during setup.

## Requirements

- Node.js 22.12 or newer
- pnpm 10 (recommended by the template)

## Run locally

From this directory:

```bash
pnpm install
pnpm dev
```

Open the local URL printed by Astro (normally `http://localhost:4321`).

Verify a production build before deployment:

```bash
pnpm build
pnpm preview
```

## Create a new post

From this `astro-site/` directory, run:

```bash
pnpm new:post -- "Your post title"
```

The generator creates a dated Markdown file under `src/content/posts/`, with a URL-friendly filename and a starter outline. New posts start with `draft: true`, so they stay out of public post lists until you review the article and change that value to `false`.

Then preview the article with `pnpm dev`. When it is ready, run `pnpm build` before committing and pushing.

## Where to edit

- Site name, description, timezone, theme features, and social links: `astro-paper.config.ts`
- Astro framework setup: `astro.config.ts`
- Blog posts: `src/content/posts/`
- About page: `src/content/pages/about.md`
- Colour tokens: `src/styles/theme.css`
- Static assets: `public/`

The first post is enabled in the migration preview so the Posts and Tags pages have real content. Review it before any production deployment; set `draft: true` again if you want to keep it out of public listings.

## Deployment safety

The current production workflow still builds the Hugo site from `kwanlung_blog/`. The Astro workflow initially performs a build check only and does not publish to GitHub Pages. See [DEPLOYMENT.md](DEPLOYMENT.md) for the planned cutover steps.
