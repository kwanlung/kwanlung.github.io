# Dean's Notebook

个人博客源码：<https://kwanlung.github.io/>

基于 [Astro](https://astro.build/) + [AstroPaper](https://github.com/satnaing/astro-paper) 主题，支持中英双语（English 默认，中文在 `/zh/` 下）。

## 仓库结构

| 目录 | 说明 |
|---|---|
| `astro-site/` | 当前站点（Astro + AstroPaper），所有开发都在这里 |
| `kwanlung_blog/` | 旧版 Hugo 站点，仅作归档保留 |

## 快速开始

要求：Node.js ≥ 22.12，pnpm 10。

```bash
cd astro-site
pnpm install
pnpm dev        # http://localhost:4321（中文 http://localhost:4321/zh/）
```

## 常用命令

均在 `astro-site/` 目录下执行：

| 命令 | 用途 |
|---|---|
| `pnpm dev` | 开发服务器，热更新 |
| `pnpm new:post -- "文章标题"` | 创建草稿文章（自动带日期文件名，`draft: true`） |
| `pnpm build` | 类型检查 + 静态构建 + Pagefind 搜索索引 |
| `pnpm preview` | 本地预览 build 产物 |
| `pnpm sync` | 同步 content collection 类型 |
| `pnpm lint` / `pnpm format` | ESLint / Prettier |

## 双语发文

文章放在 `astro-site/src/content/posts/`，靠 frontmatter 两个字段配对：

```yaml
lang: en                              # en 或 zh
translationKey: my-post-key           # 中英文版本填同一个值
```

- 英文版可用 `pnpm new:post` 生成；中文版手动建文件，`lang: zh` + 相同 `translationKey`。
- 配对后文章页的语言切换按钮直接跳对应译文；未配对的文章只出现在自己语言的列表里，切换按钮回退到对方语言的文章列表。
- `draft: true` 的文章不会出现在任何公开列表、RSS 或相邻文章导航中；dev 环境下可直接访问文章 URL 预览草稿，改 `false` 才发布。

更详细的编辑指引见 [astro-site/README.md](astro-site/README.md)。

## 部署

推送到 `main` 分支后，GitHub Actions（`.github/workflows/deploy.yaml`）自动构建 `astro-site/` 并发布到 GitHub Pages。切换上线前请先阅读 [astro-site/DEPLOYMENT.md](astro-site/DEPLOYMENT.md)。