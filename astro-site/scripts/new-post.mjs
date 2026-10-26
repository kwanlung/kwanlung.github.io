#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const title = process.argv.slice(2).join(" ").trim();

if (!title) {
  console.error('Usage: pnpm new:post -- "Your post title"');
  process.exit(1);
}

const slug = title
  .normalize("NFKD")
  .replace(/\p{M}/gu, "")
  .toLowerCase()
  .replace(/[^a-z0-9]+/g, "-")
  .replace(/^-+|-+$/g, "")
  .slice(0, 80)
  .replace(/-+$/g, "");

if (!slug) {
  console.error("Could not create a URL-friendly slug from that title.");
  process.exit(1);
}

const parts = new Intl.DateTimeFormat("en-CA", {
  timeZone: "Asia/Kuala_Lumpur",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
}).formatToParts(new Date());

const part = type => parts.find(item => item.type === type)?.value;
const date = `${part("year")}-${part("month")}-${part("day")}`;
const postsDir = path.join(process.cwd(), "src", "content", "posts");
const filePath = path.join(postsDir, `${date}-${slug}.md`);
const pubDatetime = new Date().toISOString();

const content = `---
author: "Dean Ng"
pubDatetime: ${pubDatetime}
title: ${JSON.stringify(title)}
description: "TODO: Write a one-sentence summary of this post."
tags:
  - learning
draft: true
---

## The question

What am I trying to understand?

## What I learned

Write the key ideas in your own words.

## Practical notes

Record an example, experiment, or action you took.

## Sources and further reading

- Add reliable references here when useful.

## Reflection

What changed in my understanding, and what will I do next?
`;

try {
  await mkdir(postsDir, { recursive: true });
  await writeFile(filePath, content, { encoding: "utf8", flag: "wx" });
  console.log(`Created: ${path.relative(process.cwd(), filePath)}`);
  console.log("The post is a draft. Edit the frontmatter and content before publishing.");
} catch (error) {
  if (error?.code === "EEXIST") {
    console.error(`A post already exists at: ${path.relative(process.cwd(), filePath)}`);
    process.exit(1);
  }
  console.error(error);
  process.exit(1);
}
