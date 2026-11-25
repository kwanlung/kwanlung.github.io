import type { Page } from "astro";
import type { CollectionEntry } from "astro:content";
import { getRelativeLocaleUrl } from "astro:i18n";
import { getSortedPosts } from "./getSortedPosts";
import config from "@/config";

/**
 * Builds a locale-aware pagination page at render time.
 *
 * The shared listing pages serve both the default-locale routes and the
 * /zh/ mirror routes, so pagination is re-sliced for the current locale's
 * own posts. Returns undefined when the page number is out of range for
 * this locale (caller should rewrite to the localized 404).
 */
export function getLocalePage(
  posts: CollectionEntry<"posts">[],
  locale: string,
  pageParam: string | undefined,
  pathPrefix: string,
  pageSize: number = config.posts.perPage
): Page<CollectionEntry<"posts">> | undefined {
  const localePosts = getSortedPosts(posts).filter(
    post => post.data.lang === locale
  );
  const lastPage = Math.max(1, Math.ceil(localePosts.length / pageSize));
  const currentPage = pageParam === undefined ? 1 : Number(pageParam);

  if (
    !Number.isInteger(currentPage) ||
    currentPage < 1 ||
    currentPage > lastPage
  ) {
    return undefined;
  }

  const start = (currentPage - 1) * pageSize;
  const data = localePosts.slice(start, start + pageSize);
  const pageUrl = (page: number) =>
    getRelativeLocaleUrl(
      locale,
      page === 1 ? pathPrefix : `${pathPrefix}/${page}`
    );

  return {
    data,
    start,
    end: start + data.length - 1,
    size: pageSize,
    total: localePosts.length,
    currentPage,
    lastPage,
    url: {
      current: pageUrl(currentPage),
      first: pageUrl(1),
      last: pageUrl(lastPage),
      prev: currentPage > 1 ? pageUrl(currentPage - 1) : undefined,
      next: currentPage < lastPage ? pageUrl(currentPage + 1) : undefined,
    },
  };
}