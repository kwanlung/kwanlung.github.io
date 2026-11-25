import type { UIStrings } from "../types";

export default {
  nav: {
    home: "首页",
    posts: "文章",
    tags: "标签",
    about: "关于",
    archives: "归档",
    search: "搜索",
  },
  post: {
    publishedAt: "发布于",
    updatedAt: "更新于",
    sharePostIntro: "分享这篇文章：",
    sharePostOn: "分享到 {{platform}}",
    sharePostViaEmail: "通过邮件分享",
    tagLabel: "标签",
    backToTop: "回到顶部",
    goBack: "返回",
    editPage: "编辑此页",
    previousPost: "上一篇",
    nextPost: "下一篇",
  },
  pagination: {
    prev: "上一页",
    next: "下一页",
    page: "页",
  },
  home: {
    socialLinks: "社交链接",
    featured: "精选文章",
    recentPosts: "最新文章",
    allPosts: "全部文章",
    heroTitle: "嗨，我是 Dean。",
    heroIntro:
      "我是一名软件工程师，对技术和世界的运行方式充满好奇。这本笔记记录我学到的东西、做过的项目，以及想更深入理解的想法。",
    heroIntro2:
      "我写后端工程与 AI、金融与经济、生活与反思，也记录构建有意义职业生涯的过程。目标很简单：持续学习、清晰思考，把想法变成行动。",
  },
  footer: {
    copyright: "版权所有",
    allRightsReserved: "保留所有权利。",
  },
  pages: {
    tagTitle: "标签",
    tagDesc: "该标签下的所有文章",

    tagsTitle: "标签",
    tagsDesc: "文章中使用的所有标签。",

    postsTitle: "文章",
    postsDesc: "我发布的所有文章。",

    archivesTitle: "归档",
    archivesDesc: "我归档的所有文章。",

    searchTitle: "搜索",
    searchDesc: "搜索任意文章……",
  },
  a11y: {
    skipToContent: "跳到正文",
    openMenu: "打开菜单",
    closeMenu: "关闭菜单",
    toggleTheme: "切换主题",
    searchPlaceholder: "搜索文章……",
    noResults: "未找到结果",
    goToPreviousPage: "上一页",
    goToNextPage: "下一页",
    switchLanguage: "切换语言",
  },
  notFound: {
    title: "404 未找到",
    message: "页面不存在",
    goHome: "返回首页",
  },
} satisfies UIStrings;