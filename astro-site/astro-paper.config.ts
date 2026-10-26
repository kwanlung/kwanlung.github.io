import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://kwanlung.github.io/",
    title: "Dean's Notebook",
    description: "Notes on software engineering, AI, finance, economics, and personal growth.",
    author: "Dean Ng",
    profile: "https://github.com/kwanlung",
    ogImage: "default-og.svg",
    lang: "en",
    timezone: "Asia/Kuala_Lumpur",
    dir: "ltr",
  },
  posts: {
    perPage: 8,
    perIndex: 8,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: true,
    showArchives: true,
    showBackButton: true,
    editPost: {
      enabled: true,
      url: "https://github.com/kwanlung/kwanlung.github.io/edit/main/astro-site/",
    },
    search: "pagefind",
  },
  socials: [
    {
      name: "github",
      url: "https://github.com/kwanlung",
      linkTitle: "Dean on GitHub — code and projects",
    },
    {
      name: "linkedin",
      url: "https://www.linkedin.com/in/deanng00/",
      linkTitle: "Connect with Dean on LinkedIn",
    },
    {
      name: "kaggle",
      url: "https://www.kaggle.com/kwanlungng",
      linkTitle: "Dean on Kaggle — data science notebooks",
    },
    {
      name: "x",
      url: "https://x.com/DEANi18015616",
      linkTitle: "Dean on X",
    },
    {
      name: "cv",
      url: "/resume/DeanNgKwanLung-SoftwareEngineer.pdf",
      linkTitle: "View Dean's résumé (PDF)",
    },
    {
      name: "mail",
      url: "mailto:kwanlung123@gmail.com",
      linkTitle: "Email Dean",
    },
    {
      name: "instagram",
      url: "https://www.instagram.com/nkwanlungggg/",
      linkTitle: "Dean on Instagram — personal interests and travel",
    },
  ],
  shareLinks: [
    { name: "whatsapp", url: "https://wa.me/?text=" },
    { name: "facebook", url: "https://www.facebook.com/sharer.php?u=" },
    { name: "x", url: "https://x.com/intent/post?url=" },
    { name: "telegram", url: "https://t.me/share/url?url=" },
    { name: "pinterest", url: "https://pinterest.com/pin/create/button/?url=" },
    { name: "mail", url: "mailto:?subject=See%20this%20post&body=" },
  ],
});