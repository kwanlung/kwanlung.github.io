// Endpoints are not covered by i18n page fallback routes, so the zh RSS
// feed needs an explicit route. GET reads context.currentLocale ("zh").
export { GET } from "../rss.xml";