/**
 * Every page's name and one-line purpose, in one place.
 *
 * The name IS the page's `<h1>`, its sidebar label, its ⌘K entry and its
 * browser-tab title. The trading app learned this the hard way (F-037): a
 * second list is how one page ends up with three names. No "use client" here,
 * so server layouts (tab titles) and client components (shell) share it.
 */
export type RouteInfo = { title: string; summary: string };

export const ROUTES: Record<string, RouteInfo> = {
  "/": {
    title: "Ask",
    summary: "Ask a question about your reports and get an answer with page citations.",
  },
  "/documents": {
    title: "Documents",
    summary: "Your PDF library, page by page, as each parser reads it.",
  },
  "/indexes": {
    title: "Indexes",
    summary: "The search indexes built from your documents, one per embedding setup.",
  },
  "/learn": {
    title: "How it works",
    summary: "Every step from PDF to cited answer, explained on your own documents.",
  },
  "/evaluation": {
    title: "Evaluation",
    summary: "Test questions with known answers, and how each setup scores on them.",
  },
  "/settings": {
    title: "Settings",
    summary: "Which models answer, where they run, and which techniques are on.",
  },
  "/health": {
    title: "System Health",
    summary: "Technical checks on the AI models, documents, indexes and this computer.",
  },
};

export const ROUTE_TITLES: Record<string, string> = Object.fromEntries(
  Object.entries(ROUTES).map(([href, r]) => [href, r.title]),
);

/** The suffix every tab title carries, and the title of `/` on its own. */
export const APP_NAME = "RAG Studio";
