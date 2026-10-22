/**
 * How a search setup is named everywhere it is shown: the Ask picker, a run's
 * line in the history, the ladder, the search lab. One name per method
 * (docs/product-language.md): Dense ("by meaning"), BM25 ("by words"),
 * Hybrid (both, fused by rank with RRF).
 */
import type { SearchMethod, SearchRanks } from "@/lib/api";

export const SEARCH_LABELS: Record<SearchMethod, string> = { dense: "Dense", bm25: "BM25", hybrid: "Hybrid" };

/** The "Limit per section" setting the app offers: at most 2 passages per section ahead of the rest. */
export const SECTION_CAP = 2;

/** "Dense", "Hybrid · limit 2 per section". Runs made before M3 have no method: they were dense. */
export function searchName(method: string | null | undefined, cap?: number | null): string {
  const name = SEARCH_LABELS[(method ?? "dense") as SearchMethod] ?? method ?? "Dense";
  return cap ? `${name} · limit ${cap} per section` : name;
}

/** Query rewriting (docs/product-language.md): plain names first; decomposition, multi-query and HyDE in details. */
export const REWRITE_LABELS: Record<string, string> = {
  decompose: "Split into parts",
  multi: "Several phrasings",
  hyde: "Draft answer first",
};
export const REWRITE_EXPLAIN: Record<string, string> = {
  decompose: "Search each part of a many-part question, then pool the results.",
  multi: "Search three other ways of wording it, then combine the lists.",
  hyde: "Write a short draft answer and search with that.",
};

/** The rounds Multi-step search can take (the engine accepts 2 to 6). */
export const AGENT_ROUNDS = [2, 3, 4, 5, 6] as const;

/**
 * The M5 switches of a recorded setup, each in words, in the order the
 * pipeline applies them: "Several phrasings", "Multi-step search, 3 rounds",
 * "reranked by Remote AI · 4B". Empty when none is on.
 */
export function setupExtras(c: {
  rerank?: string | null;
  rerank_name?: string | null;
  rewrite?: string | null;
  agent_rounds?: number | null;
}): string[] {
  const out: string[] = [];
  if (c.rewrite) out.push(REWRITE_LABELS[c.rewrite] ?? c.rewrite);
  if (c.agent_rounds) out.push(`Multi-step search, ${c.agent_rounds} rounds`);
  if (c.rerank_name || c.rerank) out.push(`reranked by ${c.rerank_name ?? c.rerank}`);
  return out;
}

const RANK_NAMES: Record<string, string> = {
  dense: "dense",
  bm25: "BM25",
  original: "the question",
  hyde: "drafted passage",
  reranked: "reranked",
};

function rankName(list: string): string {
  if (RANK_NAMES[list]) return RANK_NAMES[list];
  if (/^q\d+$/.test(list)) return `phrasing ${list.slice(1)}`;
  if (/^r\d+$/.test(list)) return `round ${list.slice(1)}`;
  return list;
}

/** "dense rank 4 · BM25 rank 10", "BM25: not in the top 50", "reranked rank 1", or "" when there are no ranks. */
export function ranksText(ranks: SearchRanks | null | undefined): string {
  if (!ranks) return "";
  return Object.entries(ranks)
    .map(([list, r]) => {
      const name = rankName(list);
      if (r) return `${name} rank ${r}`;
      return list === "reranked" ? "" : `${name}: not in the top 50`;
    })
    .filter(Boolean)
    .join(" · ");
}
