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

/** "dense rank 4 · BM25 rank 10", "BM25: not in the top 50", or "" when there are no ranks. */
export function ranksText(ranks: SearchRanks | null | undefined): string {
  if (!ranks) return "";
  return (["dense", "bm25"] as const)
    .filter((m) => m in ranks)
    .map((m) => {
      const r = ranks[m];
      const name = m === "dense" ? "dense" : "BM25";
      return r ? `${name} rank ${r}` : `${name}: not in the top 50`;
    })
    .join(" · ");
}
