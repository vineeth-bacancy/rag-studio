import { useEffect } from "react";
import useSWR from "swr";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { IndexesResponse, PromptVersion, RerankersResponse, SearchMethod } from "@/lib/api";
import { REWRITE_EXPLAIN, REWRITE_LABELS, SEARCH_LABELS, SECTION_CAP } from "@/lib/search";
import { useAsk } from "./ask-store";

/**
 * Which index answers come from, how it is searched, and which prompt version
 * the model gets. Only indexes that can be asked now can be chosen (built,
 * and for a Colab index, the runtime connected); the others say why. BM25 and hybrid need the index's word index. The defaults are the
 * ones the API names (row 11a, prompt v3) until the scorecard says otherwise.
 */
export function SetupPicker({ disabled }: { disabled?: boolean }) {
  const { data: indexes } = useSWR<IndexesResponse>("/indexes");
  const { data: prompts } = useSWR<{ default: string; versions: PromptVersion[] }>("/prompts");
  const profile = useAsk((s) => s.profile) ?? indexes?.default_profile ?? "";
  const prompt = useAsk((s) => s.prompt) ?? prompts?.default ?? "";
  const search = useAsk((s) => s.search);
  const sectionCap = useAsk((s) => s.sectionCap);
  const rerank = useAsk((s) => s.rerank);
  const setRerank = useAsk((s) => s.setRerank);
  const rewrite = useAsk((s) => s.rewrite);
  const setRewrite = useAsk((s) => s.setRewrite);
  const agentRounds = useAsk((s) => s.agentRounds);
  const setAgentRounds = useAsk((s) => s.setAgentRounds);
  const { data: rerankers } = useSWR<RerankersResponse>("/rerankers");
  const setProfile = useAsk((s) => s.setProfile);
  const setPrompt = useAsk((s) => s.setPrompt);
  const setSearch = useAsk((s) => s.setSearch);
  const setSectionCap = useAsk((s) => s.setSectionCap);

  const chosen = indexes?.profiles.find((p) => p.id === profile);
  const wordIndex = chosen?.word_index ?? true;
  // An index without a word index can only be searched by meaning: the
  // picker shows that switch rather than sending a request the API refuses.
  useEffect(() => {
    if (chosen && !chosen.word_index && search !== "dense") setSearch("dense");
  }, [chosen, search, setSearch]);

  const profileItems = Object.fromEntries((indexes?.profiles ?? []).map((p) => [p.id, p.label]));
  const promptItems = Object.fromEntries((prompts?.versions ?? []).map((v) => [v.version, `Prompt ${v.version}`]));
  const methods = indexes?.search_methods ?? (["dense", "bm25", "hybrid"] as SearchMethod[]).map((id) => ({
    id, label: SEARCH_LABELS[id], explain: "",
  }));
  const searchItems = Object.fromEntries(methods.map((m) => [m.id, m.label]));

  return (
    <div className="flex min-w-0 flex-col gap-1">
      <div className="flex min-w-0 flex-wrap items-center gap-1.5">
        <label htmlFor="ask-index" className="sr-only">
          Index
        </label>
        <Select value={profile} items={profileItems} onValueChange={(v) => v && setProfile(v as string)} disabled={disabled}>
          <SelectTrigger id="ask-index" size="sm" className="max-w-52 text-xs" title="Which index to search">
            <SelectValue placeholder="Index" />
          </SelectTrigger>
          <SelectContent>
            {(indexes?.profiles ?? []).map((p) => (
              <SelectItem key={p.id} value={p.id} disabled={!p.active || p.can?.query.ok === false}>
                <span className="flex flex-col">
                  <span>{p.label}</span>
                  {/* Built is not enough since M4: a Colab index needs the
                      runtime connected. The engine says why (indexes.py). */}
                  {(!p.active || p.can?.query.ok === false) && (
                    <span className="text-[11px] text-muted-foreground">
                      {!p.active ? "Not built yet: build it on the Indexes page." : p.can?.query.reason}
                    </span>
                  )}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <label htmlFor="ask-search" className="sr-only">
          Search
        </label>
        <Select
          value={search}
          items={searchItems}
          onValueChange={(v) => v && setSearch(v as SearchMethod)}
          disabled={disabled}
        >
          <SelectTrigger id="ask-search" size="sm" className="text-xs" title="How the index is searched">
            <SelectValue placeholder="Search" />
          </SelectTrigger>
          <SelectContent>
            {methods.map((m) => {
              const off = m.id !== "dense" && !wordIndex;
              return (
                <SelectItem key={m.id} value={m.id} disabled={off}>
                  <span className="flex flex-col">
                    <span>{m.label}</span>
                    <span className="text-[11px] text-muted-foreground">
                      {off ? `This index has no word index yet. Run: ./rag ingest --profile ${profile}` : m.explain}
                    </span>
                  </span>
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
        <label
          className="flex items-center gap-1.5 rounded-md px-1.5 py-1 text-xs text-muted-foreground"
          title={`At most ${SECTION_CAP} passages from one section ahead of the rest; contents pages and reference lists at most 1. Passages with no known section are never limited.`}
        >
          <Switch
            size="sm"
            checked={sectionCap !== null}
            onCheckedChange={(on) => setSectionCap(on ? SECTION_CAP : null)}
            disabled={disabled}
          />
          Limit per section
        </label>
        <label htmlFor="ask-rerank" className="sr-only">
          Reranker
        </label>
        <Select
          value={rerank ?? "off"}
          items={{ off: "Reranker off", ...Object.fromEntries((rerankers?.rerankers ?? []).map((r) => [r.id, r.label])) }}
          onValueChange={(v) => setRerank(!v || v === "off" ? null : (v as string))}
          disabled={disabled}
        >
          <SelectTrigger
            id="ask-rerank"
            size="sm"
            className="text-xs"
            title={`A slower model reads each of the top ${rerankers?.depth ?? 20} passages with the question and reorders them`}
          >
            <SelectValue placeholder="Reranker" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="off">Reranker off</SelectItem>
            {(rerankers?.rerankers ?? []).map((r) => (
              <SelectItem key={r.id} value={r.id} disabled={!r.can.ok}>
                <span className="flex flex-col">
                  <span>{r.label}</span>
                  {!r.can.ok && <span className="text-[11px] text-muted-foreground">{r.can.reason}</span>}
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <label htmlFor="ask-rewrite" className="sr-only">
          Query rewriting
        </label>
        <Select
          value={rewrite ?? "off"}
          items={{ off: "Rewriting off", ...REWRITE_LABELS }}
          onValueChange={(v) => setRewrite(!v || v === "off" ? null : (v as string))}
          disabled={disabled}
        >
          <SelectTrigger
            id="ask-rewrite"
            size="sm"
            className="text-xs"
            title="Remote AI rewrites the question before searching, then the results are combined"
          >
            <SelectValue placeholder="Query rewriting" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="off">Rewriting off</SelectItem>
            {Object.entries(REWRITE_LABELS).map(([id, label]) => (
              <SelectItem key={id} value={id} disabled={rerankers?.rewrite?.ok === false}>
                <span className="flex flex-col">
                  <span>{label}</span>
                  <span className="text-[11px] text-muted-foreground">
                    {rerankers?.rewrite?.ok === false ? rerankers.rewrite.reason : REWRITE_EXPLAIN[id]}
                  </span>
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <label
          className="flex items-center gap-1.5 rounded-md px-1.5 py-1 text-xs text-muted-foreground"
          title={
            rerankers?.rewrite?.ok === false
              ? `${rerankers.rewrite.reason}`
              : "Remote AI looks at what was found and decides whether to search again, up to 3 rounds"
          }
        >
          <Switch
            size="sm"
            checked={agentRounds !== null}
            onCheckedChange={(on) => setAgentRounds(on ? 3 : null)}
            disabled={disabled || rerankers?.rewrite?.ok === false}
          />
          Multi-step search
        </label>
        <label htmlFor="ask-prompt" className="sr-only">
          Prompt version
        </label>
        <Select value={prompt} items={promptItems} onValueChange={(v) => v && setPrompt(v as string)} disabled={disabled}>
          <SelectTrigger id="ask-prompt" size="sm" className="text-xs" title="Which system prompt version the model gets">
            <SelectValue placeholder="Prompt" />
          </SelectTrigger>
          <SelectContent>
            {(prompts?.versions ?? []).map((v) => (
              <SelectItem key={v.version} value={v.version}>
                Prompt {v.version}
                {v.version === prompts?.default ? " (default)" : ""}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {prompt === "v4" && chosen && !chosen.labels && (
        <p className="px-1 text-[11px] text-muted-foreground">
          Prompt v4 tells the model each passage starts with its section label, and this index has none. Pick a
          structure-aware index, or prompt v3.
        </p>
      )}
    </div>
  );
}
