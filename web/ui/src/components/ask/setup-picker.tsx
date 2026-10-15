import { useEffect } from "react";
import useSWR from "swr";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import type { IndexesResponse, PromptVersion, SearchMethod } from "@/lib/api";
import { SEARCH_LABELS, SECTION_CAP } from "@/lib/search";
import { useAsk } from "./ask-store";

/**
 * Which index answers come from, how it is searched, and which prompt version
 * the model gets. Only built indexes can be chosen; the others say how to
 * build them. BM25 and hybrid need the index's word index. The defaults stay
 * the baseline until the scorecard says otherwise.
 */
export function SetupPicker({ disabled }: { disabled?: boolean }) {
  const { data: indexes } = useSWR<IndexesResponse>("/indexes");
  const { data: prompts } = useSWR<{ default: string; versions: PromptVersion[] }>("/prompts");
  const profile = useAsk((s) => s.profile);
  const prompt = useAsk((s) => s.prompt);
  const search = useAsk((s) => s.search);
  const sectionCap = useAsk((s) => s.sectionCap);
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
              <SelectItem key={p.id} value={p.id} disabled={!p.active}>
                <span className="flex flex-col">
                  <span>{p.label}</span>
                  {!p.active && (
                    <span className="text-[11px] text-muted-foreground">
                      Not built yet. Run: ./rag ingest --profile {p.id}
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
