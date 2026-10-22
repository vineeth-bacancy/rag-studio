"use client";

import { useState } from "react";
import useSWR from "swr";
import { ArrowDownUp, Loader2, Search } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { TechnicalDetails } from "@/components/common/technical-details";
import type { DocumentFile, GoldenSet, IndexesResponse, RerankersResponse, SearchLab as SearchLabData } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { num } from "@/lib/format";
import { SECTION_CAP } from "@/lib/search";
import { cn } from "@/lib/utils";
import { SearchColumns, type LineSet } from "./search-columns";

/** Test questions that show each method failing differently, by id. */
const PRESETS: { set: string; id: string; why: string }[] = [
  { set: "set-b", id: "B4", why: "Dense finds growth, but U.S. and euro-area growth: right topic, wrong subject." },
  { set: "set-a", id: "A6", why: "The risks paragraph never says \"executive summary\"; only a section label can." },
  { set: "set-a", id: "A5", why: "Chapter 3's reference list is full of topic words and states no finding." },
  { set: "set-d", id: "D6", why: "The number is in a table row: a few words, many numbers." },
];

const LAB_QUERY_LIMIT = 500;

/**
 * Step 7, the search lab: one question searched three ways at once, on any
 * index. Every list is the engine's real retriever (rag_studio/retrieve.py),
 * 10 deep; where the first answer page ranked is counted 50 deep, the depth
 * an evaluation run uses. "Rerank" adds the dense top 20 re-read by a
 * reranker (the same `retrieve.run` Ask and evaluation use).
 */
export function SearchLab() {
  const { data: indexes } = useSWR<IndexesResponse>("/indexes");
  const { data: golden } = useSWR<{ sets: GoldenSet[] }>("/golden");
  const { data: docData } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const [draft, setDraft] = useState<string | null>(null);
  const [asked, setAsked] = useState<string | null>(null);
  const [pickedProfile, setProfile] = useState<string | null>(null);
  // Until one is picked, the index the app uses by default (the API names it).
  const profile = pickedProfile ?? indexes?.default_profile ?? "";
  const [capped, setCapped] = useState(false);
  const [showScores, setShowScores] = useState(false);

  const presets = PRESETS.flatMap((p) => {
    const q = golden?.sets.find((s) => s.id === p.set)?.questions.find((x) => x.id === p.id);
    return q ? [{ ...p, question: q.question }] : [];
  });
  // The first preset is asked as soon as the test questions load, so the page
  // opens on a real comparison rather than an empty form.
  const question = asked ?? presets[0]?.question ?? null;
  const text = draft ?? question ?? "";

  const key = question && profile
    ? `/learn/search?${new URLSearchParams({ q: question, profile, ...(capped ? { cap: String(SECTION_CAP) } : {}) })}`
    : null;
  const { data, error, isLoading, mutate } = useSWR<SearchLabData>(key, { keepPreviousData: true });
  const built = (indexes?.profiles ?? []).filter((p) => p.active);
  const chosen = built.find((p) => p.id === profile);

  // Reranking is slow (20-35 s on this CPU), so it runs only when asked, and
  // a request counts only for the question and index it was asked for.
  const { data: rerankers } = useSWR<RerankersResponse>("/rerankers");
  const usable = (rerankers?.rerankers ?? []).filter((r) => r.can.ok);
  const [pickedReranker, setReranker] = useState<string | null>(null);
  const reranker = usable.find((r) => r.id === pickedReranker) ?? usable.find((r) => r.where === "colab") ?? usable[0];
  const [rerankFor, setRerankFor] = useState<{ key: string; id: string } | null>(null);
  const rerankKey = key && rerankFor?.key === key ? `${key}&${new URLSearchParams({ rerank: rerankFor.id })}` : null;
  const reranked = useSWR<SearchLabData>(rerankKey, { revalidateOnFocus: false });
  const [lineSet, setLineSet] = useState<LineSet>("fusion");
  const shown = reranked.data ?? data;

  if (!indexes || !golden) return <CardSkeleton height="h-96" />;

  const ask = (q: string) => {
    const t = q.trim();
    if (!t) return;
    setAsked(t);
    setDraft(null);
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle as="h2">Search lab</CardTitle>
          <CardDescription>
            The same question, three ways, and on request a fourth: the dense list reranked. Hover a passage to
            follow it from list to list; answer pages of a test question are marked.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <form
            className="flex flex-col gap-2 sm:flex-row"
            onSubmit={(e) => {
              e.preventDefault();
              ask(text);
            }}
          >
            <label htmlFor="lab-question" className="sr-only">
              Question
            </label>
            <Input
              id="lab-question"
              value={text}
              maxLength={LAB_QUERY_LIMIT}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Type a question..."
              className="flex-1"
            />
            <Button type="submit" disabled={!text.trim()}>
              <Search aria-hidden />
              Search
            </Button>
          </form>

          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Test questions to try">
            {presets.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => ask(p.question)}
                aria-pressed={question === p.question}
                title={p.why}
                className={cn(
                  "rounded-md px-2.5 py-1 text-left text-xs ring-1 ring-foreground/10 transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
                  question === p.question && "bg-muted ring-foreground/40",
                )}
              >
                <span className="font-medium tabular-nums">{p.id}</span> {p.question}
              </button>
            ))}
          </div>
          {presets.find((p) => p.question === question) && (
            <p className="text-xs text-pretty text-muted-foreground">{presets.find((p) => p.question === question)!.why}</p>
          )}

          <div className="flex flex-wrap items-end gap-x-5 gap-y-3">
            <div className="min-w-0 space-y-1">
              <label htmlFor="lab-index" className="text-xs font-medium text-muted-foreground">
                Index
              </label>
              <Select
                value={profile}
                items={Object.fromEntries(built.map((p) => [p.id, p.label]))}
                onValueChange={(v) => v && setProfile(v as string)}
              >
                <SelectTrigger id="lab-index" className="w-full max-w-80 sm:w-80">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {built.map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <label className="flex cursor-pointer items-center gap-2 pb-1 text-sm" title={`At most ${SECTION_CAP} passages from one section ahead of the rest in the hybrid list; contents pages and reference lists at most 1.`}>
              <Switch checked={capped} onCheckedChange={setCapped} />
              Limit per section
            </label>
            <label className="flex cursor-pointer items-center gap-2 pb-1 text-sm">
              <Switch checked={showScores} onCheckedChange={setShowScores} />
              Show scores
            </label>
          </div>
          {chosen && <p className="max-w-prose text-xs text-pretty text-muted-foreground">{chosen.description}</p>}
          {capped && (
            <p className="text-xs text-muted-foreground">
              The limit applies to the hybrid list. Passages with no known section (the baseline index has none) are
              never limited.
            </p>
          )}
        </CardContent>
      </Card>

      {error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : !shown ? (
        <div role="status" aria-busy="true" className="space-y-2">
          <CardSkeleton height="h-96" />
          <p className="text-xs text-muted-foreground">Embedding the question and searching three ways.</p>
        </div>
      ) : (
        <div className={cn("space-y-3", isLoading && "opacity-70")}>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            <span>
              {shown.test_question ? `Test question ${shown.test_question}` : "Your question"} · {shown.label} index ·{" "}
              {num(shown.rows)} passages · top {shown.k} each
            </span>
            {shown.targets.length > 0 && (
              <span>
                · Answer pages:{" "}
                {shown.targets.map((t) => `${docTitle(t.doc_id, docData?.documents)} p. ${t.pages.join(", ")}`).join("; ")}
              </span>
            )}
            {isLoading && (
              <span className="flex items-center gap-1" role="status">
                <Loader2 className="size-3.5 animate-spin" aria-hidden />
                Searching...
              </span>
            )}
            {shown.lists.reranked && (
              // Hybrid can only sit beside two columns, so one set of lines is drawn at a time.
              <span className="ml-auto flex items-center gap-2">
                <span id="lab-lines">Lines</span>
                <Tabs value={lineSet} onValueChange={(v) => setLineSet(v as LineSet)}>
                  <TabsList aria-labelledby="lab-lines" className="h-7">
                    <TabsTrigger value="fusion" className="px-2 text-xs">
                      Fusion
                    </TabsTrigger>
                    <TabsTrigger value="rerank" className="px-2 text-xs">
                      Reranking
                    </TabsTrigger>
                  </TabsList>
                </Tabs>
              </span>
            )}
          </div>
          <SearchColumns
            data={shown}
            docs={docData?.documents}
            showScores={showScores}
            lineSet={shown.lists.reranked ? lineSet : "fusion"}
            rerankSlot={
              reranked.error ? (
                <div className="space-y-2">
                  <p className="text-xs text-pretty text-destructive">{reranked.error.message}</p>
                  <Button size="sm" variant="outline" onClick={() => void reranked.mutate()}>
                    Try again
                  </Button>
                </div>
              ) : rerankKey ? (
                <p className="flex items-start gap-1.5 text-xs text-pretty text-muted-foreground" role="status">
                  <Loader2 className="mt-0.5 size-3.5 shrink-0 animate-spin" aria-hidden />
                  {reranker?.where === "local"
                    ? "Reading 20 passages with the question. About 20-35 seconds on this computer, longer the first time while the model loads."
                    : "Reading 20 passages with the question on Remote AI. A few seconds."}
                </p>
              ) : !reranker ? (
                <p className="text-xs text-pretty text-muted-foreground">
                  {(rerankers?.rerankers ?? [])[0]?.can.reason ?? "No reranker can run now."}{" "}
                  {(rerankers?.rerankers ?? [])[0]?.can.fix && (
                    <code className="font-mono">{rerankers!.rerankers[0].can.fix}</code>
                  )}
                </p>
              ) : (
                <div className="space-y-2">
                  {usable.length > 1 && (
                    <>
                      <label htmlFor="lab-reranker" className="sr-only">
                        Reranker
                      </label>
                      <Select
                        value={reranker.id}
                        items={Object.fromEntries(usable.map((r) => [r.id, r.label]))}
                        onValueChange={(v) => v && setReranker(v as string)}
                      >
                        <SelectTrigger id="lab-reranker" size="sm" className="w-full text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {usable.map((r) => (
                            <SelectItem key={r.id} value={r.id}>
                              {r.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </>
                  )}
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={!key}
                    onClick={() => {
                      if (!key) return;
                      setRerankFor({ key, id: reranker.id });
                      setLineSet("rerank");
                    }}
                  >
                    <ArrowDownUp aria-hidden />
                    Rerank
                  </Button>
                  {usable.length === 1 && <p className="text-[11px] text-muted-foreground">With {reranker.label}.</p>}
                </div>
              )
            }
          />
          <TechnicalDetails
            rows={{
              Fusion: `RRF: each passage scores the sum of 1 / (${shown.rrf_k} + rank) over the lists it is in; 50 per list`,
              "Embed question": shown.timings.embed_ms !== undefined ? `${shown.timings.embed_ms} ms` : "-",
              "Dense search": shown.timings.search_ms !== undefined ? `${shown.timings.search_ms} ms, exact, cosine` : "-",
              "BM25 search": shown.timings.bm25_ms !== undefined ? `${shown.timings.bm25_ms} ms` : "-",
              Reranker: shown.lists.reranked
                ? `${shown.lists.reranked.label}: ${num(Math.round(shown.lists.reranked.ms))} ms for the dense top 20`
                : "-",
              Scores:
                "Dense: cosine similarity. BM25: its own scale, comparable within one question only. Hybrid: the RRF score. Reranked: how much more the reranker leans to \"yes, this passage answers it\" than to \"no\" (a logit difference; only the order matters).",
            }}
          />
        </div>
      )}
    </div>
  );
}
