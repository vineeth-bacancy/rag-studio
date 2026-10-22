"use client";

import { useState } from "react";
import useSWR from "swr";
import { ChevronRight, Loader2, Play, Target } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { WhatHappened } from "@/components/ask/trace";
import { HitLabel, OutcomeBadge } from "@/components/eval/badges";
import { SearchSteps } from "@/components/eval/search-steps";
import type {
  DocumentFile,
  GoldenSet,
  IndexesResponse,
  RerankersResponse,
  RewriteLab as RewriteLabData,
  RewriteLabTechnique,
  RewriteLive,
} from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { NO_DATA } from "@/lib/format";
import { REWRITE_LABELS } from "@/lib/search";
import { cn } from "@/lib/utils";

/** Test questions each technique was built for, by id. */
const PRESETS: { set: string; id: string; why: string }[] = [
  { set: "set-b", id: "B4", why: "Worded like a topic, not like the report's sentence: other phrasings may find the subject." },
  { set: "set-a", id: "A2", why: "Two questions in one sentence: what replaced FDsys, and when." },
  { set: "set-d", id: "D10", why: "Its two answers sit in two documents, so one search rarely finds both." },
  { set: "set-e", id: "E1", why: "A follow-up: \"And for 2027?\" means nothing without the turn before it." },
];

const NAMES: Record<string, string> = {
  asked: "As asked",
  ...REWRITE_LABELS,
  agent: "Multi-step search",
  followup: "Made whole by Remote AI",
  as_typed: "Asked as typed",
};

const HOW: Record<string, string> = {
  asked: "One search with the question exactly as written.",
  decompose: "Remote AI splits it into up to 3 questions; each is searched, and the lists are taken in turn.",
  multi: "Remote AI writes 3 other ways a report might say it; all 4 lists are fused by rank.",
  hyde: "Remote AI drafts a short passage that would answer it; that passage is searched like a document.",
  agent: "After each search Remote AI reads what was found and answers or searches again, up to 3 rounds.",
  followup: "Remote AI rewrites the follow-up into a full question using the turn before it, and that is searched.",
  as_typed: "The follow-up is searched exactly as typed, with no memory of the conversation.",
};

/**
 * Step 8: query rewriting and multi-step search. One test question as each
 * technique searched it in recorded evaluation runs (what was searched for,
 * each round, where the answer page ranked), so it works with Remote AI off;
 * with Remote AI connected one technique can be run live.
 */
export function RewriteLab() {
  const { data: golden } = useSWR<{ sets: GoldenSet[] }>("/golden");
  const { data: indexes } = useSWR<IndexesResponse>("/indexes");
  const { data: docData } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const docs = docData?.documents;
  const [qid, setQid] = useState(PRESETS[0].id);
  const [picked, setProfile] = useState<string | null>(null);
  const profile = picked ?? indexes?.default_profile ?? null;
  const built = (indexes?.profiles ?? []).filter((p) => p.active);

  const key = profile ? `/learn/rewrite?${new URLSearchParams({ question: qid, profile })}` : null;
  const { data, error, isLoading, mutate } = useSWR<RewriteLabData>(key, { keepPreviousData: true });

  const presets = PRESETS.flatMap((p) => {
    const q = golden?.sets.find((s) => s.id === p.set)?.questions.find((x) => x.id === p.id);
    return q ? [{ ...p, question: q.question }] : [];
  });
  const preset = presets.find((p) => p.id === qid);

  if (!golden || !indexes) return <CardSkeleton height="h-96" />;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle as="h2">Rewriting, replayed</CardTitle>
          <CardDescription>
            What each technique searched for when a run asked this question, and where the answer page ranked. Read
            from recorded runs: nothing is asked of a model while you look.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-muted-foreground">Test question</p>
            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Test question to replay">
              {presets.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  onClick={() => setQid(p.id)}
                  aria-pressed={qid === p.id}
                  title={p.why}
                  className={cn(
                    "rounded-md px-2.5 py-1 text-left text-xs ring-1 ring-foreground/10 transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
                    qid === p.id && "bg-muted ring-foreground/40",
                  )}
                >
                  <span className="font-medium tabular-nums">{p.id}</span> {p.question}
                </button>
              ))}
            </div>
            {preset && <p className="text-xs text-pretty text-muted-foreground">{preset.why}</p>}
          </div>
          <div className="min-w-0 space-y-1">
            <label htmlFor="rewrite-index" className="text-xs font-medium text-muted-foreground">
              Index
            </label>
            <Select
              value={profile ?? ""}
              items={Object.fromEntries(built.map((p) => [p.id, p.label]))}
              onValueChange={(v) => v && setProfile(v as string)}
            >
              <SelectTrigger id="rewrite-index" className="w-full max-w-96 sm:w-96">
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
        </CardContent>
      </Card>

      {error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : !data ? (
        <CardSkeleton height="h-64" />
      ) : (
        <div className={cn("space-y-3", isLoading && "opacity-70")}>
          {data.history && data.history.length > 0 && (
            <p className="text-xs text-pretty text-muted-foreground">
              Earlier turn: <span className="text-foreground">{data.history.at(-1)!.question}</span> · then:{" "}
              <span className="text-foreground">{data.question}</span>
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            Answer pages:{" "}
            {data.targets.length
              ? data.targets.map((t) => `${docTitle(t.doc_id, docs)} p. ${t.pages.join(", ")}`).join("; ")
              : NO_DATA}
          </p>
          <ol className="space-y-2">
            {data.techniques.map((t) => (
              <TechniqueRow key={t.technique} t={t} docs={docs} />
            ))}
          </ol>
        </div>
      )}

      {data && !data.history && profile && <LiveRun question={data.question} profile={profile} docs={docs} />}
    </div>
  );
}

function TechniqueRow({ t, docs }: { t: RewriteLabTechnique; docs: DocumentFile[] | undefined }) {
  const [open, setOpen] = useState(false);
  const r = t.result;
  const reranker = typeof t.search?.rerank === "string" ? t.search.rerank : null;
  return (
    <li className="@container/result rounded-lg bg-card ring-1 ring-foreground/10">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        disabled={!r}
        aria-expanded={r ? open : undefined}
        className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2.5 text-left focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none disabled:cursor-default"
      >
        {r && (
          <ChevronRight className={cn("size-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-90")} aria-hidden />
        )}
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium">{NAMES[t.technique] ?? t.technique}</span>
          <span className="block text-xs text-pretty text-muted-foreground">{HOW[t.technique]}</span>
        </span>
        {r ? (
          <span className="flex items-center gap-2 text-xs">
            <HitLabel result={r} />
            {r.outcome && <OutcomeBadge outcome={r.outcome} error={r.error} />}
          </span>
        ) : (
          <Badge variant="outline" className="font-normal text-muted-foreground">
            Not measured yet
          </Badge>
        )}
      </button>
      {r && (
        <p className="px-3 pb-2 text-[11px] text-muted-foreground tabular-nums">
          {t.ladder ? `Ladder row ${t.ladder.row}: ${t.ladder.label}` : t.label} · run {t.run_id}
          {t.mode === "retrieval" ? " · search only" : ""}
          {reranker ? ` · reranked (${reranker})` : ""}
          {r.best_rank !== null ? ` · answer page at rank ${r.best_rank} searching 50 deep` : " · answer page not in the top 50"}
        </p>
      )}
      {r && open && <SearchSteps r={r} docs={docs} />}
      {r && open && t.technique === "asked" && (
        <p className="border-t px-4 py-3 text-xs text-muted-foreground">One search, nothing rewritten.</p>
      )}
    </li>
  );
}

const LIVE = ["decompose", "multi", "hyde", "agent"] as const;

/** One technique, now, on Remote AI: the same pipeline Ask uses. */
function LiveRun({ question, profile, docs }: { question: string; profile: string; docs: DocumentFile[] | undefined }) {
  const { data: rerankers } = useSWR<RerankersResponse>("/rerankers");
  const can = rerankers?.rewrite;
  const [technique, setTechnique] = useState<(typeof LIVE)[number]>("multi");
  // A live result counts only for the question, technique and index it was asked with.
  const next = `/learn/rewrite/live?${new URLSearchParams({ q: question, technique, profile })}`;
  const [asked, setAsked] = useState<string | null>(null);
  const live = useSWR<RewriteLive>(asked === next ? asked : null, { revalidateOnFocus: false });

  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">Run it live</CardTitle>
        <CardDescription>
          The same question, searched now with one technique on Remote AI. A rewrite made before for these exact words
          is replayed from memory, so the result matches the runs above.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {can && !can.ok ? (
          <p className="text-xs text-pretty text-muted-foreground">{can.reason}</p>
        ) : (
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="live-technique" className="sr-only">
              Technique
            </label>
            <Select
              value={technique}
              items={Object.fromEntries(LIVE.map((x) => [x, NAMES[x]]))}
              onValueChange={(v) => v && setTechnique(v as (typeof LIVE)[number])}
            >
              <SelectTrigger id="live-technique" size="sm" className="w-56 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {LIVE.map((x) => (
                  <SelectItem key={x} value={x}>
                    {NAMES[x]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button size="sm" variant="outline" disabled={!can?.ok || live.isLoading} onClick={() => setAsked(next)}>
              {live.isLoading ? <Loader2 className="animate-spin" aria-hidden /> : <Play aria-hidden />}
              Run live
            </Button>
          </div>
        )}
        {live.error && <ErrorState message={live.error.message} error={live.error} onRetry={() => void live.mutate()} />}
        {live.isLoading && (
          <p className="text-xs text-muted-foreground" role="status">
            Asking Remote AI and searching. A few seconds; multi-step search can take up to a minute and a half.
          </p>
        )}
        {live.data && (
          <div className="space-y-3">
            <p className="text-xs tabular-nums">
              <span className="text-muted-foreground">First answer page: </span>
              {live.data.targets.length === 0
                ? "answer pages unknown"
                : live.data.first_answer_rank === null
                  ? "not in the top 50"
                  : `rank ${live.data.first_answer_rank}`}
            </p>
            <WhatHappened steps={live.data.steps} />
            <ol className="space-y-1">
              {live.data.hits.slice(0, 5).map((h) => (
                <li key={h.chunk_id} className="flex items-baseline gap-2 text-xs">
                  <span className="font-semibold tabular-nums">{h.rank}</span>
                  <span className="min-w-0 truncate">{docTitle(h.doc_id, docs)}</span>
                  <span className="shrink-0 text-muted-foreground tabular-nums">p. {h.page}</span>
                  {h.answer_page && (
                    <Badge className="h-4 gap-1 bg-sky-600 px-1.5 text-[10px] font-normal text-white dark:bg-sky-500 dark:text-black">
                      <Target className="size-2.5" aria-hidden />
                      Answer page
                    </Badge>
                  )}
                </li>
              ))}
            </ol>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
