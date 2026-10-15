"use client";

import { useState } from "react";
import useSWR from "swr";
import { Shuffle, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { TechnicalDetails } from "@/components/common/technical-details";
import type { EmbedLab as EmbedLabData, EmbedHit, IndexesResponse } from "@/lib/api";
import { bytes, num } from "@/lib/format";
import { cn } from "@/lib/utils";
import { docColor, EmbedMap } from "./embed-map";

const QUESTIONS = [
  "What are the main risks to the global outlook?",
  "¿Cuál es el pronóstico de crecimiento mundial para 2026?",
  "How much recycled aluminum is in the iMac stand?",
];
const DEFAULT_ORDER = ["docling-structure-embeddinggemma2", "docling-structure-qwen3-0.6b", "docling-structure", "baseline"];

/** The exact text sent to the model, with the profile's prefix marked. */
function Sent({ prefix, text }: { prefix: string; text: string }) {
  return (
    <code className="block rounded-md bg-muted px-2.5 py-2 font-mono text-xs break-words whitespace-pre-wrap">
      {prefix && <mark className="rounded-sm bg-chart-1/15 px-0.5 text-chart-1">{prefix.replace(/\n/g, "↵\n")}</mark>}
      {text}
    </code>
  );
}

function HitList({
  data, hits, focus, onFocus, kept,
}: {
  data: EmbedLabData;
  hits: EmbedHit[];
  focus?: string | null;
  onFocus?: (id: string | null) => void;
  kept?: Set<string>;
}) {
  return (
    <ol className="divide-y text-xs">
      {hits.map((h) => (
        <li
          key={h.id}
          tabIndex={onFocus ? 0 : undefined}
          onMouseEnter={() => onFocus?.(h.id)}
          onMouseLeave={() => onFocus?.(null)}
          onFocus={() => onFocus?.(h.id)}
          onBlur={() => onFocus?.(null)}
          className={cn(
            "grid grid-cols-[1.25rem_minmax(0,1fr)_3.5rem] gap-x-2 rounded-sm py-2 outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50",
            focus === h.id && "bg-muted/60",
          )}
        >
          <span className="font-semibold tabular-nums">{h.rank}</span>
          <div className="min-w-0 space-y-0.5">
            <p className="flex items-center gap-1.5 font-medium">
              <span className="size-2 shrink-0 rounded-full" style={{ background: docColor(data, h.doc) }} aria-hidden />
              <span className="truncate">{data.documents.find((d) => d.doc_id === h.doc)?.title ?? h.doc}</span>
              <span className="shrink-0 text-muted-foreground">p. {h.page}</span>
              {kept && (kept.has(h.id) ? (
                <span className="shrink-0 text-muted-foreground">· also in the right top 5</span>
              ) : null)}
            </p>
            <p className="line-clamp-2 text-pretty text-muted-foreground">{h.text}</p>
          </div>
          <span className="text-right font-mono tabular-nums">{h.score.toFixed(3)}</span>
        </li>
      ))}
    </ol>
  );
}

function Dots({ kept, of }: { kept: number; of: number }) {
  return (
    <span className="inline-flex gap-1" aria-label={`${kept} of ${of}`}>
      {Array.from({ length: of }, (_, i) => (
        <span
          key={i}
          className={cn("size-2.5 rounded-full ring-1", i < kept ? "bg-foreground ring-foreground" : "ring-foreground/30")}
          aria-hidden
        />
      ))}
    </span>
  );
}

/**
 * Step 5, embed: one index's passages as points, a question landing among
 * them, Matryoshka cuts, and what happens when two models' vectors mix. All
 * computed by the engine (rag_studio/learn.py `embed_lab`) on the real index.
 */
export function EmbedLab() {
  const { data: indexes } = useSWR<IndexesResponse>("/indexes");
  const [profile, setProfile] = useState<string | null>(null);
  const [draft, setDraft] = useState<string | null>(null);
  const [asked, setAsked] = useState<string>(QUESTIONS[0]);
  const [mix, setMix] = useState<string | null>(null);
  const [focus, setFocus] = useState<string | null>(null);

  const built = (indexes?.profiles ?? []).filter((p) => p.active && p.can?.query.ok !== false);
  const chosen =
    built.find((p) => p.id === profile) ?? DEFAULT_ORDER.map((id) => built.find((p) => p.id === id)).find(Boolean) ?? built[0];
  const model = chosen?.embedding?.model;
  // One index per other model, preferring the same passages: mixing is about
  // the model, so keep everything else equal where possible.
  const otherModels = [...new Set(built.map((p) => p.embedding?.model ?? p.id))].filter((m) => m !== model);
  const others = otherModels.map((m) => {
    const same = built.filter((p) => (p.embedding?.model ?? p.id) === m);
    return same.find((p) => p.parser === chosen?.parser && p.chunker === chosen?.chunker) ?? same[0];
  });
  const mixChoice = others.find((p) => p.id === mix) ?? null;

  const key = chosen
    ? `/learn/embed?${new URLSearchParams({ profile: chosen.id, q: asked, ...(mixChoice ? { mix: mixChoice.id } : {}) })}`
    : null;
  const { data, error, isLoading, mutate } = useSWR<EmbedLabData>(key, { keepPreviousData: true });
  // keepPreviousData: draw only a result for the current choice (CLAUDE.md, M3 UI gotcha).
  const current = data && chosen && data.profile === chosen.id ? data : null;

  if (!indexes) return <CardSkeleton height="h-96" />;
  const text = draft ?? asked;
  const q = current?.question;
  const topIds = new Set(q?.nearest.map((h) => h.id) ?? []);

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle as="h2">Embedding lab</CardTitle>
          <CardDescription className="text-pretty">
            Pick an index, ask a question, and see where its vector lands among every passage.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <form
            className="flex flex-col gap-2 sm:flex-row"
            onSubmit={(e) => {
              e.preventDefault();
              if (text.trim()) {
                setAsked(text.trim());
                setDraft(null);
              }
            }}
          >
            <label htmlFor="embed-question" className="sr-only">
              Question
            </label>
            <Input
              id="embed-question"
              value={text}
              maxLength={500}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Type a question..."
              className="flex-1"
            />
            <Button type="submit" disabled={!text.trim()}>
              <Sparkles aria-hidden />
              Embed
            </Button>
          </form>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Questions to try">
            {QUESTIONS.map((x) => (
              <button
                key={x}
                type="button"
                onClick={() => {
                  setAsked(x);
                  setDraft(null);
                }}
                aria-pressed={asked === x}
                className={cn(
                  "rounded-md px-2.5 py-1 text-left text-xs ring-1 ring-foreground/10 transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
                  asked === x && "bg-muted ring-foreground/40",
                )}
              >
                {x}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-end gap-x-5 gap-y-3">
            <div className="min-w-0 space-y-1">
              <label htmlFor="embed-index" className="text-xs font-medium text-muted-foreground">
                Index
              </label>
              <Select
                value={chosen?.id ?? null}
                items={Object.fromEntries(built.map((p) => [p.id, p.label]))}
                onValueChange={(v) => {
                  if (v) {
                    setProfile(v as string);
                    setMix(null);
                  }
                }}
              >
                <SelectTrigger id="embed-index" className="w-full max-w-80 sm:w-80">
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
            {chosen?.embedding && (
              <p className="pb-1 text-xs text-muted-foreground">
                {chosen.embedding.name} · {num(chosen.embedding.dimensions)} dimensions ·{" "}
                {chosen.embedding.query_on === "colab" ? "questions embedded on the Colab GPU" : "questions embedded on this computer"}
              </p>
            )}
          </div>
        </CardContent>
      </Card>

      {error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : !current ? (
        <div role="status" aria-busy="true" className="space-y-2">
          <CardSkeleton height="h-96" />
          <p className="text-xs text-muted-foreground">
            Embedding the question. The first question to a model that runs in this app loads it (about 20 s).
          </p>
        </div>
      ) : (
        <div className={cn("space-y-4", isLoading && "opacity-70")}>
          <Card>
            <CardHeader>
              <CardTitle as="h3" className="text-base">
                {num(current.rows)} passages, {num(current.dimensions)} dimensions each, seen from two
              </CardTitle>
              <CardDescription className="text-pretty">
                The two axes keep {Math.round(current.explained * 100)}% of how the vectors differ (PCA), so nearby here
                is usually, not always, nearby for the search. The dashed lines go to the question&apos;s true five nearest
                passages, measured in all {num(current.dimensions)} dimensions.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label="Documents">
                {current.documents.map((d) => (
                  <li key={d.doc_id} className="flex items-center gap-1.5">
                    <span className="size-2.5 rounded-full" style={{ background: docColor(current, d.doc_id) }} aria-hidden />
                    {d.title}
                    <span className="text-muted-foreground tabular-nums">({num(current.points.filter((p) => p.doc === d.doc_id).length)})</span>
                  </li>
                ))}
                {q && (
                  <li className="flex items-center gap-1.5">
                    <span className="size-3 rounded-full border-2 border-foreground bg-background" aria-hidden />
                    Your question
                  </li>
                )}
              </ul>
              <EmbedMap data={current} focus={focus} onFocus={setFocus} />
            </CardContent>
          </Card>

          {q && (
            <div className="grid gap-4 @container lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle as="h3" className="text-base">The five nearest passages</CardTitle>
                  <CardDescription className="text-pretty">
                    By cosine similarity with the question, in all dimensions: what dense search returns. Hover one to
                    find it on the map.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-1">
                    <p className="text-xs font-medium">What the model read</p>
                    <Sent prefix={current.query_prefix} text={q.text} />
                    <p className="text-xs text-pretty text-muted-foreground">
                      {current.query_prefix
                        ? "The highlighted prefix is the model's own instruction for questions; passages get "
                        : "This model takes questions as they are; passages get "}
                      {current.doc_prefix ? (
                        <code className="rounded bg-muted px-1 font-mono">{JSON.stringify(current.doc_prefix)}</code>
                      ) : (
                        "no prefix"
                      )}
                      . Embedded in {num(q.embed_ms)} ms {current.embedded_on === "remote" ? "on the Colab GPU" : "on this computer"}.
                    </p>
                  </div>
                  <HitList data={current} hits={q.nearest} focus={focus} onFocus={setFocus} />
                </CardContent>
              </Card>

              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle as="h3" className="text-base">Matryoshka: shorter vectors</CardTitle>
                    <CardDescription className="text-pretty">
                      Some models are trained so that the first dimensions carry the most meaning, like nested dolls:
                      cut every vector short and search still mostly works, in a fraction of the space.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {q.matryoshka.length ? (
                      <table className="w-full text-xs">
                        <thead className="text-left text-muted-foreground">
                          <tr>
                            <th className="pb-1 font-medium">Dimensions</th>
                            <th className="pb-1 font-medium">Index size</th>
                            <th className="pb-1 font-medium">Same top 5</th>
                          </tr>
                        </thead>
                        <tbody>
                          {q.matryoshka.map((m) => (
                            <tr key={m.dimensions} className="border-t">
                              <td className="py-1.5 tabular-nums">{num(m.dimensions)}</td>
                              <td className="py-1.5 tabular-nums">{bytes(m.bytes)}</td>
                              <td className="py-1.5">
                                <span className="flex items-center gap-2">
                                  <Dots kept={m.kept} of={5} />
                                  <span className="tabular-nums text-muted-foreground">{m.kept} of 5</span>
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p className="text-xs text-pretty text-muted-foreground">
                        Not shown for {current.model_name}: which version Ollama serves is not recorded, so whether it
                        was trained for shorter vectors is unknown. Pick a Qwen3-Embedding or EmbeddingGemma 2 index.
                      </p>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle as="h3" className="flex items-center gap-2 text-base">
                      <Shuffle className="size-4 text-muted-foreground" aria-hidden />
                      Why indexes never mix models
                    </CardTitle>
                    <CardDescription className="text-pretty">
                      Embed the same question with another model and search this index with it. The app always refuses
                      this; here you can see why.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {others.length === 0 ? (
                      <p className="text-xs text-muted-foreground">Build an index with another model to try this.</p>
                    ) : (
                      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Embed the question with">
                        {others.map((o) => (
                          <button
                            key={o.id}
                            type="button"
                            onClick={() => setMix(mix === o.id ? null : o.id)}
                            aria-pressed={mix === o.id}
                            className={cn(
                              "rounded-md px-2.5 py-1 text-xs ring-1 ring-foreground/10 transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
                              mix === o.id && "bg-muted ring-foreground/40",
                            )}
                          >
                            Ask with {o.embedding?.name} ({num(o.embedding?.dimensions ?? null)})
                          </button>
                        ))}
                      </div>
                    )}
                    {current.mix && mixChoice && current.mix.profile === mixChoice.id && (
                      current.mix.possible ? (
                        <div className="space-y-2">
                          <p className="text-xs text-pretty">
                            Same {num(current.dimensions)} dimensions, so the arithmetic runs, but{" "}
                            <strong>{current.mix.kept} of 5</strong> passages match the right answer, and every cosine is
                            near {current.mix.nearest[0]?.score.toFixed(2)}: the two models put meaning in unrelated
                            directions.
                          </p>
                          <HitList data={current} hits={current.mix.nearest} kept={topIds} />
                        </div>
                      ) : (
                        <p className="text-xs text-pretty">{current.mix.reason}</p>
                      )
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          <TechnicalDetails
            rows={{
              Profile: current.profile,
              Model: current.model,
              "Question prefix": JSON.stringify(current.query_prefix),
              "Passage prefix": JSON.stringify(current.doc_prefix),
              Projection: `PCA of the unit vectors, 2 components, ${Math.round(current.explained * 1000) / 10}% of the variance`,
              Search: "Exact cosine over every stored vector (numpy), the same ranking as the index",
            }}
          />
        </div>
      )}
    </div>
  );
}
