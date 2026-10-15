"use client";

import { useState } from "react";
import useSWR from "swr";
import { Cloud, Database, Laptop, Loader2, MemoryStick } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/common/page-header";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { StatusBadge } from "@/components/common/status";
import { BuildProgress, IndexCard, where } from "@/components/indexes/index-card";
import { api, ApiError, type IndexProfileRow, type IndexesResponse } from "@/lib/api";
import { dateTimeLabel, NO_DATA, num } from "@/lib/format";

/**
 * The index profiles (spec §9.3): build each on this computer or on the
 * Colab GPU, follow the build, check that vectors match across computers.
 *
 * Ladder row 5 is drawn as model × parser: every card is one change from the
 * row-3 index of its column (the embedding model), which is what the
 * scorecard compares.
 */

const MODELS: { model: string; blurb: string }[] = [
  {
    model: "qwen3-embedding:0.6b",
    blurb: "The everyday upgrade: small enough for this computer, or seconds on the Colab GPU. Questions are embedded here.",
  },
  {
    model: "google/embeddinggemma-2",
    blurb: "Google's multilingual model, run inside this app (Ollama cannot run it on Linux). Questions are embedded here.",
  },
  {
    model: "qwen3-embedding:8b",
    blurb: "The largest. Built and asked on the Colab GPU only: asking needs Remote AI connected.",
  },
];

const PARSERS: { parser: string; title: string }[] = [
  { parser: "pymupdf", title: "PyMuPDF + structure" },
  { parser: "docling", title: "Docling + structure" },
];

function EarlierIndex({ p }: { p: IndexProfileRow }) {
  const m = p.active;
  return (
    <li className="grid gap-x-4 gap-y-1 py-3 first:pt-0 last:pb-0 @xl:grid-cols-[minmax(0,1fr)_7rem_9rem_9rem] @xl:items-center">
      <div className="min-w-0">
        <p className="text-sm font-medium">{p.label}</p>
        <p className="truncate font-mono text-xs text-muted-foreground">{p.id}</p>
      </div>
      <p className="text-xs tabular-nums @xl:text-right">{m ? `${num(m.rows)} passages` : NO_DATA}</p>
      <p className="text-xs text-muted-foreground @xl:text-right">{m ? dateTimeLabel(m.built_at) : "Not built"}</p>
      <div className="@xl:text-right">
        {m ? (
          p.can?.query.ok ? (
            <StatusBadge status="healthy" label="Ready to ask" />
          ) : (
            <StatusBadge status="warning" label="Can't ask now" />
          )
        ) : (
          <StatusBadge status="info" label="Not built" />
        )}
      </div>
    </li>
  );
}

export default function IndexesPage() {
  const { data, error, isLoading, mutate } = useSWR<IndexesResponse>("/indexes", {
    // Every 2 s while something builds, so its progress moves; quiet otherwise.
    refreshInterval: (latest) => (latest?.active_build ? 2000 : 0),
  });
  const [checking, setChecking] = useState<string | null>(null);

  const build = async (p: IndexProfileRow, on: "local" | "colab") => {
    try {
      await api.buildIndex(p.id, on);
      toast.success(`Building ${p.label} ${where(on)}`);
    } catch (e) {
      toast.error("The build did not start.", { description: e instanceof Error ? e.message : undefined });
    } finally {
      void mutate();
    }
  };

  const cancel = async (p: IndexProfileRow) => {
    try {
      await api.cancelBuild(p.id);
      toast.success("Stopping the build. Vectors made so far are kept.");
    } catch (e) {
      toast.error("Couldn't stop it here.", { description: e instanceof Error ? e.message : undefined });
    } finally {
      void mutate();
    }
  };

  const check = async (p: IndexProfileRow) => {
    setChecking(p.id);
    try {
      const r = await api.matchCheck(p.id);
      if (r.passed === null) toast.info("Checked on one computer: nothing to compare across.");
      else if (r.passed) toast.success("The vectors match across computers.");
      else toast.error("The vectors do not match across computers. See the check for which comparison fell short.");
    } catch (e) {
      toast.error("The check could not run.", { description: e instanceof ApiError ? e.message : undefined });
    } finally {
      setChecking(null);
      void mutate();
    }
  };

  if (error) {
    return (
      <div className="mx-auto max-w-5xl space-y-6">
        <PageHeader href="/indexes" />
        <ErrorState title="Couldn't load the indexes" message={error.message} error={error} onRetry={() => mutate()} />
      </div>
    );
  }

  const row5 = (data?.profiles ?? []).filter((p) => MODELS.some((m) => m.model === p.embedding?.model));
  const earlier = (data?.profiles ?? []).filter((p) => !row5.includes(p));
  const active = data?.active_build ?? null;
  const activeHere = active && active.status === "running" && !row5.some((p) => p.id === active.profile) ? active : null;

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <PageHeader href="/indexes" />

      {isLoading || !data ? (
        <CardSkeleton height="h-96" />
      ) : (
        <>
          <div className="flex flex-wrap gap-2 text-xs">
            {/* No host here: a tunnel hostname is ~50 characters and a badge never
                wraps (QA M4-03, 375 px). Settings shows it. */}
            <Badge variant="outline" className="gap-1.5 py-1 font-normal" title={data.remote?.host ?? undefined}>
              {data.remote?.connected ? <Cloud className="size-3.5" aria-hidden /> : <Laptop className="size-3.5" aria-hidden />}
              {data.remote?.connected ? "Remote AI connected" : "Remote AI not connected"}
            </Badge>
            <Badge variant="outline" className="gap-1.5 py-1 font-normal">
              <MemoryStick className="size-3.5" aria-hidden />
              {data.free_ram_gb != null ? `${data.free_ram_gb} GB of memory free` : "Free memory unknown"}
            </Badge>
          </div>

          {active && active.status === "running" && (
            <p className="flex items-start gap-2 text-sm text-pretty text-sky-800 dark:text-sky-300" role="status">
              <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin" aria-hidden />
              Building {active.label} {where(active.on)}. One build runs at a time; the other buttons wait for it.
            </p>
          )}

          <section aria-labelledby="row5" className="space-y-4">
            <div className="space-y-1">
              <h2 id="row5" className="text-lg font-semibold tracking-tight">New embedding models</h2>
              <p className="max-w-prose text-sm text-pretty text-muted-foreground">
                Ladder row 5: the two structure-aware indexes of row 3, each embedded by three newer models. Only the
                embedding model changes, so the scorecard can say what the model alone is worth.
              </p>
            </div>

            {MODELS.map(({ model, blurb }) => {
              const cards = PARSERS.map(({ parser, title }) => ({
                title,
                p: row5.find((x) => x.embedding?.model === model && x.parser === parser),
              }));
              const e = cards.find((c) => c.p)?.p?.embedding;
              if (!e) return null;
              return (
                <Card key={model} className="@container">
                  <CardHeader>
                    <CardTitle as="h3" className="flex flex-wrap items-center gap-2 text-base">
                      {e.name}
                      <Badge variant="secondary" className="font-normal tabular-nums">{num(e.dimensions)} dimensions</Badge>
                      <Badge variant="outline" className="gap-1 font-normal">
                        {e.build_on === "colab" ? <Cloud className="size-3" aria-hidden /> : <Laptop className="size-3" aria-hidden />}
                        {e.build_on === "colab" ? "Colab GPU" : e.build_on === "either" ? "Here or Colab" : "This computer"}
                      </Badge>
                    </CardTitle>
                    <CardDescription className="text-pretty">{blurb}</CardDescription>
                  </CardHeader>
                  <CardContent className="grid gap-4 @2xl:grid-cols-2">
                    {cards.map(({ title, p }) =>
                      p ? (
                        <IndexCard
                          key={p.id}
                          profile={p}
                          title={title}
                          buildActive={Boolean(active) && active?.profile !== p.id}
                          checking={checking === p.id}
                          onBuild={(on) => void build(p, on)}
                          onCancel={() => void cancel(p)}
                          onCheck={() => void check(p)}
                        />
                      ) : null,
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </section>

          <Card className="@container">
            <CardHeader>
              <CardTitle as="h2" className="flex items-center gap-2 text-base">
                <Database className="size-4 text-muted-foreground" aria-hidden />
                Earlier indexes
              </CardTitle>
              <CardDescription className="text-pretty">
                Ladder rows 1-4, all embedded by nomic-embed-text on this computer. Rebuild one with{" "}
                <code className="rounded bg-muted px-1 font-mono">./rag ingest --profile &lt;id&gt;</code>.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {activeHere && <BuildProgress job={activeHere} onCancel={() => void api.cancelBuild(activeHere.profile).catch(() => {})} />}
              <ul className="divide-y">
                {earlier.map((p) => (
                  <EarlierIndex key={p.id} p={p} />
                ))}
              </ul>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
