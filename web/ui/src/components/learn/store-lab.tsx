"use client";

import { useState } from "react";
import useSWR from "swr";
import { Gauge, Loader2 } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { TechnicalDetails } from "@/components/common/technical-details";
import { Tile, TileGrid } from "@/components/common/tiles";
import { api, type IndexesResponse, type StoreFacts } from "@/lib/api";
import { NO_DATA, bytes, dateTimeLabel, num } from "@/lib/format";
import { cn } from "@/lib/utils";

/** The word index settings, in words. Keys are LanceDB's FTS options as the engine records them. */
const SETTING_WORDS: Record<string, (v: unknown) => string> = {
  base_tokenizer: (v) => `splits text into words with the ${String(v)} tokenizer`,
  stem: (v) => (v ? "reduces words to their stem (\"forecasts\" matches \"forecast\")" : "keeps words as written"),
  remove_stop_words: (v) => (v ? "drops common words" : "keeps common words (\"the\", \"of\"): BM25 already gives them almost no weight"),
  lower_case: (v) => (v ? "ignores case" : "matches case"),
  ascii_folding: (v) => (v ? "folds accents (\"é\" matches \"e\")" : "keeps accents"),
  with_position: (v) => (v ? "records word positions, so a phrase can be searched" : "records no positions"),
};

/**
 * Step 6: an index as LanceDB keeps it. The table, its columns and one real
 * row, its versions and indexes, every kept build, and a measurement of what
 * an approximate (ANN) index would cost in recall on these vectors.
 */
export function StoreLab() {
  const { data: indexes } = useSWR<IndexesResponse>("/indexes");
  const [profile, setProfile] = useState("baseline");
  const { data, error, isLoading, mutate } = useSWR<StoreFacts>(`/learn/store?profile=${encodeURIComponent(profile)}`, {
    keepPreviousData: true,
  });
  const built = (indexes?.profiles ?? []).filter((p) => p.active);

  if (!indexes) return <CardSkeleton height="h-96" />;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle as="h2">Inside an index</CardTitle>
          <CardDescription>
            One LanceDB table per build: a row per passage, its vector beside its text, and a word index for BM25.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="space-y-1">
            <label htmlFor="store-index" className="text-xs font-medium text-muted-foreground">
              Index
            </label>
            <Select
              value={profile}
              items={Object.fromEntries(built.map((p) => [p.id, p.label]))}
              onValueChange={(v) => v && setProfile(v as string)}
            >
              <SelectTrigger id="store-index" className="w-full max-w-80 sm:w-80">
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

          {error ? (
            <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
          ) : !data ? (
            <CardSkeleton height="h-64" />
          ) : (
            <div className={cn("space-y-6 @container", isLoading && "opacity-70")}>
              <TileGrid className="grid-cols-2 @2xl:grid-cols-4">
                <Tile label="Passages (rows)" value={num(data.rows)} />
                <Tile label="Dimensions" value={num(data.dimensions)} hint="numbers per vector" />
                <Tile label="Size on disk" value={bytes(data.bytes)} hint="text, vectors and indexes" />
                <Tile label="Versions" value={num(data.versions.length)} />
              </TileGrid>

              <div className="grid gap-6 @4xl:grid-cols-2">
                <SampleRow data={data} />
                <Columns data={data} />
              </div>

              <div className="grid gap-6 @4xl:grid-cols-2">
                <Versions data={data} />
                <WordIndex data={data} />
              </div>

              <Builds data={data} />
            </div>
          )}
        </CardContent>
      </Card>

      {data && <ExactOrApproximate profile={profile} data={data} onMeasured={() => mutate()} />}
    </div>
  );
}

function SampleRow({ data }: { data: StoreFacts }) {
  const head = data.sample.vector_head;
  return (
    <section className="min-w-0 space-y-2" aria-labelledby="store-sample">
      <h3 id="store-sample" className="text-sm font-medium">
        One row
      </h3>
      <dl className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-xs">
        <dt className="text-muted-foreground">chunk_id</dt>
        <dd className="font-mono break-all">{data.sample.chunk_id}</dd>
        <dt className="text-muted-foreground">label</dt>
        <dd className="font-mono whitespace-pre-wrap">{data.sample.label ?? NO_DATA}</dd>
        <dt className="text-muted-foreground">text</dt>
        <dd className="line-clamp-4 text-pretty">{data.sample.text}</dd>
        <dt className="text-muted-foreground">vector</dt>
        <dd className="font-mono break-all tabular-nums">
          [{head.map((v) => v.toFixed(4)).join(", ")}, … {num(data.dimensions - head.length)} more]
        </dd>
      </dl>
      <p className="text-xs text-pretty text-muted-foreground">
        The vector is what dense search compares; the text and label are what BM25 and the model read.
      </p>
    </section>
  );
}

function Columns({ data }: { data: StoreFacts }) {
  return (
    <section className="min-w-0 space-y-2" aria-labelledby="store-columns">
      <h3 id="store-columns" className="text-sm font-medium">
        Columns ({num(data.columns.length)})
      </h3>
      <div className="max-h-64 overflow-y-auto rounded-lg ring-1 ring-foreground/10">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-card">
            <tr className="text-left text-muted-foreground">
              <th className="px-3 py-1.5 font-medium">Column</th>
              <th className="px-3 py-1.5 font-medium">Type</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {data.columns.map((c) => (
              <tr key={c.name}>
                <td className="px-3 py-1 font-mono">{c.name}</td>
                <td className="px-3 py-1 font-mono break-all text-muted-foreground">{c.type}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/** What each version of the table is. LanceDB never edits in place: every change is a new version. */
function Versions({ data }: { data: StoreFacts }) {
  const wordAt = data.word_index?.table_version ?? null;
  return (
    <section className="min-w-0 space-y-2" aria-labelledby="store-versions">
      <h3 id="store-versions" className="text-sm font-medium">
        Versions
      </h3>
      <ol className="space-y-1.5 text-xs">
        {data.versions.map((v, i) => (
          <li key={v.version} className="flex flex-wrap items-baseline gap-x-2">
            <span className="font-mono tabular-nums">v{v.version}</span>
            <span>
              {i === 0
                ? "Created: every passage, its vector and its metadata"
                : v.version === wordAt
                  ? "Word index added"
                  : "Changed"}
            </span>
            <span className="text-muted-foreground tabular-nums">
              · {num(v.rows)} rows · {dateTimeLabel(v.at)}
            </span>
          </li>
        ))}
      </ol>
      {data.versions.length > 1 && (
        <p className="text-xs text-pretty text-muted-foreground">
          Every version has the same {num(data.rows)} rows: adding the word index changed no passage and no vector,
          so runs made before it still describe this build.
        </p>
      )}
    </section>
  );
}

function WordIndex({ data }: { data: StoreFacts }) {
  const settings = data.word_index?.settings ?? null;
  return (
    <section className="min-w-0 space-y-2" aria-labelledby="store-word-index">
      <h3 id="store-word-index" className="text-sm font-medium">
        Indexes
      </h3>
      {data.indexes.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No index: dense search compares the question with every vector. This build has no word index yet, so BM25
          and hybrid search are unavailable. Add it with ./rag ingest --profile {data.profile}.
        </p>
      ) : (
        <ul className="space-y-1 text-xs">
          {data.indexes.map((ix) => (
            <li key={ix.name} className="flex flex-wrap items-baseline gap-x-2">
              <Badge variant="outline" className="h-4 px-1.5 text-[10px] font-normal">
                {ix.type === "FTS" ? "Word index" : ix.type}
              </Badge>
              <span>
                on <span className="font-mono">{ix.columns.join(", ")}</span>
              </span>
              <span className="text-muted-foreground tabular-nums">
                · {num(ix.rows)} rows · {bytes(ix.bytes)}
              </span>
            </li>
          ))}
        </ul>
      )}
      {settings && (
        <ul className="list-disc space-y-0.5 pl-5 text-xs text-muted-foreground">
          <li>
            Searches{" "}
            {data.sample.label
              ? "the section label and the passage: exactly what was embedded"
              : "the passage text: exactly what was embedded (this index has no section labels)"}
          </li>
          {Object.entries(settings)
            .filter(([k]) => SETTING_WORDS[k])
            .map(([k, v]) => (
              <li key={k}>{SETTING_WORDS[k](v)}</li>
            ))}
        </ul>
      )}
      <p className="text-xs text-pretty text-muted-foreground">
        There is no vector index on purpose: dense search compares every vector, exactly.
      </p>
    </section>
  );
}

function Builds({ data }: { data: StoreFacts }) {
  return (
    <section className="space-y-2" aria-labelledby="store-builds">
      <h3 id="store-builds" className="text-sm font-medium">
        Builds of this index ({num(data.builds.length)})
      </h3>
      <ul className="space-y-1 text-xs">
        {data.builds.map((b) => (
          <li key={b.table} className="flex flex-wrap items-baseline gap-x-2">
            <span className="min-w-0 font-mono break-all">{b.table}</span>
            {b.active && (
              <Badge variant="secondary" className="h-4 px-1.5 text-[10px] font-normal">
                Active
              </Badge>
            )}
            <span className="text-muted-foreground tabular-nums">
              · {num(b.rows)} passages · built {dateTimeLabel(b.built_at)}
            </span>
          </li>
        ))}
      </ul>
      <p className="text-xs text-pretty text-muted-foreground">
        A rebuild writes a new table and switches to it only once it passes a check; the previous build is kept so
        it can be switched back.
      </p>
    </section>
  );
}

/**
 * Exact search compares the question with every vector; an approximate (ANN)
 * index compares it with a few groups of them. The measurement runs the test
 * questions against both, on a scratch copy of this build's vectors.
 */
function ExactOrApproximate({ profile, data, onMeasured }: { profile: string; data: StoreFacts; onMeasured: () => void }) {
  const [running, setRunning] = useState(false);
  const ann = data.ann;

  async function measure() {
    setRunning(true);
    try {
      await api.annCheck(profile);
      onMeasured();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "The measurement did not run. Try again.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="flex items-center gap-2">
          <Gauge className="size-4 text-muted-foreground" aria-hidden />
          Exact or approximate?
        </CardTitle>
        <CardDescription>
          An approximate index skips most vectors to answer faster, and can miss the best passages. At about{" "}
          {num(data.rows)} passages exact search takes a few milliseconds, so this app never trades recall for speed.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {ann ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-muted-foreground">
                  <th className="py-1 pr-3 font-medium">Search</th>
                  <th className="py-1 pr-3 text-right font-medium">Recall@{ann.k}</th>
                  <th className="py-1 pr-3 text-right font-medium">Median time</th>
                  <th className="py-1 text-right font-medium">Index built in</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {ann.results.map((r) => (
                  <tr key={r.index}>
                    <td className="py-1.5 pr-3">{r.index === "exact" ? "Exact (every vector)" : r.label}</td>
                    <td className="py-1.5 pr-3 text-right tabular-nums">{r.recall.toFixed(3)}</td>
                    <td className="py-1.5 pr-3 text-right tabular-nums">{r.median_ms.toFixed(1)} ms</td>
                    <td className="py-1.5 text-right tabular-nums">{r.index === "exact" ? NO_DATA : `${r.build_s.toFixed(2)} s`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="pt-2 text-xs text-pretty text-muted-foreground">
              Recall@{ann.k}: the share of exact search&apos;s top {ann.k} passages the index also returned, over{" "}
              {num(ann.queries)} test questions. Measured {dateTimeLabel(ann.measured_at)}.
            </p>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            Not measured for this build yet. The measurement builds three approximate indexes on a copy of its
            vectors; the index itself is not changed.
          </p>
        )}
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="outline" size="sm" onClick={measure} disabled={running}>
            {running ? <Loader2 className="animate-spin" aria-hidden /> : <Gauge aria-hidden />}
            {running ? "Measuring..." : ann ? "Measure again" : "Measure"}
          </Button>
          {running && (
            <p className="text-xs text-muted-foreground" role="status">
              Running: a few seconds, up to a minute when the AI is busy embedding.
            </p>
          )}
        </div>
        {ann && (
          <TechnicalDetails
            rows={{
              Table: ann.table,
              Queries: `${num(ann.queries)} test questions, embedded once`,
              "Approximate indexes": "IVF_FLAT, IVF_PQ and IVF_HNSW_SQ with LanceDB's defaults, cosine",
              LanceDB: ann.lancedb_version,
            }}
          />
        )}
      </CardContent>
    </Card>
  );
}
