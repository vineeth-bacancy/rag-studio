import { Cloud, Laptop, Loader2, Square } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { StatusBadge } from "@/components/common/status";
import { TechnicalDetails } from "@/components/common/technical-details";
import { MatchCheckPanel } from "@/components/indexes/match-check";
import type { BuildJob, Can, IndexProfileRow } from "@/lib/api";
import { dateTimeLabel, NO_DATA, num, relativeTime } from "@/lib/format";

const STAGES: Record<string, string> = {
  start: "Starting",
  clean: "Reading the cleaned documents",
  docling_start: "Docling is converting a document",
  docling_done: "Docling finished a document",
  parse: "Documents read",
  sentences: "Finding sentences",
  chunk: "Passages cut",
  embed_start: "Starting to embed",
  embed: "Embedding",
  word_index: "Building the word index",
  up_to_date: "Already up to date",
  done: "Done",
};

export const where = (on: string | undefined) => (on === "colab" ? "on the Colab GPU" : "on this computer");

function minutes(m: number | null | undefined): string {
  if (m == null) return "";
  return m < 1 ? " · under a minute" : ` · about ${Math.round(m)} min`;
}

/** A running build: what it is doing, how far, and a way to stop it. */
export function BuildProgress({ job, onCancel }: { job: BuildJob; onCancel: () => void }) {
  const pct = job.stage === "embed" && job.total ? Math.round((100 * job.done) / job.total) : null;
  return (
    <div className="space-y-2 rounded-lg bg-sky-500/5 p-3 ring-1 ring-sky-500/30" role="status" aria-live="polite">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex items-center gap-1.5 text-sm font-medium text-sky-800 dark:text-sky-300">
          <Loader2 className="size-4 animate-spin" aria-hidden />
          Building {where(job.on)}
        </p>
        <Button size="sm" variant="ghost" onClick={onCancel}>
          <Square className="size-3.5" aria-hidden />
          Cancel
        </Button>
      </div>
      <Progress value={pct} aria-label="Build progress">
        <span className="text-xs text-muted-foreground">
          {STAGES[job.stage] ?? job.stage}
          {job.stage === "embed" && job.total != null && (
            <>
              : <span className="tabular-nums">{num(job.done)} of {num(job.total)}</span> passages
            </>
          )}
          {job.cached ? ` (${num(job.cached)} already embedded earlier)` : ""}
        </span>
      </Progress>
      <p className="text-xs text-muted-foreground">
        Started {relativeTime(job.started_at)}. Stopping keeps every vector made so far; building again resumes.
      </p>
    </div>
  );
}

function EndedJob({ job }: { job: BuildJob }) {
  if (job.status === "done" || job.status === "running") return null;
  const title = { failed: "The last build failed", cancelled: "The last build was cancelled", interrupted: "The last build stopped" }[
    job.status
  ];
  return (
    <Alert className={job.status === "failed" ? "border-red-500/40" : undefined}>
      <AlertTitle className="text-sm">{title}</AlertTitle>
      <AlertDescription className="text-xs break-words text-pretty">
        {job.error} <span className="text-muted-foreground">({relativeTime(job.finished_at ?? job.started_at)})</span>
      </AlertDescription>
    </Alert>
  );
}

function BuildButton({
  can, on, building, label, estimate, onBuild,
}: {
  can: Can & { minutes?: number | null };
  on: "local" | "colab";
  building: boolean;
  label: string;
  /** Only for a first build: a rebuild reuses every unchanged vector and takes seconds. */
  estimate: boolean;
  onBuild: (on: "local" | "colab") => void;
}) {
  const Icon = on === "colab" ? Cloud : Laptop;
  return (
    <div className="min-w-0 space-y-1">
      <Button size="sm" variant={on === "colab" ? "default" : "outline"} disabled={!can.ok || building} onClick={() => onBuild(on)}>
        <Icon className="size-4" aria-hidden />
        {label}
        {can.ok && estimate && <span className="font-normal opacity-80">{minutes(can.minutes)}</span>}
      </Button>
      {!can.ok && can.reason && (
        <p className="text-xs text-pretty text-muted-foreground">
          {can.reason}
          {can.fix && (
            <>
              {" "}
              <code className="rounded bg-muted px-1 font-mono">{can.fix}</code>
            </>
          )}
        </p>
      )}
    </div>
  );
}

/**
 * One index profile of ladder row 5: one parser's passages embedded by one
 * model. Everything it shows was decided by the engine (rag_studio/indexes.py):
 * whether it can build here or on Colab, and whether a question can be
 * embedded for it right now, each with the reason when not.
 */
export function IndexCard({
  profile: p,
  title,
  buildActive,
  checking,
  onBuild,
  onCancel,
  onCheck,
}: {
  profile: IndexProfileRow;
  title: string;
  buildActive: boolean;
  checking: boolean;
  onBuild: (on: "local" | "colab") => void;
  onCancel: () => void;
  onCheck: () => void;
}) {
  const m = p.active;
  const e = p.embedding;
  const job = p.job;
  const running = job?.status === "running";
  const can = p.can;
  const builds = e?.build_on ?? "local";

  return (
    <section aria-label={p.label} className="flex min-w-0 flex-col gap-3 rounded-xl bg-card p-4 ring-1 ring-foreground/10">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        {running ? (
          <StatusBadge status="running" label="Building" />
        ) : m ? (
          can?.query.ok ? (
            <StatusBadge status="healthy" label="Ready to ask" />
          ) : (
            <StatusBadge status="warning" label="Built, can't ask now" />
          )
        ) : (
          <StatusBadge status="info" label="Not built" />
        )}
      </div>

      <dl className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-3 gap-y-1 text-xs">
        <dt className="text-muted-foreground">Passages</dt>
        <dd className="tabular-nums">{m ? num(m.rows) : NO_DATA}</dd>
        <dt className="text-muted-foreground">Built</dt>
        <dd className="text-pretty">
          {m ? (
            <>
              {dateTimeLabel(m.built_at)}, {where(m.embedding.built_on)}
              {m.timings?.total_s != null && <span className="text-muted-foreground"> in {Math.round(m.timings.total_s / 60)} min</span>}
            </>
          ) : (
            NO_DATA
          )}
        </dd>
        <dt className="text-muted-foreground">Questions</dt>
        <dd className="text-pretty">
          {e?.query_on === "colab" ? "Embedded on the Colab GPU" : "Embedded on this computer"}
          {m && can && !can.query.ok && <span className="block text-amber-700 dark:text-amber-400">{can.query.reason}</span>}
        </dd>
      </dl>

      {running && job ? <BuildProgress job={job} onCancel={onCancel} /> : job && <EndedJob job={job} />}

      {can && !running && (
        <div className="flex flex-wrap items-start gap-x-3 gap-y-2">
          {builds !== "colab" && (
            <BuildButton can={can.build_local} on="local" building={buildActive} label={m ? "Rebuild here" : "Build here"} estimate={!m} onBuild={onBuild} />
          )}
          {builds !== "local" && (
            <BuildButton can={can.build_colab} on="colab" building={buildActive} label={m ? "Rebuild on Colab" : "Build on Colab"} estimate={!m} onBuild={onBuild} />
          )}
        </div>
      )}

      {m && (
        <MatchCheckPanel
          check={p.check}
          running={checking}
          disabledReason={buildActive ? "Wait for the build to finish." : null}
          onRun={onCheck}
        />
      )}

      <TechnicalDetails
        className="mt-auto"
        rows={{
          Profile: p.id,
          Model: e ? `${e.model}${e.revision ? ` @ ${e.revision.slice(0, 12)}` : ""}` : NO_DATA,
          Dimensions: e ? num(e.dimensions) : NO_DATA,
          "Passage prefix": e ? JSON.stringify(e.doc_prefix) : NO_DATA,
          "Question prefix": e ? JSON.stringify(e.query_prefix) : NO_DATA,
          ...(e?.options ? { Options: JSON.stringify(e.options) } : {}),
          ...(m
            ? {
                Table: m.table,
                Digest: m.embedding.digest.split(":").pop()?.slice(0, 12) ?? NO_DATA,
                Runtime: [m.embedding.backend, m.embedding.ollama_version ?? m.embedding.torch, m.embedding.gpu]
                  .filter(Boolean)
                  .join(" · "),
                Vectors: `${num(m.embedded ?? null)} embedded, ${num(m.reused ?? null)} kept from the last build, ${num(m.from_cache ?? null)} already embedded earlier`,
              }
            : {}),
        }}
      />
    </section>
  );
}
