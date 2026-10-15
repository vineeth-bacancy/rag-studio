"use client";

import { Suspense, useEffect } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import useSWR from "swr";
import { ClipboardCheck, GitCompareArrows } from "lucide-react";

import { PageHeader } from "@/components/common/page-header";
import { CardSkeleton, EmptyState, ErrorState } from "@/components/common/states";
import { RunCompare } from "@/components/eval/compare";
import { Ladder, type LadderRow } from "@/components/eval/ladder";
import { RunDetail } from "@/components/eval/run-detail";
import { RunHistory } from "@/components/eval/run-history";
import { StartRun } from "@/components/eval/start-run";
import type { AppStatus, DocumentFile, GoldenSet, Run, RunBrief } from "@/lib/api";

export default function EvaluationPage() {
  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <PageHeader href="/evaluation" />
      {/* The selected run lives in the URL (?run=), so a run can be linked
          and the back button closes it. useSearchParams needs a Suspense
          boundary to keep the rest of the page prerendered. */}
      <Suspense fallback={<CardSkeleton height="h-64" />}>
        <Evaluation />
      </Suspense>
    </div>
  );
}

function Evaluation() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const selected = params.get("run");
  // The compared pair also lives in the URL (?a=&b=), so a comparison can be linked.
  const compareA = params.get("a");
  const compareB = params.get("b");

  const { data: status } = useSWR<AppStatus>("/status", { refreshInterval: 30_000 });
  const { data: golden, error: goldenError } = useSWR<{ sets: GoldenSet[] }>("/golden");
  const { data: docs } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const runsKey = "/eval/runs";
  const { data: runsData, error: runsError, mutate: refreshRuns } = useSWR<{ runs: RunBrief[]; active: boolean }>(
    runsKey,
    // Poll while a run is going, so its row and the progress bar move.
    { refreshInterval: (d) => (d?.active ? 2000 : 0) },
  );
  const { data: ladder, mutate: refreshLadder } = useSWR<{ rows: LadderRow[] }>("/eval/ladder");
  const { data: run } = useSWR<Run>(selected ? `/eval/runs/${selected}` : null, {
    refreshInterval: (d) => (d?.status === "running" ? 2000 : 0),
  });

  const active = runsData?.active ?? false;
  // When a run finishes, the ladder may have a new row.
  useEffect(() => {
    if (!active) void refreshLadder();
  }, [active, refreshLadder]);

  const select = (id: string | null) => {
    const q = new URLSearchParams(params.toString());
    if (id) q.set("run", id);
    else q.delete("run");
    router.push(q.size ? `${pathname}?${q}` : pathname, { scroll: false });
  };

  const compare = (a: string | null, b: string | null, reveal = false) => {
    const q = new URLSearchParams(params.toString());
    for (const [key, value] of [["a", a], ["b", b]] as const) {
      if (value) q.set(key, value);
      else q.delete(key);
    }
    router.replace(q.size ? `${pathname}?${q}` : pathname, { scroll: false });
    if (reveal) document.getElementById("run-compare")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const finishedRuns = runsData?.runs.filter((r) => r.status === "done").length ?? 0;

  if (goldenError || runsError) {
    return <ErrorState message={(goldenError ?? runsError).message} error={goldenError ?? runsError} onRetry={() => void refreshRuns()} />;
  }
  if (!golden || !runsData) {
    return (
      <div className="space-y-4">
        <CardSkeleton height="h-48" />
        <CardSkeleton height="h-64" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {selected && run && <RunDetail run={run} docs={docs?.documents} onClose={() => select(null)} />}

      {status && !status.indexed ? (
        <EmptyState
          icon={ClipboardCheck}
          title="No search index yet"
          description="A run searches the index for every test question, so the index comes first."
          hint={
            <>
              Build it with <code className="rounded bg-muted px-1 py-0.5 font-mono">./rag ingest</code>.
            </>
          }
        />
      ) : (
        <StartRun
          sets={golden.sets}
          runs={runsData.runs}
          active={active}
          onStarted={(id) => {
            void refreshRuns();
            select(id);
          }}
        />
      )}

      {ladder && <Ladder rows={ladder.rows} />}

      {finishedRuns >= 2 ? (
        <RunCompare runs={runsData.runs} sets={golden.sets} a={compareA} b={compareB} onPick={(a, b) => compare(a, b)} />
      ) : runsData.runs.length > 0 ? (
        <EmptyState
          icon={GitCompareArrows}
          title="Nothing to compare yet"
          description="Comparing needs two finished runs: one before a change and one after it."
        />
      ) : null}

      {runsData.runs.length > 0 ? (
        <RunHistory runs={runsData.runs} selected={selected} onCompare={(a, b) => compare(a, b, true)} />
      ) : (
        <EmptyState
          icon={ClipboardCheck}
          title="No evaluation runs yet"
          description="A run asks every test question with one setup and scores each answer as Pass, Refused or Wrong. Comparing runs is how each upgrade earns its place."
        />
      )}
    </div>
  );
}
