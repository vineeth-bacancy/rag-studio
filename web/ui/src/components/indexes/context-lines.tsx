import { useState } from "react";
import useSWR from "swr";
import { Loader2, PenLine, Square } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { StatusBadge } from "@/components/common/status";
import { ApiError, api, type ContextStatus } from "@/lib/api";
import { num, relativeTime } from "@/lib/format";

const JOB_WORDS: Record<string, string> = {
  done: "finished",
  failed: "stopped with an error",
  cancelled: "was cancelled",
  interrupted: "was interrupted",
};

/**
 * A `-ctx` index's context lines: how many passages have one, the last job,
 * and a button to write the missing ones on Remote AI (qwen3:8b). The lines
 * are kept as they arrive, so a stopped job resumes where it stopped; the
 * index can only be built once every passage has its line.
 */
export function ContextLines({ profile, onChanged }: { profile: string; onChanged: () => void }) {
  const key = `/indexes/${encodeURIComponent(profile)}/context`;
  const { data, error, mutate } = useSWR<ContextStatus>(key, {
    refreshInterval: (d) => (d?.job?.status === "running" ? 3000 : 0),
  });
  const [busy, setBusy] = useState(false);

  if (error) return <p className="text-xs text-destructive">{error.message}</p>;
  if (!data) return <p className="text-xs text-muted-foreground">Counting context lines...</p>;

  const job = data.job;
  const running = job?.status === "running";
  const missing = data.passages - data.written;
  const pct = data.passages ? Math.round((100 * (running && job?.total ? job.done : data.written)) / data.passages) : 0;

  const act = async (fn: () => Promise<unknown>, ok: string) => {
    setBusy(true);
    try {
      await fn();
      toast.success(ok);
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : "That did not work. Try again.");
    } finally {
      setBusy(false);
      void mutate();
      onChanged();
    }
  };

  return (
    <div className="space-y-2 rounded-lg bg-muted/30 p-3 ring-1 ring-foreground/10">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm font-medium">Context lines</p>
        {running ? (
          <StatusBadge status="running" label="Writing" />
        ) : missing === 0 ? (
          <StatusBadge status="healthy" label="Every passage" />
        ) : (
          <StatusBadge status="info" label={`${num(missing)} missing`} />
        )}
      </div>
      <Progress value={pct} aria-label="Context lines written">
        <span className="text-xs text-muted-foreground tabular-nums">
          {num(running && job?.total ? job.done : data.written)} of {num(data.passages)} passages have a line, written by{" "}
          {data.model} on Remote AI
        </span>
      </Progress>
      {job && !running && (
        <p className="text-xs text-muted-foreground">
          The last job {JOB_WORDS[job.status] ?? job.status} {relativeTime(job.finished_at ?? job.started_at)}
          {job.summary?.lines_per_s ? ` · ${job.summary.lines_per_s} lines a second` : ""}
          {job.status === "failed" && job.error ? `: ${job.error}` : "."}
        </p>
      )}
      {missing > 0 && (
        <p className="text-xs text-pretty text-muted-foreground">
          The index can be built once every passage has its line. Lines are kept as they arrive, so writing again
          resumes where it stopped.
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        {running ? (
          <Button size="sm" variant="ghost" disabled={busy} onClick={() => void act(() => api.cancelContext(profile), "Stopping after the current batch.")}>
            <Square className="size-3.5" aria-hidden />
            Cancel
          </Button>
        ) : (
          missing > 0 && (
            <Button
              size="sm"
              variant="outline"
              disabled={busy || !data.can.ok}
              title={data.can.ok ? undefined : data.can.reason ?? undefined}
              onClick={() => void act(() => api.writeContext(profile), "Writing context lines on Remote AI.")}
            >
              {busy ? <Loader2 className="animate-spin" aria-hidden /> : <PenLine aria-hidden />}
              Write {num(missing)} on Remote AI
            </Button>
          )
        )}
        {!running && missing > 0 && !data.can.ok && <span className="text-xs text-muted-foreground">{data.can.reason}</span>}
      </div>
    </div>
  );
}
