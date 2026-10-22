import { useState } from "react";
import useSWR from "swr";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { OUTCOME } from "@/components/eval/badges";
import type { GoldenSet, RecordedAnswers, RecordedChoice } from "@/lib/api";
import { cn } from "@/lib/utils";

export interface Preset {
  set: string;
  id: string;
  why: string;
}

/** A run in a picker: its ladder row or label, the answer model and prompt, and how it scored this question. */
export function choiceLabel(c: RecordedChoice): string {
  const name = c.ladder ? `Row ${c.ladder.row}: ${c.ladder.label}` : (c.label ?? c.run_id);
  const outcome = c.outcome ? ` · ${OUTCOME[c.outcome].label}` : "";
  return `${name} · ${c.answer_model ?? "?"} · prompt ${c.prompt_version ?? "?"}${outcome}`;
}

/**
 * A test question and a recorded run that answered it, for How it works 9 and
 * 10. The runs come from `/learn/answers`: the app's default setup first, then
 * ladder runs. Until one is picked, the first is used.
 */
export function useRecordedAnswer(presets: Preset[], idPrefix: string) {
  const { data: golden } = useSWR<{ sets: GoldenSet[] }>("/golden");
  const [qid, setQid] = useState(presets[0].id);
  const [picked, setPicked] = useState<Record<string, string>>({});
  const list = useSWR<RecordedAnswers>(`/learn/answers?question=${qid}`, { keepPreviousData: true });
  // A list still showing the previous question is not this question's list.
  const runs = list.data?.question_id === qid ? list.data.runs : [];
  const runId = (picked[qid] && runs.some((r) => r.run_id === picked[qid]) ? picked[qid] : runs[0]?.run_id) ?? null;

  const options = presets.flatMap((p) => {
    const q = golden?.sets.find((s) => s.id === p.set)?.questions.find((x) => x.id === p.id);
    return q ? [{ ...p, question: q.question }] : [];
  });
  const preset = options.find((p) => p.id === qid);

  const picker = (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <p className="text-xs font-medium text-muted-foreground">Test question</p>
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Test question">
          {options.map((p) => (
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
        <label htmlFor={`${idPrefix}-run`} className="text-xs font-medium text-muted-foreground">
          Recorded answer
        </label>
        {list.data && runs.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No finished run has answered this question on an index that still exists. Run an evaluation that includes
            it first.
          </p>
        ) : (
          <Select
            value={runId ?? ""}
            items={Object.fromEntries(runs.map((r) => [r.run_id, choiceLabel(r)]))}
            onValueChange={(v) => v && setPicked({ ...picked, [qid]: v as string })}
          >
            <SelectTrigger id={`${idPrefix}-run`} className="w-full max-w-full sm:w-[36rem]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {runs.map((r) => (
                <SelectItem key={r.run_id} value={r.run_id}>
                  {choiceLabel(r)}
                  {r.default_setup ? " · the app's defaults" : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>
    </div>
  );
  return { qid, runId, picker, listError: list.error as Error | undefined, retryList: () => list.mutate() };
}
