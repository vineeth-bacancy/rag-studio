import { useState } from "react";
import useSWR from "swr";
import { Play } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { api, type AnswerModel, type GoldenSet, type RunBrief, type RunMode } from "@/lib/api";

/**
 * Start a run: which test questions, whether the model answers them, and
 * which model. The time estimate is only shown when a past run with the same
 * model measured it - never guessed.
 */
export function StartRun({
  sets,
  runs,
  active,
  onStarted,
}: {
  sets: GoldenSet[];
  runs: RunBrief[];
  active: boolean;
  onStarted: (id: string) => void;
}) {
  const { data: models } = useSWR<{ default: string; answer_models: AnswerModel[] }>("/models");
  const [chosen, setChosen] = useState<string[]>(["set-a"]);
  const [mode, setMode] = useState<RunMode>("answer");
  const [model, setModel] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  const pickedModel = model ?? models?.default ?? null;
  const note = models?.answer_models.find((m) => m.name === pickedModel)?.note;
  const scoredCount = sets
    .filter((s) => chosen.includes(s.id) && s.scored)
    .reduce((n, s) => n + s.questions.length, 0);
  const perQuestion = measuredSecondsPerAnswer(runs, pickedModel);
  const estimate = !models && mode === "answer"
    ? "Loading the installed models..."
    : mode === "retrieval"
      ? "A few seconds: search only, no model answers."
      : perQuestion !== null && scoredCount > 0
        ? `About ${Math.max(1, Math.round((perQuestion * scoredCount) / 60))} min for ${scoredCount} answers, from your last run with ${pickedModel}.`
        : scoredCount > 0
          ? `${scoredCount} answers. No earlier run with ${pickedModel} to estimate the time from.`
          : "None of the chosen sets has scored questions; only retrieval will be measured.";

  async function start() {
    setStarting(true);
    try {
      const { id } = await api.startRun({ sets: chosen, mode, model: mode === "answer" ? pickedModel : null });
      toast.success("Run started. Results appear below as each question finishes.");
      onStarted(id);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "The run did not start. Try again.");
    } finally {
      setStarting(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2">Run evaluation</CardTitle>
        <CardDescription>
          Ask every test question with the current setup and score each answer. Runs use the CPU fully, so one
          runs at a time.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <fieldset className="space-y-2">
          <legend className="mb-2 text-sm font-medium">Test questions</legend>
          {sets.map((s) => {
            const id = `set-${s.id}`;
            return (
              <label key={s.id} htmlFor={id} className="flex cursor-pointer items-start gap-3 rounded-md p-1.5 hover:bg-muted/50">
                <Checkbox
                  id={id}
                  className="mt-0.5"
                  checked={chosen.includes(s.id)}
                  onCheckedChange={(on) =>
                    setChosen((c) => (on ? [...c, s.id] : c.filter((x) => x !== s.id)))
                  }
                />
                <span className="min-w-0 space-y-0.5">
                  <span className="block text-sm">
                    {s.name} <span className="text-muted-foreground tabular-nums">· {s.questions.length} questions</span>
                    {!s.scored && <span className="text-muted-foreground"> · retrieval only</span>}
                  </span>
                  <span className="block text-xs text-pretty text-muted-foreground">{s.description}</span>
                </span>
              </label>
            );
          })}
        </fieldset>

        <fieldset className="space-y-2">
          <legend className="mb-2 text-sm font-medium">What to measure</legend>
          {/* Native radios inside the cards: arrow keys move between them. */}
          <div className="grid gap-2 sm:grid-cols-2">
            {(
              [
                ["answer", "Full scorecard", "Search, then the model answers and each answer is scored."],
                ["retrieval", "Retrieval only", "Search only: did an answer page reach the top 5?"],
              ] as const
            ).map(([value, label, hint]) => (
              <label
                key={value}
                className="cursor-pointer rounded-lg p-3 ring-1 ring-foreground/10 transition-colors hover:bg-muted/40 has-checked:bg-muted/60 has-checked:ring-foreground/40 has-focus-visible:ring-[3px] has-focus-visible:ring-ring/50"
              >
                <input
                  type="radio"
                  name="run-mode"
                  value={value}
                  checked={mode === value}
                  onChange={() => setMode(value)}
                  className="sr-only"
                />
                <span className="block text-sm font-medium">{label}</span>
                <span className="block text-xs text-muted-foreground">{hint}</span>
              </label>
            ))}
          </div>
        </fieldset>

        {mode === "answer" && (
          <div className="space-y-1.5">
            <label htmlFor="run-model" className="text-sm font-medium">
              Answer model
            </label>
            <Select value={pickedModel} onValueChange={(v) => setModel(v as string)}>
              <SelectTrigger id="run-model" className="w-full sm:w-72">
                <SelectValue placeholder="Choose a model" />
              </SelectTrigger>
              <SelectContent>
                {models?.answer_models.map((m) => (
                  <SelectItem key={m.name} value={m.name}>
                    {m.name}
                    {m.name === models.default ? " (default)" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {note && <p className="text-xs text-muted-foreground">{note}</p>}
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-4">
          <p className="text-xs text-muted-foreground">{estimate}</p>
          <Button onClick={start} disabled={active || starting || chosen.length === 0 || (mode === "answer" && !pickedModel)}>
            <Play aria-hidden />
            {active ? "A run is in progress" : starting ? "Starting..." : "Run evaluation"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

/** Seconds per scored answer in the newest finished run with this model. */
function measuredSecondsPerAnswer(runs: RunBrief[], model: string | null): number | null {
  const r = runs.find((x) => x.status === "done" && x.config.answer_model === model && x.summary?.scored);
  if (!r?.summary || !r.summary.scored) return null;
  return r.summary.answer_seconds / r.summary.scored;
}
