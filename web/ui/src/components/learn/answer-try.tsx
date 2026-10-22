import { useState } from "react";
import useSWR from "swr";
import { RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ErrorState } from "@/components/common/states";
import { OutcomeBadge } from "@/components/eval/badges";
import { MarkedAnswer } from "@/components/learn/marked-answer";
import { useDebounced } from "@/hooks/use-debounced";
import { api, type AnswerCheck } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * Edit a recorded answer and watch the engine read it: the citations it maps
 * (to the passages that run showed), the strict refusal rule and the
 * scorecard outcome, by the same code that read the original. The presets
 * are the three mistakes worth seeing.
 */
export function AnswerTry({
  runId,
  questionId,
  recorded,
  k,
  refusal,
  show = "both",
}: {
  runId: string;
  questionId: string;
  recorded: string;
  k: number;
  refusal: string;
  /** Step 10 cares about citations, step 12 about the score; both show the outcome. */
  show?: "citations" | "score" | "both";
}) {
  const [text, setText] = useState(recorded);
  const settled = useDebounced(text, 400);
  const key = ["answer-check", runId, questionId, settled] as const;
  const { data, error, mutate } = useSWR<AnswerCheck>(key, () =>
    api.answerCheck({ run: runId, question: questionId, text: settled }),
  );
  const current = data && settled === text ? data : null;

  // A refusal has no fact of its own to keep, so "add a fact" writes one that is plainly made up.
  const fact = text.replaceAll(refusal, "").trim() || "An invented fact [1].";
  const presets: { label: string; make: () => string }[] = [
    { label: `Cite passage ${k + 2}`, make: () => `${text.trim()} [${k + 2}]` },
    { label: "Add a fact before the refusal", make: () => `${fact} ${refusal}` },
    { label: "Refuse", make: () => refusal },
  ];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-1.5">
        {presets.map((p) => (
          <Button key={p.label} size="xs" variant="outline" onClick={() => setText(p.make())}>
            {p.label}
          </Button>
        ))}
        <Button size="xs" variant="ghost" onClick={() => setText(recorded)} disabled={text === recorded}>
          <RotateCcw aria-hidden />
          Restore the recorded answer
        </Button>
      </div>
      <label htmlFor={`try-${questionId}`} className="sr-only">
        Answer to check
      </label>
      <Textarea
        id={`try-${questionId}`}
        value={text}
        maxLength={4000}
        onChange={(e) => setText(e.target.value)}
        className="font-mono text-xs md:text-xs"
      />
      {error ? (
        <ErrorState message={error.message} error={error} onRetry={() => void mutate()} />
      ) : (
        <div className={cn("space-y-2 rounded-lg bg-muted/40 px-3 py-2.5", !current && "opacity-60")} aria-live="polite">
          {current ? (
            <>
              <MarkedAnswer
                text={text}
                marks={show !== "score" ? current.marks : []}
                facts={show !== "citations" ? current.facts : []}
              />
              <dl className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-[auto_1fr]">
                <dt className="text-muted-foreground">Scorecard</dt>
                <dd>
                  <OutcomeBadge outcome={current.score.outcome} />
                  {current.score.outcome === null && <span className="text-muted-foreground"> search only</span>}
                </dd>
                <dt className="text-muted-foreground">Refusal</dt>
                <dd>
                  {current.refused
                    ? "Yes: the whole answer is exactly the refusal sentence."
                    : text.includes(refusal)
                      ? "No: it contains the refusal sentence, but says something else too, so it counts as an answer."
                      : "No."}
                </dd>
                {show !== "score" && (
                  <>
                    <dt className="text-muted-foreground">Citations</dt>
                    <dd className="tabular-nums">
                      {current.citations.numbers.length === 0
                        ? "None."
                        : `Points at ${current.citations.valid.length ? current.citations.valid.map((n) => `[${n}]`).join(" ") : "no passage shown"}`}
                      {current.citations.invalid.length > 0 && (
                        <span className="text-red-700 dark:text-red-400">
                          {" "}
                          · {current.citations.invalid.map((n) => `[${n}]`).join(" ")}: no such passage (only {current.k}{" "}
                          were shown). Reported, never dropped or fixed.
                        </span>
                      )}
                    </dd>
                  </>
                )}
                {show !== "citations" && current.facts.length > 0 && (
                  <>
                    <dt className="text-muted-foreground">Expected facts</dt>
                    <dd>
                      {current.facts.map((f) => (
                        <span key={f.label} className={cn("mr-2", f.found ? "text-emerald-700 dark:text-emerald-400" : "text-red-700 dark:text-red-400")}>
                          {f.found ? "✓" : "✗"} {f.label}
                        </span>
                      ))}
                    </dd>
                  </>
                )}
              </dl>
            </>
          ) : (
            <p className="text-xs text-muted-foreground" role="status">
              Reading it…
            </p>
          )}
        </div>
      )}
    </div>
  );
}
