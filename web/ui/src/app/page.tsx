"use client";

import { useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { ArrowUp, Bot, CircleSlash, Loader2, MessageSquareText, RotateCcw, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { PageHeader } from "@/components/common/page-header";
import { EmptyState, ErrorState } from "@/components/common/states";
import { AnswerText } from "@/components/ask/answer-text";
import { useAsk, type Exchange } from "@/components/ask/ask-store";
import { PagePreview } from "@/components/ask/page-preview";
import { Sources } from "@/components/ask/sources";
import { WhatHappened, duration } from "@/components/ask/trace";
import { ApiError, type AppStatus, type DocumentFile, type GoldenSet, type Source } from "@/lib/api";
import { num } from "@/lib/format";
import { cn } from "@/lib/utils";

/** The API's limit, mirrored only to show it. The server applies it. */
const QUESTION_LIMIT = 2000;

export default function AskPage() {
  const { data: status } = useSWR<AppStatus>("/status", { refreshInterval: 30_000 });
  const { data: docData } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const { data: golden } = useSWR<{ sets: GoldenSet[] }>("/golden");
  const exchanges = useAsk((s) => s.exchanges);
  const busy = useAsk((s) => s.busy);
  const send = useAsk((s) => s.send);
  const stop = useAsk((s) => s.stop);
  const clear = useAsk((s) => s.clear);

  const [draft, setDraft] = useState("");
  const [preview, setPreview] = useState<Source | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const docs = docData?.documents;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [exchanges.length]);

  async function ask(question: string) {
    if (!question.trim() || busy) return;
    setDraft("");
    await send(question);
    inputRef.current?.focus();
  }

  const examples = golden?.sets.find((s) => s.id === "set-a")?.questions.filter((q) => q.id.startsWith("A")).slice(0, 4) ?? [];
  const noIndex = status && !status.indexed;

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <PageHeader
        href="/"
        actions={
          exchanges.length > 0 && (
            <Button variant="ghost" size="sm" onClick={clear} disabled={busy}>
              <RotateCcw aria-hidden />
              Clear
            </Button>
          )
        }
      />

      {noIndex ? (
        <EmptyState
          icon={MessageSquareText}
          title="No search index yet"
          description="Answers come from passages found in a search index of your reports. Build the index once, then ask here."
          hint={
            <>
              Build it with <code className="rounded bg-muted px-1 py-0.5 font-mono">./rag ingest</code> (about 7
              minutes on this computer).
            </>
          }
        />
      ) : (
        <>
          {exchanges.length === 0 && (
            <Card>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  {status?.index
                    ? `Answers come only from your ${num(status.documents)} reports: ${num(status.index.rows)} passages, searched by meaning. Each fact is cited, and a citation opens the page it came from.`
                    : "Answers come only from your reports. Each fact is cited, and a citation opens the page it came from."}
                </p>
                {examples.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground">Try one of your test questions</p>
                    <div className="flex flex-col gap-1.5">
                      {examples.map((q) => (
                        <button
                          key={q.id}
                          type="button"
                          onClick={() => void ask(q.question)}
                          disabled={busy}
                          className="rounded-md px-2.5 py-1.5 text-left text-sm ring-1 ring-foreground/10 transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none disabled:opacity-50"
                        >
                          {q.question}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          <div className="space-y-8">
            {exchanges.map((ex) => (
              <ExchangeView key={ex.id} ex={ex} docs={docs} onOpen={setPreview} onRetry={() => void ask(ex.question)} busy={busy} />
            ))}
          </div>
          <div ref={endRef} />

          <form
            className="sticky bottom-4 z-10"
            onSubmit={(e) => {
              e.preventDefault();
              void ask(draft);
            }}
          >
            <div className="rounded-xl bg-card/95 p-2 shadow-lg ring-1 ring-foreground/10 backdrop-blur supports-[backdrop-filter]:bg-card/85">
              <label htmlFor="question" className="sr-only">
                Your question
              </label>
              <Textarea
                id="question"
                ref={inputRef}
                value={draft}
                maxLength={QUESTION_LIMIT}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    void ask(draft);
                  }
                }}
                placeholder="Ask about your reports..."
                rows={2}
                className="max-h-48 min-h-14 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0 dark:bg-transparent"
              />
              <div className="flex items-center justify-between gap-2 px-1 pt-1">
                <p className="text-[11px] text-muted-foreground">
                  {draft.length >= QUESTION_LIMIT * 0.9
                    ? `${num(draft.length)} of ${num(QUESTION_LIMIT)} characters`
                    : "Enter to ask · Shift+Enter for a new line"}
                </p>
                {busy ? (
                  <Button type="button" size="sm" variant="outline" onClick={stop}>
                    <Square aria-hidden />
                    Stop
                  </Button>
                ) : (
                  <Button type="submit" size="sm" disabled={!draft.trim()}>
                    <ArrowUp aria-hidden />
                    Ask
                  </Button>
                )}
              </div>
            </div>
          </form>
        </>
      )}

      <PagePreview source={preview} docs={docs} open={preview !== null} onOpenChange={(o) => !o && setPreview(null)} />
    </div>
  );
}

/** What is happening right now, in words, while the answer is not done. */
function liveLabel(ex: Exchange): string {
  const last = ex.stages.at(-1)?.id;
  if (!last) return "Searching your documents...";
  if (last === "embed") return "Searching your documents...";
  if (last === "search") return "Building the prompt...";
  if (last === "prompt") {
    return ex.text
      ? `${ex.ready?.model.name ?? "The model"} is writing...`
      : `${ex.ready?.model.name ?? "The model"} is reading ${ex.ready?.sources.length ?? ""} passages. On this CPU that takes 10-40 seconds.`;
  }
  return "Finishing...";
}

function ExchangeView({
  ex,
  docs,
  onOpen,
  onRetry,
  busy,
}: {
  ex: Exchange;
  docs: DocumentFile[] | undefined;
  onOpen: (s: Source) => void;
  onRetry: () => void;
  busy: boolean;
}) {
  const sources = ex.done?.sources ?? ex.ready?.sources ?? [];
  const refused = ex.done?.refused === true;
  const cited = ex.done?.citations.valid ?? [];

  return (
    <article className="space-y-3" aria-busy={ex.status === "streaming"}>
      <div className="flex justify-end">
        <p className="max-w-[85%] rounded-2xl rounded-br-md bg-secondary px-3.5 py-2 text-sm text-pretty text-secondary-foreground">
          {ex.question}
        </p>
      </div>

      <div className="flex items-start gap-3">
        <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-muted" aria-hidden>
          <Bot className="size-4 text-muted-foreground" />
        </div>
        <div className="min-w-0 flex-1 space-y-3">
          {ex.status === "error" ? (
            <ErrorState
              title="This answer didn't complete"
              message={ex.error?.error ?? "The request failed."}
              // Only a network failure suggests starting the app service; any
              // other error came from the service itself, so it is running.
              error={ex.error?.stage === "network" ? undefined : new ApiError(ex.error?.error ?? "", 500)}
              onRetry={busy ? undefined : onRetry}
            />
          ) : refused ? (
            <div className="space-y-1.5 rounded-lg bg-muted/40 px-3.5 py-3 ring-1 ring-foreground/10">
              <p className="flex items-center gap-2 text-sm font-medium">
                <CircleSlash className="size-4 text-muted-foreground" aria-hidden />
                Not in your documents
              </p>
              <p className="text-sm text-muted-foreground text-pretty">
                None of the {sources.length} passages found states this, so no answer was made up. If the
                reports should cover it, try naming the report or the exact term it uses.
              </p>
              <p className="text-xs text-muted-foreground">
                The model replied: &ldquo;{ex.done?.answer.trim()}&rdquo;
              </p>
            </div>
          ) : ex.text ? (
            <AnswerText text={ex.text} sources={sources} docs={docs} onCite={onOpen} streaming={ex.status === "streaming"} />
          ) : null}

          {ex.status === "streaming" && (
            <p className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
              <Loader2 className="size-4 animate-spin" aria-hidden />
              {liveLabel(ex)}
            </p>
          )}
          {ex.status === "stopped" && (
            <p className="text-sm text-muted-foreground">Stopped. The partial answer above is incomplete.</p>
          )}

          {ex.done && ex.done.citations.invalid.length > 0 && (
            <p className="text-xs text-red-700 dark:text-red-400">
              The answer cites {ex.done.citations.invalid.map((n) => `[${n}]`).join(", ")}, but the model was given
              only {sources.length} passages. Treat those claims as unsupported.
            </p>
          )}

          {ex.done && (
            <p className={cn("text-xs text-muted-foreground tabular-nums")}>
              {refused ? "Refused" : `${cited.length} of ${sources.length} passages cited`} · {duration(ex.done.total_ms)} ·{" "}
              {ex.done.model.name}
            </p>
          )}

          {sources.length > 0 && (
            <Sources sources={sources} cited={ex.done ? cited : null} docs={docs} onOpen={onOpen} />
          )}
          {ex.stages.length > 0 && <WhatHappened steps={ex.done?.trace ?? ex.stages} ready={ex.ready} done={ex.done} />}
        </div>
      </div>
    </article>
  );
}
