"use client";

import { useEffect, useRef, useState } from "react";
import useSWR from "swr";
import {
  AlertTriangle,
  ArrowUp,
  Bot,
  CircleSlash,
  Loader2,
  MessageSquareText,
  RotateCcw,
  SlidersHorizontal,
  Square,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { PageHeader } from "@/components/common/page-header";
import { EmptyState, ErrorState } from "@/components/common/states";
import { AnswerText } from "@/components/ask/answer-text";
import { ConversationList, ConversationsSheet } from "@/components/ask/conversations";
import { useAsk, type Exchange } from "@/components/ask/ask-store";
import { PagePreview } from "@/components/ask/page-preview";
import { SetupPicker } from "@/components/ask/setup-picker";
import { Sources } from "@/components/ask/sources";
import { WhatHappened, duration } from "@/components/ask/trace";
import { ApiError, type AppStatus, type DocumentFile, type GoldenSet, type IndexesResponse, type Source } from "@/lib/api";
import { num } from "@/lib/format";
import { cn } from "@/lib/utils";

/** The API's limit, mirrored only to show it. The server applies it. */
const QUESTION_LIMIT = 2000;

const SEARCHED_BY = { dense: "by meaning", bm25: "by words (BM25)", hybrid: "by meaning and by words (hybrid)" } as const;

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
  // On a phone the setup switches would take half the screen: folded until asked for (QA M5-04).
  const [setupOpen, setSetupOpen] = useState(false);
  // The page preview needs the index the source came from: a cleaned index
  // locates its passage differently from the baseline.
  const [preview, setPreview] = useState<{ source: Source; profile: string } | null>(null);
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

  // The intro describes the index answers will come from: the chosen one.
  const { data: indexes } = useSWR<IndexesResponse>("/indexes");
  const profile = useAsk((s) => s.profile) ?? indexes?.default_profile;
  const search = useAsk((s) => s.search);
  const chosen = indexes?.profiles.find((p) => p.id === profile);
  const examples = golden?.sets.find((s) => s.id === "set-a")?.questions.filter((q) => q.id.startsWith("A")).slice(0, 4) ?? [];
  const noIndex = status && !status.indexed;
  // The first question loads the search model into this app and the answer
  // model into Ollama; with too little free memory that would push the
  // computer into swap, so say so up front.
  const lowMemory = status?.memory.short && !busy;

  return (
    <div className="@container mx-auto max-w-5xl">
    <div className="grid gap-6 @4xl:grid-cols-[13rem_minmax(0,1fr)]">
      {/* Saved conversations: a column beside a wide page, a Sheet below it. */}
      <aside className="hidden @4xl:block">
        <div className="sticky top-4 pt-14">
          <ConversationList />
        </div>
      </aside>
    <div className="max-w-3xl min-w-0 space-y-6">
      <PageHeader
        href="/"
        actions={
          <div className="flex items-center gap-2">
            <span className="@4xl:hidden">
              <ConversationsSheet />
            </span>
            {exchanges.length > 0 && (
              <Button variant="ghost" size="sm" onClick={clear} disabled={busy}>
                <RotateCcw aria-hidden />
                New conversation
              </Button>
            )}
          </div>
        }
      />

      {noIndex ? (
        <EmptyState
          icon={MessageSquareText}
          title="No search index yet"
          description="Answers come from passages found in a search index of your reports. Build the index once, then ask here."
          hint={
            <>
              Build it with <code className="rounded bg-muted px-1 py-0.5 font-mono">{status?.build_command}</code>.
              The first build takes about half an hour on this computer; later builds reuse what is saved.
            </>
          }
        />
      ) : lowMemory && exchanges.length === 0 ? (
        <EmptyState
          icon={MessageSquareText}
          title="Not enough free memory to answer yet"
          description={`The first question loads ${
            status?.memory.parts?.length
              ? status.memory.parts.map((p) => `${p.what} (about ${p.gb} GB)`).join(" and ")
              : "the search model"
          }${
            // One part already names its size; "in all" only adds up two (M8-08).
            (status?.memory.parts?.length ?? 0) === 1 ? "" : `: about ${status?.memory.needed_gb} GB in all`
          }, and ${status?.memory.free_gb} GB is free.`}
          hint="Close some apps, then reload this page. Asking now could slow the whole computer down."
        />
      ) : (
        <>
          {exchanges.length === 0 && (
            <Card>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  {status && chosen?.active
                    ? `Answers come only from your ${num(status.documents)} reports: ${num(chosen.active.rows)} passages in the ${chosen.label} index, searched ${SEARCHED_BY[search]}. Each fact is cited, and a citation opens the page it came from.`
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
              <ExchangeView
                key={ex.id}
                ex={ex}
                docs={docs}
                onOpen={(source) => setPreview({ source, profile: ex.ready?.index.profile ?? indexes?.default_profile ?? "" })}
                onRetry={() => void ask(ex.question)}
                onAsTyped={() => void send(ex.question, { asTyped: true })}
                busy={busy}
              />
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
              <div className="flex flex-wrap items-center justify-between gap-2 px-1 pt-1">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="text-xs text-muted-foreground sm:hidden"
                  aria-expanded={setupOpen}
                  aria-controls="ask-setup"
                  onClick={() => setSetupOpen(!setupOpen)}
                >
                  <SlidersHorizontal aria-hidden />
                  Search setup
                </Button>
                <div id="ask-setup" className={cn("w-full sm:contents", !setupOpen && "hidden")}>
                  <SetupPicker disabled={busy} />
                </div>
                <div className="ml-auto flex items-center gap-2">
                <p className="hidden text-[11px] text-muted-foreground sm:block">
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
            </div>
          </form>
        </>
      )}

      <PagePreview
        source={preview?.source ?? null}
        profile={preview?.profile ?? indexes?.default_profile ?? ""}
        docs={docs}
        open={preview !== null}
        onOpenChange={(o) => !o && setPreview(null)}
      />
    </div>
    </div>
    </div>
  );
}

/** What is happening right now, in words, while the answer is not done. */
function liveLabel(ex: Exchange): string {
  const last = ex.stages.at(-1)?.id;
  if (!last) return "Searching your documents...";
  if (last === "followup") return "Searching for the full question...";
  if (last === "rewrite") return "Searching with the rewritten question...";
  if (last === "fuse") return "Checking for instructions...";
  if (last === "rerank") return "Checking for instructions...";
  if (last === "round") return "Deciding whether to search again...";
  if (last === "embed") return "Searching your documents...";
  if (last === "search") return "Checking for instructions...";
  if (last === "guard") return "Building the prompt...";
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
  onAsTyped,
  busy,
}: {
  ex: Exchange;
  docs: DocumentFile[] | undefined;
  onOpen: (s: Source) => void;
  onRetry: () => void;
  onAsTyped: () => void;
  busy: boolean;
}) {
  const sources = ex.done?.sources ?? ex.ready?.sources ?? [];
  const refused = ex.done?.refused === true;
  // The guard step lists instruction-like phrases it found in the question.
  const guardStep = ex.stages.find((s) => s.id === "guard");
  const asked = (guardStep?.detail.question_flags as unknown as { label: string }[] | undefined) ?? [];
  const cited = ex.done?.citations.valid ?? ex.saved?.citations.map((c) => c.n) ?? [];
  // A follow-up made whole by Remote AI: say what was really searched for, and offer the question as typed.
  const searchedFor = ex.ready?.standalone_question ?? ex.done?.standalone_question ?? ex.saved?.standalone ?? null;
  const rewritten = searchedFor !== null && searchedFor !== ex.question;
  const typed = ex.asTyped === true || ex.done?.rewritten === false || ex.saved?.rewritten === false;

  return (
    <article className="space-y-3" aria-busy={ex.status === "streaming"}>
      <div className="flex flex-col items-end gap-1">
        <p className="max-w-[85%] rounded-2xl rounded-br-md bg-secondary px-3.5 py-2 text-sm text-pretty text-secondary-foreground">
          {ex.question}
        </p>
        {(rewritten || typed) && (
          <p className="flex max-w-[85%] flex-wrap items-center justify-end gap-x-2 gap-y-1 text-xs text-muted-foreground">
            {rewritten ? (
              <>
                <span>
                  Searched for: <span className="text-foreground">{searchedFor}</span>
                </span>
                {ex.status !== "streaming" && !ex.saved && (
                  <Button size="xs" variant="outline" onClick={onAsTyped} disabled={busy}>
                    Ask as typed
                  </Button>
                )}
              </>
            ) : (
              <span>Asked as typed: not rewritten.</span>
            )}
          </p>
        )}
        {asked.length > 0 && (
          <p className="flex max-w-[85%] items-start gap-1.5 text-xs text-amber-700 dark:text-amber-400">
            <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
            {asked[0].label}. The rules still apply: answers come only from your documents.
          </p>
        )}
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
              action={
                ex.error?.can_ask_as_typed && !busy ? (
                  <Button size="sm" variant="outline" onClick={onAsTyped}>
                    Ask as typed
                  </Button>
                ) : undefined
              }
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
            <AnswerText
              text={ex.text}
              sources={sources}
              saved={ex.saved?.sources}
              docs={docs}
              onCite={onOpen}
              streaming={ex.status === "streaming"}
            />
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

          {ex.saved && (
            <p className="text-xs text-muted-foreground">
              Saved answer{ex.saved.citations.length > 0 ? ` · cites ${ex.saved.citations.map((c) => `${c.doc_id} p${c.page}`).join(", ")}` : ""}
              {ex.saved.ms ? ` · ${duration(ex.saved.ms)}` : ""}
            </p>
          )}

          {ex.done && (
            <p className={cn("text-xs text-muted-foreground tabular-nums")}>
              {refused ? "Refused" : `${cited.length} of ${sources.length} passages cited`} · {duration(ex.done.total_ms)} ·{" "}
              {ex.done.model.name}
            </p>
          )}
          {ex.done?.chat_warning && (
            <p className="flex items-start gap-1.5 text-xs text-amber-700 dark:text-amber-400">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden />
              {ex.done.chat_warning}
            </p>
          )}

          {sources.length > 0 && (
            <Sources
              sources={sources}
              cited={ex.done ? cited : null}
              docs={docs}
              onOpen={onOpen}
              promptVersion={ex.ready?.prompt.version}
              contextModel={(ex.ready?.index.config as { context?: { model?: string } } | undefined)?.context?.model}
            />
          )}
          {ex.stages.length > 0 && <WhatHappened steps={ex.done?.trace ?? ex.stages} ready={ex.ready} done={ex.done} />}
        </div>
      </div>
    </article>
  );
}
