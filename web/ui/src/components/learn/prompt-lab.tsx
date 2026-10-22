"use client";

import useSWR from "swr";
import { CheckCircle2, Info, TriangleAlert } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { TechnicalDetails } from "@/components/common/technical-details";
import { OutcomeBadge } from "@/components/eval/badges";
import { useRecordedAnswer, type Preset } from "@/components/learn/recorded-picker";
import type { DocumentFile, PromptLab as PromptLabData, PromptPart, PromptVersionView } from "@/lib/api";
import { docTitle } from "@/lib/documents";
import { NO_DATA } from "@/lib/format";
import { cn } from "@/lib/utils";

const PRESETS: Preset[] = [
  { set: "set-a", id: "A1", why: "A plain fact question: the prompt as every answer gets it." },
  { set: "set-a", id: "A6", why: "One blank line in the prompt flipped this one between a refusal and a false attribution." },
  { set: "set-d", id: "T3", why: "The question itself tries to override the rules. Prompt v2 answered it; v3 refused." },
  { set: "set-d", id: "D10", why: "Its answer needs two documents: five long passages, the fullest prompt in the test set." },
];

/** Colours of the budget bar's parts. Green is kept for Healthy and Pass (docs/product-language.md). */
const PART_STYLE: Record<string, { label: string; className: string }> = {
  system: { label: "Rules (system prompt)", className: "bg-zinc-500 dark:bg-zinc-400" },
  passage: { label: "Passages", className: "bg-sky-600 dark:bg-sky-500" },
  question: { label: "Question", className: "bg-violet-600 dark:bg-violet-400" },
  frame: { label: "Framing (Context:, fence, reminder)", className: "bg-zinc-300 dark:bg-zinc-600" },
  answer: { label: "The answer", className: "bg-fuchsia-600 dark:bg-fuchsia-400" },
};

/**
 * Step 9: the prompt. One recorded answer's exact messages, read back from
 * its run (the passages from the run's own index build), the context budget
 * they used, why passages are numbered, and prompts v2, v3 and v4 for the
 * same passages, each with its outcome where a run measured it.
 */
export function PromptLab() {
  const { data: docData } = useSWR<{ documents: DocumentFile[] }>("/documents");
  const docs = docData?.documents;
  const { qid, runId, picker, listError, retryList } = useRecordedAnswer(PRESETS, "prompt");
  const key = runId ? `/learn/prompt?${new URLSearchParams({ run: runId, question: qid })}` : null;
  const { data, error, isLoading, mutate } = useSWR<PromptLabData>(key, { keepPreviousData: true });
  const shown = data && data.run.id === runId && data.question_id === qid ? data : null;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle as="h2">A recorded answer</CardTitle>
          <CardDescription>
            Pick a test question and a run that answered it. The prompt is rebuilt from that run: its passages are read
            back from the index build it searched, by the ids it recorded. Nothing is asked of a model.
          </CardDescription>
        </CardHeader>
        <CardContent>{picker}</CardContent>
      </Card>

      {listError ? (
        <ErrorState message={listError.message} error={listError} onRetry={retryList} />
      ) : error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : !runId ? null : !shown ? (
        <CardSkeleton height="h-96" />
      ) : (
        <div className={cn("space-y-4", isLoading && "opacity-70")}>
          <Messages data={shown} />
          <Budget data={shown} />
          <Numbered data={shown} docs={docs} />
          <Versions data={shown} />
        </div>
      )}
    </div>
  );
}

function ExactNote({ exact }: { exact: PromptLabData["exact"] }) {
  if (exact === "identical") {
    return (
      <p className="flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-400">
        <CheckCircle2 className="size-3.5" aria-hidden />
        Byte for byte what the model read: the run recorded a hash of the messages, and these match it.
      </p>
    );
  }
  if (exact === "differs") {
    return (
      <p className="flex items-center gap-1.5 text-xs text-amber-700 dark:text-amber-400">
        <TriangleAlert className="size-3.5" aria-hidden />
        Needs attention: these differ from what the run sent, so its index build or the prompt code changed since.
      </p>
    );
  }
  return (
    <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
      <Info className="size-3.5" aria-hidden />
      Rebuilt from the run’s passages and prompt version (its hash matches). Runs made before M6 did not record a hash
      of the whole messages, so this cannot be proven byte for byte.
    </p>
  );
}

/** One message, with the parts a reader should notice marked. The text itself is shown exactly. */
function MessageText({ text, markers }: { text: string; markers: PromptLabData["markers"] }) {
  const lines = text.split("\n");
  return (
    <pre className="max-h-[28rem] overflow-auto rounded-md bg-muted/50 p-3 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
      {lines.map((line, i) => {
        const cls = /^\[\d+\] \S+:p\d+:c\d+$/.test(line)
          ? "font-semibold text-sky-800 dark:text-sky-300"
          : line === markers.fence_open || line === markers.fence_close || line === markers.reminder
            ? "bg-amber-500/15 text-amber-900 dark:text-amber-300"
            : line.startsWith("Question: ")
              ? "font-semibold"
              : /^(Document|Section): /.test(line)
                ? "text-muted-foreground italic"
                : "";
        return (
          <span key={i} className={cn("block", cls)}>
            {line || " "}
          </span>
        );
      })}
    </pre>
  );
}

function Messages({ data }: { data: PromptLabData }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          The exact messages
        </CardTitle>
        <CardDescription>
          Two messages: the rules (the system prompt, version {data.version}) and one user message with the numbered
          passages and the question{data.searched_for !== data.question ? ", as the follow-up was searched" : ""}. The
          model sees nothing else: not the conversation, not the page, not the score.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <ExactNote exact={data.exact} />
        {data.messages.map((m) => (
          <div key={m.role} className="space-y-1">
            <p className="text-xs font-medium">
              {m.role === "system" ? "System" : "User"}{" "}
              <span className="font-normal text-muted-foreground tabular-nums">
                · {m.content.length.toLocaleString()} characters
              </span>
            </p>
            <MessageText text={m.content} markers={data.markers} />
          </div>
        ))}
        <TechnicalDetails
          rows={{
            Run: data.run.id,
            "Prompt sha256": data.prompt_sha256,
            Model: data.model.model ?? NO_DATA,
            Thinking: data.model.think === false ? "off (think: false, always sent)" : String(data.model.think),
            Temperature: data.model.temperature ?? NO_DATA,
          }}
        />
      </CardContent>
    </Card>
  );
}

function partKey(p: PromptPart): string {
  return p.part === "passage" ? `passage-${p.n}` : p.part;
}

function Bar({ parts, total, label }: { parts: { key: string; style: string; tokens: number; title: string }[]; total: number; label: string }) {
  return (
    <div className="space-y-1">
      <p className="text-xs text-muted-foreground">{label}</p>
      <div className="flex h-5 w-full overflow-hidden rounded-md ring-1 ring-foreground/15" role="img" aria-label={label}>
        {parts.map((p) => (
          <div
            key={p.key}
            className={cn("h-full border-r border-background last:border-r-0", p.style)}
            style={{ width: `${(100 * p.tokens) / total}%` }}
            title={`${p.title}: ${p.tokens.toLocaleString()} tokens`}
          />
        ))}
      </div>
    </div>
  );
}

function Budget({ data }: { data: PromptLabData }) {
  const b = data.budget;
  const promptTokens = b.prompt_tokens ?? b.estimated_tokens;
  const ctxWindow = b.num_ctx ?? 8192;
  const segments = data.parts.map((p, i) => ({
    key: partKey(p),
    style: cn(PART_STYLE[p.part].className, p.part === "passage" && i % 2 === 0 && "opacity-75"),
    tokens: p.tokens,
    title: p.part === "passage" ? `Passage [${p.n}]` : PART_STYLE[p.part].label,
  }));
  const withAnswer = [
    ...segments,
    ...(b.answer_tokens ? [{ key: "answer", style: PART_STYLE.answer.className, tokens: b.answer_tokens, title: "The answer" }] : []),
  ];
  const used = promptTokens + (b.answer_tokens ?? 0);
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          The context budget
        </CardTitle>
        <CardDescription>
          The prompt and the answer share one window of {ctxWindow.toLocaleString()} tokens (num_ctx). Ollama does not
          refuse a prompt that is too long: it silently drops the start and answers anyway, so the rules would be the
          first thing lost.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Bar
          parts={withAnswer}
          total={ctxWindow}
          label={`In the ${ctxWindow.toLocaleString()}-token window: ${Math.round((100 * used) / ctxWindow)}% used`}
        />
        <Bar parts={segments} total={promptTokens} label="Inside the prompt" />
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {Object.entries(PART_STYLE).map(([k, s]) => (
            <li key={k} className="flex items-center gap-1.5">
              <span className={cn("size-2.5 rounded-sm", s.className)} aria-hidden />
              {s.label}
            </li>
          ))}
        </ul>
        <dl className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-[auto_1fr]">
          <dt className="text-muted-foreground">Prompt</dt>
          <dd className="tabular-nums">
            {b.prompt_tokens !== null
              ? `${b.prompt_tokens.toLocaleString()} tokens, counted by Ollama when it was sent`
              : `about ${b.estimated_tokens.toLocaleString()} tokens (estimated at 4 characters per token)`}{" "}
            · {b.chars.toLocaleString()} characters
          </dd>
          <dt className="text-muted-foreground">The answer</dt>
          <dd className="tabular-nums">{b.answer_tokens !== null ? `${b.answer_tokens} tokens` : NO_DATA}</dd>
          <dt className="text-muted-foreground">Characters per token</dt>
          <dd className="tabular-nums">
            {b.chars_per_token ?? NO_DATA}
            <span className="text-muted-foreground">
              {" "}
              · prose here runs 3-5. Above {b.truncation_chars_per_token}, the model read far less than it was sent, and
              the trace says the start was likely dropped{b.truncated ? " (it did here)" : ""}.
            </span>
          </dd>
        </dl>
        <p className="text-xs text-pretty text-muted-foreground">
          {b.split === "measured"
            ? "Ollama counts the whole prompt; the split between its parts is by characters, so the part sizes are close, not exact."
            : "This run did not record Ollama's count, so every size here is estimated."}
        </p>
      </CardContent>
    </Card>
  );
}

function Numbered({ data, docs }: { data: PromptLabData; docs: DocumentFile[] | undefined }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          Why passages are numbered
        </CardTitle>
        <CardDescription>
          The model points at a passage by its number, one short token like [2], and never writes a page. Code keeps
          this table and turns each number into a document and page after the answer is written (step 10). A page the
          model typed itself could be invented; a number can only point at what it was shown.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ol className="divide-y text-xs">
          {data.sources.map((s) => (
            <li key={s.n} className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 py-1.5">
              <span className="font-mono font-semibold text-sky-800 tabular-nums dark:text-sky-300">[{s.n}]</span>
              <span className="min-w-0 flex-1 truncate">{docTitle(s.doc_id, docs)}</span>
              <span className="text-muted-foreground tabular-nums">
                p. {s.page}
                {s.page_label && s.page_label !== String(s.page) ? ` (printed ${s.page_label})` : ""}
              </span>
              {s.cited ? (
                <Badge variant="secondary" className="h-4 px-1.5 text-[10px] font-normal">
                  Cited in the answer
                </Badge>
              ) : (
                <span className="text-[10px] text-muted-foreground">not cited</span>
              )}
            </li>
          ))}
        </ol>
      </CardContent>
    </Card>
  );
}

function VersionColumn({ v }: { v: PromptVersionView }) {
  const changed = new Set(v.changed);
  return (
    <div className="flex min-w-0 flex-col gap-2 rounded-lg p-3 ring-1 ring-foreground/10">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold">{v.version}</span>
        {v.current && (
          <Badge variant="secondary" className="h-4 px-1.5 text-[10px] font-normal">
            This run
          </Badge>
        )}
        <span className="font-mono text-[10px] text-muted-foreground">{v.sha256.slice(0, 12)}</span>
      </div>
      <p className="text-xs text-muted-foreground">
        {v.against ? `${v.changed.length} line${v.changed.length === 1 ? "" : "s"} new in ${v.version} (marked)` : "The tested baseline rules"}
        {" · "}
        {v.fenced ? "passages fenced off as data" : "passages not fenced"}
      </p>
      <pre className="max-h-72 overflow-auto rounded-md bg-muted/50 p-2 font-mono text-[10.5px] leading-relaxed whitespace-pre-wrap">
        {v.system.split("\n").map((line, i) => (
          <span key={i} className={cn("block", changed.has(i) && "bg-amber-500/15 text-amber-900 dark:text-amber-300")}>
            {line || " "}
          </span>
        ))}
      </pre>
      <div className="mt-auto space-y-1 border-t pt-2 text-xs">
        {v.measured ? (
          <>
            <p className="flex flex-wrap items-center gap-2">
              <OutcomeBadge outcome={v.measured.outcome} />
              <span className="text-muted-foreground">{v.measured.this_run ? "this run" : `run ${v.measured.run_id}`}</span>
            </p>
            <p className="line-clamp-4 text-pretty">{v.measured.answer}</p>
          </>
        ) : (
          <p className="text-muted-foreground">Not measured with these passages and this model.</p>
        )}
      </div>
      <TechnicalDetails label="User message for these passages">
        <pre className="max-h-64 overflow-auto font-mono text-[10px] whitespace-pre-wrap">{v.user}</pre>
      </TechnicalDetails>
    </div>
  );
}

function Versions({ data }: { data: PromptLabData }) {
  return (
    <Card className="@container">
      <CardHeader>
        <CardTitle as="h2" className="text-base">
          v2, v3 and v4 side by side
        </CardTitle>
        <CardDescription>
          The same passages under each prompt version. v3 adds one rule and fences the passages off as quoted data (the
          injection guard, step 13); v4 relaxes rule 4 once passages carry section labels. Every version is recorded with
          its hash, because whitespace alone has moved answers here.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3 @3xl:grid-cols-3">
        {data.versions.map((v) => (
          <VersionColumn key={v.version} v={v} />
        ))}
      </CardContent>
    </Card>
  );
}
