import { useState } from "react";
import useSWR from "swr";
import { Play } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  api,
  type AnswerModel,
  type GoldenSet,
  type IndexesResponse,
  type PromptVersion,
  type RerankersResponse,
  type RunBrief,
  type RunMode,
  type SearchMethod,
} from "@/lib/api";
import { AGENT_ROUNDS, REWRITE_EXPLAIN, REWRITE_LABELS, SEARCH_LABELS, SECTION_CAP } from "@/lib/search";

/**
 * Start a run: which test questions, which index they search and how, whether
 * the model answers them, which model and which prompt. The time estimate is only
 * shown when a past run with the same model measured it - never guessed.
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
  const { data: indexes } = useSWR<IndexesResponse>("/indexes");
  const { data: prompts } = useSWR<{ default: string; versions: PromptVersion[] }>("/prompts");
  const [chosen, setChosen] = useState<string[]>(["set-a"]);
  const [mode, setMode] = useState<RunMode>("answer");
  const [model, setModel] = useState<string | null>(null);
  const [pickedProfileId, setProfile] = useState<string | null>(null);
  const profile = pickedProfileId ?? indexes?.default_profile ?? "";
  const [prompt, setPrompt] = useState<string | null>(null);
  const [search, setSearch] = useState<SearchMethod>("dense");
  const [capped, setCapped] = useState(false);
  const [rerank, setRerank] = useState<string | null>(null);
  const [rewrite, setRewrite] = useState<string | null>(null);
  const [agentRounds, setAgentRounds] = useState<number | null>(null);
  const [asTyped, setAsTyped] = useState(false);
  const { data: rerankers } = useSWR<RerankersResponse>("/rerankers");
  const [starting, setStarting] = useState(false);

  // Rewriting, multi-step search and Set E's follow-ups are done by Remote AI
  // (qwen3:8b on Colab), never by a silent local stand-in.
  const remote = rerankers?.rewrite;
  const remoteOk = remote?.ok === true;
  const followupSets = sets.filter((s) => s.remote_only && chosen.includes(s.id));
  const blocked =
    (rewrite || agentRounds) && remote && !remoteOk
      ? remote.reason
      : followupSets.length > 0 && !asTyped && remote && !remoteOk
        ? `${followupSets.map((s) => s.name).join(", ")} is rewritten by Remote AI, which is not connected. Connect it in Settings, or ask the follow-ups as typed.`
        : null;

  const pickedModel = model ?? models?.default ?? null;
  const pickedPrompt = prompt ?? prompts?.default ?? null;
  const profileLabels = Object.fromEntries((indexes?.profiles ?? []).map((p) => [p.id, p.label]));
  const unbuilt = (indexes?.profiles ?? []).filter((p) => !p.active);
  const pickedProfile = indexes?.profiles.find((p) => p.id === profile);
  // Without a word index only dense search can run; the choice falls back
  // visibly rather than sending a run the API refuses.
  const wordIndex = pickedProfile?.word_index ?? true;
  const pickedSearch: SearchMethod = wordIndex ? search : "dense";
  const methods = indexes?.search_methods ?? (["dense", "bm25", "hybrid"] as SearchMethod[]).map((id) => ({
    id, label: SEARCH_LABELS[id], explain: "",
  }));
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
      const { id } = await api.startRun({
        sets: chosen,
        mode,
        model: mode === "answer" ? pickedModel : null,
        profile,
        search: pickedSearch,
        section_cap: capped ? SECTION_CAP : null,
        ...(rerank ? { rerank } : {}),
        ...(rewrite ? { rewrite } : {}),
        ...(agentRounds ? { agent_rounds: agentRounds } : {}),
        ...(followupSets.length > 0 && asTyped ? { as_typed: true } : {}),
        ...(mode === "answer" && pickedPrompt ? { prompt: pickedPrompt } : {}),
      });
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
          Ask every test question with the chosen setup and score each answer. Runs use the CPU fully, so one
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
                    {s.remote_only && <span className="text-muted-foreground"> · needs Remote AI</span>}
                  </span>
                  <span className="block text-xs text-pretty text-muted-foreground">{s.description}</span>
                  {s.remote_only && remote && !remoteOk && (
                    <span className="block text-xs text-pretty text-amber-700 dark:text-amber-400">
                      Remote AI is not connected, so its follow-ups can only be asked as typed.
                    </span>
                  )}
                </span>
              </label>
            );
          })}
          {followupSets.length > 0 && (
            <div className="space-y-1 pt-1 pl-1.5">
              <label className="flex w-fit cursor-pointer items-center gap-2 text-sm">
                <Switch checked={asTyped} onCheckedChange={setAsTyped} />
                Ask follow-ups as typed
              </label>
              <p className="max-w-prose text-xs text-pretty text-muted-foreground">
                {asTyped
                  ? "Each follow-up is searched exactly as written (\"And for 2027?\"), with no rewrite: the run shows what the conversation's memory is worth."
                  : "Remote AI first makes each follow-up a question that stands alone, using the recorded earlier turn. The rewrite is checked, then retrieval, then the answer."}
              </p>
            </div>
          )}
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

        <div className="space-y-1.5">
          <label htmlFor="run-index" className="text-sm font-medium">
            Index
          </label>
          <Select items={profileLabels} value={profile} onValueChange={(v) => setProfile(v as string)}>
            <SelectTrigger id="run-index" className="w-full sm:w-72">
              <SelectValue placeholder="Choose an index" />
            </SelectTrigger>
            <SelectContent>
              {indexes?.profiles.map((p) => (
                <SelectItem key={p.id} value={p.id} disabled={!p.active || p.can?.query.ok === false}>
                  {p.label}
                  {!p.active ? " · not built yet" : p.can?.query.ok === false ? " · can't be asked now" : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {pickedProfile && <p className="max-w-prose text-xs text-pretty text-muted-foreground">{pickedProfile.description}</p>}
          {/* One line, not one per index: with 13 profiles a list of build
              commands crowded the form, and some build only on Colab. */}
          {unbuilt.length > 0 && (
            <p className="text-xs text-pretty text-muted-foreground">
              {unbuilt.length === 1 ? `${unbuilt[0].label} is` : `${unbuilt.length} indexes are`} not built yet: build
              them on the Indexes page.
            </p>
          )}
          {pickedProfile?.active && pickedProfile.can?.query.ok === false && (
            <p className="text-xs text-pretty text-amber-700 dark:text-amber-400">{pickedProfile.can.query.reason}</p>
          )}
        </div>

        <div className="space-y-1.5">
          <label htmlFor="run-search" className="text-sm font-medium">
            Search
          </label>
          <Select
            items={Object.fromEntries(methods.map((m) => [m.id, m.label]))}
            value={pickedSearch}
            onValueChange={(v) => v && setSearch(v as SearchMethod)}
          >
            <SelectTrigger id="run-search" className="w-full sm:w-72">
              <SelectValue placeholder="Choose a search method" />
            </SelectTrigger>
            <SelectContent>
              {methods.map((m) => (
                <SelectItem key={m.id} value={m.id} disabled={m.id !== "dense" && !wordIndex}>
                  <span className="flex flex-col">
                    <span>{m.label}</span>
                    {m.explain && <span className="text-[11px] text-muted-foreground">{m.explain}</span>}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {!wordIndex && pickedProfile?.active && (
            <p className="text-xs text-muted-foreground">
              This index has no word index yet, so BM25 and hybrid search are unavailable. Add it with{" "}
              <code className="rounded bg-muted px-1 py-0.5 font-mono">./rag ingest --profile {pickedProfile.id}</code>
            </p>
          )}
          <label className="flex w-fit cursor-pointer items-center gap-2 pt-1 text-sm">
            <Switch checked={capped} onCheckedChange={setCapped} />
            Limit per section
          </label>
          <p className="max-w-prose text-xs text-pretty text-muted-foreground">
            At most {SECTION_CAP} passages from one section ahead of the rest; contents pages and reference lists at
            most 1. Nothing is dropped: a limited passage moves down the list. Passages with no known section are never
            limited.
          </p>
        </div>

        <div className="space-y-1.5">
          <label htmlFor="run-rerank" className="text-sm font-medium">
            Reranker
          </label>
          <Select
            items={{ off: "Off", ...Object.fromEntries((rerankers?.rerankers ?? []).map((r) => [r.id, r.label])) }}
            value={rerank ?? "off"}
            onValueChange={(v) => setRerank(!v || v === "off" ? null : (v as string))}
          >
            <SelectTrigger id="run-rerank" className="w-full sm:w-72">
              <SelectValue placeholder="Off" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="off">Off</SelectItem>
              {(rerankers?.rerankers ?? []).map((r) => (
                <SelectItem key={r.id} value={r.id} disabled={!r.can.ok}>
                  <span className="flex flex-col">
                    <span>{r.label}</span>
                    {!r.can.ok && <span className="text-[11px] text-muted-foreground">{r.can.reason}</span>}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="max-w-prose text-xs text-pretty text-muted-foreground">
            A slower model reads each of the top {rerankers?.depth ?? 20} passages with the question and puts them in a
            better order. On this computer it adds 20 to 35 seconds to every question, and the run searches every
            question first, then answers them.
          </p>
        </div>

        <div className="space-y-1.5">
          <label htmlFor="run-rewrite" className="text-sm font-medium">
            Query rewriting
          </label>
          <Select
            items={{ off: "Off", ...REWRITE_LABELS }}
            value={rewrite ?? "off"}
            onValueChange={(v) => setRewrite(!v || v === "off" ? null : (v as string))}
          >
            <SelectTrigger id="run-rewrite" className="w-full sm:w-72">
              <SelectValue placeholder="Off" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="off">Off</SelectItem>
              {Object.entries(REWRITE_LABELS).map(([id, label]) => (
                <SelectItem key={id} value={id} disabled={!remoteOk}>
                  <span className="flex flex-col">
                    <span>{label}</span>
                    <span className="text-[11px] text-muted-foreground">
                      {remote && !remoteOk ? remote.reason : REWRITE_EXPLAIN[id]}
                    </span>
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <p className="max-w-prose text-xs text-pretty text-muted-foreground">
            Remote AI rewrites each question before it is searched, and the lists are combined. Every rewrite is
            remembered, so a later run with the same setup reuses it.
          </p>
        </div>

        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <label className="flex w-fit cursor-pointer items-center gap-2 text-sm font-medium">
              <Switch
                checked={agentRounds !== null}
                onCheckedChange={(on) => setAgentRounds(on ? 3 : null)}
                disabled={!remoteOk && agentRounds === null}
              />
              Multi-step search
            </label>
            {agentRounds !== null && (
              <>
                <label htmlFor="run-rounds" className="sr-only">
                  Rounds at most
                </label>
                <Select
                  items={Object.fromEntries(AGENT_ROUNDS.map((n) => [String(n), `Up to ${n} rounds`]))}
                  value={String(agentRounds)}
                  onValueChange={(v) => v && setAgentRounds(Number(v))}
                >
                  <SelectTrigger id="run-rounds" size="sm" className="w-36">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {AGENT_ROUNDS.map((n) => (
                      <SelectItem key={n} value={String(n)}>
                        Up to {n} rounds
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </>
            )}
          </div>
          <p className="max-w-prose text-xs text-pretty text-muted-foreground">
            {remote && !remoteOk && agentRounds === null
              ? remote.reason
              : "Round 1 is the normal search. Then Remote AI reads what was found and either answers or searches once more, perhaps in one document, until it has enough or reaches the limit."}
          </p>
        </div>

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

        {mode === "answer" && (
          <div className="space-y-1.5">
            <label htmlFor="run-prompt" className="text-sm font-medium">
              Prompt
            </label>
            <Select value={pickedPrompt} onValueChange={(v) => setPrompt(v as string)}>
              <SelectTrigger id="run-prompt" className="w-full sm:w-72">
                <SelectValue placeholder="Choose a prompt version" />
              </SelectTrigger>
              <SelectContent>
                {prompts?.versions.map((v) => (
                  <SelectItem key={v.version} value={v.version}>
                    {v.version}
                    {v.version === prompts.default ? " (default)" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              The system prompt is versioned to the byte: one blank line changed two answers. Every run records which
              one it used.
            </p>
            {pickedPrompt === "v4" && pickedProfile && !pickedProfile.labels && (
              <p className="text-xs text-muted-foreground">
                Prompt v4 tells the model each passage starts with its section label, and this index has none.
              </p>
            )}
          </div>
        )}

        <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-4">
          {blocked ? (
            <p className="max-w-prose text-xs text-pretty text-amber-700 dark:text-amber-400">{blocked}</p>
          ) : (
            <p className="text-xs text-muted-foreground">{estimate}</p>
          )}
          <Button
            onClick={start}
            disabled={
              active ||
              starting ||
              chosen.length === 0 ||
              (mode === "answer" && !pickedModel) ||
              !pickedProfile?.active ||
              blocked !== null
            }
          >
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
