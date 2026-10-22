"use client";

import { useState } from "react";
import useSWR from "swr";
import {
  AlertTriangle, CheckCircle2, ChevronDown, Cloud, Copy, Cpu, ExternalLink, Laptop, Loader2, PlugZap,
  RefreshCw, Unplug, XCircle, Zap,
} from "lucide-react";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { StatusBadge } from "@/components/common/status";
import {
  api, ApiError, type ColabCell, type ConnectResult, type LlmSnapshot, type RemoteProbe, type SmokeTest,
} from "@/lib/api";
import { NO_DATA, num, relativeTime } from "@/lib/format";
import { cn } from "@/lib/utils";

/**
 * Where the AI models run, and how to move work to a Colab GPU.
 *
 * The browser half of `./rag llm`: every button posts to the local API, which
 * runs the same Python as the terminal, so the two can never disagree about
 * which computer is embedding.
 *
 * One rule shapes this file: **a credential travels browser -> API, never
 * API -> browser.** The Connect field sends a code the user pasted; nothing
 * this page renders has ever held a token, and the setup cell offered here is
 * the tokenless one. See docs/02-remote-gpu.md.
 */

const COLAB_URL = "https://colab.research.google.com/#create=true";

function seconds(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return NO_DATA;
  return s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`;
}

/** What to do about a failure, by its kind (the stage says how far it got). */
function Remedy({ kind }: { kind?: string | null }) {
  const text: Record<string, string> = {
    unreachable:
      "The runtime is gone or its tunnel is down: Colab reclaims an idle runtime after about 90 minutes. " +
      "Run the same setup cell again and connect with what it shows.",
    timeout: "It accepted the request but did not finish in time. A runtime without a GPU is that slow; check the GPU line.",
    auth: "The runtime refused the token. A runtime that made its own token issues a new one each time: paste its new connect code.",
    model: "The runtime is up but does not serve the model. Run the setup cell with that model.",
    not_connected: "Nothing is connected. Use Connect with the code the setup cell showed.",
  };
  const body = kind ? text[kind] : null;
  return body ? <p className="text-xs text-pretty text-muted-foreground">{body}</p> : null;
}

// --------------------------------------------------------------------------- connect

function ConnectDialog({
  open, onOpenChange, tokenRemembered, codePrefix, onConnected,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  tokenRemembered: boolean;
  codePrefix: string;
  onConnected: (result: ConnectResult) => void;
}) {
  const [value, setValue] = useState("");
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const trimmed = value.trim();
  const isCode = trimmed.startsWith(codePrefix);
  const isUrl = /^https?:\/\//.test(trimmed);
  // A bare URL carries no token. This computer remembers the one it put in
  // the last cell it made (./rag llm colab), which is what makes a reconnect
  // one field.
  const needsToken = isUrl && !tokenRemembered;

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const result = await api.llmConnect(isCode ? { code: trimmed } : { url: trimmed, token: token.trim() || undefined });
      onConnected(result);
      if (result.ok) {
        const first = result.embedded[0];
        toast.success("Remote AI is connected", {
          description: first ? `${first.model} embedded a test sentence in ${seconds(first.seconds)}.` : undefined,
        });
        onOpenChange(false);
        setValue("");
        setToken("");
      } else {
        // Saved but not working: kept open, because the next step is fixing
        // the runtime and pressing Connect again.
        setError(result.error ?? "The runtime was saved but did not answer.");
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "App service unavailable. Start it with ./rag web.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {/* A connect code is one unbroken base64 string. Grid items default to
          min-width: auto, and the textarea sizes to its content, so a pasted
          code once widened the dialog to 3,155 px and pushed Connect off
          screen (QA M4-01). Fixed size, break anywhere, min-w-0, and the
          dialog scrolls if it is ever taller than the window. */}
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Connect a Colab GPU</DialogTitle>
          <DialogDescription>
            Paste the connect code the setup cell showed. It goes to this app&apos;s own service on 127.0.0.1,
            which checks it and keeps it outside the project folder. Your browser never talks to the runtime.
          </DialogDescription>
        </DialogHeader>

        <div className="min-w-0 space-y-3">
          <div className="min-w-0 space-y-1.5">
            <Label htmlFor="connect-target">Connect code, or the runtime&apos;s URL</Label>
            <Textarea
              id="connect-target"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder={`${codePrefix}eyJ1Ijoi…   or   https://xxxx.trycloudflare.com`}
              rows={3}
              className="max-h-40 resize-none overflow-y-auto font-mono text-xs break-all field-sizing-fixed"
              spellCheck={false}
              autoComplete="off"
              autoFocus
            />
            <p className="text-xs text-muted-foreground" aria-live="polite">
              {isCode
                ? "A connect code: it carries the URL and the token together."
                : isUrl
                  ? tokenRemembered
                    ? "A URL. This computer will use the token it remembers, from the last connection or the last ./rag llm colab cell."
                    : "A URL with no remembered token: give the token below, or paste the whole connect code."
                  : `A connect code starts with ${codePrefix}; a URL starts with https://.`}
            </p>
          </div>
          {needsToken && (
            <div className="space-y-1.5">
              <Label htmlFor="connect-token">Token</Label>
              <Input
                id="connect-token"
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                className="font-mono text-xs"
                spellCheck={false}
                autoComplete="off"
              />
            </div>
          )}
          {error && (
            <Alert variant="destructive" live="polite">
              <AlertTriangle className="size-4" aria-hidden />
              <AlertTitle>Not connected yet</AlertTitle>
              <AlertDescription className="text-xs break-words">{error}</AlertDescription>
            </Alert>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={busy}>
            Cancel
          </Button>
          <Button
            onClick={() => void submit()}
            disabled={busy || (!isCode && !isUrl) || (needsToken && !token.trim())}
            aria-busy={busy}
          >
            {busy ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <Zap className="size-4" aria-hidden />}
            {busy ? "Connecting and testing…" : "Connect and test"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// --------------------------------------------------------------------------- setup

/**
 * What the runtime is for. Each is one setup cell: the models Ollama serves
 * there and the GPU worker's jobs (tools/colab/worker.py). qwen3:8b and the
 * 4B reranker are loaded one at a time on a T4.
 */
const JOBS: { id: string; label: string; what: string; models?: string; worker?: string }[] = [
  { id: "embed", label: "Embeddings", what: "Builds and searches the indexes made on Colab (Qwen3-Embedding 0.6B and 8B)." },
  {
    id: "rewrite",
    label: "Rewriting and context lines",
    what: "qwen3:8b rewrites questions and follow-ups, steers multi-step search, and writes context lines.",
    models: "qwen3:8b",
    worker: "context",
  },
  {
    id: "rerank",
    label: "Larger reranker",
    what: "Qwen3-Reranker 4B on the GPU. Starts fastest: it installs no Ollama.",
    worker: "qwen3-4b",
  },
  {
    id: "all",
    label: "Rewriting and rerankers",
    what: "Both of the above, plus the 0.6B reranker. The 4B swaps in when a run asks for it.",
    models: "qwen3:8b",
    worker: "context,qwen3-0.6b-gpu,qwen3-4b",
  },
];

function ColabSetup() {
  const [cell, setCell] = useState<ColabCell | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [jobId, setJobId] = useState(JOBS[0].id);
  const job = JOBS.find((j) => j.id === jobId) ?? JOBS[0];

  const load = async (next = job, force = false) => {
    if ((cell && !force) || loading) return;
    setLoading(true);
    try {
      setCell(await api.llmColab(next.models || next.worker ? { models: next.models, worker: next.worker } : undefined));
    } catch (e) {
      toast.error("Couldn't build the setup cell.", { description: e instanceof Error ? e.message : undefined });
    } finally {
      setLoading(false);
    }
  };

  const copy = async () => {
    if (!cell) return;
    try {
      await navigator.clipboard.writeText(cell.cell);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast.success(`Setup cell copied (${cell.kilobytes} KB)`);
    } catch {
      toast.error("Couldn't copy to the clipboard. Select the cell below and copy it.");
    }
  };

  return (
    <Collapsible onOpenChange={(open) => open && void load()}>
      <CollapsibleTrigger
        render={
          <button
            type="button"
            className="group flex w-full items-center justify-between gap-2 rounded-md py-1 text-left text-sm font-medium focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
          />
        }
      >
        <span className="flex items-center gap-2">
          <Cloud className="size-4 text-muted-foreground" aria-hidden />
          Set up a Colab GPU runtime
        </span>
        <ChevronDown className="size-4 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]:rotate-180" aria-hidden />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="space-y-3 pt-3 text-sm">
          <ol className="list-decimal space-y-1.5 pl-5 text-pretty text-muted-foreground">
            <li>
              Open a new notebook, then <strong className="text-foreground">Runtime › Change runtime type › T4 GPU</strong>.
              Do this first: Colab hands out runtimes without a GPU without saying so.
            </li>
            <li>Copy the setup cell, paste it into one code cell, and run it (about 3-5 minutes).</li>
            <li>
              When it has checked the lock on its tunnel, it shows a connect code once. Paste that into{" "}
              <strong className="text-foreground">Connect</strong>.
            </li>
          </ol>
          <fieldset className="space-y-1.5">
            <legend className="text-xs font-medium text-muted-foreground">What the runtime is for</legend>
            <div className="grid gap-1.5 sm:grid-cols-2">
              {JOBS.map((j) => (
                <label
                  key={j.id}
                  className={cn(
                    "flex cursor-pointer gap-2 rounded-md p-2 text-xs ring-1 ring-foreground/10 has-[:focus-visible]:ring-[3px] has-[:focus-visible]:ring-ring/50",
                    j.id === jobId && "bg-muted ring-foreground/30",
                  )}
                >
                  <input
                    type="radio"
                    name="colab-job"
                    value={j.id}
                    checked={j.id === jobId}
                    onChange={() => {
                      setJobId(j.id);
                      setCell(null);
                      void load(j, true);
                    }}
                    className="mt-0.5 accent-foreground"
                  />
                  <span>
                    <span className="block font-medium">{j.label}</span>
                    <span className="block text-pretty text-muted-foreground">{j.what}</span>
                  </span>
                </label>
              ))}
            </div>
          </fieldset>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" onClick={() => void copy()} disabled={!cell}>
              {loading ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : copied ? (
                <CheckCircle2 className="size-4" aria-hidden />
              ) : (
                <Copy className="size-4" aria-hidden />
              )}
              {copied ? "Copied" : cell ? `Copy the setup cell (${cell.kilobytes} KB)` : "Building…"}
            </Button>
            <a
              href={COLAB_URL}
              target="_blank"
              rel="noreferrer noopener"
              className="inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-sm font-medium hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
            >
              <ExternalLink className="size-4" aria-hidden />
              Open Colab
            </a>
          </div>
          <Alert>
            <AlertTriangle className="size-4" aria-hidden />
            <AlertTitle className="text-sm">This cell carries no token</AlertTitle>
            <AlertDescription className="space-y-1.5 text-xs text-pretty">
              <p>
                The runtime makes its own and shows it once, inside the connect code, in an output it blanks when you
                press Enter. Copy it first. No credential travels from this app to your browser, and none stays in the
                notebook Colab saves to your Drive.
              </p>
              <p>
                For a one-field reconnect after Colab reclaims the runtime, make the cell in a terminal instead:{" "}
                <code className="rounded bg-muted px-1 font-mono">./rag llm colab --copy</code>. That cell carries a
                token this computer remembers, so a new runtime needs only its URL pasted back.
              </p>
            </AlertDescription>
          </Alert>
          {cell && (
            <>
              <p className="text-xs text-pretty text-muted-foreground">
                {cell.models.length > 0 ? (
                  <>
                    It serves{" "}
                    {cell.models.map((m, i) => (
                      <span key={m}>
                        {i > 0 && (i === cell.models.length - 1 ? " and " : ", ")}
                        <code className="rounded bg-muted px-1 font-mono">{m}</code>
                      </span>
                    ))}{" "}
                    on Ollama {cell.ollama_version ?? "(latest)"}, the same version as this computer, so both sides run
                    the same model code.
                  </>
                ) : (
                  <>It installs no Ollama.</>
                )}
                {cell.worker && cell.worker.length > 0 && (
                  <>
                    {" "}
                    Its GPU worker runs <code className="rounded bg-muted px-1 font-mono">{cell.worker.join(", ")}</code>{" "}
                    with the same library versions as this computer, pinned to exact model revisions.
                  </>
                )}{" "}
                It writes {cell.files.join(", ")} and checks each against a digest first, so a truncated paste stops
                instead of running half a script.
              </p>
              <pre
                tabIndex={0}
                aria-label="The Colab setup cell"
                className="max-h-52 overflow-auto rounded-lg bg-muted p-3 font-mono text-[10px] leading-relaxed focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none"
              >
                {cell.cell}
              </pre>
            </>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

// --------------------------------------------------------------------------- the card

function RuntimeFacts({ probe }: { probe: RemoteProbe }) {
  const gpu = probe.gpu?.gpus?.[0];
  return (
    <>
      <dt className="text-muted-foreground">GPU</dt>
      <dd className="min-w-0 text-xs">
        {gpu ? (
          <>
            {gpu.name} · {num(gpu.memory_used_mb)} / {num(gpu.memory_total_mb)} MB in use
          </>
        ) : (
          <span className="text-amber-700 dark:text-amber-400">None: {probe.gpu?.reason ?? "not reported"}</span>
        )}
      </dd>
      <dt className="text-muted-foreground">Models</dt>
      <dd className="min-w-0 space-y-0.5 text-xs">
        {(probe.models ?? []).length === 0 && NO_DATA}
        {(probe.models ?? []).map((m) => {
          const loaded = probe.loaded?.find((l) => l.name === m.name);
          return (
            <p key={m.name} className="flex flex-wrap items-baseline gap-x-2">
              <span className="font-mono">{m.name}</span>
              <span className={cn("text-muted-foreground", loaded && !loaded.on_gpu && "text-red-700 dark:text-red-400")}>
                {loaded ? (loaded.on_gpu ? "loaded on the GPU" : "loaded on the CPU") : "not loaded"}
              </span>
            </p>
          );
        })}
      </dd>
      <dt className="text-muted-foreground">Ollama</dt>
      <dd className="text-xs">
        {probe.ollama_version ?? NO_DATA}
        {probe.latency_ms != null && <span className="text-muted-foreground"> · answered in {num(probe.latency_ms)} ms</span>}
      </dd>
      {probe.worker && (
        <>
          <dt className="text-muted-foreground">GPU worker</dt>
          <dd className="min-w-0 text-xs">
            <span className="font-mono">{probe.worker.served.join(", ") || "no reranker"}</span>
            {probe.worker.context && <span className="text-muted-foreground"> · writes context lines</span>}
            <span className="text-muted-foreground">
              {" "}
              · {probe.worker.loaded ? `${probe.worker.loaded} loaded` : "nothing loaded"}
              {probe.worker.vram?.peak_mb ? `, peak ${num(probe.worker.vram.peak_mb)} MB` : ""}
            </span>
          </dd>
        </>
      )}
    </>
  );
}

export function ModelBackend() {
  const { data, error, isLoading, mutate } = useSWR<LlmSnapshot>("/llm/status");
  const connected = Boolean(data?.connection.connected);
  // The GPU and model facts cost one request through the tunnel: asked once
  // when the page opens, and again on Refresh, never on a timer.
  const probed = useSWR<LlmSnapshot>(connected ? "/llm/status?probe=true" : null, {
    revalidateOnFocus: false,
    revalidateIfStale: false,
  });
  const [connectOpen, setConnectOpen] = useState(false);
  const [busy, setBusy] = useState<"test" | "refresh" | "disconnect" | null>(null);
  const [test, setTest] = useState<SmokeTest | null>(null);

  const adopt = (snapshot: LlmSnapshot) => {
    void mutate({ connection: snapshot.connection, remote: null }, { revalidate: false });
    if (snapshot.remote) void probed.mutate(snapshot, { revalidate: false });
  };

  const runTest = async () => {
    setBusy("test");
    setTest(null);
    try {
      const result = await api.llmTest();
      setTest(result);
      adopt(result);
      if (result.ok) toast.success(`Remote AI answered in ${seconds(result.seconds)}`);
      else toast.error(`The test failed at the ${result.stage} step`);
    } catch (e) {
      toast.error("App service unavailable, so the test could not run.", {
        description: e instanceof Error ? e.message : undefined,
      });
    } finally {
      setBusy(null);
    }
  };

  const refresh = async () => {
    setBusy("refresh");
    try {
      adopt(await api.llmStatus(true));
    } catch {
      toast.error("Couldn't refresh. Try again.");
    } finally {
      setBusy(null);
    }
  };

  const disconnect = async () => {
    setBusy("disconnect");
    try {
      adopt(await api.llmDisconnect());
      setTest(null);
      toast.success("Disconnected. Everything runs on this computer.");
    } catch (e) {
      toast.error("Couldn't disconnect.", { description: e instanceof Error ? e.message : undefined });
    } finally {
      setBusy(null);
    }
  };

  if (error) return <ErrorState title="Couldn't load the model backend" message={error.message} error={error} onRetry={() => mutate()} />;
  if (isLoading || !data) return <CardSkeleton height="h-64" />;

  const conn = data.connection;
  const probe = probed.data?.remote ?? null;
  const reachable = probe?.reachable;
  const onCpu = (probe?.loaded ?? []).filter((m) => !m.on_gpu).map((m) => m.name);
  const noGpu = probe?.reachable && !probe.gpu?.gpus?.length;
  // A runtime set up for another job (rewriting, a reranker) does not serve the Colab indexes'
  // embedders: worth knowing, not a fault (M5-01).
  const missing = probe?.missing ?? [];
  const status = !connected
    ? null
    : probe == null
      ? probed.isLoading
        ? "running"
        : "info"
      : !reachable
        ? "unavailable"
        : noGpu || onCpu.length
          ? "warning"
          : "healthy";

  return (
    <Card
      id="model-backend"
      className={cn("scroll-mt-20", status === "unavailable" && "ring-red-500/40", status === "warning" && "ring-amber-500/40")}
    >
      <CardHeader>
        <CardTitle as="h2" className="flex flex-wrap items-center gap-2 text-base">
          Model backend
          <Badge
            variant="outline"
            className={cn(
              "gap-1 font-normal",
              connected ? "border-violet-500/40 text-violet-700 dark:text-violet-400" : "border-sky-500/40 text-sky-700 dark:text-sky-400",
            )}
          >
            {connected ? <Cloud className="size-3" aria-hidden /> : <Laptop className="size-3" aria-hidden />}
            {connected ? "Remote AI connected" : "Local AI only"}
          </Badge>
          {status && status !== "info" && (
            <StatusBadge
              status={status}
              label={status === "running" ? "Checking" : status === "healthy" ? "Healthy" : undefined}
            />
          )}
        </CardTitle>
        <CardDescription className="text-pretty">
          Which computer runs the AI models. A Colab GPU you start yourself builds the large indexes in seconds and
          embeds their questions; passages, citations and answers stay on this computer.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {!connected ? (
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="flex items-start gap-3 rounded-lg p-4 ring-1 ring-foreground/10">
              <Laptop className="mt-0.5 size-5 shrink-0 text-muted-foreground" aria-hidden />
              <div className="min-w-0 space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="font-medium">Local AI</p>
                  <StatusBadge status="healthy" label="In use" />
                </div>
                <p className="text-sm text-pretty text-muted-foreground">
                  Ollama and the in-app models on this computer. Private and always there, but slow for large models.
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3 rounded-lg border border-dashed p-4">
              <Cloud className="mt-0.5 size-5 shrink-0 text-muted-foreground" aria-hidden />
              <div className="min-w-0 space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="font-medium">Remote AI</p>
                  <StatusBadge status="info" label="Not connected" />
                </div>
                <p className="text-sm text-pretty text-muted-foreground">
                  A Colab T4 GPU. Needed for the Qwen3-Embedding 8B indexes; optional for the rest.
                </p>
              </div>
            </div>
          </div>
        ) : (
          <dl className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-sm">
            <dt className="text-muted-foreground">Runtime</dt>
            <dd className="min-w-0 font-mono text-xs break-all">
              {conn.host}
              {conn.authenticated && <span className="ml-2 font-sans text-muted-foreground">locked with a token</span>}
            </dd>
            {probe?.reachable && <RuntimeFacts probe={probe} />}
            <dt className="text-muted-foreground">Connected</dt>
            <dd className="text-xs text-muted-foreground">{relativeTime(conn.connected_at)}</dd>
            <dt className="text-muted-foreground">Last test</dt>
            <dd className="text-xs">
              {conn.last_error ? (
                <span className="text-red-700 dark:text-red-400">
                  Failed {relativeTime(conn.last_error_at)}: {conn.last_error}
                </span>
              ) : conn.last_ok_at ? (
                <span className="text-muted-foreground">Passed {relativeTime(conn.last_ok_at)}</span>
              ) : (
                <span className="text-muted-foreground">Never tested</span>
              )}
            </dd>
          </dl>
        )}

        {connected && probe && !probe.reachable && (
          <Alert variant="destructive" live="polite">
            <XCircle className="size-4" aria-hidden />
            <AlertTitle className="text-sm">The runtime is not answering</AlertTitle>
            <AlertDescription className="space-y-1.5">
              <p className="text-xs break-words">{probe.error}</p>
              <Remedy kind={probe.error_kind} />
            </AlertDescription>
          </Alert>
        )}
        {(noGpu || onCpu.length > 0) && (
          <Alert className="border-amber-500/50 bg-amber-500/10">
            <AlertTriangle className="size-4 text-amber-600 dark:text-amber-400" aria-hidden />
            <AlertTitle className="text-sm text-amber-700 dark:text-amber-400">
              {noGpu ? "This runtime has no GPU" : `${onCpu.join(", ")} runs on the runtime's CPU`}
            </AlertTitle>
            <AlertDescription className="text-xs text-pretty">
              It works, at about the speed of this computer. In Colab choose Runtime › Change runtime type › T4 GPU,
              then run the setup cell again.
            </AlertDescription>
          </Alert>
        )}
        {missing.length > 0 && (
          <p className="text-xs text-pretty text-muted-foreground">
            This runtime does not serve <span className="font-mono">{missing.join(", ")}</span>, so the indexes built on
            Colab cannot be asked while it is connected. To use them, run the setup cell for Embeddings.
          </p>
        )}

        {test && (
          <div
            className={cn(
              "space-y-1 rounded-lg border px-3 py-2 text-xs",
              test.ok ? "border-emerald-500/40 bg-emerald-500/5" : "border-red-500/40 bg-red-500/5",
            )}
            role="status"
          >
            <p className="flex items-center gap-1.5 text-sm font-medium">
              {test.ok ? (
                <CheckCircle2 className="size-4 text-emerald-600 dark:text-emerald-400" aria-hidden />
              ) : (
                <XCircle className="size-4 text-red-600 dark:text-red-400" aria-hidden />
              )}
              {test.ok ? `Tested each model the runtime serves in ${seconds(test.seconds)}` : `Failed at the ${test.stage} step`}
            </p>
            {test.ok &&
              test.embedded.map((r) => (
                <p key={r.model} className="text-muted-foreground">
                  <span className="font-mono">{r.model}</span> embedded a sentence · {num(r.dimensions)} dimensions ·{" "}
                  {seconds(r.seconds)}
                </p>
              ))}
            {test.ok &&
              (test.chatted ?? []).map((r) => (
                <p key={r.model} className="text-muted-foreground">
                  <span className="font-mono">{r.model}</span> answered a one-word chat · {seconds(r.seconds)}
                </p>
              ))}
            {test.ok && test.worker && (
              <p className="text-muted-foreground">
                GPU worker serves <span className="font-mono">{test.worker.served.join(", ") || "no reranker"}</span>
                {test.worker.loaded ? ` · ${test.worker.loaded} loaded` : ""}
              </p>
            )}
            {!test.ok && (
              <>
                <p className="break-words text-red-700 dark:text-red-400">{test.error}</p>
                <Remedy kind={test.error_kind} />
              </>
            )}
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          <Button size="sm" onClick={() => setConnectOpen(true)}>
            <PlugZap className="size-4" aria-hidden />
            {connected ? "Reconnect" : "Connect a GPU"}
          </Button>
          {connected && (
            <>
              <Button size="sm" variant="outline" onClick={() => void runTest()} disabled={busy !== null} aria-busy={busy === "test"}>
                {busy === "test" ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <Zap className="size-4" aria-hidden />}
                {busy === "test" ? "Testing…" : "Test"}
              </Button>
              <Button size="sm" variant="outline" onClick={() => void disconnect()} disabled={busy !== null}>
                {busy === "disconnect" ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <Unplug className="size-4" aria-hidden />}
                Disconnect
              </Button>
              <Button size="sm" variant="ghost" onClick={() => void refresh()} disabled={busy !== null} aria-busy={busy === "refresh"}>
                <RefreshCw className={cn("size-4", busy === "refresh" && "animate-spin")} aria-hidden />
                Refresh
              </Button>
            </>
          )}
        </div>

        <div className="rounded-lg bg-muted/40 text-xs ring-1 ring-foreground/5">
          <p className="px-3 pt-2 pb-1 font-medium">Where each job runs</p>
          <ul className="divide-y">
            <li className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 px-3 py-1.5">
              <span className="w-24 shrink-0 font-medium">Embedding</span>
              <span className="min-w-0 flex-1 text-pretty text-muted-foreground">
                Set by each index: the Qwen3-Embedding 8B indexes on Remote AI, every other index on this computer.
              </span>
            </li>
            <li className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 px-3 py-1.5">
              <span className="w-24 shrink-0 font-medium">Rewriting</span>
              <span className="min-w-0 flex-1 text-pretty text-muted-foreground">
                Remote AI only (qwen3:8b): query rewriting, follow-up questions, multi-step search and context lines.
                Without it these switches are off and say why.
              </span>
            </li>
            <li className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 px-3 py-1.5">
              <span className="w-24 shrink-0 font-medium">Reranker</span>
              <span className="min-w-0 flex-1 text-pretty text-muted-foreground">
                The one you pick: the 0.6B on this computer (about 39 s a question), or the 0.6B, 4B or 8B on Remote
                AI&apos;s GPU worker (seconds).
              </span>
            </li>
            <li className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 px-3 py-1.5">
              <span className="w-24 shrink-0 font-medium">Answer, judge</span>
              <span className="min-w-0 flex-1 text-pretty text-muted-foreground">
                <Cpu className="mr-1 inline size-3 align-[-1px]" aria-hidden />
                Local AI.
              </span>
            </li>
            <li className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 px-3 py-1.5">
              <span className="w-24 shrink-0 font-medium">Fallback</span>
              <span className="min-w-0 flex-1 text-pretty text-muted-foreground">
                Strict. Nothing switches to another model or computer on its own: an index that needs the GPU says so
                instead.
              </span>
            </li>
          </ul>
        </div>

        <p className="text-xs text-pretty text-muted-foreground">
          The same from a terminal: <code className="rounded bg-muted px-1 font-mono">./rag llm status</code>,{" "}
          <code className="rounded bg-muted px-1 font-mono">connect</code>, <code className="rounded bg-muted px-1 font-mono">test</code>,{" "}
          <code className="rounded bg-muted px-1 font-mono">disconnect</code>. The connection is kept in{" "}
          <code className="rounded bg-muted px-1 font-mono">{conn.state_path}</code>, readable by you only.
        </p>

        <Separator />
        <ColabSetup />
      </CardContent>

      <ConnectDialog
        open={connectOpen}
        onOpenChange={setConnectOpen}
        tokenRemembered={conn.token_remembered}
        codePrefix={conn.code_prefix}
        onConnected={(result) => {
          void mutate({ connection: result.connection, remote: null }, { revalidate: false });
          if (result.remote) void probed.mutate({ connection: result.connection, remote: result.remote }, { revalidate: false });
          setTest(result);
        }}
      />
    </Card>
  );
}
