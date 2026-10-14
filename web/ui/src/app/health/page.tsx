"use client";

import { useState } from "react";
import useSWR from "swr";
import { Check, Copy, RefreshCw, Stethoscope } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { PageHeader } from "@/components/common/page-header";
import { CardSkeleton, ErrorState } from "@/components/common/states";
import { STATUS, StatusBadge, StatusIcon } from "@/components/common/status";
import { TechnicalDetails } from "@/components/common/technical-details";
import { RolesCard } from "@/components/settings/roles-card";
import type { HealthCheck, HealthReport, InstalledModel, Status } from "@/lib/api";
import { NO_DATA, bytes, dateTimeLabel, relativeTime, tokens } from "@/lib/format";
import { cn } from "@/lib/utils";

/** Ollama's capability codes, in the app's words. */
const CAPABILITY: Record<string, string> = {
  completion: "Text",
  embedding: "Embedding",
  thinking: "Thinking",
  tools: "Tools",
  vision: "Images",
  insert: "Fill-in",
};

const OVERALL: Record<Status, string> = {
  healthy: "Healthy",
  running: "Running",
  warning: "Working, with warnings",
  needs_attention: "Needs attention",
  unavailable: "Unavailable",
  info: "Healthy",
};

export default function HealthPage() {
  const { data, error, isLoading, mutate, isValidating } = useSWR<HealthReport>("/health");

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <PageHeader
        href="/health"
        actions={
          <Button size="sm" variant="outline" onClick={() => mutate()} disabled={isValidating} aria-busy={isValidating}>
            <RefreshCw className={cn("size-4", isValidating && "animate-spin")} aria-hidden />
            {isValidating ? "Checking..." : "Check again"}
          </Button>
        }
      />

      {error ? (
        <ErrorState message={error.message} error={error} onRetry={() => mutate()} />
      ) : isLoading || !data ? (
        <div className="space-y-4">
          <CardSkeleton height="h-24" />
          <CardSkeleton height="h-64" />
        </div>
      ) : (
        <HealthReportView data={data} />
      )}
    </div>
  );
}

function HealthReportView({ data }: { data: HealthReport }) {
  const s = STATUS[data.overall];
  const shown = (["healthy", "running", "warning", "needs_attention", "unavailable", "info"] as Status[]).filter(
    (k) => data.counts[k] > 0,
  );

  return (
    <>
      <Card className={cn(data.overall !== "healthy" && ["border", s.ring])}>
        <CardContent className="flex flex-wrap items-center gap-x-8 gap-y-4">
          <div className="flex items-center gap-3">
            <div className={cn("flex size-11 items-center justify-center rounded-full", s.bg)}>
              <Stethoscope className={cn("size-5", s.text)} aria-hidden />
            </div>
            <div>
              <p className="font-medium">{OVERALL[data.overall]}</p>
              <p className="text-xs text-muted-foreground">Checked {dateTimeLabel(data.checked_at)}</p>
            </div>
          </div>
          <ul className="flex flex-wrap gap-x-5 gap-y-2 text-sm" aria-label="Checks by status">
            {shown.map((k) => (
              <li key={k} className="flex items-center gap-1.5">
                <StatusIcon status={k} />
                <span className="font-semibold tabular-nums">{data.counts[k]}</span>
                <span className="text-muted-foreground">{STATUS[k].label.toLowerCase()}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        {data.sections.map((section) => (
          <Card key={section.id}>
            <CardHeader>
              <CardTitle as="h2" className="text-base">{section.name}</CardTitle>
            </CardHeader>
            <CardContent className="divide-y">
              {section.checks.map((check) => (
                <CheckRow key={check.id} check={check} />
              ))}
            </CardContent>
          </Card>
        ))}
      </div>

      <RolesCard roles={data.roles} />
      <ModelsCard data={data} />

      <TechnicalDetails
        label="About this app"
        rows={{
          Version: data.app.version,
          Python: data.app.python,
          Platform: data.app.platform,
          Ollama: data.ollama.version ?? NO_DATA,
          Endpoint: data.app.ollama_url,
        }}
      />
      <p className="text-xs text-muted-foreground">
        Suggested fixes are shown, never run for you. Run them yourself in a terminal so nothing changes
        your system without you seeing it.
      </p>
    </>
  );
}

function CheckRow({ check }: { check: HealthCheck }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(check.fix ?? "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
      toast.success("Command copied.");
    } catch {
      toast.error("Couldn't copy. Select the command and copy it by hand.");
    }
  };
  const technical = Object.fromEntries(Object.entries(check.technical).map(([k, v]) => [k, String(v)]));

  return (
    <div className="flex items-start gap-3 py-3 first:pt-0 last:pb-0">
      <StatusIcon status={check.status} className="mt-0.5" />
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
          <p className="text-sm font-medium">{check.label}</p>
          <StatusBadge status={check.status} className="text-[10px]" />
        </div>
        <p className="text-sm text-pretty text-muted-foreground">{check.message}</p>
        {check.fix && (
          <div className="flex flex-wrap items-center gap-2">
            <code className="rounded bg-muted px-2 py-1 font-mono text-[11px] break-all">{check.fix}</code>
            <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-xs" onClick={copy}>
              {copied ? <Check className="size-3 text-emerald-500" aria-hidden /> : <Copy className="size-3" aria-hidden />}
              Copy fix
            </Button>
          </div>
        )}
        <TechnicalDetails rows={technical} />
      </div>
    </div>
  );
}

function ModelsCard({ data }: { data: HealthReport }) {
  const roleOf = (m: InstalledModel) =>
    data.roles
      .filter((r) => r.model && (r.model === m.name || `${r.model}:latest` === m.name))
      .map((r) => r.label);

  if (!data.ollama.up) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle as="h2" className="text-base">Installed models</CardTitle>
        <CardDescription>
          {data.ollama.installed.length} models in Ollama on this computer.{" "}
          {data.ollama.resident.length > 0
            ? `${data.ollama.resident.length} loaded in memory right now.`
            : "None loaded in memory right now."}
        </CardDescription>
      </CardHeader>
      {/* Columns respond to the card's width, not the window's: with the
          sidebar open at 768px the card is only ~464px wide (QA M0-04). */}
      <CardContent className="@container px-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-4">Model</TableHead>
              <TableHead className="hidden text-right @sm:table-cell">Size</TableHead>
              <TableHead className="hidden text-right @xl:table-cell">Context</TableHead>
              <TableHead className="pr-4 @2xl:pr-2">Can do</TableHead>
              <TableHead className="hidden pr-4 @2xl:table-cell">Used for</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.ollama.installed.map((m) => {
              const resident = data.ollama.resident.find((r) => r.name === m.name);
              const roles = roleOf(m);
              return (
                <TableRow key={m.name}>
                  <TableCell className="pl-4 align-top whitespace-normal">
                    <p className="font-mono text-xs font-medium whitespace-nowrap">{m.name}</p>
                    <p className="text-[11px] text-muted-foreground">
                      {[m.parameter_size, m.quantization].filter(Boolean).join(" · ") || NO_DATA}
                      <span className="@sm:hidden"> · {bytes(m.size)}</span>
                      {resident && ` · loaded, unloads ${relativeTime(resident.expires_at)}`}
                    </p>
                    {/* When the "Used for" column is hidden, its content moves here. */}
                    {roles.length > 0 && (
                      <p className="text-[11px] text-muted-foreground @2xl:hidden">Used for {roles.join(", ")}</p>
                    )}
                  </TableCell>
                  <TableCell className="hidden text-right align-top tabular-nums @sm:table-cell">{bytes(m.size)}</TableCell>
                  <TableCell className="hidden text-right align-top tabular-nums @xl:table-cell">
                    {tokens(m.context_length)}
                  </TableCell>
                  <TableCell className="pr-4 align-top whitespace-normal @2xl:pr-2">
                    <div className="flex flex-wrap gap-1">
                      {m.capabilities.map((c) => (
                        <Badge key={c} variant="secondary" className="text-[10px] font-normal">
                          {CAPABILITY[c] ?? c}
                        </Badge>
                      ))}
                    </div>
                  </TableCell>
                  <TableCell className="hidden pr-4 align-top text-xs whitespace-normal text-muted-foreground @2xl:table-cell">
                    {roles.length ? roles.join(", ") : NO_DATA}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
