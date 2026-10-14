"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import useSWR from "swr";
import { AlertTriangle, CheckCircle2, CircleDot, Layers, XCircle } from "lucide-react";

import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupContent,
  SidebarGroupLabel, SidebarHeader, SidebarMenu, SidebarMenuButton, SidebarMenuItem,
  SidebarRail, useSidebar,
} from "@/components/ui/sidebar";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { AppStatus } from "@/lib/api";
import { APP_NAME } from "@/lib/routes";
import { NAV, isActive, navLabel } from "./nav";

type Tone = "ok" | "warn" | "bad" | "muted";

function StatusRow({
  label, value, tone, hint, href,
}: {
  label: string;
  value: string;
  tone: Tone;
  hint?: string;
  /** Where the fix is. A red row you cannot act on is only an alarm. */
  href?: string;
}) {
  const Icon = tone === "ok" ? CheckCircle2 : tone === "warn" ? AlertTriangle : tone === "bad" ? XCircle : CircleDot;
  const colour =
    tone === "ok" ? "text-emerald-500"
    : tone === "warn" ? "text-amber-500"
    : tone === "bad" ? "text-red-500"
    : "text-muted-foreground";

  const contents = (
    <>
      <span className="flex items-center gap-1.5 text-muted-foreground">
        <Icon className={cn("size-3 shrink-0", colour)} aria-hidden />
        {label}
      </span>
      <span className="truncate font-medium tabular-nums">{value}</span>
    </>
  );
  const shell = "flex items-center justify-between gap-2 px-2 py-1 text-xs group-data-[collapsible=icon]:hidden";
  if (!hint) return <div className={shell}>{contents}</div>;

  // A real button or link as the trigger, so the hint (often the only copy of
  // the fix) is reachable by keyboard, not by hover alone.
  const triggerClass = cn(
    shell,
    "w-full rounded-sm text-left focus-visible:ring-[3px] focus-visible:ring-sidebar-ring/50 focus-visible:outline-none",
    href ? "hover:bg-sidebar-accent" : "cursor-help",
  );
  return (
    <Tooltip>
      <TooltipTrigger
        render={href ? <Link href={href} className={triggerClass} /> : <button type="button" className={triggerClass} />}
      >
        {contents}
      </TooltipTrigger>
      <TooltipContent side="right" className="max-w-64">{hint}</TooltipContent>
    </Tooltip>
  );
}

export function AppSidebar() {
  const pathname = usePathname();
  const { isMobile, setOpenMobile } = useSidebar();
  const { data: status, error } = useSWR<AppStatus>("/status", { refreshInterval: 30_000 });
  // On a phone the sidebar is a sheet over the page; picking a page must close
  // it, or the page you asked for stays hidden behind the menu.
  const closeOnMobile = () => {
    if (isMobile) setOpenMobile(false);
  };

  const ai = status?.ai;
  const aiTone: Tone = !ai ? "muted" : !ai.up ? "bad" : ai.answer_model_installed === false ? "warn" : ai.busy_with.length ? "warn" : "ok";
  const aiValue = !ai ? "..." : !ai.up ? "Unavailable" : ai.busy_with.length ? "Busy" : (ai.answer_model ?? "No model");
  const aiHint = !ai
    ? undefined
    : !ai.up
      ? "Local AI is not responding. Open System Health to see why."
      : ai.busy_with.length
        ? `Another app has ${ai.busy_with.join(", ")} loaded, so answers will be slower until it unloads.`
        : ai.answer_model_installed === false
          ? `The answer model ${ai.answer_model} is not installed. System Health has the command.`
          : `Answers come from ${ai.answer_model}, on this computer.`;

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="border-b">
        <Link
          href="/"
          className="flex items-center gap-2 rounded-md px-2 py-1.5 focus-visible:ring-[3px] focus-visible:ring-sidebar-ring/50 focus-visible:outline-none group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:gap-0 group-data-[collapsible=icon]:px-0"
        >
          <div className="flex size-7 shrink-0 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <Layers className="size-4" aria-hidden />
          </div>
          <div className="min-w-0 group-data-[collapsible=icon]:hidden">
            <p className="truncate text-sm leading-tight font-semibold">{APP_NAME}</p>
            <p className="truncate text-[11px] text-muted-foreground">Local · cited answers</p>
          </div>
        </Link>
      </SidebarHeader>

      {/* Sidebar, SidebarContent and SidebarGroup are divs; without `as` the
          app would have no navigation landmark at all. */}
      <SidebarContent as="nav" aria-label="Primary">
        {NAV.map((section) => (
          <SidebarGroup key={section.group}>
            <SidebarGroupLabel>{section.group}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {section.items.map((item) => {
                  const label = navLabel(item.href);
                  return (
                    <SidebarMenuItem key={item.href}>
                      {/* Base UI composition: `render`, not Radix's `asChild`. */}
                      <SidebarMenuButton
                        render={<Link href={item.href} onClick={closeOnMobile} />}
                        isActive={isActive(pathname, item.href)}
                        tooltip={label}
                      >
                        <item.icon aria-hidden />
                        <span>{label}</span>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  );
                })}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>

      <SidebarFooter className="border-t">
        <p className="px-2 pb-1 text-[10px] font-medium tracking-wide text-muted-foreground uppercase group-data-[collapsible=icon]:hidden">
          Status
        </p>
        {error ? (
          <StatusRow
            label="App service"
            value="Unavailable"
            tone="bad"
            hint="Nothing can load until the app service is running. Start it with ./rag web."
          />
        ) : (
          <>
            <StatusRow
              label={ai?.backend === "remote" ? "Remote AI" : "Local AI"}
              value={aiValue}
              tone={aiTone}
              hint={aiHint}
              href={ai && (!ai.up || ai.answer_model_installed === false) ? "/health" : undefined}
            />
            <StatusRow
              label={navLabel("/documents")}
              value={status ? String(status.documents) : "..."}
              tone={!status ? "muted" : status.documents > 0 ? "ok" : "warn"}
            />
            <StatusRow
              label="Index"
              value={!status ? "..." : status.indexed ? "Built" : "Not built"}
              tone={!status ? "muted" : status.indexed ? "ok" : "muted"}
              hint={status && !status.indexed ? "No search index yet. Build one with ./rag ingest." : undefined}
              href={status && !status.indexed ? "/indexes" : undefined}
            />
          </>
        )}
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
