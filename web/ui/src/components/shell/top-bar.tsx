"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import useSWR from "swr";
import { Moon, Search, Sun } from "lucide-react";
import { useTheme } from "next-themes";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { AppStatus } from "@/lib/api";
import { ROUTE_TITLES } from "@/lib/routes";
import { useCommandPalette } from "./command-palette";

function pageTitle(pathname: string) {
  const root = `/${pathname.split("/").filter(Boolean)[0] ?? ""}`;
  return ROUTE_TITLES[root] ?? ROUTE_TITLES["/"];
}

export function TopBar() {
  const pathname = usePathname();
  const { resolvedTheme, setTheme } = useTheme();
  const setPaletteOpen = useCommandPalette((s) => s.setOpen);
  // Same key as the sidebar, so SWR serves both from one request.
  const { data: status } = useSWR<AppStatus>("/status", { refreshInterval: 30_000 });
  const title = pageTitle(pathname);

  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center gap-2 border-b bg-background/95 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <SidebarTrigger className="-ml-1" />
      {/* Same variant as the component's default (data-vertical:self-stretch),
          so tailwind-merge replaces it. A plain \`self-center\` loses to the
          attribute selector, and with a fixed height the line pins to the top. */}
      <Separator orientation="vertical" className="mr-1 data-vertical:h-4 data-vertical:self-center" />

      {/* Context, not a heading: each page renders its own <h1> (PageHeader),
          so a screen reader meets the title once. */}
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-muted-foreground">{title}</p>
      </div>

      <div className="flex items-center gap-1">
        {status && !status.ai.up && (
          // A link, not a label: the reason and the fix are on System Health.
          <Link href="/health" className="mr-1 hidden rounded-md focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none md:flex">
            <Badge variant="outline" className="gap-1 border-amber-500/40 text-[11px] text-amber-700 hover:bg-amber-500/10 dark:text-amber-400">
              AI unavailable
            </Badge>
          </Link>
        )}

        <Button
          variant="outline"
          size="sm"
          className="hidden h-8 gap-2 px-2.5 text-muted-foreground sm:flex"
          onClick={() => setPaletteOpen(true)}
        >
          <Search className="size-3.5" aria-hidden />
          <span className="text-xs">Search...</span>
          <kbd className="ml-1 rounded border bg-muted px-1 font-mono text-[10px]">⌘K</kbd>
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="size-8 sm:hidden"
          onClick={() => setPaletteOpen(true)}
          aria-label="Search"
        >
          <Search className="size-4" aria-hidden />
        </Button>

        <Tooltip>
          <TooltipTrigger
            render={
              <Button
                variant="ghost"
                size="icon"
                className="relative size-8"
                onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
                aria-label="Switch between light and dark"
              />
            }
          >
            <Sun className="size-4 scale-100 rotate-0 transition-transform dark:scale-0 dark:-rotate-90" aria-hidden />
            <Moon className="absolute size-4 scale-0 rotate-90 transition-transform dark:scale-100 dark:rotate-0" aria-hidden />
          </TooltipTrigger>
          <TooltipContent>Light / dark</TooltipContent>
        </Tooltip>
      </div>
    </header>
  );
}
