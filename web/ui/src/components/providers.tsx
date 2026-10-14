"use client";

import { ThemeProvider } from "next-themes";
import { SWRConfig } from "swr";

import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { fetcher } from "@/lib/api";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem disableTransitionOnChange>
      <SWRConfig
        value={{
          fetcher,
          // Local data that changes when you act, not on its own. One refetch on
          // returning to the tab keeps a long-open page honest without polling.
          revalidateOnFocus: true,
          focusThrottleInterval: 30_000,
          shouldRetryOnError: false,
          dedupingInterval: 5000,
        }}
      >
        <TooltipProvider delay={200}>
          {children}
          <Toaster position="bottom-right" richColors closeButton />
        </TooltipProvider>
      </SWRConfig>
    </ThemeProvider>
  );
}
