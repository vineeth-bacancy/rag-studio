"use client";

import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "./app-sidebar";
import { CommandPalette } from "./command-palette";
import { TopBar } from "./top-bar";

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <AppSidebar />
      {/*
        SidebarInset already renders a <main>, so the skip link targets it
        directly; tabIndex={-1} lets focus actually land there.
      */}
      <SidebarInset id="main" tabIndex={-1} className="min-w-0 focus:outline-none">
        <TopBar />
        <div className="min-w-0 flex-1 px-4 py-6 sm:px-6 lg:px-8">{children}</div>
      </SidebarInset>
      <CommandPalette />
    </SidebarProvider>
  );
}
