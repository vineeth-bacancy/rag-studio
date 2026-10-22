"use client";

import { useState } from "react";
import useSWR, { mutate } from "swr";
import { History, MessageSquarePlus, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { useAsk } from "@/components/ask/ask-store";
import { api, type ChatBrief } from "@/lib/api";
import { num, relativeTime } from "@/lib/format";
import { cn } from "@/lib/utils";

/**
 * The saved conversations, newest first. A conversation lives on this
 * computer (data/chat/); opening one shows its questions and answers and
 * lets you carry on. "New conversation" starts another and keeps this one.
 */
export function ConversationList({ onPicked, className }: { onPicked?: () => void; className?: string }) {
  const { data } = useSWR<{ chats: ChatBrief[] }>("/chats", { refreshInterval: 0 });
  const chatId = useAsk((s) => s.chatId);
  const busy = useAsk((s) => s.busy);
  const openChat = useAsk((s) => s.openChat);
  const clear = useAsk((s) => s.clear);
  const [confirming, setConfirming] = useState<string | null>(null);

  async function open(id: string) {
    try {
      openChat(await api.chat(id));
      onPicked?.();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "This conversation could not be opened. It may have been deleted.");
      void mutate("/chats");
    }
  }

  async function remove(id: string) {
    try {
      await api.deleteChat(id);
      if (id === chatId) clear();
      void mutate("/chats");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "The conversation could not be deleted. Try again.");
    } finally {
      setConfirming(null);
    }
  }

  const chats = data?.chats ?? [];
  return (
    <nav aria-label="Conversations" className={cn("space-y-2", className)}>
      <Button
        variant="outline"
        size="sm"
        className="w-full justify-start"
        disabled={busy}
        onClick={() => {
          clear();
          onPicked?.();
        }}
      >
        <MessageSquarePlus aria-hidden />
        New conversation
      </Button>
      {chats.length === 0 ? (
        <p className="px-1 text-xs text-pretty text-muted-foreground">
          Nothing saved yet. Each question you ask is kept in a conversation on this computer, so you can ask a
          follow-up later.
        </p>
      ) : (
        <ul className="space-y-0.5">
          {chats.map((c) => (
            <li key={c.id} className="group flex items-center gap-1">
              <button
                type="button"
                disabled={busy}
                onClick={() => void open(c.id)}
                aria-current={c.id === chatId ? "true" : undefined}
                className={cn(
                  "min-w-0 flex-1 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-muted focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none disabled:opacity-50",
                  c.id === chatId && "bg-muted font-medium",
                )}
              >
                <span className="block truncate">{c.title ?? "Untitled"}</span>
                <span className="block text-[11px] font-normal text-muted-foreground tabular-nums">
                  {num(c.turns)} {c.turns === 1 ? "question" : "questions"} · {relativeTime(c.updated_at)}
                </span>
              </button>
              {confirming === c.id ? (
                <Button size="xs" variant="destructive" onClick={() => void remove(c.id)}>
                  Delete
                </Button>
              ) : (
                <Button
                  size="icon-xs"
                  variant="ghost"
                  aria-label={`Delete conversation: ${c.title ?? "Untitled"}`}
                  className="opacity-0 group-focus-within:opacity-100 group-hover:opacity-100 focus-visible:opacity-100"
                  onClick={() => setConfirming(c.id)}
                >
                  <Trash2 aria-hidden />
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
    </nav>
  );
}

/** Below a wide container the list lives in a Sheet. */
export function ConversationsSheet() {
  const [open, setOpen] = useState(false);
  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger render={<Button variant="outline" size="sm" />}>
        <History aria-hidden />
        Conversations
      </SheetTrigger>
      <SheetContent side="left" className="w-80 max-w-[90vw]">
        <SheetHeader>
          <SheetTitle>Conversations</SheetTitle>
          <SheetDescription>Saved on this computer.</SheetDescription>
        </SheetHeader>
        <div className="px-4 pb-4">
          <ConversationList onPicked={() => setOpen(false)} />
        </div>
      </SheetContent>
    </Sheet>
  );
}
