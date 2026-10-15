import useSWR from "swr";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { IndexProfileRow, PromptVersion } from "@/lib/api";
import { useAsk } from "./ask-store";

/**
 * Which index answers come from, and which prompt version the model gets.
 * Only built indexes can be chosen; the others say how to build them. The
 * defaults stay the baseline until the scorecard says otherwise.
 */
export function SetupPicker({ disabled }: { disabled?: boolean }) {
  const { data: indexes } = useSWR<{ profiles: IndexProfileRow[] }>("/indexes");
  const { data: prompts } = useSWR<{ default: string; versions: PromptVersion[] }>("/prompts");
  const profile = useAsk((s) => s.profile);
  const prompt = useAsk((s) => s.prompt);
  const setProfile = useAsk((s) => s.setProfile);
  const setPrompt = useAsk((s) => s.setPrompt);

  const profileItems = Object.fromEntries((indexes?.profiles ?? []).map((p) => [p.id, p.label]));
  const promptItems = Object.fromEntries((prompts?.versions ?? []).map((v) => [v.version, `Prompt ${v.version}`]));

  return (
    <div className="flex min-w-0 flex-wrap items-center gap-1.5">
      <label htmlFor="ask-index" className="sr-only">
        Index
      </label>
      <Select value={profile} items={profileItems} onValueChange={(v) => v && setProfile(v as string)} disabled={disabled}>
        <SelectTrigger id="ask-index" size="sm" className="max-w-52 text-xs" title="Which index to search">
          <SelectValue placeholder="Index" />
        </SelectTrigger>
        <SelectContent>
          {(indexes?.profiles ?? []).map((p) => (
            <SelectItem key={p.id} value={p.id} disabled={!p.active}>
              <span className="flex flex-col">
                <span>{p.label}</span>
                {!p.active && (
                  <span className="text-[11px] text-muted-foreground">
                    Not built yet. Run: ./rag ingest --profile {p.id}
                  </span>
                )}
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <label htmlFor="ask-prompt" className="sr-only">
        Prompt version
      </label>
      <Select value={prompt} items={promptItems} onValueChange={(v) => v && setPrompt(v as string)} disabled={disabled}>
        <SelectTrigger id="ask-prompt" size="sm" className="text-xs" title="Which system prompt version the model gets">
          <SelectValue placeholder="Prompt" />
        </SelectTrigger>
        <SelectContent>
          {(prompts?.versions ?? []).map((v) => (
            <SelectItem key={v.version} value={v.version}>
              Prompt {v.version}
              {v.version === prompts?.default ? " (default)" : ""}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
