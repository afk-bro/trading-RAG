import { useState, useEffect } from "react";
import { createPortal } from "react-dom";
import { useCreateWorkspace } from "@/hooks/use-workspaces";

const SUGGESTIONS = [
  "Apex 50k \u00b7 NQ \u00b7 NY AM",
  "Unicorn v1 \u00b7 30d",
  `Research \u00b7 ${new Date().toISOString().slice(0, 10)}`,
];

interface Props {
  onCreated: (id: string) => void;
  onClose: () => void;
}

export function CreateWorkspaceModal({ onCreated, onClose }: Props) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const { mutate, isPending, error } = useCreateWorkspace();

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  function handleCreate() {
    if (!name.trim()) return;
    mutate(
      { name: name.trim(), description: description.trim() || undefined },
      { onSuccess: (ws) => onCreated(ws.id) },
    );
  }

  return createPortal(
    <div
      className="fixed inset-0 z-[400] flex items-center justify-center bg-black/50"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="w-full max-w-md bg-bg-secondary rounded-lg border border-border p-6">
        <h3 className="text-lg text-text-emphasis mb-4">New Workspace</h3>

        {/* Quick suggestions */}
        <div className="flex flex-wrap gap-2 mb-4">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => setName(s)}
              className="px-2 py-1 text-xs rounded-md bg-bg-tertiary text-text-muted
                         hover:text-foreground transition-colors"
            >
              {s}
            </button>
          ))}
        </div>

        <label className="block text-sm text-text-muted mb-1">Name *</label>
        <input
          className="w-full bg-background border border-border rounded-md px-3 py-2 mb-3
                     text-foreground placeholder:text-text-muted focus:outline-none
                     focus:border-accent text-sm"
          placeholder="Workspace name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleCreate()}
          autoFocus
        />

        <label className="block text-sm text-text-muted mb-1">
          Description
        </label>
        <input
          className="w-full bg-background border border-border rounded-md px-3 py-2 mb-4
                     text-foreground placeholder:text-text-muted focus:outline-none
                     focus:border-accent text-sm"
          placeholder="Optional description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />

        {error && (
          <p className="text-danger text-xs mb-3">
            {error instanceof Error ? error.message : "Failed to create"}
          </p>
        )}

        <div className="flex justify-end gap-2">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-text-muted hover:text-foreground
                       rounded-md transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={!name.trim() || isPending}
            className="px-4 py-2 bg-accent text-background rounded-md text-sm
                       font-medium hover:opacity-90 transition-opacity
                       disabled:opacity-50"
          >
            {isPending ? "Creating..." : "Create"}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
