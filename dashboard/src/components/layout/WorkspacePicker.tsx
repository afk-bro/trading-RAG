import { useState } from "react";
import { useWorkspaces } from "@/hooks/use-workspaces";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

interface Props {
  onSelect: (id: string) => void;
}

export function WorkspacePicker({ onSelect }: Props) {
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const { data, isLoading } = useWorkspaces();

  const workspaces = data?.workspaces ?? [];

  function submit() {
    const trimmed = input.trim();
    if (!UUID_RE.test(trimmed)) {
      setError("Enter a valid UUID");
      return;
    }
    onSelect(trimmed);
  }

  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="w-full max-w-md p-6 bg-bg-secondary rounded-lg border border-border">
        <h2 className="text-xl text-text-emphasis mb-4">Select Workspace</h2>

        {/* Workspace list from API */}
        {isLoading ? (
          <div className="text-sm text-text-muted mb-4">Loading workspaces...</div>
        ) : workspaces.length > 0 ? (
          <div className="flex flex-col gap-2 mb-4">
            {workspaces.map((ws) => (
              <button
                key={ws.id}
                onClick={() => onSelect(ws.id)}
                className="text-left px-3 py-3 rounded-md border border-border
                           hover:border-accent hover:bg-bg-tertiary transition-colors"
              >
                <div className="text-sm font-medium text-text-emphasis">{ws.name}</div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-xs text-text-muted">{ws.slug}</span>
                  <span className="text-xs text-text-muted">
                    {new Date(ws.created_at).toLocaleDateString()}
                  </span>
                </div>
              </button>
            ))}
          </div>
        ) : (
          <div className="text-sm text-text-muted mb-4">No workspaces found</div>
        )}

        {/* UUID fallback */}
        <p className="text-text-muted text-xs mb-2">Or enter UUID directly</p>
        <div className="flex gap-2 mb-3">
          <input
            className="flex-1 bg-background border border-border rounded-md px-3 py-2
                       text-foreground placeholder:text-text-muted focus:outline-none
                       focus:border-accent text-sm"
            placeholder="Workspace UUID"
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              setError(null);
            }}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
          <button
            onClick={submit}
            className="px-4 py-2 bg-accent text-background rounded-md text-sm
                       font-medium hover:opacity-90 transition-opacity"
          >
            Go
          </button>
        </div>

        {error && <p className="text-danger text-xs mb-3">{error}</p>}
      </div>
    </div>
  );
}
