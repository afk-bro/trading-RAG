import { useState, useEffect } from "react";

const STORAGE_KEY = "dashboard_recent_workspaces";

function getRecent(): string[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]") as string[];
  } catch {
    return [];
  }
}

function addRecent(id: string) {
  const list = getRecent().filter((w) => w !== id);
  list.unshift(id);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list.slice(0, 5)));
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

interface Props {
  onSelect: (id: string) => void;
}

export function WorkspacePicker({ onSelect }: Props) {
  const [input, setInput] = useState("");
  const [recent, setRecent] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setRecent(getRecent());
  }, []);

  function submit() {
    const trimmed = input.trim();
    if (!UUID_RE.test(trimmed)) {
      setError("Enter a valid UUID");
      return;
    }
    addRecent(trimmed);
    onSelect(trimmed);
  }

  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="w-full max-w-md p-6 bg-bg-secondary rounded-lg border border-border">
        <h2 className="text-xl text-text-emphasis mb-4">Select Workspace</h2>

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

        {recent.length > 0 && (
          <div>
            <p className="text-text-muted text-xs mb-2">Recent</p>
            <div className="flex flex-col gap-1">
              {recent.map((id) => (
                <button
                  key={id}
                  onClick={() => {
                    addRecent(id);
                    onSelect(id);
                  }}
                  className="text-left px-3 py-2 rounded-md text-sm text-foreground
                             hover:bg-bg-tertiary font-mono transition-colors truncate"
                >
                  {id}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
