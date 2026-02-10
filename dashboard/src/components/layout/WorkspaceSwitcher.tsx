import { useState, useRef, useEffect } from "react";
import { ChevronDown, Plus } from "lucide-react";
import { useWorkspaces } from "@/hooks/use-workspaces";
import { CreateWorkspaceModal } from "./CreateWorkspaceModal";

interface Props {
  currentId: string;
  onSelect: (id: string) => void;
}

export function WorkspaceSwitcher({ currentId, onSelect }: Props) {
  const [open, setOpen] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { data, isLoading } = useWorkspaces();

  const workspaces = data?.workspaces ?? [];
  const current = workspaces.find((w) => w.id === currentId);
  const label = current?.name ?? (currentId ? `${currentId.slice(0, 8)}...` : "Select workspace");

  // Close on click-outside
  useEffect(() => {
    if (!open) return;
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-haspopup="listbox"
        className="flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium
                   bg-bg-tertiary text-text-muted hover:text-foreground transition-colors"
      >
        <span className="max-w-[160px] truncate">{label}</span>
        <ChevronDown className="w-3 h-3" />
      </button>

      {open && (
        <div className="absolute left-0 top-full mt-1 z-[300] w-64 bg-bg-secondary
                        border border-border rounded-lg shadow-lg overflow-hidden">
          {isLoading ? (
            <div className="px-3 py-2 text-xs text-text-muted">Loading...</div>
          ) : workspaces.length === 0 ? (
            <div className="px-3 py-2 text-xs text-text-muted">No workspaces</div>
          ) : (
            <div role="listbox" className="max-h-60 overflow-y-auto">
              {workspaces.map((ws) => (
                <button
                  key={ws.id}
                  onClick={() => {
                    onSelect(ws.id);
                    setOpen(false);
                  }}
                  className={`w-full text-left px-3 py-2 text-sm transition-colors
                    ${ws.id === currentId
                      ? "bg-bg-tertiary text-text-emphasis"
                      : "text-foreground hover:bg-bg-tertiary"
                    }`}
                >
                  <div className="font-medium truncate">{ws.name}</div>
                  <div className="text-xs text-text-muted truncate">{ws.slug}</div>
                </button>
              ))}
            </div>
          )}

          <div className="border-t border-border">
            <button
              onClick={() => {
                setOpen(false);
                setShowCreate(true);
              }}
              className="w-full flex items-center gap-2 px-3 py-2 text-sm text-text-muted
                         hover:text-foreground hover:bg-bg-tertiary transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              New workspace
            </button>
          </div>
        </div>
      )}

      {showCreate && (
        <CreateWorkspaceModal
          onCreated={(id) => {
            setShowCreate(false);
            onSelect(id);
          }}
          onClose={() => setShowCreate(false)}
        />
      )}
    </div>
  );
}
