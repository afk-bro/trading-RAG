import { createContext, useContext, useCallback, useSyncExternalStore } from "react";

const STORAGE_KEY = "active_workspace_id";

// ---------------------------------------------------------------------------
// Tiny external store so localStorage writes are reactive across components
// ---------------------------------------------------------------------------

let listeners: Array<() => void> = [];

function subscribe(cb: () => void) {
  listeners.push(cb);
  return () => {
    listeners = listeners.filter((l) => l !== cb);
  };
}

function getSnapshot(): string {
  return localStorage.getItem(STORAGE_KEY) ?? "";
}

function setStored(id: string) {
  if (id) {
    localStorage.setItem(STORAGE_KEY, id);
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
  listeners.forEach((l) => l());
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

interface WorkspaceCtx {
  workspaceId: string;
  setWorkspaceId: (id: string) => void;
}

const Ctx = createContext<WorkspaceCtx>({
  workspaceId: "",
  setWorkspaceId: () => {},
});

export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const workspaceId = useSyncExternalStore(subscribe, getSnapshot);

  const setWorkspaceId = useCallback((id: string) => {
    setStored(id);
  }, []);

  return (
    <Ctx.Provider value={{ workspaceId, setWorkspaceId }}>
      {children}
    </Ctx.Provider>
  );
}

/**
 * Read the active workspace ID from context.
 * Reactive — re-renders when workspace changes anywhere in the app.
 */
export function useWorkspaceId(): string {
  return useContext(Ctx).workspaceId;
}

/**
 * Get the setter to change the active workspace.
 */
export function useSetWorkspaceId(): (id: string) => void {
  return useContext(Ctx).setWorkspaceId;
}

/**
 * Read workspace ID from localStorage (non-reactive, for api client).
 */
export function getActiveWorkspaceId(): string {
  return localStorage.getItem(STORAGE_KEY) ?? "";
}
