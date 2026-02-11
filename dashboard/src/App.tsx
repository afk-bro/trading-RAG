import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkspaceProvider } from "@/context/workspace";
import { AuthProvider, useAuth } from "@/context/auth";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { DashboardPage } from "@/pages/DashboardPage";
import { BacktestsPage } from "@/pages/BacktestsPage";
import { BacktestRunPage } from "@/pages/BacktestRunPage";
import { ComparePage } from "@/pages/ComparePage";
import { LoginPage } from "@/pages/LoginPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function RequireAuth({ children }: { children: React.ReactNode }) {
  const { session, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-text-muted text-sm">Loading...</div>
      </div>
    );
  }

  // Allow if Supabase session exists OR legacy admin token is set
  const hasAdminToken = !!localStorage.getItem("admin_token");
  if (!session && !hasAdminToken) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

export function App() {
  return (
    <AuthProvider>
      <WorkspaceProvider>
        <QueryClientProvider client={queryClient}>
          <BrowserRouter basename="/dashboard">
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route
                element={
                  <RequireAuth>
                    <ErrorBoundary>
                      <DashboardShell />
                    </ErrorBoundary>
                  </RequireAuth>
                }
              >
                <Route index element={<DashboardPage />} />
                <Route path="backtests" element={<BacktestsPage />} />
                <Route path="backtests/compare" element={<ComparePage />} />
                <Route path="backtests/:runId" element={<BacktestRunPage />} />
              </Route>
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </BrowserRouter>
        </QueryClientProvider>
      </WorkspaceProvider>
    </AuthProvider>
  );
}
