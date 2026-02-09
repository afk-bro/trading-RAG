import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { WorkspaceProvider } from "@/context/workspace";
import { DashboardShell } from "@/components/layout/DashboardShell";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { DashboardPage } from "@/pages/DashboardPage";
import { BacktestsPage } from "@/pages/BacktestsPage";
import { BacktestRunPage } from "@/pages/BacktestRunPage";
import { ComparePage } from "@/pages/ComparePage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export function App() {
  return (
    <WorkspaceProvider>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter basename="/dashboard">
          <Routes>
            <Route element={<ErrorBoundary><DashboardShell /></ErrorBoundary>}>
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
  );
}
