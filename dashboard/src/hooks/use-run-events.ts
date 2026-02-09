import { useQuery } from "@tanstack/react-query";
import { getRunEvents } from "@/api/endpoints";

export function useRunEvents(runId: string | null) {
  return useQuery({
    queryKey: ["run-events", runId],
    queryFn: () => getRunEvents(runId!),
    enabled: !!runId,
    staleTime: Infinity, // Events don't change for a completed run
  });
}
