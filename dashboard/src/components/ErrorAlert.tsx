import { AlertTriangle } from "lucide-react";

interface Props {
  message: string;
  details?: string;
  onRetry?: () => void;
}

export function ErrorAlert({ message, details, onRetry }: Props) {
  return (
    <div className="bg-danger/10 border border-danger/30 rounded-lg p-4">
      <div className="flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-danger flex-shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0 space-y-2">
          <p className="text-sm font-medium text-danger">{message}</p>
          {details && (
            <details className="text-xs text-text-muted">
              <summary className="cursor-pointer hover:text-foreground">
                Details
              </summary>
              <pre className="mt-1 p-2 bg-bg-tertiary rounded text-[11px] overflow-x-auto whitespace-pre-wrap break-all">
                {details}
              </pre>
            </details>
          )}
          {onRetry && (
            <button
              onClick={onRetry}
              className="px-3 py-1.5 text-xs font-medium rounded-md bg-danger/15 text-danger
                         hover:bg-danger/25 transition-colors"
            >
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
