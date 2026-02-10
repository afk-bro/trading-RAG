import { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  retryCount: number;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, retryCount: 0 };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("ErrorBoundary caught:", error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center min-h-[300px] p-6">
          <div className="bg-bg-secondary border border-danger/30 rounded-lg p-8 max-w-md w-full text-center space-y-4">
            <AlertTriangle className="w-10 h-10 text-danger mx-auto" />
            <h2 className="text-lg font-semibold text-text-emphasis">
              Something went wrong
            </h2>
            {this.state.error?.message && (
              <p className="text-xs font-mono text-text-muted break-all">
                {this.state.error.message}
              </p>
            )}
            <div className="flex items-center justify-center gap-3">
              <button
                onClick={() =>
                  this.setState((s) => ({
                    hasError: false,
                    error: null,
                    retryCount: s.retryCount + 1,
                  }))
                }
                className="px-4 py-2 text-sm font-medium rounded-md bg-accent text-white
                           hover:bg-accent/90 transition-colors"
              >
                Try again
              </button>
              <button
                onClick={() => window.location.reload()}
                className="px-4 py-2 text-sm font-medium rounded-md border border-border
                           text-text-muted hover:text-foreground hover:bg-bg-tertiary transition-colors"
              >
                Reload page
              </button>
            </div>
          </div>
        </div>
      );
    }

    return <div key={this.state.retryCount}>{this.props.children}</div>;
  }
}
