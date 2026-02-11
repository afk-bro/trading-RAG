import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Auth } from "@supabase/auth-ui-react";
import { ThemeSupa } from "@supabase/auth-ui-shared";
import { supabase } from "@/lib/supabase";
import { useAuth } from "@/context/auth";
import { Activity } from "lucide-react";

export function LoginPage() {
  const { session, loading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (session) {
      navigate("/", { replace: true });
    }
  }, [session, navigate]);

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-text-muted text-sm">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <div className="flex items-center justify-center gap-2 mb-8">
          <Activity className="w-6 h-6 text-accent" />
          <h1 className="text-xl font-semibold text-foreground">
            Trading Dashboard
          </h1>
        </div>
        <div className="rounded-lg border border-border bg-bg-secondary p-6">
          <Auth
            supabaseClient={supabase}
            appearance={{
              theme: ThemeSupa,
              variables: {
                default: {
                  colors: {
                    brand: "hsl(var(--accent))",
                    brandAccent: "hsl(var(--accent))",
                    inputBackground: "hsl(var(--bg-tertiary))",
                    inputText: "hsl(var(--foreground))",
                    inputBorder: "hsl(var(--border))",
                  },
                },
              },
            }}
            providers={[]}
            redirectTo={window.location.origin + "/dashboard"}
          />
        </div>
      </div>
    </div>
  );
}
