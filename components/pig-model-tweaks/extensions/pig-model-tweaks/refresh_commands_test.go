package pigmodeltweaks

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The refresh commands must leave a persistent, visible result in the
// conversation: a toast alone is easy to miss, and a silent command reads as
// "did nothing" even when the host refreshed the catalog successfully.

func TestRefreshCommandReportsPersistentMessageAndToast(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(openRouterListFixture))
	}))
	defer server.Close()
	original := liveOpenRouterCatalog
	liveOpenRouterCatalog = &openRouterCatalog{client: server.Client(), url: server.URL}
	t.Cleanup(func() { liveOpenRouterCatalog = original })

	host := StartHost(t, Extension(), HostOptions{
		OnCall: func(method string, args map[string]any) (map[string]any, string) {
			if method == "getModelRegistryState" {
				return map[string]any{"models": []any{}, "providers": map[string]any{}, "registered": []any{}}, ""
			}
			return map[string]any{}, ""
		},
	})
	if failure := host.Command(refreshOpenRouterName, ""); failure != "" {
		t.Fatalf("command failure = %q", failure)
	}

	messages := host.CallsTo("sendMessage")
	if len(messages) != 1 {
		t.Fatalf("sendMessage calls = %d, want 1", len(messages))
	}
	message, _ := messages[0].Args["message"].(map[string]any)
	if message["customType"] != refreshMessageType || message["display"] != true {
		t.Errorf("message = %v, want a displayed %q", message, refreshMessageType)
	}
	content, _ := message["content"].(string)
	if !strings.Contains(content, refreshOpenRouterName) || !strings.Contains(content, "openrouter") {
		t.Errorf("content = %q, want command and provider", content)
	}
	options, _ := messages[0].Args["options"].(map[string]any)
	if options["triggerTurn"] != false {
		t.Errorf("options = %v, want triggerTurn false", options)
	}

	toasts := host.CallsTo("ui.notify")
	if len(toasts) != 1 {
		t.Fatalf("ui.notify calls = %d, want 1", len(toasts))
	}
	if toast, _ := toasts[0].Args["message"].(string); !strings.Contains(toast, refreshOpenRouterName) {
		t.Errorf("toast = %q, want command name", toast)
	}
}

func TestRefreshCommandFallsBackToToastWhenMessageFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(openRouterListFixture))
	}))
	defer server.Close()
	original := liveOpenRouterCatalog
	liveOpenRouterCatalog = &openRouterCatalog{client: server.Client(), url: server.URL}
	t.Cleanup(func() { liveOpenRouterCatalog = original })

	host := StartHost(t, Extension(), HostOptions{
		OnCall: func(method string, args map[string]any) (map[string]any, string) {
			switch method {
			case "getModelRegistryState":
				return map[string]any{"models": []any{}, "providers": map[string]any{}, "registered": []any{}}, ""
			case "sendMessage":
				return nil, "messages unavailable"
			default:
				return map[string]any{}, ""
			}
		},
	})
	if failure := host.Command(refreshOpenRouterName, ""); failure != "" {
		t.Fatalf("command failure = %q", failure)
	}
	toasts := host.CallsTo("ui.notify")
	if len(toasts) != 1 {
		t.Fatalf("ui.notify calls = %d, want 1 fallback", len(toasts))
	}
	if toast, _ := toasts[0].Args["message"].(string); !strings.Contains(toast, refreshOpenRouterName) || !strings.Contains(toast, "openrouter") {
		t.Errorf("fallback toast = %q, want detailed result", toast)
	}
}

func TestMergeCatalogModelsPreservesHostConfig(t *testing.T) {
	existing := []map[string]any{
		{"id": "anthropic/claude-sonnet-4.5", "api": "anthropic-messages", "baseUrl": "https://openrouter.ai/api", "compat": map[string]any{"forceAdaptiveThinking": true}},
		{"id": "old/alias", "api": "openai-completions"},
	}
	fetched := []map[string]any{
		{"id": "anthropic/claude-sonnet-4.5", "api": "openai-completions", "baseUrl": "https://openrouter.ai/api/v1"},
		{"id": "new/model", "api": "openai-completions", "baseUrl": "https://openrouter.ai/api/v1"},
	}
	merged := mergeCatalogModels(existing, fetched)
	if len(merged) != 3 {
		t.Fatalf("merged = %d, want 3", len(merged))
	}
	if merged[0]["api"] != "anthropic-messages" || merged[0]["compat"] == nil {
		t.Errorf("existing config not preserved: %v", merged[0])
	}
	if merged[1]["id"] != "new/model" {
		t.Errorf("new model = %v", merged[1])
	}
	if merged[2]["id"] != "old/alias" {
		t.Errorf("old-only model = %v", merged[2])
	}
}

func TestRefreshDetailReportsNewAndRemovedModels(t *testing.T) {
	tests := []struct {
		name         string
		before       []string
		after        []string
		haveBefore   bool
		haveAfter    bool
		wantContains []string
	}{
		{
			name: "no change", haveBefore: true, haveAfter: true,
			before: []string{"a", "b"}, after: []string{"a", "b"},
			wantContains: []string{"2 models available", "No new models"},
		},
		{
			name: "models added", haveBefore: true, haveAfter: true,
			before: []string{"a"}, after: []string{"a", "b", "c"},
			wantContains: []string{"3 models available", "2 new: b, c"},
		},
		{
			name: "models removed", haveBefore: true, haveAfter: true,
			before: []string{"a", "b"}, after: []string{"a"},
			wantContains: []string{"1 model available", "1 removed: b"},
		},
		{
			name: "no before snapshot", haveAfter: true,
			after: []string{"a"}, wantContains: []string{"1 model available"},
		},
		{
			name: "no after snapshot", haveBefore: true,
			before: []string{"a"}, wantContains: []string{"could not be read back"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := refreshDetail("opencode-go", tt.before, tt.haveBefore, tt.after, tt.haveAfter)
			for _, want := range tt.wantContains {
				if !strings.Contains(got, want) {
					t.Errorf("detail = %q, want it to contain %q", got, want)
				}
			}
		})
	}
}

func TestRefreshDetailCapsLongModelLists(t *testing.T) {
	before := []string{"old"}
	after := []string{"old", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"}
	got := refreshDetail("openrouter", before, true, after, true)
	if !strings.Contains(got, "9 new: m1, m2, m3, m4, m5, m6, m7, m8, … (+1 more)") {
		t.Errorf("detail = %q, want capped list", got)
	}
}
