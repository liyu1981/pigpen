package pigmodeltweaks

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const openRouterListFixture = `{
  "data": [
    {
      "id": "anthropic/claude-sonnet-4.5", "name": "Anthropic: Claude Sonnet 4.5",
      "context_length": 1000000,
      "architecture": {"input_modalities": ["text", "image"]},
      "pricing": {"prompt": "0.000003", "completion": "0.000015", "input_cache_read": "0.0000003"},
      "top_provider": {"max_completion_tokens": 64000},
      "supported_parameters": ["tools", "reasoning", "reasoning_effort"]
    },
    {
      "id": "new/model", "name": "New Model",
      "context_length": 128000,
      "architecture": {"input_modalities": ["text"]},
      "pricing": {"prompt": "0", "completion": "0"},
      "top_provider": {"max_completion_tokens": 262144},
      "supported_parameters": ["tools"]
    }
  ]
}`

func TestOpenRouterCatalogConvertsList(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(openRouterListFixture))
	}))
	defer server.Close()

	catalog := &openRouterCatalog{client: server.Client(), url: server.URL}
	models, err := catalog.load(context.Background())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("models = %d, want 2", len(models))
	}
	first := models[0]
	if first["id"] != "anthropic/claude-sonnet-4.5" || first["name"] != "Anthropic: Claude Sonnet 4.5" {
		t.Errorf("identity = %v", first)
	}
	if first["api"] != "openai-completions" || first["baseUrl"] != "https://openrouter.ai/api/v1" {
		t.Errorf("endpoint = %v / %v", first["api"], first["baseUrl"])
	}
	if first["contextWindow"] != 1000000 || first["maxTokens"] != 64000 {
		t.Errorf("limits = %v / %v", first["contextWindow"], first["maxTokens"])
	}
	if first["reasoning"] != true {
		t.Errorf("reasoning = %v, want true", first["reasoning"])
	}
	input, _ := first["input"].([]any)
	if len(input) != 2 || input[1] != "image" {
		t.Errorf("input = %v, want text+image", input)
	}
	cost, _ := first["cost"].(map[string]any)
	if cost["input"] != 3.0 || cost["output"] != 15.0 || cost["cacheRead"] != 0.3 || cost["cacheWrite"] != 0.0 {
		t.Errorf("cost = %v", cost)
	}

	second := models[1]
	if second["reasoning"] != false {
		t.Errorf("reasoning = %v, want false", second["reasoning"])
	}
	if second["maxTokens"] != 128000 {
		t.Errorf("maxTokens = %v, want context cap", second["maxTokens"])
	}
}

func TestOpenRouterCatalogRejectsEmptyList(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"data":[]}`))
	}))
	defer server.Close()
	if _, err := (&openRouterCatalog{client: server.Client(), url: server.URL}).load(context.Background()); err == nil {
		t.Fatal("empty catalog should be an error")
	}
}

func TestMergeOpenRouterModelsPreservesHostConfig(t *testing.T) {
	existing := []map[string]any{
		{"id": "anthropic/claude-sonnet-4.5", "api": "anthropic-messages", "baseUrl": "https://openrouter.ai/api", "compat": map[string]any{"forceAdaptiveThinking": true}},
		{"id": "old/alias", "api": "openai-completions"},
	}
	fetched := []map[string]any{
		{"id": "anthropic/claude-sonnet-4.5", "api": "openai-completions", "baseUrl": "https://openrouter.ai/api/v1"},
		{"id": "new/model", "api": "openai-completions", "baseUrl": "https://openrouter.ai/api/v1"},
	}
	merged := mergeOpenRouterModels(existing, fetched)
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

func TestRefreshOpenRouterRegistersMergedModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(openRouterListFixture))
	}))
	defer server.Close()

	original := liveOpenRouterCatalog
	liveOpenRouterCatalog = &openRouterCatalog{client: server.Client(), url: server.URL}
	t.Cleanup(func() { liveOpenRouterCatalog = original })

	state := []any{map[string]any{
		"id": "anthropic/claude-sonnet-4.5", "provider": "openrouter",
		"api": "anthropic-messages", "baseUrl": "https://openrouter.ai/api",
	}}
	host := StartHost(t, Extension(), HostOptions{
		OnCall: func(method string, args map[string]any) (map[string]any, string) {
			switch method {
			case "getModelRegistryState":
				return map[string]any{"models": state, "providers": map[string]any{}, "registered": []any{}}, ""
			case "registerProvider":
				config, _ := args["config"].(map[string]any)
				models, _ := config["models"].([]any)
				state = nil
				for _, model := range models {
					entry, _ := model.(map[string]any)
					entry["provider"] = "openrouter"
					state = append(state, entry)
				}
				return map[string]any{}, ""
			default:
				return map[string]any{}, ""
			}
		},
	})
	if failure := host.Command(refreshOpenRouterName, ""); failure != "" {
		t.Fatalf("command failure = %q", failure)
	}
	registrations := host.CallsTo("registerProvider")
	if len(registrations) != 1 || registrations[0].Args["name"] != "openrouter" {
		t.Fatalf("registrations = %v", registrations)
	}
	config, _ := registrations[0].Args["config"].(map[string]any)
	models, _ := config["models"].([]any)
	if len(models) != 2 {
		t.Fatalf("registered models = %d, want 2", len(models))
	}
	first, _ := models[0].(map[string]any)
	if first["api"] != "anthropic-messages" {
		t.Errorf("known model lost its host config: %v", first)
	}
	messages := host.CallsTo("sendMessage")
	if len(messages) != 1 {
		t.Fatalf("sendMessage calls = %d, want 1", len(messages))
	}
	message, _ := messages[0].Args["message"].(map[string]any)
	content, _ := message["content"].(string)
	if !strings.Contains(content, "new/model") {
		t.Errorf("result message = %q, want new model name", content)
	}
}
