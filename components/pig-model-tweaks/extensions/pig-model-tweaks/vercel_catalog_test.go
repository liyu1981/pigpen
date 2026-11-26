package pigmodeltweaks

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const vercelListFixture = `{
  "object": "list",
  "data": [
    {
      "id": "anthropic/claude-sonnet-5.5-batch", "name": "Claude Sonnet 5.5 (batch)",
      "type": "language", "context_window": 200000, "max_tokens": 64000,
      "tags": ["reasoning", "tool-use", "vision"],
      "pricing": {"input": "0.000003", "output": "0.000015", "input_cache_read": "0.0000003", "input_cache_write": "0.00000375"}
    },
    {
      "id": "vendor/numeric-price", "name": "Numeric Price",
      "type": "language", "context_window": 128000, "max_tokens": 8192,
      "tags": ["tool-use"],
      "pricing": {"input": 0.0000001, "output": 0.0000002}
    },
    {
      "id": "vendor/untitled", "type": "language", "tags": ["tool-use"],
      "pricing": {"input": "0.00000012", "output": "0.00000024", "input_cache_read": "0.00000012"}
    },
    {
      "id": "vendor/image-model", "name": "An Image Model",
      "type": "image", "tags": ["image-generation"], "context_window": 4096, "max_tokens": 4096,
      "pricing": {"input": "0.01", "output": "0.04"}
    },
    {
      "id": "vendor/realtime", "name": "A Realtime Model",
      "type": "realtime", "tags": ["tool-use"], "context_window": 4096, "max_tokens": 4096,
      "pricing": {"input": "0.01", "output": "0.04"}
    },
    {
      "id": "vendor/no-tools", "name": "A Model Without Tools",
      "type": "language", "tags": ["web-search"], "context_window": 8192, "max_tokens": 1024,
      "pricing": {"input": "0.01", "output": "0.04"}
    },
    {
      "id": "vendor/evaluation", "name": "A Classifier",
      "type": "evaluation", "context_window": 32000, "max_tokens": 4096,
      "pricing": {"input": "0.042", "output": "0"}
    }
  ]
}`

func TestVercelCatalogConvertsList(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(vercelListFixture))
	}))
	defer server.Close()

	catalog := &vercelCatalog{client: server.Client(), url: server.URL}
	models, err := catalog.load(context.Background())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(models) != 3 {
		t.Fatalf("models = %d, want 3 (the typed and tool-less ones are not chat models)", len(models))
	}

	first := models[0]
	if first["id"] != "anthropic/claude-sonnet-5.5-batch" || first["name"] != "Claude Sonnet 5.5 (batch)" {
		t.Errorf("identity = %v", first)
	}
	if first["api"] != "anthropic-messages" || first["baseUrl"] != vercelBaseURL {
		t.Errorf("endpoint = %v / %v", first["api"], first["baseUrl"])
	}
	compat, _ := first["compat"].(map[string]any)
	if compat["allowEmptySignature"] != true {
		t.Errorf("compat = %v, want the gateway's empty-signature allowance", compat)
	}
	if first["reasoning"] != true {
		t.Errorf("reasoning = %v, want true", first["reasoning"])
	}
	if first["contextWindow"] != 200000 || first["maxTokens"] != 64000 {
		t.Errorf("limits = %v / %v", first["contextWindow"], first["maxTokens"])
	}
	input, _ := first["input"].([]any)
	if len(input) != 2 || input[1] != "image" {
		t.Errorf("input = %v, want text+image", input)
	}
	cost, _ := first["cost"].(map[string]any)
	if cost["input"] != 3.0 || cost["output"] != 15.0 || cost["cacheRead"] != 0.3 || cost["cacheWrite"] != 3.75 {
		t.Errorf("cost = %v, want per-million rates", cost)
	}

	second := models[1]
	if second["reasoning"] != false {
		t.Errorf("reasoning = %v, want false", second["reasoning"])
	}
	if input, _ := second["input"].([]any); len(input) != 1 {
		t.Errorf("input = %v, want text only", input)
	}
	cost, _ = second["cost"].(map[string]any)
	if cost["input"] != 0.1 || cost["output"] != 0.2 {
		t.Errorf("numeric price = %v, want per-million rates", cost)
	}

	third := models[2]
	if third["id"] != "vendor/untitled" || third["name"] != "vendor/untitled" {
		t.Errorf("a list item without a name = %v, want the id as its name", third)
	}
	if third["contextWindow"] != 4096 || third["maxTokens"] != 4096 {
		t.Errorf("undeclared limits = %v / %v, want the fallback", third["contextWindow"], third["maxTokens"])
	}
	cost, _ = third["cost"].(map[string]any)
	if cost["cacheRead"] != 0.12 || cost["cacheWrite"] != 0.0 {
		t.Errorf("cache cost = %v", cost)
	}
}

func TestVercelCatalogRejectsEmptyList(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[]}`))
	}))
	defer server.Close()
	if _, err := (&vercelCatalog{client: server.Client(), url: server.URL}).load(context.Background()); err == nil {
		t.Fatal("empty catalog should be an error")
	}
}

func TestRefreshVercelRegistersMergedModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(vercelListFixture))
	}))
	defer server.Close()

	original := liveVercelCatalog
	liveVercelCatalog = &vercelCatalog{client: server.Client(), url: server.URL}
	t.Cleanup(func() { liveVercelCatalog = original })

	state := []any{map[string]any{
		"id": "anthropic/claude-sonnet-5.5-batch", "provider": "vercel-ai-gateway",
		"api": "anthropic-messages", "baseUrl": vercelBaseURL, "contextWindow": 111111,
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
					entry["provider"] = "vercel-ai-gateway"
					state = append(state, entry)
				}
				return map[string]any{}, ""
			default:
				return map[string]any{}, ""
			}
		},
	})
	if failure := host.Command(refreshVercelName, ""); failure != "" {
		t.Fatalf("command failure = %q", failure)
	}
	registrations := host.CallsTo("registerProvider")
	if len(registrations) != 1 || registrations[0].Args["name"] != "vercel-ai-gateway" {
		t.Fatalf("registrations = %v", registrations)
	}
	config, _ := registrations[0].Args["config"].(map[string]any)
	models, _ := config["models"].([]any)
	if len(models) != 3 {
		t.Fatalf("registered models = %d, want the 3 fetched, the known one with its host config", len(models))
	}
	first, _ := models[0].(map[string]any)
	// The registry snapshot crosses the fake host as JSON, so its numbers
	// come back as float64.
	if window, _ := first["contextWindow"].(float64); window != 111111 {
		t.Errorf("known model lost its host config: %v", first)
	}
	messages := host.CallsTo("sendMessage")
	if len(messages) != 1 {
		t.Fatalf("sendMessage calls = %d, want 1", len(messages))
	}
	message, _ := messages[0].Args["message"].(map[string]any)
	content, _ := message["content"].(string)
	if !strings.Contains(content, refreshVercelName) || !strings.Contains(content, "vercel-ai-gateway") {
		t.Errorf("result message = %q, want command and provider", content)
	}
	if !strings.Contains(content, "vendor/numeric-price") {
		t.Errorf("result message = %q, want the new model name", content)
	}
}
