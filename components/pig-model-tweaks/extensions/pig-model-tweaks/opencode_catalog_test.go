package pigmodeltweaks

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

const openCodeListFixture = `{
  "object": "list",
  "data": [
    {
      "id": "hy4-flash-free", "name": "Hy4 flash Free", "type": "chat",
      "limit": {"context": 1048576, "output": 65536},
      "cost": {"text": {"input": {"price": 0, "cache": {"read": 0}}, "output": {"price": 0}}},
      "capabilities": {"reasoning": {"supported": true, "effort": ["high"]}},
      "modalities": {"input": ["text", "image/jpeg"]},
      "api": {"chat-completions": {"base_url": "https://opencode.ai/inference/go/openai/v1"}},
      "recommended_api": "chat-completions"
    },
    {
      "id": "qwen4-max", "name": "Qwen4 Max", "type": "chat",
      "limit": {"context": 1000000, "output": 262144},
      "cost": {"text": {"input": {"price": 1.5, "cache": {"read": 0.075, "write": {"5m": 1.875, "1h": 3}}}, "output": {"price": 4.5}}},
      "capabilities": {"reasoning": {"supported": true, "toggle": true, "effort": ["low", "high", "xhigh"]}},
      "modalities": {"input": ["text"]},
      "api": {"messages": {"base_url": "https://opencode.ai/inference/go/anthropic/v1"}},
      "recommended_api": "messages"
    }
  ]
}`

func TestOpenCodeCatalogConvertsInferenceList(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(openCodeListFixture))
	}))
	defer server.Close()

	catalog := newOpenCodeCatalog()
	models, err := catalog.load(context.Background(), server.URL)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("models = %d, want 2", len(models))
	}

	first := models[0]
	if first["id"] != "hy4-flash-free" || first["name"] != "Hy4 flash Free" {
		t.Errorf("first model identity = %v", first)
	}
	if first["api"] != "openai-completions" {
		t.Errorf("api = %v, want openai-completions", first["api"])
	}
	if first["baseUrl"] != "https://opencode.ai/inference/go/openai/v1" {
		t.Errorf("baseUrl = %v", first["baseUrl"])
	}
	if first["contextWindow"] != 1048576 || first["maxTokens"] != 65536 {
		t.Errorf("limits = %v/%v", first["contextWindow"], first["maxTokens"])
	}
	input, _ := first["input"].([]any)
	if len(input) != 2 || input[1] != "image" {
		t.Errorf("input = %v, want text+image", input)
	}
	thinking, _ := first["thinkingLevelMap"].(map[string]any)
	if thinking["high"] != "high" || thinking["off"] != nil {
		t.Errorf("thinking = %v", thinking)
	}

	second := models[1]
	if second["api"] != "anthropic-messages" || second["baseUrl"] != "https://opencode.ai/inference/go/anthropic/v1" {
		t.Errorf("second model endpoint = %v / %v", second["api"], second["baseUrl"])
	}
	cost, _ := second["cost"].(map[string]any)
	if cost["input"] != 1.5 || cost["output"] != 4.5 || cost["cacheRead"] != 0.075 {
		t.Errorf("cost = %v", cost)
	}
	thinking, _ = second["thinkingLevelMap"].(map[string]any)
	if thinking["xhigh"] != "xhigh" || thinking["off"] != "off" {
		t.Errorf("toggle thinking = %v", thinking)
	}
}

func TestOpenCodeCatalogRejectsEmptyList(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[]}`))
	}))
	defer server.Close()
	if _, err := newOpenCodeCatalog().load(context.Background(), server.URL); err == nil {
		t.Fatal("empty catalog should be an error")
	}
}

func TestRefreshCommandsRegisterOpenCodeModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(openCodeListFixture))
	}))
	defer server.Close()

	original := liveOpenCodeCatalog
	liveOpenCodeCatalog = &openCodeCatalog{client: server.Client(), zenURL: server.URL, goURL: server.URL}
	t.Cleanup(func() { liveOpenCodeCatalog = original })

	for _, tc := range []struct {
		command  string
		provider string
	}{
		{refreshZenName, "opencode"},
		{refreshGoName, "opencode-go"},
	} {
		t.Run(tc.command, func(t *testing.T) {
			var registered []any
			host := StartHost(t, Extension(), HostOptions{
				OnCall: func(method string, args map[string]any) (map[string]any, string) {
					switch method {
					case "getModelRegistryState":
						return map[string]any{"models": registered, "providers": map[string]any{}, "registered": []any{}}, ""
					case "registerProvider":
						config, _ := args["config"].(map[string]any)
						models, _ := config["models"].([]any)
						for _, model := range models {
							entry, _ := model.(map[string]any)
							entry["provider"] = tc.provider
							registered = append(registered, entry)
						}
						return map[string]any{}, ""
					default:
						return map[string]any{}, ""
					}
				},
			})
			if failure := host.Command(tc.command, ""); failure != "" {
				t.Fatalf("command failure = %q", failure)
			}
			registrations := host.CallsTo("registerProvider")
			if len(registrations) != 1 {
				t.Fatalf("registerProvider calls = %d, want 1", len(registrations))
			}
			if registrations[0].Args["name"] != tc.provider {
				t.Errorf("provider = %v, want %s", registrations[0].Args["name"], tc.provider)
			}
			config, _ := registrations[0].Args["config"].(map[string]any)
			models, _ := config["models"].([]any)
			if len(models) != 2 {
				t.Fatalf("registered models = %d, want 2", len(models))
			}
			first, _ := models[0].(map[string]any)
			if first["id"] != "hy4-flash-free" {
				t.Errorf("first model = %v", first)
			}
			if calls := host.CallsTo("refreshModelRegistry"); len(calls) != 0 {
				t.Errorf("host refresh calls = %d, want 0 for OpenCode providers", len(calls))
			}
			messages := host.CallsTo("sendMessage")
			if len(messages) != 1 {
				t.Fatalf("sendMessage calls = %d, want 1", len(messages))
			}
			message, _ := messages[0].Args["message"].(map[string]any)
			content, _ := message["content"].(string)
			if !strings.Contains(content, "hy4-flash-free") {
				t.Errorf("result message = %q, want added model name", content)
			}
		})
	}
}

func TestOpenCodeAPIProfileMapping(t *testing.T) {
	for _, tc := range []struct {
		protocol string
		want     string
	}{
		{"chat-completions", "openai-completions"},
		{"responses", "openai-responses"},
		{"messages", "anthropic-messages"},
		{"unknown", "openai-completions"},
	} {
		model := openCodeModel{RecommendedAPI: tc.protocol, API: map[string]openCodeAPI{tc.protocol: {BaseURL: "https://example.test/v1"}}}
		api, base := openCodeAPIFor(model)
		if api != tc.want || base != "https://example.test/v1" {
			t.Errorf("protocol %s -> %s/%s, want %s", tc.protocol, api, base, tc.want)
		}
	}
}

func TestFlexCostTTLMap(t *testing.T) {
	var raw json.RawMessage = json.RawMessage(`{"5m":1.875,"1h":3}`)
	if got := flexCost(raw); got != 1.875 {
		t.Errorf("flexCost = %v, want 1.875", got)
	}
	raw = json.RawMessage(`0.075`)
	if got := flexCost(raw); got != 0.075 {
		t.Errorf("flexCost = %v, want 0.075", got)
	}
}
