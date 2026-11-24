package pigmodeltweaks

// Layer-1 tests for the keyless OpenCode Zen provider and the model-refresh
// commands. All I/O goes through local servers or the fake host, so the tests
// need no network.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

// freeCollect drains a provider stream and fails the test if it never ends.
func freeCollect(t *testing.T, stream *sdk.ModelEventStream) ([]map[string]any, map[string]any) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var events []map[string]any
	for event := range stream.Events(ctx) {
		events = append(events, event)
	}
	if ctx.Err() != nil {
		t.Fatalf("stream did not finish: %v", ctx.Err())
	}
	return events, stream.Result()
}

func freeEventTypes(events []map[string]any) []string {
	types := make([]string, 0, len(events))
	for _, event := range events {
		types = append(types, fmt.Sprint(event["type"]))
	}
	return types
}

func freeTerminalMessage(t *testing.T, events []map[string]any) map[string]any {
	t.Helper()
	if len(events) == 0 {
		t.Fatal("no stream events")
	}
	last := events[len(events)-1]
	message, ok := last["message"].(map[string]any)
	if !ok {
		message, ok = last["error"].(map[string]any)
	}
	if !ok {
		t.Fatalf("no message in terminal event %v", last)
	}
	return message
}

func freeTranscript(text string) map[string]any {
	return map[string]any{"messages": []any{map[string]any{"role": "user", "content": text}}}
}

func freeTestModel(api string) map[string]any {
	return map[string]any{
		"id": "hy3-free", "name": "Hy3 (Free)", "api": api,
		"provider": freeProviderID, "reasoning": false, "input": []any{"text"},
	}
}

func freeServeJSON(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, body)
	}))
}

func TestFreeIDFilter(t *testing.T) {
	accepted := []string{"m-free", "opencode/m-free", "BIG-PICKLE", "opencode/big-pickle", "x-FREE"}
	for _, id := range accepted {
		if !isFreeID(id) {
			t.Errorf("isFreeID(%q) = false, want true", id)
		}
	}
	for _, id := range []string{"gpt-5", "big-pickle-pro", "free", "opencode/x-free-thing"} {
		if isFreeID(id) {
			t.Errorf("isFreeID(%q) = true, want false", id)
		}
	}
}

func TestFreeDiscoveryEnrichesFromModelsDev(t *testing.T) {
	zen := freeServeJSON(t, `{"data":[
		{"id":"opencode/shark-free","name":"Shark"},
		{"id":"big-pickle"},
		{"id":"muse-spark-free"},
		{"id":"gpt-5"}
	]}`)
	defer zen.Close()
	meta := freeServeJSON(t, `{"opencode":{"models":{
		"shark":{"name":"Shark Meta","reasoning":true,
			"reasoning_options":[{"type":"effort","values":["minimal","high","none"]},{"type":"toggle"}],
			"limit":{"context":200000,"output":8192},"modalities":{"input":["text","image"]}},
		"muse-spark":{"name":"Muse Spark","reasoning":false,
			"limit":{"context":65536,"output":4096},"provider":{"npm":"@ai-sdk/openai"}}
	}}}`)
	defer meta.Close()

	fc := &freeCatalog{client: zen.Client(), zenURL: zen.URL, metaURL: meta.URL}
	found := fc.load(context.Background())
	if len(found) != 3 {
		t.Fatalf("discovered %d models, want 3: %+v", len(found), found)
	}
	byID := map[string]freeInfo{}
	for _, info := range found {
		byID[info.ID] = info
	}
	shark := byID["opencode/shark-free"]
	if shark.Name != "Shark" || !shark.Reasoning || shark.Context != 200000 || shark.Output != 8192 {
		t.Errorf("shark metadata = %+v", shark)
	}
	if !reflect.DeepEqual(shark.Input, []string{"text", "image"}) {
		t.Errorf("shark input = %v", shark.Input)
	}
	if got := freeEffortMapValue(shark.Thinking, "off"); got != "off" {
		t.Errorf("shark off = %v, want off", got)
	}
	if got := freeEffortMapValue(shark.Thinking, "minimal"); got != "minimal" {
		t.Errorf("shark minimal = %v", got)
	}
	if got := freeEffortMapValue(shark.Thinking, "max"); got != nil {
		t.Errorf("hidden level max = %v, want nil", got)
	}
	if byID["opencode/big-pickle"].Context != freeDefaultContext || byID["opencode/big-pickle"].Name != "Big Pickle (Free)" {
		t.Errorf("default metadata = %+v", byID["opencode/big-pickle"])
	}
	if byID["opencode/muse-spark-free"].API != "openai-responses" {
		t.Errorf("muse api = %q, want openai-responses", byID["opencode/muse-spark-free"].API)
	}
	config := freeModelConfig(shark)
	if config["id"] != "shark-free" || config["api"] != "openai-completions" {
		t.Errorf("provider config = %v", config)
	}
}

func freeEffortMapValue(levels map[string]*string, level string) any {
	value, exists := levels[level]
	if !exists || value == nil {
		return nil
	}
	return *value
}

func TestFreeDiscoveryFailureIsEmpty(t *testing.T) {
	fc := &freeCatalog{zenURL: "http://127.0.0.1:1/models", metaURL: "http://127.0.0.1:1/meta"}
	if got := fc.load(context.Background()); len(got) != 0 {
		t.Fatalf("failed discovery = %+v, want empty", got)
	}
}

func TestFreeProviderStartsEmptyAndIsKeyless(t *testing.T) {
	p := newFreeProvider()
	models, err := p.getModels()
	if err != nil || len(models) != 0 {
		t.Fatalf("models = %v, %v; want empty", models, err)
	}
	decl := p.declaration()
	if decl.ID != freeProviderID || decl.Stream == nil || decl.StreamSimple == nil {
		t.Fatalf("unexpected declaration: %+v", decl)
	}
	if decl.Headers["Authorization"] != "" {
		t.Errorf("default headers must not include Authorization: %v", decl.Headers)
	}
	result, err := decl.Auth.APIKey.Resolve(sdk.APIKeyAuthInput{})
	if err != nil {
		t.Fatalf("resolve auth: %v", err)
	}
	if result.Auth["apiKey"] != "none" || result.Auth["headers"] == nil {
		t.Errorf("auth = %v", result.Auth)
	}
}

func TestFreeRefreshRestoresSnapshot(t *testing.T) {
	p := newFreeProvider()
	stored := map[string]any{"models": []any{
		map[string]any{"id": "kept", "provider": freeProviderID, "api": "openai-completions", "name": "Kept"},
		map[string]any{"id": "foreign", "provider": "openrouter", "api": "openai-completions"},
		map[string]any{"id": "bad-api", "provider": freeProviderID, "api": "anthropic-messages"},
	}}
	var published json.RawMessage
	called := false
	err := p.refresh(sdk.RefreshModelsContext{
		Stored:       stored,
		AllowNetwork: false,
		Publish: func(publication sdk.ModelsPublication) (bool, error) {
			called = true
			published = publication.Persist
			if err := publication.Update(); err != nil {
				return false, err
			}
			return true, nil
		},
	})
	if err != nil || !called || len(published) != 0 {
		t.Fatalf("restore: err=%v called=%v persist=%s", err, called, published)
	}
	models, _ := p.getModels()
	if len(models) != 1 || models[0]["id"] != "kept" {
		t.Fatalf("restored = %v", models)
	}
	if _, exists := models[0]["provider"]; exists {
		t.Errorf("restored model still has provider field: %v", models[0])
	}
}

func TestFreeRefreshPublishesAndKeepsLastGoodSnapshot(t *testing.T) {
	zen := freeServeJSON(t, `{"data":[{"id":"shark-free","name":"Shark"}]}`)
	defer zen.Close()
	meta := freeServeJSON(t, `{"opencode":{"models":{}}}`)
	defer meta.Close()
	p := newFreeProvider()
	p.catalog.zenURL, p.catalog.metaURL = zen.URL, meta.URL
	var persisted map[string]any
	publish := func(publication sdk.ModelsPublication) (bool, error) {
		if len(publication.Persist) > 0 {
			if err := json.Unmarshal(publication.Persist, &persisted); err != nil {
				return false, err
			}
		}
		return true, publication.Update()
	}
	if err := p.refresh(sdk.RefreshModelsContext{AllowNetwork: true, Publish: publish}); err != nil {
		t.Fatalf("refresh: %v", err)
	}
	models, _ := p.getModels()
	if len(models) != 1 || models[0]["id"] != "shark-free" {
		t.Fatalf("refreshed = %v", models)
	}
	if len(persisted) != 2 || persisted["checkedAt"] == nil {
		t.Errorf("persisted snapshot = %v", persisted)
	}

	// A failed discovery must not wipe the catalog we just published.
	empty := freeServeJSON(t, `{"data":[]}`)
	defer empty.Close()
	p.catalog.zenURL = empty.URL
	if err := p.refresh(sdk.RefreshModelsContext{AllowNetwork: true, Publish: publish}); err != nil {
		t.Fatalf("empty refresh: %v", err)
	}
	models, _ = p.getModels()
	if len(models) != 1 || models[0]["id"] != "shark-free" {
		t.Fatalf("empty refresh wiped the catalog: %v", models)
	}
}

func TestFreeChatStream(t *testing.T) {
	var body map[string]any
	var headers http.Header
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers = r.Header.Clone()
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"choices":[{"delta":{"reasoning_content":"think"},"index":0}]}`,
			`data: {"choices":[{"delta":{"content":"hello"},"index":0}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"go","arguments":"{\"x\":"}}]},"index":0}]}`,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"index":0}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":4}}}`,
			"data: [DONE]",
			"",
		}, "\n"))
	}))
	defer server.Close()

	p := newFreeProvider()
	p.catalog.client = server.Client()
	model := freeTestModel("openai-completions")
	model["baseUrl"] = server.URL
	opts := sdk.ProviderStreamOptions{Values: map[string]any{
		"maxTokens": 321, "temperature": 0.4,
		"headers": map[string]any{"Authorization": nil, "X-Test": "yes"},
	}}
	stream, err := p.stream(model, freeTranscript("hi"), opts)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	events, _ := freeCollect(t, stream)
	types := freeEventTypes(events)
	if types[0] != "start" || types[len(types)-1] != "done" {
		t.Fatalf("event types = %v", types)
	}
	for _, want := range []string{"thinking_delta", "text_delta", "toolcall_start", "toolcall_delta", "toolcall_end"} {
		found := false
		for _, typ := range types {
			if typ == want {
				found = true
			}
		}
		if !found {
			t.Errorf("missing %s in %v", want, types)
		}
	}
	message := freeTerminalMessage(t, events)
	if message["stopReason"] != "toolUse" {
		t.Errorf("stop reason = %v, want toolUse", message["stopReason"])
	}
	usage := message["usage"].(map[string]any)
	if usage["input"] != 6 || usage["cacheRead"] != 4 || usage["output"] != 3 {
		t.Errorf("usage = %v", usage)
	}
	if body["max_tokens"] != float64(321) || body["stream"] != true || body["model"] != "hy3-free" {
		t.Errorf("request body = %v", body)
	}
	if headers.Get("x-opencode-client") != "cli" || headers.Get("x-opencode-project") != "global" {
		t.Errorf("missing OpenCode headers: %v", headers)
	}
	if headers.Get("Authorization") != "" || headers.Get("X-Test") != "yes" {
		t.Errorf("header override failed: %v", headers)
	}
	tools, _ := body["stream_options"].(map[string]any)
	if tools["include_usage"] != true {
		t.Errorf("stream_options = %v", body["stream_options"])
	}
}

func TestFreeResponsesStream(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"type":"response.output_text.delta","output_index":0,"delta":"hey"}`,
			`data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"a\":}"}`,
			`data: {"type":"response.function_call_arguments.done","output_index":1,"arguments":"{\"a\":1}"}`,
			`data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","call_id":"c1","name":"look","arguments":"{\"a\":1}"}}`,
			`data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":9,"output_tokens":2,"total_tokens":11}}}`,
			"data: [DONE]",
			"",
		}, "\n"))
	}))
	defer server.Close()

	p := newFreeProvider()
	p.catalog.client = server.Client()
	model := freeTestModel("openai-responses")
	model["baseUrl"] = server.URL
	stream, err := p.stream(model, freeTranscript("hi"), sdk.ProviderStreamOptions{Values: map[string]any{"maxTokens": 40}})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	events, _ := freeCollect(t, stream)
	message := freeTerminalMessage(t, events)
	if message["stopReason"] != "toolUse" {
		t.Errorf("stop reason = %v", message["stopReason"])
	}
	if body["store"] != false || body["stream"] != true || body["max_output_tokens"] != float64(40) {
		t.Errorf("request body = %v", body)
	}
	input, _ := body["input"].([]any)
	if len(input) == 0 {
		t.Errorf("request has no input: %v", body)
	}
}

func TestFreeChatHTTPErrorAndAbort(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":"bad request"}`)
	}))
	defer server.Close()
	p := newFreeProvider()
	p.catalog.client = server.Client()
	model := freeTestModel("openai-completions")
	model["baseUrl"] = server.URL

	events, _ := freeCollect(t, mustFreeStream(t, p, model, freeTranscript("hi"), sdk.ProviderStreamOptions{}))
	last := events[len(events)-1]
	if last["type"] != "error" || !strings.Contains(fmt.Sprint(last["error"]), "400") {
		t.Errorf("error event = %v", last)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	stream, err := p.stream(model, freeTranscript("hi"), sdk.ProviderStreamOptions{Signal: ctx})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	events, _ = freeCollect(t, stream)
	if last := events[len(events)-1]; last["type"] != "error" || last["reason"] != "aborted" {
		t.Errorf("abort event = %v", last)
	}
}

func TestFreeRequestHooks(t *testing.T) {
	var seenStatus int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body["model"] != "replaced" {
			t.Errorf("payload hook was not applied: %v", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"index\":0}],\"usage\":null}\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\",\"index\":0}]}\ndata: [DONE]\n")
	}))
	defer server.Close()
	p := newFreeProvider()
	p.catalog.client = server.Client()
	model := freeTestModel("openai-completions")
	model["baseUrl"] = server.URL
	opts := sdk.ProviderStreamOptions{
		Values: map[string]any{},
		OnPayload: func(payload any, _ map[string]any) (any, error) {
			replacement := payload.(map[string]any)
			replacement["model"] = "replaced"
			return replacement, nil
		},
		OnResponse: func(response map[string]any, _ map[string]any) error {
			seenStatus, _ = response["status"].(int)
			return nil
		},
	}
	events, _ := freeCollect(t, mustFreeStream(t, p, model, freeTranscript("hi"), opts))
	if events[len(events)-1]["type"] != "done" || seenStatus != 200 {
		t.Errorf("events = %v, status = %d", freeEventTypes(events), seenStatus)
	}
}

func TestFreeRefreshRespectsAbort(t *testing.T) {
	p := newFreeProvider()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	called := false
	err := p.refresh(sdk.RefreshModelsContext{
		AllowNetwork: true,
		Signal:       ctx,
		Publish: func(sdk.ModelsPublication) (bool, error) {
			called = true
			return true, nil
		},
	})
	if err != nil || called {
		t.Fatalf("aborted refresh: err=%v publish called=%v", err, called)
	}
}

func TestFreeChatRequestShape(t *testing.T) {
	transcript := map[string]any{"messages": []any{
		map[string]any{
			"role": "system", "content": "Be brief.",
			"sections":     map[string]string{"zz": "last", "aa": "first"},
			"toolsAdded":   []any{map[string]any{"name": "search", "description": "Find things"}},
			"toolsRemoved": []any{map[string]any{"name": "unused"}},
		},
		map[string]any{"role": "user", "content": "hello"},
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "thinking", "thinking": "hmm"},
			map[string]any{"type": "text", "text": "sure"},
			map[string]any{"type": "toolCall", "id": "c1", "name": "search", "arguments": map[string]any{"q": "x"}},
		}},
		map[string]any{"role": "toolResult", "toolCallId": "c1", "content": []any{
			map[string]any{"type": "text", "text": "found"},
			map[string]any{"type": "image", "data": "AAA", "mimeType": "image/png"},
		}},
	}}
	model := freeTestModel("openai-completions")
	model["reasoning"] = true
	model["thinkingLevelMap"] = map[string]any{"high": "high", "off": nil}
	request, err := freeChatRequest(model, transcript, sdk.ProviderStreamOptions{Values: map[string]any{
		"reasoningEffort": "high",
	}})
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	if request["reasoning_effort"] != "high" {
		t.Errorf("reasoning effort = %v", request["reasoning_effort"])
	}
	messages, _ := request["messages"].([]any)
	if len(messages) < 4 {
		t.Fatalf("messages = %v", messages)
	}
	system := messages[0].(map[string]any)["content"].(string)
	if !strings.Contains(system, "Be brief.") || !strings.HasSuffix(system, "last") || !strings.Contains(system, "first") {
		t.Errorf("system prompt = %q", system)
	}
	assistant := messages[2].(map[string]any)
	if assistant["reasoning_content"] != "hmm" || assistant["tool_calls"] == nil {
		t.Errorf("assistant replay = %v", assistant)
	}
	tool := messages[3].(map[string]any)
	if tool["role"] != "tool" || tool["tool_call_id"] != "c1" {
		t.Errorf("tool message = %v", tool)
	}
	tools, _ := request["tools"].([]any)
	if len(tools) != 1 {
		t.Errorf("tools = %v", tools)
	}
}

func TestRefreshCommandHostError(t *testing.T) {
	host := StartHost(t, Extension(), HostOptions{
		OnCall: func(method string, args map[string]any) (map[string]any, string) {
			if method == "refreshModelRegistry" {
				return nil, "catalog unavailable"
			}
			return map[string]any{}, ""
		},
	})
	failure := host.Command(refreshGoName, "")
	if !strings.Contains(failure, refreshGoName) || !strings.Contains(failure, "catalog unavailable") {
		t.Errorf("command failure = %q", failure)
	}
}

func mustFreeStream(t *testing.T, p *freeProvider, model, transcript map[string]any, opts sdk.ProviderStreamOptions) *sdk.ModelEventStream {
	t.Helper()
	stream, err := p.stream(model, transcript, opts)
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	return stream
}

func TestRefreshCommandsCallHost(t *testing.T) {
	for _, target := range refreshTargets {
		t.Run(target.command, func(t *testing.T) {
			host := StartHost(t, Extension(), HostOptions{
				OnCall: func(method string, args map[string]any) (map[string]any, string) {
					switch method {
					case "refreshModelRegistry":
						return map[string]any{"aborted": false, "errors": map[string]any{}}, ""
					case "getModelRegistryState":
						return map[string]any{"models": []any{}, "providers": map[string]any{}, "registered": []any{}}, ""
					default:
						return map[string]any{}, ""
					}
				},
			})
			_ = host.Command(target.command, "")
			calls := host.CallsTo("refreshModelRegistry")
			if len(calls) != 1 {
				t.Fatalf("refresh calls = %d, want 1", len(calls))
			}
			providers, _ := calls[0].Args["providers"].([]any)
			if len(providers) != 1 || providers[0] != target.provider {
				t.Errorf("providers = %v, want [%s]", providers, target.provider)
			}
			if calls[0].Args["allowNetwork"] != true || calls[0].Args["force"] != true {
				t.Errorf("refresh args = %v, want allowNetwork+force", calls[0].Args)
			}
		})
	}
}
