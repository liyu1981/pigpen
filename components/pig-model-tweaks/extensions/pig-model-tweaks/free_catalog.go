package pigmodeltweaks

// This file is an independent implementation of a keyless OpenCode Zen model
// catalog. It is MIT licensed, like the rest of this extension. The behavior
// (free-id filtering, models.dev enrichment, conservative defaults and a
// persisted snapshot) is described in the package README; no source from the
// GPL-licensed pi-opencode-free package was copied.

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	// freeProviderID is the model prefix: opencode-free/<model>.
	freeProviderID = "opencode-free"
	// freeBaseURL is the Zen API base used for chat and Responses calls.
	freeBaseURL = "https://opencode.ai/zen/v1"
	// freeProviderName is the display name in pickers.
	freeProviderName = "OpenCode Direct (Free)"
	// freeClientUA identifies the OpenCode CLI, which the free tier expects.
	freeClientUA = "opencode/0.0.0-dev"

	freeZenCatalogURL  = "https://opencode.ai/zen/v1/models"
	freeMetaCatalogURL = "https://models.dev/api.json"

	// Defaults used when models.dev has no entry for a free model.
	freeDefaultContext = 128000
	freeDefaultOutput  = 16384
)

// freeThinkingOrder is the vocabulary Pi uses for thinking levels, in order.
var freeThinkingOrder = []string{"off", "minimal", "low", "medium", "high", "xhigh", "max"}

// freeHeaders are the OpenCode client identity headers. The free tier is
// keyless: no Authorization header is ever added here.
func freeHeaders() map[string]string {
	return map[string]string{
		"x-opencode-client":  "cli",
		"x-opencode-project": "global",
		"User-Agent":         freeClientUA,
	}
}

// freeInfo is one free model after discovery and before conversion into a
// provider model config.
type freeInfo struct {
	ID        string
	Name      string
	Reasoning bool
	Context   int
	Output    int
	Input     []string
	Thinking  map[string]*string
	API       string
}

// freeMeta is the subset of a models.dev entry the catalog needs.
type freeMeta struct {
	Name      string
	Reasoning bool
	Input     []string
	Context   int
	Output    int
	Efforts   []string
	HasToggle bool
	NPM       string
}

// freeCatalog fetches the two catalogs. The fields are injectable so tests can
// point the fetcher at local servers without touching the network.
type freeCatalog struct {
	client  *http.Client
	zenURL  string
	metaURL string
	budget  time.Duration
}

func newFreeCatalog() *freeCatalog {
	return &freeCatalog{client: http.DefaultClient}
}

// load discovers free models. A failure of either source yields no models at
// all: it is the provider's job to keep the previous snapshot in that case.
func (fc *freeCatalog) load(ctx context.Context) []freeInfo {
	client := fc.client
	if client == nil {
		client = http.DefaultClient
	}
	budget := fc.budget
	if budget <= 0 {
		budget = 3 * time.Second
	}
	zenURL, metaURL := fc.zenURL, fc.metaURL
	if zenURL == "" {
		zenURL = freeZenCatalogURL
	}
	if metaURL == "" {
		metaURL = freeMetaCatalogURL
	}
	if ctx == nil {
		ctx = context.Background()
	}
	requestCtx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()

	var (
		advertised []freeAdvertised
		zenOK      bool
		meta       map[string]freeMeta
		wg         sync.WaitGroup
	)
	wg.Add(2)
	go func() {
		defer wg.Done()
		var payload struct {
			Data []freeAdvertised `json:"data"`
		}
		ok, err := fetchJSON(requestCtx, client, zenURL, freeHeaders(), &payload)
		if err == nil && ok {
			advertised = payload.Data
			zenOK = true
		}
	}()
	go func() {
		defer wg.Done()
		meta = fetchFreeMeta(requestCtx, client, metaURL)
	}()
	wg.Wait()
	if !zenOK {
		return nil
	}
	return selectFreeModels(advertised, meta)
}

// freeAdvertised is one row of the Zen /models response.
type freeAdvertised struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

// fetchJSON performs a GET and decodes a successful JSON body into dst. It
// reports whether the response was usable.
func fetchJSON(ctx context.Context, client *http.Client, url string, headers map[string]string, dst any) (bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false, err
	}
	for name, value := range headers {
		req.Header.Set(name, value)
	}
	resp, err := client.Do(req)
	if err != nil {
		return false, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return false, nil
	}
	if err := json.NewDecoder(resp.Body).Decode(dst); err != nil {
		return false, err
	}
	return true, nil
}

// fetchFreeMeta reads the opencode section of models.dev.
func fetchFreeMeta(ctx context.Context, client *http.Client, url string) map[string]freeMeta {
	var raw struct {
		OpenCode struct {
			Models map[string]struct {
				Name      string `json:"name"`
				Reasoning bool   `json:"reasoning"`
				Efforts   []struct {
					Type   string   `json:"type"`
					Values []string `json:"values"`
				} `json:"reasoning_options"`
				Limit struct {
					Context int `json:"context"`
					Output  int `json:"output"`
				} `json:"limit"`
				Modalities struct {
					Input []string `json:"input"`
				} `json:"modalities"`
				Provider struct {
					NPM string `json:"npm"`
				} `json:"provider"`
			} `json:"models"`
		} `json:"opencode"`
	}
	ok, err := fetchJSON(ctx, client, url, nil, &raw)
	if err != nil || !ok || len(raw.OpenCode.Models) == 0 {
		return nil
	}
	out := make(map[string]freeMeta, len(raw.OpenCode.Models))
	for id, model := range raw.OpenCode.Models {
		meta := freeMeta{
			Name: model.Name, Reasoning: model.Reasoning, Input: model.Modalities.Input,
			Context: model.Limit.Context, Output: model.Limit.Output, NPM: model.Provider.NPM,
		}
		for _, option := range model.Efforts {
			switch option.Type {
			case "effort":
				meta.Efforts = option.Values
			case "toggle":
				meta.HasToggle = true
			}
		}
		out[id] = meta
	}
	return out
}

// isFreeID accepts ids ending in -free and the well-known big-pickle id, with
// or without the opencode/ prefix.
func isFreeID(id string) bool {
	bare := strings.TrimPrefix(id, "opencode/")
	lower := strings.ToLower(bare)
	return strings.HasSuffix(lower, "-free") || lower == "big-pickle"
}

// selectFreeModels keeps the free rows and enriches them from models.dev.
func selectFreeModels(rows []freeAdvertised, meta map[string]freeMeta) []freeInfo {
	out := make([]freeInfo, 0, len(rows))
	for _, row := range rows {
		if !isFreeID(row.ID) {
			continue
		}
		bare := strings.TrimPrefix(row.ID, "opencode/")
		base := strings.TrimSuffix(bare, "-free")
		details, ok := meta[bare]
		if !ok {
			details, ok = meta[base]
		}
		name := row.Name
		if name == "" && ok {
			name = details.Name
		}
		if name == "" {
			name = prettyFreeName(row.ID)
		}
		info := freeInfo{
			ID: "opencode/" + bare, Name: name, Input: details.Input,
			Context: freeDefaultContext, Output: freeDefaultOutput,
		}
		if ok {
			info.Reasoning = details.Reasoning
			if details.Context > 0 {
				info.Context = details.Context
			}
			if details.Output > 0 {
				info.Output = details.Output
			}
			info.Thinking = freeEffortMap(details)
			if details.NPM == "@ai-sdk/openai" {
				info.API = "openai-responses"
			}
		}
		out = append(out, info)
	}
	return out
}

func prettyFreeName(id string) string {
	base := strings.TrimSuffix(strings.TrimPrefix(id, "opencode/"), "-free")
	fields := strings.Fields(strings.NewReplacer("_", " ", "-", " ").Replace(base))
	for i, word := range fields {
		runes := []rune(word)
		fields[i] = strings.ToUpper(string(runes[0])) + string(runes[1:])
	}
	return strings.Join(fields, " ") + " (Free)"
}

// freeEffortMap maps models.dev effort values onto Pi's thinking levels. A nil
// entry hides a level; "off" is offered when explicitly listed or when the
// model only has a reasoning toggle.
func freeEffortMap(meta freeMeta) map[string]*string {
	if !meta.Reasoning || len(meta.Efforts) == 0 {
		return nil
	}
	levels := make(map[string]*string, len(freeThinkingOrder))
	for _, level := range freeThinkingOrder {
		levels[level] = nil
	}
	offSeen := false
	for _, effort := range meta.Efforts {
		switch effort {
		case "none":
			off := "off"
			levels["off"] = &off
			offSeen = true
		default:
			if _, exists := levels[effort]; exists {
				value := effort
				levels[effort] = &value
			}
		}
	}
	if meta.HasToggle && !offSeen {
		off := "off"
		levels["off"] = &off
	}
	return levels
}

// freeModelConfig converts discovered information into the provider config
// shape the host expects.
func freeModelConfig(info freeInfo) map[string]any {
	api := info.API
	if api == "" {
		api = "openai-completions"
	}
	input := []any{"text"}
	for _, modality := range info.Input {
		if modality == "image" {
			input = append(input, "image")
			break
		}
	}
	var thinking any
	if info.Thinking != nil {
		levels := make(map[string]any, len(info.Thinking))
		for level, value := range info.Thinking {
			if value == nil {
				levels[level] = nil
			} else {
				levels[level] = *value
			}
		}
		thinking = levels
	}
	return map[string]any{
		"id":               strings.TrimPrefix(info.ID, "opencode/"),
		"name":             info.Name,
		"api":              api,
		"reasoning":        info.Reasoning,
		"thinkingLevelMap": thinking,
		"input":            input,
		"contextWindow":    info.Context,
		"maxTokens":        info.Output,
		"cost":             map[string]any{"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
		"compat": map[string]any{
			"supportsStore":                               false,
			"supportsDeveloperRole":                       false,
			"supportsFinishReason":                        false,
			"maxTokensField":                              "max_tokens",
			"requiresReasoningContentOnAssistantMessages": true,
		},
	}
}

// freeStoredConfig adds the fields needed to recognize our own snapshot on the
// next read; freeRestoredConfig removes them again.
func freeStoredConfig(config map[string]any) map[string]any {
	stored := make(map[string]any, len(config)+2)
	for key, value := range config {
		stored[key] = value
	}
	stored["provider"] = freeProviderID
	stored["baseUrl"] = freeBaseURL
	return stored
}

func freeRestoredConfig(stored map[string]any) map[string]any {
	config := make(map[string]any, len(stored))
	for key, value := range stored {
		if key == "provider" || key == "baseUrl" {
			continue
		}
		config[key] = value
	}
	return config
}

// freeSnapshot reads the persisted catalog, dropping entries that are not ours
// or that we cannot serve.
func freeSnapshot(stored map[string]any) []map[string]any {
	items, _ := stored["models"].([]any)
	out := make([]map[string]any, 0, len(items))
	for _, item := range items {
		model, ok := item.(map[string]any)
		if !ok || model == nil {
			continue
		}
		if id, _ := model["id"].(string); id == "" {
			continue
		}
		if provider, _ := model["provider"].(string); provider != freeProviderID {
			continue
		}
		if api, _ := model["api"].(string); api != "" && api != "openai-completions" && api != "openai-responses" {
			continue
		}
		out = append(out, freeRestoredConfig(model))
	}
	return out
}
