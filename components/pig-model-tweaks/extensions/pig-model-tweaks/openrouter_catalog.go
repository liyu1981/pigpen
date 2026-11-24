package pigmodeltweaks

// OpenRouter publishes its live catalog as a keyless JSON list. This file
// turns that list into PiG provider model configs. Models PiG already knows
// keep their host-provided config (protocol, endpoint, compat, thinking map),
// so switching the source adds models without changing how existing ones are
// served. New models get OpenRouter's OpenAI-compatible defaults.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

const openRouterModelsURL = "https://openrouter.ai/api/v1/models"

// openRouterCatalog fetches the public OpenRouter model list. The URL and
// client are fields so layer-1 tests can point them at a local server.
type openRouterCatalog struct {
	client *http.Client
	url    string
}

func newOpenRouterCatalog() *openRouterCatalog {
	return &openRouterCatalog{
		client: &http.Client{Timeout: 10 * time.Second},
		url:    openRouterModelsURL,
	}
}

func (c *openRouterCatalog) load(ctx context.Context) ([]map[string]any, error) {
	client := c.client
	if client == nil {
		client = http.DefaultClient
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("accept", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("%s: %s", c.url, response.Status)
	}
	var list struct {
		Data []openRouterModel `json:"data"`
	}
	if err := json.NewDecoder(response.Body).Decode(&list); err != nil {
		return nil, fmt.Errorf("%s: decode models: %w", c.url, err)
	}
	configs := make([]map[string]any, 0, len(list.Data))
	for _, model := range list.Data {
		if model.ID == "" {
			continue
		}
		configs = append(configs, openRouterModelConfig(model))
	}
	if len(configs) == 0 {
		return nil, fmt.Errorf("%s: no models in the catalog", c.url)
	}
	return configs, nil
}

// openRouterModel is the subset of an OpenRouter list item the catalog needs.
type openRouterModel struct {
	ID            string `json:"id"`
	Name          string `json:"name"`
	ContextLength int    `json:"context_length"`
	Architecture  struct {
		InputModalities []string `json:"input_modalities"`
	} `json:"architecture"`
	Pricing struct {
		Prompt          string `json:"prompt"`
		Completion      string `json:"completion"`
		InputCacheRead  string `json:"input_cache_read"`
		InputCacheWrite string `json:"input_cache_write"`
	} `json:"pricing"`
	TopProvider struct {
		MaxCompletionTokens *int `json:"max_completion_tokens"`
	} `json:"top_provider"`
	SupportedParameters []string `json:"supported_parameters"`
}

// openRouterModelConfig converts one list item into a provider model config.
// OpenRouter serves every model through its OpenAI-compatible endpoint, and
// prices are per token in the API while PiG's cost is per million tokens.
func openRouterModelConfig(model openRouterModel) map[string]any {
	name := model.Name
	if name == "" {
		name = model.ID
	}
	input := []any{"text"}
	for _, modality := range model.Architecture.InputModalities {
		if modality == "image" {
			input = append(input, "image")
			break
		}
	}
	maxTokens := 0
	if model.TopProvider.MaxCompletionTokens != nil {
		maxTokens = *model.TopProvider.MaxCompletionTokens
	}
	if maxTokens > model.ContextLength {
		maxTokens = model.ContextLength
	}
	reasoning := false
	for _, parameter := range model.SupportedParameters {
		if parameter == "reasoning" || parameter == "reasoning_effort" || parameter == "include_reasoning" {
			reasoning = true
			break
		}
	}
	return map[string]any{
		"id":            model.ID,
		"name":          name,
		"api":           "openai-completions",
		"baseUrl":       "https://openrouter.ai/api/v1",
		"reasoning":     reasoning,
		"input":         input,
		"contextWindow": model.ContextLength,
		"maxTokens":     maxTokens,
		"cost": map[string]any{
			"input":      openRouterPrice(model.Pricing.Prompt),
			"output":     openRouterPrice(model.Pricing.Completion),
			"cacheRead":  openRouterPrice(model.Pricing.InputCacheRead),
			"cacheWrite": openRouterPrice(model.Pricing.InputCacheWrite),
		},
	}
}

func openRouterPrice(raw string) float64 {
	if raw == "" {
		return 0
	}
	price, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0
	}
	return price * 1_000_000
}

// mergeOpenRouterModels keeps the host's config for models it already knows
// and adds the ones OpenRouter published that were not in the old catalog.
// Order follows the API list, with old-only models appended so no alias is
// silently dropped.
func mergeOpenRouterModels(existing, fetched []map[string]any) []map[string]any {
	known := make(map[string]map[string]any, len(existing))
	for _, model := range existing {
		if id, _ := model["id"].(string); id != "" {
			known[id] = model
		}
	}
	out := make([]map[string]any, 0, len(fetched)+len(known))
	seen := make(map[string]bool, len(fetched))
	for _, model := range fetched {
		id, _ := model["id"].(string)
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		if old, ok := known[id]; ok {
			out = append(out, old)
		} else {
			out = append(out, model)
		}
	}
	for _, model := range existing {
		id, _ := model["id"].(string)
		if id != "" && !seen[id] {
			out = append(out, model)
			seen[id] = true
		}
	}
	return out
}
