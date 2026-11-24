package pigmodeltweaks

// The built-in `opencode` and `opencode-go` catalogs come from PiG's own
// catalog service and can lag behind OpenCode. The keyless inference model
// lists on opencode.ai are public, need no API key, and carry the model's
// protocol, endpoint and metadata, so this file turns them into PiG provider
// model configs. Requests still use the host's own auth for those providers;
// only the catalog is replaced.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

const (
	// zenModelsURL is the public OpenCode Models list.
	zenModelsURL = "https://opencode.ai/inference/v1/models"
	// goModelsURL is the public OpenCode Go list.
	goModelsURL = "https://opencode.ai/inference/go/v1/models"
)

// openCodeCatalog fetches and converts one of the public model lists. The
// client and URLs are fields so tests can point them at local servers.
type openCodeCatalog struct {
	client *http.Client
	zenURL string
	goURL  string
}

func newOpenCodeCatalog() *openCodeCatalog {
	return &openCodeCatalog{
		client: &http.Client{Timeout: 10 * time.Second},
		zenURL: zenModelsURL,
		goURL:  goModelsURL,
	}
}

// registerOpenCodeCatalogs installs both OpenCode catalogs while the extension
// is being built, so even `--list-models` sees them. A failure leaves the
// host's own catalog in place; only a complete, non-empty list is registered.
func registerOpenCodeCatalogs(ext *sdk.Extension, catalog *openCodeCatalog) {
	// A short budget keeps an offline or slow network from delaying startup.
	startup := &openCodeCatalog{
		client: &http.Client{Timeout: 4 * time.Second},
		zenURL: catalog.zenURL,
		goURL:  catalog.goURL,
	}
	targets := []struct {
		provider string
		url      string
	}{
		{"opencode", startup.zenURL},
		{"opencode-go", startup.goURL},
	}
	var wg sync.WaitGroup
	for _, target := range targets {
		target := target
		wg.Add(1)
		go func() {
			defer wg.Done()
			models, err := startup.load(context.Background(), target.url)
			if err != nil || len(models) == 0 {
				return
			}
			ext.RegisterProvider(target.provider, sdk.ProviderConfig{"models": models})
		}()
	}
	wg.Wait()
}

// load fetches one keyless list and converts it into the `models` array of a
// provider registration. The list is never cached here: the host's registry
// owns the live catalog once these models are registered.
func (c *openCodeCatalog) load(ctx context.Context, url string) ([]map[string]any, error) {
	if url == "" {
		return nil, fmt.Errorf("no OpenCode models URL")
	}
	client := c.client
	if client == nil {
		client = http.DefaultClient
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
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
		return nil, fmt.Errorf("%s: %s", url, response.Status)
	}
	var list struct {
		Data []openCodeModel `json:"data"`
	}
	if err := json.NewDecoder(response.Body).Decode(&list); err != nil {
		return nil, fmt.Errorf("%s: decode models: %w", url, err)
	}
	configs := make([]map[string]any, 0, len(list.Data))
	for _, model := range list.Data {
		if model.ID == "" {
			continue
		}
		configs = append(configs, openCodeModelConfig(model))
	}
	if len(configs) == 0 {
		return nil, fmt.Errorf("%s: no models in the catalog", url)
	}
	return configs, nil
}

// openCodeModel is the subset of an inference-list item the catalog needs.
// The cost cache fields can be a number or a map of TTLs, so they are raw.
type openCodeModel struct {
	ID    string `json:"id"`
	Name  string `json:"name"`
	Limit struct {
		Context int `json:"context"`
		Output  int `json:"output"`
	} `json:"limit"`
	Cost struct {
		Text struct {
			Input struct {
				Price float64         `json:"price"`
				Cache json.RawMessage `json:"cache"`
			} `json:"input"`
			Output struct {
				Price float64 `json:"price"`
			} `json:"output"`
		} `json:"text"`
	} `json:"cost"`
	Capabilities struct {
		Reasoning struct {
			Supported bool     `json:"supported"`
			Effort    []string `json:"effort"`
			Toggle    bool     `json:"toggle"`
		} `json:"reasoning"`
	} `json:"capabilities"`
	Modalities struct {
		Input []string `json:"input"`
	} `json:"modalities"`
	API            map[string]openCodeAPI `json:"api"`
	RecommendedAPI string                 `json:"recommended_api"`
}

type openCodeAPI struct {
	BaseURL string `json:"base_url"`
}

// openCodeAPIFor chooses the protocol and endpoint OpenCode recommends for a
// model, falling back to the first declared one.
func openCodeAPIFor(model openCodeModel) (api, baseURL string) {
	names := make([]string, 0, len(model.API))
	for name := range model.API {
		names = append(names, name)
	}
	slices.Sort(names)
	if len(names) == 0 {
		return "", ""
	}
	chosen := names[0]
	if model.RecommendedAPI != "" {
		if _, ok := model.API[model.RecommendedAPI]; ok {
			chosen = model.RecommendedAPI
		}
	}
	switch chosen {
	case "chat-completions":
		api = "openai-completions"
	case "responses":
		api = "openai-responses"
	case "messages":
		api = "anthropic-messages"
	default:
		// A protocol PiG does not implement would fail at request time;
		// Chat Completions is the safest common denominator.
		api = "openai-completions"
	}
	return api, model.API[chosen].BaseURL
}

// openCodeModelConfig converts one inference-list item into the model config
// shape the host expects from a provider registration.
func openCodeModelConfig(model openCodeModel) map[string]any {
	api, baseURL := openCodeAPIFor(model)
	name := model.Name
	if name == "" {
		name = model.ID
	}
	input := []any{"text"}
	for _, modality := range model.Modalities.Input {
		if strings.HasPrefix(modality, "image/") || modality == "image" {
			input = append(input, "image")
			break
		}
	}
	var thinking any
	if model.Capabilities.Reasoning.Supported {
		meta := freeMeta{
			Reasoning: true,
			Efforts:   model.Capabilities.Reasoning.Effort,
			HasToggle: model.Capabilities.Reasoning.Toggle,
		}
		levels := freeEffortMap(meta)
		if len(levels) > 0 {
			thinking = effortMapValues(levels)
		}
	}
	return map[string]any{
		"id":               model.ID,
		"name":             name,
		"api":              api,
		"baseUrl":          baseURL,
		"reasoning":        model.Capabilities.Reasoning.Supported,
		"thinkingLevelMap": thinking,
		"input":            input,
		"contextWindow":    model.Limit.Context,
		"maxTokens":        model.Limit.Output,
		"cost": map[string]any{
			"input":      model.Cost.Text.Input.Price,
			"output":     model.Cost.Text.Output.Price,
			"cacheRead":  flexCost(model.Cost.Text.Input.Cache),
			"cacheWrite": 0.0,
		},
		"compat": map[string]any{
			"supportsStore":                               false,
			"supportsDeveloperRole":                       false,
			"supportsFinishReason":                        false,
			"maxTokensField":                              "max_tokens",
			"requiresReasoningContentOnAssistantMessages": true,
		},
	}
}

func effortMapValues(levels map[string]*string) map[string]any {
	out := make(map[string]any, len(levels))
	for level, value := range levels {
		if value == nil {
			out[level] = nil
		} else {
			out[level] = *value
		}
	}
	return out
}

// flexCost decodes a cache price that is either a number or a map of TTLs.
func flexCost(raw json.RawMessage) float64 {
	if len(raw) == 0 {
		return 0
	}
	var price float64
	if json.Unmarshal(raw, &price) == nil {
		return price
	}
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) != nil {
		return 0
	}
	// The inference list nests the cache price under "read" and "write".
	if read, ok := object["read"]; ok {
		return flexCost(read)
	}
	var tiers map[string]float64
	if json.Unmarshal(raw, &tiers) != nil {
		return 0
	}
	// TTL maps use durations such as "5m" and "1h"; prefer the shortest one
	// when more than one price is offered.
	for _, key := range []string{"5m", "1m", "1h", "24h"} {
		if price, ok := tiers[key]; ok {
			return price
		}
	}
	keys := make([]string, 0, len(tiers))
	for key := range tiers {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	if len(keys) == 0 {
		return 0
	}
	return tiers[keys[0]]
}
