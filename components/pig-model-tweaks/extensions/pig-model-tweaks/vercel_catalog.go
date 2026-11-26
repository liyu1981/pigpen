package pigmodeltweaks

// Vercel AI Gateway publishes its whole catalog as a keyless JSON list. This
// file turns the chat models of that list into PiG provider model configs the
// way PiG's own catalog generator builds them from the same endpoint: every
// model of the gateway is served through its Anthropic-compatible Messages
// endpoint, so a model the host does not know yet needs that protocol, that
// endpoint and the gateway's empty-signature compat before a request can go
// out. Models the host already knows keep their host-authored config, so a
// refresh adds models without changing how the current ones are served.

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"slices"
	"strconv"
	"time"
)

const (
	// vercelModelsURL is the public Vercel AI Gateway model list.
	vercelModelsURL = "https://ai-gateway.vercel.sh/v1/models"
	// vercelBaseURL is the endpoint the gateway serves every chat model from.
	vercelBaseURL = "https://ai-gateway.vercel.sh"
)

// vercelCatalog fetches the public gateway list. The URL and client are
// fields so layer-1 tests can point them at a local server.
type vercelCatalog struct {
	client *http.Client
	url    string
}

func newVercelCatalog() *vercelCatalog {
	return &vercelCatalog{
		client: &http.Client{Timeout: 10 * time.Second},
		url:    vercelModelsURL,
	}
}

// load fetches one keyless list and converts it into the `models` array of a
// provider registration. The list is never cached here: the host's registry
// owns the live catalog once these models are registered.
func (c *vercelCatalog) load(ctx context.Context) ([]map[string]any, error) {
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
		Data []vercelModel `json:"data"`
	}
	if err := json.NewDecoder(response.Body).Decode(&list); err != nil {
		return nil, fmt.Errorf("%s: decode models: %w", c.url, err)
	}
	configs := make([]map[string]any, 0, len(list.Data))
	for _, model := range list.Data {
		if !vercelIsChatModel(model) {
			continue
		}
		configs = append(configs, vercelModelConfig(model))
	}
	if len(configs) == 0 {
		return nil, fmt.Errorf("%s: no models in the catalog", c.url)
	}
	return configs, nil
}

// vercelIsChatModel keeps the models a chat session can use. The one list
// also carries image, embedding, speech and other typed models, and the
// evaluation models the host serves as classifiers, so the type has to say
// `language` (a payload that omits the field entirely is taken only when the
// model advertises tools). Tool use is what PiG's loop needs: a language
// model without it, such as an image generator behind the same list, would
// fail at the first tool call. PiG's own catalog generator filters the same
// way.
func vercelIsChatModel(model vercelModel) bool {
	if model.Type != "" && model.Type != "language" {
		return false
	}
	return slices.Contains(model.Tags, "tool-use")
}

// vercelModel is the subset of a gateway list item the catalog needs. The
// pricing object holds a string per token for the fields below and tier
// objects for the ones this catalog does not use, so it stays raw.
type vercelModel struct {
	ID            string          `json:"id"`
	Name          string          `json:"name"`
	Type          string          `json:"type"`
	ContextWindow int             `json:"context_window"`
	MaxTokens     int             `json:"max_tokens"`
	Tags          []string        `json:"tags"`
	Pricing       json.RawMessage `json:"pricing"`
}

// vercelModelConfig converts one gateway list item into the model config the
// host expects from a provider registration, with the values PiG's own
// generator writes for this provider: Anthropic Messages on the gateway's
// base URL, the `allowEmptySignature` compat every built-in gateway model
// carries, per-million costs and a 4096 fallback for a list item that
// declares no limits.
func vercelModelConfig(model vercelModel) map[string]any {
	name := model.Name
	if name == "" {
		name = model.ID
	}
	input := []any{"text"}
	if slices.Contains(model.Tags, "vision") {
		input = append(input, "image")
	}
	contextWindow := model.ContextWindow
	if contextWindow <= 0 {
		contextWindow = 4096
	}
	maxTokens := model.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096
	}
	return map[string]any{
		"id":            model.ID,
		"name":          name,
		"api":           "anthropic-messages",
		"baseUrl":       vercelBaseURL,
		"reasoning":     slices.Contains(model.Tags, "reasoning"),
		"input":         input,
		"contextWindow": contextWindow,
		"maxTokens":     maxTokens,
		"compat": map[string]any{
			"allowEmptySignature": true,
		},
		"cost": map[string]any{
			"input":      vercelPricePerMillion(model.Pricing, "input"),
			"output":     vercelPricePerMillion(model.Pricing, "output"),
			"cacheRead":  vercelPricePerMillion(model.Pricing, "input_cache_read"),
			"cacheWrite": vercelPricePerMillion(model.Pricing, "input_cache_write"),
		},
	}
}

// vercelPricePerMillion reads one price of the raw pricing object and
// converts it from the gateway's per-token rate to PiG's per-million rate. A
// missing field, a tier object or an unparseable value is a zero rate rather
// than a failed refresh: the model is still usable, only its listed cost is
// unknown.
func vercelPricePerMillion(pricing json.RawMessage, key string) float64 {
	if len(pricing) == 0 {
		return 0
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(pricing, &fields); err != nil {
		return 0
	}
	raw, ok := fields[key]
	if !ok {
		return 0
	}
	var perToken float64
	if err := json.Unmarshal(raw, &perToken); err == nil {
		return vercelPerMillion(perToken)
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if parsed, err := strconv.ParseFloat(text, 64); err == nil {
			return vercelPerMillion(parsed)
		}
	}
	return 0
}

// vercelPerMillion converts a per-token price to the per-million rate PiG
// prices with, rounded to six decimals — the precision PiG's own generated
// catalog keeps for this provider.
func vercelPerMillion(perToken float64) float64 {
	perMillion := perToken * 1_000_000
	return math.Round(perMillion*1_000_000) / 1_000_000
}
