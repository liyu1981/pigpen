package pigmodeltweaks

// The keyless OpenCode Zen provider. Registration does no I/O: the catalog is
// empty until the host restores a snapshot or allows a network refresh.

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"strings"
	"sync"
	"time"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

// freeProvider owns the mutable model catalog and the I/O boundary used by the
// provider callbacks.
type freeProvider struct {
	catalog   *freeCatalog
	mu        sync.Mutex
	models    []map[string]any
	newCallID func() string
}

func newFreeProvider() *freeProvider {
	return &freeProvider{catalog: newFreeCatalog(), newCallID: newFreeCallID}
}

// registerFreeProvider adds the provider to an extension.
func registerFreeProvider(ext *sdk.Extension) {
	p := newFreeProvider()
	if err := ext.RegisterNativeProvider(p.declaration()); err != nil {
		panic(err) // a static declaration; the tests catch programming errors
	}
}

func (p *freeProvider) declaration() *sdk.Provider {
	headers := freeHeaders()
	base := freeBaseURL
	source := freeProviderID
	return &sdk.Provider{
		ID:      freeProviderID,
		Name:    freeProviderName,
		BaseURL: &base,
		Headers: headers,
		Auth: sdk.ProviderAuth{APIKey: &sdk.APIKeyAuth{
			Name: "OpenCode Zen (free models need no key)",
			Check: func(sdk.APIKeyAuthInput) (*sdk.AuthCheck, error) {
				return &sdk.AuthCheck{Type: "api_key", Source: &source}, nil
			},
			Resolve: func(sdk.APIKeyAuthInput) (*sdk.AuthResult, error) {
				// Returning the headers here as well as on the declaration makes
				// sure they reach requests on the host's native provider path.
				return &sdk.AuthResult{
					Auth:   map[string]any{"apiKey": "none", "headers": headers},
					Source: &source,
				}, nil
			},
		}},
		GetModels:     p.getModels,
		RefreshModels: p.refresh,
		Stream:        p.stream,
		StreamSimple:  p.stream,
	}
}

func (p *freeProvider) getModels() ([]map[string]any, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return copyModelMaps(p.models), nil
}

// refresh implements the host's model lifecycle. With the network off the
// stored snapshot is restored; with it on a fresh discovery is published and
// persisted. An empty discovery never wipes a previously good catalog.
func (p *freeProvider) refresh(in sdk.RefreshModelsContext) error {
	apply := func(models []map[string]any) func() error {
		return func() error {
			p.mu.Lock()
			p.models = models
			p.mu.Unlock()
			return nil
		}
	}
	if !in.AllowNetwork {
		restored := freeSnapshot(in.Stored)
		if len(restored) == 0 {
			return nil
		}
		_, err := in.Publish(sdk.ModelsPublication{Update: apply(restored)})
		return err
	}
	if in.Signal != nil && in.Signal.Err() != nil {
		return nil
	}
	found := p.catalog.load(in.Signal)
	if len(found) == 0 {
		return nil
	}
	configs := make([]map[string]any, 0, len(found))
	stored := make([]map[string]any, 0, len(found))
	for _, info := range found {
		config := freeModelConfig(info)
		configs = append(configs, config)
		stored = append(stored, freeStoredConfig(config))
	}
	snapshot, err := json.Marshal(map[string]any{
		"models":    stored,
		"checkedAt": time.Now().UnixMilli(),
	})
	if err != nil {
		return err
	}
	_, err = in.Publish(sdk.ModelsPublication{Persist: snapshot, Update: apply(configs)})
	return err
}

// copyModelMaps deep-copies a model catalog so callers cannot mutate ours.
func copyModelMaps(models []map[string]any) []map[string]any {
	out := make([]map[string]any, len(models))
	for i, model := range models {
		out[i] = copyJSONMap(model)
	}
	return out
}

func copyJSONMap(value map[string]any) map[string]any {
	out := make(map[string]any, len(value))
	for key, item := range value {
		out[key] = copyJSONValue(item)
	}
	return out
}

func copyJSONValue(value any) any {
	switch v := value.(type) {
	case map[string]any:
		return copyJSONMap(v)
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = copyJSONValue(item)
		}
		return out
	case []map[string]any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = copyJSONMap(item)
		}
		return out
	default:
		return value
	}
}

func newFreeCallID() string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return "call_" + hex.EncodeToString(b)
}

// freeRequestBase chooses the endpoint base, preferring a per-model override.
func freeRequestBase(model map[string]any) string {
	base := freeBaseURL
	if override, _ := model["baseUrl"].(string); override != "" {
		base = override
	}
	return strings.TrimRight(base, "/")
}
