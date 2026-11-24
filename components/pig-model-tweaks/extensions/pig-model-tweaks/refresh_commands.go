package pigmodeltweaks

// Commands that replace a provider's model catalog with the one published by
// the provider itself. OpenRouter and OpenCode both expose public, keyless
// model lists; the host keeps ownership of credentials and request routing.

import (
	"context"
	"fmt"
	"slices"
	"strings"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

// The public catalog sources are variables so layer-1 tests can point them at
// local servers instead of the live endpoints.
var (
	liveOpenCodeCatalog   = newOpenCodeCatalog()
	liveOpenRouterCatalog = newOpenRouterCatalog()
)

const (
	refreshOpenRouterName = "pmt-refresh-openrouter-models"
	refreshZenName        = "pmt-refresh-opencode-zen-models"
	refreshGoName         = "pmt-refresh-opencode-go-models"
)

// refreshMessageType is the custom message type used for the visible result of
// a refresh command. It starts no turn, like the status messages of the other
// components, so running the command does not send anything to a model.
const refreshMessageType = "pmt-model-refresh"

// refreshTargets are the built-in provider ids whose catalogs the commands
// refresh. "opencode" is OpenCode Zen and "opencode-go" is its Go endpoint.
var refreshTargets = []struct {
	command  string
	provider string
	summary  string
}{
	{refreshOpenRouterName, "openrouter", "Refreshes the OpenRouter model catalog"},
	{refreshZenName, "opencode", "Refreshes the OpenCode Zen model catalog"},
	{refreshGoName, "opencode-go", "Refreshes the OpenCode Go model catalog"},
}

func registerRefreshCommands(ext *sdk.Extension) {
	for _, target := range refreshTargets {
		target := target
		ext.Command(target.command, target.summary, func(ctx sdk.Context, _ string) error {
			return refreshProviderModels(ctx, target.command, target.provider)
		})
	}
}

func refreshProviderModels(ctx sdk.Context, command, provider string) error {
	// Snapshot the live catalog first so the reply can say what actually
	// changed, not just that a refresh was requested. A failed read is not a
	// refresh failure; it only removes the before/after detail from the reply.
	before, haveBefore := providerModelIDs(ctx, provider)

	var err error
	switch provider {
	case "opencode", "opencode-go":
		err = refreshFromOpenCode(ctx, provider)
	case "openrouter":
		err = refreshFromOpenRouter(ctx)
	}
	if err != nil {
		return fmt.Errorf("%s: %w", command, err)
	}

	after, haveAfter := providerModelIDs(ctx, provider)
	reportRefresh(ctx, command, refreshDetail(provider, before, haveBefore, after, haveAfter))
	return nil
}

// refreshFromOpenCode replaces a built-in OpenCode provider's catalog with the
// models from its keyless inference list. Registering only the models merges
// with the host's existing provider config, so credentials and request routing
// stay with the host while the catalog comes from OpenCode itself.
func refreshFromOpenCode(ctx sdk.Context, provider string) error {
	url := liveOpenCodeCatalog.zenURL
	if provider == "opencode-go" {
		url = liveOpenCodeCatalog.goURL
	}
	models, err := liveOpenCodeCatalog.load(context.Background(), url)
	if err != nil {
		return err
	}
	return ctx.ModelRegistry().RegisterProvider(provider, sdk.ProviderConfig{"models": models})
}

// refreshFromOpenRouter merges OpenRouter's public catalog with the models the
// host already knows. Existing models keep their host-authored protocol,
// endpoint and compat settings; only models missing from the host catalog are
// added with OpenRouter's OpenAI-compatible defaults.
func refreshFromOpenRouter(ctx sdk.Context) error {
	models, err := liveOpenRouterCatalog.load(context.Background())
	if err != nil {
		return err
	}
	existing := providerModelConfigs(ctx, "openrouter")
	return ctx.ModelRegistry().RegisterProvider("openrouter", sdk.ProviderConfig{
		"models": mergeOpenRouterModels(existing, models),
	})
}

// refreshDetail turns the before/after snapshots into a single sentence that
// tells the user whether the catalog changed at all.
func refreshDetail(provider string, before []string, haveBefore bool, after []string, haveAfter bool) string {
	if !haveAfter {
		return fmt.Sprintf("%s: catalog refreshed; the model list could not be read back", provider)
	}
	noun := "models"
	if len(after) == 1 {
		noun = "model"
	}
	summary := fmt.Sprintf("%s: catalog refreshed, %d %s available", provider, len(after), noun)
	if !haveBefore {
		return summary + "."
	}
	added, removed := modelDiff(before, after)
	if len(added) == 0 && len(removed) == 0 {
		return summary + ". No new models — the catalog was already up to date."
	}
	summary += "."
	if len(added) > 0 {
		summary += fmt.Sprintf(" %d new: %s.", len(added), joinModelIDs(added))
	}
	if len(removed) > 0 {
		summary += fmt.Sprintf(" %d removed: %s.", len(removed), joinModelIDs(removed))
	}
	return summary
}

// modelDiff returns the sorted ids added to and removed from the catalog.
func modelDiff(before, after []string) (added, removed []string) {
	old, new := make(map[string]struct{}, len(before)), make(map[string]struct{}, len(after))
	for _, id := range before {
		old[id] = struct{}{}
	}
	for _, id := range after {
		new[id] = struct{}{}
	}
	for id := range new {
		if _, ok := old[id]; !ok {
			added = append(added, id)
		}
	}
	for id := range old {
		if _, ok := new[id]; !ok {
			removed = append(removed, id)
		}
	}
	slices.Sort(added)
	slices.Sort(removed)
	return added, removed
}

// joinModelIDs keeps a long list readable: names are capped so a catalog with
// dozens of additions still fits in one status line.
func joinModelIDs(ids []string) string {
	const limit = 8
	if len(ids) > limit {
		return fmt.Sprintf("%s, … (+%d more)", strings.Join(ids[:limit], ", "), len(ids)-limit)
	}
	return strings.Join(ids, ", ")
}

// providerModelIDs reads the live model list for one provider.
func providerModelIDs(ctx sdk.Context, provider string) ([]string, bool) {
	models, err := ctx.ModelRegistry().GetAll()
	if err != nil {
		return nil, false
	}
	ids := []string{}
	for _, model := range models {
		if owner, _ := model["provider"].(string); owner == provider {
			if id, _ := model["id"].(string); id != "" {
				ids = append(ids, id)
			}
		}
	}
	slices.Sort(ids)
	return ids, true
}

// providerModelConfigs returns the full model configs the host currently has
// for one provider, so an extension refresh can preserve their settings.
func providerModelConfigs(ctx sdk.Context, provider string) []map[string]any {
	models, err := ctx.ModelRegistry().GetAll()
	if err != nil {
		return nil
	}
	out := []map[string]any{}
	for _, model := range models {
		if owner, _ := model["provider"].(string); owner == provider {
			out = append(out, model)
		}
	}
	return out
}

// reportRefresh puts the result in front of the user. A displayed message that
// starts no turn is persistent in the transcript; a transient toast is sent as
// well for immediate feedback, and as the fallback when the host cannot inject
// a message into the conversation (for example outside the TUI).
func reportRefresh(ctx sdk.Context, command, detail string) {
	message := command + ": " + detail
	no := false
	if err := ctx.SendMessage(refreshMessageType, message, true, sdk.SendMessageOptions{TriggerTurn: &no}); err != nil {
		ctx.Notify(message, "info")
		return
	}
	ctx.Notify(command+": refresh complete — see the result in the conversation", "info")
}
