package pigmodeltweaks

// Commands that ask the host to re-fetch a provider's model catalog over the
// network. The host owns the built-in catalogs, so the extension uses its
// public model-registry refresh instead of writing to the host's model store.

import (
	"fmt"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

const (
	refreshOpenRouterName = "pmt-refresh-openrouter-models"
	refreshZenName        = "pmt-refresh-opencode-zen-models"
	refreshGoName         = "pmt-refresh-opencode-go-models"
)

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
	allowNetwork, force := true, true
	result, err := ctx.ModelRegistry().Refresh(sdk.ModelsRefreshOptions{
		AllowNetwork: &allowNetwork,
		Force:        &force,
		Providers:    []string{provider},
	})
	if err != nil {
		return fmt.Errorf("%s: %w", command, err)
	}
	if result.Aborted {
		ctx.Notify(command+": refresh aborted", "info")
		return nil
	}
	if message, failed := result.Errors[provider]; failed {
		ctx.Notify(fmt.Sprintf("%s: refresh failed for %s: %s", command, provider, message), "error")
		return nil
	}
	count := 0
	if models, err := ctx.ModelRegistry().GetAll(); err == nil {
		for _, model := range models {
			if owner, _ := model["provider"].(string); owner == provider {
				count++
			}
		}
	}
	ctx.Notify(fmt.Sprintf("%s: %s refreshed (%d models)", command, provider, count), "info")
	return nil
}
