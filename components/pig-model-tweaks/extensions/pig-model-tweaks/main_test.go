package pigmodeltweaks

import (
	"os"
	"testing"
)

// The tests give the extension a home through PIG_HOME. A host's own answer
// (PIG_EXT_HOST_INFO, which sdk.AgentDir reads) must not leak in from the
// environment the tests run in.
func TestMain(m *testing.M) {
	_ = os.Unsetenv("PIG_EXT_HOST_INFO")
	// Layer-1 tests must never call the live opencode.ai endpoints. A closed
	// local port makes the session-start catalog sync fail immediately; tests
	// that exercise the conversion point their own catalog at httptest.
	liveOpenCodeCatalog = &openCodeCatalog{
		client: newOpenCodeCatalog().client,
		zenURL: "http://127.0.0.1:0/zen",
		goURL:  "http://127.0.0.1:0/go",
	}
	liveOpenRouterCatalog = &openRouterCatalog{
		client: newOpenRouterCatalog().client,
		url:    "http://127.0.0.1:0/openrouter",
	}
	os.Exit(m.Run())
}
