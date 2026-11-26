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
	// Layer-1 tests must never call the live provider model lists. A closed
	// local port makes the fetch fail immediately; tests that exercise a
	// conversion point their own catalog at httptest.
	liveOpenCodeCatalog = &openCodeCatalog{
		client: newOpenCodeCatalog().client,
		zenURL: "http://127.0.0.1:0/zen",
		goURL:  "http://127.0.0.1:0/go",
	}
	liveOpenRouterCatalog = &openRouterCatalog{
		client: newOpenRouterCatalog().client,
		url:    "http://127.0.0.1:0/openrouter",
	}
	liveVercelCatalog = &vercelCatalog{
		client: newVercelCatalog().client,
		url:    "http://127.0.0.1:0/vercel",
	}
	os.Exit(m.Run())
}
