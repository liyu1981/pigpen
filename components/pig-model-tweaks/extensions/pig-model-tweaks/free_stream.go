package pigmodeltweaks

// Streaming transport and event emission for the keyless Zen provider. This is
// an independent MIT implementation of the OpenAI-compatible wire protocol the
// free models speak; see free_catalog.go for the licensing note.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

// stream starts a turn and returns immediately. Everything after that happens
// on a goroutine, and failures are delivered as stream events rather than Go
// errors so the host sees the same event sequence for a bad response as for a
// successful one.
func (p *freeProvider) stream(model, transcript map[string]any, opts sdk.ProviderStreamOptions) (*sdk.ModelEventStream, error) {
	out := sdk.CreateAssistantMessageEventStream()
	ctx := opts.Signal
	if ctx == nil {
		ctx = context.Background()
	}
	api, _ := model["api"].(string)
	go func() {
		switch api {
		case "", "openai-completions":
			p.chatTurn(ctx, out, model, transcript, opts)
		case "openai-responses":
			p.responseTurn(ctx, out, model, transcript, opts)
		default:
			emitter := newFreeEmitter(out, model, sdk.ProviderStreamOptions{})
			emitter.fail(fmt.Errorf("opencode-free cannot serve API %q", api), false)
		}
	}()
	return out, nil
}

// freeEmitter builds the assistant message one event at a time. Each event
// carries a deep snapshot, so later mutations never change a delivered event.
type freeEmitter struct {
	out     *sdk.ModelEventStream
	modelID string
	api     string
	info    map[string]any
	blocks  []map[string]any
	usage   map[string]any
	stop    string
	open    string
	clock   func() time.Time
	callID  func() string
	observe func(any, map[string]any) error
}

func newFreeEmitter(out *sdk.ModelEventStream, model map[string]any, opts sdk.ProviderStreamOptions) *freeEmitter {
	id, _ := model["id"].(string)
	api, _ := model["api"].(string)
	if api == "" {
		api = "openai-completions"
	}
	e := &freeEmitter{
		out: out, modelID: id, api: api, info: model,
		stop: "stop", clock: time.Now, callID: newFreeCallID,
	}
	if opts.OnProviderStreamEvent != nil {
		e.observe = opts.OnProviderStreamEvent
	}
	return e
}

func (e *freeEmitter) snapshot() map[string]any {
	content := make([]any, len(e.blocks))
	for i, block := range e.blocks {
		content[i] = copyJSONMap(block)
	}
	return map[string]any{
		"role": "assistant", "api": e.api, "provider": freeProviderID,
		"model": e.modelID, "content": content, "stopReason": e.stop,
		"timestamp": e.clock().UnixMilli(), "usage": copyJSONMap(e.usage),
	}
}

func freeEmptyUsage() map[string]any {
	return map[string]any{
		"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0,
		"cost": map[string]any{"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
	}
}

func (e *freeEmitter) begin() {
	e.usage = freeEmptyUsage()
	e.out.Push(map[string]any{"type": "start", "partial": e.snapshot()})
}

// closeBlock emits the end event for the text or thinking block currently
// being appended to, if any.
func (e *freeEmitter) closeBlock() {
	if e.open == "" {
		return
	}
	last := e.blocks[len(e.blocks)-1]
	key := e.open
	e.out.Push(map[string]any{
		"type": key + "_end", "contentIndex": len(e.blocks) - 1,
		"content": last[key], "partial": e.snapshot(),
	})
	e.open = ""
}

// appendText starts a new block when the kind changes and emits one delta.
func (e *freeEmitter) appendText(kind, text string) {
	if text == "" {
		return
	}
	if e.open != kind {
		e.closeBlock()
		e.blocks = append(e.blocks, map[string]any{"type": kind, kind: ""})
		e.open = kind
		e.out.Push(map[string]any{
			"type": kind + "_start", "contentIndex": len(e.blocks) - 1,
			"partial": e.snapshot(),
		})
	}
	last := e.blocks[len(e.blocks)-1]
	last[kind] = last[kind].(string) + text
	e.out.Push(map[string]any{
		"type": kind + "_delta", "contentIndex": len(e.blocks) - 1,
		"delta": text, "partial": e.snapshot(),
	})
}

func (e *freeEmitter) beginTool(id, name string) int {
	e.closeBlock()
	if id == "" && e.callID != nil {
		id = e.callID()
	}
	index := len(e.blocks)
	e.blocks = append(e.blocks, map[string]any{
		"type": "toolCall", "id": id, "name": name, "arguments": map[string]any{},
	})
	e.out.Push(map[string]any{"type": "toolcall_start", "contentIndex": index, "partial": e.snapshot()})
	return index
}

func (e *freeEmitter) toolDelta(index int, fragment string) {
	if fragment == "" {
		return
	}
	e.out.Push(map[string]any{
		"type": "toolcall_delta", "contentIndex": index,
		"delta": fragment, "partial": e.snapshot(),
	})
}

func (e *freeEmitter) endTool(index int, args map[string]any) {
	if args == nil {
		args = map[string]any{}
	}
	block := e.blocks[index]
	block["arguments"] = copyJSONMap(args)
	e.out.Push(map[string]any{
		"type": "toolcall_end", "contentIndex": index,
		"toolCall": copyJSONMap(block), "partial": e.snapshot(),
	})
}

func (e *freeEmitter) hasTools() bool {
	for _, block := range e.blocks {
		if block["type"] == "toolCall" {
			return true
		}
	}
	return false
}

func (e *freeEmitter) complete(reason string) {
	if reason == "" {
		reason = "stop"
	}
	if reason == "stop" && e.hasTools() {
		reason = "toolUse"
	}
	e.closeBlock()
	e.stop = reason
	e.out.Push(map[string]any{"type": "done", "reason": reason, "message": e.snapshot()})
}

func (e *freeEmitter) fail(err error, aborted bool) {
	reason, text := "error", err.Error()
	if aborted {
		reason, text = "aborted", "Request was aborted"
	}
	e.closeBlock()
	e.stop = reason
	message := e.snapshot()
	message["errorMessage"] = text
	e.out.Push(map[string]any{"type": "error", "reason": reason, "error": message})
}

func (e *freeEmitter) noteEvent(data string) error {
	if e.observe == nil {
		return nil
	}
	var value any
	if err := json.Unmarshal([]byte(data), &value); err != nil {
		return err
	}
	return e.observe(value, e.info)
}

// setUsage records token counts. A total is derived when the server omits it.
func setFreeUsage(usage map[string]any, input, output, cached, reasoning, total int) {
	if total <= 0 {
		total = input + output + cached
	}
	usage["input"] = max(0, input-cached)
	usage["output"] = output
	usage["cacheRead"] = cached
	usage["cacheWrite"] = 0
	usage["totalTokens"] = total
	if reasoning > 0 {
		usage["reasoning"] = reasoning
	}
}

// readSSE calls fn for every data line of an SSE body. io.EOF means the body
// ended (including an explicit [DONE] marker).
func readSSE(body io.Reader, fn func(string) error) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64<<10), 4<<20)
	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r")
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			return io.EOF
		}
		if err := fn(data); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return io.EOF
}

// postStream sends one request, retrying transient failures. The timeout
// applies to the response headers only; the body can stream afterwards.
func (p *freeProvider) postStream(ctx context.Context, endpoint string, body []byte, opts sdk.ProviderStreamOptions) (*http.Response, context.CancelFunc, string, error) {
	retries := 0
	if n, ok := freeNumber(opts.Values["maxRetries"]); ok && n > 0 {
		retries = int(n)
	}
	maxDelay := 60 * time.Second
	if n, ok := freeNumber(opts.Values["maxRetryDelayMs"]); ok && n >= 0 {
		maxDelay = time.Duration(n) * time.Millisecond
	}
	var note string
	for attempt := 0; ; attempt++ {
		resp, cancel, err := freePost(ctx, p.catalog.client, endpoint, body, opts)
		var delay time.Duration
		switch {
		case err != nil:
			if ctx.Err() != nil || attempt >= retries {
				return nil, nil, "", err
			}
			delay = freeBackoff(attempt)
		case freeRetryable(resp.StatusCode) && attempt < retries:
			delay = freeRetryAfter(resp.Header, time.Now())
			if delay < 0 {
				delay = freeBackoff(attempt)
			}
			if maxDelay > 0 && delay > maxDelay {
				note = fmt.Sprintf(" (retry requested after %ds, over maxRetryDelayMs %d)", int(delay.Seconds()), maxDelay.Milliseconds())
				return resp, cancel, note, nil
			}
			_ = resp.Body.Close()
			cancel()
		default:
			return resp, cancel, note, nil
		}
		timer := time.NewTimer(delay)
		select {
		case <-timer.C:
		case <-ctx.Done():
			timer.Stop()
			return nil, nil, "", ctx.Err()
		}
	}
}

func freePost(ctx context.Context, client *http.Client, endpoint string, body []byte, opts sdk.ProviderStreamOptions) (*http.Response, context.CancelFunc, error) {
	requestCtx, cancel := context.WithCancel(ctx)
	req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		cancel()
		return nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	for name, value := range freeHeaders() {
		req.Header.Set(name, value)
	}
	applyFreeHeaders(req.Header, opts.Values)

	var timedOut atomic.Bool
	var timer *time.Timer
	ms, _ := freeNumber(opts.Values["timeoutMs"])
	if ms > 0 {
		timer = time.AfterFunc(time.Duration(ms)*time.Millisecond, func() {
			timedOut.Store(true)
			cancel()
		})
	}
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if timer != nil {
		timer.Stop()
	}
	if err != nil {
		cancel()
		switch {
		case timedOut.Load():
			return nil, nil, fmt.Errorf("OpenCode Zen timed out after %d ms", int(ms))
		case freeDialError(err):
			return nil, nil, fmt.Errorf("OpenCode Zen is not reachable at %s: %w", endpoint, err)
		case requestCtx.Err() == nil:
			return nil, nil, fmt.Errorf("OpenCode Zen request failed: %w", err)
		default:
			return nil, nil, err
		}
	}
	return resp, cancel, nil
}

// applyFreeHeaders lets callers override or remove the default request headers.
// A nil value removes a header, as in Pi's request options.
func applyFreeHeaders(header http.Header, values map[string]any) {
	switch typed := values["headers"].(type) {
	case map[string]any:
		for name, value := range typed {
			if value == nil {
				header.Del(name)
			} else if text, ok := value.(string); ok {
				header.Set(name, text)
			}
		}
	case map[string]string:
		for name, value := range typed {
			header.Set(name, value)
		}
	case map[string]*string:
		for name, value := range typed {
			if value == nil {
				header.Del(name)
			} else {
				header.Set(name, *value)
			}
		}
	}
}

func freeRetryable(status int) bool {
	return status == http.StatusRequestTimeout || status == http.StatusConflict ||
		status == http.StatusTooManyRequests || status >= 500
}

func freeBackoff(attempt int) time.Duration {
	return 500 * time.Millisecond << min(attempt, 4)
}

func freeRetryAfter(header http.Header, now time.Time) time.Duration {
	if ms, err := strconv.ParseFloat(header.Get("Retry-After-Ms"), 64); err == nil && ms >= 0 {
		return time.Duration(ms * float64(time.Millisecond))
	}
	value := strings.TrimSpace(header.Get("Retry-After"))
	if secs, err := strconv.ParseFloat(value, 64); err == nil && secs >= 0 {
		return time.Duration(secs * float64(time.Second))
	}
	if when, err := http.ParseTime(value); err == nil {
		return max(when.Sub(now), 0)
	}
	return -1
}

func freeDialError(err error) bool {
	var op *net.OpError
	return errors.As(err, &op)
}

func freeNumber(value any) (float64, bool) {
	switch n := value.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		value, err := n.Float64()
		return value, err == nil
	default:
		return 0, false
	}
}

func freeString(value any) string {
	text, _ := value.(string)
	return text
}

func freeHTTPError(resp *http.Response, note string) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	text := strings.TrimSpace(string(body))
	if text == "" {
		text = resp.Status
	}
	return fmt.Errorf("Zen HTTP %d: %s%s", resp.StatusCode, text, note)
}

func freeObserveResponse(opts sdk.ProviderStreamOptions, resp *http.Response, model map[string]any) error {
	if opts.OnResponse == nil {
		return nil
	}
	headers := make(map[string]string, len(resp.Header))
	for name, values := range resp.Header {
		headers[strings.ToLower(name)] = strings.Join(values, ", ")
	}
	return opts.OnResponse(map[string]any{"status": resp.StatusCode, "headers": headers}, model)
}

func freePayload(opts sdk.ProviderStreamOptions, payload any, model map[string]any) (any, error) {
	if opts.OnPayload == nil {
		return payload, nil
	}
	replacement, err := opts.OnPayload(payload, model)
	if err != nil {
		return nil, err
	}
	if replacement != nil {
		return replacement, nil
	}
	return payload, nil
}

// freeJSON encodes a request through the host's payload hook.
func freeJSON(payload any) ([]byte, error) {
	return json.Marshal(payload)
}

// freeToolArguments decodes the JSON object carried by a streamed tool call.
func freeToolArguments(raw string) (map[string]any, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return map[string]any{}, nil
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return nil, fmt.Errorf("OpenCode Zen sent tool arguments that are not a JSON object: %.120q", raw)
	}
	if args == nil {
		args = map[string]any{}
	}
	return args, nil
}

func freeFirst(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
