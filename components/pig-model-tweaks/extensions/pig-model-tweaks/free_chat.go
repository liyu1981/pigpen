package pigmodeltweaks

// Chat Completions request construction and stream parsing for the keyless
// OpenCode Zen provider. Independent MIT implementation.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

func (p *freeProvider) chatTurn(ctx context.Context, out *sdk.ModelEventStream, model, transcript map[string]any, opts sdk.ProviderStreamOptions) {
	e := newFreeEmitter(out, model, opts)
	if p.newCallID != nil {
		e.callID = p.newCallID
	}
	e.begin()
	request, err := freeChatRequest(model, transcript, opts)
	var payload any = request
	if err == nil {
		payload, err = freePayload(opts, payload, model)
	}
	if err != nil {
		e.fail(err, ctx.Err() != nil)
		return
	}
	body, err := freeJSON(payload)
	if err != nil {
		e.fail(fmt.Errorf("cannot encode OpenCode Zen request: %w", err), false)
		return
	}
	resp, cancel, note, err := p.postStream(ctx, freeRequestBase(model)+"/chat/completions", body, opts)
	if err != nil {
		e.fail(err, ctx.Err() != nil)
		return
	}
	defer cancel()
	defer func() { _ = resp.Body.Close() }()
	if err := freeObserveResponse(opts, resp, model); err != nil {
		e.fail(err, ctx.Err() != nil)
		return
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		e.fail(freeHTTPError(resp, note), false)
		return
	}
	freeReadChat(ctx, resp.Body, e)
}

// freeChatRequest translates Pi's transcript into Zen's chat-completions body.
func freeChatRequest(model, transcript map[string]any, opts sdk.ProviderStreamOptions) (map[string]any, error) {
	system, tools := freeSystemParts(transcript)
	messages := make([]any, 0, len(system)+1)
	if system != "" {
		messages = append(messages, map[string]any{"role": "system", "content": system})
	}
	images := freeModelImages(model)
	items, _ := transcript["messages"].([]any)
	for index := 0; index < len(items); index++ {
		message, ok := items[index].(map[string]any)
		if !ok || message == nil {
			continue
		}
		switch freeString(message["role"]) {
		case "system":
			// Already folded into the leading system message.
		case "user":
			if m := freeUserMessage(message, images); m != nil {
				messages = append(messages, m)
			}
		case "assistant":
			if m := freeAssistantMessage(message, freeString(model["reasoning"]) == "true" || model["reasoning"] == true); m != nil {
				messages = append(messages, m)
			}
		case "toolResult":
			// Consecutive tool results become consecutive tool messages, and any
			// images they carry are replayed as one following user message.
			var attached []map[string]any
			for {
				text, pictures := freeContentParts(message["content"], images)
				if text == "" {
					if len(pictures) > 0 || freeHasImagePart(message["content"]) {
						text = "(see attached image)"
					} else {
						text = "(no tool output)"
					}
				}
				messages = append(messages, map[string]any{
					"role": "tool", "tool_call_id": message["toolCallId"], "content": text,
				})
				attached = append(attached, pictures...)
				if index+1 >= len(items) {
					break
				}
				next, ok := items[index+1].(map[string]any)
				if !ok || next == nil || freeString(next["role"]) != "toolResult" {
					break
				}
				message = next
				index++
			}
			if len(attached) > 0 {
				parts := append([]any{map[string]any{"type": "text", "text": "Attached image(s) from tool result:"}}, freeAnyMaps(attached)...)
				messages = append(messages, map[string]any{"role": "user", "content": parts})
			}
		}
	}

	request := map[string]any{
		"model":          freeModelID(model),
		"messages":       messages,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}
	if len(tools) > 0 {
		request["tools"] = freeChatTools(tools)
	}
	if choice, exists := opts.Values["toolChoice"]; exists && choice != nil {
		request["tool_choice"] = choice
	}
	if temp, ok := freeNumber(opts.Values["temperature"]); ok {
		request["temperature"] = temp
	}
	if maxTokens, ok := freeNumber(opts.Values["maxTokens"]); ok && maxTokens > 0 {
		request["max_tokens"] = int(maxTokens)
	}
	if reasoning, ok := model["reasoning"].(bool); ok && reasoning {
		if effort, ok := freeEffort(model, opts.Values); ok {
			request["reasoning_effort"] = effort
		}
	}
	// Sampling parameters are applied last and may override named fields.
	if sampling, ok := opts.Values["samplingParams"].(map[string]any); ok {
		for key, value := range sampling {
			request[key] = value
		}
	}
	return request, nil
}

func freeModelID(model map[string]any) string {
	return strings.TrimPrefix(freeString(model["id"]), "opencode/")
}

func freeModelImages(model map[string]any) bool {
	switch input := model["input"].(type) {
	case []any:
		for _, value := range input {
			if value == "image" {
				return true
			}
		}
	case []string:
		for _, value := range input {
			if value == "image" {
				return true
			}
		}
	}
	return false
}

func freeUserMessage(message map[string]any, images bool) map[string]any {
	if text, ok := message["content"].(string); ok {
		return map[string]any{"role": "user", "content": text}
	}
	text, pictures := freeContentParts(message["content"], images)
	if text == "" && len(pictures) == 0 {
		return nil
	}
	if len(pictures) == 0 {
		return map[string]any{"role": "user", "content": text}
	}
	parts := make([]any, 0, len(pictures)+1)
	if text != "" {
		parts = append(parts, map[string]any{"type": "text", "text": text})
	}
	parts = append(parts, freeAnyMaps(pictures)...)
	return map[string]any{"role": "user", "content": parts}
}

func freeAssistantMessage(message map[string]any, reasoning bool) map[string]any {
	var text, thinking []string
	var calls []any
	blocks, _ := message["content"].([]any)
	for _, raw := range blocks {
		block, ok := raw.(map[string]any)
		if !ok || block == nil {
			continue
		}
		switch freeString(block["type"]) {
		case "text":
			if value := freeString(block["text"]); value != "" {
				text = append(text, value)
			}
		case "thinking":
			if value := freeString(block["thinking"]); value != "" {
				thinking = append(thinking, value)
			}
		case "toolCall":
			arguments, _ := block["arguments"].(map[string]any)
			if arguments == nil {
				arguments = map[string]any{}
			}
			encoded, _ := json.Marshal(arguments)
			calls = append(calls, map[string]any{
				"id": block["id"], "type": "function",
				"function": map[string]any{"name": block["name"], "arguments": string(encoded)},
			})
		}
	}
	content := strings.Join(text, "")
	if content == "" && len(calls) == 0 {
		return nil
	}
	out := map[string]any{"role": "assistant", "content": content}
	if len(calls) > 0 {
		out["tool_calls"] = calls
	}
	if reasoning {
		out["reasoning_content"] = strings.Join(thinking, "\n")
	}
	return out
}

// freeContentParts splits text and image blocks from a message's content.
func freeContentParts(content any, images bool) (string, []map[string]any) {
	if text, ok := content.(string); ok {
		return text, nil
	}
	var text []string
	var pictureParts []map[string]any
	blocks, _ := content.([]any)
	for _, raw := range blocks {
		block, ok := raw.(map[string]any)
		if !ok || block == nil {
			continue
		}
		switch freeString(block["type"]) {
		case "text":
			if value := freeString(block["text"]); value != "" {
				text = append(text, value)
			}
		case "image":
			if images {
				if data, mime := freeString(block["data"]), freeString(block["mimeType"]); data != "" {
					pictureParts = append(pictureParts, map[string]any{
						"type":      "image_url",
						"image_url": map[string]any{"url": "data:" + mime + ";base64," + data},
					})
				}
			}
		}
	}
	return strings.Join(text, "\n"), pictureParts
}

func freeHasImagePart(content any) bool {
	blocks, _ := content.([]any)
	for _, raw := range blocks {
		if block, ok := raw.(map[string]any); ok && block != nil && freeString(block["type"]) == "image" {
			return true
		}
	}
	return false
}

func freeAnyMaps(values []map[string]any) []any {
	out := make([]any, len(values))
	for i, value := range values {
		out[i] = value
	}
	return out
}

func freeChatTools(tools []map[string]any) []any {
	out := make([]any, 0, len(tools))
	for _, tool := range tools {
		parameters, _ := tool["parameters"].(map[string]any)
		if parameters == nil {
			parameters = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": tool["name"], "description": tool["description"], "parameters": parameters,
			},
		})
	}
	return out
}

// freeSystemParts collapses system messages into the current system prompt and
// the tools currently offered, in declaration order.
func freeSystemParts(transcript map[string]any) (string, []map[string]any) {
	var prompt []string
	var sectionOrder []string
	sections := map[string]string{}
	var toolOrder []string
	tools := map[string]map[string]any{}
	items, _ := transcript["messages"].([]any)
	for _, raw := range items {
		message, ok := raw.(map[string]any)
		if !ok || message == nil || freeString(message["role"]) != "system" {
			continue
		}
		if text := freeContentText(message["content"]); text != "" {
			prompt = append(prompt, text)
		}
		freeCollectSections(message["sections"], &sectionOrder, sections)
		for _, removed := range freeMaps(message["toolsRemoved"]) {
			name := freeString(removed["name"])
			if name == "" {
				continue
			}
			delete(tools, name)
			toolOrder = freeWithout(toolOrder, name)
		}
		for _, added := range freeMaps(message["toolsAdded"]) {
			name := freeString(added["name"])
			if name == "" {
				continue
			}
			if _, exists := tools[name]; !exists {
				toolOrder = append(toolOrder, name)
			}
			tools[name] = added
		}
	}
	parts := append([]string{}, prompt...)
	for _, name := range sectionOrder {
		if text := sections[name]; text != "" {
			parts = append(parts, text)
		}
	}
	out := make([]map[string]any, 0, len(toolOrder))
	for _, name := range toolOrder {
		if tool := tools[name]; tool != nil {
			out = append(out, tool)
		}
	}
	return strings.Join(parts, "\n\n"), out
}

func freeCollectSections(raw any, order *[]string, sections map[string]string) {
	set := func(name, value string) {
		if name == "" {
			return
		}
		found := false
		for _, existing := range *order {
			if existing == name {
				found = true
				break
			}
		}
		if !found {
			*order = append(*order, name)
		}
		sections[name] = value
	}
	switch sectionsIn := raw.(type) {
	case map[string]string:
		names := make([]string, 0, len(sectionsIn))
		for name := range sectionsIn {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			set(name, sectionsIn[name])
		}
	case map[string]any:
		names := make([]string, 0, len(sectionsIn))
		for name := range sectionsIn {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			if value, ok := sectionsIn[name].(string); ok {
				set(name, value)
			} else {
				delete(sections, name)
			}
		}
	case []any:
		for _, entry := range sectionsIn {
			section, ok := entry.(map[string]any)
			if !ok || section == nil {
				continue
			}
			if value, ok := section["value"].(string); ok {
				set(freeString(section["name"]), value)
			} else {
				delete(sections, freeString(section["name"]))
			}
		}
	}
}

func freeContentText(content any) string {
	if text, ok := content.(string); ok {
		return text
	}
	var parts []string
	blocks, _ := content.([]any)
	for _, raw := range blocks {
		if block, ok := raw.(map[string]any); ok && block != nil && freeString(block["type"]) == "text" {
			if text := freeString(block["text"]); text != "" {
				parts = append(parts, text)
			}
		}
	}
	return strings.Join(parts, "\n")
}

func freeMaps(value any) []map[string]any {
	items, _ := value.([]any)
	out := make([]map[string]any, 0, len(items))
	for _, item := range items {
		if m, ok := item.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

func freeWithout(names []string, name string) []string {
	out := names[:0]
	for _, existing := range names {
		if existing != name {
			out = append(out, existing)
		}
	}
	return out
}

// freeEffort maps the requested thinking level onto the model's effort values.
func freeEffort(model map[string]any, values map[string]any) (string, bool) {
	level := ""
	for _, key := range []string{"reasoningEffort", "reasoning", "effort"} {
		if value := freeString(values[key]); value != "" {
			level = value
			break
		}
	}
	if enabled, exists := values["thinkingEnabled"].(bool); exists {
		if !enabled {
			level = "off"
		} else if level == "" {
			level = "high"
		}
	}
	if levels, ok := model["thinkingLevelMap"].(map[string]any); ok && levels != nil {
		if mapped, exists := levels[level]; exists {
			if mapped == nil {
				// A null entry hides the level from the picker. "off" truly
				// disables effort; any other named level is still sent as-is.
				if level == "off" {
					return "", false
				}
				return level, true
			}
			if text, ok := mapped.(string); ok {
				return text, true
			}
		}
	}
	if level == "off" {
		return "", false
	}
	return level, true
}

// chatChunk is one SSE data frame from /chat/completions.
type chatChunk struct {
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
	Choices []struct {
		Delta struct {
			Content   string  `json:"content"`
			Reasoning *string `json:"reasoning_content"`
			Thinking  string  `json:"reasoning"`
			Calls     []struct {
				Index    *int   `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		Finish *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		Prompt        int `json:"prompt_tokens"`
		Complete      int `json:"completion_tokens"`
		Total         int `json:"total_tokens"`
		PromptDetails *struct {
			Cached int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
		CompleteDetails *struct {
			Reasoning int `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
}

type chatToolDraft struct {
	block     int
	id        string
	name      string
	arguments strings.Builder
	done      bool
}

// freeReadChat consumes the SSE stream, emitting events as they arrive.
func freeReadChat(ctx context.Context, body io.Reader, e *freeEmitter) {
	drafts := map[int]*chatToolDraft{}
	var order []int
	finish := ""
	sawContent := false

	draftFor := func(index int, id, name string) *chatToolDraft {
		draft := drafts[index]
		if draft == nil {
			draft = &chatToolDraft{block: e.beginTool(id, name), id: id, name: name}
			drafts[index] = draft
			order = append(order, index)
		} else {
			if draft.id == "" && id != "" {
				draft.id = id
				e.blocks[draft.block]["id"] = id
			}
			if draft.name == "" && name != "" {
				draft.name = name
				e.blocks[draft.block]["name"] = name
			}
		}
		return draft
	}
	finishTools := func() error {
		for _, index := range order {
			draft := drafts[index]
			if draft.done {
				continue
			}
			args, err := freeToolArguments(draft.arguments.String())
			if err != nil {
				return err
			}
			e.endTool(draft.block, args)
			draft.done = true
		}
		return nil
	}

	err := readSSE(body, func(data string) error {
		if err := e.noteEvent(data); err != nil {
			return err
		}
		var chunk chatChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("OpenCode Zen sent a non-JSON line: %.80q", data)
		}
		if chunk.Error != nil {
			return errors.New(freeFirst(chunk.Error.Message, chunk.Error.Type, chunk.Error.Code, "OpenCode Zen returned an error"))
		}
		if usage := chunk.Usage; usage != nil {
			cached, reasoning := 0, 0
			if usage.PromptDetails != nil {
				cached = usage.PromptDetails.Cached
			}
			if usage.CompleteDetails != nil {
				reasoning = usage.CompleteDetails.Reasoning
			}
			setFreeUsage(e.usage, usage.Prompt, usage.Complete, cached, reasoning, usage.Total)
		}
		for _, choice := range chunk.Choices {
			delta := choice.Delta
			if delta.Reasoning != nil {
				e.appendText("thinking", *delta.Reasoning)
				if *delta.Reasoning != "" {
					sawContent = true
				}
			} else if delta.Thinking != "" {
				e.appendText("thinking", delta.Thinking)
				sawContent = true
			}
			if delta.Content != "" {
				e.appendText("text", delta.Content)
				sawContent = true
			}
			for i, call := range delta.Calls {
				index := i
				if call.Index != nil {
					index = *call.Index
				}
				draft := draftFor(index, call.ID, call.Function.Name)
				if call.Function.Arguments != "" {
					draft.arguments.WriteString(call.Function.Arguments)
					e.toolDelta(draft.block, call.Function.Arguments)
					sawContent = true
				}
			}
			if choice.Finish != nil && *choice.Finish != "" {
				finish = *choice.Finish
				if err := finishTools(); err != nil {
					return err
				}
				e.closeBlock()
			}
		}
		return nil
	})
	if err != nil && !errors.Is(err, io.EOF) {
		e.fail(err, ctx.Err() != nil)
		return
	}
	if finish == "" && !sawContent {
		e.fail(errors.New("OpenCode Zen stream ended before it finished"), ctx.Err() != nil)
		return
	}
	if err := finishTools(); err != nil {
		e.fail(err, ctx.Err() != nil)
		return
	}
	reason := "stop"
	switch finish {
	case "length":
		reason = "length"
	case "tool_calls", "function_call":
		reason = "toolUse"
	}
	e.complete(reason)
}
