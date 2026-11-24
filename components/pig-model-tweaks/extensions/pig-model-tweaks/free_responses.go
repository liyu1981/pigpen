package pigmodeltweaks

// Responses API request construction and stream parsing for the keyless
// OpenCode Zen provider. Independent MIT implementation.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

func (p *freeProvider) responseTurn(ctx context.Context, out *sdk.ModelEventStream, model, transcript map[string]any, opts sdk.ProviderStreamOptions) {
	e := newFreeEmitter(out, model, opts)
	if p.newCallID != nil {
		e.callID = p.newCallID
	}
	e.begin()
	request, err := freeResponseRequest(model, transcript, opts)
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
	resp, cancel, note, err := p.postStream(ctx, freeRequestBase(model)+"/responses", body, opts)
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
	freeReadResponses(ctx, resp.Body, e)
}

func freeResponseRequest(model, transcript map[string]any, opts sdk.ProviderStreamOptions) (map[string]any, error) {
	system, tools := freeSystemParts(transcript)
	input := make([]any, 0)
	if system != "" {
		input = append(input, map[string]any{"role": "system", "content": system})
	}
	images := freeModelImages(model)
	items, _ := transcript["messages"].([]any)
	messageIndex := 0
	for _, raw := range items {
		message, ok := raw.(map[string]any)
		if !ok || message == nil {
			continue
		}
		switch freeString(message["role"]) {
		case "system":
		case "user":
			if parts := freeResponseInput(message["content"], images); len(parts) > 0 {
				input = append(input, map[string]any{"role": "user", "content": parts})
			}
		case "assistant":
			blocks, _ := message["content"].([]any)
			textNumber := 0
			for _, blockRaw := range blocks {
				block, ok := blockRaw.(map[string]any)
				if !ok || block == nil {
					continue
				}
				switch freeString(block["type"]) {
				case "text":
					id := fmt.Sprintf("msg_pi_%d", messageIndex)
					if textNumber > 0 {
						id = fmt.Sprintf("msg_pi_%d_%d", messageIndex, textNumber)
					}
					textNumber++
					input = append(input, map[string]any{
						"type": "message", "role": "assistant", "status": "completed", "id": id,
						"content": []any{map[string]any{"type": "output_text", "text": block["text"], "annotations": []any{}}},
					})
				case "toolCall":
					arguments, _ := block["arguments"].(map[string]any)
					if arguments == nil {
						arguments = map[string]any{}
					}
					encoded, _ := json.Marshal(arguments)
					input = append(input, map[string]any{
						"type": "function_call", "call_id": block["id"], "name": block["name"],
						"arguments": string(encoded),
					})
				}
			}
		case "toolResult":
			input = append(input, map[string]any{
				"type": "function_call_output", "call_id": message["toolCallId"],
				"output": freeResponseOutput(message["content"], images),
			})
		}
		messageIndex++
	}
	request := map[string]any{
		"model": freeModelID(model), "input": input, "stream": true, "store": false,
	}
	if len(tools) > 0 {
		converted := make([]any, 0, len(tools))
		for _, tool := range tools {
			parameters, _ := tool["parameters"].(map[string]any)
			if parameters == nil {
				parameters = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			converted = append(converted, map[string]any{
				"type": "function", "name": tool["name"],
				"description": tool["description"], "parameters": parameters,
			})
		}
		request["tools"] = converted
	}
	if choice, exists := opts.Values["toolChoice"]; exists && choice != nil {
		request["tool_choice"] = choice
	}
	if temp, ok := freeNumber(opts.Values["temperature"]); ok {
		request["temperature"] = temp
	}
	if maxTokens, ok := freeNumber(opts.Values["maxTokens"]); ok && maxTokens > 0 {
		request["max_output_tokens"] = max(16, int(maxTokens))
	}
	if reasoning, ok := model["reasoning"].(bool); ok && reasoning {
		if effort, ok := freeEffort(model, opts.Values); ok {
			request["reasoning"] = map[string]any{"effort": effort, "summary": "auto"}
			request["include"] = []any{"reasoning.encrypted_content"}
		} else {
			level := freeEffortLevel(opts.Values)
			levels, _ := model["thinkingLevelMap"].(map[string]any)
			if off, exists := levels["off"]; level == "" && (!exists || off != nil) {
				effort := "none"
				if text, ok := off.(string); ok {
					effort = text
				}
				request["reasoning"] = map[string]any{"effort": effort}
			}
		}
	}
	if sampling, ok := opts.Values["samplingParams"].(map[string]any); ok {
		for key, value := range sampling {
			request[key] = value
		}
	}
	return request, nil
}

// freeEffortLevel reads the requested thinking level without consulting the
// model's level map.
func freeEffortLevel(values map[string]any) string {
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
	return level
}

func freeResponseInput(content any, images bool) []any {
	parts := make([]any, 0)
	if text, ok := content.(string); ok {
		return append(parts, map[string]any{"type": "input_text", "text": text})
	}
	blocks, _ := content.([]any)
	for _, raw := range blocks {
		block, ok := raw.(map[string]any)
		if !ok || block == nil {
			continue
		}
		switch freeString(block["type"]) {
		case "text":
			parts = append(parts, map[string]any{"type": "input_text", "text": block["text"]})
		case "image":
			if images {
				if data, mime := freeString(block["data"]), freeString(block["mimeType"]); data != "" {
					parts = append(parts, map[string]any{
						"type": "input_image", "detail": "auto",
						"image_url": "data:" + mime + ";base64," + data,
					})
				}
			}
		}
	}
	return parts
}

func freeResponseOutput(content any, images bool) any {
	var text []string
	var pictures []any
	sawImage := false
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
			sawImage = true
			if images {
				if data, mime := freeString(block["data"]), freeString(block["mimeType"]); data != "" {
					pictures = append(pictures, map[string]any{
						"type": "input_image", "detail": "auto",
						"image_url": "data:" + mime + ";base64," + data,
					})
				}
			}
		}
	}
	joined := strings.Join(text, "\n")
	if len(pictures) == 0 {
		if joined != "" {
			return joined
		}
		if sawImage {
			return "(see attached image)"
		}
		return "(no tool output)"
	}
	parts := make([]any, 0, len(pictures)+1)
	if joined != "" {
		parts = append(parts, map[string]any{"type": "input_text", "text": joined})
	}
	return append(parts, pictures...)
}

type responseEvent struct {
	Type        string          `json:"type"`
	OutputIndex int             `json:"output_index"`
	Response    *responseMeta   `json:"response"`
	Item        json.RawMessage `json:"item"`
	Delta       string          `json:"delta"`
	Arguments   string          `json:"arguments"`
	Error       json.RawMessage `json:"error"`
	Code        string          `json:"code"`
	Message     string          `json:"message"`
}

type responseMeta struct {
	Status string `json:"status"`
	Usage  *struct {
		Input        int `json:"input_tokens"`
		Output       int `json:"output_tokens"`
		Total        int `json:"total_tokens"`
		InputDetails *struct {
			Cached int `json:"cached_tokens"`
		} `json:"input_tokens_details"`
		OutputDetails *struct {
			Reasoning int `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
	Error      *struct{ Code, Message string } `json:"error"`
	Incomplete *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
}

type responseItem struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Content   []struct {
		Type    string `json:"type"`
		Text    string `json:"text"`
		Refusal string `json:"refusal"`
	} `json:"content"`
	Summary []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"summary"`
}

type responseDraft struct {
	kind      string
	block     int
	id        string
	name      string
	args      strings.Builder
	done      bool
	textSeen  bool
	thinkSeen bool
}

func decodeResponseItem(raw json.RawMessage) *responseItem {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var item responseItem
	if err := json.Unmarshal(raw, &item); err != nil {
		return nil
	}
	return &item
}

func responseErrorText(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var value struct {
		Message string `json:"message"`
		Code    string `json:"code"`
		Type    string `json:"type"`
	}
	if json.Unmarshal(raw, &value) != nil {
		return ""
	}
	return freeFirst(value.Message, value.Code, value.Type)
}

// freeReadResponses consumes the Responses stream and emits Pi events.
func freeReadResponses(ctx context.Context, body io.Reader, e *freeEmitter) {
	drafts := map[int]*responseDraft{}
	var order []int
	status, incomplete := "", ""
	terminal := false

	draftFor := func(index int, id, name string) *responseDraft {
		draft := drafts[index]
		if draft != nil {
			if draft.kind != "function_call" {
				return nil
			}
			if draft.id == "" && id != "" {
				draft.id = id
				e.blocks[draft.block]["id"] = id
			}
			if draft.name == "" && name != "" {
				draft.name = name
				e.blocks[draft.block]["name"] = name
			}
			return draft
		}
		draft = &responseDraft{kind: "function_call", block: e.beginTool(id, name), id: id, name: name}
		drafts[index] = draft
		order = append(order, index)
		return draft
	}
	kindFor := func(index int, kind string) *responseDraft {
		draft := drafts[index]
		if draft == nil {
			draft = &responseDraft{kind: kind, block: -1}
			drafts[index] = draft
		}
		return draft
	}
	finishTools := func() error {
		for _, index := range order {
			draft := drafts[index]
			if draft == nil || draft.done {
				continue
			}
			args, err := freeToolArguments(draft.args.String())
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
		var event responseEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			return fmt.Errorf("OpenCode Zen sent a non-JSON line: %.80q", data)
		}
		if message := responseErrorText(event.Error); message != "" {
			return errors.New(message)
		}
		switch event.Type {
		case "response.output_item.added":
			item := decodeResponseItem(event.Item)
			if item == nil {
				return nil
			}
			switch item.Type {
			case "function_call":
				draft := draftFor(event.OutputIndex, freeFirst(item.CallID, item.ID), item.Name)
				if draft != nil && item.Arguments != "" {
					draft.args.WriteString(item.Arguments)
					e.toolDelta(draft.block, item.Arguments)
				}
			case "reasoning":
				kindFor(event.OutputIndex, "reasoning")
			case "message":
				kindFor(event.OutputIndex, "message")
			}
		case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
			draft := kindFor(event.OutputIndex, "reasoning")
			e.appendText("thinking", event.Delta)
			draft.thinkSeen = draft.thinkSeen || event.Delta != ""
		case "response.reasoning_summary_part.done":
			if draft := drafts[event.OutputIndex]; draft != nil && draft.kind == "reasoning" {
				e.appendText("thinking", "\n\n")
			}
		case "response.output_text.delta", "response.refusal.delta":
			draft := kindFor(event.OutputIndex, "message")
			e.appendText("text", event.Delta)
			draft.textSeen = draft.textSeen || event.Delta != ""
		case "response.function_call_arguments.delta":
			if draft := draftFor(event.OutputIndex, "", ""); draft != nil {
				draft.args.WriteString(event.Delta)
				e.toolDelta(draft.block, event.Delta)
			}
		case "response.function_call_arguments.done":
			if draft := draftFor(event.OutputIndex, "", ""); draft != nil {
				appendTail(draft, event.Arguments, e)
			}
		case "response.output_item.done":
			item := decodeResponseItem(event.Item)
			if item == nil {
				return nil
			}
			switch item.Type {
			case "function_call":
				draft := draftFor(event.OutputIndex, freeFirst(item.CallID, item.ID), item.Name)
				if draft != nil && !draft.done {
					appendTail(draft, item.Arguments, e)
					args, err := freeToolArguments(draft.args.String())
					if err != nil {
						return err
					}
					e.endTool(draft.block, args)
					draft.done = true
				}
			case "message":
				draft := drafts[event.OutputIndex]
				var text strings.Builder
				for _, part := range item.Content {
					if part.Type == "output_text" {
						text.WriteString(part.Text)
					} else {
						text.WriteString(part.Refusal)
					}
				}
				if text.Len() > 0 && (draft == nil || !draft.textSeen) {
					e.appendText("text", text.String())
					if draft != nil {
						draft.textSeen = true
					}
				}
			case "reasoning":
				draft := drafts[event.OutputIndex]
				var text strings.Builder
				for _, part := range item.Summary {
					text.WriteString(part.Text)
				}
				for _, part := range item.Content {
					text.WriteString(part.Text)
				}
				if text.Len() > 0 && (draft == nil || !draft.thinkSeen) {
					e.appendText("thinking", text.String())
					if draft != nil {
						draft.thinkSeen = true
					}
				}
			}
		case "response.completed", "response.incomplete":
			if event.Response == nil {
				return errors.New("OpenCode Zen sent a terminal response without metadata")
			}
			if usage := event.Response.Usage; usage != nil {
				cached, reasoning := 0, 0
				if usage.InputDetails != nil {
					cached = usage.InputDetails.Cached
				}
				if usage.OutputDetails != nil {
					reasoning = usage.OutputDetails.Reasoning
				}
				setFreeUsage(e.usage, usage.Input, usage.Output, cached, reasoning, usage.Total)
			}
			status = event.Response.Status
			if event.Response.Incomplete != nil {
				incomplete = event.Response.Incomplete.Reason
			}
			terminal = true
		case "response.done":
			// A trailing bookkeeping event; the terminal one above matters.
		case "response.failed":
			message := "The response failed without details"
			if event.Response != nil && event.Response.Error != nil {
				message = freeFirst(event.Response.Error.Message, event.Response.Error.Code, message)
			}
			return errors.New(message)
		case "error":
			return errors.New(freeFirst(event.Message, event.Code, "OpenCode Zen returned an error"))
		}
		return nil
	})
	if err != nil && !errors.Is(err, io.EOF) {
		e.fail(err, ctx.Err() != nil)
		return
	}
	if !terminal {
		e.fail(errors.New("OpenCode Zen Responses stream ended before a terminal response"), ctx.Err() != nil)
		return
	}
	if err := finishTools(); err != nil {
		e.fail(err, ctx.Err() != nil)
		return
	}
	reason := "stop"
	if status == "incomplete" && strings.Contains(strings.ToLower(incomplete), "max_output_tokens") {
		reason = "length"
	}
	e.complete(reason)
}

// appendTail replaces accumulated arguments with a final value, emitting only
// the portion not already streamed.
func appendTail(draft *responseDraft, final string, e *freeEmitter) {
	if final == "" {
		return
	}
	current := draft.args.String()
	if tail, ok := strings.CutPrefix(final, current); ok {
		e.toolDelta(draft.block, tail)
	} else {
		e.toolDelta(draft.block, final)
	}
	draft.args.Reset()
	draft.args.WriteString(final)
}
