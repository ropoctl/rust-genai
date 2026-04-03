//! Replay integration tests for the OpenAI Responses API adapter.
//!
//! These tests use pre-recorded cassettes from `tests/data/yakbak/openai_resp/`
//! and assert that content and tool calls flow through correctly.

mod support;

use genai::chat::*;
use serde_json::json;
use support::yakbak::replay_client;
use support::{TestResult, extract_stream_end};

#[tokio::test]
async fn test_yakbak_openai_resp_reasoning_stream() -> TestResult<()> {
	let (client, _server) = replay_client("openai_resp", "reasoning_stream").await?;

	let chat_req = ChatRequest::new(vec![
		ChatMessage::system("Think carefully, then answer concisely in at most two sentences."),
		ChatMessage::user(
			"A fair coin is tossed repeatedly until either HHT or THH appears. Which pattern is more likely to appear first, and what is the probability?",
		),
	]);
	let options = ChatOptions::default()
		.with_reasoning_effort(ReasoningEffort::Medium)
		.with_capture_content(true)
		.with_capture_reasoning_content(true)
		.with_capture_usage(true);

	let stream_res = client
		.exec_chat_stream("openai_resp::gpt-5.4-mini", chat_req, Some(&options))
		.await?;
	let extract = extract_stream_end(stream_res.stream).await?;

	// Exact text content
	assert_eq!(
		extract.content.as_deref(),
		Some("**THH** is more likely to appear first. In fact, \\(P(\\text{HHT first}) = \\tfrac14\\) and \\(P(\\text{THH first}) = \\tfrac34\\)."),
		"Text should match recorded response exactly"
	);

	let reasoning = extract.reasoning_content.ok_or("Should have streamed reasoning summary content")?;
	assert!(
		reasoning.contains("**Considering toss sequences**"),
		"Should include the first streamed reasoning summary section"
	);
	assert!(
		reasoning.contains("**Cross-checking Penney's game**"),
		"Should include the later streamed reasoning summary section"
	);

	let usage = extract.stream_end.captured_usage.as_ref().ok_or("Should have usage")?;
	assert_eq!(usage.prompt_tokens, Some(54));
	assert_eq!(usage.completion_tokens, Some(1262));
	assert_eq!(usage.total_tokens, Some(1316));

	let thought_sigs = extract
		.stream_end
		.captured_thought_signatures()
		.ok_or("Should have thought signatures (encrypted reasoning content)")?;
	assert_eq!(thought_sigs.len(), 1, "Should have exactly one thought signature");
	assert!(
		thought_sigs[0].len() > 100,
		"Encrypted content should be substantial, got {} bytes",
		thought_sigs[0].len()
	);

	Ok(())
}

#[tokio::test]
async fn test_yakbak_openai_resp_stream_tools() -> TestResult<()> {
	let (client, _server) = replay_client("openai_resp", "reasoning_stream_tools").await?;

	let chat_req = ChatRequest::new(vec![
		ChatMessage::system("You are a helpful assistant. Use tools when needed."),
		ChatMessage::user("What is the temperature in C and weather, in Paris, France"),
	])
	.append_tool(Tool::new("get_weather").with_schema(json!({
		"type": "object",
		"properties": {
			"city": { "type": "string", "description": "The city name" },
			"country": { "type": "string", "description": "The most likely country of this city name" },
			"unit": { "type": "string", "enum": ["C", "F"], "description": "Temperature unit" }
		},
		"required": ["city", "country", "unit"],
	})));

	let options = ChatOptions::default()
		.with_reasoning_effort(ReasoningEffort::Low)
		.with_capture_content(true)
		.with_capture_reasoning_content(true)
		.with_capture_tool_calls(true)
		.with_capture_usage(true);

	let stream_res = client
		.exec_chat_stream("openai_resp::gpt-5.4-mini", chat_req, Some(&options))
		.await?;
	let extract = extract_stream_end(stream_res.stream).await?;

	// Exactly one tool call
	let tool_calls = extract.stream_end.captured_tool_calls().ok_or("Should have tool calls")?;
	assert_eq!(tool_calls.len(), 1);

	// Exact tool call details
	let tc = &tool_calls[0];
	assert_eq!(tc.fn_name, "get_weather");
	assert_eq!(
		tc.fn_arguments,
		json!({"city": "Paris", "country": "France", "unit": "C"})
	);
	assert!(!tc.call_id.is_empty());

	// Exact usage
	let usage = extract.stream_end.captured_usage.as_ref().ok_or("Should have usage")?;
	assert_eq!(usage.prompt_tokens, Some(106));
	assert_eq!(usage.completion_tokens, Some(45));
	assert_eq!(usage.total_tokens, Some(151));

	Ok(())
}

/// Demonstrates and verifies the UTF-8 chunking fix in WebStream::poll_next().
///
/// When a multi-byte UTF-8 character (e.g. Japanese 日 = 3 bytes) straddles an HTTP
/// chunk boundary, the old code's `String::from_utf8()` would fail. The fix buffers
/// incomplete trailing bytes across chunks.
///
/// This tape has Japanese text positioned at byte offset 8191 (8KB boundary).
#[tokio::test]
async fn test_yakbak_openai_resp_utf8_chunking_bug() -> TestResult<()> {
	let (client, _server) = replay_client("openai_resp", "utf8_chunking_bug").await?;

	let chat_req = ChatRequest::new(vec![ChatMessage::user("Say something in Japanese.")]);
	let options = ChatOptions::default().with_capture_content(true).with_capture_usage(true);

	let stream_res = client
		.exec_chat_stream("openai_resp::o3-mini", chat_req, Some(&options))
		.await?;
	let extract = extract_stream_end(stream_res.stream).await?;

	// The content should contain Japanese characters intact
	let content = extract.content.ok_or("Should have streamed content")?;
	assert!(
		content.contains("日本語のテスト"),
		"Content should contain Japanese text, got: {}...",
		&content[content.len().saturating_sub(100)..]
	);

	// Usage should be captured
	let usage = extract.stream_end.captured_usage.as_ref().ok_or("Should have usage")?;
	assert_eq!(usage.prompt_tokens, Some(10));
	assert_eq!(usage.completion_tokens, Some(50));
	assert_eq!(usage.total_tokens, Some(60));

	Ok(())
}
