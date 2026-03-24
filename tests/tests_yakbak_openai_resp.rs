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
		ChatMessage::system("Answer in one sentence."),
		ChatMessage::user("Why is the sky blue?"),
	]);
	let options = ChatOptions::default()
		.with_reasoning_effort(ReasoningEffort::Low)
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
		Some("The sky appears blue because molecules in Earth\u{2019}s atmosphere scatter shorter-wavelength sunlight, like blue and violet, more strongly than longer wavelengths, and our eyes perceive the scattered light as blue."),
		"Text should match recorded response exactly"
	);

	// Exact usage
	let usage = extract.stream_end.captured_usage.as_ref().ok_or("Should have usage")?;
	assert_eq!(usage.prompt_tokens, Some(21));
	assert_eq!(usage.completion_tokens, Some(51));
	assert_eq!(usage.total_tokens, Some(72));

	// Encrypted reasoning content captured as thought_signatures
	let sigs = extract.stream_end.captured_thought_signatures().ok_or("Should have thought_signatures (encrypted_content)")?;
	assert_eq!(sigs.len(), 1, "Should have exactly one encrypted_content blob");
	assert_eq!(sigs[0], "gAAAAABpwxu6LuEvELP__ILDLadfJctmwel4WISviDlxOMX_n_k7RFNRmD9yqmrOsGIfPpgAh76NlhIPAOt1_7-vJjzIhcaTc_x3MgKoiWHx7BETl8UoUQsG07u194CMqU8vYQUWVHw0z9NdRnSBMU8isJc2Wt1myGHoIQWZiL3ECjQQ1t01gtdWZ7-1dYzgte5gGw8mOi1KZBlKP8FSvDINZDQNDwPlawGToT5ehJSYBuSnrpV2QBnpRY6A4sNG3wUd6T91hjDmQNClxWlyGThHN3aFcUq-A0q7HR2WBlbrKKcIxgsTV-_3VRYrg0uABbvGwX0eBvFmS2rkfDyvl9iRbpV5S0LBiGZks-B4v7nuT1AYmIh5uOTctUG5xRvXjD4kXVVp6jE1m1cxPF69QlMaxk0VidBXmBiXbVd09bmLtck8H40SD0XGXQ1rxj-PM_XzxNGBRKXlak2UQhA5S90JoOjBfQD1mkrDv3NkLz8llw2-53kVl8zc_kgyw8Rfk1RlKYVdetp4sRARUsg08yoqcqrtNgTk9oWgLp2Lj53Gq1356AbtyLdPUCQouI3esElwxjhKWircojP_M2vFkBXNE4bHtNRIr7_vaZeFcHF1O8SE0s2xzcTajPfPApjapfJ4ppNCHNxgMf1vhzMzPEp-1_8-zmt1bG_ima5jwe-r2SSedKjtXW9Q-T9cq68cP3ORtu4yA3AjW3J3K0DKEcEbmCK25Yscsi8uQhOuHnFTPP21x-TfB4i59r0PEtLfxhPlyQveGSjx8j2tPhzedIjjd9qfAYaPiKjNyBSGF39vxUtqYR29uSPV4A4xM57bdBmBA6l5QYYW");

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

	// Reasoning summary should be captured (from response.reasoning_summary_text.delta)
	let reasoning = extract.reasoning_content.as_deref().ok_or("Should have reasoning content")?;
	assert!(reasoning.starts_with("**Getting weather for Paris**"), "reasoning should start with expected header");
	assert!(reasoning.contains("fetch the weather"), "reasoning should mention fetching weather");

	// Exactly one tool call
	let tool_calls = extract.stream_end.captured_tool_calls().ok_or("Should have tool calls")?;
	assert_eq!(tool_calls.len(), 1);

	// Exact tool call details
	let tc = &tool_calls[0];
	assert_eq!(tc.fn_name, "get_weather");
	assert_eq!(tc.fn_arguments, json!({"city": "Paris", "country": "France", "unit": "C"}));
	assert!(!tc.call_id.is_empty());

	// Exact usage
	let usage = extract.stream_end.captured_usage.as_ref().ok_or("Should have usage")?;
	assert_eq!(usage.prompt_tokens, Some(106));
	assert_eq!(usage.completion_tokens, Some(41));
	assert_eq!(usage.total_tokens, Some(147));

	// Encrypted reasoning content captured as thought_signatures
	let sigs = extract.stream_end.captured_thought_signatures().ok_or("Should have thought_signatures (encrypted_content)")?;
	assert_eq!(sigs.len(), 1, "Should have exactly one encrypted_content blob");
	assert_eq!(sigs[0], "gAAAAABpwxu6qKZn-6kiRA7yRzLjYy-VENTy0TqK50nHCtYdXbsZi4aNy5owaLB9qDxlvAmzfwjT25dpjJOxOJc70UJZaNffJFZdMoTlVkb2vkLOh-PthD3d0J9pbfYPrhY1FM0uFFWHn0xgOLf1W_qaWcUUQpTPMdQN2lgWvOuwMfG4_tAYXRC3vMB5U7RKOV6qXGorRQykJLSZSbio6RDPgDTg8ZTDr3R9nLgkiW_ObMsUKd3JcgbizRc5B-QPMVDOMN8zqZKE9_wLKbuERnofL_4dY6sluCBJfOMtb5O03eE9E6sWFukidsJnnObKJ3gK25Rg7qIMItAUs_ytmaJx_UW3ZeCRm2IJ8rYOSuv6zAZ9OBnyioF6KJ_3AMIgEDjeM_MPG4FNhD0MqEb7tpZr_fw3e0BiScJ0zUJUufPZ4H98xICRsMOor98Q2MxsnhHTgSodGNbBh-lSKThoxL6gOSHS9N73OGmlMUh_cWRz21hUXks2tVp1dQniK__1TW7Zmgl3PeVHIH6d5qKSenj0D7jodHkVFlpgonFFFVCTale2T4wGrmw5Qfor9ONM5RWcjaINIfAIJGYaiguZ6v4muGY6R8NuUVvhLe5Kafk-RNZTaFRXY4FFiN9YQNXEXrezY2hvWwuZStFlAO_0ADAKLEzU2fx-yPzVtd8ufDdMTQD6d1DXnlhP4WhR9ZcMZhCm_zQHwIp57o6Q16DKhFtUOTBQfX-9teGqkJd_lsOWA7kWDWb0UyhquHRt8o0jqs6no6y2FrfzQEe9X5K4Wa_odT2vKm6AMN38PP_jnoNZk18-u-EcUV_mve7P-d_2lAr59aH-ZYf6");

	Ok(())
}
