use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::GenerationConfig;

pub const DEFAULT_OLLAMA_ENDPOINT: &str = "http://localhost:11434";
pub const DEFAULT_OPENAI_ENDPOINT: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratedQa {
    pub question: String,
    pub answer: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct LlmRequestConfig<'a> {
    pub model: &'a str,
    pub temperature: f32,
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("LLM HTTP request failed")]
    Http(#[from] reqwest::Error),
    #[error("LLM API returned HTTP {status}: {body}")]
    Api { status: StatusCode, body: String },
    #[error("LLM returned malformed JSON: {source}; response excerpt: {excerpt}")]
    InvalidJson {
        #[source]
        source: serde_json::Error,
        excerpt: String,
    },
    #[error("LLM returned JSON with an unexpected schema: {reason}; response excerpt: {excerpt}")]
    UnexpectedSchema { reason: String, excerpt: String },
    #[error("LLM response did not contain a completion")]
    MissingCompletion,
}

impl LlmError {
    pub(crate) fn is_retryable(&self) -> bool {
        match self {
            Self::Api { status, .. } => {
                *status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
            }
            Self::Http(source) => source.is_timeout() || source.is_connect(),
            Self::InvalidJson { .. } | Self::UnexpectedSchema { .. } | Self::MissingCompletion => {
                true
            }
        }
    }
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete_structured(
        &self,
        prompt: &str,
        config: &LlmRequestConfig<'_>,
        response_schema: &serde_json::Value,
    ) -> Result<String, LlmError>;
}

pub async fn generate_questions(
    client: &dyn LlmClient,
    prompt: &str,
    config: &GenerationConfig,
) -> Result<Vec<GeneratedQa>, LlmError> {
    let response_schema = generated_response_schema(config.questions_per_chunk);
    let request_config = LlmRequestConfig {
        model: &config.model,
        temperature: config.temperature,
    };
    let completion = client
        .complete_structured(prompt, &request_config, &response_schema)
        .await?;
    parse_generated_questions(&completion)
}

#[derive(Debug, Clone)]
pub struct OllamaClient {
    http: Client,
    endpoint: String,
}

impl OllamaClient {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            http: Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
        }
    }
}

#[derive(Debug, Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    format: &'a serde_json::Value,
    options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
}

#[async_trait]
impl LlmClient for OllamaClient {
    async fn complete_structured(
        &self,
        prompt: &str,
        config: &LlmRequestConfig<'_>,
        response_schema: &serde_json::Value,
    ) -> Result<String, LlmError> {
        let response = self
            .http
            .post(format!("{}/api/generate", self.endpoint))
            .json(&OllamaRequest {
                model: config.model,
                prompt,
                stream: false,
                format: response_schema,
                options: OllamaOptions {
                    temperature: config.temperature,
                },
            })
            .send()
            .await?;
        let body = checked_body(response).await?;
        let response: OllamaResponse =
            serde_json::from_str(&body).map_err(|source| LlmError::InvalidJson {
                source,
                excerpt: response_excerpt(&body),
            })?;

        Ok(response.response)
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleClient {
    http: Client,
    endpoint: String,
    api_key: Option<String>,
}

impl OpenAiCompatibleClient {
    pub fn new(endpoint: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            http: Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_owned(),
            api_key,
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    messages: [OpenAiMessage<'a>; 1],
    temperature: f32,
    response_format: OpenAiResponseFormat<'a>,
}

#[derive(Debug, Serialize)]
struct OpenAiResponseFormat<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    json_schema: OpenAiJsonSchema<'a>,
}

#[derive(Debug, Serialize)]
struct OpenAiJsonSchema<'a> {
    name: &'static str,
    strict: bool,
    schema: &'a serde_json::Value,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    content: String,
}

#[async_trait]
impl LlmClient for OpenAiCompatibleClient {
    async fn complete_structured(
        &self,
        prompt: &str,
        config: &LlmRequestConfig<'_>,
        response_schema: &serde_json::Value,
    ) -> Result<String, LlmError> {
        let mut request = self
            .http
            .post(format!("{}/chat/completions", self.endpoint))
            .json(&OpenAiRequest {
                model: config.model,
                messages: [OpenAiMessage {
                    role: "user",
                    content: prompt,
                }],
                temperature: config.temperature,
                response_format: OpenAiResponseFormat {
                    kind: "json_schema",
                    json_schema: OpenAiJsonSchema {
                        name: "hkb_response",
                        strict: true,
                        schema: response_schema,
                    },
                },
            });
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = request.send().await?;
        let body = checked_body(response).await?;
        let response: OpenAiResponse =
            serde_json::from_str(&body).map_err(|source| LlmError::InvalidJson {
                source,
                excerpt: response_excerpt(&body),
            })?;
        let completion = response
            .choices
            .into_iter()
            .next()
            .ok_or(LlmError::MissingCompletion)?;

        Ok(completion.message.content)
    }
}

async fn checked_body(response: reqwest::Response) -> Result<String, LlmError> {
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        return Err(LlmError::Api { status, body });
    }
    Ok(body)
}

fn parse_generated_questions(raw: &str) -> Result<Vec<GeneratedQa>, LlmError> {
    let value = parse_json_completion(raw)?;
    let items = match value {
        serde_json::Value::Array(items) => items,
        serde_json::Value::Object(mut object) => object
            .remove("items")
            .or_else(|| object.remove("questions"))
            .and_then(|items| items.as_array().cloned())
            .ok_or_else(|| LlmError::UnexpectedSchema {
                reason: "expected an array or an object containing an `items` array".to_owned(),
                excerpt: response_excerpt(raw),
            })?,
        _ => {
            return Err(LlmError::UnexpectedSchema {
                reason: "expected a JSON array or object".to_owned(),
                excerpt: response_excerpt(raw),
            });
        }
    };

    serde_json::from_value(serde_json::Value::Array(items)).map_err(|source| {
        LlmError::UnexpectedSchema {
            reason: source.to_string(),
            excerpt: response_excerpt(raw),
        }
    })
}

pub(crate) fn parse_json_completion(raw: &str) -> Result<serde_json::Value, LlmError> {
    serde_json::from_str(json_candidate(raw)).map_err(|source| LlmError::InvalidJson {
        source,
        excerpt: response_excerpt(raw),
    })
}

fn json_candidate(raw: &str) -> &str {
    let trimmed = raw.trim();
    if let Some(opening) = trimmed.find("```") {
        let after_fence = &trimmed[opening + 3..];
        let after_language = after_fence
            .strip_prefix("json")
            .unwrap_or(after_fence)
            .trim_start();
        return after_language
            .split_once("```")
            .map_or(after_language, |(json, _)| json)
            .trim();
    }

    trimmed
}

fn generated_response_schema(questions_per_chunk: usize) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": questions_per_chunk,
                "maxItems": questions_per_chunk,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": { "type": "string" },
                        "answer": { "type": "string" },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" }
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1
                        }
                    },
                    "required": ["question", "answer"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["items"],
        "additionalProperties": false
    })
}

fn response_excerpt(response: &str) -> String {
    const MAX_CHARACTERS: usize = 500;

    let mut excerpt = response.chars().take(MAX_CHARACTERS).collect::<String>();
    if response.chars().count() > MAX_CHARACTERS {
        excerpt.push_str("...");
    }
    excerpt
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    use reqwest::StatusCode;

    use super::{
        LlmError, OllamaClient, OpenAiCompatibleClient, generate_questions,
        parse_generated_questions,
    };
    use crate::types::GenerationConfig;

    #[test]
    fn parses_wrapped_json() -> Result<(), Box<dyn std::error::Error>> {
        let items = parse_generated_questions(
            r#"{"items":[{"question":"What?","answer":"This.","tags":[],"confidence":0.8}]}"#,
        )?;

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].question, "What?");
        assert_eq!(items[0].confidence, Some(0.8));
        Ok(())
    }

    #[test]
    fn accepts_bare_arrays_inside_code_fences() -> Result<(), Box<dyn std::error::Error>> {
        let items =
            parse_generated_questions("```json\n[{\"question\":\"Q\",\"answer\":\"A\"}]\n```")?;

        assert_eq!(items.len(), 1);
        assert!(items[0].tags.is_empty());
        Ok(())
    }

    #[test]
    fn accepts_questions_alias_and_surrounding_prose() -> Result<(), Box<dyn std::error::Error>> {
        let items = parse_generated_questions(
            "Here is the result:\n```json\n{\"questions\":[{\"question\":\"Q\",\"answer\":\"A\"}]}\n```",
        )?;

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].answer, "A");
        Ok(())
    }

    #[test]
    fn malformed_json_reports_reason_and_excerpt() {
        let error = parse_generated_questions(r#"{"items":[{"question":"Q"}"#)
            .err()
            .map(|error| error.to_string());

        assert!(error.as_deref().is_some_and(|message| {
            message.contains("malformed JSON")
                && message.contains("line 1 column")
                && message.contains("response excerpt")
        }));
    }

    #[test]
    fn retries_rate_limits_and_server_errors_but_not_authentication_errors() {
        let rate_limit = LlmError::Api {
            status: StatusCode::TOO_MANY_REQUESTS,
            body: String::new(),
        };
        let server_error = LlmError::Api {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: String::new(),
        };
        let authentication = LlmError::Api {
            status: StatusCode::UNAUTHORIZED,
            body: String::new(),
        };

        assert!(rate_limit.is_retryable());
        assert!(server_error.is_retryable());
        assert!(!authentication.is_retryable());
    }

    #[tokio::test]
    async fn ollama_client_uses_generate_endpoint() -> Result<(), Box<dyn std::error::Error>> {
        let generated = r#"{"items":[{"question":"Q","answer":"A"}]}"#;
        let response = serde_json::json!({ "response": generated }).to_string();
        let (endpoint, server) = mock_server(response)?;
        let client = OllamaClient::new(endpoint);

        let items = generate_questions(&client, "prompt", &GenerationConfig::default()).await?;
        let request = server
            .join()
            .map_err(|_| std::io::Error::other("mock server panicked"))??;

        assert_eq!(items.len(), 1);
        assert!(request.starts_with("POST /api/generate "));
        assert!(request.contains("\"stream\":false"));
        assert!(request.contains("\"minItems\":3"));
        assert!(request.contains("\"required\":[\"items\"]"));
        Ok(())
    }

    #[tokio::test]
    async fn openai_client_sends_optional_bearer_auth() -> Result<(), Box<dyn std::error::Error>> {
        let generated = r#"{"items":[{"question":"Q","answer":"A"}]}"#;
        let response = serde_json::json!({
            "choices": [{ "message": { "content": generated } }]
        })
        .to_string();
        let (endpoint, server) = mock_server(response)?;
        let client = OpenAiCompatibleClient::new(endpoint, Some("secret".to_owned()));

        let items = generate_questions(&client, "prompt", &GenerationConfig::default()).await?;
        let request = server
            .join()
            .map_err(|_| std::io::Error::other("mock server panicked"))??;

        assert_eq!(items.len(), 1);
        assert!(request.starts_with("POST /chat/completions "));
        assert!(
            request
                .to_lowercase()
                .contains("authorization: bearer secret")
        );
        assert!(request.contains("\"response_format\":{\"type\":\"json_schema\""));
        assert!(request.contains("\"name\":\"hkb_response\",\"strict\":true"));
        assert!(request.contains("\"required\":[\"items\"]"));
        Ok(())
    }

    fn mock_server(
        response_body: String,
    ) -> Result<(String, thread::JoinHandle<std::io::Result<String>>), std::io::Error> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept()?;
            let mut request = Vec::new();
            let mut buffer = [0; 4_096];

            loop {
                let bytes_read = stream.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..bytes_read]);
                if request_is_complete(&request) {
                    break;
                }
            }

            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            )?;

            Ok(String::from_utf8_lossy(&request).into_owned())
        });

        Ok((format!("http://{address}"), server))
    }

    fn request_is_complete(request: &[u8]) -> bool {
        let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
            return false;
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        });

        content_length.is_some_and(|length| request.len() >= header_end + 4 + length)
    }
}
