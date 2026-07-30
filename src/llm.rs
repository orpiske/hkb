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

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("LLM HTTP request failed")]
    Http(#[from] reqwest::Error),
    #[error("LLM API returned HTTP {status}: {body}")]
    Api { status: StatusCode, body: String },
    #[error("LLM returned invalid JSON")]
    InvalidJson(#[source] serde_json::Error),
    #[error("LLM response did not contain a completion")]
    MissingCompletion,
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn generate_questions(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Vec<GeneratedQa>, LlmError>;
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
    format: &'a str,
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
    async fn generate_questions(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Vec<GeneratedQa>, LlmError> {
        let response = self
            .http
            .post(format!("{}/api/generate", self.endpoint))
            .json(&OllamaRequest {
                model: &config.model,
                prompt,
                stream: false,
                format: "json",
                options: OllamaOptions {
                    temperature: config.temperature,
                },
            })
            .send()
            .await?;
        let body = checked_body(response).await?;
        let response: OllamaResponse =
            serde_json::from_str(&body).map_err(LlmError::InvalidJson)?;

        parse_generated_questions(&response.response)
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
    async fn generate_questions(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<Vec<GeneratedQa>, LlmError> {
        let mut request = self
            .http
            .post(format!("{}/chat/completions", self.endpoint))
            .json(&OpenAiRequest {
                model: &config.model,
                messages: [OpenAiMessage {
                    role: "user",
                    content: prompt,
                }],
                temperature: config.temperature,
            });
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }

        let response = request.send().await?;
        let body = checked_body(response).await?;
        let response: OpenAiResponse =
            serde_json::from_str(&body).map_err(LlmError::InvalidJson)?;
        let completion = response
            .choices
            .into_iter()
            .next()
            .ok_or(LlmError::MissingCompletion)?;

        parse_generated_questions(&completion.message.content)
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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum GeneratedEnvelope {
    Wrapped { items: Vec<GeneratedQa> },
    Bare(Vec<GeneratedQa>),
}

fn parse_generated_questions(raw: &str) -> Result<Vec<GeneratedQa>, LlmError> {
    let json = strip_code_fence(raw);
    let envelope: GeneratedEnvelope = serde_json::from_str(json).map_err(LlmError::InvalidJson)?;

    Ok(match envelope {
        GeneratedEnvelope::Wrapped { items } | GeneratedEnvelope::Bare(items) => items,
    })
}

fn strip_code_fence(raw: &str) -> &str {
    let trimmed = raw.trim();
    let Some(after_opening) = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
    else {
        return trimmed;
    };

    after_opening
        .strip_suffix("```")
        .unwrap_or(after_opening)
        .trim()
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    use super::{LlmClient, OllamaClient, OpenAiCompatibleClient, parse_generated_questions};
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

    #[tokio::test]
    async fn ollama_client_uses_generate_endpoint() -> Result<(), Box<dyn std::error::Error>> {
        let generated = r#"{"items":[{"question":"Q","answer":"A"}]}"#;
        let response = serde_json::json!({ "response": generated }).to_string();
        let (endpoint, server) = mock_server(response)?;
        let client = OllamaClient::new(endpoint);

        let items = client
            .generate_questions("prompt", &GenerationConfig::default())
            .await?;
        let request = server
            .join()
            .map_err(|_| std::io::Error::other("mock server panicked"))??;

        assert_eq!(items.len(), 1);
        assert!(request.starts_with("POST /api/generate "));
        assert!(request.contains("\"stream\":false"));
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

        let items = client
            .generate_questions("prompt", &GenerationConfig::default())
            .await?;
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
