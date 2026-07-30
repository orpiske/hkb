use crate::types::Chunk;

pub const PROMPT_VERSION: &str = "qa-v1";

pub const PROMPT_TEMPLATE: &str = r#"You create grounded question-and-answer training data.
Generate exactly {{questions_per_chunk}} items from the supplied source.
Every answer must be supported only by the source. Be concise and do not speculate.
Return JSON with this shape:
{"items":[{"question":"...","answer":"...","tags":["..."],"confidence":0.0}]}
Confidence is optional and, when present, must be between 0 and 1.

Source: {{path}}:{{start_line}}-{{end_line}}
<source>
{{chunk_text}}
</source>"#;

pub fn build_qa_prompt(chunk: &Chunk, questions_per_chunk: usize) -> String {
    PROMPT_TEMPLATE
        .replace("{{questions_per_chunk}}", &questions_per_chunk.to_string())
        .replace("{{path}}", &chunk.path)
        .replace("{{start_line}}", &chunk.start_line.to_string())
        .replace("{{end_line}}", &chunk.end_line.to_string())
        .replace("{{chunk_text}}", &chunk.text)
}

#[cfg(test)]
mod tests {
    use super::build_qa_prompt;
    use crate::types::Chunk;

    #[test]
    fn prompt_contains_generation_rules_and_source_provenance() {
        let chunk = Chunk {
            chunk_id: "chunk-1".to_owned(),
            path: "README.md".to_owned(),
            language: "markdown".to_owned(),
            start_line: 4,
            end_line: 8,
            content_hash: "hash".to_owned(),
            text: "HKB builds datasets.".to_owned(),
        };

        let prompt = build_qa_prompt(&chunk, 3);

        assert!(prompt.contains("exactly 3 items"));
        assert!(prompt.contains("README.md:4-8"));
        assert!(prompt.contains("<source>\nHKB builds datasets.\n</source>"));
    }
}
