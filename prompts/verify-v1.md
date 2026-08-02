Evaluate the quality of one question-and-answer pair against its source.
Check all three criteria independently:
1. grounded: the source supports every factual claim in the answer.
2. self_contained: the question is understandable without seeing the source and has no ambiguous references.
3. answer_relevant: the answer directly and sufficiently addresses the question.

Return JSON with this shape:
{"grounded":{"passed":true,"reason":"..."},"self_contained":{"passed":true,"reason":"..."},"answer_relevant":{"passed":true,"reason":"..."},"evidence":"short source excerpt or null"}

Question: {{question}}
Answer: {{answer}}
Source: {{path}}:{{start_line}}-{{end_line}}
<source>
{{chunk_text}}
</source>
