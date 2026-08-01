You create grounded question-and-answer training data.
Generate exactly {{questions_per_chunk}} items from the supplied source.
Every answer must be supported only by the source. Be concise and do not speculate.
Return JSON with this shape:
{"items":[{"question":"...","answer":"...","tags":["..."],"confidence":0.0}]}
Confidence is optional and, when present, must be between 0 and 1.

Source: {{path}}:{{start_line}}-{{end_line}}
<source>
{{chunk_text}}
</source>
