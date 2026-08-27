# Evaluation strategy

Mini SIA evaluates retrieval and generation separately so a regression is diagnosable.

| Metric | What it detects | Default interpretation |
| --- | --- | --- |
| Retrieval recall | Missing expected documents | Did the retriever find the evidence? |
| Reciprocal rank | Relevant evidence ranked too low | How soon did useful evidence appear? |
| Context precision | Relevant sources ranked too low | How early did useful evidence appear? |
| Answer coverage | Missing required answer concepts | Did the answer include key facts? |
| Citation validity | Missing or out-of-range citations | Can the answer be traced to sources? |

Context precision is source-level average precision: it rewards relevant sources near
the top without labeling every chunk in a source document. The aggregate quality score
is the unweighted mean of these five metrics. In a real
deployment, calibrate weights and thresholds with human reviewers and production
traffic. Do not treat substring-based answer coverage as a substitute for expert
judgment; it is a stable regression signal for CI.

Add typical, edge, and adversarial cases to `evals/golden.jsonl`. Run the deterministic
suite on every change, and run the OpenAI-provider suite before prompt/model releases.
