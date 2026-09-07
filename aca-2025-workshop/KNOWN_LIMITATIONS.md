# Known limitations at publication

This publication preserves the tested workshop; it does not claim that later review findings have been fixed.

- Adapter reload checks model/corpus/weight hashes, but the saved dataset.json is not bound to the recorded training/validation hashes. Treat it as an integrity gap; do not edit adapter dataset files and describe them as the original experiment. Bind and verify the dataset fingerprint before relying on final comparisons.
- A two-step smoke adapter can be selected for the final test. The smoke flag is recorded, but the code does not block that combination. Use smoke adapters only for mechanics checks; use full training and frozen settings for substantive evaluation.
- Selected UHC plan evidence omits the plan-level out-of-network maximum field marked Not Applicable. It does not establish a personal out-of-network cap or resolve balance billing. The underlying limitation remains relevant to the prebuilt sample.
- Default document mode supplies a disclosed prefix of complete chunks, not all documents. Graph retrieval uses explicit source joins and structural links with lexical seeds, not Microsoft GraphRAG community summaries.
- Automatic JSON/citation-ID checks are not answer accuracy. Current results include format failures and semantically weak answers. Independent correctness, entailment and caveat review remains necessary.

These issues were documented in a later model-assisted review. They do not invalidate the recorded CPU execution, but they limit what conclusions can be drawn from it. Full training and Colab/GPU execution remain unverified.
