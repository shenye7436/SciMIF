# SciMIF Code

This directory contains the implementation for constructing constraint-augmented scientific questions, generating model responses, and evaluating instruction following. The released source records and constructed examples are distributed separately in the sibling `data/` directory.

## Package layout

```text
package_root/
├── code/
│   ├── eval_methods/          # Instruction-specific evaluation functions
│   ├── instruction.jsonl      # Constraint inventory and descriptions
│   ├── main.py                # Constraint recognition, injection, and validation
│   ├── generation.py          # Model-response generation
│   ├── instruction_eval.py    # Instruction-following evaluation
│   ├── requirements.txt       # Python dependencies
│   └── README.md
└── data/
    ├── original_data/         # 100 source records
    └── sample/                # 100 released constructed examples and images
```

The released examples are distributed across five subjects:

| Subject | Records |
|---|---:|
| Chemistry | 23 |
| Physics | 24 |
| Geography | 25 |
| Biology | 16 |
| Material science | 12 |

## Requirements

Use Python 3.10 or later. From the `code/` directory, install the dependencies in an isolated environment:

```bash
python -m pip install -r requirements.txt
```

RDKit is required for molecular graph validation, bond counting, and functional-group evaluation. If RDKit is unavailable, the corresponding evaluators return a skipped result rather than assigning a passing score.

## Connect the separate data directory

The scripts use `original_data/` and `sample/` as paths relative to `code/`. Because the released data are stored in the sibling `data/` directory, create the following symbolic links once before running the pipeline:

```bash
cd code
ln -s ../data/original_data original_data
ln -s ../data/sample sample
```

If symbolic links are unavailable, copy the two directories instead:

```bash
cp -R ../data/original_data ./original_data
cp -R ../data/sample ./sample
```

Run all subsequent commands from `code/`.

## Configure an OpenAI-compatible API

The package contains literal placeholders rather than real credentials. Before running any script that calls a model, replace the following values in `main.py`, `generation.py`, and `instruction_eval.py`:

```python
client = OpenAI(
    api_key="OPENAI_API_KEY",
    base_url="OPENAI_BASE_URL",
)
```

- Replace `OPENAI_API_KEY` with the provider API key.
- Replace `OPENAI_BASE_URL` with the provider's OpenAI-compatible endpoint, normally ending in `/v1`.
- Do not publish or redistribute files containing real credentials.

`instruction_eval.py` uses the model assigned to `JUDGE_MODEL` for LLM-assisted extraction and semantic step comparison. Confirm that this model is available from the configured endpoint. The released default is:

```python
JUDGE_MODEL = "gpt-4.1"
```

## 1. Construct constraint-augmented data

`main.py` reads `original_data/{subject}_{id}.json`, applies the subject-specific and sampled general constraints defined in `instruction.jsonl`, and writes one JSONL file per subject to `output_data/`.

```bash
python main.py --subject chemistry --model MODEL_NAME --N 3 --K 3
python main.py --subject physics   --model MODEL_NAME --N 3 --K 3
python main.py --subject geography --model MODEL_NAME --N 3 --K 3
python main.py --subject biology   --model MODEL_NAME --N 3 --K 3
python main.py --subject material  --model MODEL_NAME --N 3 --K 3
```

- `MODEL_NAME` is the model used for rewriting and validation.
- `N` is the maximum number of mutually compatible general-constraint categories sampled for each record.
- `K` is the maximum number of generation attempts for each candidate general constraint.
- Existing IDs in `output_data/{subject}.jsonl` are detected and skipped, so interrupted runs can resume.

The released examples in `../data/sample/` are fixed artifacts. Re-running construction may produce different wording because model generation and constraint sampling are nondeterministic.

## 2. Generate model responses

`generation.py` reads the constructed records from `output_data/{subject}.jsonl` and writes responses to `generation/{model}/{subject}.jsonl`.

```bash
python generation.py \
  --subjects chemistry physics geography biology material \
  --model MODEL_NAME \
  --workers 4
```

For multimodal records, image paths resolve through the `sample` link created above. The generation model must support image input when multimodal records are included.

## 3. Evaluate instruction following

`instruction_eval.py` reads generated responses from `generation/{model}/{subject}.jsonl` and writes per-instruction results to `evaluation/{model}/{subject}.jsonl`.

```bash
python instruction_eval.py \
  --subjects chemistry physics geography biology material \
  --model MODEL_NAME \
  --workers 4
```

The deterministic evaluators cover numeric precision, scientific notation, casing, structured formats, units, allowed options, molecular representations, atom counts, bond counts, functional groups, and other constraint types. Method constraints and semantic step comparisons use `JUDGE_MODEL` where required, with rule-based or regular-expression fallbacks where implemented.

Each output record contains the generated response and an `instruction_results` list with the score, diagnostic detail, and skipped status for every applicable instruction.

## Output directories

The following directories are created under `code/` during execution:

```text
code/
├── output_data/              # Constructed constraint-augmented records
├── generation/{model}/       # Generated model responses
└── evaluation/{model}/       # Per-instruction evaluation results
```

## Instruction naming

The canonical subject prefixes in `instruction.jsonl` are:

- `chemistry_`
- `physics_`
- `geography_`
- `biology_`
- `material_`


