# SciMIF: Understanding Multimodal Instruction Following in Scientific Domains

SciMIF is a benchmark for evaluating whether Multimodal Large Language Models (MLLMs) can follow complex instructions in scientific domains. Unlike conventional scientific benchmarks that primarily evaluate answer correctness, SciMIF evaluates whether a model satisfies explicit scientific and general constraints, including required methods, units, formats, terminology, reasoning procedures, numerical precision, and response structures.

## Installation

```bash
git clone https://github.com/shenye7436/SciMIF.git
cd SciMIF

python -m pip install -r requirements.txt
```

## Data Sources

SciMIF is constructed by augmenting samples from publicly available scientific datasets across five disciplines.

| Discipline | Source Datasets |
|---|---|
| Chemistry | ChemEval, S2-TOMG-Bench-mini |
| Geography | EarthSE, IMAGEO-Bench |
| Biology | Mol-Instructions, LAB-Bench |
| Material | MaScQA, MatCha, LLM4Mat-Bench, MatSciBench |
| Physics | UGPhysics, PhysReason, PhysUniBench |


## Usage

### 1. Construct Constraint-Augmented Questions

The source datasets are not included in this repository. `data/input_data.json` and `data/output_data.jsonl` are format examples only and are not read automatically by the scripts. Prepare the real input as a JSON array following `data/input_data.json`; record IDs are assigned from the array indices. In an input record, `scientific_instruction_list` contains the candidate scientific instructions; the legacy field name `instruction_list` is also supported.

Run the construction script:
```bash
python data_construction.py \
  --subject chemistry \
  --input-path /path/to/chemistry.json \
  --output-path output_data/chemistry.jsonl \
  --model MODEL_NAME \
  --N 3 \
  --K 3
```

Constructed questions are saved to the path supplied with `--output-path`, for example:

```text
output_data/{subject}.jsonl
```

`output_data` should not be treated as the final benchmark directly. The constructed samples require human verification.


A sample contains the following fields:

- `id`: Sample ID.
- `task`: Scientific task type.
- `original_question`: Original scientific question.
- `edit_question`: Question after constraint injection.
- `answer`: Reference answer.
- `choose_instruction`: Applied scientific constraints.
- `instruction_list`: All constraints associated with the sample. The `source` field is `original` for constraints already present in the original question, `core_task` for candidate scientific constraints supplied by the input record, or `added_general` for additional general constraints introduced during construction.
- `image`: Optional image path or list of image paths for multimodal samples.

### 2. Generate Model Responses

```bash
python generation.py \
  --subjects chemistry physics geography life materials \
  --input-dir output_data \
  --output-dir generation \
  --model MODEL_NAME \
  --image-root images \
  --workers 4
```

For each subject, `generation.py` reads `{input-dir}/{subject}.jsonl`. Generated responses are saved to:

```text
generation/{model}/{subject}.jsonl
```

The selected model must support image inputs when processing multimodal samples.

### 3. Evaluate Instruction Following

```bash
python instruction_eval.py \
  --subjects chemistry physics geography life materials \
  --model MODEL_NAME \
  --workers 4
```

Evaluation results are saved to:

```text
evaluation/{model}/{subject}.jsonl
```

Each output record contains the generated response and the evaluation result for every applicable instruction.

### 4. Evaluate Answer Correctness

We use [CompassVerifier-32B](https://github.com/open-compass/CompassVerifier) as the judge model for answer-correctness evaluation. CompassVerifier dependencies are optional and require an environment supported by vLLM:

```bash
python -m pip install -r requirements-verifier.txt
```

Run answer-correctness evaluation:

```bash
python correctness_eval.py \
  --subjects chemistry physics geography life materials \
  --model MODEL_NAME \
  --generation-dir generation \
  --evaluation-dir evaluation \
  --output-dir correctness \
  --verifier-model opencompass/CompassVerifier-32B \
  --batch-size 8
```

For each subject, results are separated by the binary verifier score:

```text
correctness/{model}/{subject}_score1.jsonl
correctness/{model}/{subject}_score0.jsonl
```

If the corresponding instruction-evaluation file is available, its `instruction_results` are included in each correctness record.
