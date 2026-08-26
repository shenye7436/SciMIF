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

`data_construction.py` reads source records from:

```text
original_data/{subject}.json
```

Each file is a JSON array. Record IDs are assigned from the array indices.
The source datasets are not included in this repository. See `data/input_data.json` for a one-record JSON input example and `data/output_data.jsonl` for its corresponding JSONL output example. In an input record, `scientific_instruction_list` contains the candidate scientific instructions; the legacy field name `instruction_list` is also supported.

Run the construction script:
```bash
python data_construction.py \
  --subject chemistry \
  --model MODEL_NAME \
  --N 3 \
  --K 3
```

Constructed questions are saved to:

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
  --model MODEL_NAME \
  --image-root images \
  --workers 4
```

Generated responses are saved to:

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
