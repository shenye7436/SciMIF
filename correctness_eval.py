#!/usr/bin/env python3
"""Evaluate answer correctness with CompassVerifier through vLLM."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from verifier_prompts import CV_COT_PROMPT, CV_PROMPT


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON at {path}:{line_number}: {exc}') from exc


def parse_verifier_score(judgement_text: str) -> Tuple[int, str]:
    """Map CompassVerifier's A/B/C judgement to a binary correctness score."""
    text = (judgement_text or '').strip()
    boxed_labels = re.findall(r'\\boxed\{([ABC])\}', text.upper())
    if boxed_labels:
        label = boxed_labels[-1]
        score = 1 if label == 'A' else 0
        return score, f'parsed boxed label {label}'

    standalone_labels = re.findall(r'(?<![A-Z])([ABC])(?![A-Z])', text.upper())
    if standalone_labels:
        label = standalone_labels[-1]
        score = 1 if label == 'A' else 0
        return score, f'parsed label {label}'

    normalized = text.lower()
    if re.search(r'\b(?:incorrect|invalid|false|no)\b', normalized):
        return 0, 'parsed negative keyword'
    if re.search(r'\b(?:correct|true|yes)\b', normalized):
        return 1, 'parsed positive keyword'
    return 0, 'unparsed judgement text'


def load_instruction_results(evaluation_file: Path) -> Dict[Any, List[Dict[str, Any]]]:
    if not evaluation_file.is_file():
        return {}
    return {
        row['id']: row.get('instruction_results', [])
        for row in read_jsonl(evaluation_file)
        if row.get('id') is not None
    }


def evaluate_subject(
    subject: str,
    model: str,
    generation_dir: Path,
    evaluation_dir: Path,
    output_dir: Path,
    tokenizer: Any,
    verifier: Any,
    sampling_params: Any,
    prompt_template: str,
    batch_size: int,
) -> Tuple[int, int]:
    generation_file = generation_dir / model / f'{subject}.jsonl'
    evaluation_file = evaluation_dir / model / f'{subject}.jsonl'
    if not generation_file.is_file():
        raise FileNotFoundError(f'Generation file does not exist: {generation_file}')

    instruction_results = load_instruction_results(evaluation_file)
    if not evaluation_file.is_file():
        print(f'[warning] Instruction evaluation file not found: {evaluation_file}')

    model_output_dir = output_dir / model
    model_output_dir.mkdir(parents=True, exist_ok=True)
    score1_path = model_output_dir / f'{subject}_score1.jsonl'
    score0_path = model_output_dir / f'{subject}_score0.jsonl'
    correct_count = 0
    incorrect_count = 0

    def flush_batch(batch: List[Dict[str, Any]], score1_file: Any, score0_file: Any) -> None:
        nonlocal correct_count, incorrect_count
        if not batch:
            return
        outputs = verifier.generate([item['model_input'] for item in batch], sampling_params)
        for item, output in zip(batch, outputs):
            judgement = output.outputs[0].text
            score, detail = parse_verifier_score(judgement)
            record = {
                'id': item['id'],
                'subject': subject,
                'verifier_score': score,
                'verifier_detail': detail,
                'verifier_judgement': judgement,
                'instruction_results': instruction_results.get(item['id'], []),
            }
            target = score1_file if score == 1 else score0_file
            target.write(json.dumps(record, ensure_ascii=False) + '\n')
            if score == 1:
                correct_count += 1
            else:
                incorrect_count += 1

    batch: List[Dict[str, Any]] = []
    with score1_path.open('w', encoding='utf-8') as score1_file, score0_path.open('w', encoding='utf-8') as score0_file:
        for row in read_jsonl(generation_file):
            if row.get('id') is None:
                continue
            prompt = prompt_template.format(
                question=row.get('edit_question', ''),
                gold_answer=row.get('answer') or '',
                llm_response=row.get('response', ''),
            )
            messages = [{'role': 'user', 'content': prompt}]
            model_input = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            batch.append({'id': row['id'], 'model_input': model_input})
            if len(batch) >= batch_size:
                flush_batch(batch, score1_file, score0_file)
                batch = []
        flush_batch(batch, score1_file, score0_file)

    print(f'[{subject}] correct={correct_count} incorrect={incorrect_count}')
    print(f'[{subject}] wrote {score1_path} and {score0_path}')
    return correct_count, incorrect_count


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate generated-answer correctness with CompassVerifier.')
    parser.add_argument('--subject', default='chemistry', help='One subject; ignored when --subjects is provided.')
    parser.add_argument('--subjects', nargs='+', default=None, metavar='NAME', help='One or more subjects.')
    parser.add_argument('--model', required=True, help='Model response subdirectory name.')
    parser.add_argument('--generation-dir', default='generation', type=Path, help='Generation result root directory.')
    parser.add_argument('--evaluation-dir', default='evaluation', type=Path, help='Instruction evaluation root directory.')
    parser.add_argument('--output-dir', default='correctness', type=Path, help='Correctness result root directory.')
    parser.add_argument('--verifier-model', default='opencompass/CompassVerifier-3B', help='CompassVerifier model name or path.')
    parser.add_argument('--use-cot', action='store_true', help='Use the chain-of-thought verifier prompt.')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit('Install CompassVerifier dependencies with: pip install -r requirements-verifier.txt') from exc

    tokenizer = AutoTokenizer.from_pretrained(args.verifier_model)
    verifier = LLM(model=args.verifier_model, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    prompt_template = CV_COT_PROMPT if args.use_cot else CV_PROMPT
    subjects = args.subjects if args.subjects else [args.subject]

    for subject in subjects:
        subject = (subject or '').strip()
        if subject:
            evaluate_subject(
                subject=subject,
                model=args.model,
                generation_dir=args.generation_dir.expanduser().resolve(),
                evaluation_dir=args.evaluation_dir.expanduser().resolve(),
                output_dir=args.output_dir.expanduser().resolve(),
                tokenizer=tokenizer,
                verifier=verifier,
                sampling_params=sampling_params,
                prompt_template=prompt_template,
                batch_size=args.batch_size,
            )


if __name__ == '__main__':
    main()
