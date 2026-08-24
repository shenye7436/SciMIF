import os
import json
import random
import argparse
import re
import threading
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
INSTRUCTION_NAME_ALIASES = {'life_unit_consistency': 'biology_unit_consistency', 'life_entity_relationship_format_validity': 'biology_entity_relationship_format_validity', 'life_sequence_length_constraint': 'biology_sequence_length_constraint', 'life_method_constraint': 'biology_method_constraint', 'life_analysis_steps_constraint': 'biology_analysis_steps_constraint', 'materials_unit_consistency': 'material_unit_consistency', 'materials_characterization_technique_format_constraint': 'material_characterization_technique_format_constraint', 'materials_property_prediction_constraint': 'material_property_prediction_constraint', 'materials_method_constraint': 'material_method_constraint', 'materials_analysis_steps_constraint': 'material_analysis_steps_constraint'}

def normalize_key(s):
    if not s:
        return ''
    clean = str(s).strip('[]').strip().lower()
    return INSTRUCTION_NAME_ALIASES.get(clean, clean)

def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_existing_results(path: Path):
    if not path.exists():
        return ({}, set())
    data_dict = {}
    existing_ids = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                idx = obj.get('id')
                if idx is not None:
                    data_dict[idx] = obj
                    existing_ids.add(idx)
            except:
                continue
    return (data_dict, existing_ids)

def save_jsonl_append(path: Path, item: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_instruction_library(instruction_jsonl: Path):
    general_lib = {}
    subject_lib = {}
    group_mapping = {}
    inst_to_group = {}
    if not instruction_jsonl.exists():
        print(f'Fatal error: instruction library not found: {instruction_jsonl}')
        return (general_lib, subject_lib, group_mapping, inst_to_group)
    with open(instruction_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                clean_line = line.strip().rstrip(',')
                obj = json.loads(clean_line)
                for name, content in obj.items():
                    clean_name = normalize_key(name)
                    if clean_name.startswith('general_'):
                        general_lib[clean_name] = content
                        group = content.get('group', 'default')
                        group_mapping.setdefault(group, set()).add(clean_name)
                        inst_to_group[clean_name] = group
                    else:
                        subject_lib[clean_name] = content
            except:
                continue
    print(f'Loaded {len(general_lib)} general instructions and {len(subject_lib)} scientific instructions.')
    return (general_lib, subject_lib, group_mapping, inst_to_group)

class GeneralInstructionLimiter:

    def __init__(self, existing_results, glib, limit=100):
        self.limit = limit
        self.lock = threading.Lock()
        self.counter = defaultdict(int)
        for item in existing_results.values():
            for inst in item.get('instruction_list', []):
                name = normalize_key(inst.get('instruction_name'))
                if name in glib:
                    self.counter[name] += 1

    def can_use(self, name):
        with self.lock:
            return self.counter[name] < self.limit

    def record(self, name):
        with self.lock:
            self.counter[name] += 1

    def distribution(self):
        return dict(sorted(self.counter.items(), key=lambda x: -x[1]))

def call_llm(client, model, prompt):
    try:
        res = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': prompt}], temperature=0.2)
        return res.choices[0].message.content.strip()
    except:
        return ''

def build_extract_prompt(question, general_lib):
    instruction_descriptions = {name: content.get('description', '') for name, content in general_lib.items()}
    return f'\n# Task\nIdentify ONLY the task instructions from the [Instruction List] that are **explicitly and literally stated** in the [Input Question].\n\n# Strict Criteria for Numerical Annotations\nFor `decimal_annotation`, `scientific_anotation`, or any precision-related instructions, **ONLY** identify them if the question contains **Explicit Keywords**.\n\n# Data\n- [Input Question]: {question}\n- [Instruction List]: {json.dumps(instruction_descriptions, indent=2, ensure_ascii=False)}\n\n# Output\nReturn ONLY a JSON list of names: ["name1", "name2"].\n'

def build_add_prompt(prev_question, answer, instruction_name, instruction_info):
    param_desc = instruction_info.get('required_parameters', 'None')
    return f"""\n#Task Rewrite the [Original Question] to strictly incorporate the [New Instruction]. \n\n#Data \n##Original Question: \n{prev_question} \n##Reference Answer (STRICTLY CONFIDENTIAL): \n{answer} \n##New Instruction: \n{instruction_info.get('description', '')} \n##Parameter Requirement: \n{param_desc} \n\n#Rules: \n1. **NO STRUCTURED ANSWER FIELD**: You MUST NOT include any structured answer field such as: - answer: ... - "answer": ... - 'answer': ... - any JSON key named "answer" The rewritten question must not contain any explicit answer field or answer key-value pair. \n2. **CONCRETE PARAMETERS**: If the instruction requires parameters (e.g., "specific units", "template"), you MUST choose a specific value/template and state it clearly in the question. \n3. **PRESERVE SCIENTIFIC INTEGRITY & CONTEXT**: \n   - Keep ALL original clinical summaries, case studies, or source texts. \n   - You are rewriting the INSTRUCTION part, but the DATA/SOURCE TEXT part must remain 100% intact.\n4. **JSON OUTPUT ONLY**: - "new_question": The rewritten question text. - "concrete_parameter": The specific value/unit/template you used. \n5. The rewritten question text MUST NOT include complete reference answer or any part of it that reveals the solution.\n\nReturn ONLY the JSON.\n"""

def enhance_question(client, model, question, answer, item_instruction_list, glib, slib, gmap, i2g, N, K, limiter):
    current_question = question
    final_instruction_meta = []
    used_general_insts = set()
    used_groups = set()
    raw_extract = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': build_extract_prompt(question, glib)}], temperature=0.0).choices[0].message.content.strip()
    try:
        json_match = re.search('\\[.*\\]', raw_extract, re.DOTALL)
        if json_match:
            detected = json.loads(json_match.group())
            for name in detected:
                clean_name = normalize_key(name)
                if clean_name in glib:
                    info = glib[clean_name]
                    meta_entry = {'instruction_name': clean_name, 'source': 'original'}
                    used_general_insts.add(clean_name)
                    group = i2g.get(clean_name)
                    if group:
                        used_groups.add(group)
                    final_instruction_meta.append(meta_entry)
    except:
        pass
    for subj_inst_raw in item_instruction_list:
        clean_subj = normalize_key(subj_inst_raw)
        if clean_subj in slib:
            info = slib[clean_subj]
            meta_entry = {'instruction_name': clean_subj, 'source': 'core_task'}
            raw_res = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': build_add_prompt(current_question, answer, clean_subj, info)}], temperature=0.2).choices[0].message.content.strip()
            try:
                json_match = re.search('\\{.*\\}', raw_res, re.DOTALL)
                if json_match:
                    res_data = json.loads(json_match.group())
                    new_q = res_data.get('new_question', '').strip()
                    concrete_p = res_data.get('concrete_parameter', '').strip()
                    if str(answer).lower() in new_q.lower():
                        new_q = new_q.replace(str(answer), '[REDACTED]')
                    if new_q and len(new_q) > 10:
                        current_question = new_q
                        if 'required_parameters' in info:
                            meta_entry['required_parameters'] = concrete_p if concrete_p else info['required_parameters']
            except:
                pass
            final_instruction_meta.append(meta_entry)
    available_groups = list(set(gmap.keys()) - used_groups)
    random.shuffle(available_groups)
    selected_groups = available_groups[:N]
    for gid in selected_groups:
        candidates = list(gmap[gid])
        random.shuffle(candidates)
        category_accepted = False
        for cand in candidates:
            if not limiter.can_use(cand):
                continue
            info = glib[cand]
            for _attempt in range(K):
                raw_res = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': build_add_prompt(current_question, answer, cand, info)}], temperature=0.2).choices[0].message.content.strip()
                try:
                    json_match = re.search('\\{.*\\}', raw_res, re.DOTALL)
                    if not json_match:
                        continue
                    res_data = json.loads(json_match.group())
                    new_q = res_data.get('new_question', '').strip()
                    concrete_p = res_data.get('concrete_parameter', '').strip()
                    if str(answer).lower() in new_q.lower():
                        continue
                    v1_prompt = f"Does the following question strictly require the model to follow this instruction: '{info.get('description')}'?\n                    Reply 'True' or 'False'.\n                    Question: {new_q}"
                    v1 = call_llm(client, model, v1_prompt)
                    v2_prompt = f"""[Role]: Expert Quality Assurance Specialist\n\n[Task]: Determine if the [Original Answer] remains semantically sufficient to satisfy the [New Question].\n\n[Data]\n- Original Answer: {answer}\n- New Question: {new_q}\n\n[Validation Criteria (Reply 'True' if):]\n1. Core Content: The factual information and entities in the [Original Answer] are still the correct solution.\n2. Numeric Flexibility: Different representations of the same value are allowed (e.g., 0.5 == 5e-1, 1000 == 1,000).\n3. Case/Format: Changes in English case (apple vs APPLE), punctuation, or formatting (list vs table) are acceptable.\n4. Structural Instructions: If the New Question adds "response structure" requirements (e.g., "put answer at the end", "reason first", "JSON format"), the [Original Answer] is still valid as long as it provides the factual core.\n\n[Invalidation Criteria (Reply 'False' if):]\n1. Task Type Drift: The New Question changes the required output type. \n   - If the instruction mandates a "Multiple Choice" format (e.g., "Select A, B, C, or D"), but the [Original Answer] is not a single option letter/text.\n   - If the instruction mandates a "Yes/No" or "True/False" response, but the [Original Answer] is a detailed description or list.\n   - If the instruction asks for a specific count (e.g., "How many...?"), but the [Original Answer] provides the names of items instead of the digit.\n2. Missing Info: The New Question requires information that simply does not exist in the [Original Answer].\n\nReply ONLY 'True' or 'False'."""
                    v2 = call_llm(client, model, v2_prompt)
                    if 'true' in v1.lower() and 'true' in v2.lower():
                        current_question = new_q
                        used_general_insts.add(cand)
                        used_groups.add(gid)
                        limiter.record(cand)
                        entry = {'instruction_name': cand, 'source': 'added_general'}
                        if 'required_parameters' in info:
                            entry['required_parameters'] = concrete_p if concrete_p else info['required_parameters']
                        final_instruction_meta.append(entry)
                        category_accepted = True
                        break
                except:
                    continue
            if category_accepted:
                break
    choose_instruction = [normalize_key(name) for name in item_instruction_list if normalize_key(name) in slib]
    return (current_question, final_instruction_meta, choose_instruction)

def load_subject_records(base_dir: Path, subject: str):
    canonical_subject = {'life': 'biology', 'materials': 'material'}.get(subject, subject)
    input_dir = base_dir / 'original_data'
    paths = sorted(input_dir.glob(f'{canonical_subject}_*.json'), key=lambda path: int(path.stem.rsplit('_', 1)[1]))
    if not paths:
        raise FileNotFoundError(f'No original-data files found for subject {canonical_subject!r} in {input_dir}')
    records = [load_json(path) for path in paths]
    for path, record in zip(paths, records):
        if not isinstance(record.get('id'), int):
            raise ValueError(f'Missing integer id in {path}')
        if not isinstance(record.get('query'), str):
            raise ValueError(f'Missing query in {path}')
    return (canonical_subject, records)

def process(subject, model, N, K):
    BASE_DIR = Path(__file__).resolve().parent
    instruction_lib_path = BASE_DIR / 'instruction.jsonl'
    subject, raw_data = load_subject_records(BASE_DIR, subject)
    output_file = BASE_DIR / 'output_data' / f'{subject}.jsonl'
    glib, slib, gmap, i2g = load_instruction_library(instruction_lib_path)
    client = OpenAI(
        api_key="OPENAI_API_KEY",
        base_url="OPENAI_BASE_URL",
    )
    total_count = len(raw_data)
    existing_results, existing_ids = load_existing_results(output_file)
    limiter = GeneralInstructionLimiter(existing_results, glib, limit=100)
    print('\nCurrent general-instruction counts:')
    for k, v in list(limiter.distribution().items())[:10]:
        print(f'{k}: {v}')
    final_output_map = existing_results.copy()
    record_ids = [item['id'] for item in raw_data]
    tasks_to_do = [item for item in raw_data if item['id'] not in existing_ids]
    print(f'\n--- Processing subject: {subject} ---')
    print(f'Total records: {total_count}')
    print(f'Already processed: {len(existing_ids)}')
    print(f'Pending records: {len(tasks_to_do)}')

    def save_physical_reorder(data_map):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for record_id in record_ids:
                if record_id in data_map:
                    f.write(json.dumps(data_map[record_id], ensure_ascii=False) + '\n')

    def process_item(idx, item):
        answer = item.get('target') or item.get('answer') or item.get('target_answer')
        if not answer:
            answer = None
        try:
            final_q, final_meta, choose_inst = enhance_question(client, model, item['query'], answer, item.get('instruction_list', []), glib, slib, gmap, i2g, N, K, limiter)
            res = {'id': idx, 'task': item.get('task', 'Unknown_Task'), 'original_question': item['query'], 'edit_question': final_q, 'answer': answer, 'choose_instruction': choose_inst, 'instruction_list': final_meta}
            if 'image' in item and item['image']:
                res['image'] = str(Path('sample') / item['image'])
            return res
        except Exception:
            traceback.print_exc()
            return {'id': idx, 'status': 'failed_enhancement', 'original_question': item['query']}
    if not tasks_to_do:
        print('All records have already been processed.')
    else:
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_idx = {executor.submit(process_item, item['id'], item): item['id'] for item in tasks_to_do}
            pbar = tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc=f'Enhancing {subject}')
            write_counter = 0
            for future in pbar:
                idx = future_to_idx[future]
                result = future.result()
                if result:
                    final_output_map[idx] = result
                    write_counter += 1
                    if write_counter % 1 == 0:
                        save_physical_reorder(final_output_map)
        save_physical_reorder(final_output_map)
        print(f'\nProcessing complete. Results were written in ID order to: {output_file}')
        print('\n===== General Instruction Distribution =====')
        dist = limiter.distribution()
        for k, v in dist.items():
            print(f'{k}: {v}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--N', type=int, default=3, help='maximum number of general-constraint categories sampled per item')
    parser.add_argument('--K', type=int, default=3, help='maximum generation attempts for each candidate general constraint')
    args = parser.parse_args()
    process(args.subject, args.model, args.N, args.K)
