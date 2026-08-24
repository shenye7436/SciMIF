import json
import os
import inspect
import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm
from openai import OpenAI
JUDGE_MODEL = 'gpt-4.1'

def build_llm_client(client, model_name):

    def call(prompt: str) -> str:
        for i in range(3):
            try:
                resp = client.chat.completions.create(model=model_name, messages=[{'role': 'user', 'content': prompt}], temperature=0, max_tokens=1024)
                content = resp.choices[0].message.content
                if isinstance(content, list):
                    return ''.join([c.get('text', '') for c in content if isinstance(c, dict)])
                return content
            except Exception as e:
                if i == 2:
                    raise
                time.sleep(2 * (i + 1))
    print(f'[DEBUG] Using model: {model_name}')
    return call
SCIENCE_MAP = {'chemistry_unit_consistency': ('eval_methods.unit_matching', 'evaluate_unit_consistency'), 'physics_unit_consistency': ('eval_methods.unit_matching', 'evaluate_unit_consistency'), 'geography_unit_consistency': ('eval_methods.unit_matching', 'evaluate_unit_consistency'), 'biology_unit_consistency': ('eval_methods.unit_matching', 'evaluate_unit_consistency'), 'material_unit_consistency': ('eval_methods.unit_matching', 'evaluate_unit_consistency'), 'chemistry_molecular_format_validity': ('eval_methods.chemistry_format_validation', 'evaluate_molecular_format'), 'chemistry_entity_option_constraint': ('eval_methods.options_matching', 'evaluate_options_constraint'), 'chemistry_atom_count_constraint': ('eval_methods.chemistry_count_atom_checking', 'evaluate_atom_count'), 'chemistry_atom_bond_constraint': ('eval_methods.chemistry_count_bond_checking', 'evaluate_bond_count'), 'chemistry_atom_group_constraint': ('eval_methods.chemistry_count_group_checking', 'evaluate_group_count'), 'chemistry_method_constraint': ('eval_methods.analysis_method_checking', 'evaluate_method_constraint'), 'chemistry_reaction_steps_constraint': ('eval_methods.analysis_step_checking', 'evaluate_analysis_steps'), 'chemistry_analysis_steps_constraint': ('eval_methods.analysis_step_checking', 'evaluate_analysis_steps'), 'physics_method_constraint': ('eval_methods.analysis_method_checking', 'evaluate_method_constraint'), 'physics_analysis_steps_constraint': ('eval_methods.analysis_step_checking', 'evaluate_analysis_steps'), 'geography_method_constraint': ('eval_methods.analysis_method_checking', 'evaluate_method_constraint'), 'geography_analysis_steps_constraint': ('eval_methods.analysis_step_checking', 'evaluate_analysis_steps'), 'biology_method_constraint': ('eval_methods.analysis_method_checking', 'evaluate_method_constraint'), 'biology_analysis_steps_constraint': ('eval_methods.analysis_step_checking', 'evaluate_analysis_steps'), 'material_method_constraint': ('eval_methods.analysis_method_checking', 'evaluate_method_constraint'), 'material_analysis_steps_constraint': ('eval_methods.analysis_step_checking', 'evaluate_analysis_steps'), 'geography_address_format_validity': ('eval_methods.geography_format_geocoding_validation', 'evaluate_geography_address'), 'geography_scene_option_constraint': ('eval_methods.options_matching', 'evaluate_options_constraint'), 'biology_entity_relationship_format_validity': ('eval_methods.life_format_entity_relationship_validation', 'evaluate_entity_relationship'), 'biology_sequence_length_constraint': ('eval_methods.life_sequence_length_checking', 'evaluate_sequence_length'), 'material_characterization_technique_format_constraint': ('eval_methods.materials_format_characterization_technique_validation', 'evaluate_characterization_technique'), 'material_property_prediction_constraint': ('eval_methods.materials_property_prediction_checking', 'evaluate_property_prediction')}
LEGACY_SCIENCE_ALIASES = {'life_unit_consistency': SCIENCE_MAP['biology_unit_consistency'], 'life_method_constraint': SCIENCE_MAP['biology_method_constraint'], 'life_analysis_steps_constraint': SCIENCE_MAP['biology_analysis_steps_constraint'], 'life_entity_relationship_format_validity': SCIENCE_MAP['biology_entity_relationship_format_validity'], 'life_sequence_length_constraint': SCIENCE_MAP['biology_sequence_length_constraint'], 'materials_unit_consistency': SCIENCE_MAP['material_unit_consistency'], 'materials_method_constraint': SCIENCE_MAP['material_method_constraint'], 'materials_analysis_steps_constraint': SCIENCE_MAP['material_analysis_steps_constraint'], 'materials_characterization_technique_format_constraint': SCIENCE_MAP['material_characterization_technique_format_constraint'], 'materials_property_prediction_constraint': SCIENCE_MAP['material_property_prediction_constraint']}
GENERAL_MAP = {'general_decimal_annotation': ('eval_methods.general_checking', 'check_decimal_format'), 'general_scientific_annotation': ('eval_methods.general_checking', 'check_scientific_format'), 'general_wrap_up': ('eval_methods.general_checking', 'check_wrap_up'), 'general_all_uppercase': ('eval_methods.general_checking', 'check_uppercase'), 'general_all_lowercase': ('eval_methods.general_checking', 'check_lowercase'), 'general_json_constraint': ('eval_methods.general_checking', 'check_json_format'), 'general_list_constraint': ('eval_methods.general_checking', 'check_list_format'), 'general_tuple_constraint': ('eval_methods.general_checking', 'check_tuple_format'), 'general_dictionary_constraint': ('eval_methods.general_checking', 'check_dictionary_format'), 'general_markdown_constraint': ('eval_methods.general_checking', 'check_markdown_format'), 'general_html_constraint': ('eval_methods.general_checking', 'check_html_format'), 'general_xml_constraint': ('eval_methods.general_checking', 'check_xml_format'), 'general_csv_constraint': ('eval_methods.general_checking', 'check_csv_format'), 'general_choose_from': ('eval_methods.general_checking', 'check_choose_from'), 'general_judge': ('eval_methods.general_checking', 'check_judge'), 'general_number_response': ('eval_methods.general_checking', 'check_number_response'), 'general_response_structure': ('eval_methods.general_checking', 'check_response_structure')}
INSTRUCTION_EVALUATOR_MAP = {**SCIENCE_MAP, **LEGACY_SCIENCE_ALIASES, **GENERAL_MAP}
LLM_EXTRACTION_INSTRUCTIONS = {
    'chemistry_molecular_format_validity',
    'chemistry_atom_count_constraint',
    'chemistry_atom_bond_constraint',
    'chemistry_atom_group_constraint',
    'chemistry_method_constraint',
    'physics_method_constraint',
    'geography_method_constraint',
    'biology_method_constraint',
    'material_method_constraint',
}
SCIENCE_INSTRUCTION_NAMES = set(SCIENCE_MAP) | set(LEGACY_SCIENCE_ALIASES)

def get_evaluator(instruction_name: str):
    if instruction_name not in INSTRUCTION_EVALUATOR_MAP:
        return None
    mod_name, func_name = INSTRUCTION_EVALUATOR_MAP[instruction_name]
    mod = __import__(mod_name, fromlist=[func_name])
    return getattr(mod, func_name)

def evaluate_single_instruction(response: str, item: Dict[str, Any], instruction: Dict[str, Any], instruction_record: Dict, llm_client=None, judge_model: Optional[str]=None) -> Dict[str, Any]:
    instruction_name = instruction.get('instruction_name', '')
    required_parameters = instruction.get('required_parameters', '')
    evaluator = get_evaluator(instruction_name)
    if evaluator is None:
        return {'score': 0.0, 'detail': 'skipped', 'skipped': True}
    call_kwargs = {'required_parameters': required_parameters, 'instruction_description': instruction_record.get(instruction_name, {}).get('description', ''), 'edit_question': item.get('edit_question', ''), 'reference_answer': item.get('answer', ''), 'item': item, 'instruction_name': instruction_name}
    if 'analysis_step' in instruction_name or 'reaction_steps' in instruction_name or instruction_name == 'general_number_response' or (instruction_name in LLM_EXTRACTION_INSTRUCTIONS):
        call_kwargs['llm_client'] = llm_client
        call_kwargs['judge_model'] = judge_model
    try:
        sig = inspect.signature(evaluator)
        params = sig.parameters
        accepts_varkw = any((p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()))
        filtered_kwargs = call_kwargs if accepts_varkw else {k: v for k, v in call_kwargs.items() if k in params}
        result = evaluator(response, **filtered_kwargs)
        if not isinstance(result, dict):
            return {'score': 0.0, 'detail': f'Unexpected evaluator result type: {type(result)}'}
        score = result.get('score', 0.0)
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        return {**result, 'score': score_f, 'detail': result.get('detail', ''), 'skipped': bool(result.get('skipped', False))}
    except Exception as e:
        return {'score': 0.0, 'detail': f'error: {e}', 'skipped': False}

def evaluate_item(item: Dict[str, Any], llm_client=None, instruction_record_path: Optional[str]=None, instruction_filter: Optional[List[str]]=None, judge_model: Optional[str]=None) -> Dict[str, Any]:
    instruction_list = item.get('instruction_list', [])
    response = item.get('response', '')
    results = []
    for inst in instruction_list:
        inst_name = inst.get('instruction_name', '')
        r = evaluate_single_instruction(response, item, inst, {}, llm_client, judge_model)
        results.append({'instruction_name': inst_name, 'score': r.get('score', 0.0), 'detail': r.get('detail', ''), 'skipped': bool(r.get('skipped', False))})
        if 'casing_scope' in r:
            results[-1]['casing_scope'] = r['casing_scope']
    return {'id': item.get('id', ''), 'response': response, 'instruction_results': results}
_WORKER_CLIENT = None

def _init_worker():
    global _WORKER_CLIENT
    _WORKER_CLIENT = OpenAI(
        api_key="OPENAI_API_KEY",
        base_url="OPENAI_BASE_URL",
    )

def _eval_one(idx: int, item: Dict[str, Any], judge_model: str):
    global _WORKER_CLIENT
    llm_fn = build_llm_client(_WORKER_CLIENT, judge_model)
    eval_data = evaluate_item(item, llm_client=llm_fn, judge_model=judge_model)
    return (idx, eval_data)

def run_eval_batch(subject: str, model: str, workers: int=4):
    model = (model or '').strip()
    if not model:
        raise SystemExit('Specify the generation subdirectory with --model.')
    input_file = os.path.join('generation', model, f'{subject}.jsonl')
    out_dir = os.path.join('evaluation', model)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f'{subject}.jsonl')
    if not os.path.isfile(input_file):
        raise SystemExit(f'Input file does not exist: {input_file}')
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f]
    print(f'input: {input_file} | output: {output_file} | judge_model={JUDGE_MODEL}')
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex, open(output_file, 'w', encoding='utf-8') as fout:
        futures = [ex.submit(_eval_one, i, l, JUDGE_MODEL) for i, l in enumerate(lines)]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            _, data = fut.result()
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='Evaluate generation/{model}/{subject}.jsonl and write evaluation results.')
    parser.add_argument('--subject', default='chemistry', help='One subject; ignored when --subjects is provided.')
    parser.add_argument('--subjects', nargs='+', default=None, metavar='NAME', help='One or more subjects, for example: chemistry physics geography biology material.')
    parser.add_argument('--model', required=True, help='Generation subdirectory containing model responses.')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel evaluator processes.')
    args = parser.parse_args()
    subjects = list(args.subjects) if args.subjects else [args.subject]
    for subj in subjects:
        subj = (subj or '').strip()
        if not subj:
            continue
        print(f'\n========== subject: {subj} ==========')
        run_eval_batch(subj, args.model, workers=args.workers)
