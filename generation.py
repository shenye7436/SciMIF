import json
import os
import base64
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
from openai import OpenAI
from io import BytesIO
_WORKER_CLIENT = None
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

def encode_image(image_path, max_size=(512, 512), quality=60):
    if not os.path.exists(image_path):
        return None
    try:
        if not PILLOW_AVAILABLE:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f'[image processing failed] {image_path}: {e}')
        return None

def safe_get_text(response):
    try:
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for c in content:
                    if isinstance(c, dict) and 'text' in c:
                        texts.append(c['text'])
                return ''.join(texts)
        return None
    except Exception as e:
        print(f'[Parsing failed]: {e}')
        return None

def _init_worker():
    global _WORKER_CLIENT
    _WORKER_CLIENT = OpenAI(
        api_key="OPENAI_API_KEY",
        base_url="OPENAI_BASE_URL",
    )

def resolve_image_paths(image_value, image_root):
    if not image_value:
        return []
    values = image_value if isinstance(image_value, list) else [image_value]
    image_paths = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        image_paths.append(value if os.path.isabs(value) else os.path.join(image_root, value))
    return image_paths

def _call_model_worker(prompt, image_paths, model_name):
    global _WORKER_CLIENT
    if _WORKER_CLIENT is None:
        _WORKER_CLIENT = OpenAI(
            api_key="OPENAI_API_KEY",
            base_url="OPENAI_BASE_URL",
        )
    try:
        content = [{'type': 'text', 'text': prompt}]
        for image_path in image_paths:
            base64_image = encode_image(image_path)
            if base64_image:
                content.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}})
        if len(content) > 1:
            response = _WORKER_CLIENT.chat.completions.create(model=model_name, messages=[{'role': 'user', 'content': content}], max_tokens=4096, temperature=0.7)
        else:
            response = _WORKER_CLIENT.chat.completions.create(model=model_name, messages=[{'role': 'user', 'content': prompt}], max_tokens=4096, temperature=0.7)
        text = safe_get_text(response)
        return text if text else 'failed:'
    except Exception as e:
        print(f'[model request failed]: {e}')
        return f'failed:{str(e)}'

def load_existing_ids(output_path):
    existing_ids = set()
    if not os.path.exists(output_path):
        return existing_ids
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'id' in data:
                    existing_ids.add(data['id'])
            except:
                pass
    print(f'Detected {len(existing_ids)} existing responses')
    return existing_ids

def _process_one(idx, data, model_name, image_root):
    image_paths = resolve_image_paths(data.get('image'), image_root)
    response_text = _call_model_worker(data.get('edit_question'), image_paths, model_name)
    if not response_text or response_text.startswith('failed') or 'data_inspection_failed' in str(response_text):
        print(f"\n[response rejected] id={data.get('id')}")
        print(f"question: {data.get('edit_question')[:80]}\n")
        return (idx, None)
    return (idx, {'id': data.get('id'), 'edit_question': data.get('edit_question'), 'answer': data.get('answer'), 'response': response_text, 'instruction_list': data.get('instruction_list')})

def process_jsonl(subject, workers=4, model_name=None, image_root='images', input_dir='output_data', output_dir='generation'):
    model_name = (model_name or '').strip()
    if not model_name:
        raise SystemExit('Specify the response model with --model.')
    image_root = os.path.abspath(image_root)
    input_dir = os.path.abspath(os.path.expanduser(input_dir))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    input_path = os.path.join(input_dir, f'{subject}.jsonl')
    out_dir = os.path.join(output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'{subject}.jsonl')
    existing_ids = load_existing_ids(output_path)
    if not os.path.isfile(input_path):
        print(f'[skipped] input file does not exist: {input_path}')
        return
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    lines = [l for l in lines if l['id'] not in existing_ids]
    print(f'[{subject}] pending: {len(lines)} | model={model_name!r}')
    batch_size = workers * 2
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex, open(output_path, 'a', encoding='utf-8') as fout:
        for i in tqdm(range(0, len(lines), batch_size)):
            batch = lines[i:i + batch_size]
            futures = [ex.submit(_process_one, i + j, d, model_name, image_root) for j, d in enumerate(batch)]
            for fut in as_completed(futures):
                try:
                    _, item = fut.result()
                    if item:
                        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                        fout.flush()
                except Exception as e:
                    print(f'[error]: {e}')
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='Generate model responses from output_data/{subject}.jsonl.')
    parser.add_argument('--subject', default='chemistry', help='One subject; ignored when --subjects is provided.')
    parser.add_argument('--subjects', nargs='+', default=None, metavar='NAME', help='One or more subjects, for example: chemistry physics geography life materials.')
    parser.add_argument('--workers', type=int, default=20, help='Number of parallel worker processes.')
    parser.add_argument('--model', required=True, help='Model name used for response generation.')
    parser.add_argument('--input-dir', required=True, help='Directory containing {subject}.jsonl construction outputs.')
    parser.add_argument('--output-dir', required=True, help='Root directory for generated model responses.')
    parser.add_argument('--image-root', default='images', help='Root directory containing subject/benchmark/images paths.')
    args = parser.parse_args()
    subjects = list(args.subjects) if args.subjects else [args.subject]
    for subj in subjects:
        subj = (subj or '').strip()
        if not subj:
            continue
        print(f'\n========== subject: {subj} ==========')
        process_jsonl(subj, args.workers, model_name=args.model, image_root=args.image_root, input_dir=args.input_dir, output_dir=args.output_dir)
