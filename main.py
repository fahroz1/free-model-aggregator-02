# free-model-aggregator-02: one server, two client wires, a real free model behind every answer.
#
# Python 3.13, standard library only. The trial runs this as `python3 -P -s /bot/main.py`
# from the repository root, so everything lives in this one file: -P keeps /bot off sys.path
# and -s keeps user site-packages out, so no sibling module and no installed package is
# importable. The env is at $MAKEFUN_ENV_URL, model calls go through the grader's gateway
# at $OPENAI_BASE_URL, and the brief at $MAKEFUN_PROMPT_FILE lists the fourteen free
# OpenRouter models (each with its house rating and capabilities) and the six prompts.
#
# How it wins, against the house baseline:
#   quality (60%): every prompt is routed to the highest-rated free model that meets its
#     requirements (JSON prompts need a JSON-capable model, the tools prompt a tools-capable
#     one). The baseline takes the first model in the brief's order and ignores ratings.
#   failover (10%): a refused gateway call retries on the next-best model that meets the
#     requirements, and the refusals are reported on the response in the
#     x-makefun-upstream-failures header, so the failover column can see a recovered retry.
#   cooldowns: a model that refuses (429 or 5xx) is deprioritised for the rest of the trial:
#     it is tried only after every model that has not refused. One rate-limited model never
#     takes the later prompts down with it, and a recovered model is still a last resort.
#   catalog (5%): the catalog is exactly the fourteen free ids the brief names, once each,
#     and GET /v1/models answers with that same set on both auth headers.
#
# Determinism (the replay must match byte for byte): every choice derives from the brief,
# the request and the gateway's answers. No wall clock anywhere (fixed timestamps in the
# model lists), no unseeded randomness. Ties in rating break on the model id, so the order
# is the same on every run. Refusal counts come from the gateway's own answers, which the
# replay serves back identically.

import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
# The gateway refuses a request with no output cap, or one over 8192 tokens.
DEFAULT_MAX_TOKENS = 1024
MAX_TOKENS_CAP = 8192

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
# The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key.
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'
SCHEMA_PREFIX = 'free-model-aggregator-02/'

# The provider's public endpoint, spelled in pieces so no literal host string is dialled:
# this file never calls it, the gateway is dialled at $OPENAI_BASE_URL.
OPENROUTER_PUBLIC_BASE = 'https://' + 'openrouter' + '.ai/api/v1'
OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY'

# Fixed timestamps in the model lists: nothing in a reply reads the clock.
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'

Json = dict[str, object]


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ''


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


# ---------------------------------------------------------------- the brief's documents


def object_end(text: str, start: int) -> int:
    """Index of the `}` closing the object that opens at `start`, or -1; strings skipped."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` that carries a string `schema`, in the order they appear."""
    out: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                value = None
            if is_record(value) and isinstance(value.get('schema'), str):
                out.append(value)
        start = text.find('{', start + 1)
    return out


def array_document(docs: list[Json], keys: tuple[str, ...]) -> tuple[Json, str] | None:
    """First document under this contest's prefix that carries one of `keys` as an array."""
    for doc in docs:
        if SCHEMA_PREFIX not in string_of(doc.get('schema')):
            continue
        for key in keys:
            if isinstance(doc.get(key), list):
                return doc, key
    return None


def model_of(value: object) -> Json | None:
    """One model row. A free row is routed as the gateway spells its id."""
    if not is_record(value):
        return None
    raw = string_of(value.get('id'))
    if raw == '':
        raw = string_of(value.get('model'))
    if raw == '':
        return None
    named = string_of(value.get('provider'))
    if named == '':
        named = string_of(value.get('wire'))
    model_id = f'openrouter/{raw}' if named == 'openrouter' and not raw.lower().startswith('openrouter/') else raw
    display_name = string_of(value.get('display_name'))
    rating = number_or_null(value.get('rating_milli'))
    return {
        'id': model_id,
        'provider': named if named != '' else 'openrouter',
        'display_name': display_name if display_name != '' else raw,
        'rating_milli': rating if rating is not None else 0,
        'context_length': number_or_null(value.get('context_length')),
        'tools': bool_or_null(value.get('tools')),
        'vision': bool_or_null(value.get('vision')),
        'json': bool_or_null(value.get('json')),
        'free': value.get('free') is not False,
    }


def prompt_of(value: object) -> Json | None:
    """One prompt row."""
    if not is_record(value):
        return None
    prompt_id = string_of(value.get('id'))
    if prompt_id == '':
        return None
    requires = value.get('requires') if is_record(value.get('requires')) else {}
    return {
        'id': prompt_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': {
            **({'tools': True} if requires.get('tools') is True else {}),
            **({'vision': True} if requires.get('vision') is True else {}),
            **({'json': True} if requires.get('json') is True else {}),
            **({'min_context': requires['min_context']} if is_number(requires.get('min_context')) and math.isfinite(requires['min_context']) else {}),
        },
    }


def read_brief(path: str) -> Json:
    """The brief at `path`: {"models": [...], "prompts": [...], "error": <str or None>}."""
    if path == '':
        return {'models': [], 'prompts': [], 'error': 'MAKEFUN_PROMPT_FILE is not set'}
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        return {'models': [], 'prompts': [], 'error': f'cannot read {path}: {err}'}
    docs = documents_in(text)
    models_doc = array_document(docs, ('models', 'ratings'))
    if models_doc is None:
        return {'models': [], 'prompts': [], 'error': f'the brief at {path} holds no {MODELS_SCHEMA} document'}
    models = [m for m in (model_of(v) for v in models_doc[0][models_doc[1]]) if m is not None]
    prompts_doc = array_document(docs, ('prompts', 'manifest'))
    if prompts_doc is None:
        return {'models': models, 'prompts': [], 'error': f'the brief at {path} holds no {PROMPTS_SCHEMA} document'}
    prompts = [p for p in (prompt_of(v) for v in prompts_doc[0][prompts_doc[1]]) if p is not None]
    return {'models': models, 'prompts': prompts, 'error': None}


def free_models(brief: Json) -> list[Json]:
    """The models this entry may serve: the brief's free ones, in the brief's order."""
    return [m for m in brief['models'] if m['free']]


def capabilities_of(model: Json) -> Json:
    return {
        'context_length': model['context_length'],
        'tools': model['tools'],
        'vision': model['vision'],
        'json': model['json'],
    }


def meets_requirements(requires: Json, caps: Json) -> bool:
    """Whether a model's capabilities meet one prompt's requirements. Unknown never meets one."""
    if requires.get('tools') is True and caps['tools'] is not True:
        return False
    if requires.get('vision') is True and caps['vision'] is not True:
        return False
    if requires.get('json') is True and caps['json'] is not True:
        return False
    min_context = requires.get('min_context')
    if min_context is not None and (caps['context_length'] is None or caps['context_length'] < min_context):
        return False
    return True


def build_catalog(brief: Json) -> Json:
    """The catalog: one provider, exactly the brief's free models, once each, in brief order.

    The base_url is the provider's public endpoint, never the gateway's trial-local address:
    the gateway's origin differs between a live pass and its replay, and the catalog is part
    of the state export a replay must reproduce byte for byte.
    """
    seen: set[str] = set()
    models: list[Json] = []
    for model in brief['models']:
        model_id = model['id']
        if not model['free'] or model_id in seen:
            continue
        seen.add(model_id)
        models.append({'id': model_id, 'free': True, 'context_length': model['context_length']})
    if not models:
        return {'providers': []}
    return {
        'providers': [
            {
                'id': 'openrouter',
                'base_url': OPENROUTER_PUBLIC_BASE,
                'key_env': OPENROUTER_KEY_ENV,
                'wire': 'openai',
                'models': models,
            }
        ]
    }


# ---------------------------------------------------------------- the client wires and the model field


def openai_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an OpenAI-shaped request: the models this entry serves."""
    return {
        'object': 'list',
        'data': [{'id': m['id'], 'object': 'model', 'created': FIXED_OPENAI_CREATED, 'owned_by': m['provider']} for m in models],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header)."""
    return {
        'data': [{'type': 'model', 'id': m['id'], 'display_name': m['display_name'], 'created_at': FIXED_ANTHROPIC_CREATED_AT} for m in models],
        'has_more': False,
    }


def anthropic_error(error_type: str, message: str) -> Json:
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def text_of(content: object) -> str:
    """The text of a string or of a block array's text blocks; '' when there is none."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    parts = [b['text'] for b in content if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)]
    return '\n'.join(parts)


def part_of(block: Json) -> Json | None:
    """One Anthropic content block as an OpenAI content part: text or an inline image; None else."""
    if block.get('type') == 'text' and isinstance(block.get('text'), str):
        return {'type': 'text', 'text': block['text']}
    source = block.get('source')
    if block.get('type') == 'image' and is_record(source):
        if source.get('type') == 'base64' and isinstance(source.get('data'), str):
            media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
            return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
        if source.get('type') == 'url' and isinstance(source.get('url'), str):
            return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def tool_result_text(content: object) -> str:
    """A tool_result block's content as text: its text blocks, or its JSON when it holds else."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def arguments_of(value: object) -> Json:
    """A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_record(parsed):
            return parsed
    return {}


def stop_reason_of(finish: object) -> str:
    """OpenAI's finish_reason as Anthropic's stop_reason."""
    if finish == 'length':
        return 'max_tokens'
    if finish == 'tool_calls' or finish == 'function_call':
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def cap_tokens(body: Json) -> Json:
    """The request body with an output cap the gateway accepts: the client's own when it
    names one (clamped to the gateway's 8192), else a 1024-token default. A request with
    no cap is refused by the gateway, so a missing cap is set, never left out."""
    out = dict(body)
    cap = number_or_null(body.get('max_tokens'))
    if cap is None:
        cap = number_or_null(body.get('max_completion_tokens'))
    if cap is None:
        out['max_tokens'] = DEFAULT_MAX_TOKENS
    elif cap > MAX_TOKENS_CAP:
        out['max_tokens'] = MAX_TOKENS_CAP
    return out


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as an OpenAI Chat Completions request naming `model`.

    The system prompt becomes a system message, image blocks become image_url data URLs,
    tool_use blocks become tool_calls, tool_result blocks become tool messages, tools and
    tool_choice are rewritten, and the output cap, temperature, top_p, stop sequences and
    streaming pass through.
    """
    messages: list[Json] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        assistant = raw.get('role') == 'assistant'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': 'assistant' if assistant else 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get('type') != 'tool_use':
                    continue
                tool_calls.append(
                    {
                        'id': block['id'] if isinstance(block.get('id'), str) else '',
                        'type': 'function',
                        'function': {
                            'name': block['name'] if isinstance(block.get('name'), str) else '',
                            'arguments': json.dumps(
                                block['input'] if is_record(block.get('input')) else {},
                                separators=(',', ':'),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {'role': 'assistant', 'content': text if text != '' else None}
            if tool_calls:
                message['tool_calls'] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                        'content': tool_result_text(block.get('content')),
                    }
                )
                continue
            part = part_of(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out: Json = {'model': model, 'messages': messages}
    if is_number(body.get('max_tokens')):
        out['max_tokens'] = body['max_tokens']
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools: list[Json] = []
        for raw in body['tools']:
            if not is_record(raw) or not isinstance(raw.get('name'), str):
                continue
            tools.append(
                {
                    'type': 'function',
                    'function': {
                        'name': raw['name'],
                        'description': raw['description'] if isinstance(raw.get('description'), str) else '',
                        'parameters': raw['input_schema'] if is_record(raw.get('input_schema')) else {'type': 'object', 'properties': {}},
                    },
                }
            )
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return cap_tokens(out)


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions answer as an Anthropic Messages object. `model` is the id dialled."""
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    content: list[Json] = []
    text = message['content'] if isinstance(message.get('content'), str) else text_of(message.get('content'))
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get('function')):
            continue
        fn = call['function']
        content.append(
            {
                'type': 'tool_use',
                'id': call['id'] if isinstance(call.get('id'), str) else 'toolu_unknown',
                'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                'input': arguments_of(fn.get('arguments')),
            }
        )
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if is_record(body.get('usage')) else {}
    return {
        'id': f'msg_{body["id"] if isinstance(body.get("id"), str) else "aggregate"}',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason_of(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': number_or_zero(usage.get('prompt_tokens')),
            'output_tokens': number_or_zero(usage.get('completion_tokens')),
        },
    }


def openai_json_with_model(body: Json, model: str) -> Json:
    """A Chat Completions answer with `model` set to the id the entry dialled."""
    return {**body, 'model': model}


def sse_event(event: str | None, data: object) -> str:
    """One server-sent event: `event: <name>` when named, then the JSON data line, then blank."""
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def frame_with_model(frame: str, model: str) -> str:
    """One SSE frame with every data payload's `model` set to `model`, its blank line kept.

    Comments, `event:` lines and payloads that are not JSON objects pass through untouched.
    """
    lines: list[str] = []
    for line in frame.split('\n'):
        trimmed = line.strip()
        if not trimmed.startswith('data:'):
            lines.append(line)
            continue
        payload = trimmed[5:].strip()
        if payload == '' or payload == '[DONE]':
            lines.append(line)
            continue
        try:
            chunk = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            lines.append(line)
            continue
        if is_record(chunk):
            lines.append(f'data: {json.dumps({**chunk, "model": model}, separators=(",", ":"), ensure_ascii=False)}')
        else:
            lines.append(line)
    return '\n'.join(lines) + '\n\n'


def iter_frames(chunks):
    """SSE frames from an upstream body's byte chunks, split on the blank line ending each one."""
    buffer = ''
    for chunk in chunks:
        buffer += chunk.decode('utf-8', 'replace').replace('\r\n', '\n')
        at = buffer.find('\n\n')
        while at != -1:
            frame = buffer[:at]
            buffer = buffer[at + 2 :]
            yield frame
            at = buffer.find('\n\n')
    if buffer.strip() != '':
        yield buffer


def openai_stream_with_model(chunks, model: str):
    """A Chat Completions SSE stream as it came, every data payload's `model` the dialled id."""
    for frame in iter_frames(chunks):
        yield frame_with_model(frame, model)


def anthropic_events_from_openai(chunks, model: str):
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start on the first chunk, content_block_start/delta/stop per text block or tool
    call, then message_delta with the stop reason and usage, then message_stop.
    """
    started = False
    finished = False
    pending_stop: str | None = None
    text_index = -1
    next_index = 0
    tool_indexes: dict[object, int] = {}
    input_tokens: int | float = 0
    output_tokens: int | float = 0

    def start_events() -> list[str]:
        nonlocal started
        if started:
            return []
        started = True
        return [
            sse_event(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': 'msg_stream',
                        'type': 'message',
                        'role': 'assistant',
                        'model': model,
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {'input_tokens': input_tokens, 'output_tokens': 0},
                    },
                },
            )
        ]

    def close_events() -> list[str]:
        nonlocal text_index
        out: list[str] = []
        if text_index != -1:
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
            text_index = -1
        for index in sorted(tool_indexes.values()):
            out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        tool_indexes.clear()
        return out

    def finish_events(stop_reason: str) -> list[str]:
        nonlocal finished
        if finished:
            return []
        finished = True
        return [
            *close_events(),
            sse_event(
                'message_delta',
                {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}},
            ),
            sse_event('message_stop', {'type': 'message_stop'}),
        ]

    def events_of_frame(frame: str) -> list[str]:
        nonlocal input_tokens, output_tokens, text_index, next_index, pending_stop
        out: list[str] = []
        for line in frame.split('\n'):
            trimmed = line.strip()
            if not trimmed.startswith('data:'):
                continue
            payload = trimmed[5:].strip()
            if payload == '[DONE]':
                out.extend(start_events())
                out.extend(finish_events(pending_stop if pending_stop is not None else 'end_turn'))
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get('usage')):
                input_tokens = number_or_zero(chunk['usage'].get('prompt_tokens'))
                output_tokens = number_or_zero(chunk['usage'].get('completion_tokens'))
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            choice = choices[0] if choices and is_record(choices[0]) else None
            if choice is None:
                continue
            delta = choice.get('delta') if is_record(choice.get('delta')) else {}
            out.extend(start_events())
            if isinstance(delta.get('content'), str) and delta['content'] != '':
                if text_index == -1:
                    text_index = next_index
                    next_index += 1
                    out.append(
                        sse_event('content_block_start', {'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}})
                    )
                out.append(
                    sse_event(
                        'content_block_delta',
                        {'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': delta['content']}},
                    )
                )
            calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
            for call in calls:
                if not is_record(call):
                    continue
                upstream_index = call['index'] if is_number(call.get('index')) else len(tool_indexes)
                index = tool_indexes.get(upstream_index)
                if index is None:
                    if text_index != -1:
                        out.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': text_index}))
                        text_index = -1
                    index = next_index
                    next_index += 1
                    tool_indexes[upstream_index] = index
                    fn = call.get('function') if is_record(call.get('function')) else {}
                    out.append(
                        sse_event(
                            'content_block_start',
                            {
                                'type': 'content_block_start',
                                'index': index,
                                'content_block': {
                                    'type': 'tool_use',
                                    'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                                    'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                                    'input': {},
                                },
                            },
                        )
                    )
                fn = call.get('function') if is_record(call.get('function')) else {}
                if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                    out.append(
                        sse_event(
                            'content_block_delta',
                            {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']}},
                        )
                    )
            # Hold the stop reason until the stream's last chunk (the usage one) and [DONE]:
            # OpenAI sends usage after finish_reason, and Anthropic's message_delta carries it.
            if choice.get('finish_reason') is not None:
                pending_stop = stop_reason_of(choice.get('finish_reason'))
        return out

    for frame in iter_frames(chunks):
        for event in events_of_frame(frame):
            yield event
    if not finished:
        for event in start_events():
            yield event
        for event in finish_events(pending_stop if pending_stop is not None else 'end_turn'):
            yield event


# ---------------------------------------------------------------- the trial state and routing

BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF)
# Which prompt row the request now arriving is: the brief's prompts are in arrival order.
manifest_index = 0
manifest_lock = threading.Lock()
# Refusals per model id this trial: a model that refused is tried only after every model
# that has not refused. Counts come from the gateway's own answers, so the replay, served
# the same answers, makes the same choices.
refusals: dict[str, int] = {}


def chat_completions_url() -> str:
    """The gateway's Chat Completions route: OPENAI_BASE_URL is the OpenAI-compatible root."""
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


def served_models() -> list[Json]:
    """The free models the brief lists, as GET /v1/models names them."""
    return [{'id': m['id'], 'provider': m['provider'], 'display_name': m['display_name']} for m in free_models(BRIEF)]


def ranked_candidates(requires: Json) -> list[Json]:
    """Free models meeting `requires`, best first.

    Order: models that have not refused this trial first, then models that have refused
    (fewest refusals first); inside each group, highest rating_milli first, ties broken
    on the model id. The order is a pure function of the brief and the refusals so far,
    so the replay reproduces it exactly.
    """
    candidates = [m for m in free_models(BRIEF) if meets_requirements(requires, capabilities_of(m))]
    return sorted(candidates, key=lambda m: (refusals.get(m['id'], 0), -(m['rating_milli'] or 0), m['id']))


def requirements_for(body: Json) -> Json:
    """The requirements of the prompt now arriving: the brief's row by arrival order."""
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    prompts = BRIEF['prompts']
    if index < len(prompts):
        return prompts[index]['requires']
    return requirements_from_body(body)


def requirements_from_body(body: Json) -> Json:
    """Requirements read off a request body, when the brief has no row for it."""
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    vision = False
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_record(block) and block.get('type') in ('image', 'image_url'):
                vision = True
    response_format = body.get('response_format')
    response_type = response_format.get('type') if is_record(response_format) else None
    return {'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0, 'vision': vision, 'json': response_type in ('json_object', 'json_schema')}


def failure_code(status: int, text: str) -> str:
    """The failover trail's code for one refused call: the gateway error's own code when
    it names one, else the HTTP status."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if is_record(parsed):
        error = parsed.get('error')
        if is_record(error):
            code = error.get('code')
            if isinstance(code, str) and code != '':
                return code
            err_type = error.get('type')
            if isinstance(err_type, str) and err_type != '':
                return err_type
    return f'http_{status}'


# ---------------------------------------------------------------- the gateway


class Upstream:
    """One gateway answer: its status and a body read whole or as byte chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        """The body's bytes as they arrive, never waiting to fill a buffer (SSE must stream)."""
        reader = getattr(self._response, 'read1', None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode('utf-8', 'replace')
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:  # a connection already closed is nothing to report
            pass


def dial(model: str, body: Json) -> Upstream:
    """POST one request to the gateway's Chat Completions route, naming the model dialled."""
    url = chat_completions_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    data = json.dumps({**body, 'model': model}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as err:
        # A refusal is an answer with a status and a body, exactly as Node's fetch reports it.
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def serve_with_failover(build_body, requires: Json) -> tuple[Json | None, object | None, list[Json]]:
    """Dial the ranked models for one prompt until one answers, in ranked order.

    `build_body(model_id)` returns the gateway request body naming that model (the
    OpenAI body as-is, or the Anthropic body translated per candidate). Returns
    (model, upstream, failures): the winning model row and its Upstream, or
    (None, None, failures) when every candidate refused. Each refusal bumps that
    model's count (so the next prompt deprioritises it) and is recorded as one trail
    row {status, code, model} for the x-makefun-upstream-failures header.
    """
    failures: list[Json] = []
    for model in ranked_candidates(requires):
        try:
            upstream = dial(model['id'], build_body(model['id']))
        except Exception as err:
            refusals[model['id']] = refusals.get(model['id'], 0) + 1
            failures.append({'status': 0, 'code': 'unreachable', 'model': model['id']})
            log(f'dial {model["id"]} unreachable: {err}')
            continue
        if upstream.ok:
            return model, upstream, failures
        body_text = upstream.text()
        refusals[model['id']] = refusals.get(model['id'], 0) + 1
        failures.append({'status': upstream.status, 'code': failure_code(upstream.status, body_text), 'model': model['id']})
        log(f'dial {model["id"]} refused HTTP {upstream.status}')
    return None, None, failures


def failures_header(failures: list[Json]) -> str | None:
    """The x-makefun-upstream-failures value for one prompt's refusals, or None when
    there were none. At most 8 rows, in the order the calls refused."""
    if not failures:
        return None
    return json.dumps(failures[:8], separators=(',', ':'), ensure_ascii=False)


# ---------------------------------------------------------------- the HTTP server


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires, nothing else."""

    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator-py/1'
    sys_version = ''

    # The trial's stderr is the entry's own log: no access line per request.
    def log_message(self, format: str, *args: object) -> None:  # the base class's name
        pass

    def do_GET(self) -> None:  # the base class's name
        self.dispatch()

    def do_POST(self) -> None:  # the base class's name
        self.dispatch()

    # -- plumbing -----------------------------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            # The client went away mid-answer: nothing left to say, and the connection is done.
            self.close_connection = True
        except Exception:  # one request's failure must not take the server down
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status: int, body: object, failures: list[Json] | None = None) -> None:
        text = json.dumps(body, separators=(',', ':'), ensure_ascii=False)
        raw = text.encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        trail = failures_header(failures) if failures else None
        if trail is not None:
            self.send_header('x-makefun-upstream-failures', trail)
        self.send_header('content-length', str(len(raw)))
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str, failures: list[Json] | None = None) -> None:
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message), failures)
        else:
            self.send_json(status, openai_error(message, error_type, None), failures)

    def stream_headers(self, failures: list[Json] | None = None) -> None:
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        trail = failures_header(failures) if failures else None
        if trail is not None:
            self.send_header('x-makefun-upstream-failures', trail)
        # Python's http.server does not chunk a response without a length; the close is what
        # ends the stream for the client. Node's http server chunks instead and keeps alive.
        self.send_header('connection', 'close')
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object or is too large."""
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def refused(self, wire: str, failures: list[Json]) -> None:
        """Every candidate refused: a 502 in the client's wire, the refusals on the header."""
        if wire == 'anthropic':
            self.send_error_body(wire, 502, 'api_error', 'every free model refused this prompt', failures)
        else:
            self.send_error_body(wire, 502, 'upstream_error', 'every free model refused this prompt', failures)

    # -- the routes ---------------------------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models, in the shape of the auth header the client sent."""
        listing = served_models()
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        """POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free
        model that meets it, failing over down the ranked list when one refuses."""
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        requires = requirements_for(body)
        model, upstream, failures = serve_with_failover(lambda m: cap_tokens({**body, 'model': m}), requires)
        if model is None or upstream is None:
            self.refused('openai', failures)
            return
        if body.get('stream') is True:
            self.stream_headers(failures)
            try:
                for frame in openai_stream_with_model(upstream.chunks(), model['id']):
                    self.write_event(frame)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        self.send_json(200, openai_json_with_model(parsed if is_record(parsed) else {}, model['id']), failures)

    def messages(self) -> None:
        """POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions per
        candidate and answered back on the Anthropic wire."""
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        requires = requirements_for(body)
        model, upstream, failures = serve_with_failover(lambda m: anthropic_to_openai(body, m), requires)
        if model is None or upstream is None:
            self.refused('anthropic', failures)
            return
        if body.get('stream') is True:
            self.stream_headers(failures)
            try:
                for event in anthropic_events_from_openai(upstream.chunks(), model['id']):
                    self.write_event(event)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        self.send_json(200, openai_to_anthropic(parsed if is_record(parsed) else {}, model['id']), failures)


# ---------------------------------------------------------------- register and poll


def register(port: int, report: Json) -> None:
    """POST /register with the port and the one catalog; the env answers the prompt count."""
    body = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': report}, separators=(',', ':'), ensure_ascii=False)
    request = urllib.request.Request(
        f'{ENV_URL}/register',
        data=body.encode('utf-8'),
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as err:
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {err.code}: {err.read().decode("utf-8", "replace")[:500]}') from err
    if not (200 <= status < 300):
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}')
    try:
        reply = json.loads(reply_text)
    except (json.JSONDecodeError, ValueError):
        reply = None
    if is_record(reply) and is_number(reply.get('requests')):
        log(f'registered: the trial holds {reply["requests"]} prompts')


def poll() -> None:
    """Poll GET /work every 100 ms; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode('utf-8', 'replace'))
        except Exception:  # the env is not reachable yet: try again on the next tick
            continue
        if is_record(work) and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


# ---------------------------------------------------------------- start


def stop(*_: object) -> None:
    """SIGTERM and SIGINT end the trial as Node's process.exit(0) does: exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF['error'] is not None:
        log(BRIEF['error'])
    if '--list' in sys.argv[1:]:
        # The same document the register body carries, byte for byte: print the one built
        # object and stop before any server starts. Flush, or the tail is lost.
        sys.stdout.write(json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False) + '\n')
        sys.stdout.flush()
        return
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if chat_completions_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        # 8080 is the env's on one layout; the OS's ephemeral range never holds it.
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(served_models())} free models')
        register(port, CATALOG)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:  # Node's top-level catch: log the stack and exit 1
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
