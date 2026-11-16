"""Local trial test for the aggregator: mock env + mock gateway, drives the 6 prompts
from the kit's practice/trial-0 workload and checks every wire.

Run: python3 test_trial.py
Expects: main.py in the same dir, kit at ../free-model-aggregator-02-kit/free-model-aggregator-02-kit
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KIT = os.path.expanduser('~/workspace/makefun/free-model-aggregator-02-kit/free-model-aggregator-02-kit')
WORKLOAD = json.load(open(f'{KIT}/practice/trial-0/input/env/workload.json'))
PROMPT_FILE = f'{KIT}/practice/trial-0/input/prompt.txt'
TOKEN_RE = re.compile(r'[A-Z]+-\d+')

# Mock gateway: 429 the first call to inkling-2-small (failover test), else answer 200.
gateway_calls = []
fail_once = {'openrouter/thinkingmachines/inkling-2-small:free': 1}


def provider_model(dialled: str) -> str:
    """The provider's own spelling: the dialled id without the gateway's openrouter/ prefix."""
    return dialled[len('openrouter/'):] if dialled.startswith('openrouter/') else dialled


def token_of(body: dict) -> str:
    text = json.dumps(body)
    m = TOKEN_RE.search(text)
    return m.group(0) if m else 'TOKEN'


class MockGateway(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get('content-length', 0))
        body = json.loads(self.rfile.read(length))
        model = body.get('model', '')
        gateway_calls.append(model)
        if fail_once.get(model, 0) > 0:
            fail_once[model] -= 1
            err = json.dumps({'error': {'message': 'rate limited', 'type': 'rate_limit_error', 'code': 'rate_limit_exceeded'}}).encode()
            self.send_response(429)
            self.send_header('content-type', 'application/json')
            self.send_header('content-length', str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return
        token = token_of(body)
        is_json = isinstance(body.get('response_format'), dict)
        content = json.dumps({'token': token}) if is_json else token
        prov = provider_model(model)
        if body.get('stream') is True:
            self.send_response(200)
            self.send_header('content-type', 'text/event-stream')
            self.send_header('cache-control', 'no-cache')
            self.send_header('connection', 'close')
            self.end_headers()
            self.close_connection = True
            # Split the content into two deltas, then a usage chunk, then [DONE].
            half = len(content) // 2
            for part in (content[:half], content[half:]):
                chunk = {'id': 'chatcmpl-mock', 'object': 'chat.completion.chunk', 'created': 1,
                         'model': prov, 'choices': [{'index': 0, 'delta': {'content': part}, 'finish_reason': None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            usage_chunk = {'id': 'chatcmpl-mock', 'object': 'chat.completion.chunk', 'created': 1,
                           'model': prov, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                           'usage': {'prompt_tokens': 20, 'completion_tokens': 5, 'total_tokens': 25}}
            self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
            self.wfile.write(b'data: [DONE]\n\n')
            return
        answer = {'id': 'chatcmpl-mock', 'object': 'chat.completion', 'created': 1, 'model': prov,
                  'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
                  'usage': {'prompt_tokens': 20, 'completion_tokens': 5, 'total_tokens': 25}}
        raw = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class MockEnv(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    registered = {}
    sent = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get('content-length', 0))
        body = json.loads(self.rfile.read(length))
        MockEnv.registered = body
        raw = json.dumps({'ok': True, 'requests': 6}).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        done = MockEnv.sent >= 6
        raw = json.dumps({'done': done, 'next': None, 'pending': max(0, 6 - MockEnv.sent)}).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def http(method, url, body=None, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'content-type': 'application/json', **(headers or {})}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read().decode(), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode(), dict(err.headers)


def main():
    gw = ThreadingHTTPServer(('127.0.0.1', 0), MockGateway)
    env = ThreadingHTTPServer(('127.0.0.1', 0), MockEnv)
    gw_port, env_port = gw.server_address[1], env.server_address[1]
    threading.Thread(target=gw.serve_forever, daemon=True).start()
    threading.Thread(target=env.serve_forever, daemon=True).start()

    env_vars = dict(os.environ)
    env_vars.update({
        'MAKEFUN_ENV_URL': f'http://127.0.0.1:{env_port}',
        'MAKEFUN_PROMPT_FILE': PROMPT_FILE,
        'OPENAI_BASE_URL': f'http://127.0.0.1:{gw_port}/v1',
        'OPENAI_API_KEY': 'makefun-trial-test',
        'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{gw_port}',
        'ANTHROPIC_API_KEY': 'makefun-trial-test',
    })
    proc = subprocess.Popen([sys.executable, 'main.py'], env=env_vars,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Wait for registration.
    for _ in range(50):
        if MockEnv.registered:
            break
        time.sleep(0.1)
    assert MockEnv.registered, 'server never registered'
    catalog = MockEnv.registered['catalog']
    cat_ids = [m['id'] for m in catalog['providers'][0]['models']]
    print(f'registered port {MockEnv.registered["port"]}, catalog models: {len(cat_ids)}')
    assert len(cat_ids) == 14, f'catalog has {len(cat_ids)} models, want 14'
    base = f'http://127.0.0.1:{MockEnv.registered["port"]}'

    # GET /v1/models on both wires.
    status, text, _ = http('GET', f'{base}/v1/models', headers={'Authorization': 'Bearer x'})
    data = json.loads(text)
    assert status == 200 and [m['id'] for m in data['data']] == cat_ids, 'OpenAI model list mismatch'
    status, text, _ = http('GET', f'{base}/v1/models', headers={'x-api-key': 'x', 'anthropic-version': '2023-06-01'})
    data = json.loads(text)
    assert status == 200 and [m['id'] for m in data['data']] == cat_ids, 'Anthropic model list mismatch'
    print('GET /v1/models: both wires list the 14 ids OK')

    # Drive the 6 prompts in manifest order.
    results = []
    for item in WORKLOAD['manifest']:
        pid = item['id']
        body = WORKLOAD['bodies'][pid]
        path = '/v1/chat/completions' if item['api'] == 'openai' else '/v1/messages'
        headers = {'Authorization': 'Bearer x'} if item['api'] == 'openai' else {'x-api-key': 'x', 'anthropic-version': '2023-06-01'}
        status, text, hdrs = http('POST', f'{base}{path}', body=body, headers=headers)
        MockEnv.sent += 1
        token = item['token']
        if item['stream']:
            # Streaming: the token may be split across deltas; reassemble the text first.
            pieces = []
            for m in re.finditer(r'"(?:content|text)":\s*"((?:[^"\\]|\\.)*)"', text):
                try:
                    pieces.append(json.loads(f'"{m.group(1)}"'))
                except Exception:
                    pass
            ok = status == 200 and token in ''.join(pieces)
        else:
            ok = status == 200 and token in text
        model = ''
        if item['api'] == 'openai':
            try:
                model = json.loads(text).get('model', '')
            except Exception:
                # streaming: model appears on chunks
                m = re.search(r'"model":\s*"([^"]+)"', text)
                model = m.group(1) if m else ''
        else:
            try:
                model = json.loads(text).get('model', '')
            except Exception:
                m = re.search(r'"model":\s*"([^"]+)"', text)
                model = m.group(1) if m else ''
        trail = hdrs.get('x-makefun-upstream-failures')
        results.append((pid, item['api'], item['stream'], status, ok, model, trail))
        print(f'{pid} {item["api"]} stream={item["stream"]} status={status} token_ok={ok} model={model} trail={trail}')

    # The poll loop should exit the server once work is done.
    try:
        proc.wait(timeout=5)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        exit_code = 'STILL RUNNING (poll did not exit)'
    print(f'server exit code: {exit_code}')

    print(f'gateway calls ({len(gateway_calls)}):')
    for c in gateway_calls:
        print('  ', c)
    # Failover check: inkling-2-small was 429ed once; some prompt must have been served
    # by the next-best model, and the trail header must name the refusal.
    assert any(r[5] == 'openrouter/nvidia/nemotron-3.5-super-vl:free' for r in results), 'expected a super-vl serve after the inkling 429'
    assert any(r[6] for r in results), 'expected an x-makefun-upstream-failures header on a recovered prompt'
    assert all(r[4] for r in results), 'not every prompt carried its token back'
    assert exit_code == 0, f'server did not exit 0 (got {exit_code})'
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
