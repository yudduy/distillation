import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate as e
import select_dataset as selection
from llm import LlmResponse

TEMPLATE = '{{equation1}} {{ equation2 }}'


def uncapped_env(**changes):
    env={'DISTILL_LIVE':'1','OPENROUTER_API_KEY':'test','DISTILL_ALLOW_UNCAPPED':'1',
         'GITHUB_ACTIONS':'true','GITHUB_REPOSITORY':'yudduy/distillation',
         'GITHUB_WORKFLOW_REF':'yudduy/distillation/.github/workflows/benchmark.yml@refs/heads/main'}
    env.update(changes)
    return env


def response(text='VERDICT: TRUE', **kw):
    return LlmResponse(text=text, finish_reason='stop', **kw)


def test_vendor_pin():
    e.verify_vendor()


@pytest.mark.parametrize('suffix,valid', [('a'*(10240-len(TEMPLATE)), True), ('a'*(10241-len(TEMPLATE)), False), ('é'*5107, False)])
def test_prompt_byte_boundary(tmp_path, suffix, valid):
    path = tmp_path / 'prompt.txt'
    path.write_text(TEMPLATE + suffix)
    if valid:
        assert len(e.read_prompt(path)[1]) == 10240
    else:
        with pytest.raises(e.InvalidRun): e.read_prompt(path)


@pytest.mark.parametrize('data', [b'', b'{{equation1}}', b'{equation1} {equation2}', b'\xff'])
def test_prompt_invalid(tmp_path, data):
    path = tmp_path / 'prompt.txt'; path.write_bytes(data)
    with pytest.raises(e.InvalidRun): e.read_prompt(path)


def test_prompt_symlinks(tmp_path):
    target = tmp_path / 'real'; target.mkdir(); (target / 'p').write_text(TEMPLATE)
    link = tmp_path / 'linked'; link.symlink_to(target, target_is_directory=True)
    with pytest.raises(e.InvalidRun): e.read_prompt(link / 'p')
    file = tmp_path / 'file'; file.symlink_to(target / 'p')
    with pytest.raises(e.InvalidRun): e.read_prompt(file)
    with pytest.raises(e.InvalidRun): e.read_prompt(tmp_path / 'missing')


@pytest.mark.asyncio
async def test_600_outcomes_and_answer_failures():
    rows = [{'equation1': str(i), 'equation2': 'x=x', 'answer': i < 100} for i in range(200)]
    active = peak = 0
    async def complete(alias, prompt):
        nonlocal active, peak
        active += 1; peak = max(peak, active); await asyncio.sleep(0); active -= 1
        return response('VERDICT: TRUE' if alias == e.MODEL_IDS[0] else 'unclear')
    result = await e.evaluate(rows, TEMPLATE, complete)
    assert result['score'] == 100 / 600
    assert result['metrics']['outcomes'] == 600
    assert result['metrics']['parseFailures'] == 400
    assert peak == 6
    assert result['metrics']['models'][e.MODEL_IDS[0]]['accuracy'] == .5


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['error', 'deadline', 'finish'])
async def test_infrastructure_failure_cancels(failure):
    async def complete(*_):
        if failure == 'error': raise RuntimeError('SECRET PRIVATE QUESTION')
        if failure == 'deadline': await asyncio.sleep(10)
        return LlmResponse(text='', finish_reason='error')
    with pytest.raises((RuntimeError, asyncio.TimeoutError, e.InvalidRun)):
        await e.evaluate([{'equation1':'a','equation2':'b','answer':True}], TEMPLATE, complete, timeout=.01)


@pytest.mark.asyncio
async def test_refusal_is_wrong():
    async def complete(*_): return response('VERDICT: TRUE', refusal='refused')
    result = await e.evaluate([{'equation1':'a','equation2':'b','answer':True}], TEMPLATE, complete)
    assert result['score'] == 0


@pytest.mark.parametrize('env', [{}, {'DISTILL_LIVE':'1'}, {'DISTILL_LIVE':'1','OPENROUTER_API_KEY':'test','DISTILL_MAX_SPEND_USD':'nan'}, {'DISTILL_LIVE':'1','OPENROUTER_API_KEY':'test','DISTILL_MAX_SPEND_USD':'-1'}])
def test_no_implicit_spend(env):
    with pytest.raises(e.InvalidRun): e.live_settings(env)


@pytest.mark.parametrize('change', [{'limit':None},{'limit':11},{'limit_remaining':None},{'limit_remaining':0},{'limit_reset':'daily'},{'include_byok_in_limit':False}])
def test_provider_budget_fail_closed(change):
    data={'limit':10,'limit_remaining':9,'limit_reset':None,'include_byok_in_limit':True}
    e.check_key_budget(data,10)
    with pytest.raises(e.InvalidRun): e.check_key_budget({**data,**change},10)


def test_uncapped_mode_is_limited_to_operator_workflow():
    key, cap = e.live_settings(uncapped_env())
    assert key == 'test' and cap.is_infinite()
    for changes in (
        {'GITHUB_ACTIONS':'false'},
        {'GITHUB_REPOSITORY':'someone/else'},
        {'GITHUB_WORKFLOW_REF':'yudduy/distillation/.github/workflows/other.yml@refs/heads/main'},
    ):
        with pytest.raises(e.InvalidRun, match='uncapped_hosted_context_invalid'):
            e.live_settings(uncapped_env(**changes))


@pytest.mark.asyncio
async def test_uncapped_hosted_run_skips_key_cap_but_requires_non_byok_usage(monkeypatch, tmp_path):
    monkeypatch.setattr(e, 'ROOT', tmp_path)
    requests = 0
    async def handler(req):
        nonlocal requests
        if req.url.path.endswith('/key'):
            # This key metadata would be rejected by ordinary capped mode.
            return httpx.Response(200, json={'data': {'limit': None, 'limit_remaining': None}})
        requests += 1
        body = json.loads(req.content)
        provider = 'Novita' if body['model'].startswith('google/') else 'DeepInfra'
        return httpx.Response(200, json={
            'provider': provider,
            'usage': {'cost': '0.001', 'prompt_tokens': 1, 'completion_tokens': 1,
                      'is_byok': False},
            'choices': [{'finish_reason': 'stop', 'message': {'content': 'VERDICT: TRUE'}}],
        })
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(handler))
    result = await e.live_evaluate(
        [{'equation1':'a','equation2':'b','answer':True}], TEMPLATE, uncapped_env())
    assert result['score'] == 1 and requests == 3
    assert result['metrics']['billing']['confirmedSpendUsd'] == '0.003'
    artifact = json.loads((tmp_path/'evaluation-costs.json').read_text())
    assert artifact['status'] == 'succeeded' and artifact['billing'] == result['metrics']['billing']


@pytest.mark.asyncio
@pytest.mark.parametrize('is_byok', [True, None])
async def test_uncapped_hosted_run_rejects_byok_or_unreported_billing(monkeypatch, tmp_path, is_byok):
    monkeypatch.setattr(e, 'ROOT', tmp_path)
    async def handler(req):
        if req.url.path.endswith('/key'):
            return httpx.Response(200, json={'data': {}})
        body = json.loads(req.content)
        provider = 'Novita' if body['model'].startswith('google/') else 'DeepInfra'
        usage = {'cost': '0.001', 'prompt_tokens': 1, 'completion_tokens': 1}
        if is_byok is not None:
            usage['is_byok'] = is_byok
        return httpx.Response(200, json={
            'provider': provider,
            'usage': usage,
            'choices': [{'finish_reason': 'stop', 'message': {'content': 'VERDICT: TRUE'}}],
        })
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(handler))
    with pytest.raises(e.InvalidRun, match='hosted_billing_mode_invalid'):
        await e.live_evaluate(
            [{'equation1':'a','equation2':'b','answer':True}], TEMPLATE, uncapped_env())


@pytest.mark.asyncio
async def test_real_client_routing_and_8192_cap(monkeypatch):
    requests=[]
    async def handler(req):
        if req.url.path.endswith('/key'):
            return httpx.Response(200,json={'data':{'limit':1,'limit_remaining':1,'limit_reset':None,'include_byok_in_limit':True}})
        body=json.loads(req.content); requests.append(body)
        provider='Novita' if body['model'].startswith('google/') else 'DeepInfra'
        return httpx.Response(200,json={'provider':provider,'usage':{'cost':'0.001'},'choices':[{'finish_reason':'stop','message':{'content':'VERDICT: TRUE'}}]})
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(handler))
    result=await e.live_evaluate([{'equation1':'a','equation2':'b','answer':True}],TEMPLATE,{'DISTILL_LIVE':'1','OPENROUTER_API_KEY':'test','DISTILL_MAX_SPEND_USD':'1'})
    assert result['score']==1
    assert len(requests)==3
    assert {r['model'] for r in requests} == {'openai/gpt-oss-120b','meta-llama/llama-3.3-70b-instruct','google/gemma-4-31b-it'}
    for r in requests:
        assert r['max_tokens']==8192 and r['temperature']==0 and r['seed']==0
        assert r['provider']['allow_fallbacks'] is False
        assert r['provider']['require_parameters'] is True
        assert r['provider']['data_collection']=='deny' and r['provider']['zdr'] is True
        assert r['provider']['max_price']['request']==0
        assert len(r['messages'])==1 and r['messages'][0]['role']=='user'
    by_model={r['model']:r for r in requests}
    assert by_model['openai/gpt-oss-120b']['provider']['order']==['deepinfra/turbo']
    assert by_model['openai/gpt-oss-120b']['provider']['quantizations']==['bf16']
    assert by_model['openai/gpt-oss-120b']['provider']['max_price']=={'prompt':.15,'completion':.6,'request':0}
    assert by_model['openai/gpt-oss-120b']['reasoning']=={'effort':'low'}
    assert by_model['meta-llama/llama-3.3-70b-instruct']['provider']['order']==['deepinfra/turbo']
    assert by_model['meta-llama/llama-3.3-70b-instruct']['provider']['quantizations']==['fp8']
    assert by_model['meta-llama/llama-3.3-70b-instruct']['provider']['max_price']=={'prompt':.1,'completion':.32,'request':0}
    assert 'reasoning' not in by_model['meta-llama/llama-3.3-70b-instruct']
    assert by_model['google/gemma-4-31b-it']['provider']['order']==['novita/bf16']
    assert by_model['google/gemma-4-31b-it']['provider']['quantizations']==['bf16']
    assert by_model['google/gemma-4-31b-it']['provider']['max_price']=={'prompt':.14,'completion':.4,'request':0}
    assert by_model['google/gemma-4-31b-it']['reasoning']=={'effort':'none'}
    billing=result['metrics']['billing']
    assert {key:billing[key] for key in ('attempts','confirmedSpendUsd','unresolvedReservedUsd')}=={
        'attempts':3,'confirmedSpendUsd':'0.003','unresolvedReservedUsd':'0'}
    for alias in e.MODEL_IDS:
        assert billing['models'][alias]['attempts']==1
        assert billing['models'][alias]['costReportedAttempts']==1
        assert billing['models'][alias]['httpStatusCounts']=={'200':1}


@pytest.mark.asyncio
async def test_stale_score_removed_without_credentials(tmp_path,monkeypatch):
    monkeypatch.setattr(e,'ROOT',tmp_path)
    (tmp_path/'score.json').write_text('{"score":1}')
    (tmp_path/'evaluation-costs.json').write_text('{"status":"stale"}')
    (tmp_path/'evaluation-costs.json.tmp').write_text('stale')
    (tmp_path/'submission').mkdir(); (tmp_path/'submission/prompt.txt').write_text(TEMPLATE)
    with pytest.raises(e.InvalidRun): await e.run(SimpleNamespace(ranked=False),{})
    assert not (tmp_path/'score.json').exists()
    assert not (tmp_path/'evaluation-costs.json').exists()
    assert not (tmp_path/'evaluation-costs.json.tmp').exists()


@pytest.mark.asyncio
async def test_failed_hosted_evaluation_persists_sanitized_costs_without_score(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'ROOT', tmp_path)
    (tmp_path/'submission').mkdir()
    (tmp_path/'submission/prompt.txt').write_text(TEMPLATE)
    async def handler(req):
        if req.url.path.endswith('/key'):
            return httpx.Response(200, json={'data': {}})
        return httpx.Response(200, json={
            'provider': 'PRIVATE WRONG PROVIDER',
            'usage': {'cost':'0.001', 'prompt_tokens':1, 'completion_tokens':1,
                      'is_byok':False},
            'choices':[{'finish_reason':'stop', 'message':{'content':'PRIVATE RESPONSE'}}],
        })
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(handler))
    with pytest.raises(e.InvalidRun, match='provider_route_mismatch'):
        await e.run(SimpleNamespace(ranked=False), uncapped_env())
    assert not (tmp_path/'score.json').exists()
    serialized = (tmp_path/'evaluation-costs.json').read_text()
    artifact = json.loads(serialized)
    assert set(artifact) == {'contractVersion', 'status', 'billing'}
    assert artifact['status'] == 'failed' and artifact['billing']['attempts'] > 0
    assert 'PRIVATE' not in serialized and 'attemptId' not in serialized


@pytest.mark.asyncio
async def test_hosted_progress_is_coarse_and_sanitized(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(e, 'ROOT', tmp_path)
    async def handler(req):
        if req.url.path.endswith('/key'):
            return httpx.Response(200, json={'data': {}})
        body = json.loads(req.content)
        provider = 'Novita' if body['model'].startswith('google/') else 'DeepInfra'
        return httpx.Response(200, json={
            'provider':provider,
            'usage':{'cost':'0.001', 'prompt_tokens':1, 'completion_tokens':1,
                     'is_byok':False},
            'choices':[{'finish_reason':'stop', 'message':{'content':'VERDICT: TRUE'}}],
        })
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(handler))
    rows = [{'equation1':f'PRIVATE {i}', 'equation2':'SECRET', 'answer':True}
            for i in range(20)]
    await e.live_evaluate(rows, TEMPLATE, uncapped_env(OPENROUTER_API_KEY='PRIVATE KEY'))
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(events) == 3
    assert {event['model'] for event in events} == set(e.MODEL_IDS)
    assert all(event == {'event':'evaluation_progress', 'model':event['model'], 'completed':20}
               for event in events)
    assert 'PRIVATE' not in json.dumps(events)


@pytest.mark.asyncio
async def test_cost_write_error_does_not_mask_evaluation_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'ROOT', tmp_path)
    async def handler(req):
        if req.url.path.endswith('/key'):
            return httpx.Response(200, json={'data': {}})
        return httpx.Response(200, json={
            'provider':'WrongProvider',
            'usage':{'cost':'0', 'is_byok':False},
            'choices':[{'finish_reason':'stop', 'message':{'content':'VERDICT: TRUE'}}],
        })
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(handler))
    monkeypatch.setattr(e, 'write_cost_artifact', lambda *_: (_ for _ in ()).throw(OSError()))
    with pytest.raises(e.InvalidRun, match='provider_route_mismatch'):
        await e.live_evaluate(
            [{'equation1':'a','equation2':'b','answer':True}], TEMPLATE, uncapped_env())


def entries():
    out=[]
    for i in range(1,141):
        for variant in [{'implication':{'lhs':f'Equation{i}','rhs':f'Equation{i+200}','finite':False}}, {'facts':{'satisfied':[f'Equation{i}'],'refuted':[f'Equation{i+400}'],'finite':True}}]:
            out.append({'proven':True,'variant':variant,'name':f'theorem{i}','filename':'proof.lean','line':i})
    return out


def test_selection_balance_provenance_and_exclusions():
    source=entries(); equations={i:f'x = x * y ({i})' for i in range(1,1000)}
    source += [{'proven':False,'variant':{'implication':{'lhs':'Equation600','rhs':'Equation601','finite':False}}}]
    excluded={(1,201),(2,202)}
    a=selection.select(source,excluded,equations,b'a'*32)
    assert a==selection.select(source,excluded,equations,b'a'*32)
    assert a!=selection.select(source,excluded,equations,b'b'*32)
    assert len(a)==200 and sum(row['answer'] for row in a)==100
    assert not ({(row['eq1_id'],row['eq2_id']) for row in a} & excluded)
    assert all(row['provenance']['sourceSha']==selection.SOURCE_SHA for row in a)


def test_conflicts_finite_only_and_invalid_pairs_excluded():
    source=entries()
    source += [{'proven':True,'name':'x','filename':'x','line':1,'variant':{'implication':{'lhs':'Equation1','rhs':'Equation401','finite':False}}}]
    source += [{'proven':True,'name':'x','filename':'x','line':1,'variant':{'implication':{'lhs':'Equation700','rhs':'Equation701','finite':True}}}]
    positive,negative=selection.candidates(source,set())
    assert (1,401) not in positive and (1,401) not in negative
    assert (700,701) not in positive
    with pytest.raises(ValueError): selection.external(ROOT/'private')


def test_ranked_dataset_checksum_balance(tmp_path):
    rows=selection.select(entries(),set(),{i:str(i) for i in range(1,1000)},b'a'*32)
    data=b''.join((json.dumps(r)+'\n').encode() for r in rows)
    path=tmp_path/'data'; path.write_bytes(data)
    manifest=tmp_path/'manifest'; manifest.write_text(json.dumps({'contractVersion':e.CONTRACT,'version':'v1','sha256':e.digest(data)}))
    assert len(e.load_dataset(path,ranked=True,manifest_path=manifest)[0])==200
    path.write_bytes(data+b'\n')
    with pytest.raises(e.InvalidRun): e.load_dataset(path,ranked=True,manifest_path=manifest)

@pytest.mark.asyncio
async def test_publish_aggregate_only_and_remove_score_on_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(e, 'ROOT', tmp_path)
    (tmp_path/'submission').mkdir()
    (tmp_path/'submission/prompt.txt').write_text(TEMPLATE)
    env={'DISTILL_LIVE':'1','OPENROUTER_API_KEY':'test-secret','DISTILL_MAX_SPEND_USD':'1'}
    async def mock(rows, template, _env, **kwargs):
        async def complete(*_): return response('VERDICT: TRUE\nprivate reasoning must not escape')
        return await e.evaluate(rows,template,complete,**kwargs)
    await e.run(SimpleNamespace(ranked=False),env,mock)
    output=(tmp_path/'score.json').read_text()
    assert 'private reasoning' not in output and 'test-secret' not in output
    payload=json.loads(output)
    assert_problem_text_only_in_panel(payload)
    assert payload['metrics']['datasetKind']=='public' and payload['metrics']['stage']=='public_complete'
    async def failed(*_, **__): raise RuntimeError('PRIVATE error body')
    with pytest.raises(RuntimeError): await e.run(SimpleNamespace(ranked=False),env,failed)
    assert not (tmp_path/'score.json').exists()
    assert 'private' not in capsys.readouterr().out.lower()


def test_cli_does_not_echo_exception_body(tmp_path,monkeypatch,capsys):
    async def failed(*_): raise RuntimeError('PRIVATE prompt and key')
    monkeypatch.setattr(e,'run',failed)
    monkeypatch.setattr(sys,'argv',['evaluate.py'])
    assert e.main()==1
    assert capsys.readouterr().err=='Math Distillation: evaluation_failed\n'


@pytest.mark.asyncio
async def test_route_mismatch_invalidates_run(monkeypatch):
    async def handler(req):
        if req.url.path.endswith('/key'):
            return httpx.Response(200,json={'data':{'limit':1,'limit_remaining':1,'limit_reset':None,'include_byok_in_limit':True}})
        return httpx.Response(200,json={'provider':'WrongProvider','choices':[{'finish_reason':'stop','message':{'content':'VERDICT: TRUE'}}]})
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(handler))
    with pytest.raises(e.InvalidRun,match='provider_route_mismatch'):
        await e.live_evaluate([{'equation1':'a','equation2':'b','answer':True}],TEMPLATE,{'DISTILL_LIVE':'1','OPENROUTER_API_KEY':'test','DISTILL_MAX_SPEND_USD':'1'})


@pytest.mark.asyncio
async def test_local_budget_blocks_before_dispatch(monkeypatch):
    calls=0
    async def handler(req):
        nonlocal calls
        if req.url.path.endswith('/key'):
            return httpx.Response(200,json={'data':{'limit':0.001,'limit_remaining':0.001,'limit_reset':None,'include_byok_in_limit':True}})
        calls += 1
        return httpx.Response(200,json={})
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(handler))
    with pytest.raises(e.InvalidRun,match='local_spending_cap_exceeded'):
        await e.live_evaluate([{'equation1':'a','equation2':'b','answer':True}],TEMPLATE,{'DISTILL_LIVE':'1','OPENROUTER_API_KEY':'test','DISTILL_MAX_SPEND_USD':'0.001'})
    assert calls == 0


@pytest.mark.asyncio
async def test_attempt_limit_is_per_model_prompt():
    calls=0
    async def handler(req):
        nonlocal calls
        calls += 1
        return httpx.Response(500,json={})
    ledger=e.BudgetLedger(e.Decimal('10'))
    transport=e.GuardedTransport(httpx.MockTransport(handler),ledger)
    route=e.ROUTES['llama-3-3-70b-instruct']
    body={'model':route.model,'messages':[{'role':'user','content':'same prompt'}],
          'max_tokens':8192,'temperature':0,'seed':0,'reasoning':{'effort':'none'},
          'provider':{'order':[route.endpoint],'quantizations':['fp8'],'allow_fallbacks':False}}
    async with httpx.AsyncClient(transport=transport) as client:
        for _ in range(e.MAX_ATTEMPTS_PER_PAIR):
            assert (await client.post('https://openrouter.ai/api/v1/chat/completions',json=body)).status_code==500
        with pytest.raises(e.InvalidRun,match='attempt_limit_exceeded'):
            await client.post('https://openrouter.ai/api/v1/chat/completions',json=body)
    assert calls==e.MAX_ATTEMPTS_PER_PAIR


@pytest.mark.asyncio
async def test_private_receipts_are_sanitized_and_aggregated(tmp_path, monkeypatch):
    receipts=tmp_path/'attempts.jsonl'
    async def handler(req):
        if req.url.path.endswith('/key'):
            return httpx.Response(200,json={'data':{'limit':1,'limit_remaining':1,'limit_reset':None,'include_byok_in_limit':True}})
        provider='Novita' if json.loads(req.content)['model'].startswith('google/') else 'DeepInfra'
        return httpx.Response(200,json={
            'provider':provider,
            'usage':{'prompt_tokens':17,'completion_tokens':9,'cost':'0.002',
                     'is_byok':False,
                     'prompt_tokens_details':{'cached_tokens':3},
                     'completion_tokens_details':{'reasoning_tokens':4}},
            'choices':[{'finish_reason':'stop','message':{'content':'VERDICT: TRUE\nPRIVATE RESPONSE'}}],
        })
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(handler))
    result=await e.live_evaluate(
        [{'equation1':'PRIVATE EQUATION ONE','equation2':'PRIVATE EQUATION TWO','answer':True}],TEMPLATE,
        {'DISTILL_LIVE':'1','OPENROUTER_API_KEY':'PRIVATE KEY','DISTILL_MAX_SPEND_USD':'1',
         'DISTILL_PRIVATE_RECEIPTS':str(receipts)})
    assert receipts.stat().st_mode & 0o777 == 0o600
    rows=[json.loads(line) for line in receipts.read_text().splitlines()]
    assert len(rows)==3
    assert len({row['attemptId'] for row in rows})==3
    assert len({row['reservationId'] for row in rows})==3
    assert {row['endpoint'] for row in rows}=={'deepinfra/turbo','novita/bf16'}
    assert all(row['httpStatus']==200 and row['outcome']=='completed' for row in rows)
    assert all((row['tokensIn'],row['tokensOut'],row['reasoningTokens'],row['cachedTokens'],row['isByok'])==(17,9,4,3,False) for row in rows)
    assert all(row['actualCostUsd']=='0.002' and row['unresolvedReservationUsd']=='0' for row in rows)
    serialized=receipts.read_text()
    assert 'PRIVATE' not in serialized and 'equation' not in serialized.lower()
    billing=result['metrics']['billing']
    assert billing['confirmedSpendUsd']=='0.006'
    public_result=json.dumps(result)
    assert 'attemptId' not in public_result and 'reservationId' not in public_result and 'PRIVATE' not in public_result
    for alias in e.MODEL_IDS:
        assert billing['models'][alias]['tokensIn']==17
        assert billing['models'][alias]['tokensOut']==9
        assert billing['models'][alias]['reasoningTokens']==4
        assert billing['models'][alias]['cachedTokens']==3
        assert billing['models'][alias]['isByokReportedAttempts']==1
        assert billing['models'][alias]['byokAttempts']==0


@pytest.mark.asyncio
async def test_transport_failure_writes_unresolved_receipt(tmp_path):
    receipts_path=tmp_path/'failures.jsonl'
    receipts=e.PrivateReceipts(receipts_path)
    async def handler(req):
        raise httpx.ConnectError('PRIVATE FAILURE',request=req)
    ledger=e.BudgetLedger(e.Decimal('10'))
    transport=e.GuardedTransport(httpx.MockTransport(handler),ledger,receipts)
    route=e.ROUTES['llama-3-3-70b-instruct']
    body={'model':route.model,'messages':[{'role':'user','content':'PRIVATE PROMPT'}],
          'max_tokens':8192,'temperature':0,'seed':0,'reasoning':{'effort':'none'},
          'provider':{'order':[route.endpoint],'quantizations':['fp8'],'allow_fallbacks':False}}
    with pytest.raises(httpx.ConnectError):
        async with httpx.AsyncClient(transport=transport) as client:
            await client.post('https://openrouter.ai/api/v1/chat/completions',json=body)
    receipts.close()
    row=json.loads(receipts_path.read_text())
    assert row['outcome']=='transport_error' and row['httpStatus'] is None
    assert row['actualCostUsd'] is None and e.Decimal(row['unresolvedReservationUsd'])>0
    assert 'PRIVATE' not in receipts_path.read_text()
    model=ledger.metrics()['models']['llama-3-3-70b-instruct']
    assert model['failedAttempts']==1 and model['transportFailures']==1


def test_private_receipt_path_is_new_and_external(tmp_path):
    with pytest.raises(e.InvalidRun,match='private_receipts_must_be_external'):
        e.PrivateReceipts(Path('relative.jsonl'))
    with pytest.raises(e.InvalidRun,match='private_receipts_must_be_external'):
        e.PrivateReceipts(e.ROOT/'inside.jsonl')
    existing=tmp_path/'existing.jsonl'; existing.write_text('old')
    with pytest.raises(e.InvalidRun,match='private_receipts_unavailable'):
        e.PrivateReceipts(existing)


@pytest.mark.asyncio
async def test_receipts_distinguish_missing_usage_from_explicit_zero(tmp_path):
    receipt_path=tmp_path/'coverage.jsonl'
    receipts=e.PrivateReceipts(receipt_path)
    calls=0
    async def handler(req):
        nonlocal calls
        calls += 1
        payload={} if calls==1 else {
            'usage':{'prompt_tokens':0,'completion_tokens':0,'cost':0,'is_byok':True,
                     'prompt_tokens_details':{'cached_tokens':0},
                     'completion_tokens_details':{'reasoning_tokens':0}}}
        return httpx.Response(200,json=payload)
    ledger=e.BudgetLedger(e.Decimal('10'))
    transport=e.GuardedTransport(httpx.MockTransport(handler),ledger,receipts)
    route=e.ROUTES['gemma-4-31b-it']
    body={'model':route.model,'messages':[{'role':'user','content':'sensitive'}],
          'max_tokens':8192,'temperature':0,'seed':0,'reasoning':{'effort':'none'},
          'provider':{'order':[route.endpoint],'quantizations':['bf16'],'allow_fallbacks':False}}
    async with httpx.AsyncClient(transport=transport) as client:
        for _ in range(2):
            await client.post('https://openrouter.ai/api/v1/chat/completions',json=body)
    receipts.close()
    missing,zero=[json.loads(line) for line in receipt_path.read_text().splitlines()]
    assert (missing['tokensIn'],missing['actualCostUsd'],missing['isByok'])==(None,None,None)
    assert e.Decimal(missing['unresolvedReservationUsd'])>0
    assert (zero['tokensIn'],zero['tokensOut'],zero['reasoningTokens'],zero['cachedTokens'])==(0,0,0,0)
    assert zero['actualCostUsd']=='0' and zero['unresolvedReservationUsd']=='0' and zero['isByok'] is True
    model=ledger.metrics()['models']['gemma-4-31b-it']
    assert model['usageReportedAttempts']==1 and model['costReportedAttempts']==1
    assert model['tokensInReportedAttempts']==1 and model['cachedTokensReportedAttempts']==1
    assert model['isByokReportedAttempts']==1 and model['byokAttempts']==1


@pytest.mark.asyncio
async def test_gzip_response_is_decoded_once():
    import gzip
    payload = {'usage': {'cost': 0.001, 'prompt_tokens': 10, 'completion_tokens': 5},
               'choices': [{'finish_reason': 'stop', 'message': {'content': 'VERDICT: TRUE'}}]}
    compressed = gzip.compress(json.dumps(payload).encode())
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield compressed
    async def handler(request):
        return httpx.Response(200, headers={'content-encoding': 'gzip', 'content-length': str(len(compressed))}, stream=Stream())
    ledger = e.BudgetLedger(e.Decimal('1'))
    route = e.ROUTES['llama-3-3-70b-instruct']
    body = {'model': route.model, 'messages': [{'role': 'user', 'content': 'x=x'}],
            'max_tokens': 8192, 'temperature': 0, 'seed': 0,
            'provider': {'order': [route.endpoint], 'quantizations': ['fp8'], 'allow_fallbacks': False}}
    async with httpx.AsyncClient(transport=e.GuardedTransport(httpx.MockTransport(handler), ledger)) as client:
        result = await client.post('https://openrouter.ai/api/v1/chat/completions', json=body)
        assert result.json() == payload
        assert 'content-encoding' not in result.headers
    assert ledger.metrics()['attempts'] == 1


@pytest.mark.asyncio
async def test_model_queues_respect_route_caps_without_consuming_call_timeout():
    active = {alias: 0 for alias in e.MODEL_IDS}
    peak = {alias: 0 for alias in e.MODEL_IDS}
    async def complete(alias, _prompt):
        active[alias] += 1
        peak[alias] = max(peak[alias], active[alias])
        try:
            await asyncio.sleep(.02)
            return response()
        finally:
            active[alias] -= 1
    rows = [{'equation1':str(i),'equation2':'x=x','answer':True} for i in range(8)]
    result = await e.evaluate(rows, TEMPLATE, complete, timeout=.03, concurrency=6,
                              model_concurrency=e.MODEL_CONCURRENCY)
    assert result['metrics']['outcomes'] == 24
    assert peak['gpt-oss-120b'] == 1
    assert 1 <= peak['llama-3-3-70b-instruct'] <= 3
    assert 1 <= peak['gemma-4-31b-it'] <= 6


@pytest.mark.asyncio
async def test_model_scheduler_replaces_completed_lane_without_global_head_of_line():
    permits = {alias: asyncio.Queue() for alias in e.MODEL_IDS}
    started = []
    active = {alias: 0 for alias in e.MODEL_IDS}
    peak = {alias: 0 for alias in e.MODEL_IDS}

    async def complete(alias, prompt):
        started.append((alias, prompt))
        active[alias] += 1
        peak[alias] = max(peak[alias], active[alias])
        try:
            await permits[alias].get()
            return response()
        finally:
            active[alias] -= 1

    rows = [{'equation1': str(i), 'equation2': 'x=x', 'answer': True} for i in range(5)]
    evaluation = asyncio.create_task(e.evaluate(
        rows, TEMPLATE, complete, concurrency=6, model_concurrency=e.MODEL_CONCURRENCY))

    for _ in range(100):
        if len(started) == 6:
            break
        await asyncio.sleep(0)
    initial = {alias: [name for name, _ in started].count(alias) for alias in e.MODEL_IDS}
    assert initial == {'gpt-oss-120b': 1, 'llama-3-3-70b-instruct': 3,
                       'gemma-4-31b-it': 2}

    # Completing the sole GPT request must admit the next GPT request. With
    # nested FIFO semaphores, already queued Gemma requests take the global
    # slot and can starve this capped lane until much later.
    permits[e.MODEL_IDS[0]].put_nowait(None)
    for _ in range(100):
        if [alias for alias, _ in started].count(e.MODEL_IDS[0]) == 2:
            break
        await asyncio.sleep(0)
    assert [alias for alias, _ in started].count(e.MODEL_IDS[0]) == 2

    for alias in e.MODEL_IDS:
        for _ in rows:
            permits[alias].put_nowait(None)
    result = await asyncio.wait_for(evaluation, 1)

    assert result['metrics']['outcomes'] == len(rows) * len(e.MODEL_IDS)
    assert result['score'] == 1
    assert all(peak[alias] <= e.MODEL_CONCURRENCY[alias] for alias in e.MODEL_IDS)
    assert sorted(started) == sorted(
        (alias, e.render_prompt(TEMPLATE, row['equation1'], row['equation2']))
        for row in rows for alias in e.MODEL_IDS)


@pytest.mark.asyncio
async def test_research_drain_keeps_other_results_without_partial_score():
    completed = []
    async def complete(alias, prompt):
        if alias == e.MODEL_IDS[0]:
            raise RuntimeError('upstream unavailable')
        await asyncio.sleep(0.01)
        completed.append(alias)
        return response()
    with pytest.raises(RuntimeError):
        await e.evaluate([{'equation1': 'x=x', 'equation2': 'x=x', 'answer': True}], TEMPLATE,
                         complete, drain_errors=True)
    assert set(completed) == set(e.MODEL_IDS[1:])


# --- Staged evaluation: public screen, ranked rounds, impossibility stopping ---

PUBLIC_SHA256 = '5e9692acf9ba4b0ffa3f235b946831447a2b827b8dec3061a81b97d80c3e163c'
EXECUTION_CONFIG_SHA256 = '8c636e88d1e01d77c1770fa60cbfafd0cf1fa99180efcf5b78bdd6b55d3df4d3'
CONFIG_SHA256 = '0b9b3578eb55e6ff4e495fcaf2ca73608b908bbd28477072e9ab15e579cc76fe'
RECORD_URL = 'https://yukon.test/api/benchmarks/c2f61f98-21dc-41fe-aade-05f6dd41f60a'


def public_rows():
    return e.load_dataset(e.SCREEN_DATASET, ranked=False)[0]


def ranked_rows(count=200):
    return [{'id': f'yukon_{i+1:04}', 'eq1_id': 1000 + i, 'eq2_id': 3000 + i,
             'equation1': f'PRIVATE {i}', 'equation2': 'SECRET', 'answer': i % 2 == 0,
             'provenance': {'name': 'PRIVATE THEOREM'}} for i in range(count)]


def public_answers():
    return {f"{row['equation1']} {row['equation2']}": row['answer'] for row in public_rows()}


def verdict(answer):
    return f"VERDICT: {'TRUE' if answer else 'FALSE'}"


def openrouter(answer):
    """MockTransport handler: `answer(prompt, model)` returns the completion text."""
    async def handler(req):
        if req.url.path.endswith('/key'):
            return httpx.Response(200, json={'data': {'limit': 1, 'limit_remaining': 1, 'limit_reset': None,
                                                      'include_byok_in_limit': True}})
        body = json.loads(req.content)
        provider = 'Novita' if body['model'].startswith('google/') else 'DeepInfra'
        text = answer(body['messages'][0]['content'], body['model'])
        return httpx.Response(200, json={
            'provider': provider,
            'usage': {'cost': '0.001', 'prompt_tokens': 1, 'completion_tokens': 1, 'is_byok': False},
            'choices': [{'finish_reason': 'stop', 'message': {'content': text}}],
        })
    return handler


def correct_public_then(private_text=None):
    """Answer the public panel correctly; answer ranked prompts with `private_text`, or correctly."""
    answers = public_answers()
    def answer(prompt, _model):
        if 'PRIVATE' in prompt:
            return verdict(int(prompt.split()[1]) % 2 == 0) if private_text is None else private_text
        return verdict(answers[prompt])
    return answer


def record_endpoint(monkeypatch, payload=None, *, status=200, raw=None, exc=None):
    async def handler(req):
        assert 'authorization' not in {k.lower() for k in req.headers}
        if exc is not None:
            raise exc
        if raw is not None:
            return httpx.Response(status, content=raw)
        return httpx.Response(status, json=payload)
    monkeypatch.setattr(e, 'record_transport', lambda: httpx.MockTransport(handler))


def hosted_fixture(tmp_path, tmp_path_factory, monkeypatch, rows=None, **env_changes):
    monkeypatch.setattr(e, 'ROOT', tmp_path)
    (tmp_path / 'submission').mkdir()
    (tmp_path / 'submission/prompt.txt').write_text(TEMPLATE)
    rows = ranked_rows() if rows is None else rows
    data = b''.join((json.dumps(row, sort_keys=True) + '\n').encode() for row in rows)
    private = tmp_path_factory.mktemp('private') / 'ranked.jsonl'
    private.write_bytes(data)
    (tmp_path / 'ranked-dataset.json').write_text(json.dumps(
        {'contractVersion': e.CONTRACT, 'version': 'yukon-equational-v1', 'sha256': e.digest(data)}))
    return uncapped_env(GITHUB_SHA='a' * 40, GITHUB_RUN_ID='7', GITHUB_RUN_ATTEMPT='1',
                        DISTILL_PRIVATE_DATASET=str(private), **env_changes)


async def hosted_run(tmp_path, tmp_path_factory, monkeypatch, answer, *, ranked=True, record=None, rows=None, **env_changes):
    env = hosted_fixture(tmp_path, tmp_path_factory, monkeypatch, rows=rows, **env_changes)
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(openrouter(answer)))
    if record is not None:
        env['DISTILL_RECORD_SOURCE_URL'] = RECORD_URL
        record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': record / 600, 'direction': '+'}})
    await e.run(SimpleNamespace(ranked=ranked), env)
    payload = json.loads((tmp_path / 'score.json').read_text())
    costs = json.loads((tmp_path / 'evaluation-costs.json').read_text())
    return payload, costs


def assert_problem_text_only_in_panel(payload):
    """`equation1`/`equation2`/`expected` may appear only as verbatim public-panel rows."""
    vendored = public_rows()
    found = []
    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('equation1', 'equation2', 'expected', 'answer'):
                    found.append(path)
                walk(value, path + (key,))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path + (index,))
    walk(payload, ())
    assert all(path[:3] == ('metrics', 'publicPanel', 'problems') and len(path) == 4 for path in found), found
    problems = payload['metrics']['publicPanel']['problems']
    assert problems == [{'id': row['id'], 'eq1Id': row['eq1_id'], 'eq2Id': row['eq2_id'],
                         'equation1': row['equation1'], 'equation2': row['equation2'], 'expected': row['answer']}
                        for row in vendored]
    assert payload['metrics']['publicPanel']['datasetSha256'] == PUBLIC_SHA256
    assert 'PRIVATE' not in json.dumps(payload) and 'SECRET' not in json.dumps(payload)


def assert_identity(payload, kind):
    metrics = payload['metrics']
    assert metrics['contractVersion'] == 'distill-v2'
    assert metrics['executionConfigSha256'] == EXECUTION_CONFIG_SHA256
    assert metrics['configSha256'] == CONFIG_SHA256
    assert metrics['datasetKind'] == kind
    assert metrics['datasetVersion'] == ('yukon-equational-v1' if kind == 'ranked' else 'public-smoke-v1')


@pytest.mark.asyncio
async def test_verdicts_align_to_question_ids():
    rows = [{'id': f'q{i}', 'equation1': str(i), 'equation2': 'x=x', 'answer': True} for i in range(3)]
    async def complete(alias, _prompt):
        return response({'gpt-oss-120b': 'VERDICT: TRUE', 'llama-3-3-70b-instruct': 'unclear',
                         'gemma-4-31b-it': 'VERDICT: FALSE'}[alias])
    plain = await e.evaluate(rows, TEMPLATE, complete)
    assert 'verdicts' not in plain['metrics']
    result = await e.evaluate(rows, TEMPLATE, complete, include_verdicts=True)
    assert result['metrics']['verdicts'] == {'gpt-oss-120b': '111', 'llama-3-3-70b-instruct': 'uuu',
                                             'gemma-4-31b-it': '000'}
    with pytest.raises(e.InvalidRun, match='invalid_verdict_policy'):
        await e.evaluate(rows, TEMPLATE, complete, model_concurrency=e.MODEL_CONCURRENCY,
                         include_verdicts=True, shuffle_seed='x')


def test_aggregate_tolerates_a_model_with_zero_completions():
    rows = [{'id': 'a', 'equation1': 'a', 'equation2': 'b', 'answer': True}] * 2
    outcomes = [('gpt-oss-120b', True, 1, 2), None, None, ('gpt-oss-120b', None, 0, 0), None, None]
    result = e._aggregate(outcomes, rows)
    assert result['metrics']['outcomes'] == 2 and result['metrics']['questionCount'] == 2
    assert result['metrics']['models']['gpt-oss-120b'] == {'correct': 1, 'total': 2, 'accuracy': .5, 'parseFailures': 1}
    assert result['metrics']['models']['gemma-4-31b-it'] == {'correct': 0, 'total': 0, 'accuracy': 0.0, 'parseFailures': 0}
    assert result['metrics']['tokensIn'] == 1 and result['metrics']['tokensOut'] == 2
    empty = e._aggregate([None] * 6, rows)
    assert empty['score'] == 0 and empty['metrics']['outcomes'] == 0


def permit_fake():
    permits = {alias: asyncio.Queue() for alias in e.MODEL_IDS}
    started = []
    async def complete(alias, prompt):
        started.append((alias, prompt))
        return response(await permits[alias].get())
    return permits, started, complete


async def settle(predicate, spins=200):
    for _ in range(spins):
        if predicate():
            return
        await asyncio.sleep(0)
    assert predicate()


@pytest.mark.asyncio
async def test_stop_fires_only_at_round_boundary_and_admits_nothing_after():
    permits, started, complete = permit_fake()
    rows = [{'equation1': str(i), 'equation2': 'x=x', 'answer': True} for i in range(10)]
    evaluation = asyncio.create_task(e.evaluate(rows, TEMPLATE, complete, model_concurrency=e.MODEL_CONCURRENCY,
                                                record_correct=20, round_questions=5))
    await settle(lambda: len(started) == 6)
    # Ten wrong outcomes already make 20 unreachable, yet the round of 15 must drain first.
    for alias, wrong in zip(e.MODEL_IDS, (4, 3, 3), strict=True):
        for _ in range(wrong):
            permits[alias].put_nowait('VERDICT: FALSE')
    await settle(lambda: len(started) == 15)
    await asyncio.sleep(0.01)
    assert not evaluation.done()
    for alias in e.MODEL_IDS:
        for _ in range(5):
            permits[alias].put_nowait('VERDICT: TRUE')
    result = await asyncio.wait_for(evaluation, 1)
    assert len(started) == 15, 'round 1 drained completely and round 2 never started'
    assert result['score'] == 0
    stopped = result['metrics']['stopped']
    assert stopped['roundsTotal'] == 2 and stopped['roundsCleared'] == 1 and stopped['roundQuestions'] == 5
    assert stopped['completedOutcomes'] == 15 == stopped['roundsCleared'] * 15
    assert stopped['correctSoFar'] + (30 - stopped['completedOutcomes']) <= stopped['recordCorrect'] == 20
    assert result['metrics']['outcomes'] == 15
    assert 'verdicts' not in result['metrics']


@pytest.mark.asyncio
@pytest.mark.parametrize('wrong_rows,stops', [(0, False), (1, True)])
async def test_stop_boundary_is_exact(wrong_rows, stops):
    # 10 rows, rounds of 5, record 27/30: after round one, 15 - 3w + 15 <= 27 iff w >= 1 wrong row.
    rows = [{'equation1': str(i), 'equation2': 'x=x', 'answer': True} for i in range(10)]
    async def complete(_alias, prompt):
        index = int(prompt.split()[0])
        return response('VERDICT: FALSE' if index < wrong_rows else 'VERDICT: TRUE')
    result = await e.evaluate(rows, TEMPLATE, complete, model_concurrency=e.MODEL_CONCURRENCY,
                              record_correct=27, round_questions=5)
    if stops:
        assert result['score'] == 0 and result['metrics']['stopped']['roundsCleared'] == 1
        assert result['metrics']['stopped']['completedOutcomes'] == 15
    else:
        assert 'stopped' not in result['metrics'] and result['score'] == 1 and result['metrics']['outcomes'] == 30


@pytest.mark.asyncio
async def test_complete_run_never_reports_stopped_even_below_record():
    rows = [{'equation1': str(i), 'equation2': 'x=x', 'answer': True} for i in range(4)]
    async def complete(*_): return response('VERDICT: FALSE')
    result = await e.evaluate(rows, TEMPLATE, complete, model_concurrency=e.MODEL_CONCURRENCY,
                              record_correct=12, round_questions=4)
    assert 'stopped' not in result['metrics'] and result['metrics']['outcomes'] == 12 and result['score'] == 0


@pytest.mark.asyncio
async def test_stop_requires_scheduler_branch():
    rows = [{'equation1': 'a', 'equation2': 'b', 'answer': True}]
    async def complete(*_): return response()
    for kwargs in ({'record_correct': 1}, {'round_questions': 1}, {'shuffle_seed': 'x'}):
        with pytest.raises(e.InvalidRun, match='invalid_stop_policy'):
            await e.evaluate(rows, TEMPLATE, complete, **kwargs)
    with pytest.raises(e.InvalidRun, match='invalid_stop_policy'):
        await e.evaluate(rows, TEMPLATE, complete, model_concurrency=e.MODEL_CONCURRENCY, record_correct=-1)
    with pytest.raises(e.InvalidRun, match='invalid_stop_policy'):
        await e.evaluate(rows, TEMPLATE, complete, model_concurrency=e.MODEL_CONCURRENCY, round_questions=0)


@pytest.mark.asyncio
async def test_shuffle_changes_order_but_not_complete_counts():
    rows = [{'equation1': str(i), 'equation2': 'x=x', 'answer': i % 3 == 0} for i in range(30)]
    async def make(seed=None, round_questions=None):
        seen = []
        async def complete(_alias, prompt):
            seen.append(int(prompt.split()[0]))
            return response('VERDICT: TRUE')
        result = await e.evaluate(rows, TEMPLATE, complete, model_concurrency=e.MODEL_CONCURRENCY,
                                  shuffle_seed=seed, round_questions=round_questions)
        return result, seen
    plain, plain_order = await make()
    a, a_order = await make('a' * 40, 10)
    b, b_order = await make('b' * 40, 10)
    same, same_order = await make('a' * 40, 10)
    assert a['metrics']['models'] == b['metrics']['models'] == plain['metrics']['models']
    assert a['score'] == b['score'] == plain['score'] == 10 / 30
    assert a['metrics']['outcomes'] == 90 and 'stopped' not in a['metrics']
    assert a_order == same_order and a_order != b_order and a_order != plain_order
    assert sorted(a_order) == sorted(b_order) == sorted(plain_order)


@pytest.mark.asyncio
async def test_shuffle_is_deterministic_per_seed_and_stopping_round_varies():
    rows = [{'equation1': str(i), 'equation2': 'x=x', 'answer': True} for i in range(60)]
    async def complete(_alias, prompt):
        return response('VERDICT: FALSE' if int(prompt.split()[0]) < 36 else 'VERDICT: TRUE')
    async def stop_round(seed):
        result = await e.evaluate(rows, TEMPLATE, complete, model_concurrency=e.MODEL_CONCURRENCY,
                                  shuffle_seed=seed, round_questions=10, record_correct=100)
        stopped = result['metrics']['stopped']
        assert stopped['completedOutcomes'] == stopped['roundsCleared'] * 30
        assert stopped['correctSoFar'] + (180 - stopped['completedOutcomes']) <= 100
        return stopped['roundsCleared']
    seeds = [format(i, '040x') for i in range(8)]
    rounds = [await stop_round(seed) for seed in seeds]
    assert rounds == [await stop_round(seed) for seed in seeds], 'same seed, same stopping round'
    assert len(set(rounds)) > 1, rounds


@pytest.mark.parametrize('score,expected', [(473 / 600, 473), (0.788333333333333, 473), (0, 0), (1, 600),
                                            (0.7883, None), (1.2, None), (-0.1, None), ('0.5', None),
                                            (True, None), (float('nan'), None)])
def test_record_correct_rounding(score, expected):
    assert e.record_correct_from_score(score) == expected


@pytest.mark.asyncio
async def test_fetch_record_reads_public_benchmark_row(monkeypatch):
    live_float = 0.788333333333333  # exactly what the dev API returned for 473/600
    record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': live_float, 'direction': '+'}})
    assert await e.fetch_record(RECORD_URL) == {'source': 'yukon-public-api', 'score': live_float, 'correct': 473}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['http500', 'timeout', 'minimize', 'maximize_word', 'not_json', 'inexact', 'missing', 'plain_http'])
async def test_fetch_record_fails_open(monkeypatch, failure):
    url = RECORD_URL
    if failure == 'http500':
        record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': .5, 'direction': '+'}}, status=500)
    elif failure == 'timeout':
        record_endpoint(monkeypatch, exc=httpx.ReadTimeout('PRIVATE'))
    elif failure == 'minimize':
        record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': .5, 'direction': '-'}})
    elif failure == 'maximize_word':
        record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': .5, 'direction': 'maximize'}})
    elif failure == 'not_json':
        record_endpoint(monkeypatch, raw=b'<html>')
    elif failure == 'inexact':
        record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': .7883, 'direction': '+'}})
    elif failure == 'missing':
        record_endpoint(monkeypatch, {'benchmark': {}})
    else:
        record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': .5, 'direction': '+'}})
        url = 'http://yukon.test/api/benchmarks/x'
    assert await e.fetch_record(url) is None


@pytest.mark.asyncio
async def test_unavailable_record_runs_complete_and_is_reported(tmp_path, tmp_path_factory, monkeypatch):
    env = hosted_fixture(tmp_path, tmp_path_factory, monkeypatch, DISTILL_RECORD_SOURCE_URL=RECORD_URL)
    record_endpoint(monkeypatch, raw=b'', status=503)
    calls = {}
    async def fake(rows, template, _env, **kwargs):
        calls.update(kwargs)
        async def complete(*_): return response()
        result = await e.evaluate(rows, template, complete, model_concurrency=e.MODEL_CONCURRENCY,
                                  record_correct=kwargs['record_correct'], shuffle_seed=kwargs['shuffle_seed'],
                                  round_questions=kwargs['round_questions'])
        result['metrics']['stage'] = 'ranked_complete'
        return result
    await e.run(SimpleNamespace(ranked=True), env, fake)
    assert calls['record_correct'] is None and calls['round_questions'] == 20
    assert calls['shuffle_seed'] != 'a' * 40 and len(calls['shuffle_seed']) == 32 and int(calls['shuffle_seed'], 16) >= 0
    payload = json.loads((tmp_path / 'score.json').read_text())
    policy = payload['metrics']['evaluationPolicy']
    assert policy['record'] == 'unavailable' and policy['stopWhenImpossible'] is True
    assert policy['screen'] == {'panel': 'public-smoke-v1', 'minCorrect': 32, 'maxParseFailures': 6}
    assert policy['roundQuestions'] == 20 and policy['order'] == 'shuffled-question-major' and policy['shuffleSeed'] == 'private-random'
    assert payload['metrics']['outcomes'] == 600 and payload['score'] == .5


@pytest.mark.asyncio
async def test_stop_switch_off_skips_the_record_fetch(tmp_path, tmp_path_factory, monkeypatch):
    env = hosted_fixture(tmp_path, tmp_path_factory, monkeypatch, DISTILL_RECORD_SOURCE_URL=RECORD_URL,
                         DISTILL_STOP_WHEN_IMPOSSIBLE='0')
    async def fetched(*_): raise AssertionError('record must not be fetched when stopping is off')
    monkeypatch.setattr(e, 'fetch_record', fetched)
    async def fake(rows, template, _env, **kwargs):
        assert kwargs['record_correct'] is None
        return {'score': 0, 'metrics': {'stage': 'ranked_complete', 'models': {}, 'outcomes': 0}}
    await e.run(SimpleNamespace(ranked=True), env, fake)
    policy = json.loads((tmp_path / 'score.json').read_text())['metrics']['evaluationPolicy']
    assert policy['record'] == 'disabled' and policy['stopWhenImpossible'] is False


@pytest.mark.asyncio
@pytest.mark.parametrize('env_changes,reason,fetched', [
    ({}, 'url_unset', None),
    ({'DISTILL_RECORD_SOURCE_URL': RECORD_URL, 'DISTILL_STOP_WHEN_IMPOSSIBLE': '0'}, 'disabled', None),
    ({'DISTILL_RECORD_SOURCE_URL': RECORD_URL}, 'unavailable', None),
    ({'DISTILL_RECORD_SOURCE_URL': RECORD_URL}, 'fetched', 473),
])
async def test_record_event_is_always_logged(tmp_path, tmp_path_factory, monkeypatch, capsys, env_changes, reason, fetched):
    env = hosted_fixture(tmp_path, tmp_path_factory, monkeypatch, **env_changes)
    if fetched is None:
        record_endpoint(monkeypatch, raw=b'', status=503)
    else:
        record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': fetched / 600, 'direction': '+'}})
    async def fake(*_, **__):
        return {'score': 0, 'metrics': {'stage': 'ranked_complete', 'models': {}, 'outcomes': 0}}
    await e.run(SimpleNamespace(ranked=True), env, fake)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    record = [event for event in events if event.get('event') == 'record']
    assert record == [{'event': 'record', 'recordCorrect': fetched, 'reason': reason}]


@pytest.mark.parametrize('name,value,code', [
    ('DISTILL_SCREEN_MIN_CORRECT', '-1', 'screen_policy_invalid'),
    ('DISTILL_SCREEN_MIN_CORRECT', '61', 'screen_policy_invalid'),
    ('DISTILL_SCREEN_MIN_CORRECT', 'x', 'screen_policy_invalid'),
    ('DISTILL_SCREEN_MAX_PARSE_FAILURES', '1.5', 'screen_policy_invalid'),
    ('DISTILL_ROUND_QUESTIONS', '0', 'round_policy_invalid'),
    ('DISTILL_ROUND_QUESTIONS', '201', 'round_policy_invalid'),
    ('DISTILL_STOP_WHEN_IMPOSSIBLE', 'yes', 'stop_policy_invalid'),
])
def test_policy_env_validation(name, value, code):
    with pytest.raises(e.InvalidRun, match=code):
        (e.screen_policy if name.startswith('DISTILL_SCREEN') else e.stop_policy)({name: value})
    assert e.screen_policy({}) == (32, 6) == e.screen_policy({'DISTILL_SCREEN_MIN_CORRECT': '', 'DISTILL_SCREEN_MAX_PARSE_FAILURES': ' '})
    assert e.screen_policy({'DISTILL_SCREEN_MIN_CORRECT': '0', 'DISTILL_SCREEN_MAX_PARSE_FAILURES': '60'}) == (0, 60)
    assert e.stop_policy({}) == (True, None, 20)
    assert e.stop_policy({'DISTILL_STOP_WHEN_IMPOSSIBLE': '0', 'DISTILL_RECORD_SOURCE_URL': RECORD_URL, 'DISTILL_ROUND_QUESTIONS': '10'}) == (False, RECORD_URL, 10)


def test_screen_panel_disjoint_from_ranked(tmp_path, monkeypatch):
    screen = public_rows()
    e.assert_screen_disjoint(screen, ranked_rows())
    overlap = ranked_rows()
    overlap[7] = {**overlap[7], 'eq1_id': 5, 'eq2_id': 625}
    with pytest.raises(e.InvalidRun, match='screen_panel_overlap'):
        e.assert_screen_disjoint(screen, overlap)
    # select() with SAIR exclusions covering the public pairs cannot emit them.
    pairs = {(row['eq1_id'], row['eq2_id']) for row in screen}
    source = [{'proven': True, 'name': 't', 'filename': 'f', 'line': 1,
               'variant': {'implication': {'lhs': f'Equation{a}', 'rhs': f'Equation{b}', 'finite': False}}}
              for a, b in pairs]
    source += [{'proven': True, 'name': 't', 'filename': 'f', 'line': 1,
                'variant': {'implication': {'lhs': f'Equation{i}', 'rhs': f'Equation{i+100}', 'finite': False}}} for i in range(200, 320)]
    source += [{'proven': True, 'name': 't', 'filename': 'f', 'line': 1,
                'variant': {'facts': {'satisfied': [f'Equation{i}'], 'refuted': [f'Equation{i+100}'], 'finite': True}}} for i in range(500, 620)]
    rows = selection.select(source, pairs, {i: str(i) for i in range(1, 4695)}, b'a' * 32)
    e.assert_screen_disjoint(screen, rows)
    # prepare_private.py fails at fixture load, before any spend.
    import base64, gzip, prepare_private
    data = b''.join((json.dumps(row) + '\n').encode() for row in overlap)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'contractVersion': e.CONTRACT, 'version': 'v1', 'sha256': e.digest(data)}))
    monkeypatch.setattr(prepare_private, 'load_dataset',
                        lambda path, ranked: e.load_dataset(path, ranked=ranked, manifest_path=manifest))
    monkeypatch.setenv('DISTILL_PRIVATE_DATASET', str(tmp_path / 'ranked.jsonl'))
    monkeypatch.setenv('DISTILL_PRIVATE_DATA_GZIP_BASE64', base64.b64encode(gzip.compress(data)).decode())
    with pytest.raises(e.InvalidRun, match='screen_panel_overlap'):
        prepare_private.main()


@pytest.mark.asyncio
async def test_run_rejects_overlapping_ranked_panel_before_spend(tmp_path, tmp_path_factory, monkeypatch):
    overlap = ranked_rows()
    overlap[0] = {**overlap[0], 'eq1_id': 90, 'eq2_id': 1428}
    env = hosted_fixture(tmp_path, tmp_path_factory, monkeypatch, rows=overlap)
    async def never(*_, **__): raise AssertionError('evaluator must not run')
    with pytest.raises(e.InvalidRun, match='screen_panel_overlap'):
        await e.run(SimpleNamespace(ranked=True), env, never)
    assert not (tmp_path / 'score.json').exists()


def test_screen_failure_writes_sentinel_without_ranked_spend(tmp_path, tmp_path_factory, monkeypatch, capsys):
    # Synchronous on purpose: this drives the real CLI entry point, which owns the event loop.
    env = hosted_fixture(tmp_path, tmp_path_factory, monkeypatch, DISTILL_RECORD_SOURCE_URL=RECORD_URL)
    record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': 473 / 600, 'direction': '+'}})
    def answer(prompt, _model):
        assert 'PRIVATE' not in prompt, 'ranked rows must never be sent after a failed screen'
        return 'unclear'
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(openrouter(answer)))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', '--ranked'])
    assert e.main() == 0
    serialized = (tmp_path / 'score.json').read_text()
    payload = json.loads(serialized)
    assert payload['score'] == 0
    metrics = payload['metrics']
    assert metrics['stage'] == 'screen_failed' and metrics['partial'] is True
    assert metrics['outcomes'] == 0 and metrics['questionCount'] == 200 and metrics['parseFailures'] == 0
    assert all(metrics['models'][alias] == {'correct': 0, 'total': 0, 'accuracy': 0.0, 'parseFailures': 0} for alias in e.MODEL_IDS)
    assert metrics['billing']['attempts'] == 60
    panel = metrics['publicPanel']
    assert panel['datasetKind'] == 'public' and panel['datasetVersion'] == 'public-smoke-v1' and panel['questionCount'] == 20
    assert panel['correct'] == 0 and panel['total'] == 60 and panel['parseFailures'] == 60
    assert panel['floor'] == {'minCorrect': 32, 'maxParseFailures': 6, 'passed': False}
    assert panel['tokensIn'] == 60 and panel['tokensOut'] == 60, 'screen usage is reported on the panel'
    assert metrics['tokensIn'] == 0 and metrics['tokensOut'] == 0, 'top-level counters keep ranked semantics'
    assert panel['verdicts'] == {alias: 'u' * 20 for alias in e.MODEL_IDS}
    assert all(panel['models'][alias] == {'correct': 0, 'total': 20, 'accuracy': 0.0, 'parseFailures': 20} for alias in e.MODEL_IDS)
    assert metrics['evaluationPolicy']['record'] == {'source': 'yukon-public-api', 'score': 473 / 600, 'correct': 473}
    assert 'stopped' not in metrics
    assert_problem_text_only_in_panel(payload)
    assert_identity(payload, 'ranked')
    costs = json.loads((tmp_path / 'evaluation-costs.json').read_text())
    assert costs['status'] == 'screen_failed' and costs['billing']['attempts'] == 60
    out = capsys.readouterr().out
    assert json.loads(out.splitlines()[0]) == {'event': 'record', 'recordCorrect': 473, 'reason': 'fetched'}
    assert json.loads(out.splitlines()[-1]) == {'score': 0, 'datasetKind': 'ranked', 'stage': 'screen_failed'}
    assert 'PRIVATE' not in out and 'SECRET' not in out


@pytest.mark.asyncio
async def test_screen_pass_then_ranked_complete(tmp_path, tmp_path_factory, monkeypatch):
    payload, costs = await hosted_run(tmp_path, tmp_path_factory, monkeypatch, correct_public_then(), record=473)
    metrics = payload['metrics']
    assert metrics['stage'] == 'ranked_complete' and 'partial' not in metrics and 'stopped' not in metrics
    assert metrics['outcomes'] == 600 and payload['score'] == 1
    assert all(metrics['models'][alias]['total'] == 200 for alias in e.MODEL_IDS)
    assert metrics['publicPanel']['floor'] == {'minCorrect': 32, 'maxParseFailures': 6, 'passed': True}
    assert metrics['publicPanel']['correct'] == 60 and metrics['publicPanel']['verdicts'] == {alias: '1' * 20 for alias in e.MODEL_IDS}
    assert metrics['billing']['attempts'] == 660
    assert metrics['publicPanel']['tokensIn'] == 60 and metrics['tokensIn'] == 600
    assert costs['status'] == 'succeeded'
    assert 'verdicts' not in metrics and 'questionIds' not in json.dumps(payload)
    assert_problem_text_only_in_panel(payload)
    assert_identity(payload, 'ranked')


@pytest.mark.asyncio
async def test_ranked_stopped_sentinel_is_sanitized(tmp_path, tmp_path_factory, monkeypatch):
    payload, costs = await hosted_run(tmp_path, tmp_path_factory, monkeypatch,
                                      correct_public_then('unclear'), record=473)
    metrics = payload['metrics']
    assert payload['score'] == 0
    assert metrics['stage'] == 'ranked_stopped' and metrics['partial'] is True
    stopped = metrics['stopped']
    # Every ranked answer is a parse failure: 600 - 60k <= 473 first holds after round 3.
    assert stopped == {'roundsCleared': 3, 'roundsTotal': 10, 'roundQuestions': 20, 'completedOutcomes': 180,
                       'correctSoFar': 0, 'recordCorrect': 473}
    assert stopped['roundsCleared'] * 60 == stopped['completedOutcomes'] == metrics['outcomes']
    assert stopped['correctSoFar'] + (600 - stopped['completedOutcomes']) <= 473
    assert metrics['parseFailures'] == 180 and sum(m['total'] for m in metrics['models'].values()) == 180
    assert metrics['billing']['attempts'] == 240
    assert metrics['publicPanel']['floor']['passed'] is True
    assert costs['status'] == 'stopped'
    assert 'verdicts' not in metrics and 'questionIds' not in json.dumps(payload)
    assert_problem_text_only_in_panel(payload)
    assert_identity(payload, 'ranked')


@pytest.mark.asyncio
async def test_public_run_panel_matches_hosted_screen(tmp_path, tmp_path_factory, monkeypatch):
    answers = public_answers()
    def flaky(prompt, model):
        if 'PRIVATE' in prompt:
            return 'VERDICT: TRUE'
        index = list(answers).index(prompt)
        if model.startswith('google/') and index % 4 == 0:
            return 'no idea'
        return verdict(answers[prompt] if index % 5 else not answers[prompt])
    hosted, _ = await hosted_run(tmp_path, tmp_path_factory, monkeypatch, flaky, record=473)
    local_root = tmp_path_factory.mktemp('local')
    monkeypatch.setattr(e, 'ROOT', local_root)
    (local_root / 'submission').mkdir()
    (local_root / 'submission/prompt.txt').write_text(TEMPLATE)
    await e.run(SimpleNamespace(ranked=False), {'DISTILL_LIVE': '1', 'OPENROUTER_API_KEY': 'k', 'DISTILL_MAX_SPEND_USD': '1'})
    local = json.loads((local_root / 'score.json').read_text())
    assert local['metrics']['stage'] == 'public_complete' and 'partial' not in local['metrics']
    assert local['metrics']['publicPanel'] == hosted['metrics']['publicPanel']
    assert local['metrics']['publicPanel']['floor'] == {'minCorrect': 32, 'maxParseFailures': 6, 'passed': True}
    assert local['metrics']['evaluationPolicy']['record'] == 'disabled'
    assert local['score'] == local['metrics']['publicPanel']['correct'] / 60
    assert 'verdicts' not in local['metrics']
    assert_problem_text_only_in_panel(local)
    assert_identity(local, 'public')
    assert_identity(hosted, 'ranked')
    assert local['metrics']['publicPanel']['parseFailures'] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize('stage_answer', ['screen_failed', 'stopped'])
async def test_cost_artifact_failure_on_sentinel_stage_removes_score(tmp_path, tmp_path_factory, monkeypatch, stage_answer):
    env = hosted_fixture(tmp_path, tmp_path_factory, monkeypatch, DISTILL_RECORD_SOURCE_URL=RECORD_URL)
    record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': 473 / 600, 'direction': '+'}})
    answer = (lambda prompt, _model: 'unclear') if stage_answer == 'screen_failed' else correct_public_then('unclear')
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(openrouter(answer)))
    monkeypatch.setattr(e, 'write_cost_artifact', lambda *_: (_ for _ in ()).throw(OSError()))
    with pytest.raises(e.InvalidRun, match='cost_artifact_unavailable'):
        await e.run(SimpleNamespace(ranked=True), env)
    assert not (tmp_path / 'score.json').exists() and not (tmp_path / 'score.json.tmp').exists()


@pytest.mark.asyncio
async def test_private_shuffle_seed_never_leaves_the_process(tmp_path, tmp_path_factory, monkeypatch, capsys):
    sentinel = 'f00dfacef00dfacef00dfacef00dface'
    monkeypatch.setattr(e, 'private_shuffle_seed', lambda: sentinel)
    env = hosted_fixture(tmp_path, tmp_path_factory, monkeypatch, DISTILL_RECORD_SOURCE_URL=RECORD_URL)
    record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': 473 / 600, 'direction': '+'}})
    monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(openrouter(correct_public_then('unclear'))))
    seen = {}
    real_evaluate = e.evaluate
    async def spy(rows, template, complete, **kwargs):
        seen.setdefault('seeds', []).append(kwargs.get('shuffle_seed'))
        return await real_evaluate(rows, template, complete, **kwargs)
    monkeypatch.setattr(e, 'evaluate', spy)
    await e.run(SimpleNamespace(ranked=True), env)
    assert seen['seeds'] == [None, sentinel], 'screen unshuffled, ranked shuffled with the private seed'
    for text in ((tmp_path / 'score.json').read_text(), (tmp_path / 'evaluation-costs.json').read_text(), capsys.readouterr().out):
        assert sentinel not in text and 'f00dface' not in text
    payload = json.loads((tmp_path / 'score.json').read_text())
    assert payload['metrics']['stage'] == 'ranked_stopped'
    assert payload['metrics']['evaluationPolicy']['shuffleSeed'] == 'private-random'
    assert 'a' * 40 == payload['metrics']['candidateSha']


def test_private_shuffle_seed_is_fresh_per_call():
    seeds = {e.private_shuffle_seed() for _ in range(5)}
    assert len(seeds) == 5 and all(len(seed) == 32 and int(seed, 16) >= 0 for seed in seeds)


@pytest.mark.asyncio
async def test_same_candidate_can_stop_at_different_rounds(tmp_path_factory, monkeypatch):
    # Half the private rows are answered correctly, half unparseable, so the stopping round depends
    # on which rows the private order puts first. The two seeds are pinned (found by replaying the
    # shuffle offline): 43 bad rows are needed for the stop, seed ...0 reaches it in round 5, ...1 in 4.
    answers = public_answers()
    def answer(prompt, _model):
        if 'PRIVATE' in prompt:
            index = int(prompt.split()[1])
            return verdict(index % 2 == 0) if index < 100 else 'unclear'
        return verdict(answers[prompt])
    rounds = {}
    for seed in ('0' * 32, '0' * 31 + '1'):
        monkeypatch.setattr(e, 'private_shuffle_seed', lambda seed=seed: seed)
        root = tmp_path_factory.mktemp('run')
        env = hosted_fixture(root, tmp_path_factory, monkeypatch, DISTILL_RECORD_SOURCE_URL=RECORD_URL)
        record_endpoint(monkeypatch, {'benchmark': {'currentBestScore': 473 / 600, 'direction': '+'}})
        monkeypatch.setattr(e, 'base_transport', lambda: httpx.MockTransport(openrouter(answer)))
        await e.run(SimpleNamespace(ranked=True), env)
        payload = json.loads((root / 'score.json').read_text())
        stopped = payload['metrics']['stopped']
        assert stopped['completedOutcomes'] == stopped['roundsCleared'] * 60
        assert payload['metrics']['candidateSha'] == 'a' * 40
        rounds[seed] = stopped['roundsCleared']
    assert rounds == {'0' * 32: 5, '0' * 31 + '1': 4}


@pytest.mark.asyncio
async def test_unknown_stage_is_never_written(tmp_path, tmp_path_factory, monkeypatch):
    env = hosted_fixture(tmp_path, tmp_path_factory, monkeypatch)
    async def fake(*_, **__):
        return {'score': 0.5, 'metrics': {'stage': 'diagnostic', 'models': {}, 'outcomes': 0}}
    with pytest.raises(e.InvalidRun, match='stage_invalid'):
        await e.run(SimpleNamespace(ranked=True), env, fake)
    assert not (tmp_path / 'score.json').exists()
    assert e.STAGES == ('public_complete', 'screen_failed', 'ranked_stopped', 'ranked_complete')


def test_execution_config_key_set_unchanged():
    assert set(e.EXECUTION_CONFIG) == set(e.MODEL_IDS)
    assert e.digest(json.dumps(e.EXECUTION_CONFIG, sort_keys=True, separators=(',', ':')).encode()) == EXECUTION_CONFIG_SHA256
    assert e.digest((e.VENDOR / 'evaluation_models.json').read_bytes()) == CONFIG_SHA256
    assert e.digest(e.SCREEN_DATASET.read_bytes()) == PUBLIC_SHA256


# --- select_dataset.py public dev sets ---

def mocked_sources(monkeypatch, published_pairs=((1, 201),)):
    equations = '\n'.join(f'x = x * y ({i})' for i in range(1, 4695)).encode()
    published = b''.join((json.dumps({'eq1_id': a, 'eq2_id': b}) + '\n').encode() for a, b in published_pairs)
    def fetch(url):
        if url.endswith('full_entries.json'):
            return json.dumps(entries()).encode()
        if url.endswith('equations.txt'):
            return equations
        if '/api/datasets/' in url:
            return json.dumps({'sha': selection.SAIR_SHA, 'siblings': [{'rfilename': 'data/evaluation.jsonl'}, {'rfilename': 'README.md'}]}).encode()
        if url.endswith('data/evaluation.jsonl'):
            return published
        raise AssertionError(f'unexpected fetch {url}')
    monkeypatch.setattr(selection, 'fetch', fetch)


def test_public_dev_set_is_seeded_excluded_and_never_private(tmp_path, monkeypatch):
    mocked_sources(monkeypatch)
    out = tmp_path / 'dev.jsonl'
    manifest = selection.write_public(7, 40, out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 40 and sum(row['answer'] for row in rows) == 20
    assert manifest == {'datasetKind': 'public-dev', 'publicSeed': 7, 'count': 40, 'sha256': e.digest(out.read_bytes())}
    assert {row['id'] for row in rows} == {f'dev_{i:04}' for i in range(1, 41)}
    assert (1, 201) not in {(row['eq1_id'], row['eq2_id']) for row in rows}
    assert all(row['provenance']['sourceSha'] == selection.SOURCE_SHA for row in rows)
    assert 'yukon_' not in out.read_text()
    again = tmp_path / 'again.jsonl'
    selection.write_public(7, 40, again)
    assert again.read_bytes() == out.read_bytes()
    other = tmp_path / 'other.jsonl'
    selection.write_public(8, 40, other)
    assert other.read_bytes() != out.read_bytes()
    with pytest.raises(FileExistsError):
        selection.write_public(7, 40, out)
    with pytest.raises(ValueError):
        selection.write_public(7, 41, tmp_path / 'odd.jsonl')
    with pytest.raises(ValueError):
        selection.write_public(7, 400, tmp_path / 'big.jsonl')
    assert selection.public_seed_bytes(7) != selection.public_seed_bytes(8)
    assert len(selection.public_seed_bytes(7)) == 32
    # A public dev set loads through the evaluator's public (non-ranked) validator.
    assert len(e.load_dataset(out, ranked=False)[0]) == 40


def test_public_dev_set_cli_requires_explicit_seed(tmp_path, monkeypatch, capsys):
    mocked_sources(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['select_dataset.py', '--out', str(tmp_path / 'x.jsonl')])
    with pytest.raises(SystemExit):
        selection.main()
    monkeypatch.setattr(sys, 'argv', ['select_dataset.py', '--public-seed', '3', '--count', '10', '--out', str(tmp_path / 'y.jsonl')])
    selection.main()
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed['publicSeed'] == 3 and printed['count'] == 10 and (tmp_path / 'y.jsonl').exists()
    monkeypatch.setattr(sys, 'argv', ['select_dataset.py', '--public-seed', '3', '--output-dir', str(tmp_path / 'z')])
    with pytest.raises(SystemExit):
        selection.main()
