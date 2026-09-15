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
        assert len(r['messages'])==1 and r['messages'][0]['role']=='user'
    by_model={r['model']:r for r in requests}
    assert by_model['openai/gpt-oss-120b']['provider']['order']==['deepinfra/bf16']
    assert by_model['openai/gpt-oss-120b']['provider']['quantizations']==['bf16']
    assert by_model['openai/gpt-oss-120b']['reasoning']=={'effort':'low'}
    assert by_model['meta-llama/llama-3.3-70b-instruct']['provider']['order']==['deepinfra/turbo']
    assert by_model['meta-llama/llama-3.3-70b-instruct']['provider']['quantizations']==['fp8']
    assert 'reasoning' not in by_model['meta-llama/llama-3.3-70b-instruct']
    assert by_model['google/gemma-4-31b-it']['provider']['order']==['novita/bf16']
    assert by_model['google/gemma-4-31b-it']['provider']['quantizations']==['bf16']
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
    (tmp_path/'submission').mkdir(); (tmp_path/'submission/prompt.txt').write_text(TEMPLATE)
    with pytest.raises(e.InvalidRun): await e.run(SimpleNamespace(ranked=False),{})
    assert not (tmp_path/'score.json').exists()


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
    async def mock(rows, template, _env):
        async def complete(*_): return response('VERDICT: TRUE\nprivate reasoning must not escape')
        return await e.evaluate(rows,template,complete)
    await e.run(SimpleNamespace(ranked=False),env,mock)
    output=(tmp_path/'score.json').read_text()
    assert 'private reasoning' not in output and 'test-secret' not in output
    assert 'equation1' not in output and 'expected' not in output
    assert json.loads(output)['metrics']['datasetKind']=='public'
    async def failed(*_): raise RuntimeError('PRIVATE error body')
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
    assert {row['endpoint'] for row in rows}=={'deepinfra/bf16','deepinfra/turbo','novita/bf16'}
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
