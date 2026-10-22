import json
from types import SimpleNamespace

import srim_agent as agent

TOOL_RESULT = json.dumps({'prediction': {'energy_MeV': 18.0, 'fluence': 1.283e16, 'roi_start_nm': 469.5,
                                         'max_spread': 0.1, 'hours': 26.7}})


def test_numbers_in_reads_scientific_forms():
    values = [v for v, _, _ in agent.numbers_in('1.28 × 10¹⁶, 1.28e+16, 1.28x10^16, 1,000 ions, 205 600 s')]
    assert values[:3] == [1.28e16] * 3 and values[3:] == [1000, 205600]


def test_check_accepts_tool_numbers_and_rounding():
    answer = 'Predicted: 18 MeV, 1.28 × 10¹⁶ ions/cm² (1.3e16), ROI from 470 nm, ±10 % flatness, 26.7 h.'
    assert agent.check_numbers(answer, [TOOL_RESULT]) == []


def test_check_flags_invented_numbers():
    answer = 'Use 22 MeV and 2.5e16 ions/cm2; the ROI starts at 469.5 nm.'
    assert agent.check_numbers(answer, [TOOL_RESULT]) == ['22', '2.5e16']


def test_check_does_not_accept_wrong_rounding():
    # 1.283e16 rounds to 1.28e16, not 1.29e16
    assert agent.check_numbers('1.29e16 ions/cm2', [TOOL_RESULT]) == ['1.29e16']


# --- the loop, with a scripted stand-in for the Groq client ------------------------------------------------

def _message(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


def _tool_call(name, args, id_='c1'):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class ScriptedClient:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.seen.append([dict(m) for m in kwargs['messages']])
        return self.replies.pop(0)


def test_loop_calls_tools_and_returns_answer():
    client = ScriptedClient([
        _message(tool_calls=[_tool_call('beam_time', {'fluence': 1e16, 'current_nA': 50, 'charge_state': 3})]),
        _message('About 26.7 hours.'),
    ])
    a = agent.Agent(client=client)
    events = []
    reply = a.ask('How long for 1e16 at 50 nA, 3+?', lambda k, d: events.append(k))
    assert reply.text == 'About 26.7 hours.' and reply.unverified == []
    assert reply.tool_calls[0]['name'] == 'beam_time' and reply.tool_calls[0]['result']['hours'] == 26.7
    assert events == ['tool_call', 'tool_result']
    assert client.seen[1][-1]['role'] == 'tool'


def test_invented_number_triggers_one_rewrite_and_clean_history():
    client = ScriptedClient([
        _message('It takes 99.9 hours.'),          # no tool gave 99.9
        _message('I need the fluence first.'),
    ])
    a = agent.Agent(client=client)
    events = []
    reply = a.ask('How long will it take?', lambda k, d: events.append((k, d)))
    assert events[0] == ('rewrite', {'unsupported': ['99.9']})
    assert reply.text == 'I need the fluence first.' and reply.unverified == []
    assert '[Automatic check]' in client.seen[1][-1]['content']
    assert [m['role'] for m in a.messages] == ['system', 'user', 'assistant']   # draft and check removed


def test_unsupported_numbers_after_rewrite_are_reported():
    client = ScriptedClient([_message('99.9 hours.'), _message('Still 99.9 hours.')])
    reply = agent.Agent(client=client).ask('How long?')
    assert reply.unverified == ['99.9']


# --- approval before TRIM ----------------------------------------------------------------------------------

PLAN_ARGS = {'material': 'W', 'ion': 'W', 'target_dpa': 1, 'facility': 'Helsinki 5 MV tandem'}


def _plan_client(final):
    return ScriptedClient([_message(tool_calls=[_tool_call('plan_beam', PLAN_ARGS)]), _message(final)])


def test_trim_run_waits_for_approval(monkeypatch):
    ran = []
    monkeypatch.setattr(agent.tools, 'call_tool', lambda *a, **k: ran.append(a) or json.dumps({'ok': True}))
    a = agent.Agent(client=_plan_client('Confirmed.'))
    reply = a.ask('Confirm with TRIM')
    assert reply.pending == {'name': 'plan_beam', 'args': PLAN_ARGS} and reply.text == '' and not ran
    try:
        a.ask('something else')
        assert False, 'a new question must wait for the approval'
    except RuntimeError:
        pass
    reply = a.resume(True)
    assert len(ran) == 1 and reply.text == 'Confirmed.' and reply.pending is None
    assert reply.tool_calls[0]['name'] == 'plan_beam'


def test_declined_trim_run_is_not_run_and_the_model_is_told(monkeypatch):
    ran = []
    monkeypatch.setattr(agent.tools, 'call_tool', lambda *a, **k: ran.append(a) or json.dumps({'ok': True}))
    client = _plan_client('Keeping the prediction.')
    a = agent.Agent(client=client)
    a.ask('Confirm with TRIM')
    reply = a.resume(False)
    assert not ran and reply.text == 'Keeping the prediction.'
    assert reply.tool_calls[0]['result']['declined'] is True
    assert client.seen[-1][-1]['role'] == 'tool'   # the model saw the decline as the tool's answer


def test_approval_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(agent.tools, 'call_tool', lambda *a, **k: json.dumps({'ok': True}))
    reply = agent.Agent(client=_plan_client('Done.'), approval=False).ask('Confirm with TRIM')
    assert reply.pending is None and reply.text == 'Done.'
