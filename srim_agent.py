# LLM interface (Part 2): a chat agent that answers beam-planning questions by calling the physics core
# through srim_tools, on Groq's OpenAI-compatible API. A plain tool-calling loop, no framework.
#
# The rule that makes it a science tool rather than a chatbot: every number in an answer must come from a
# tool result (or the user). check_numbers() enforces it after each answer; an answer that quotes a number
# no tool produced is sent back once to be rewritten, and if it still does, the reply carries the numbers
# as unverified so the app can show a warning.
#
#   python srim_agent.py                      chat in the terminal
#   python srim_agent.py "question"           one question
#
# Needs GROQ_API_KEY (a user environment variable). GROQ_MODEL overrides the model.

import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import srim_tools as tools

DEFAULT_MODEL = 'openai/gpt-oss-120b'
MAX_STEPS = 10            # model calls per question, tool rounds included
MAX_REWRITES = 1          # times an answer with unsupported numbers is sent back

SYSTEM_PROMPT = """You help nuclear materials scientists plan heavy-ion irradiation experiments: which ion \
energy and fluence give a target dpa (displacements per atom) in a region of interest (ROI) below the \
surface of a sample. The physics is done by tools built on SRIM-2013 TRIM and a surrogate model trained on it.

Rules:
1. Every number you state about a beam, dose, depth, time or rate must come from a tool result in this \
conversation, or from the user. Copy numbers as the tool gives them (you may round to fewer significant \
figures). Never estimate, convert units or do arithmetic yourself: for beam time call beam_time.
2. screen_beam numbers are PREDICTED by a surrogate model: say so. plan_beam numbers are CONFIRMED by TRIM: \
say so. Never present a prediction as confirmed.
3. If the user has not given the target dpa, the material, the ion, and a facility or energy range, ask for \
what is missing. Do not guess them. Call list_options when unsure of a material or facility name.
4. Results list defaults_used: tell the user briefly which defaults the answer assumed (ROI start, width, \
flatness, implant margin, objective) so they can change them.
5. Pass on flags and failure messages, including the values they suggest (for example "raise the maximum \
energy to about 21.2 MeV"). Never claim that another facility, energy or setting would work unless a tool \
result in this conversation shows it: check with energy_window or screen_beam first, and if you have not \
checked, say so.
5b. Say where numbers come from: "status" in each result tells you (surrogate prediction, SR ranges, or \
TRIM).
6. Use screen_beam first. plan_beam runs TRIM for 2-15 minutes: call it only when the user asks for \
verified or confirmed numbers.
7. Be brief and concrete. The users are physicists: use units, skip basics. For a beam answer: one sentence \
with the answer and whether it is predicted or confirmed, then a two-column table (Quantity | Value) with \
these rows when the result has them: beam energy, fluence, ROI depth (start-end and width), dpa range in the \
ROI, peak dpa, charge state and terminal voltage, and beam time if asked. Do not repeat inputs the user gave \
(current, area). Then one line naming the defaults assumed. Do not name tools or JSON fields, and do not list numbers the user did not need; they \
can open the tool calls for the rest."""

# Added when the app draws charts under the answer (not in the terminal)
CHARTS_NOTE = """
8. The app shows charts under your answer for every screen_beam and plan_beam result: the dpa and \
implanted-ion depth profiles with the ROI marked, and the objective against beam energy. You can point to \
them ("see the depth profile below"), but read numbers only from the tool results, not from the charts."""


# ---------------------------------------------------------------------------
# Number check: every number in an answer must appear in a source (tool result, user message, prompt)
# ---------------------------------------------------------------------------

_SUPERSCRIPT = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
_NUMBER = re.compile(
    r'(?<![\w.])(\d+(?:\.\d+)?)'                              # mantissa
    r'(?:\s*[×x*]\s*10\s*(?:\^\s*)?([-+−]?\d+|[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)'   # x 10^16, × 10¹⁶
    r'|[eE]([-+]?\d+))?')                                     # 1.28e16, 1.28e+16


def numbers_in(text: str) -> List[tuple]:
    # (value, significant figures, as written) for every number in the text
    text = re.sub(r'(?<=\d)[,    ](?=\d{3}\b)', '', text)   # 1,000 and 205 600 -> one number
    out = []
    for m in _NUMBER.finditer(text):
        mantissa, exp = m.group(1), m.group(2) or m.group(3)
        value = float(mantissa)
        if exp:
            value *= 10.0 ** int(exp.translate(_SUPERSCRIPT).replace('−', '-'))
        digits = mantissa.replace('.', '').lstrip('0') or '0'
        sig = len(digits.rstrip('0')) if '.' not in mantissa else len(digits)
        out.append((value, max(sig, 1), m.group(0)))
    return out


def _matches(quoted: float, sig: int, source: float) -> bool:
    # quoted is source rounded to `sig` significant figures (or quoted exactly)
    if source == 0 or quoted == 0:
        return source == quoted
    rounded = float(f'{source:.{sig}g}')
    return math.isclose(quoted, rounded, rel_tol=1e-9) or math.isclose(quoted, source, rel_tol=1e-9)


def check_numbers(answer: str, sources: List[str]) -> List[str]:
    # Numbers in the answer that no source supports. Small integers (list numbering, "2 options") are
    # allowed; percentages may quote a fraction (0.1 -> 10 %).
    pool = [v for s in sources for v, _, _ in numbers_in(s)]
    bad = []
    for value, sig, written in numbers_in(answer):
        if value == int(value) and abs(value) <= 10:
            continue
        candidates = (value, value / 100)
        if not any(_matches(q, sig, s) for q in candidates for s in pool):
            bad.append(written)
    return bad


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def _windows_user_env(name: str) -> Optional[str]:
    # A user environment variable saved after this process's parent started (e.g. a VS Code terminal opened
    # before the key was set) is in the registry but not in os.environ
    if sys.platform != 'win32':
        return None
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as k:
            return winreg.QueryValueEx(k, name)[0] or None
    except OSError:
        return None


@dataclass
class Reply:
    text: str
    unverified: List[str] = field(default_factory=list)   # numbers no tool or user gave (should be empty)
    tool_calls: List[dict] = field(default_factory=list)  # {name, args, result} in order


EventFn = Callable[[str, dict], None]   # ('tool_call' | 'tool_result' | 'progress' | 'rewrite', data)


class Agent:
    """A conversation. ask() can be called repeatedly; the history is kept in self.messages."""

    def __init__(self, model: Optional[str] = None, client=None, temperature: float = 0.2,
                 srim_dir: Optional[str] = None, charts: bool = False):
        self.srim_dir = srim_dir   # SRIM folder for screen_beam and plan_beam (default: srim_core's)
        prompt = SYSTEM_PROMPT + (CHARTS_NOTE if charts else '')
        if client is None:
            from groq import Groq
            key = os.environ.get('GROQ_API_KEY') or _windows_user_env('GROQ_API_KEY')
            if not key:
                raise RuntimeError('GROQ_API_KEY is not set. Set it as a user environment variable '
                                   '(Start > "Edit environment variables for your account").')
            client = Groq(api_key=key)
        self.client, self.temperature = client, temperature
        self.model = model or os.environ.get('GROQ_MODEL') or DEFAULT_MODEL
        self.messages: List[dict] = [{'role': 'system', 'content': prompt}]
        self.sources: List[str] = [prompt, json.dumps(tools.TOOLS)]

    def _complete(self):
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(
                    model=self.model, messages=self.messages, tools=tools.TOOLS, tool_choice='auto',
                    temperature=self.temperature)
            except Exception as e:   # rate limits on the free tier: wait and retry
                if 'rate' not in type(e).__name__.lower() or attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

    def ask(self, text: str, on_event: Optional[EventFn] = None) -> Reply:
        emit = on_event or (lambda kind, data: None)
        self.messages.append({'role': 'user', 'content': text})
        self.sources.append(text)
        calls, rejected = [], []
        for _ in range(MAX_STEPS):
            msg = self._complete().choices[0].message
            if msg.tool_calls:
                self.messages.append({'role': 'assistant', 'content': msg.content or '', 'tool_calls': [
                    {'id': c.id, 'type': 'function',
                     'function': {'name': c.function.name, 'arguments': c.function.arguments}}
                    for c in msg.tool_calls]})
                for c in msg.tool_calls:
                    try:
                        args = json.loads(c.function.arguments or '{}')
                    except json.JSONDecodeError:
                        args = None
                    emit('tool_call', {'name': c.function.name, 'args': args})
                    raw = {}
                    result = (tools.call_tool(c.function.name, args, srim_dir=self.srim_dir, raw=raw,
                                              progress=lambda *p: emit('progress', dict(zip(
                                                  ('stage', 'step', 'total', 'message'), p))))
                              if args is not None else
                              json.dumps({'ok': False, 'message': 'Arguments were not valid JSON.'}))
                    # raw (full result objects, for charts) stays local; only `result` goes to the model
                    calls.append({'name': c.function.name, 'args': args, 'result': json.loads(result),
                                  'raw': raw or None})
                    emit('tool_result', calls[-1])
                    self.sources.append(result)
                    self.messages.append({'role': 'tool', 'tool_call_id': c.id, 'content': result})
                continue

            answer = msg.content or ''
            bad = check_numbers(answer, self.sources)
            self.messages.append({'role': 'assistant', 'content': answer})
            if bad and len(rejected) < 2 * MAX_REWRITES:
                emit('rewrite', {'unsupported': bad})
                self.messages.append({'role': 'user', 'content': (
                    f'[Automatic check] These numbers in your answer do not appear in any tool result: '
                    f'{", ".join(bad)}. Rewrite the answer using only numbers from tool results; call a tool '
                    f'if you need a number you do not have. Do not mention this check.')})
                rejected += self.messages[-2:]
                continue
            # keep the history clean: drop rejected drafts and the check messages
            self.messages = [m for m in self.messages if not any(m is r for r in rejected)]
            return Reply(answer, bad, calls)
        return Reply('I could not finish this within the step limit. Try a narrower question.', [], calls)


# ---------------------------------------------------------------------------
# Terminal chat
# ---------------------------------------------------------------------------

def _print_event(kind: str, data: dict):
    dim = lambda s: f'\033[2m{s}\033[0m'   # noqa: E731
    if kind == 'tool_call':
        print(dim(f'  -> {data["name"]}({json.dumps(data["args"])})'))
    elif kind == 'tool_result':
        print(dim(f'  <- {data["result"].get("status") or ("ok" if data["result"].get("ok", True) else "failed")}'))
    elif kind == 'progress':
        print(dim(f'     [{data["stage"]} {data["step"]}/{data["total"]}] {data["message"]}'))
    elif kind == 'rewrite':
        print(dim(f'  (check: unsupported numbers {data["unsupported"]}, asking for a rewrite)'))


def main(argv: List[str]):
    agent = Agent()
    print(f'srim_opt agent ({agent.model}). Ctrl+C to quit.\n')
    questions = [' '.join(argv)] if argv else None
    while True:
        try:
            q = questions.pop(0) if questions else (None if questions is not None else input('you> ').strip())
        except (EOFError, KeyboardInterrupt):
            return
        if q is None:
            return
        if not q:
            continue
        reply = agent.ask(q, _print_event)
        print(f'\nagent> {reply.text}\n')
        if reply.unverified:
            print(f'WARNING: these numbers did not come from a tool: {", ".join(reply.unverified)}\n')


if __name__ == '__main__':
    main(sys.argv[1:])
