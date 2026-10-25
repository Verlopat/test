"""Offline regression checks for simple_router.py, with no external API calls."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

# Inject only the existing module interfaces required by simple_router.py.
# No .env or provider clients are loaded during offline tests.
@dataclass(frozen=True)
class Candidate:
    provider: str
    model: str
    quota_group: str | None = None

class CandidateError(Exception):
    def __init__(self, message, *, kind='other', retry_after=None):
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after

fake_model_router = types.ModuleType('model_router')
fake_model_router.KEY_NAMES = {'Gemini': 'GEMINI_API_KEY', 'LLM7.io': 'LLM7_API_KEY',
                               'Groq': 'GROQ_API_KEY', 'OpenRouter': 'OPENROUTER_API_KEY',
                               'Z.ai': 'ZAI_API_KEY'}
fake_model_router.Router = object
fake_model_router.load_config = lambda path: []
fake_model_router.redacted = str
fake_errors = types.ModuleType('router_errors')
fake_errors.CandidateError = CandidateError
fake_validation = types.ModuleType('answer_validation')
fake_validation.matches_expected = lambda answer, expected: answer.strip().rstrip('.') == expected.strip()
with patch.dict(sys.modules, {'model_router': fake_model_router, 'router_errors': fake_errors,
                              'answer_validation': fake_validation}):
    spec = importlib.util.spec_from_file_location('simple_router', Path(__file__).with_name('simple_router.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


class FakeRouter:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = []
        self.expansions = []

    def expand(self, candidate, seen):
        self.expansions.append((candidate.provider, candidate.model, seen))
        return [candidate]

    def generate(self, candidate, prompt, probe=False):
        self.calls.append((candidate.provider, candidate.model, prompt, probe))
        behavior = self.behavior.get(candidate.model, 'answer')
        if isinstance(behavior, list):
            result = behavior.pop(0)
        else:
            result = behavior
        if isinstance(result, Exception):
            raise result
        return result


class SimpleRouterTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.env = patch.dict(os.environ, {
            'GEMINI_API_KEY': 'placeholder', 'LLM7_API_KEY': 'placeholder',
            'GROQ_API_KEY': 'placeholder', 'ZAI_API_KEY': 'placeholder',
            'OPENROUTER_API_KEY': 'placeholder',
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.pause = patch.object(mod.time, 'sleep')
        self.sleep = self.pause.start()
        self.addCleanup(self.pause.stop)

    def query(self, models, behavior, **options):
        fake = FakeRouter(behavior)
        result = mod.ask(fake, 'Full question: context stays intact\nWhat next?',
                         models, log=self.lines.append, **options)
        return result, fake

    def test_503_retries_same_model_then_uses_backup(self):
        models = [Candidate('Gemini', 'strong'), Candidate('Gemini', 'backup')]
        error = CandidateError('HTTP 503', kind='server')
        (answer, winner), fake = self.query(models, {'strong': [error, error], 'backup': 'success'})
        self.assertEqual((answer, winner.model), ('success', 'backup'))
        self.assertEqual([c[1] for c in fake.calls], ['strong', 'strong', 'backup'])
        self.assertEqual([c[2] for c in fake.calls], ['Full question: context stays intact\nWhat next?']*3)
        self.assertTrue(all(c[3] is False for c in fake.calls))
        self.sleep.assert_called_once_with(2.0)

    def test_timeout_retry_and_success(self):
        models = [Candidate('Gemini', 'first'), Candidate('Groq', 'second')]
        (answer, winner), fake = self.query(models, {'first': [CandidateError('timeout', kind='timeout'), '156']})
        self.assertEqual(winner.model, 'first')
        self.assertEqual(len(fake.calls), 2)

    def test_rate_limit_with_long_retry_after_skips_shared_quota(self):
        models = [Candidate('Gemini', 'first'), Candidate('Gemini', 'backup'),
                  Candidate('Groq', 'working')]
        rate = CandidateError('HTTP 429', kind='rate_limit', retry_after=180)
        (answer, winner), fake = self.query(models, {'first': rate, 'backup': 'should not call', 'working': 'good'})
        self.assertEqual(winner.model, 'working')
        self.assertEqual([call[1] for call in fake.calls], ['first', 'working'])
        self.sleep.assert_not_called()

    def test_rate_limit_with_short_retry_after_retries(self):
        models = [Candidate('Gemini', 'first')]
        rate = CandidateError('HTTP 429', kind='rate_limit', retry_after=3)
        (answer, winner), fake = self.query(models, {'first': [rate, 'ok']})
        self.assertEqual(winner.model, 'first')
        self.sleep.assert_called_once_with(3)

    def test_known_independent_group_fallback(self):
        models = [Candidate('LLM7.io', 'one', 'pool1'), Candidate('LLM7.io', 'two', 'pool2')]
        rate = CandidateError('HTTP 429', kind='rate_limit', retry_after=99)
        (answer, winner), fake = self.query(models, {'one': rate, 'two': 'fallback'})
        self.assertEqual(winner.model, 'two')

    def test_wrong_api_key_skips_provider(self):
        models = [Candidate('Gemini', 'one'), Candidate('Gemini', 'two'), Candidate('Groq', 'three')]
        dead = set()
        (answer, winner), fake = self.query(models, {'one': CandidateError('401', kind='auth'), 'three': 'yes'}, dead_providers=dead)
        self.assertEqual(winner.model, 'three')
        self.assertEqual([c[1] for c in fake.calls], ['one', 'three'])
        self.assertIn('Gemini', dead)

    def test_bad_catalog_auth_skips_provider(self):
        fake = FakeRouter({'b': 'fine'})
        def expand(candidate, seen):
            if candidate.provider == 'Gemini':
                raise CandidateError('401', kind='auth')
            return [candidate]
        fake.expand = expand
        answer, winner = mod.ask(fake, 'prompt', [Candidate('Gemini', 'a'), Candidate('Gemini', 'a2'), Candidate('Groq', 'b')], log=self.lines.append)
        self.assertEqual(winner.provider, 'Groq')

    def test_bad_reply_and_empty_fallback(self):
        models = [Candidate('Gemini', 'one'), Candidate('Gemini', 'two'), Candidate('Groq', 'three')]
        (answer, winner), fake = self.query(models, {'one': 'Service unavailable',
                                                   'two': '  ', 'three': 'research answer'})
        self.assertEqual(winner.model, 'three')
        self.assertEqual(len(fake.calls), 3)
        self.sleep.assert_not_called()

    def test_legitimate_rate_limit_explanation_is_not_discarded(self):
        text = 'The rate limit is 100 requests per minute, which determines your throughput.'
        self.assertTrue(mod.is_real_answer(text))
        self.assertFalse(mod.is_real_answer('Rate limit, try again later'))
        self.assertFalse(mod.is_real_answer('Sorry, the server is overloaded.'))
        self.assertTrue(mod.is_real_answer('An API may say service unavailable during overload.'))

    def test_known_wrong_answer_fallback_only_with_expect(self):
        models = [Candidate('Gemini', 'one'), Candidate('Groq', 'two')]
        (answer, winner), fake = self.query(models, {'one': '155', 'two': '156'}, expected='156')
        self.assertEqual(winner.model, 'two')
        self.assertEqual(answer, '156')
        self.assertEqual(len(fake.calls), 2)

    def test_duplicate_candidate_id_skipped(self):
        models = [Candidate('Gemini', 'one'), Candidate('Gemini', 'one'), Candidate('Groq', 'three')]
        err = CandidateError('unavailable', kind='model_unavailable')
        (answer, winner), fake = self.query(models, {'one': err, 'three': 'ok'})
        self.assertEqual([c[1] for c in fake.calls], ['one', 'three'])
        self.assertIs(fake.expansions[0][2], fake.expansions[1][2])

    def test_context_length_skips_without_retry(self):
        models = [Candidate('Gemini', 'small'), Candidate('Gemini', 'large')]
        (answer, winner), fake = self.query(models, {'small': CandidateError('context too large', kind='context_length'),
                                                     'large': 'works'})
        self.assertEqual(winner.model, 'large')
        self.sleep.assert_not_called()

    def test_unreachable_model_moves_without_retry(self):
        models = [Candidate('Gemini', 'first'), Candidate('Groq', 'backup')]
        (answer, winner), fake = self.query(models, {'first': CandidateError('network problem', kind='connection'),
                                                   'backup': 'works'})
        self.assertEqual([c[1] for c in fake.calls], ['first', 'backup'])
        self.sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
