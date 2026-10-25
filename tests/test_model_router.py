"""Offline regression tests: no credentials and no external network."""
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_router as mr


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data, self.status_code = data, status
        self.ok = 200 <= status < 300
        self.headers = headers or {}
        import json
        self.text = json.dumps(data)

    def json(self):
        return self.data


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.router = mr.Router(timeout=2)
        self.notes = []

    def test_provider_fallback_on_503_then_success(self):
        choices = [mr.Candidate('Groq', 'openai/gpt-oss-120b'),
                   mr.Candidate('Z.ai', 'glm-4.7-flash')]

        def stub(method, url, **kwargs):
            if 'groq.com' in url:
                return Response({'error': 'overloaded'}, 503)
            return Response({'choices': [{'message': {'content': '156'}, 'finish_reason': 'stop'}]})

        with patch.dict(os.environ, {'GROQ_API_KEY': 'testing-groq', 'ZAI_API_KEY': 'testing-zai'}), \
             patch.object(mr.requests, 'request', side_effect=stub):
            answer, winner, attempts = self.router.ask('12*13', choices, log=self.notes.append)

        self.assertEqual(answer, '156')
        self.assertEqual(winner.provider, 'Z.ai')
        self.assertEqual([a[2] for a in attempts], ['HTTP 503: provider/server error', 'OK'])

    def test_wrong_known_answer_triggers_next(self):
        choices = [mr.Candidate('Groq', 'a'), mr.Candidate('Z.ai', 'b')]

        def stub(method, url, **kwargs):
            content = '155' if 'groq.com' in url else '156'
            return Response({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]})

        with patch.dict(os.environ, {'GROQ_API_KEY': 'test', 'ZAI_API_KEY': 'test'}), \
             patch.object(mr.requests, 'request', side_effect=stub):
            result = self.router.ask('12*13', choices, expected='156', log=self.notes.append)

        self.assertEqual(result[1].provider, 'Z.ai')

    def test_empty_or_incomplete_reasoning_not_returned(self):
        with self.assertRaises(mr.CandidateError):
            mr.clean_answer('<think>reasoning only</think>')
        with self.assertRaises(mr.CandidateError):
            mr.clean_answer('hello <think>truncated')
        self.assertEqual(mr.clean_answer('156<think>inner</think>'), '156')

    def test_openrouter_only_zero_price_free_models(self):
        catalog = {'data': [
            {'id': 'a:free', 'pricing': {'prompt': '0', 'completion': '0.0'}},
            {'id': 'b:free', 'pricing': {'prompt': '0', 'completion': '0.1'}},
            {'id': 'c:free', 'pricing': {'prompt': '0', 'completion': '0'}},
            {'id': 'paid', 'pricing': {'prompt': '0', 'completion': '0'}}
        ]}
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test'}), \
             patch.object(mr.requests, 'request', return_value=Response(catalog)):
            selected = self.router.expand(mr.Candidate('OpenRouter', 'auto-free'), set())
            self.assertEqual([c.model for c in selected], ['a:free', 'c:free'])
            with self.assertRaises(mr.CandidateError):
                self.router.expand(mr.Candidate('OpenRouter', 'b:free'), set())

    def test_unconfigured_provider_skips_to_next(self):
        choices = [mr.Candidate('NVIDIA NIM', 'model'), mr.Candidate('Z.ai', 'glm-4.7-flash')]
        response = Response({'choices': [{'message': {'content': 'good'}, 'finish_reason': 'stop'}]})

        with patch.dict(os.environ, {'NVIDIA_API_KEY': '', 'ZAI_API_KEY': 'test'}), \
             patch.object(mr.requests, 'request', return_value=response):
            result = self.router.ask('Hello', choices, log=self.notes.append)

        self.assertEqual(result[1].provider, 'Z.ai')
        self.assertTrue(any(x.startswith('SKIP NVIDIA NIM') for x in self.notes))

    def test_llm7_reasoning_clean_and_truncation_fails(self):
        success = Response({'choices': [{'message': {'content': '<think>stuff</think>156'}, 'finish_reason': 'stop'}]})
        length = Response({'choices': [{'message': {'content': '156'}, 'finish_reason': 'length'}]})

        with patch.dict(os.environ, {'GROQ_API_KEY': 'test'}):
            with patch.object(mr.requests, 'request', return_value=success):
                answer = self.router.generate(mr.Candidate('Groq', 'model'), 'prompt')
            self.assertEqual(answer, '156')
            with patch.object(mr.requests, 'request', return_value=length):
                with self.assertRaises(mr.CandidateError):
                    self.router.generate(mr.Candidate('Groq', 'model'), 'prompt')

    def test_config_rank_and_no_mistral(self):
        path = Path(__file__).resolve().parents[1] / 'model_priority.json'
        candidates = mr.load_config(path)
        self.assertGreaterEqual(len(candidates), 8)
        self.assertNotIn('Mistral', {c.provider for c in candidates})

    def test_429_auth_failure_then_nvidia_success(self):
        choices = [mr.Candidate('Groq', 'a'), mr.Candidate('Z.ai', 'b'),
                   mr.Candidate('NVIDIA NIM', 'c')]

        def stub(method, url, **kwargs):
            if 'groq.com' in url:
                return Response({}, 429)
            if 'api.z.ai' in url:
                return Response({}, 401)
            return Response({'choices': [{'message': {'content': 'winner'}, 'finish_reason': 'stop'}]})

        with patch.dict(os.environ, {'GROQ_API_KEY': 'test', 'ZAI_API_KEY': 'test', 'NVIDIA_API_KEY': 'test'}), \
             patch.object(mr.requests, 'request', side_effect=stub):
            result = self.router.ask('hello', choices, log=self.notes.append)

        self.assertEqual(result[1].provider, 'NVIDIA NIM')
        self.assertEqual(len(result[2]), 3)


    def test_numeric_answer_formats(self):
        accepted = [
            '156', '156.0', 'The answer is **156**.',
            'Answer: 156', '12 * 13 = 156',
            '12 multiplied by 13 is 156',
            'Result = 156',
            r'\boxed{156}',
            'Calculation:\nAnswer: 156'
        ]
        for response in accepted:
            with self.subTest(response=response):
                self.assertTrue(mr.matches_expected(response, '156'))
        for response in ('155', 'Answer: 155', '12 * 13 = 155',
                         'A random unrelated mention of 156 somewhere.',
                         '156 is not the answer: it is 155'):
            with self.subTest(response=response):
                self.assertFalse(mr.matches_expected(response, '156'))
        self.assertFalse(mr.matches_expected('Answer: yes', 'no'))

    def test_inline_reasoning_no_longer_concatenates_numbers(self):
        self.assertEqual(mr.clean_answer('156<think>thinking</think>156'), '156')
        self.assertEqual(mr.clean_answer('156<think>thinking</think> 156'), '156')
        self.assertEqual(mr.clean_answer('<think>thinking</think>156'), '156')

    def test_formatted_known_answer_accepted_before_fallback(self):
        candidates = [mr.Candidate('Groq', 'first'), mr.Candidate('Z.ai', 'second')]
        response = Response({'choices': [{'message': {'content': 'The answer is **156**.'},
                                          'finish_reason': 'stop'}]})
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test', 'ZAI_API_KEY': 'test'}), \
             patch.object(mr.requests, 'request', return_value=response) as http:
            answer, winner, attempts = self.router.ask(
                '12 * 13', candidates, expected='156', log=self.notes.append)
        self.assertEqual(winner.provider, 'Groq')
        self.assertEqual(answer, 'The answer is **156**.')
        self.assertEqual(http.call_count, 1)
        self.assertEqual(len(attempts), 1)

    def test_rejected_answer_preview_is_opt_in(self):
        candidates = [mr.Candidate('Groq', 'first')]
        response = Response({'choices': [{'message': {'content': 'The answer is 155'},
                                          'finish_reason': 'stop'}]})
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test'}), \
             patch.object(mr.requests, 'request', return_value=response):
            with self.assertRaises(mr.CandidateError):
                self.router.ask('test', candidates, expected='156', log=self.notes.append)
        self.assertFalse(any('155' in note for note in self.notes))
        self.notes.clear()
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test'}), \
             patch.object(mr.requests, 'request', return_value=response):
            with self.assertRaises(mr.CandidateError):
                self.router.ask('test', candidates, expected='156',
                                show_rejected=True, log=self.notes.append)
        self.assertTrue(any('155' in note for note in self.notes))



    def test_probe_fail_then_gemini_backup_before_other_provider(self):
        candidates = [
            mr.Candidate('Gemini', 'gemini-3.8-flash'),
            mr.Candidate('Gemini', 'gemini-3.6-flash'),
            mr.Candidate('Groq', 'openai/gpt-oss-120b'),
        ]
        calls = []

        def fake_api(method, url, **kwargs):
            if method == 'GET':
                return {'name': 'models/found'}
            prompt = kwargs['payload']['contents'][0]['parts'][0]['text']
            calls.append((url, prompt))
            if 'gemini-3.8-flash' in url:
                raise mr.CandidateError('HTTP 503: provider overloaded')
            answer = 'OK' if prompt.startswith('Hi.') else '156'
            return {'candidates': [{'content': {'parts': [{'text': answer}]},
                                    'finishReason': 'STOP'}]}

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test', 'GROQ_API_KEY': 'test'}), \
             patch.object(mr, 'api_call', side_effect=fake_api):
            answer, winner, attempts = self.router.ask(
                'What is 12 times 13?', candidates, expected='156',
                probe=True, log=self.notes.append)
        self.assertEqual(answer, '156')
        self.assertEqual(winner, candidates[1])
        self.assertEqual(len(calls), 3)
        self.assertTrue(all('gemini' in url for url, _ in calls))
        self.assertEqual([prompt for _, prompt in calls],
                         ['Hi. Reply with OK.', 'Hi. Reply with OK.', 'What is 12 times 13?'])
        self.assertEqual(len(attempts), 2)
        self.assertTrue(any('PROBE FAIL Gemini/gemini-3.8-flash' in x for x in self.notes))
        self.assertTrue(any('PROBE OK Gemini/gemini-3.6-flash' in x for x in self.notes))
        self.assertTrue(any('QUERY PASS Gemini/gemini-3.6-flash' in x for x in self.notes))

    def test_probe_ok_real_query_fails_then_backup(self):
        candidates = [mr.Candidate('Gemini', 'gemini-3.8-flash'),
                      mr.Candidate('Gemini', 'gemini-3.6-flash')]
        calls = []

        def fake_api(method, url, **kwargs):
            if method == 'GET':
                return {'name': 'models/found'}
            prompt = kwargs['payload']['contents'][0]['parts'][0]['text']
            calls.append((url, prompt, kwargs['timeout'],
                          kwargs['payload']['generationConfig']['maxOutputTokens']))
            if 'gemini-3.8-flash' in url and not prompt.startswith('Hi.'):
                raise mr.CandidateError('HTTP 429: rate limit or quota exceeded')
            answer = 'OK' if prompt.startswith('Hi.') else '156'
            return {'candidates': [{'content': {'parts': [{'text': answer}]},
                                    'finishReason': 'STOP'}]}

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test'}), \
             patch.object(mr, 'api_call', side_effect=fake_api):
            answer, winner, attempts = self.router.ask(
                'real question', candidates, expected='156',
                probe=True, log=self.notes.append)
        self.assertEqual(answer, '156')
        self.assertEqual(winner.model, 'gemini-3.6-flash')
        self.assertEqual(len(calls), 4)
        self.assertEqual([x[2] for x in calls], [8.0, 2, 8.0, 2])
        self.assertEqual([x[3] for x in calls], [192, 700, 192, 700])
        self.assertTrue(any(x.startswith('QUERY FAIL Gemini/gemini-3.8-flash') for x in self.notes))

    def test_all_gemini_fail_then_llm7_main_fails_then_llm7_backup(self):
        candidates = [
            mr.Candidate('Gemini', 'gemini-3.8-flash'),
            mr.Candidate('Gemini', 'gemini-3.6-flash'),
            mr.Candidate('LLM7.io', 'minimax-m2.7'),
            mr.Candidate('LLM7.io', 'codestral-latest'),
            mr.Candidate('Groq', 'last'),
        ]
        calls = []

        def fake_api(method, url, **kwargs):
            if method == 'GET' and 'gemini' in url:
                return {'name': 'models/ok'}
            if method == 'GET' and 'llm7' in url:
                return {'data': [{'id': x, 'tier': 'turbo', 'model_type': 'chat',
                                  'usage_based_only': False}
                                 for x in ('minimax-m2.7', 'codestral-latest')]}
            if 'gemini' in url:
                raise mr.CandidateError('provider overloaded')
            model = kwargs['payload']['model']
            prompt = kwargs['payload']['messages'][0]['content']
            calls.append((model, prompt))
            if model == 'minimax-m2.7':
                raise mr.CandidateError('rate limit')
            return {'choices': [{'message': {'content': 'OK' if prompt.startswith('Hi.') else '156'},
                                 'finish_reason': 'stop'}]}

        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test', 'LLM7_API_KEY': 'test',
                                     'GROQ_API_KEY': 'test'}), \
             patch.object(mr, 'api_call', side_effect=fake_api), \
             patch.object(self.router, '_pace_llm7'):
            answer, winner, attempts = self.router.ask(
                '12 * 13?', candidates, expected='156',
                probe=True, log=self.notes.append)
        self.assertEqual(answer, '156')
        self.assertEqual(winner, mr.Candidate('LLM7.io', 'codestral-latest'))
        self.assertEqual([model for model, _ in calls],
                         ['minimax-m2.7', 'codestral-latest', 'codestral-latest'])
        self.assertEqual(len(attempts), 4)
        self.assertEqual(self.notes[-1].split()[0:2], ['QUERY', 'PASS'])

    def test_provider_order_defaults_to_gemini_then_llm7(self):
        names = [c.provider for c in mr.load_config(
            Path(__file__).resolve().parents[1] / 'model_priority.json')]
        self.assertEqual(names[:3], ['Gemini'] * 3)
        self.assertEqual(names[3:7], ['LLM7.io'] * 4)
        self.assertNotIn('Mistral', names)

    def test_no_probe_option_preserves_single_request(self):
        candidates = [mr.Candidate('Groq', 'openai/gpt-oss-120b')]
        response = Response({'choices': [{'message': {'content': '156'},
                                          'finish_reason': 'stop'}]})
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test'}), \
             patch.object(mr.requests, 'request', return_value=response) as mock_request:
            answer, winner, _ = self.router.ask('12*13', candidates,
                                                expected='156', probe=False, log=self.notes.append)
        self.assertEqual(answer, '156')
        self.assertEqual(mock_request.call_count, 1)

    def test_llm7_pacing_between_catalog_and_probe(self):
        with patch.object(mr.time, 'monotonic', side_effect=[100.0, 100.2, 101.4]), \
             patch.object(mr.time, 'sleep') as sleeper:
            router = mr.Router()
            router._pace_llm7()
            router._pace_llm7()
        sleeper.assert_called_once()
        self.assertAlmostEqual(sleeper.call_args[0][0], 0.9, places=4)



if __name__ == '__main__':
    unittest.main()
