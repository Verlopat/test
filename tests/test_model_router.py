"""Offline regression tests: no credentials and no external network."""
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_router as mr


class Response:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status
        self.ok = 200 <= status < 300

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
        self.assertEqual([a[2] for a in attempts], ['HTTP 503: provider overloaded', 'OK'])

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


if __name__ == '__main__':
    unittest.main()
