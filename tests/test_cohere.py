"""Offline tests for Cohere's Chat v2 adapter and configured backups."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import model_router as mr
from router_errors import CandidateError, from_http


class CohereTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'COHERE_API_KEY': 'fake-trial-key'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.router = mr.Router(timeout=5)
        self.candidate = mr.Candidate('Cohere', 'command-a-plus-05-2026')

    def test_command_model_chat_request_and_response(self):
        response = {
            'finish_reason': 'COMPLETE',
            'message': {'role': 'assistant', 'content': [
                {'type': 'text', 'text': 'The answer is '},
                {'type': 'tool', 'text': 'this should not appear'},
                {'type': 'text', 'text': '156'}
            ]}
        }
        with patch.object(mr, 'api_call', return_value=response) as send:
            text = self.router.generate(self.candidate, 'What is 12 * 13?')
        self.assertEqual(text, 'The answer is \n156')
        args, kwargs = send.call_args
        self.assertEqual(args, ('POST', 'https://api.cohere.com/v2/chat'))
        self.assertEqual(kwargs['key'], 'fake-trial-key')
        self.assertEqual(kwargs['payload']['model'], 'command-a-plus-05-2026')
        self.assertEqual(kwargs['payload']['messages'], [{'role': 'user', 'content': 'What is 12 * 13?'}])
        self.assertEqual(kwargs['payload']['stream'], False)
        self.assertEqual(kwargs['timeout'], 5)

    def test_probe_uses_short_prompt_and_respects_probe_flag(self):
        response = {'finish_reason': 'COMPLETE', 'message': {'content': [
            {'type': 'text', 'text': 'OK'}]}}
        with patch.object(mr, 'api_call', return_value=response) as send:
            text = self.router.generate(self.candidate, 'original', probe=True)
        self.assertEqual(text, 'OK')
        self.assertEqual(send.call_args.kwargs['payload']['messages'][0]['content'],
                         'Hi. Reply with OK.')
        self.assertEqual(send.call_args.kwargs['timeout'], 8.0)

    def test_truncated_and_error_cohere_replies_rejected(self):
        for reason in ('MAX_TOKENS', 'ERROR', 'TIMEOUT', 'TOOL_CALL'):
            with self.subTest(reason=reason), patch.object(mr, 'api_call', return_value={
                'finish_reason': reason,
                'message': {'content': [{'type': 'text', 'text': 'partial'}]}}):
                with self.assertRaises(CandidateError) as raised:
                    self.router.generate(self.candidate, 'question')
                self.assertEqual(raised.exception.kind, 'invalid_response')

    def test_empty_and_invalid_cohere_replies_rejected(self):
        for message in ({'content': []}, {'content': 'wrong shape'}, 'bad shape'):
            with self.subTest(message=message), patch.object(mr, 'api_call',
                    return_value={'finish_reason': 'COMPLETE', 'message': message}):
                with self.assertRaises(CandidateError):
                    self.router.generate(self.candidate, 'question')

    def test_invalid_cohere_token_code_classified_as_provider_auth(self):
        failure = from_http(498, 'token invalid')
        self.assertEqual(failure.kind, 'auth')

    def test_four_cohere_models_are_grouped_in_default_order(self):
        entries = mr.load_config(Path(__file__).resolve().parents[1] / 'model_priority.json')
        names = [(entry.provider, entry.model) for entry in entries]
        cohere = [entry for entry in entries if entry.provider == 'Cohere']
        self.assertEqual([entry.model for entry in cohere], [
            'command-a-plus-05-2026', 'command-a-03-2025',
            'command-r-plus-08-2024', 'command-r7b-12-2024',
        ])
        self.assertTrue(names.index(('Z.ai', 'glm-4.7-flash')) <
                        names.index(('Cohere', 'command-a-plus-05-2026')))
        self.assertTrue(names.index(('Cohere', 'command-r7b-12-2024')) <
                        names.index(('NVIDIA NIM', 'nvidia/nemotron-3-super-120b-a12b')))


if __name__ == '__main__':
    unittest.main()
