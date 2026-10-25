"""Offline, deterministic checks of cached health and classified fallback."""
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import model_router as mr
from router_errors import CandidateError, from_http, retry_after_seconds
from router_health import HealthCache
from router_policy import ask_cached, record_council_role


class FakeClock:
    def __init__(self):
        self.now = 10000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeRouter:
    def __init__(self, behavior=None):
        self.timeout = 22.0
        self.probe_timeout = 6.0
        self.calls = []
        self.behavior = behavior or {}

    def expand(self, candidate, _used):
        return [candidate]

    def generate(self, candidate, prompt, *, probe=False, request_timeout=None):
        self.calls.append((candidate.provider, candidate.model, prompt, probe,
                           request_timeout))
        return self.behavior.get(candidate.model, lambda _: ("OK" if probe else "156"))(
            (candidate, prompt, probe, request_timeout)
        )


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "health.json"
        self.clock = FakeClock()
        self.env = patch.dict(os.environ, {
            "GEMINI_API_KEY": "test-key",
            "LLM7_API_KEY": "test-key",
            "GROQ_API_KEY": "test-key",
            "ZAI_API_KEY": "test-key",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.notes = []

    def cache(self):
        return HealthCache(self.path, 600, clock=self.clock)

    def run_router(self, router, candidates, *, prompt="Full research question\nWith context.", probe=True):
        return ask_cached(router, prompt, candidates, cache=self.cache(),
                          probe=probe, expected=None, log=self.notes.append)

    def test_first_use_probes_then_cache_reuse_and_expiration(self):
        candidates = [mr.Candidate("Gemini", "gemini-a")]
        router = FakeRouter()
        self.run_router(router, candidates)
        self.assertEqual(len(router.calls), 2)
        self.assertTrue(router.calls[0][3])
        self.assertFalse(router.calls[1][3])
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        saved = self.path.read_text()
        self.assertNotIn("test-key", saved)
        self.assertNotIn("Full research question", saved)

        second_process = FakeRouter()
        self.run_router(second_process, candidates)
        self.assertEqual(len(second_process.calls), 1)
        self.assertFalse(second_process.calls[0][3])
        self.assertTrue(any("PROBE CACHED" in n for n in self.notes))

        self.clock.advance(601)
        third_process = FakeRouter()
        self.run_router(third_process, candidates)
        self.assertEqual(len(third_process.calls), 2)
        self.assertTrue(third_process.calls[0][3])

    def test_authentication_failure_blocks_provider_not_other_providers(self):
        def broken(call):
            raise CandidateError("Invalid key", kind="auth")
        candidates = [
            mr.Candidate("Gemini", "gemini-one"),
            mr.Candidate("Gemini", "gemini-two"),
            mr.Candidate("Groq", "groq-working"),
        ]
        router = FakeRouter({"gemini-one": broken, "gemini-two": broken})
        answer, winning, attempts = self.run_router(router, candidates)
        self.assertEqual(winning.provider, "Groq")
        self.assertEqual(answer, "156")
        self.assertEqual([c[1] for c in router.calls],
                         ["gemini-one", "groq-working", "groq-working"])
        self.assertTrue(self.cache().blocked("Gemini", "gemini-two"))
        self.assertTrue(self.cache().healthy("Groq", "groq-working"))

    def test_retry_after_shared_quota_blocks_backup(self):
        def limited(call):
            raise CandidateError("HTTP 429", kind="rate_limit", retry_after=120)
        candidates = [
            mr.Candidate("LLM7.io", "first"),
            mr.Candidate("LLM7.io", "second"),
            mr.Candidate("Groq", "ok"),
        ]
        router = FakeRouter({"first": limited, "second": limited})
        result = self.run_router(router, candidates)
        self.assertEqual(result[1].provider, "Groq")
        self.assertNotIn("second", [call[1] for call in router.calls])
        self.assertEqual(self.cache().blocked("LLM7.io", "second")[1], 120)
        self.clock.advance(121)
        self.assertIsNone(self.cache().blocked("LLM7.io", "second"))

    def test_separate_quota_group_allows_backup(self):
        def limited(call):
            raise CandidateError("HTTP 429", kind="rate_limit", retry_after=120)
        candidates = [
            mr.Candidate("LLM7.io", "first", quota_group="turbo"),
            mr.Candidate("LLM7.io", "backup", quota_group="different_pool"),
        ]
        router = FakeRouter({"first": limited})
        answer, winning, attempts = self.run_router(router, candidates)
        self.assertEqual(winning.model, "backup")
        self.assertEqual([row[1] for row in router.calls], ["first", "backup", "backup"])

    def test_server_error_retries_once_short_then_falls_back(self):
        counter = {"bad": 0}
        def flaky(call):
            counter["bad"] += 1
            raise CandidateError("HTTP 503", kind="server")
        candidates = [mr.Candidate("Gemini", "bad"), mr.Candidate("Groq", "good")]
        router = FakeRouter({"bad": flaky})
        with patch("router_policy.time.sleep"):
            result = self.run_router(router, candidates, probe=False)
        self.assertEqual(result[1].model, "good")
        self.assertEqual(counter["bad"], 2)
        self.assertEqual(router.calls[1][4], 8.0)
        self.assertTrue(self.cache().blocked("Gemini", "bad"))

    def test_transient_retry_once_can_recover(self):
        counter = {"bad": 0}
        def flaky(call):
            counter["bad"] += 1
            if counter["bad"] == 1:
                raise CandidateError("timeout", kind="timeout")
            return "156"
        router = FakeRouter({"bad": flaky})
        with patch("router_policy.time.sleep"):
            result = self.run_router(router, [mr.Candidate("Gemini", "bad")], probe=False)
        self.assertEqual(result[1].model, "bad")
        self.assertEqual(counter["bad"], 2)
        self.assertEqual(len(router.calls), 2)
        self.assertEqual(router.calls[1][4], 8.0)

    def test_context_length_does_not_mark_unhealthy(self):
        def too_small(call):
            raise CandidateError("Context too long", kind="context_length")
        candidates = [
            mr.Candidate("Gemini", "small"),
            mr.Candidate("Gemini", "big"),
        ]
        prompt = "Conversation context:\n" + ("Very relevant line.\n" * 100)
        router = FakeRouter({"small": too_small})
        result = self.run_router(router, candidates, prompt=prompt, probe=False)
        self.assertEqual(result[1].model, "big")
        self.assertIsNone(self.cache().blocked("Gemini", "small"))
        self.assertEqual(router.calls[-1][2], prompt)
        self.assertEqual(router.calls[0][2], prompt)

    def test_configured_small_context_skips_without_blacklisting(self):
        prompt = "Relevant context:\n" + ("a" * 5000)
        candidates = [
            mr.Candidate("Gemini", "small", context_window=1000),
            mr.Candidate("Gemini", "big", context_window=64000),
        ]
        router = FakeRouter()
        result = self.run_router(router, candidates, prompt=prompt, probe=False)
        self.assertEqual(result[1].model, "big")
        self.assertEqual([x[1] for x in router.calls], ["big"])
        self.assertIsNone(self.cache().blocked("Gemini", "small"))

    def test_council_cross_provider_base_model_reuse(self):
        path = Path(self.dir.name) / "roles.json"
        first = mr.Candidate("Z.ai", "glm-4.7-flash")
        second = mr.Candidate("Cloudflare", "@cf/zai-org/glm-4.7-flash")
        third = mr.Candidate("LLM7.io", "glm-4.7-flash")
        self.assertEqual(record_council_role(path, "runA", "A", first), [])
        second_warnings = record_council_role(path, "runA", "B", second)
        self.assertTrue(any("MODEL REUSE" in message for message in second_warnings))
        third_warnings = record_council_role(path, "runA", "C", third)
        self.assertTrue(any("INDEPENDENCE WARNING" in message for message in third_warnings))

    def test_successful_fallback_becomes_preferred_across_cli_runs(self):
        def broken(call):
            raise CandidateError("server overloaded", kind="server")
        options = [
            mr.Candidate("Gemini", "top"),
            mr.Candidate("Groq", "fallback"),
        ]
        first = FakeRouter({"top": broken})
        with patch("router_policy.time.sleep"):
            self.run_router(first, options)
        self.assertEqual(self.cache().preferred_model(), ("Groq", "fallback"))

        # Gemini's short cooldown has expired, but Groq remains healthy.
        # The next command should use Groq directly without another probe.
        self.clock.advance(31)
        second = FakeRouter({"top": broken})
        self.run_router(second, options)
        self.assertEqual([entry[1] for entry in second.calls], ["fallback"])
        self.assertFalse(second.calls[0][3])
        self.assertTrue(any("PREFERRED CACHED Groq/fallback" in n for n in self.notes))

        # After 10 minutes, strict original priority applies again.
        self.clock.advance(601)
        third = FakeRouter({"top": broken})
        with patch("router_policy.time.sleep"):
            self.run_router(third, options)
        self.assertEqual(third.calls[0][1], "top")
        self.assertIsNotNone(self.cache().preferred_model())

    def test_no_sticky_restores_priority_even_with_valid_preference(self):
        options = [
            mr.Candidate("Gemini", "top"),
            mr.Candidate("Groq", "fallback"),
        ]
        c = self.cache()
        c.success("Groq", "fallback", prefer=True)
        router = FakeRouter()
        answer, winner, _ = ask_cached(router, "Exact context\nQuestion",
                                        options, cache=self.cache(), probe=True,
                                        sticky=False, log=self.notes.append)
        self.assertEqual(winner.model, "top")
        self.assertEqual(router.calls[0][1], "top")
        self.assertTrue(router.calls[0][3])

    def test_successful_real_answer_is_cached_even_without_probe(self):
        router = FakeRouter()
        self.run_router(router, [mr.Candidate("Groq", "good")], probe=False)
        self.assertTrue(self.cache().healthy("Groq", "good"))

    def test_council_model_attribution_and_independence_warning(self):
        path = Path(self.dir.name) / ".council_roles.json"
        model = mr.Candidate("Groq", "openai/gpt-oss-120b")
        self.assertEqual(record_council_role(path, "run1", "proposer", model), [])
        warns = record_council_role(path, "run1", "critic", model, fallback_used=True)
        self.assertTrue(any("MODEL REUSE" in x for x in warns))
        warns = record_council_role(path, "run1", "verifier", model, fallback_used=True)
        self.assertTrue(any("INDEPENDENCE WARNING" in x for x in warns))
        data = json.loads(path.read_text())
        self.assertEqual(len(data["runs"]["run1"]["roles"]), 3)
        self.assertTrue(data["runs"]["run1"]["roles"]["critic"]["fallback_used"])
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertNotIn("test-key", path.read_text())


class FailureClassificationTests(unittest.TestCase):
    def test_retry_after_seconds_and_date(self):
        self.assertEqual(retry_after_seconds("45"), 45)
        self.assertEqual(retry_after_seconds("-9"), 0)
        self.assertIsNone(retry_after_seconds("nonsense"))
        from datetime import datetime, timezone
        base = datetime(2026, 10, 25, 19, 0, tzinfo=timezone.utc)
        self.assertEqual(
            retry_after_seconds("Sun, 25 Oct 2026 19:02:00 GMT", now=base), 120
        )

    def test_context_and_model_not_found_have_different_kinds(self):
        self.assertEqual(from_http(400, '{"error": "maximum context window exceeded"}').kind,
                         "context_length")
        self.assertEqual(from_http(404, "model not found").kind,
                         "model_unavailable")
        self.assertEqual(from_http(401, "invalid api key").kind, "auth")
        self.assertEqual(from_http(429, "busy", retry_after="75").retry_after, 75)
        self.assertEqual(from_http(503, "server busy").kind, "server")

    def test_config_model_ids_and_display_names_are_distinct(self):
        cfg_path = Path(tempfile.mkdtemp()) / "rank.json"
        cfg_path.write_text(json.dumps({
            "priority": [{"provider": "Gemini", "model_id": "gemini-3.8-flash",
                          "display_name": "Best Gemini",
                          "context_window": 100000}]
        }))
        items = mr.load_config(cfg_path)
        self.assertEqual(items[0].model, "gemini-3.8-flash")
        self.assertEqual(items[0].display_name, "Best Gemini")
        self.assertEqual(items[0].context_window, 100000)


if __name__ == "__main__":
    unittest.main()
