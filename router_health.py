"""Private, persistent health and cooldown state for the provider-first router.

Stores no API keys, prompts or model answers. All times are epoch seconds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

DEFAULT_TTL_SECONDS = 600
COOLDOWN_SECONDS = {
    "auth": 3600,
    "model_unavailable": 3600,
    "rate_limit": 60,
    "timeout": 30,
    "server": 30,
    "connection": 30,
    "invalid_response": 45,
    "request_invalid": 45,
    "other": 30,
}


class HealthCache:
    def __init__(self, path: Path, ttl_seconds: int = DEFAULT_TTL_SECONDS,
                 clock=None):
        self.path = Path(path)
        self.ttl = ttl_seconds
        self.clock = clock or time.time
        self.data = {"version": 1, "models": {}, "providers": {}, "groups": {},
                     "preferred": None}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded.get("version") == 1:
                    for key in ("models", "providers", "groups"):
                        value = loaded.get(key)
                        if isinstance(value, dict):
                            self.data[key] = value
                    preferred = loaded.get("preferred")
                    if isinstance(preferred, dict):
                        self.data["preferred"] = preferred
            except (ValueError, OSError, UnicodeError):
                # A corrupt/inaccessible state file must never expose secrets or
                # prevent normal calls: fail open and rebuild it on next save.
                pass

    @staticmethod
    def model_key(provider: str, model: str) -> str:
        return json.dumps([provider, model], ensure_ascii=False)

    @staticmethod
    def quota_key(provider: str, quota_group: str | None) -> str:
        # Unless explicitly separated in config, quota is assumed shared
        # within this provider. Never invent independent quotas.
        return json.dumps([provider, quota_group or "__provider_shared__"],
                          ensure_ascii=False)

    def _until(self, value) -> float:
        if isinstance(value, dict):
            value = value.get("until", value.get("cooldown_until", 0))
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    def blocked(self, provider: str, model: str, quota_group: str | None = None):
        """Return (reason, seconds remaining) or None."""
        now = self.clock()
        checks = (
            ("provider", self.data["providers"].get(provider)),
            ("quota group", self.data["groups"].get(self.quota_key(provider, quota_group))),
            ("model", self.data["models"].get(self.model_key(provider, model))),
        )
        for label, value in checks:
            until = self._until(value)
            if until > now:
                detail = value.get("reason", label) if isinstance(value, dict) else label
                return f"{label}: {detail}", int(until - now + 0.999)
        return None

    def healthy(self, provider: str, model: str, quota_group: str | None = None) -> bool:
        if self.blocked(provider, model, quota_group):
            return False
        entry = self.data["models"].get(self.model_key(provider, model), {})
        return self._until(entry.get("healthy_until", 0)) > self.clock() if isinstance(entry, dict) else False

    def preferred_model(self) -> tuple[str, str] | None:
        """Temporarily prefer the last successful real-answer model."""
        entry = self.data.get("preferred")
        if not isinstance(entry, dict):
            return None
        provider, model = entry.get("provider"), entry.get("model")
        if not isinstance(provider, str) or not isinstance(model, str):
            return None
        if self._until(entry.get("until")) <= self.clock():
            return None
        if not self.healthy(provider, model, entry.get("quota_group")):
            return None
        return provider, model

    def success(self, provider: str, model: str, *, prefer: bool = False,
                quota_group: str | None = None):
        key = self.model_key(provider, model)
        self.data["models"][key] = {
            "healthy_until": self.clock() + self.ttl,
            "last_success": self.clock(),
            "cooldown_until": 0,
            "reason": "",
        }
        if prefer:
            self.data["preferred"] = {
                "provider": provider,
                "model": model,
                "quota_group": quota_group,
                "until": self.clock() + self.ttl,
            }
        self.save()

    def failure(self, provider: str, model: str, kind: str,
                quota_group: str | None = None, retry_after: float | None = None):
        if kind == "context_length":
            # Context constraints are specific to this prompt, not model health.
            return
        now = self.clock()
        preferred = self.data.get("preferred")
        if (isinstance(preferred, dict) and preferred.get("provider") == provider
                and preferred.get("model") == model):
            self.data["preferred"] = None
        delay = COOLDOWN_SECONDS.get(kind, COOLDOWN_SECONDS["other"])
        if kind == "rate_limit" and retry_after is not None:
            delay = max(1, min(86400, float(retry_after)))
        entry = {"until": now + delay, "reason": kind}
        if kind == "auth":
            self.data["providers"][provider] = entry
        elif kind == "rate_limit":
            self.data["groups"][self.quota_key(provider, quota_group)] = entry
        else:
            key = self.model_key(provider, model)
            previous = self.data["models"].get(key, {})
            if not isinstance(previous, dict):
                previous = {}
            previous.update({
                "healthy_until": 0,
                "cooldown_until": now + delay,
                "reason": kind,
            })
            self.data["models"][key] = previous
        self.save()

    def save(self):
        """Atomically persist a private 0600 state file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", delete=False
            ) as stream:
                temporary = Path(stream.name)
                os.chmod(temporary, 0o600)
                json.dump(self.data, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
