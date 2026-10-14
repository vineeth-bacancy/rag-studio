"""The request guard: each attack is sent, and none may reach a route.

See web/api/guard.py for why each rule exists.
"""
from __future__ import annotations

import pytest

from guard import hostname, refusal

ORIGINS = frozenset({"http://127.0.0.1:3100", "http://localhost:3100",
                     "http://127.0.0.1:8775", "http://localhost:8775"})


def scope(method="GET", **headers):
    return {"type": "http", "method": method,
            "headers": [(k.replace("_", "-").encode(), v.encode()) for k, v in headers.items()]}


@pytest.mark.parametrize("value, expected", [
    ("127.0.0.1:8775", "127.0.0.1"),
    ("LOCALHOST:3100", "localhost"),
    ("[::1]:8775", "::1"),
    ("evil.example", "evil.example"),
])
def test_hostname(value, expected):
    assert hostname(value) == expected


def test_dns_rebinding_host_is_refused_even_for_reads():
    assert refusal(scope(host="attacker.example:8775"), ORIGINS)


def test_rebinding_through_the_ui_proxy_is_refused():
    # The Next proxy rewrites Host to the API and carries the browser's host
    # in X-Forwarded-Host; checking Host alone would let this through.
    assert refusal(scope(host="127.0.0.1:8775", x_forwarded_host="attacker.example:3100"), ORIGINS)


def test_cross_site_post_is_refused():
    assert refusal(scope("POST", host="127.0.0.1:8775", sec_fetch_site="cross-site"), ORIGINS)


def test_foreign_origin_post_is_refused():
    assert refusal(scope("POST", host="127.0.0.1:8775", origin="https://evil.example"), ORIGINS)


def test_null_origin_post_is_refused():
    assert refusal(scope("POST", host="127.0.0.1:8775", origin="null"), ORIGINS)


def test_own_ui_post_is_allowed():
    assert refusal(scope("POST", host="127.0.0.1:8775", origin="http://localhost:3100",
                         sec_fetch_site="same-origin"), ORIGINS) is None


def test_script_without_origin_is_allowed():
    # curl and test clients send no Origin; browsers always do on POST.
    assert refusal(scope("POST", host="127.0.0.1:8775"), ORIGINS) is None


def test_guard_runs_before_routes(client):
    r = client.get("/api/status", headers={"host": "attacker.example"})
    assert r.status_code == 403
    assert "this computer" in r.json()["detail"]
