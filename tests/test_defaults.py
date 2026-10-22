"""The app's defaults (M5, 2026-10-16): row 5d and prompt v3."""
from rag_studio import config, health, profiles, prompt


def test_the_default_is_row_5d_with_prompt_v3():
    assert profiles.DEFAULT == "docling-structure-embeddinggemma2" and profiles.DEFAULT in profiles.PROFILES
    assert prompt.DEFAULT_VERSION == "v3" and prompt.DEFAULT_VERSION in prompt.SYSTEM_PROMPTS


def test_the_embed_role_follows_the_default_index():
    role = next(r for r in config.ROLES if r.key == "embed")
    assert role.model == profiles.get(profiles.DEFAULT).embedding.model and role.runs_on == "python"


def test_not_built_messages_name_the_right_command():
    assert profiles.build_command("baseline") == "./rag ingest"
    assert profiles.build_command(profiles.DEFAULT) == f"./rag index build {profiles.DEFAULT}"


def test_status_names_the_default_and_the_memory_it_needs(tmp_data):
    s = health.status()
    assert s["default_profile"] == profiles.DEFAULT and s["build_command"].startswith("./rag index build")
    assert set(s["memory"]) == {"needed_gb", "free_gb", "short"}
