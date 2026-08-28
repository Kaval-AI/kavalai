"""The bundled agent skills, and the command that installs them.

A skill is documentation a client's coding agent reads instead of the docs, and
it rots the same way — except the agent cannot tell that it has. These tests
pin the parts a rename would silently invalidate: the node types, the eval
schema, the matchers and the environment variable names a skill spells out.
They cannot check prose, but a rename is what actually breaks a client.
"""

import inspect
import re
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from kavalai import skills
from kavalai.eval.eval_runner import EvalCase, EvalSuite
from kavalai.eval.simple_evaluator import MATCHERS
from kavalai.workflow import models

ROOT = Path(__file__).parent.parent
SKILLS_DIR = ROOT / "kavalai" / ".agents" / "skills"
ENV_EXAMPLE = ROOT / ".env.example"

EXPECTED_SKILLS = {
    "kavalai",
    "kavalai-workflows",
    "kavalai-tools",
    "kavalai-serving",
    "kavalai-rag",
    "kavalai-eval",
}

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def skill_dirs() -> list[Path]:
    return sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())


def skill_text(name: str) -> str:
    """Everything a skill says, its reference files included."""
    directory = SKILLS_DIR / name
    return "\n".join(p.read_text() for p in sorted(directory.rglob("*.md")))


def all_skill_text() -> str:
    return "\n".join(p.read_text() for p in sorted(SKILLS_DIR.rglob("*.md")))


def node_types() -> set[str]:
    """Every ``type:`` value a node in a workflow graph can carry."""
    return {
        member.model_fields["type"].default
        for member in vars(models).values()
        if inspect.isclass(member)
        and issubclass(member, BaseModel)
        and isinstance(member.model_fields.get("type"), FieldInfo)
        and isinstance(member.model_fields["type"].default, str)
        and member.__name__.endswith("Node")
    }


def front_matter(skill: Path) -> dict[str, str]:
    """Parse a skill's front matter into a mapping.

    The format is one ``key: value`` per line; values are single-line, so a
    plain split is enough and avoids a YAML dependency in the test.
    """
    match = FRONT_MATTER.match((skill / "SKILL.md").read_text())
    assert match, f"{skill.name} has no front matter"
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


class TestBundle:
    """The skills that ship, and the shape they ship in."""

    def test_the_expected_skills_ship(self):
        assert {p.name for p in skill_dirs()} == EXPECTED_SKILLS

    @pytest.mark.parametrize("skill", skill_dirs(), ids=lambda p: p.name)
    def test_front_matter_names_the_directory(self, skill):
        fields = front_matter(skill)
        assert fields["name"] == skill.name
        # The description is the only thing an agent reads when deciding
        # whether to load the skill at all, so an empty or terse one is a bug.
        assert len(fields["description"]) > 80

    @pytest.mark.parametrize("skill", skill_dirs(), ids=lambda p: p.name)
    def test_reference_files_are_reachable(self, skill):
        """A reference nothing points at is a reference nothing reads."""
        body = (skill / "SKILL.md").read_text()
        for reference in (skill / "references").glob("*.md"):
            assert f"references/{reference.name}" in body

    def test_the_entry_skill_routes_to_every_other(self):
        body = skill_text("kavalai")
        for name in EXPECTED_SKILLS - {"kavalai"}:
            assert name in body


class TestDrift:
    """What a rename in the library would invalidate."""

    def test_every_node_type_is_documented(self):
        """A node type the workflow skill never mentions cannot be written."""
        documented = skill_text("kavalai-workflows")
        for name in node_types():
            assert re.search(rf"\b{re.escape(name)}\b", documented), name

    def test_every_input_type_is_documented(self):
        documented = skill_text("kavalai-workflows")
        for name in models.ArgumentInfo.model_fields["type"].annotation.__args__:
            assert f"`{name}`" in documented

    def test_every_matcher_is_documented(self):
        documented = skill_text("kavalai-eval")
        for matcher in MATCHERS:
            assert f"`{matcher}`" in documented

    def test_every_eval_key_is_documented(self):
        documented = skill_text("kavalai-eval")
        for field in (*EvalSuite.model_fields, *EvalCase.model_fields):
            assert f"`{field}`" in documented

    def test_environment_variables_exist(self):
        """Every variable a skill spells out is one the code actually reads.

        The other direction is `test_config_drift.py`'s job: a skill is allowed
        to leave a variable out, but never to invent one.
        """
        documented = set(re.findall(r"[A-Z][A-Z0-9_]{4,}", all_skill_text()))
        known = set(
            re.findall(r"^([A-Z][A-Z0-9_]*)=", ENV_EXAMPLE.read_text(), re.MULTILINE)
        )
        # Prose and code fences carry unrelated capitals; only judge the names
        # that look like ours or like a provider credential.
        candidates = {
            name
            for name in documented
            if name.startswith(("KAVALAI_", "FASTEMBED_"))
            or name.endswith(("_API_KEY", "_CLIENT_ID", "_CLIENT_SECRET"))
        }
        assert candidates <= known, sorted(candidates - known)


class TestInstall:
    """`kavalai-skills`."""

    def test_available_skills_are_the_bundled_ones(self):
        assert {p.name for p in skills.available_skills()} == EXPECTED_SKILLS

    def test_install_copies_every_skill(self, tmp_path):
        installed, skipped = skills.install_skills(tmp_path)
        assert set(installed) == EXPECTED_SKILLS
        assert skipped == []
        assert (tmp_path / "kavalai-workflows" / "SKILL.md").is_file()
        assert (tmp_path / "kavalai-workflows" / "references" / "nodes.md").is_file()

    def test_install_one(self, tmp_path):
        installed, _ = skills.install_skills(tmp_path, ["kavalai-rag"])
        assert installed == ["kavalai-rag"]
        assert not (tmp_path / "kavalai").exists()

    def test_an_unknown_name_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="No such skill: nope"):
            skills.install_skills(tmp_path, ["nope"])

    def test_existing_skills_are_kept_unless_forced(self, tmp_path):
        skills.install_skills(tmp_path, ["kavalai-rag"])
        edited = tmp_path / "kavalai-rag" / "SKILL.md"
        edited.write_text("mine")

        installed, skipped = skills.install_skills(tmp_path, ["kavalai-rag"])
        assert installed == [] and skipped == ["kavalai-rag"]
        assert edited.read_text() == "mine", "a client's own edits were discarded"

        installed, skipped = skills.install_skills(
            tmp_path, ["kavalai-rag"], force=True
        )
        assert installed == ["kavalai-rag"] and skipped == []
        assert edited.read_text() != "mine"

    def test_describe_reads_the_front_matter(self):
        skill = SKILLS_DIR / "kavalai-eval"
        assert skills.describe(skill).startswith("Write and run Kaval.AI")

    def test_main_installs_and_reports(self, tmp_path, capsys):
        assert skills.main(["install", "--target", str(tmp_path)]) == skills.EXIT_OK
        assert "installed" in capsys.readouterr().out

        assert skills.main(["install", "--target", str(tmp_path)]) == skills.EXIT_OK
        assert "--force to overwrite" in capsys.readouterr().out

    def test_main_lists(self, capsys):
        assert skills.main(["list"]) == skills.EXIT_OK
        out = capsys.readouterr().out
        assert all(name in out for name in EXPECTED_SKILLS)

    def test_main_reports_an_unknown_name(self, tmp_path, capsys):
        code = skills.main(["install", "nope", "--target", str(tmp_path)])
        assert code == skills.EXIT_ERROR
        assert "No such skill" in capsys.readouterr().err
