"""
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# The agent skills Kaval.AI ships, and the command that copies them into a
# project.
#
# The skills live in the wheel at ``kavalai/.agents/skills/<name>/SKILL.md``,
# the layout other libraries publish theirs under, so an agent that reads an
# installed package's ``.agents`` directory finds them with no install step at
# all. ``kavalai-skills install`` is the route that does not depend on that: it
# copies them into the project's own skills directory, where every agent looks.

import argparse
import shutil
import sys
from importlib import resources
from pathlib import Path

SKILLS_PACKAGE_PATH = ("kavalai", ".agents", "skills")

DEFAULT_TARGET = Path(".claude") / "skills"

EXIT_OK = 0

EXIT_ERROR = 1


def bundled_skills_dir() -> Path:
    """Locate the skills directory inside the installed package.

    Returns:
        Path of ``kavalai/.agents/skills``.

    Raises:
        FileNotFoundError: When the package was installed without its skill
            files, which means the packaging data went missing rather than the
            skills being optional.
    """
    root = resources.files(SKILLS_PACKAGE_PATH[0])
    path = Path(str(root)).joinpath(*SKILLS_PACKAGE_PATH[1:])
    if not path.is_dir():
        raise FileNotFoundError(
            f"Kaval.AI was installed without its skills ({path} is missing). "
            "Reinstall `kavalai`, or copy the skills from a source checkout."
        )
    return path


def available_skills() -> list[Path]:
    """The bundled skills, in name order.

    Returns:
        One directory per skill, each holding a ``SKILL.md``.
    """
    root = bundled_skills_dir()
    return sorted(p for p in root.iterdir() if (p / "SKILL.md").is_file())


def install_skills(
    target: Path,
    names: list[str] | None = None,
    force: bool = False,
) -> tuple[list[str], list[str]]:
    """Copy the bundled skills into ``target``.

    Args:
        target: Directory to copy into. Created if it does not exist.
        names: Skills to install. ``None`` installs all of them.
        force: Overwrite a skill that is already there. Without it an existing
            skill is left alone and reported as skipped, so a client's own
            edits are never silently discarded.

    Returns:
        The names installed, and the names skipped because they already
        existed.

    Raises:
        ValueError: When ``names`` mentions a skill that is not bundled.
    """
    skills = available_skills()
    by_name = {p.name: p for p in skills}
    if names:
        unknown = sorted(set(names) - set(by_name))
        if unknown:
            raise ValueError(
                f"No such skill: {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(by_name))}."
            )
        skills = [by_name[name] for name in names]

    target.mkdir(parents=True, exist_ok=True)
    installed, skipped = [], []
    for skill in skills:
        destination = target / skill.name
        if destination.exists() and not force:
            skipped.append(skill.name)
            continue
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(skill, destination)
        installed.append(skill.name)
    return installed, skipped


def describe(skill: Path) -> str:
    """Read a skill's one-line description out of its front matter.

    Args:
        skill: The skill directory.

    Returns:
        The ``description`` field, truncated for a terminal, or an empty
        string when the file has none.
    """
    for line in (skill / "SKILL.md").read_text().splitlines():
        if line.startswith("description:"):
            description = line.partition(":")[2].strip()
            return description[:100] + "…" if len(description) > 100 else description
    return ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="kavalai-skills",
        description="Install the Kaval.AI agent skills into a project.",
    )
    parser.add_argument(
        "command",
        choices=["install", "list"],
        help="`install` copies the skills into a project; `list` shows them.",
    )
    parser.add_argument(
        "names",
        nargs="*",
        help="Skills to install. Omit to install all of them.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"Where to install them (default: {DEFAULT_TARGET}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite skills that are already installed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``kavalai-skills`` console script.

    Returns:
        ``0`` when the command did what was asked, ``1`` when it could not.
    """
    args = parse_args(argv)

    try:
        if args.command == "list":
            for skill in available_skills():
                print(f"{skill.name}\n    {describe(skill)}")
            return EXIT_OK

        installed, skipped = install_skills(args.target, args.names, args.force)
    except (FileNotFoundError, ValueError) as e:
        print(f"{e}", file=sys.stderr)
        return EXIT_ERROR

    for name in installed:
        print(f"installed  {args.target / name}")
    for name in skipped:
        print(f"exists     {args.target / name} (--force to overwrite)")
    if not installed and not skipped:
        print("Nothing to install.")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
