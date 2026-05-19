"""Tests for ADK skill loading and L1 metadata for the food CPI agent."""

from __future__ import annotations

from food_price_forecasting.analyst_agent import (
    FOOD_CPI_SKILL_DIR,
    FOOD_PRICE_FORECASTER_INSTRUCTION,
)
from google.adk.skills import load_skill_from_dir
from google.adk.skills.models import Skill
from google.adk.skills.prompt import format_skills_as_xml


def _food_skill() -> Skill:
    return load_skill_from_dir(FOOD_CPI_SKILL_DIR)


class TestFoodCpiSkillL1:
    """SKILL.md frontmatter serialises to L1 XML without duplicating the forecaster instruction."""

    def test_skill_frontmatter_loads(self) -> None:
        skill = _food_skill()
        assert skill.name == "forecast-food-cpi"
        assert skill.frontmatter.description.strip()

    def test_skill_l1_xml_contains_name_and_description(self) -> None:
        skill = _food_skill()
        xml = format_skills_as_xml([skill])
        assert "<available_skills>" in xml
        assert "forecast-food-cpi" in xml
        snippet = skill.frontmatter.description.strip()[:80]
        assert snippet in xml

    def test_forecaster_instruction_does_not_duplicate_l1(self) -> None:
        skill = _food_skill()
        instruction = FOOD_PRICE_FORECASTER_INSTRUCTION.lower()
        assert "forecast-food-cpi" not in instruction
        assert "load_skill" not in instruction
        description_snippet = skill.frontmatter.description.strip()[:80].lower()
        assert description_snippet not in instruction
