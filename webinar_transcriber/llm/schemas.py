"""Pydantic response schemas for Instructor-backed LLM integrations."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, Field

SchemaModelT = TypeVar("SchemaModelT", bound=BaseModel)


class SectionTextResponse(BaseModel):
    """Structured LLM response for one polished section body."""

    tldr: str = ""
    transcript_text: str = ""


class ReportSectionUpdate(BaseModel):
    """Replacement content for one report section."""

    id: str
    title: str


class ReportPolishResponse(BaseModel):
    """Structured LLM response for report polishing."""

    summary: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    section_updates: list[ReportSectionUpdate] = Field(default_factory=list)
