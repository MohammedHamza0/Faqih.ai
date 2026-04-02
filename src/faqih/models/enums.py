"""Domain enumerations for the Fiqh RAG system."""

from __future__ import annotations

from enum import StrEnum


class ChunkType(StrEnum):
    """Classification of a Fiqh text chunk by its content role."""

    HUKM = "hukm"              # Ruling / Legal judgment
    DALIL = "dalil"            # Evidence / Proof
    TALIL = "talil"            # Reasoning / Justification
    KHILAF = "khilaf"          # Scholarly disagreement
    SHART = "shart"            # Condition / Prerequisite
    ISTITHNA = "istithna"      # Exception
    GENERAL = "general"        # Unclassified content


class Madhab(StrEnum):
    """The four major Sunni schools of Islamic jurisprudence."""

    HANAFI = "hanafi"
    MALIKI = "maliki"
    SHAFII = "shafii"
    HANBALI = "hanbali"


class RelationType(StrEnum):
    """Types of relationships between entities in the knowledge graph."""

    AGREES_WITH = "AGREES_WITH"
    DISAGREES_WITH = "DISAGREES_WITH"
    QUALIFIES = "QUALIFIES"
    EVIDENCED_BY = "EVIDENCED_BY"
    EXCEPTION_OF = "EXCEPTION_OF"
    REFERENCES = "REFERENCES"
    DERIVED_FROM = "DERIVED_FROM"
    HAS_CONTENT = "HAS_CONTENT"


class QuestionType(StrEnum):
    """Classification of user question intent."""

    FATWA = "fatwa"            # Seeking a ruling
    MUQARANA = "muqarana"      # Comparative across madhabs
    TASIL = "tasil"            # Foundational / Theoretical
    TARIKH = "tarikh"          # Historical


class DetailLevel(StrEnum):
    """Desired level of detail in the response."""

    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"


class EntityType(StrEnum):
    """Types of entities extracted from Fiqh texts."""

    MASALA = "masala"          # Legal issue / question
    SCHOLAR = "scholar"        # Named scholar
    BOOK = "book"             # Reference book
    AYAH = "ayah"             # Quranic verse
    HADITH = "hadith"         # Prophetic tradition
    IJMA = "ijma"             # Consensus
    QIYAS = "qiyas"           # Analogical reasoning
    TERM = "term"             # Technical Fiqh term
