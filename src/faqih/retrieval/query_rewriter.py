"""Query rewriter with HyDE and multi-query expansion."""

from __future__ import annotations

import logging

from faqih.services.embedding import EmbeddingService
from faqih.services.llm import LLMClient

logger = logging.getLogger(__name__)

# ─── HyDE Prompt ────────────────────────────────────────────

HYDE_SYSTEM_PROMPT = """أنت عالم فقه إسلامي متخصص. عندما يُطرح عليك سؤال فقهي، اكتب إجابة مثالية مفصلة بأسلوب كتب الفقه الكلاسيكية.

قواعد:
- اكتب بأسلوب المتون والشروح الفقهية
- اذكر الأحكام والأدلة والشروط
- استخدم المصطلحات الفقهية الدقيقة
- اكتب إجابة من 200-400 كلمة
- لا تذكر أنك نموذج ذكاء اصطناعي
"""

# ─── Multi-Query Expansion Prompt ───────────────────────────

EXPANSION_SYSTEM_PROMPT = """أنت خبير في صياغة الأسئلة الفقهية. أعد صياغة السؤال التالي بثلاث طرق مختلفة تحافظ على نفس المعنى لكن بمصطلحات وتراكيب مختلفة.

أعد النتيجة بصيغة JSON:
{
  "queries": ["صياغة 1", "صياغة 2", "صياغة 3"]
}
"""


class QueryRewriter:
    """
    Rewrites user queries for improved retrieval.

    Strategies:
    1. HyDE: Generate hypothetical answer → embed as query vector
    2. Multi-Query: Generate 3 alternative formulations
    """

    def __init__(self, llm: LLMClient, embedding: EmbeddingService):
        self._llm = llm
        self._embedding = embedding

    async def generate_hyde_embedding(self, query: str) -> list[float]:
        """
        Generate a HyDE (Hypothetical Document Embedding) query vector.

        Creates a hypothetical Fiqh-style answer, then embeds it
        as the query vector (not shown to the user).
        """
        messages = [
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        hypothetical_answer = await self._llm.complete(messages, temperature=0.5)
        logger.debug("HyDE answer: %s", hypothetical_answer[:200])

        # Embed the hypothetical answer
        embedding = self._embedding.encode_single(hypothetical_answer)
        return embedding

    async def expand_queries(self, query: str) -> list[str]:
        """
        Generate alternative formulations of the query.

        Returns the original query + 3 reformulations.
        """
        messages = [
            {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        try:
            result = await self._llm.complete_json(messages, temperature=0.7)
            expanded = result.get("queries", [])
            logger.info("Expanded to %d alternative queries", len(expanded))
        except Exception as e:
            logger.warning("Query expansion failed: %s", e)
            expanded = []

        # Original query first, then expansions
        return [query] + expanded[:3]

    async def rewrite(self, query: str) -> dict:
        """
        Full query rewriting pipeline.

        Returns dict with:
        - hyde_embedding: list[float] — the HyDE query vector
        - expanded_queries: list[str] — original + 3 reformulations
        """
        hyde_embedding, expanded = await asyncio.gather(
            self.generate_hyde_embedding(query),
            self.expand_queries(query),
        )

        return {
            "hyde_embedding": hyde_embedding,
            "expanded_queries": expanded,
        }


import asyncio  # noqa: E402 — needed for gather in rewrite()
