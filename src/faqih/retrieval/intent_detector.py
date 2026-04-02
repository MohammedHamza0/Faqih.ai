"""Intent detection for Fiqh queries."""

from __future__ import annotations

import logging

from faqih.models.schemas import QueryIntent
from faqih.services.llm import LLMClient

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """أنت محلل أسئلة فقهية. حلل السؤال التالي واستخرج المعلومات المطلوبة.

أعد النتيجة بصيغة JSON:
{
  "madhab_preference": "hanafi|maliki|shafii|hanbali|all",
  "question_type": "fatwa|muqarana|tasil|tarikh",
  "detail_level": "brief|standard|detailed",
  "language": "arabic|english|mixed"
}

قواعد:
- madhab_preference: "all" إذا لم يذكر المستخدم مذهباً محدداً أو يسأل مقارنة
- question_type:
  - fatwa: يسأل عن حكم شرعي محدد
  - muqarana: يسأل عن الاختلاف بين المذاهب أو المقارنة
  - tasil: يسأل عن الأصل أو الدليل أو التأصيل
  - tarikh: يسأل عن تاريخ حكم أو مسألة
- detail_level: "detailed" إذا طلب تفصيلاً، "brief" إذا طلب اختصاراً
- language: حدد لغة السؤال
"""


class IntentDetector:
    """
    Lightweight intent classifier for Fiqh queries.

    Extracts: madhab preference, question type, detail level, language.
    Uses few-shot LLM prompting (upgradeable to fine-tuned model later).
    """

    def __init__(self, llm: LLMClient):
        self._llm = llm

    async def detect(self, query: str) -> QueryIntent:
        """Detect the intent of a Fiqh query."""
        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        try:
            result = await self._llm.complete_json(messages, temperature=0.1)
            intent = QueryIntent(
                madhab_preference=result.get("madhab_preference", "all"),
                question_type=result.get("question_type", "fatwa"),
                detail_level=result.get("detail_level", "standard"),
                language=result.get("language", "arabic"),
            )
            logger.info("Detected intent: %s", intent.model_dump())
            return intent
        except Exception as e:
            logger.warning("Intent detection failed (%s), using defaults", e)
            return QueryIntent()
