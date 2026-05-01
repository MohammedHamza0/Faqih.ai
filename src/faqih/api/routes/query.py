"""Query endpoint with SSE streaming."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from faqih.api.dependencies import (
    get_conversation_memory,
    get_generator,
    get_intent_detector,
    get_query_rewriter,
    get_retrieval_engine,
    get_semantic_cache,
)
from faqih.cache.semantic_cache import SemanticCache
from faqih.generation.citation_linker import CitationLinker
from faqih.generation.generator import FiqhGenerator
from faqih.memory.conversation import ConversationMemory
from faqih.retrieval.engine import RetrievalEngine
from faqih.retrieval.intent_detector import IntentDetector
from faqih.retrieval.query_rewriter import QueryRewriter
from faqih.services.llm import LLMExhaustedError

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    query: str


@router.post("/sessions/{session_id}/query")
async def query_endpoint(
    session_id: str,
    body: QueryRequest,
    request: Request,
    rewriter: QueryRewriter = Depends(get_query_rewriter),
    intent_detector: IntentDetector = Depends(get_intent_detector),
    retrieval: RetrievalEngine = Depends(get_retrieval_engine),
    generator: FiqhGenerator = Depends(get_generator),
    memory: ConversationMemory = Depends(get_conversation_memory),
    cache: SemanticCache = Depends(get_semantic_cache),
):
    """
    Process a Fiqh query and stream the response via SSE.
    """
    query_text = body.query

    # ── Check semantic cache ────────────────────────────
    cached_response = await cache.get(query_text)
    if cached_response:
        # Return cached response via SSE
        async def cached_stream():
            yield f"data: {json.dumps({'type': 'cached', 'content': True}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'answer', 'content': cached_response.get('answer', '')}, ensure_ascii=False)}\n\n"
            citations = cached_response.get("citations", [])
            if citations:
                yield f"data: {json.dumps({'type': 'citations', 'content': citations}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(cached_stream(), media_type="text/event-stream")

    # ── Process query pipeline ──────────────────────────
    async def process_and_stream():
        try:
            # Step 1: Detect intent + rewrite query (parallel)
            intent_task = intent_detector.detect(query_text)
            rewrite_task = rewriter.rewrite(query_text)

            intent, rewrite_result = await asyncio.gather(intent_task, rewrite_task)

            yield f"data: {json.dumps({'type': 'intent', 'content': intent.model_dump()}, ensure_ascii=False)}\n\n"

            # Step 2: Get conversation context
            entity_history = await memory.get_entity_history(session_id)

            # Step 3: Hybrid retrieval
            fused_results = await retrieval.retrieve(
                query_text=query_text,
                hyde_embedding=rewrite_result["hyde_embedding"],
                query_entities=entity_history,
                intent=intent,
                expanded_queries=rewrite_result["expanded_queries"],
            )

            # Get chunk data from Qdrant payloads
            qdrant = request.app.state.qdrant
            chunks_data = []
            for result in fused_results:
                point = await qdrant.get_point(
                    request.app.state.settings.qdrant_collection, result.chunk_id
                )
                if point and point.payload:
                    chunks_data.append(point.payload)

            yield f"data: {json.dumps({'type': 'retrieval', 'content': {'count': len(chunks_data)}}, ensure_ascii=False)}\n\n"

            # Step 4: Get conversation history
            conversation_history = await memory.build_context_from_history(session_id)

            # Step 5: Stream generation
            full_answer = []
            async for token in generator.generate_stream(
                query=query_text,
                chunks_data=chunks_data,
                intent=intent,
                conversation_history=conversation_history,
            ):
                full_answer.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

            answer_text = "".join(full_answer)

            # Step 6: Link citations
            linker = CitationLinker()
            citations = linker.link_citations(answer_text, chunks_data)
            citations_data = [c.model_dump(mode="json") for c in citations]

            yield f"data: {json.dumps({'type': 'citations', 'content': citations_data}, ensure_ascii=False)}\n\n"

            # Step 7: Update conversation memory
            mentioned_entities = [c.get("book_title", "") for c in chunks_data[:3]]
            await memory.add_turn(session_id, "user", query_text, entities=mentioned_entities)
            await memory.add_turn(
                session_id,
                "assistant",
                answer_text,
                chunk_ids=[r.chunk_id for r in fused_results],
            )

            # Step 8: Cache the response
            await cache.set(
                query_text,
                {
                    "answer": answer_text,
                    "citations": citations_data,
                },
            )

            yield "data: [DONE]\n\n"

        except LLMExhaustedError:
            logger.error("All LLM providers exhausted for query: %s", query_text[:100])
            error_msg = "عذرًا، الخدمة غير متاحة حاليًا. نعمل على حل المشكلة، يرجى المحاولة لاحقًا."
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Query pipeline error: %s", e, exc_info=True)
            error_msg = "حدث خطأ أثناء معالجة السؤال. يرجى المحاولة مرة أخرى."
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(process_and_stream(), media_type="text/event-stream")
