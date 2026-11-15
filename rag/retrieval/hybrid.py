"""Hybrid retrieval: dense + sparse + structured fusion via Reciprocal Rank Fusion.

RRF formula: score(d) = sum over systems s of weight_s / (rrf_k + rank_s(d))
where rank_s(d) is the 1-based rank of document d in the result list of
system s. This avoids mixing incompatible raw score scales (cosine
similarity vs. BM25 scores) — only ranks are fused.

Structured results are never discarded: exact structured records are returned
alongside the fused document list, and structured evidence also contributes a
pseudo-ranked list to the fusion when a place/entity was detected.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from rag.config import RAGConfig
from rag.ingestion.enhanced_loader import map_status
from rag.query_processing import ProcessedQuery, QueryProcessor
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.sparse import SparseIndex
from rag.retrieval.structured import StructuredRetriever
from rag.schemas import (
    RetrievalResult,
    RetrievedDocument,
    SourceMetadata,
    SourceType,
    StructuredPlace,
    StructuredRecord,
    VerificationMetadata,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuses dense, sparse, and structured retrieval with RRF."""

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseIndex,
        structured: StructuredRetriever,
        query_processor: QueryProcessor,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.structured = structured
        self.qp = query_processor
        self.config = config or RAGConfig()

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        processed: Optional[ProcessedQuery] = None,
    ) -> RetrievalResult:
        started_docs: List[RetrievedDocument] = []
        methods: List[str] = []
        result_top_k = top_k or self.config.top_k

        pq = processed or self.qp.process(query)
        effective_filters = {**pq.filters, **(filters or {})}
        warnings: List[str] = list(pq.warnings)

        # ---- structured retrieval (exact) -----------------------------
        structured_places: List[StructuredPlace] = []
        structured_pseudo_rank: List[Tuple[str, int]] = []  # (place_id, rank)

        if pq.detected_places:
            places = self.structured.records_by_ids(pq.detected_places)
            structured_places.extend(places)
            structured_pseudo_rank.extend(
                (p.place_id, i) for i, p in enumerate(places, start=1)
            )
            methods.append("structured")
        if pq.structured_fields:
            # exact field lookups for each detected/targeted place
            target_ids = pq.detected_places or []
            if target_ids:
                methods.append("structured_fields")
            else:
                warnings.append("structured_intent_without_place")
                # structured intent (prices/hours/...) with NO specific
                # place, but a known city: fetch the city's places so the
                # requested fields are answered PER PLACE ("the main
                # attractions in Luxor, their opening hours and ticket
                # prices") instead of rendering only the city overview
                if pq.detected_city:
                    city_places = self.structured.find_places_by_city(
                        pq.detected_city
                    )
                    structured_places.extend(city_places)
                    structured_pseudo_rank.extend(
                        (p.place_id, i)
                        for i, p in enumerate(city_places, start=1)
                    )
                    methods.append("structured")
        elif not pq.detected_places and pq.detected_city:
            city_places = self.structured.find_places_by_city(pq.detected_city)
            structured_places.extend(city_places)
            structured_pseudo_rank.extend(
                (p.place_id, i) for i, p in enumerate(city_places, start=1)
            )
            methods.append("structured")
        if pq.intent == "definition" and pq.detected_places and not structured_places:
            places = self.structured.records_by_ids(pq.detected_places)
            structured_places.extend(places)
            methods.append("structured")

        # dedupe structured records
        seen_ids = set()
        unique_structured: List[StructuredPlace] = []
        for p in structured_places:
            if p.place_id in seen_ids:
                continue
            seen_ids.add(p.place_id)
            unique_structured.append(p)

        # ---- dense retrieval -------------------------------------------
        dense_docs: List[RetrievedDocument] = []
        if pq.normalized:
            dense_docs = self.dense.retrieve(
                query, top_k=self.config.dense_top_k, filters=effective_filters
            )
            methods.append("dense") if dense_docs else None
            started_docs.extend(dense_docs)

        # ---- sparse retrieval ------------------------------------------
        sparse_hits: List[Dict[str, Any]] = []
        if pq.normalized:
            sparse_hits = self.sparse.search(
                query, top_k=self.config.sparse_top_k
            )
            if sparse_hits:
                methods.append("sparse")

        # ---- RRF fusion --------------------------------------------------
        fused = self._rrf_fuse(
            dense_docs=dense_docs,
            sparse_hits=sparse_hits,
            structured_pseudo_rank=structured_pseudo_rank,
        )

        # attach structured records' verification/source data to documents
        by_place = {p.place_id: p for p in unique_structured}
        for doc in fused:
            place = by_place.get(doc.place_id or "")
            if place:
                doc.verification = place.verification
                doc.source = place.source

        if not fused:
            warnings.append("no_relevant_results")

        extras = self._collect_extras(pq)

        return RetrievalResult(
            query=query,
            documents=fused[:result_top_k],
            structured_records=unique_structured,
            structured_extras=extras,
            intent=pq.intent,
            intents=list(pq.intents or [pq.intent] if pq.intent else []),
            detected_entities={
                "places": pq.detected_places,
                "city": pq.detected_city,
                "cities": pq.detected_cities,
                "structured_fields": pq.structured_fields,
                "days": pq.days,
            },
            filters=effective_filters,
            methods_used=sorted(set(m for m in methods if m)),
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Enhanced-KB structured extras (cities, activities, itineraries, rules)
    # ------------------------------------------------------------------
    def _collect_extras(self, pq: ProcessedQuery) -> List[StructuredRecord]:
        """Fetch non-place structured records relevant to the query.

        These extras come only from the enhanced KB collections and preserve
        the original records verbatim. Transportation intent yields an explicit
        ``unavailable`` record when the routes collection is empty — routes are
        never invented.
        """
        if not self.structured.kb.is_enhanced:
            return []
        kb = self.structured.kb
        extras: List[StructuredRecord] = []
        seen_rids: set = set()

        def add(record_type: str, record: Dict[str, Any], title: str) -> None:
            rid = str(record.get("id") or record_type)
            if rid in seen_rids:
                return  # no duplicate sections (e.g. same rule for 2 places)
            seen_rids.add(rid)
            dq = record.get("data_quality")
            if record.get("verification_status"):
                raw_status = record.get("verification_status")
            elif isinstance(dq, dict) and dq:
                # per-field statuses: take the first entry (records in the
                # enhanced KB are uniformly unverified unless marked otherwise)
                first = next(iter(dq.values()))
                raw_status = first.get("status") if isinstance(first, dict) else None
            else:
                raw_status = None
            extras.append(
                StructuredRecord(
                    record_type=record_type,
                    record_id=str(record.get("id") or rid),
                    title=title,
                    data=record,
                    verification=VerificationMetadata(
                        status=map_status(raw_status),
                        last_verified=record.get("last_verified"),
                    ),
                    source=SourceMetadata(
                        kb_name=kb.kb_name,
                        kb_version=kb.kb_version,
                        kb_last_updated=kb.kb_last_updated,
                        source_file="rehlaai_rag_data_enhanced.json",
                        record_id=record.get("id"),
                        disclaimer=kb.disclaimer,
                    ),
                )
            )

        city = pq.detected_city

        # city record — one per mentioned destination (Bug B: a query naming
        # two cities previously retrieved evidence for only one of them,
        # chosen nondeterministically)
        requested_cities = list(pq.detected_cities) or ([city] if city else [])
        for city_name in requested_cities:
            city_rec = self.structured.find_city(city_name)
            if city_rec:
                add("city", city_rec, city_rec.get("name_en") or city_name)

        # activities intent: activities of the detected city (or all)
        matched_intents = set(pq.intents or [pq.intent])
        if "activities" in matched_intents:
            activities = (
                self.structured.find_activities_by_city(city) if city else kb.activities
            )
            for act in activities:
                add("activity", act, act.get("name_en") or act.get("id"))

        # itinerary intent: itineraries of the detected city (or all).
        # When the user requested a specific number of days, return ONLY the
        # best-matching itinerary — returning every KB plan produced answers
        # with several unrelated itineraries ("4-day trip" -> six plans).
        # The renderer still enforces one plan and exact day counts.
        if "itinerary" in matched_intents:
            if len(requested_cities) > 1:
                # multi-destination query (Bug B): combine KB plans so every
                # requested destination gets evidence, without duplicates
                selected = self._select_multi_city_itineraries(
                    requested_cities, pq.days
                )
            else:
                itineraries = (
                    self.structured.find_itineraries_by_city(city) if city else kb.itineraries
                )
                requested_days = pq.days
                if requested_days and len(itineraries) > 1:

                    def _day_distance(it: Dict[str, Any]) -> tuple:
                        try:
                            count = int(
                                it.get("days_count") or len(it.get("days") or [])
                            )
                        except (TypeError, ValueError):
                            count = 0
                        # records without a usable day count rank last; ties
                        # prefer a plan that can cover the FULL requested
                        # length over one that falls short of it
                        if not count:
                            return (99, 99, 1)
                        # destination purity (Bug A): when the plan must be
                        # TRUNCATED to the requested length, a combined plan
                        # that merely includes the requested city must not
                        # outrank a pure-city plan ("4 days in Aswan"
                        # previously selected the 5-day Luxor->Aswan plan and
                        # started in Luxor). The penalty applies only in the
                        # truncation case, so the exact-length combined-plan
                        # selections (5-day Luxor, budget trip without a
                        # city) and short-trip choices are unchanged.
                        it_city = str(it.get("city") or "").strip().lower()
                        impure = (
                            bool(city)
                            and bool(it_city)
                            and it_city != city.lower()
                            and count > requested_days
                        )
                        return (
                            1 if impure else 0,
                            abs(count - requested_days),
                            0 if count >= requested_days else 1,
                        )

                    best = min(itineraries, key=_day_distance)
                    itineraries = [best]
                selected = itineraries
            for it in selected:
                add("itinerary", it, it.get("title_en") or it.get("id"))

        # transportation intent: explicit unavailable record when empty
        # (placed first — it directly answers the query)
        if "transportation" in (pq.structured_fields or []) or "transportation" in matched_intents:
            routes = self.structured.get_transportation_routes()
            if not kb.transportation_routes:
                extras.append(
                    StructuredRecord(
                        record_type="transportation_route",
                        record_id="transportation_routes_empty",
                        title="Transportation routes",
                        data={
                            "status": "unavailable",
                            "note": routes.note or kb.transportation_note,
                        },
                        verification=VerificationMetadata(
                            status=VerificationStatus.UNAVAILABLE,
                            notes=routes.note,
                        ),
                        source=SourceMetadata(
                            kb_name=kb.kb_name,
                            kb_version=kb.kb_version,
                            source_file="rehlaai_rag_data_enhanced.json",
                            record_id="transportation_routes_empty",
                            disclaimer=kb.disclaimer,
                        ),
                    )
                )

        # planning rules attached to the detected places
        for place_id in pq.detected_places:
            for rule in self.structured.get_planning_rules_for_place(place_id):
                add("planning_rule", rule, f"Planning rule {rule.get('id')}")

        return extras

    # ------------------------------------------------------------------
    # Multi-destination itinerary selection (Bug B)
    # ------------------------------------------------------------------
    def _select_multi_city_itineraries(
        self, cities: List[str], requested_days: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Select itinerary evidence for a multi-destination query.

        Reuses the existing KB plans — no new itinerary engine. When one KB
        plan already covers every requested city ("Luxor and Aswan"), it is
        returned as-is and the existing renderer/validator handle it like any
        single plan. Otherwise the best exact-city plan per destination (in
        mention order) is merged into ONE synthetic record: already-shown
        stops are dropped, fully-duplicate days are skipped, days are
        renumbered. Every stop comes verbatim from a KB plan — nothing is
        invented; destinations without KB evidence contribute no days.
        """
        def _best_plan(city_l: str) -> Optional[Dict[str, Any]]:
            plans = [
                it for it in self.structured.kb.itineraries
                if str(it.get("city") or "").strip().lower() == city_l
            ]
            if not plans:
                return None
            if requested_days:
                def _dist(it: Dict[str, Any]) -> tuple:
                    try:
                        count = int(
                            it.get("days_count") or len(it.get("days") or [])
                        )
                    except (TypeError, ValueError):
                        count = 0
                    if not count:
                        return (99, 1)
                    return (
                        abs(count - requested_days),
                        0 if count >= requested_days else 1,
                    )
                return min(plans, key=_dist)
            # no length requested: the plan with the most day evidence
            return max(plans, key=lambda it: len(it.get("days") or []))

        # one KB plan that already covers ALL requested cities wins outright
        for it in self.structured.kb.itineraries:
            it_city = str(it.get("city") or "").lower()
            if it_city and all(c.lower() in it_city for c in cities):
                return [it]

        merged_days: List[Dict[str, Any]] = []
        used_place_ids: set = set()
        titles_en: List[str] = []
        titles_ar: List[str] = []
        for city_name in cities:
            plan = _best_plan(city_name.strip().lower())
            if plan is None:
                continue  # no evidence for this destination — never invented
            city_rec = self.structured.find_city(city_name)

            def _display(key: str) -> str:
                if city_rec and city_rec.get(key):
                    return str(city_rec[key])
                return city_name

            titles_en.append(_display("name_en"))
            titles_ar.append(_display("name_ar"))
            for day in plan.get("days") or []:
                stops = [
                    stop for stop in (day.get("stops") or [])
                    if str(stop.get("place_id") or "") not in used_place_ids
                ]
                if not stops:
                    continue  # day fully duplicated by an earlier destination
                for stop in stops:
                    used_place_ids.add(str(stop.get("place_id") or ""))
                merged_days.append({
                    "day": len(merged_days) + 1,
                    "title": day.get("title"),
                    "stops": stops,
                    "tip": day.get("tip"),
                })

        if not merged_days:
            return []  # nothing retrievable: the honest "not available" path
        record = {
            "id": "itinerary_combined",
            "title_en": " and ".join(dict.fromkeys(titles_en)),
            "title_ar": " و".join(dict.fromkeys(titles_ar)),
            "city": " and ".join(dict.fromkeys(titles_en)),
            "days_count": len(merged_days),
            "days": merged_days,
            "interests": [],
            "pace": "",
            # merged plan: honestly unverified (the KB has no such plan)
            "data_quality": {
                "days": {
                    "status": "unverified",
                    "source_url": None,
                    "verification_date": None,
                },
            },
        }
        return [record]

    # ------------------------------------------------------------------
    def _rrf_fuse(
        self,
        dense_docs: List[RetrievedDocument],
        sparse_hits: List[Dict[str, Any]],
        structured_pseudo_rank: List[Tuple[str, int]],
    ) -> List[RetrievedDocument]:
        """Fuse ranked lists with weighted Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        first_seen: Dict[str, RetrievedDocument] = {}
        k = self.config.rrf_k

        for rank, doc in enumerate(dense_docs, start=1):
            scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + (
                self.config.dense_weight / (k + rank)
            )
            first_seen.setdefault(doc.doc_id, doc)

        for rank, hit in enumerate(sparse_hits, start=1):
            doc_id = hit["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + (
                self.config.sparse_weight / (k + rank)
            )
            if doc_id not in first_seen:
                meta = hit.get("metadata", {})
                first_seen[doc_id] = RetrievedDocument(
                    doc_id=doc_id,
                    content=hit.get("content", ""),
                    place_id=meta.get("place_id"),
                    collection=str(meta.get("collection", "document")),
                    section=meta.get("section"),
                    metadata=dict(meta),
                    score=None,
                    rank=rank,
                    source_type=SourceType.SPARSE,
                )

        # structured evidence contributes to fusion so that exact matches for
        # a detected entity are not ranked below weak semantic hits
        for place_id, rank in structured_pseudo_rank:
            # structured pseudo-docs use the place's practical section id
            for doc_id, doc in first_seen.items():
                if doc.place_id == place_id and doc.section == "practical":
                    scores[doc_id] = scores.get(doc_id, 0.0) + (
                        self.config.structured_weight / (k + rank)
                    )

        ordered_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
        fused: List[RetrievedDocument] = []
        for rank, doc_id in enumerate(ordered_ids, start=1):
            doc = first_seen[doc_id].model_copy(deep=True)
            doc.score = scores[doc_id]
            doc.rank = rank
            if first_seen[doc_id].source_type == SourceType.SPARSE:
                doc.source_type = SourceType.HYBRID if doc_id in {
                    d.doc_id for d in dense_docs
                } else SourceType.SPARSE
            else:
                doc.source_type = SourceType.HYBRID if doc_id in {
                    h["doc_id"] for h in sparse_hits
                } else doc.source_type
            fused.append(doc)
        return fused

    # ------------------------------------------------------------------
    def deduplicate_by_content(self, docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Remove documents with identical content (keeps the highest rank)."""
        seen: set = set()
        out: List[RetrievedDocument] = []
        for doc in docs:
            key = doc.content.strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(doc)
        return out
