
# -*- coding: utf-8 -*-
"""검색·답변 지표.

작은 셋(40문항)에서는 전체 평균이 서사를 만들지 못한다.
유형별로 쪼개야 "조항번호형이 Hybrid로 얼마나 올랐는가"를 확인할 수 있다.
"""
from __future__ import annotations

from collections import defaultdict


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    """gold 청크가 상위 k 안에 하나라도 있으면 1."""
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def mrr(retrieved: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold_set:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not k:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / k


def aggregate(
    rows: list[dict],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """전체 및 질문 유형별 평가 지표를 집계한다."""

    def summarize(group: list[dict]) -> dict:
        n = len(group)
        block = {}

        # 1. 검색 지표
        for k in ks:
            key = f"recall@{k}"
            block[key] = round(
                sum(row.get(key, 0.0) for row in group) / n, 4
            ) if n else 0.0

        block["mrr"] = round(
            sum(row.get("mrr", 0.0) for row in group) / n, 4
        ) if n else 0.0

        block["n"] = n

        # 2. 답변 관련 지표
        answered_rows = [
            row for row in group
            if "answered" in row
        ]

        if answered_rows:
            block["answer_rate"] = round(
                sum(bool(row["answered"]) for row in answered_rows)
                / len(answered_rows),
                4,
            )

        # 3. 오거절률
        answerable_rows = [
            row for row in answered_rows
            if row.get("answerable", True)
        ]

        if answerable_rows:
            block["false_refusal_rate"] = round(
                sum(not row["answered"] for row in answerable_rows)
                / len(answerable_rows),
                4,
            )

        # 4. 미답변 정확도
        unanswerable_rows = [
            row for row in group
            if row.get("answerable") is False
        ]

        if unanswerable_rows:
            block["abstain_accuracy"] = round(
                sum(
                    (
                        row.get("answered") is False
                        or row.get("grade") == "insufficient"
                    )
                    for row in unanswerable_rows
                ) / len(unanswerable_rows),
                4,
            )

        # 5. 응답 지연시간
        latencies = sorted(
            row["latency_ms"]
            for row in group
            if row.get("latency_ms") is not None
        )

        if latencies:
            block["latency_ms_avg"] = round(
                sum(latencies) / len(latencies),
                4,
            )

            p95_index = max(
                0,
                int(len(latencies) * 0.95) - 1,
            )

            block["latency_ms_p95"] = round(
                latencies[p95_index],
                4,
            )

        return block

    # 6. 전체 집계
    overall = summarize(rows)

    # 7. 질문 유형별 집계
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["type"]].append(row)

    by_type = {
        type_name: summarize(group)
        for type_name, group in sorted(grouped.items())
    }

    return {
        "overall": overall,
        "by_type": by_type,
    }


def compare(tables: dict[str, dict], metric: str = "recall@5") -> list[dict]:
    """3단계 비교표. 파이프라인별 유형 분해를 한 표로 세운다."""
    types = sorted({t for tb in tables.values() for t in tb["by_type"]})
    rows = []

    for t in types:
        row = {"type": t}
        for name, tb in tables.items():
            row[name] = tb["by_type"].get(t, {}).get(metric)
        rows.append(row)

    row = {"type": "전체"}

    for name, tb in tables.items():
        row[name] = tb["overall"].get(metric)

    rows.append(row)
    return rows
