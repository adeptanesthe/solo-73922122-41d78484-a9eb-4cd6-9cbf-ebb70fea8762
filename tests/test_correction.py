"""纠错模块测试:候选约束、整句联合最优、同分取替换最少、确认流程原子性。"""
import itertools
import math
import random

import pytest
from fastapi.testclient import TestClient

from app.correction import (
    AdjacencyStats,
    RecordNotFound,
    RecordTooLong,
    Replacement,
    RevisionConflict,
    SuggestionMismatch,
    _choose_sequence,
    build_adjacency_stats,
    is_edit_distance_one,
)
from app.index import MAX_DOCS, IndexManager, search
from app.main import create_app


@pytest.fixture()
def client():
    return TestClient(create_app(IndexManager()))


def upsert(client, docs, deletes=None):
    return client.post("/documents/batch", json={"upserts": docs, "deletes": deletes or []})


# ---------------------------------------------------------------- 基础构件


def test_edit_distance_one():
    assert is_edit_distance_one("轴承过垫", "轴承过热")  # 替换
    assert is_edit_distance_one("轴承", "轴承过")  # 插入
    assert is_edit_distance_one("轴承过热", "轴承过")  # 删除
    assert not is_edit_distance_one("轴承", "轴承")
    assert not is_edit_distance_one("轴承", "轴承过热")  # 差 2 个字
    assert not is_edit_distance_one("abc", "xyz")


def test_stats_exclude_target_record():
    docs = {
        "A": ("t", ("own1", "own2")),
        "B": ("t", ("x", "y")),
        "C": ("t", ("x", "y")),
    }
    st = build_adjacency_stats(docs, "A")
    assert ("own1", "own2") not in st.bigrams  # 目标记录自身不提供证据
    assert st.bigrams[("x", "y")] == 2
    assert st.context["x"] == 2
    assert st.n_vocab == 2


# ---------------------------------------------------------------- 整句选择


def test_replaces_typo_with_corpus_supported_word(client):
    upsert(client, [
        {"id": "A", "text": "t", "tokens": ["数控机床", "轴承过垫", "更换轴承"]},
        *[
            {"id": f"B{i}", "text": "t", "tokens": ["数控机床", "轴承过热", "更换轴承"]}
            for i in range(4)
        ],
    ])
    body = client.post("/corrections/suggest", json={"id": "A"}).json()
    assert body["id"] == "A" and body["revision"] == 1
    assert body["replacements"] == [
        {"position": 1, "original": "轴承过垫", "candidate": "轴承过热"}
    ]
    assert body["suggested_tokens"] == ["数控机床", "轴承过热", "更换轴承"]
    assert body["original_tokens"] == ["数控机床", "轴承过垫", "更换轴承"]


def test_no_adjacent_evidence_returns_original():
    # 其余记录每篇只有一个词:没有任何相邻词对
    st = AdjacencyStats(
        vocab=frozenset({"a", "b"}),
        bigrams={},
        context={},
        n_vocab=2,
        by_length={1: ("a", "b")},
    )
    tokens = ("a", "b")
    suggested, score, reps = _choose_sequence(tokens, st)
    assert suggested == tokens and reps == () and score == 0.0


def test_single_token_returns_original():
    st = AdjacencyStats(
        vocab=frozenset({"a", "b"}), bigrams={("a", "b"): 3}, context={"a": 3},
        n_vocab=2, by_length={1: ("a", "b")},
    )
    assert _choose_sequence(("a",), st) == (("a",), 0.0, ())


def test_tie_chooses_fewer_replacements():
    # 构造:(u,v) 与 (u2,v) 的相邻概率相差一倍,恰好抵消多一次替换的 1/2 惩罚。
    # 替换 v 得 1 次替换;同时替换 u->u2 得多 1 次替换,两者整句得分相同 => 取替换少的。
    st = AdjacencyStats(
        vocab=frozenset({"aa", "ab", "xx", "xy", "ww"}),
        bigrams={("aa", "xy"): 1, ("ab", "xy"): 3, ("xy", "ww"): 1},
        context={"aa": 3, "ab": 3, "xy": 1},
        n_vocab=5,
        by_length={2: ("aa", "ab", "ww", "xx", "xy")},
    )
    tokens = ("aa", "xx", "ww")
    suggested, score, reps = _choose_sequence(tokens, st)
    assert suggested == ("aa", "xy", "ww")
    assert len(reps) == 1 and reps[0].position == 1

    # 直接核对:两次替换的组合与一次替换得分相同
    def logp(u, v):
        return math.log(st.bigrams.get((u, v), 0) + 1) - math.log(st.context.get(u, 0) + st.n_vocab)

    one = logp("aa", "xy") + logp("xy", "ww") + math.log(0.5)
    two = logp("ab", "xy") + logp("xy", "ww") + 2 * math.log(0.5)
    assert math.isclose(one, two, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(score, one, rel_tol=1e-12, abs_tol=1e-12)


def test_dp_matches_bruteforce_on_random_corpora():
    rng = random.Random(11)
    vocab = ["a1", "a2", "b1", "b2", "c1", "c2"]
    for _ in range(200):
        bigrams = {}
        for u in vocab:
            for v in vocab:
                c = rng.choice([0, 0, 1, 2, 5])
                if c:
                    bigrams[(u, v)] = c
        context = {}
        for (u, _v), c in bigrams.items():
            context[u] = context.get(u, 0) + c
        by_length = {2: tuple(vocab)}
        st = AdjacencyStats(
            vocab=frozenset(vocab), bigrams=bigrams, context=context,
            n_vocab=len(vocab), by_length=by_length,
        )
        tokens = tuple(rng.choice(vocab) for _ in range(rng.randint(0, 4)))
        suggested, score, reps = _choose_sequence(tokens, st)

        # 穷举所有 原词/编辑距离 1 候选 的组合,按 (得分, -替换数) 取最大
        if tokens and bigrams:
            choices = []
            for tok in tokens:
                cands = [tok] + [
                    w for w in sorted(vocab) if is_edit_distance_one(w, tok)
                ]
                choices.append(cands)

            def seq_score(seq):
                return sum(
                    math.log(bigrams.get((u, v), 0) + 1)
                    - math.log(context.get(u, 0) + len(vocab))
                    for u, v in zip(seq, seq[1:])
                ) + math.log(0.5) * sum(1 for a, b in zip(tokens, seq) if a != b)

            best = max(
                itertools.product(*choices),
                key=lambda seq: (round(seq_score(seq), 12), -sum(1 for a, b in zip(tokens, seq) if a != b)),
            )
            assert math.isclose(score, seq_score(best), rel_tol=1e-12, abs_tol=1e-12)
            assert len(reps) == sum(1 for a, b in zip(tokens, best) if a != b)
        else:
            assert suggested == tokens


# ---------------------------------------------------------------- 确认流程


def _setup_typo(client):
    return upsert(client, [
        {"id": "A", "text": "A 正文", "tokens": ["数控机床", "轴承过垫", "更换轴承"]},
        *[
            {"id": f"B{i}", "text": "t", "tokens": ["数控机床", "轴承过热", "更换轴承"]}
            for i in range(4)
        ],
    ])


def test_confirm_applies_tokens_and_index_atomically(client):
    _setup_typo(client)
    sug = client.post("/corrections/suggest", json={"id": "A"}).json()
    # 纠错前:错词可检索,正确词序不影响其他记录
    assert [h["id"] for h in client.post("/search", json={"terms": ["轴承过垫"], "k": 5}).json()["hits"]] == ["A"]

    r = client.post(
        "/corrections/confirm",
        json={"id": "A", "revision": sug["revision"], "token": sug["token"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["revision"] == sug["revision"] + 1
    assert body["tokens"] == ["数控机床", "轴承过热", "更换轴承"]
    assert body["applied"] == [
        {"position": 1, "original": "轴承过垫", "candidate": "轴承过热"}
    ]
    # 旧分词已不可检索,新分词可检索;正文不变
    assert client.post("/search", json={"terms": ["轴承过垫"], "k": 5}).json()["hits"] == []
    hits = client.post("/search", json={"terms": ["轴承过热"], "k": 20}).json()["hits"]
    assert "A" in [h["id"] for h in hits]
    assert next(h for h in hits if h["id"] == "A")["text"] == "A 正文"
    assert client.get("/stats").json()["revision"] == body["revision"]


def test_stale_revision_rejected_after_successful_batch(client):
    _setup_typo(client)
    sug = client.post("/corrections/suggest", json={"id": "A"}).json()
    other = client.post("/corrections/suggest", json={"id": "B0"}).json()

    assert upsert(client, [{"id": "Z", "text": "t", "tokens": ["z"]}]).status_code == 200
    r = client.post(
        "/corrections/confirm",
        json={"id": "A", "revision": sug["revision"], "token": sug["token"]},
    )
    assert r.status_code == 409
    # 旧建议未生效:分词保持原样
    st = client.app.state.manager.current()
    assert list(st.docs["A"][1]) == ["数控机床", "轴承过垫", "更换轴承"]
    assert other["revision"] == sug["revision"]


def test_failed_batch_does_not_invalidate_confirm(client):
    _setup_typo(client)
    sug = client.post("/corrections/suggest", json={"id": "A"}).json()
    rev_before = client.get("/stats").json()["revision"]

    # 超过容量上限的批量更正失败:修订号不变,旧确认仍有效
    big = [
        {"id": f"X{i}", "text": "t", "tokens": ["x"]}
        for i in range(MAX_DOCS + 1)
    ]
    assert upsert(client, big).status_code == 409
    assert client.get("/stats").json()["revision"] == rev_before

    r = client.post(
        "/corrections/confirm",
        json={"id": "A", "revision": sug["revision"], "token": sug["token"]},
    )
    assert r.status_code == 200
    assert r.json()["revision"] == rev_before + 1


def test_mismatched_suggestion_token_rejected(client):
    _setup_typo(client)
    sug_a = client.post("/corrections/suggest", json={"id": "A"}).json()
    sug_b = client.post("/corrections/suggest", json={"id": "B0"}).json()
    r = client.post(
        "/corrections/confirm",
        json={"id": "A", "revision": sug_a["revision"], "token": sug_b["token"]},
    )
    assert r.status_code == 409
    st = client.app.state.manager.current()
    assert list(st.docs["A"][1]) == ["数控机床", "轴承过垫", "更换轴承"]


def test_confirm_twice_second_is_stale(client):
    _setup_typo(client)
    sug = client.post("/corrections/suggest", json={"id": "A"}).json()
    payload = {"id": "A", "revision": sug["revision"], "token": sug["token"]}
    assert client.post("/corrections/confirm", json=payload).status_code == 200
    assert client.post("/corrections/confirm", json=payload).status_code == 409


def test_suggest_404_for_missing_and_422_for_too_long(client):
    r = client.post("/corrections/suggest", json={"id": "nope"})
    assert r.status_code == 404
    upsert(client, [
        {"id": "L20", "text": "t", "tokens": ["w"] * 20},
        {"id": "L21", "text": "t", "tokens": ["w"] * 21},
    ])
    assert client.post("/corrections/suggest", json={"id": "L20"}).status_code == 200
    assert client.post("/corrections/suggest", json={"id": "L21"}).status_code == 422


def test_suggest_does_not_change_state(client):
    _setup_typo(client)
    before = client.get("/stats").json()
    client.post("/corrections/suggest", json={"id": "A"})
    assert client.get("/stats").json() == before


# ---------------------------------------------------------------- 索引层


def test_manager_suggest_and_confirm_directly():
    mgr = IndexManager()
    mgr.apply_batch([
        ("A", "t", ("aa", "ab")),
        ("B", "t", ("aa", "ac")),
        ("C", "t", ("aa", "ac")),
    ], [])
    sug = mgr.suggest_correction("A")
    assert sug.replacements == (
        Replacement(position=1, original="ab", candidate="ac"),
    )  # 仅以其余记录(不含 A 自身)为证据
    with pytest.raises(RecordNotFound):
        mgr.suggest_correction("missing")
    with pytest.raises(RevisionConflict):
        mgr.confirm_correction("A", sug.revision + 1, sug.token)
    with pytest.raises(SuggestionMismatch):
        mgr.confirm_correction("A", sug.revision, "0" * 64)
    st, applied = mgr.confirm_correction("A", sug.revision, sug.token)
    assert applied == sug.replacements
    assert st.revision == sug.revision + 1
    assert st.docs["A"][1] == ("aa", "ac")


def test_manager_confirm_uses_snapshot_and_blocks_concurrent_changes():
    mgr = IndexManager()
    mgr.apply_batch([
        ("A", "t", ("数控机床", "轴承过垫", "更换轴承")),
        ("B", "t", ("数控机床", "轴承过热", "更换轴承")),
        ("C", "t", ("数控机床", "轴承过热", "更换轴承")),
    ], [])
    sug = mgr.suggest_correction("A")
    assert sug.replacements
    st, applied = mgr.confirm_correction("A", sug.revision, sug.token)
    assert applied == sug.replacements
    hits, _ = search(st, ["轴承过热"], 5)
    assert {h[0] for h in hits} == {"A", "B", "C"}
    hits, _ = search(st, ["轴承过垫"], 5)
    assert hits == []
