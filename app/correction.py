"""分词纠错建议:相邻词统计 + 加一平滑 + 替换惩罚 + 整句联合最优。

模型(所有统计均来自“其余记录”,不含被纠错记录本身):

- 相邻词概率(加一平滑): ``P(v | u) = (c(u, v) + 1) / (c(u) + V)``
  其中 ``c(u, v)`` 是相邻词对 ``u, v`` 在其余记录中出现的次数,
  ``c(u)`` 是其余记录中以 ``u`` 开头的相邻词对总次数,``V`` 是其余记录的词表大小。
- 整条序列得分: ``score(t) = Σ_{i=1..m-1} log P(t[i+1] | t[i]) + n_rep * ln(1/2)``,
  ``n_rep`` 为相对原序列发生替换的位置数(每次替换乘 1/2 惩罚)。
- 候选: 在其余记录中出现过、且与原词编辑距离恰好为 1 的词;原词本身始终可选。
- 联合选择: 相邻位置互相耦合,用动态规划在所有候选组合中选整条序列得分最高者。
  同分时选替换次数最少的序列;再同分时按“优先保留原词、候选按字典序”的确定顺序取先者。
- 若其余记录没有任何相邻词证据(无相邻词对),返回原序列,得分记为 0.0(哨兵值)。
- 返回的得分只用于解释排序,不参与确认校验。
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass

MAX_CORRECTION_TOKENS = 20
REPLACEMENT_PENALTY = 0.5
_LOG_PENALTY = math.log(REPLACEMENT_PENALTY)  # 每次替换的 log 惩罚(负数)

# 判定“同分”的相对/绝对容差:仅消除浮点累加误差,不影响真实分数差异
_SCORE_REL_TOL = 1e-12
_SCORE_ABS_TOL = 1e-15


class CorrectionError(Exception):
    """纠错流程的基类异常。"""


class RecordNotFound(CorrectionError):
    """记录编号不存在。"""


class RecordTooLong(CorrectionError):
    """记录分词数超过纠错上限。"""


class RevisionConflict(CorrectionError):
    """确认时语料修订号已变化(期间有成功的批量更正)。"""


class SuggestionMismatch(CorrectionError):
    """确认时提交的建议与当前语料下的建议不一致。"""


@dataclass(frozen=True)
class Replacement:
    position: int  # 0 起的分词位置
    original: str
    candidate: str


@dataclass(frozen=True)
class Suggestion:
    doc_id: str
    revision: int
    original_tokens: tuple
    suggested_tokens: tuple
    replacements: tuple  # tuple[Replacement, ...]
    score: float
    token: str  # 建议指纹,确认时回传


@dataclass(frozen=True)
class AdjacencyStats:
    """其余记录的相邻词统计。"""

    vocab: frozenset  # 其余记录出现过的所有词
    bigrams: dict  # (u, v) -> 次数
    context: dict  # u -> 以 u 开头的相邻词对总次数
    n_vocab: int
    by_length: dict  # 词长 -> 该长度的词(元组),用于缩小编辑距离比较范围

    @property
    def has_evidence(self) -> bool:
        return bool(self.bigrams)


def is_edit_distance_one(a: str, b: str) -> bool:
    """判断 a、b 的编辑距离是否恰好为 1(替换/插入/删除一个字符)。"""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:  # 保证 a 更短,则只可能是 b 多插入了一个字符
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1 :]


def build_adjacency_stats(docs: dict, exclude_id: str) -> AdjacencyStats:
    """统计其余记录(排除 exclude_id)的词表与相邻词对。"""
    bigrams: Counter = Counter()
    vocab: set = set()
    for did, (_, tokens) in docs.items():
        if did == exclude_id:
            continue
        vocab.update(tokens)
        for u, v in zip(tokens, tokens[1:]):
            bigrams[(u, v)] += 1

    context: Counter = Counter()
    for (u, _v), c in bigrams.items():
        context[u] += c

    by_length: dict = {}
    for w in sorted(vocab):
        by_length.setdefault(len(w), []).append(w)

    return AdjacencyStats(
        vocab=frozenset(vocab),
        bigrams=dict(bigrams),
        context=dict(context),
        n_vocab=len(vocab),
        by_length={k: tuple(v) for k, v in by_length.items()},
    )


def _log_adjacent(u: str, v: str, stats: AdjacencyStats) -> float:
    """加一平滑的相邻词对数概率 log P(v | u)。"""
    num = stats.bigrams.get((u, v), 0) + 1
    den = stats.context.get(u, 0) + stats.n_vocab
    return math.log(num) - math.log(den)


def _candidates(token: str, stats: AdjacencyStats) -> list:
    """词表内与原词编辑距离为 1 的候选词,按字典序确定排序(不含原词)。"""
    out = []
    for length in (len(token) - 1, len(token), len(token) + 1):
        for w in stats.by_length.get(length, ()):
            if is_edit_distance_one(w, token):
                out.append(w)
    return sorted(out)


def _sequence_score(tokens, stats: AdjacencyStats) -> float:
    """整条序列的相邻词对数概率和(不含替换惩罚)。"""
    return sum(_log_adjacent(u, v, stats) for u, v in zip(tokens, tokens[1:]))


def _scores_tied(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_SCORE_REL_TOL, abs_tol=_SCORE_ABS_TOL)


def _better(a: tuple, b: tuple | None) -> bool:
    """(得分, 替换数) 严格优于 b;同分(容差内)时替换少者优。"""
    if b is None:
        return True
    sa, ra = a
    sb, rb = b
    if _scores_tied(sa, sb):
        return ra < rb
    return sa > sb


def _choose_sequence(tokens: tuple, stats: AdjacencyStats):
    """动态规划联合选择得分最高的整条序列。

    返回 (suggested_tokens, score, replacements)。得分同分时取替换最少。
    """
    m = len(tokens)
    if m < 2 or not stats.has_evidence:
        # 无其他记录的相邻词证据 / 单 token:无法比较整句,原样返回
        return tokens, 0.0, ()

    # 每个位置:原词优先,其余候选按字典序
    choices = []
    for tok in tokens:
        choices.append([(tok, 0)] + [(w, 1) for w in _candidates(tok, stats)])

    if all(len(c) == 1 for c in choices):
        return tokens, _sequence_score(tokens, stats), ()

    # prev[j] = (得分, 替换数):以 choices[i][j] 结尾的最优前缀
    # 位置 0 若是替换也要计入惩罚
    prev = [
        (_LOG_PENALTY if replaced else 0.0, replaced) for _tok, replaced in choices[0]
    ]
    back: list = []  # back[i][j] = 位置 i 选 j 时,位置 i-1 的最优下标
    for i in range(1, m):
        cur = []
        cur_back = []
        for j, (v, replaced) in enumerate(choices[i]):
            best = None
            best_k = 0
            penalty = _LOG_PENALTY if replaced else 0.0
            for k, (u, _u_rep) in enumerate(choices[i - 1]):
                pscore, preps = prev[k]
                # 得分优先;得分在容差内相等时替换次数少者优先。
                # 严格更优才替换,因此同分同替换数时保留先出现者(原词优先、候选字典序)。
                cand = (pscore + _log_adjacent(u, v, stats) + penalty, preps + replaced)
                if _better(cand, best):
                    best, best_k = cand, k
            cur.append(best)
            cur_back.append(best_k)
        prev = cur
        back.append(cur_back)

    best_idx = 0
    for j in range(1, len(prev)):
        if _better(prev[j], prev[best_idx]):
            best_idx = j
    score, _reps = prev[best_idx]

    picked = [choices[m - 1][best_idx][0]]
    for i in range(m - 1, 0, -1):
        best_idx = back[i - 1][best_idx]
        picked.append(choices[i - 1][best_idx][0])
    picked.reverse()

    replacements = tuple(
        Replacement(i, a, b) for i, (a, b) in enumerate(zip(tokens, picked)) if a != b
    )
    return tuple(picked), score, replacements


def _fingerprint(doc_id: str, revision: int, original, suggested) -> str:
    payload = json.dumps(
        {
            "id": doc_id,
            "revision": revision,
            "original": list(original),
            "suggested": list(suggested),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_suggestion(docs: dict, revision: int, doc_id: str) -> Suggestion:
    """为指定记录生成纠错建议(纯函数,不修改任何状态)。"""
    if doc_id not in docs:
        raise RecordNotFound(f"记录 {doc_id!r} 不存在")
    _text, tokens = docs[doc_id]
    tokens = tuple(tokens)
    if len(tokens) > MAX_CORRECTION_TOKENS:
        raise RecordTooLong(
            f"记录 {doc_id!r} 分词数 {len(tokens)} 超过纠错上限 {MAX_CORRECTION_TOKENS}"
        )

    stats = build_adjacency_stats(docs, doc_id)
    suggested, score, replacements = _choose_sequence(tokens, stats)
    return Suggestion(
        doc_id=doc_id,
        revision=revision,
        original_tokens=tokens,
        suggested_tokens=suggested,
        replacements=replacements,
        score=score,
        token=_fingerprint(doc_id, revision, tokens, suggested),
    )
