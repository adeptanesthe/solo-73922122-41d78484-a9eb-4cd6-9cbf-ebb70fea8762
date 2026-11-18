"""请求/响应模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DocUpsert(BaseModel):
    """单篇记录的增改:编号、正文、分词序列。"""

    id: str = Field(min_length=1, max_length=128, description="记录编号")
    text: str = Field(default="", description="记录正文")
    tokens: list[str] = Field(default_factory=list, description="分词序列(可为空,空文档计入统计)")


class BatchRequest(BaseModel):
    """批量增改删:同一批内先删后增改。"""

    upserts: list[DocUpsert] = Field(default_factory=list, description="新增/修改的记录")
    deletes: list[str] = Field(default_factory=list, description="删除的编号")


class BatchResponse(BaseModel):
    status: str = "ok"
    total: int = Field(description="生效后的总篇数")
    upserted: int = Field(description="本批提交的增改条数")
    deleted: int = Field(description="本批实际删除的篇数")
    revision: int = Field(description="生效后的语料修订号")


class SearchRequest(BaseModel):
    terms: list[str] = Field(default_factory=list, description="查询词项(服务端再去重)")
    k: int = Field(ge=1, le=20, description="返回篇数,1~20")


class Hit(BaseModel):
    id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[Hit]
    scored: int = Field(description="完整计分篇数")


class CorrectionSuggestRequest(BaseModel):
    """纠错建议请求:只给出记录编号。"""

    id: str = Field(min_length=1, max_length=128, description="现存记录编号")


class ReplacementOut(BaseModel):
    position: int = Field(description="0 起的分词位置")
    original: str = Field(description="原词")
    candidate: str = Field(description="建议替换成的词")


class CorrectionSuggestResponse(BaseModel):
    id: str
    revision: int = Field(description="生成建议时的语料修订号,确认时须原样回传")
    original_tokens: list[str]
    suggested_tokens: list[str]
    replacements: list[ReplacementOut] = Field(description="替换位置及原词/候选词")
    score: float = Field(description="整条建议序列的对数得分")
    token: str = Field(description="建议指纹,确认时须原样回传")


class CorrectionConfirmRequest(BaseModel):
    """确认纠错:携带建议中的修订号与建议指纹。"""

    id: str = Field(min_length=1, max_length=128, description="记录编号")
    revision: int = Field(ge=0, description="生成建议时的语料修订号")
    token: str = Field(min_length=1, description="生成建议时返回的建议指纹")


class CorrectionConfirmResponse(BaseModel):
    status: str = "ok"
    id: str
    revision: int = Field(description="确认生效后的语料修订号")
    applied: list[ReplacementOut] = Field(description="本次实际应用的替换")
    tokens: list[str] = Field(description="生效后的分词序列")
