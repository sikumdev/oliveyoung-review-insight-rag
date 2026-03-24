"""
graph/prompts.py
모든 LLM 프롬프트 템플릿 + Pydantic BaseModel 스키마
"""

from typing import Literal
from pydantic import BaseModel, Field


# ── IntakeResult ─────────────────────────────────────────────────────────────

class IntakeResult(BaseModel):
    query_type: Literal["추천", "비교", "필터해석", "follow_up", "refine"]
    needs_skin_type: bool
    skin_type: str = Field(default="")
    skin_tone: str = Field(default="")
    products_mentioned: list[str] = Field(default_factory=list)
    rewritten_question: str
    user_needs: list[Literal[
        "산뜻함", "끈적임없음", "보습", "자극없음",
        "백탁없음", "밀림없음", "지속력", "눈시림없음"
    ]] = Field(
        default_factory=list,
        description="추천/필터해석 시 유저가 원하는 속성. 명확히 언급된 것만, 없으면 빈 리스트"
    )



# ── ReviewInsightResult ───────────────────────────────────────────────────────

class SkinTypeReaction(BaseModel):
    지성: str | None = None
    건성: str | None = None
    복합성: str | None = None
    민감성: str | None = None
    중성: str | None = None
    복합건성: str | None = None
    복합지성: str | None = None

class ReviewInsightResult(BaseModel):
    positive_keywords: list[str] = Field(
        default_factory=list,
        description="자주 언급된 긍정 키워드 최대 5개"
    )
    negative_keywords: list[str] = Field(
        default_factory=list,
        description="자주 언급된 부정 키워드 최대 5개"
    )
    skin_type_reactions: SkinTypeReaction = Field(
        default_factory=SkinTypeReaction,
        description="피부타입별 리뷰 반응 요약"
    )
    failure_points: list[str] = Field(
        default_factory=list,
        description="실패 포인트 최대 3개 (트러블, 부작용, 맞지 않는 케이스)"
    )
    overall_sentiment: Literal["긍정", "중립", "부정"] = "긍정"
    repurchase_rate_comment: str = Field(
        default="",
        description="재구매율에 대한 한 줄 코멘트"
    )


# ── CompareAxisResult ─────────────────────────────────────────────────────────

class CompareAxisResult(BaseModel):
    axes: list[str] = Field(
        description="비교 기준 목록 (보습력, 자극, 발림성, 지속력, 끈적임, 백탁, 향, 흡수력 중). 최소 1개 최대 5개",
        min_length=1,
        max_length=5,
    )


# ── CompareSummaryResult ──────────────────────────────────────────────────────

class AxisAnalysis(BaseModel):
    axis: str = Field(description="비교 기준명 (예: 보습력, 자극, 발림성)")
    analysis: str = Field(description="해당 기준에 대한 리뷰 기반 분석 2-3문장")

class CompareSummaryResult(BaseModel):
    product_name: str
    axes: list[AxisAnalysis] = Field(
        description="비교 기준별 리뷰 기반 분석 목록"
    )
    best_for: str = Field(default="", description="이 제품이 가장 잘 맞는 사람")
    avoid_if: str = Field(default="", description="이런 사람은 피해야 함")


# ── StructuredCompareAxisResult ───────────────────────────────────────────────

class AxisJudgmentItem(BaseModel):
    axis: str = Field(description="비교 기준명 (예: 보습력, 자극)")
    winner: str | None = Field(default=None, description="우세한 제품명. 무승부이면 null")
    draw: bool = False
    reason: str = Field(description="판정 근거 한 문장")

class StructuredCompareResult(BaseModel):
    judgments: list[AxisJudgmentItem] = Field(
        description="비교 기준별 판정 결과 목록"
    )


# ── intake_node 프롬프트 ──────────────────────────────────────────────────────

INTAKE_SYSTEM = """당신은 올리브영 선크림 리뷰 기반 뷰티 어드바이저 챗봇입니다.
유저의 질문을 분석해서 아래 규칙에 따라 응답하세요.

규칙:
- query_type:
    "추천"      → 조건에 맞는 제품을 추천해달라는 요청
    "비교"      → 두 개 이상의 제품을 비교해달라는 요청
    "필터해석"  → 특정 조건(피부타입/피부톤 등)의 리뷰를 해석해달라는 요청
    "follow_up" → 이전 답변에 대한 후속 질문 (단, 비교/추천 의도가 명확하면 해당 타입으로 분류)
    "refine"    → 이전 결과를 조건 변경으로 재필터링
    위에 해당 없으면 가장 가까운 타입으로 분류 (보통 "필터해석" 또는 "follow_up")

- needs_skin_type 규칙:
    추천 / 필터해석 → 피부타입 AND 피부톤 둘 다 언급 없으면 true, 둘 중 하나라도 있으면 false
    비교 / follow_up / refine → 항상 false
    주의: skin_type이나 skin_tone을 추출했다면 needs_skin_type은 반드시 false여야 함
    주의: 피부톤(퍼스널컬러) 정보만 있어도 추천 가능 → needs_skin_type=false

- skin_type/skin_tone: 언급 없으면 빈 문자열 ""
- skin_tone 추출 기준 — 아래 값 중 하나를 정확히 반환:
    "쿨톤"   → 쿨톤, 파란끼, 분홍빛 피부, 차가운 피부톤
    "여름쿨" → 여름쿨, 여쿨
    "웜톤"   → 웜톤, 노란끼, 황톤, 노르스름한 피부, 따뜻한 피부톤
    "봄웜"   → 봄웜, 봄 퍼스널컬러, 밝은 웜톤
    "가을웜" → 가을웜, 갈웜, 어두운 웜톤, 다크 웜톤
    "뉴트럴" → 뉴트럴, 뉴트럴톤, 쿨도 웜도 아닌
    퍼스널컬러 언급 없으면 ""
    예시: "얼굴에 노란끼가 많아" → "웜톤", "피부가 파란끼 도는 편" → "쿨톤"
- products_mentioned: 비교/필터해석 시 언급된 제품명 목록, 없으면 [] (최대 4개까지 추출)
- rewritten_question: 대화 이력을 반영해 단독으로 이해 가능한 질문으로 재작성
- user_needs: 추천/필터해석 시 유저가 원하는 속성. 아래 8개 중 명확히 언급된 것만 추출
    가능한 값: 산뜻함, 끈적임없음, 보습, 자극없음, 백탁없음, 밀림없음, 지속력, 눈시림없음
    동의어 매핑:
      촉촉한/수분감/촉촉해요 → 보습
      산뜻한/가벼운/보송한  → 산뜻함
      끈적임없는/안끈적이는  → 끈적임없음
      순한/자극없는/민감해도  → 자극없음
      백탁없는/하얗게안뜨는  → 백탁없음
      화잘먹는/밀리지않는    → 밀림없음
      오래가는/지속되는      → 지속력
      눈시림없는/눈안따가운  → 눈시림없음
    애매하거나 언급 없으면 빈 리스트 []
    비교/follow_up/refine은 항상 빈 리스트 []
"""

# 대화 이력은 메시지 객체로 직접 언패킹(*state["messages"][-6:])해서 전달
# → 이 템플릿은 컨텍스트 정보(추천 제품, 선택 제품, 직전 유형)만 담당
INTAKE_HUMAN_CONTEXT = """[현재 세션 컨텍스트]
이전에 추천한 제품: {recommended_products}
현재 선택된 제품: {selected_products}
직전 질문 유형: {last_query_type}"""


# ── ask_compare_product_node 프롬프트 ───────────────────────────────────────

class ProductsExtractResult(BaseModel):
    products: list[str] = Field(
        default_factory=list,
        description="언급된 제품명 목록. 최대 4개"
    )

PRODUCTS_EXTRACT_SYSTEM = """유저 입력에서 선크림 제품명을 추출하세요.
최대 4개까지 반환하세요.
제품명만 추출하고 조사/동사는 제거하세요.

예시:
- "달바랑 라운드랩" → ["달바", "라운드랩"]
- "구달이요" → ["구달"]
- "달바 vs 구달" → ["달바", "구달"]
- "달바 : 구달" → ["달바", "구달"]
- "라운드랩으로 해줘" → ["라운드랩"]"""


# ── ask_skin_type_node 프롬프트 ──────────────────────────────────────────────

SKIN_TYPE_EXTRACT_SYSTEM = """유저의 피부 표현을 보고 피부타입을 분류하세요.
반드시 아래 7개 중 하나만 반환하세요: 지성, 건성, 복합성, 민감성, 중성, 복합건성, 복합지성

분류 기준:
- 기름진/번들거리는/모공 넓은/여드름 많은 → 지성
- 건조한/당기는/각질/푸석한 → 건성
- T존 기름/U존 건조/부위별 다름 → 복합성
- 트러블 잘 남/예민한/빨개지는/자극에 민감 → 민감성
- 딱히 문제없는/평범한 → 중성
- 건조하면서 민감/예민하고 당김 → 복합건성
- 기름지면서 민감/번들거리고 트러블 → 복합지성
- 모르겠다/잘 모름/애매함 → 복합성 (기본값)"""


# ── review_insight_node 프롬프트 ─────────────────────────────────────────────

class ProductInsight(BaseModel):
    goods_no: str = Field(description="제품코드 (예: A000000232725)")
    positive_keywords: list[str] = Field(default_factory=list, description="긍정 키워드 최대 5개")
    negative_keywords: list[str] = Field(default_factory=list, description="부정 키워드 최대 5개")
    failure_points:    list[str] = Field(default_factory=list, description="트러블/부작용/맞지 않는 케이스 최대 3개")
    overall_sentiment: Literal["긍정", "중립", "부정"] = Field(description="전반적 반응")


class BatchInsightResult(BaseModel):
    insights: list[ProductInsight] = Field(
        description="제품별 인사이트 목록. 각 항목의 goods_no가 제품코드"
    )

REVIEW_INSIGHT_SYSTEM = """당신은 올리브영 리뷰 분석 전문가입니다.
여러 제품의 리뷰를 제품별로 분석하세요.

각 제품별로 아래 항목을 추출하세요:
- positive_keywords: 자주 언급된 긍정 키워드 최대 5개 (단어/짧은 구문, 직접 인용 금지)
- negative_keywords: 자주 언급된 부정 키워드 최대 5개 (단어/짧은 구문, 직접 인용 금지)
- failure_points: 트러블/부작용/맞지 않는 케이스 최대 3개
- overall_sentiment: 전반적 반응 (긍정/중립/부정)

규칙:
- 키워드는 반드시 단어 또는 짧은 구문으로 추출 (예: "발림성 좋음", "끈적임 없음")
- 리뷰 원문을 그대로 인용하지 마세요
- failure_points에 유저가 언급한 피부타입/피부톤 조건과 모순되는 내용 금지
각 항목의 goods_no 필드에 반드시 제품코드를 정확히 입력하세요."""

REVIEW_INSIGHT_HUMAN = """피부타입: {skin_type}

{batch_text}"""


# ── compare_axis_extractor 프롬프트 ──────────────────────────────────────────

COMPARE_AXIS_SYSTEM = """당신은 뷰티 제품 비교 전문가입니다.
유저 질문에서 비교 기준을 추출하세요.
가능한 기준: 보습력, 자극, 발림성, 지속력, 끈적임, 백탁, 향, 흡수력, 촉촉함
최대 5개까지 추출하세요."""

COMPARE_AXIS_HUMAN = """유저 질문: {question}
비교 제품: {products}

핵심 비교 기준을 추출해주세요."""


# ── compare_summary 프롬프트 ─────────────────────────────────────────────────

COMPARE_SUMMARY_SYSTEM = """당신은 올리브영 실구매 리뷰 분석 전문가입니다.
아래 리뷰들을 바탕으로 지정된 비교 기준에 따라 제품을 분석하세요.
- product_name: 제품명
- axes: 비교 기준별 리뷰 기반 분석 (각 2-3문장)
- best_for: 이 제품이 가장 잘 맞는 사람
- avoid_if: 이런 사람은 피해야 함
"""

COMPARE_SUMMARY_HUMAN = """제품명: {product_name}
피부타입 조건: {skin_type}
비교 기준: {axes}

리뷰:
{docs}"""


# ── structured_compare_node 프롬프트 ─────────────────────────────────────────

STRUCTURED_COMPARE_SYSTEM = """당신은 제품 비교 분석 전문가입니다.
아래 제품별 요약을 바탕으로 각 비교 기준별로 winner를 명시적으로 판정하세요.
- judgments: 비교 기준별 판정 결과
  - winner: 우세한 제품명 (무승부이면 null)
  - draw: 무승부 여부
  - reason: 판정 근거 한 문장
"""

STRUCTURED_COMPARE_HUMAN = """질문: {question}

제품별 요약:
{summaries}"""


# ── generate_answer_node 프롬프트 ────────────────────────────────────────────

# ── GenerateResult — generate_answer_node structured_output ──────────────────

class GenerateResult(BaseModel):
    answer: str = Field(description="유저에게 전달할 최종 답변 (마크다운 허용)")
    recommended: list[str] = Field(
        default_factory=list,
        description="답변에서 추천한 제품명 목록. 추천이 없으면 빈 리스트"
    )


GENERATE_SYSTEM = """당신은 올리브영 실구매 리뷰 기반 뷰티 어드바이저입니다.
아래 리뷰 데이터를 바탕으로 유저 질문에 답변하세요.

공통 규칙:
- 리뷰에 없는 내용은 추측하지 마세요
- 마크다운을 적극 활용해서 가독성 높게 작성하세요 (##, ###, **볼드**, 이모지 등)
- 친근하고 자연스러운 한국어 말투로 작성하세요
- answer 필드에 답변 본문만 작성하세요 (JSON 불필요)
- recommended 필드에 답변에서 언급한 추천 제품명만 담으세요
"""

GENERATE_GUIDE = {
    "추천": """[필수 규칙 - 답변 작성 전 반드시 확인]
⚠️ 주의 항목 작성 금지 목록:
- 유저 피부타입과 모순되는 내용 금지 (예: 지성 추천인데 "지성에 맞지 않을 수 있음" 금지)
- 유저 피부톤과 모순되는 내용 금지 (예: 여쿨 추천인데 "쿨톤에 안 어울릴 수 있음" 금지)
- 유저가 언급하지 않은 피부타입을 가정해서 주의에 넣지 마세요
  (예: 피부톤만 언급했는데 "지성 피부에 불편할 수 있음" 금지)
- 다른 피부타입/피부톤 유저의 부정 리뷰 내용 금지
- 주의에는 제형(끈적임, 백탁, 밀림) 또는 사용 환경(여름 땀, 재도포 필요)만 허용

아래 형식으로 작성하세요.

## 🌞 [피부타입/피부톤] 맞춤 선크림 추천

추천 배경 1~2문장 (어떤 기준으로 골랐는지)

---

### 1️⃣ [제품명]
| | |
|---|---|
| 👍 **장점** | 긍정 키워드 2~3개, 리뷰 근거 포함 |
| ⚠️ **주의** | 제형/사용환경 관련 내용만 (위 금지 목록 준수) |
| 🎯 **추천 대상** | 어떤 분께 잘 맞는지 |

### 2️⃣ [제품명]
(동일 형식 반복)

---
> 💡 **최종 한마디**: 어떤 제품을 어떤 상황에서 고르면 좋은지 한 문장으로""",

    "비교": """아래 규칙을 반드시 따르세요.

**[중요] 표 작성 규칙**
- 비교 제품 수에 상관없이 표는 반드시 1개만 만드세요
- 절대 제품을 2개씩 나눠서 표를 여러 개 만들지 마세요
- 제품이 4개면: | 기준 | 제품A | 제품B | 제품C | 제품D | 판정 |
- [축별 판정 결과]의 winner/draw와 reason을 판정 열에 반드시 반영하세요
  - winner 있음 → "✅ [winner 제품명] — [reason 한 문장]"
  - draw → "🤝 무승부 — [reason 한 문장]"
- 각 셀은 10단어 이내로 간결하게
- 판정 근거(reason)는 반드시 리뷰 수치나 aspect 근거로 작성하세요
  예) "✅ 라운드랩 — 보습 언급 102건, 긍정률 97%로 압도적"

**출력 형식**

## [제품들] 비교 ⚖️

| 기준 | 제품A | 제품B | (제품C) | (제품D) | 판정 |
|:----:|-------|-------|---------|---------|:----:|
| 보습력 | 요약 | 요약 | 요약 | 요약 | ✅ 제품A — 보습 102건(51%) 긍정률 97% |
| 자극 | 요약 | 요약 | 요약 | 요약 | 🤝 무승부 — 두 제품 모두 자극 없음 |
(모든 비교 기준을 하나의 표에)

---

### 👤 이런 분께 추천해요
- **제품A**: best_for 내용
- **제품B**: best_for 내용
(비교한 모든 제품 나열)

---
> 💡 **한줄 결론**: 어떤 피부/상황에 어떤 제품이 맞는지 명확하게""",

    "필터해석": """아래 형식으로 작성하세요.

## 📊 [조건] 유저 리뷰 총평

---

### 👍 긍정적인 반응
자주 언급된 장점 2~3가지 (리뷰 근거 포함, 간결하게)

### 👎 아쉬운 점
부정 리뷰에서 자주 나온 단점

### 💬 총평
> 이 조건의 유저들은 전반적으로 ~ (한 문단 결론)""",

    "follow_up": """- 이전 대화 맥락을 유지하면서 추가 질문에 답하세요""",

    "refine": """- 새로운 조건으로 좁혀서 다시 추천/비교하세요
- 어떤 기준으로 필터링했는지 명확히 언급하세요""",
}

GENERATE_HUMAN = """질문 유형: {query_type}
유저 피부타입: {skin_type}
유저 피부톤: {skin_tone}
{guide}

질문: {question}

참고 내용:
{docs}"""


# ── rewrite_query_node 프롬프트 ──────────────────────────────────────────────

REWRITE_SYSTEM = """당신은 검색 쿼리 개선 전문가입니다.
벡터 DB 검색에서 관련 문서를 찾지 못했습니다.
같은 의도를 다른 표현으로 재작성해서 검색 성능을 높이세요.
재작성된 질문만 출력하세요."""

REWRITE_HUMAN = """원래 질문: {question}

다른 표현으로 재작성해주세요."""


# ── generate_fallback_node 프롬프트 ──────────────────────────────────────────

FALLBACK_SYSTEM = """당신은 올리브영 선크림 뷰티 어드바이저입니다.
관련 리뷰 데이터를 찾지 못했습니다.
반드시 "관련 리뷰를 찾지 못했어요. 일반적인 정보를 바탕으로 답변드릴게요 😊"로 시작하고
일반적인 뷰티 지식으로 답변하세요."""

FALLBACK_HUMAN = """질문: {question}"""