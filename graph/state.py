from typing import Literal, NotRequired
from langgraph.graph import MessagesState
from langchain_core.documents import Document


class GraphState(MessagesState):
    # 항상 존재 (기본값 보장)
    rewrite_count: int = 0
    needs_skin_type: bool = False

    # 없을 수도 있는 필드
    query_type: NotRequired[Literal["추천", "비교", "필터해석", "follow_up", "refine"] | None]
    skin_type: NotRequired[str]
    skin_tone: NotRequired[str]
    products_mentioned: NotRequired[list[str]] # intake_node에서 추출 -> 유저가 언급한 제품명 목록 
    retrieved_docs: NotRequired[list[Document]] # 검색된 리뷰들을 담는 리스트
    recommended_products: NotRequired[list[str]] # generate_answer_node에서 추출 -> LLM 답변 마지막에 붙는 JSON에서 파싱 
    product_summaries: NotRequired[list[str]] # 비교 서브그래프 제품별 LLM 요약 결과
    selected_products: NotRequired[list[str]]
    last_query_type: NotRequired[str]
    review_insights: NotRequired[dict] # review_insight_node에서 추출 -> retrieved_docs 전체를 LLM이 분석한 결과
    structured_compare_result: NotRequired[dict] # 비교 서브그래프 축별 winner/draw 결과
    compare_axes: NotRequired[list[str]] # 비교 서브 그래프 LLM이 질문에서 추출한 비교 기준 목록
    user_needs: NotRequired[list[str]]  # UserNeeds.primary → aspects 목록




 # 검색된 리뷰들을 담는 리스트
'''
    ChromaDB에서 검색 결과를 반환할 때 이 형태로 돌려줌

    Document(
        page_content="올영에서 바로 구할수 있어서 좋아요...",  # 리뷰 본문
        metadata={
            "goods_name": "비오레 UV ...",
            "score": 4,
            "skin_type": "A03",
            ...
        }
    )

 '''


'''
products_mentioned   : 예) "달바 vs 라운드랩 비교해줘" -> ["달바", " 라운드랩"]
recommended_products : 예) {"recommended": ["달바", "구달"]} → ["달바", "구달"]
                          follow_up/refine에서 "이전에 추천한 제품"으로 활용
product_summaries    : 예) ["달바: 발림성 좋고 산뜻함, 지성 추천","라운드랩: 보습 강함, 건성 추천"]
review_insights      : 예) { "달바":
                                    {
                                    "positive_keywords": ["산뜻함", "가벼움"],
                                    "negative_keywords": ["끈적임", "건조함"],
                                    "failure_points": ["민감성 트러블"],
                                    "overall_sentiment": "긍정"} ,
                            "구달": {"positive_keywords": [...], ...},
                                                                         }

                            
structured_compare_result : 예) {
                                "발림성": {"winner": "달바", "draw": False},
                                "보습력": {"winner": "라운드랩", "draw": False},
                                "자극도": {"winner": None, "draw": True}
                                }


compare_axes          :  예 )  "달바 vs 라운드랩 지성 기준 비교" → ["발림성", "끈적임", "자극도"]


'''