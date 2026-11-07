import streamlit as st
from openai import OpenAI
from pathlib import Path
from datetime import date


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="너만의 다전공 챗봇",
    page_icon="🎓",
    layout="centered"
)

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url=st.secrets["BASE_URL"]
)

MODEL_NAME = st.secrets["MODEL_NAME"]

BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"


# --------------------------------------------------
# 1-1. 지식 자료 불러오기 (knowledge 폴더)
# --------------------------------------------------

def read_text_file(path):
    """utf-8로 먼저 읽고, 실패하면 cp949(윈도우 한글)로 다시 읽는다."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return ""


@st.cache_data
def load_knowledge():
    parts = []
    if not KNOWLEDGE_DIR.exists():
        return ""
    for f in sorted(KNOWLEDGE_DIR.glob("*")):
        if f.suffix.lower() in {".md", ".txt", ".csv"}:
            text = read_text_file(f)
            if text.strip():
                parts.append(f"## [자료: {f.name}]\n{text}")
    return "\n\n".join(parts)


KNOWLEDGE = load_knowledge()


# --------------------------------------------------
# 2. AI 상담 지침
# --------------------------------------------------

BASE_PROMPT = """
당신은 경희대학교 학생을 위한 다전공 전문 상담 AI입니다.

학생의 학업 상황, 관심사, 성격, 강점, 진로 및 다전공 이수 목적을
종합적으로 분석하여 적합한 다전공을 추천하는 것이 역할입니다.

[상담 원칙]

1. 반드시 학생이 제공한 정보를 바탕으로 상담합니다.
2. 학생의 답변을 무시하거나 이미 답한 질문을 반복하지 않습니다.
3. 한 번에 질문을 여러 개 하지 않고 하나씩 질문합니다.
4. 학생의 답변에 따라 다음 질문의 내용과 방향을 조정합니다.
5. 학생의 선택을 존중하며 무조건 특정 전공을 강요하지 않습니다.
6. 학생이 이해하기 쉬운 자연스러운 존댓말을 사용합니다.
7. 근거가 부족한 정보는 추측하지 않습니다.
8. 아래 [참고 자료]에 있는 학과·규정·기준만 사실로 안내합니다.
   자료에 없는 내용은 추측하지 말고
   "해당 내용은 확인되지 않으니 단과대학 행정실에 문의해 주세요"라고 안내합니다.
9. 신청 자격, 이수 학점, 졸업 요건 등을 사실처럼 만들어내지 않습니다.

[참고 자료 활용 규칙]

1. 학과를 추천할 때는 학생의 관심사·진로를 [학과별 진로·핵심 직업역량]
   자료와 연결해서 이유를 설명합니다.
2. 추천한 학과마다 가능하면 다음을 함께 안내합니다.
   - 승인(선발) 기준: 성적 반영 비율, 학업계획서·면접 여부, 지원자격(최소 학점, 선이수 과목 등)
   - 해당 학과가 학생의 소속 캠퍼스 기준으로 다전공 이수금지 전공이 아닌지
3. 학생의 소속 캠퍼스(서울/국제)는 주전공으로 판단하되,
   불분명하면 후속 질문으로 하나만 확인합니다.
   서울캠퍼스와 국제캠퍼스는 승인 기준이 다르므로 캠퍼스를 섞어 안내하지 않습니다.
4. 학생의 학점이 지원자격(최소 평점)에 미달하면 그 사실을 솔직하게 알려주고,
   대안(다른 학과, 부전공, 마이크로디그리 등)을 함께 제시합니다.
5. 학과명이 자료마다 조금 다르게 적혀 있을 수 있습니다.
   같은 학과로 판단되더라도 확실하지 않으면 확인이 필요하다고 밝힙니다.
6. 자료끼리 내용이 충돌하면 [운영지침] > 학기별 공지·안내 순서로 따릅니다.
7. 졸업요건 변경(2026학년도 신·편입생부터 다전공/부전공/마이크로디그리 중
   택1 필수)과 세 가지 과정의 차이는 학생이 물어보거나
   다전공이 아닌 대안이 필요해 보일 때 안내합니다.
8. 신청 기간은 아래 '오늘 날짜' 기준으로 판단합니다.
   공지의 신청 기간이 이미 지났다면 종료되었다고 분명히 말하고,
   다음 신청 시기(1학기 4월 / 2학기 10월, 학사일정에 따라 변동 가능)를 안내합니다.
9. 승인 기준과 신청 일정은 학기마다 바뀔 수 있으므로,
   최종 확인은 해당 단과대학 공지와 행정실에서 해야 한다고 짧게 덧붙입니다.

[1차 추천 결과 작성 방식]

처음 네 가지 질문에 대한 답변을 모두 받은 뒤에는 다음 순서로
추천 결과를 작성합니다.

1. 학생 성향 및 학업 상황 분석
2. 추천 다전공 TOP 3
3. 각 전공을 추천한 구체적인 이유
4. 주전공과의 학문적 연관성
5. 각 전공의 장점과 고려할 점 (승인 기준과 지원자격 포함)
6. 관련 진로 및 활용 가능성
7. 학생에게 가장 적합하다고 판단하는 선택지

추천 전공은 [참고 자료]에서 확인되는 학과에 한해 확정적으로 표현합니다.
자료에서 확인할 수 없다면 전공 분야 또는 후보로 표현합니다.

[점진적 추가 상담]

1차 추천 이후에는 상담을 종료하지 않습니다.

추천 결과를 바탕으로 학생에게 추가 질문을 하나씩 제시합니다.

질문 예시:
- 코딩이나 통계 과목이 포함되어도 괜찮으신가요?
- 수학적 계산이 많은 전공도 고려할 수 있으신가요?
- 개인의 흥미와 취업 가능성 중 무엇을 더 중요하게 생각하시나요?
- 주전공과 연관성이 높은 전공과 새로운 분야 중 어느 쪽을 선호하시나요?

단, 위 질문을 그대로 반복하지 말고 학생의 답변과 추천 결과에
맞춰 적절한 질문을 선택합니다.

추가 답변을 받으면 기존 추천 결과를 다시 검토하고,
필요하면 순위와 추천 이유를 수정합니다.

학생이 충분히 정보를 제공했다면 계속 질문만 하지 말고
최종 추천 결과를 정리합니다.

항상 학생이 편안하게 대화할 수 있도록 친절하게 상담합니다.
"""


def build_system_prompt():
    # 날짜는 대화할 때마다 새로 계산해서 넣는다.
    return (
        BASE_PROMPT
        + f"\n[오늘 날짜]\n{date.today().strftime('%Y-%m-%d')}\n"
        + "\n[참고 자료]\n"
        + (KNOWLEDGE if KNOWLEDGE else "(knowledge 폴더에서 불러온 자료가 없습니다.)")
    )


# --------------------------------------------------
# 3. 4단계 초기 질문
# --------------------------------------------------

QUESTIONS = [
    "현재 주전공과 학년, 그리고 학점 알려주세요.",
    "현재 진로와 관심사를 알려주세요.",
    "자신의 장단점 혹은 좋아하는 것, 싫어하는 것을 알려주세요.",
    "다전공 이수 목적을 알려주세요."
]


# --------------------------------------------------
# 4. 세션 상태 초기화
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if "fade_transition" not in st.session_state:
    st.session_state.fade_transition = False

if "step" not in st.session_state:
    st.session_state.step = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "first_recommendation_done" not in st.session_state:
    st.session_state.first_recommendation_done = False

if "loading" not in st.session_state:
    st.session_state.loading = False



# --------------------------------------------------
# 5. 화면 디자인
# --------------------------------------------------

st.markdown("""
<style>
/* 전체 화면: 어두운 배경 */
.stApp {
    background: #080D18 !important;
    color: #FFFFFF !important;
}

/* 페이지 너비와 여백 */
.block-container {
    max-width: 900px;
    padding-top: 2.5rem;
    padding-bottom: 6rem;
}

/* 상단 메뉴 */
header[data-testid="stHeader"] {
    background: #080D18 !important;
}

/* 기본 텍스트는 흰색 */
.stApp,
.stApp p,
.stApp label,
.stApp span,
.stApp li,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp [data-testid="stCaptionContainer"] {
    color: #FFFFFF !important;
}

/* 시작 화면 컨테이너 */
.home-container {
    background: linear-gradient(145deg, #101B32, #1C3154);
    padding: 95px 35px;
    border-radius: 24px;
    text-align: center;
    margin-top: 55px;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.35);
    border: 1px solid #2D4265;
}

/* 버튼 */
.stButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 12px;
    min-height: 48px;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #1D4ED8 !important;
    color: #FFFFFF !important;
    border-color: #60A5FA !important;
}

/* 채팅 제목 */
.chat-header {
    background: #101B32;
    color: #FFFFFF !important;
    padding: 18px 24px;
    border: 1px solid #263B5B;
    border-radius: 14px;
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 25px;
}

/* --------------------------------------------------
   안정적인 카카오톡 스타일 메신저 말풍선 레이아웃
-------------------------------------------------- */

/* 채팅 메시지 전체 행 */
div[data-testid="stChatMessage"] {
    display: flex !important;
    width: 100% !important;
    margin-bottom: 16px !important;
}

/* 1. 챗봇 메시지 (왼쪽 정렬) */
div[data-testid="stChatMessage"]:nth-child(odd) {
    flex-direction: row !important;
    justify-content: flex-start !important;
}

/* 챗봇 말풍선 디자인 */
div[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stMarkdownContainer"] {
    background-color: #14233D !important;
    border: 1px solid #2B4267 !important;
    color: #FFFFFF !important;
    padding: 12px 16px !important;
    border-radius: 16px !important;
    border-top-left-radius: 4px !important;
    max-width: 75% !important;
    word-break: break-word !important;
}

/* 2. 사용자 메시지 (오른쪽 정렬) */
div[data-testid="stChatMessage"]:nth-child(even) {
    flex-direction: row-reverse !important;
    justify-content: flex-start !important;
}

/* 사용자 말풍선 디자인 (카카오톡 노란색) */
div[data-testid="stChatMessage"]:nth-child(even) [data-testid="stMarkdownContainer"] {
    background-color: #FEE500 !important;
    border: 1px solid #E5CE00 !important;
    color: #191919 !important;
    padding: 12px 16px !important;
    border-radius: 16px !important;
    border-top-right-radius: 4px !important;
    max-width: 75% !important;
    word-break: break-word !important;
}

/* 사용자 말풍선 안의 텍스트 색상 강제 고정 (어두운 색) */
div[data-testid="stChatMessage"]:nth-child(even) [data-testid="stMarkdownContainer"] p,
div[data-testid="stChatMessage"]:nth-child(even) [data-testid="stMarkdownContainer"] span,
div[data-testid="stChatMessage"]:nth-child(even) [data-testid="stMarkdownContainer"] strong {
    color: #191919 !important;
}

/* 챗봇 말풍선 안의 텍스트 색상 (흰색) */
div[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stMarkdownContainer"] p,
div[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stMarkdownContainer"] span,
div[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stMarkdownContainer"] strong {
    color: #FFFFFF !important;
}

/* 채팅 입력창 */
div[data-testid="stChatInput"] {
    background: #101B32 !important;
    border: 1px solid #52698F !important;
    border-radius: 15px !important;
}

div[data-testid="stChatInput"] textarea {
    background-color: #101B32 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #B8C6DC !important;
    -webkit-text-fill-color: #B8C6DC !important;
}

/* 진행률바 */
div[data-testid="stProgressBar"] > div {
    background-color: #26344B !important;
}
div[data-testid="stProgressBar"] > div > div {
    background-color: #3B82F6 !important;
}

hr {
    border-color: #26344B !important;
}
</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# 6. AI 응답 생성 함수
# --------------------------------------------------

def get_ai_response(messages):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1800
    )

    answer = response.choices[0].message.content

    if not answer:
        return "답변을 생성하지 못했습니다. 다시 한번 말씀해 주세요."

    return answer




# --------------------------------------------------
# 7. 시작 화면
# --------------------------------------------------

if st.session_state.page == "home":

    st.markdown("""
    <div style="
        background: linear-gradient(145deg, #101B32, #1C3154);
        padding: 85px 30px;
        border-radius: 24px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(16,27,50,0.18);
        margin-top: 45px;
    ">
        <p style="
            color: #AFC3E3;
            font-size: 13px;
            letter-spacing: 3px;
            font-weight: bold;
        ">
            KYUNG HEE UNIVERSITY
        </p>
        <h1 style="
            color: white;
            font-size: 43px;
            font-weight: 800;
            margin: 25px 0;
        ">
            너만의 다전공 챗봇
        </h1>
        <p style="
            color: #D7E1F0;
            font-size: 18px;
            line-height: 1.9;
        ">
            나의 관심사와 진로에 맞는 다전공 방향을 찾아보고,<br>
            AI와 함께 나에게 맞는 선택을 시작해 보세요.
        </p>
        <p style="
            color: #AFC3E3;
            font-size: 12px;
            margin-top: 30px;
            letter-spacing: 2px;
        ">
            PERSONALIZED ANALYSIS · CAREER · FUTURE
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    _, center, _ = st.columns([1, 1, 1])

    with center:
        if st.button("상담 시작하기 →", use_container_width=True):

            st.session_state.page = "chat"
            st.session_state.fade_transition = True
            st.session_state.step = 0
            st.session_state.messages = []
            st.session_state.answers = []
            st.session_state.first_recommendation_done = False

            st.session_state.messages.append({
                "role": "assistant",
                "content": QUESTIONS[0]
            })

            st.rerun()



# --------------------------------------------------
# 8. 채팅 화면
# --------------------------------------------------

elif st.session_state.page == "chat":

    # 시작 화면에서 채팅 화면으로 넘어올 때 한 번만 Fade 효과 적용
    if st.session_state.fade_transition:
        st.markdown("""
        <style>
        @keyframes chatFadeIn {
            from {
                opacity: 0;
                transform: translateY(12px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .block-container {
            animation: chatFadeIn 0.7s ease-out both;
        }
        </style>
        """, unsafe_allow_html=True)
        st.session_state.fade_transition = False

    st.markdown(
        '<div class="chat-header">🎓　너만의 다전공 챗봇</div>',
        unsafe_allow_html=True
    )

    st.caption("AI가 당신의 관심사와 진로에 맞춰 다전공을 함께 탐색합니다.")

    # 진행 상황 표시
    if st.session_state.step < 4:
        st.progress(
            st.session_state.step / 4,
            text=f"학생 정보 수집 ({st.session_state.step}/4)"
        )
    else:
        st.caption("학생 정보 수집 완료 · 맞춤형 추가 상담 진행 중")

    # 채팅 기록 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    user_input = st.chat_input("메시지를 입력하세요...")

    if user_input:
        # 1) 사용자 메시지를 먼저 저장하고 즉시 화면에 표시
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # 2) 초기 네 가지 질문 진행
        if st.session_state.step < 4:
            st.session_state.answers.append(user_input)
            st.session_state.step += 1

            if st.session_state.step < 4:
                next_question = QUESTIONS[st.session_state.step]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": next_question
                })

            else:
                profile = f"""
다음은 다전공 상담을 신청한 학생의 정보입니다.

1. 주전공, 학년, 학점:
{st.session_state.answers[0]}

2. 진로 및 관심사:
{st.session_state.answers[1]}

3. 장단점 및 선호:
{st.session_state.answers[2]}

4. 다전공 이수 목적:
{st.session_state.answers[3]}

이 정보를 바탕으로 [참고 자료]에서 확인되는 학과 중
적합한 다전공 후보 3개를 추천하고,
각 추천 이유와 장단점, 주전공과의 연관성 및 관련 진로를 설명해 주세요.
각 후보의 승인(선발) 기준과 지원자격, 학생의 소속 캠퍼스 기준 이수금지 여부도
자료에 근거해 함께 안내해 주세요.
자료에서 확인할 수 없는 내용은 확정된 사실처럼 말하지 말고 확인이 필요하다고 밝혀 주세요.
마지막에는 학생의 성향을 더 파악할 수 있는 후속 질문을 하나만 제시해 주세요.
"""

                ai_messages = [
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": profile}
                ]

                try:
                    with st.spinner("학생의 정보를 분석하고 있습니다..."):
                        answer = get_ai_response(ai_messages)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                    st.session_state.first_recommendation_done = True

                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"AI 응답 오류: {e}"
                    })

        # 3) 네 가지 질문 이후의 추가 상담
        else:
            ai_messages = [
                {"role": "system", "content": build_system_prompt()}
            ]
            ai_messages.extend(st.session_state.messages)

            try:
                with st.spinner("답변을 분석하고 있습니다..."):
                    answer = get_ai_response(ai_messages)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"AI 응답 오류: {e}"
                })

        # 새 메시지를 포함해 화면 다시 그리기
        st.rerun()