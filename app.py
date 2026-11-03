import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="KHU 다전공 추천 AI",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 경희대학교 다전공 추천 AI")
st.write("나의 관심사와 진로에 맞는 다전공 방향을 찾아보고, AI와 자유롭게 상담해 보세요!")

# API 연결
try:
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        base_url=st.secrets["BASE_URL"]
    )
except Exception:
    st.error("API 키를 불러오지 못했습니다. secrets.toml 설정을 확인해 주세요.")
    st.stop()

# 전체 원문 데이터 파일 불러오기
@st.cache_data
def load_all_data():
    try:
        with open("data/rules_full.txt", "r", encoding="utf-8") as f:
            rules_text = f.read()
        with open("data/majors_full.txt", "r", encoding="utf-8") as f:
            majors_text = f.read()
        return rules_text, majors_text
    except Exception as e:
        return f"규정 데이터 로드 실패: {e}", f"학과 데이터 로드 실패: {e}"

rules_text, majors_text = load_all_data()

# 💡 대화 내역(메모리) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("1. 나의 정보 입력")

major = st.text_input("현재 주전공", placeholder="예: 자유전공학부")

year = st.selectbox(
    "현재 학년",
    ["1학년", "2학년", "3학년", "4학년"]
)

interests = st.multiselect(
    "관심 있는 분야",
    ["컴퓨터·AI", "경영·경제", "생명과학", "화학·물리",
     "건축·공학", "심리·사회", "미디어·콘텐츠",
     "디자인·예술", "국제·외국어"]
)

strengths = st.multiselect(
    "자신 있는 능력",
    ["논리적 사고", "수학·분석 능력", "창의력",
     "의사소통", "문제 해결", "글쓰기", "외국어"]
)

career = st.text_input(
    "희망 진로",
    placeholder="예: AI 개발자, 마케팅 전문가, 연구원"
)

reason = st.selectbox(
    "다전공을 고민하는 이유",
    ["진로 선택의 폭을 넓히고 싶어서",
     "관심 분야를 더 깊게 공부하고 싶어서",
     "취업 경쟁력을 높이고 싶어서",
     "주전공과 다른 분야를 융합하고 싶어서",
     "아직 잘 모르겠음"]
)

extra = st.text_area(
    "추가로 알려주고 싶은 내용",
    placeholder="학업 부담, 좋아하는 과목, 피하고 싶은 분야 등을 자유롭게 작성해 주세요."
)

st.divider()

# 추천 받기 버튼
if st.button("나에게 맞는 다전공 추천받기", type="primary"):
    if not major or not interests:
        st.warning("주전공과 관심 분야를 입력해 주세요.")
    else:
        profile = f"""
        주전공: {major}
        학년: {year}
        관심 분야: {", ".join(interests)}
        강점: {", ".join(strengths) if strengths else "미입력"}
        희망 진로: {career if career else "미정"}
        다전공 선택 이유: {reason}
        추가 정보: {extra if extra else "없음"}
        """

        # 시스템 지침 설정
        instructions = f"""
        당신은 경희대학교 학생을 위한 다전공 진로 상담 전문 AI입니다.
        아래 제공되는 [경희대학교 공식 다전공 운영지침 및 단과대학별 승인 기준 전체 원문]과 [학과별 진로·역량 가이드 전체 원문]을 철저히 분석하고 바탕으로 삼아 학생에게 가장 정확하고 현실적인 다전공 방향을 제안하세요.

        ---
        [경희대학교 공식 다전공 운영지침 및 승인 기준 원문]
        {rules_text}
        ---
        [경희대학교 학과별 진로 및 핵심 역량 가이드 원문]
        {majors_text}
        ---

        반드시 다음 원칙을 지키세요.
        1. 학생의 정보를 바탕으로 추천 이유를 구체적으로 설명하세요. (제공된 학과별 핵심 역량과 연계할 것)
        2. 적합한 전공 방향을 2~3개 제안하고 우선순위를 정하세요.
        3. 제공된 운영지침 내 [별표1] 이수 금지 전공(소속 캠퍼스에 따른 제한 등)이나 신청 자격(2학기 이상 등)을 위배하는 추천은 절대 하지 마세요.
        4. 각 선택지의 장점, 단점, 어울리는 진로, 주전공과의 학문적 연관성을 설명하세요.
        5. 단과대학별 승인 기준(성적, 학업계획서, 필수 선수과목 등)을 상세히 안내하고 준비 팁을 제공하세요.
        6. 친절하고 자연스러운 존댓말을 사용하세요.
        """

        # 초기 대화 세션 구성
        st.session_state.messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": f"다음 학생 프로필을 바탕으로 다전공을 추천해 주세요:\n{profile}"}
        ]

        with st.spinner("AI가 경희대학교 학사 규정 및 전공 데이터 전체를 분석하고 있습니다..."):
            try:
                response = client.chat.completions.create(
                    model=st.secrets["MODEL_NAME"],
                    messages=st.session_state.messages,
                    max_tokens=3500
                )
                answer = response.choices[0].message.content
                if answer:
                    # AI의 답변을 대화 기억(session_state)에 추가
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.warning("AI가 빈 응답을 반환했습니다.")
            except Exception as e:
                st.error(f"AI 응답 중 오류가 발생했습니다: {e}")

# 💡 추천 결과가 이미 존재한다면 화면에 대화 내역을 출력하고 추가 채팅창 제공
if len(st.session_state.messages) > 0:
    st.subheader("2. AI 다전공 추천 및 추가 상담")
    
    # 시스템 메시지를 제외한 실제 대화 내역만 화면에 출력
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
        elif message["role"] == "user" and not message["content"].startswith("다음 학생 프로필"):
            with st.chat_message("user"):
                st.markdown(message["content"])

    # 💡 사용자가 추가로 입력할 수 있는 하단 채팅 입력창
    if user_query := st.chat_input("다전공에 대해 추가로 궁금한 점을 물어보세요 (예: 학업계획서는 어떻게 써야 할까?, 복수전공 수강 신청 팁 등)"):
        # 사용자 질문 추가
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 준비하고 있습니다..."):
                try:
                    response = client.chat.completions.create(
                        model=st.secrets["MODEL_NAME"],
                        messages=st.session_state.messages,
                        max_tokens=2500
                    )
                    answer = response.choices[0].message.content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")