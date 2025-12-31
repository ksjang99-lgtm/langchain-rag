import streamlit as st
import json
import os
from datetime import datetime
from chat_agent import openAIAgent # 실제 파일명으로 수정하세요

# 1. 페이지 설정 및 초기화
st.set_page_config(page_title="지능형 기술지원 센터", page_icon="🤖")

# 에이전트 및 대화 기록 세션 초기화
if 'agent' not in st.session_state:
    st.session_state.agent = openAIAgent()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # 전체 대화 누적용 리스트

# 2. UI 레이아웃
st.title("🤖 지능형 기술지원 센터")
st.caption("설치, 배선, 지문 등록 등 무엇이든 물어보세요.")
st.markdown("---")

# 3. 기존 대화 기록 표시 (누적 데이터)
# 화면이 리프레시될 때마다 저장된 모든 메시지를 순차적으로 출력합니다.
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        # 텍스트 답변 또는 질문 출력
        if chat["role"] == "user":
            st.write(chat["content"])
        else:
            # AI 답변은 구조화된 데이터(dict)이므로 예쁘게 출력
            res = chat["content"]
            st.subheader(f"[{res.get('type')}] {res.get('title')}")
            for step in res.get('answer_steps', []):
                st.write(f"🔹 {step}")
            
            # 이미지 출력
            if res.get('related_images'):
                cols = st.columns(len(res['related_images']))
                for idx, img_path in enumerate(res['related_images']):
                    if os.path.exists(img_path):
                        cols[idx].image(img_path, use_container_width=True)
        
        # 하단 시간 표시
        st.caption(f"🕒 {chat['timestamp']}")

# 4. 사용자 질문 입력 및 처리
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    # 현재 시간 생성
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # A. 사용자 메시지 저장 및 즉시 표시
    user_msg = {"role": "user", "content": user_input, "timestamp": now}
    st.session_state.chat_history.append(user_msg)
    
    with st.chat_message("user"):
        st.write(user_input)
        st.caption(f"🕒 {now}")

    # B. AI 답변 생성 및 저장
    with st.spinner("지식 베이스 검색 중..."):
        try:
            # AI로부터 JSON(dict) 결과 수신
            result = st.session_state.agent.answer(user_input)
            ai_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            ai_msg = {"role": "assistant", "content": result, "timestamp": ai_now}
            st.session_state.chat_history.append(ai_msg)
            
            # AI 답변 즉시 화면 표시
            with st.chat_message("assistant"):
                st.subheader(f"[{result.get('type')}] {result.get('title')}")
                for step in result.get('answer_steps', []):
                    st.write(f"🔹 {step}")
                
                if result.get('related_images'):
                    st.markdown("---")
                    cols = st.columns(len(result['related_images']))
                    for idx, img_path in enumerate(result['related_images']):
                        if os.path.exists(img_path):
                            cols[idx].image(img_path, use_container_width=True, caption=f"가이드 {idx+1}")
                
                st.caption(f"🕒 {ai_now}")
                
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

# 사이드바: 대화 초기화 기능
with st.sidebar:
    if st.button("대화 내용 초기화"):
        st.session_state.chat_history = []
        st.rerun()