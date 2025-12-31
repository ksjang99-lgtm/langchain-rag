import os
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# 1. 환경 설정 및 API 키 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# [지능형 프롬프트 관리 변수]
# ==========================================
SYSTEM_PROMPT = """너는 보안 장비 'CDR-10028'의 기술 지원 전문가야. 
사용자의 질문 의도에 따라 아래 두 가지 답변 모드 중 하나를 선택해.

1. [텍스트 모드]
   - 대상: 인사, 단순 제원 확인(무게, 온도, 크기), 개념 질문, 모델명 확인 등.
   - 특징: 간결하고 명확한 텍스트로만 답변해. '이미지 가이드' 섹션을 포함하지 마.

2. [멀티모달 모드]
   - 대상: 설치 방법, 배선 연결, 지문 등록/입력 가이드, 부품 위치 찾기, 장애 해결 등.
   - 특징: 단계별 가이드를 제공하고, 답변 마지막에 '이미지 가이드: [파일명]' 형식을 반드시 포함해.

[주의사항]
- 반드시 제공된 [매뉴얼 정보]를 근거로 답변해.
- 매뉴얼에 없는 내용은 추측하지 말고 모른다고 정직하게 말해.
"""

# ==========================================
# [RAG 시스템 클래스]
# ==========================================
class CDR10028Agent:
    def __init__(self):
        # ChromaDB 연결 (build_db.py를 통해 구축된 DB 경로)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # 임베딩 모델 설정 (DB 구축 시와 동일한 모델 사용)
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-embedding-3-small"
        )
        
        # 컬렉션 로드
        self.collection = self.chroma_client.get_collection(
            name="cdr10028_manual", 
            embedding_function=self.openai_ef
        )

    def search_knowledge(self, query):
        """질문과 관련된 지식 및 이미지 경로 검색"""
        results = self.collection.query(query_texts=[query], n_results=2)
        
        # 검색된 텍스트와 이미지 경로 추출
        context = "\n".join(results['documents'][0])
        image_paths = [m['image_path'] for m in results['metadatas'][0]]
        
        return context, list(set(image_paths))

    def answer(self, user_query):
        # 1. DB에서 관련 정보 검색
        context, image_paths = self.search_knowledge(user_query)
        
        # 2. GPT-4o-mini에게 판단 및 답변 요청
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"질문: {user_query}\n\n[매뉴얼 정보]\n{context}"}
            ],
            temperature=0.3 # 정확도를 위해 낮은 온도로 설정
        )
        
        full_answer = response.choices[0].message.content
        return full_answer, image_paths

# ==========================================
# [실행부]
# ==========================================
if __name__ == "__main__":
    agent = CDR10028Agent()
    print("--------------------------------------------------")
    print("CDR-10028 지능형 기술지원 센터입니다.")
    print("설치, 배선, 지문 등록 등 무엇이든 물어보세요. (종료: exit)")
    print("--------------------------------------------------")
    
    while True:
        user_input = input("\n사용자: ")
        if user_input.lower() in ['exit', 'quit', '종료']: 
            break
        
        # 답변 생성
        answer, images = agent.answer(user_input)
        
        # 결과 출력
        print(f"\n[AI]: {answer}")
        
        # AI가 멀티모달 모드를 선택하여 '이미지 가이드'를 언급한 경우에만 이미지 경로 출력
        if "이미지 가이드:" in answer:
            print("\n[관련 이미지 가이드 시스템]")
            for img in images:
                print(f"▶ {img}")