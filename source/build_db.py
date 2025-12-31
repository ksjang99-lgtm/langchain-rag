import json
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def build_vector_db(json_file):
    # 1. ChromaDB 로컬 저장소 설정
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 2. 임베딩 모델 설정
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small"
    )
    
    # 3. 컬렉션 생성 (기존 데이터가 있다면 삭제 후 새로 생성하거나 가져오기)
    collection = client.get_or_create_collection(
        name="cdr10028_manual",
        embedding_function=openai_ef
    )

    # 4. JSON 데이터 로드
    with open(json_file, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    documents = []
    metadatas = []
    ids = []
    
    idx = 0
    for file_entry in full_data:
        file_name = file_entry['source_file']
        for page in file_entry['data']:
            # 임베딩할 텍스트 결합
            content = f"파일: {file_name}\n내용: {page['text_summary']}\n"
            for visual in page.get('visual_elements', []):
                content += f"그림({visual['name']}): {visual['description']}\n"

            # 메타데이터 저장
            metadata = {
                "source": file_name,
                "page": page['page_num'],
                "image_path": page['local_image_path']
            }

            documents.append(content)
            metadatas.append(metadata)
            ids.append(f"id_{idx}")
            idx += 1

    # 5. DB에 데이터 추가
    print(f"{len(documents)}개의 지식 조각을 DB에 저장 중...")
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print("벡터 데이터베이스 구축이 완료되었습니다.")

if __name__ == "__main__":
    build_vector_db("integrated_knowledge_base.json")