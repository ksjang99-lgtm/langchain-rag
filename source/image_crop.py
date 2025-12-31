import fitz  # PyMuPDF
import base64
import requests
import os
import json
import glob
from dotenv import load_dotenv

# ==========================================
# 1. 설정 및 프롬프트 관리 (관리 용이성)
# ==========================================
MODEL_NAME = "gpt-4o-mini"
DATASET_PATH = "./Data/"
OUTPUT_FILE = "integrated_knowledge_base.json"
IMAGE_DIR = "extracted_images"

# 프롬프트 변수 분리
SYSTEM_PROMPT = """너는 보안 장비 설치 및 기술 지원 전문가야. 
제공된 매뉴얼 페이지 이미지를 분석하여 사용자가 현장에서 즉시 참고할 수 있는 지식 베이스를 구축해야 해."""

USER_PROMPT_TEMPLATE = """이 매뉴얼 페이지 이미지를 분석해서 아래의 JSON 형식으로만 답해줘. 
특히 도면(결선도)이나 지문 등록 가이드 그림이 있다면 매우 상세하게 기술해줘.

{
  "text_summary": "이 페이지의 핵심 텍스트 내용 요약 (설치 단계, 주의사항 등)",
  "visual_elements": [
    {
      "name": "그림 번호 또는 표 제목 (예: 그림 2, 구성품 표)",
      "description": "그림이 담고 있는 기술적 상세 내용 (예: 적색선은 +단자 연결 등)",
      "keywords": ["검색용 키워드1", "키워드2", "키워드3"]
    }
  ]
}

그림이나 표가 전혀 없다면 visual_elements는 빈 리스트 []로 응답해."""

# ==========================================
# 2. 유틸리티 함수
# ==========================================

def get_api_key():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("환경변수 'OPENAI_API_KEY'가 설정되지 않았습니다.")
    return key

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_page_with_gpt(image_path, api_key):
    base64_image = encode_image(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT_TEMPLATE},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"  # 도면 분석을 위해 고해상도 모드 사용
                        }
                    }
                ]
            }
        ],
        "response_format": { "type": "json_object" },
        "max_tokens": 1500
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return json.loads(response.json()['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error during GPT analysis: {e}")
        return None

# ==========================================
# 3. 메인 로직 (일괄 처리)
# ==========================================

def run_data_preparation():
    api_key = get_api_key()
    pdf_files = glob.glob(os.path.join(DATASET_PATH, "*.pdf"))
    
    if not pdf_files:
        print(f"'{DATASET_PATH}' 폴더에 PDF 파일이 없습니다.")
        return

    full_knowledge_base = []

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"\n[작업 시작] 파일명: {file_name}")
        
        doc = fitz.open(pdf_path)
        file_entry = {"source_file": file_name, "data": []}

        for i in range(len(doc)):
            page_num = i + 1
            print(f"  > {page_num}/{len(doc)} 페이지 처리 중...")
            
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2배 해상도 추출
            
            # temp_img_path = f"temp_{file_name}_p{page_num}.png"
            temp_img_path = os.path.join(
            IMAGE_DIR, f"{file_name}_p{page_num}.png"
        )
            pix.save(temp_img_path)

            # GPT 분석
            analysis_result = analyze_page_with_gpt(temp_img_path, api_key)
            
            if analysis_result:
                analysis_result['page_num'] = page_num
                analysis_result['local_image_path'] = temp_img_path
                file_entry['data'].append(analysis_result)
            
            # (선택 사항) 분석 후 이미지를 보관할지 삭제할지 결정 가능
            # 여기서는 RAG 답변 시 보여주기 위해 유지하도록 설계됨

        full_knowledge_base.append(file_entry)
        doc.close()

    # 결과물 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(full_knowledge_base, f, ensure_ascii=False, indent=4)
    
    print(f"\n[완료] 전체 지식 베이스가 '{OUTPUT_FILE}'에 저장되었습니다.")

os.makedirs(IMAGE_DIR, exist_ok=True)
if __name__ == "__main__":
    run_data_preparation()