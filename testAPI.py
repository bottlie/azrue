from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# --- 추가된 라이브러리 ---
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
# -------------------------
import os
import uuid
from dotenv import load_dotenv

# --- Google Gemini 라이브러리를 사용하도록 변경 ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.getenv("GOOGLE_API_KEY")

app = FastAPI()

# --- 🔽 [추가된 코드 시작] 🔽 ---
# 사용자가 웹사이트의 루트 주소('/')로 접속했을 때
# 'static' 폴더에 있는 'index.html' 파일을 응답으로 보내줍니다.
# 이렇게 하면 프론트엔드와 백엔드가 하나의 주소에서 함께 동작할 수 있습니다.
@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse(os.path.join("static", "index.html"))
# --- 🔼 [추가된 코드 끝] 🔼 ---


origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "http://localhost:63342",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIRECTORY = "/tmp/pdf_uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# --- OpenAI 대신 Google Gemini 모델을 사용하도록 변경 ---
try:
    # Google API 키가 있는지 확인
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

    # Google Gemini 임베딩 모델과 LLM 초기화
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)

except Exception as e:
    print(f"Google Gemini 모델을 초기화할 수 없습니다. API 키를 확인하세요. 오류: {e}")
    embeddings = None
    llm = None
# ----------------------------------------------------


QUESTION_PROMPT_TEMPLATE = """
당신은 주어진 컨텍스트를 분석하여 핵심적인 질문을 생성하는 AI 어시스턴트입니다.
컨텍스트의 내용을 바탕으로, 사람들이 궁금해할 만한 중요한 질문을 **한 개만** 만드세요.
질문은 간결하고 명확해야 합니다.

--- 컨텍스트 ---
{context}
-----------------

생성된 질문:
"""


@app.post("/generate-questions-from-pdf/")
async def generate_questions_from_pdf(file: UploadFile = File(...)):
    # --- Google Gemini 서비스 준비 상태를 확인하도록 변경 ---
    if not llm or not embeddings:
        raise HTTPException(status_code=503, detail="Google Gemini 서비스가 준비되지 않았습니다. 서버 로그나 API 키를 확인하세요.")

    unique_id = uuid.uuid4().hex
    file_path = os.path.join(UPLOAD_DIRECTORY, f"{unique_id}_{file.filename}")

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)

        if not split_docs:
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없거나 내용이 없습니다.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 파일 처리 중 오류 발생: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    question_generation_prompt = ChatPromptTemplate.from_template(QUESTION_PROMPT_TEMPLATE)
    question_chain = question_generation_prompt | llm | StrOutputParser()

    generated_questions = []
    docs_to_process = split_docs[:5]

    for doc in docs_to_process:
        generated_question = question_chain.invoke({"context": doc.page_content})
        generated_questions.append(generated_question.strip())

    return {
        "message": f"총 {len(split_docs)}개의 조각 중 {len(docs_to_process)}개에 대한 질문 생성이 완료되었습니다.",
        "questions": generated_questions

    }

