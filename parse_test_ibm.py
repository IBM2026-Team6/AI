"""
발표 대본 자동 생성 시스템
- docs 폴더의 문서들(paper, report, docs)을 RAG로 활용
- paper의 각 페이지별로 발표 대본 자동 생성
- LangChain + IBM Watsonx 활용
- 이미지 기반 PDF 지원 (OCR 또는 수동 입력)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# LangChain 관련 모듈
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 환경 변수 로드
load_dotenv(override=True)

# IBM Watsonx 설정
WATSONX_API = os.environ['API_KEY']
PROJECT_ID = os.environ['PROJECT_ID']
IBM_URL = os.environ['IBM_CLOUD_URL']
EMBEDDING_MODEL_ID = "ibm/granite-embedding-278m-multilingual"
LLM_MODEL_ID = "meta-llama/llama-3-3-70b-instruct"

# 프로젝트 경로 설정
BASE_DIR = Path(__file__).parent
DOCS_FOLDER = BASE_DIR / "docs"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

print("=" * 80)
print("발표 대본 자동 생성 시스템")
print("=" * 80)


class PresentationScriptGenerator:
    """발표 대본 자동 생성 클래스"""
    
    def __init__(self):
        """초기화 및 IBM Watsonx 연동"""
        print("\n시스템 초기화 중...")
        
        # IBM Watsonx 인증 정보
        credentials = {
            "url": IBM_URL,
            "apikey": WATSONX_API
        }
        
        # 임베딩 모델 초기화
        self.embeddings = WatsonxEmbeddings(
            model_id=EMBEDDING_MODEL_ID,
            url=credentials["url"],
            apikey=credentials["apikey"],
            project_id=PROJECT_ID
        )
        
        # LLM 모델 초기화
        self.llm = WatsonxLLM(
            model_id=LLM_MODEL_ID,
            url=credentials["url"],
            apikey=credentials["apikey"],
            project_id=PROJECT_ID,
            params={
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50
            }
        )
        
        self.vector_db = None
        self.retriever = None
        self.rag_chain = None
        
        print("시스템 초기화 완료")
    
    def load_reference_documents(self):
        """docs 폴더의 참고 문서들을 벡터 DB에 저장"""
        print("\n참고 문서 로딩 중...")
        
        all_documents = []
        
        # docs 폴더의 모든 PDF 파일 처리
        reference_files = ["report.pdf", "docs.pdf"]  # paper.pdf는 제외 (발표 자료이므로)
        
        for file_name in reference_files:
            file_path = DOCS_FOLDER / file_name
            
            if not file_path.exists():
                print(f"경고: {file_name} 파일을 찾을 수 없습니다.")
                continue
            
            print(f"   - {file_name} 로딩 중...")
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # 문서 출처 메타데이터 추가
            for doc in documents:
                doc.metadata['source_file'] = file_name
            
            all_documents.extend(documents)
            print(f"     {len(documents)} 페이지 로드 완료")
        
        if not all_documents:
            raise ValueError("참고 문서를 찾을 수 없습니다!")
        
        # 텍스트 분할 (Chunking)
        # Granite 임베딩 모델의 최대 토큰 길이(512)를 고려하여 chunk_size를 작게 설정
        print("\n텍스트 분할 중...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # 512 토큰 제한을 고려하여 400자로 제한
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        texts = text_splitter.split_documents(all_documents)
        print(f"   {len(texts)}개의 청크(Chunks) 생성 완료")
        
        # 벡터 DB 생성 및 저장
        print("\n벡터 DB 구축 중 (시간이 소요될 수 있습니다)...")
        self.vector_db = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=str(CHROMA_DB_PATH)
        )
        self.vector_db.persist()
        
        # Retriever 설정 (검색 결과 상위 5개)
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={'k': 5}
        )
        
        print("벡터 DB 구축 완료")
    
    def setup_rag_chain(self):
        """RAG 체인 구성"""
        print("\nRAG 체인 구성 중...")
        
        # 문서 포맷팅 함수
        def format_docs(docs):
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source_file', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                formatted.append(
                    f"[참고자료 {i}] {source} (페이지 {page})\n{doc.page_content}"
                )
            return "\n\n" + "=" * 60 + "\n\n".join(formatted)
        
        # 프롬프트 템플릿 정의
        template = """당신은 전문적인 발표 대본 작성가입니다.
아래의 [발표 슬라이드 내용]과 [참고 자료]를 바탕으로, 청중에게 전달력 있고 이해하기 쉬운 발표 대본을 작성해주세요.

**작성 지침:**
1. 발표 대상자와 주제에 맞는 적절한 어조와 전문성을 유지하세요.
2. 슬라이드의 핵심 내용을 명확하게 전달하세요.
3. 참고 자료의 구체적인 정보를 활용하여 내용을 풍부하게 만드세요.
4. 자연스러운 말투로 작성하되, 전문 용어는 정확하게 사용하세요.
5. 발표 시간을 고려하여 적절한 분량으로 작성하세요.
6. 청중의 이해를 돕는 예시나 설명을 추가하세요.

[참고 자료]
{context}

[발표 슬라이드 내용]
{slide_content}

발표 대본:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # LCEL 파이프라인 구성
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "slide_content": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("RAG 체인 구성 완료")
    
    def load_presentation_slides(self, use_text_file=False):
        """발표 자료를 페이지별로 로드
        
        Args:
            use_text_file: True이면 slide_contents.txt에서 읽기, False이면 paper.pdf에서 추출
        """
        print("\n발표 자료 로딩 중...")
        
        if use_text_file:
            # slide_contents.txt에서 슬라이드 내용 읽기
            slides_file = BASE_DIR / "slide_contents.txt"
            if not slides_file.exists():
                raise FileNotFoundError(
                    f"슬라이드 내용 파일을 찾을 수 없습니다: {slides_file}\n"
                    f"slide_contents.txt 파일을 생성하고 슬라이드 내용을 입력해주세요."
                )
            
            print("   - slide_contents.txt에서 슬라이드 내용을 읽습니다...")
            with open(slides_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # [PAGE X] 형식으로 분리
            import re
            pages = re.split(r'\[PAGE \d+\]', content)
            pages = [p.strip() for p in pages if p.strip() and not p.strip().startswith('#')]
            
            # Document 형식으로 변환
            from langchain_core.documents import Document
            slides = [
                Document(page_content=content, metadata={'page': i+1, 'source': 'slide_contents.txt'})
                for i, content in enumerate(pages)
            ]
            
            print(f"   총 {len(slides)} 페이지 로드 완료")
            return slides
        
        # PDF에서 슬라이드 로드
        paper_path = DOCS_FOLDER / "paper.pdf"
        
        if not paper_path.exists():
            raise FileNotFoundError(f"발표 자료를 찾을 수 없습니다: {paper_path}")
        
        # 먼저 PyPDFLoader로 시도
        print("   - PyPDFLoader로 시도 중...")
        loader = PyPDFLoader(str(paper_path))
        slides = loader.load()
        
        # 텍스트가 추출되지 않았다면 UnstructuredPDFLoader 시도
        if all(len(slide.page_content.strip()) == 0 for slide in slides):
            print("   경고: 텍스트가 추출되지 않았습니다. UnstructuredPDFLoader로 재시도 중...")
            try:
                loader = UnstructuredPDFLoader(str(paper_path), mode="elements")
                slides = loader.load()
                print(f"   UnstructuredPDFLoader로 {len(slides)} 페이지 로드 완료")
            except Exception as e:
                print(f"   오류: UnstructuredPDFLoader 오류: {str(e)}")
                print("\n" + "="*80)
                print("경고: paper.pdf가 이미지 기반 PDF로 보입니다.")
                print("해결 방법:")
                print("   1. slide_contents.txt 파일을 열어서 각 슬라이드의 내용을 입력하세요.")
                print("   2. main.py를 실행할 때 --use-text-file 옵션을 사용하세요:")
                print("      python main.py --use-text-file")
                print("="*80 + "\n")
                raise RuntimeError("PDF에서 텍스트를 추출할 수 없습니다. slide_contents.txt를 사용하세요.")
        else:
            print(f"   총 {len(slides)} 페이지 로드 완료")
        
        return slides
    
    def generate_script_for_slide(self, slide_content, page_num):
        """특정 슬라이드에 대한 발표 대본 생성"""
        print(f"\n{page_num}페이지 대본 생성 중...")
        
        # RAG 체인 실행
        script = self.rag_chain.invoke(slide_content)
        
        return script
    
    def generate_all_scripts(self, use_text_file=False):
        """모든 슬라이드에 대한 발표 대본 생성
        
        Args:
            use_text_file: True이면 slide_contents.txt 사용, False이면 paper.pdf 사용
        """
        print("\n" + "=" * 80)
        print("발표 대본 생성 시작")
        print("=" * 80)
        
        # 발표 자료 로드
        slides = self.load_presentation_slides(use_text_file=use_text_file)
        
        # 결과 저장용
        all_scripts = []
        
        # 각 슬라이드별로 대본 생성
        for i, slide in enumerate(slides, 1):
            slide_content = slide.page_content
            
            # 디버깅: 페이지 내용 확인
            print(f"\n페이지 {i}: 추출된 텍스트 길이 = {len(slide_content)} 문자")
            if len(slide_content) > 0:
                print(f"   미리보기: {slide_content[:100]}...")
            
            # 텍스트가 없는 경우 (이미지 기반 슬라이드)
            if len(slide_content.strip()) < 5:
                print(f"   내용이 없거나 너무 짧아 건너뜁니다.")
                print(f"   참고: 이 페이지는 이미지 기반일 수 있습니다. 수동으로 내용을 확인해주세요.")
                continue
            
            try:
                script = self.generate_script_for_slide(slide_content, i)
                
                all_scripts.append({
                    'page': i,
                    'slide_content': slide_content,
                    'script': script
                })
                
                print(f"   완료")
                
            except Exception as e:
                print(f"   오류 발생: {str(e)}")
                continue
        
        return all_scripts
    
    def save_scripts_to_file(self, scripts, output_path="presentation_scripts.txt"):
        """생성된 대본을 파일로 저장"""
        print(f"\n대본 저장 중: {output_path}")
        
        output_file = BASE_DIR / output_path
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("발표 대본 자동 생성 결과\n")
            f.write("=" * 80 + "\n\n")
            
            for script_data in scripts:
                page = script_data['page']
                slide_content = script_data['slide_content']
                script = script_data['script']
                
                f.write(f"\n{'=' * 80}\n")
                f.write(f"슬라이드 {page}페이지\n")
                f.write(f"{'=' * 80}\n\n")
                
                f.write("[슬라이드 내용]\n")
                f.write("-" * 80 + "\n")
                f.write(slide_content[:500])  # 처음 500자만 표시
                if len(slide_content) > 500:
                    f.write("\n... (내용 생략) ...")
                f.write("\n" + "-" * 80 + "\n\n")
                
                f.write("[발표 대본]\n")
                f.write("-" * 80 + "\n")
                f.write(script)
                f.write("\n" + "-" * 80 + "\n\n")
        
        print(f"   저장 완료: {output_file}")
        
        return output_file


def main():
    """메인 실행 함수"""
    try:
        # 명령줄 인자 확인
        use_text_file = '--use-text-file' in sys.argv
        
        if use_text_file:
            print("\n텍스트 모드: slide_contents.txt에서 슬라이드 내용을 읽습니다.\n")
        else:
            print("\nPDF 모드: paper.pdf에서 슬라이드 내용을 추출합니다.\n")
        
        # 1. 시스템 초기화
        generator = PresentationScriptGenerator()
        
        # 2. 참고 문서 로드 및 벡터 DB 구축
        generator.load_reference_documents()
        
        # 3. RAG 체인 구성
        generator.setup_rag_chain()
        
        # 4. 모든 슬라이드에 대한 대본 생성
        scripts = generator.generate_all_scripts(use_text_file=use_text_file)
        
        # 5. 결과 저장
        output_file = generator.save_scripts_to_file(scripts)
        
        # 6. 결과 요약
        print("\n" + "=" * 80)
        print("발표 대본 생성 완료")
        print("=" * 80)
        print(f"총 {len(scripts)}개 슬라이드의 대본이 생성되었습니다.")
        print(f"저장 위치: {output_file}")
        print("\n생성된 대본을 확인하고 필요에 따라 수정하여 사용하세요.")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
