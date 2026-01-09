from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
from src.core.rag_processor import RAGProcessor

# 初始化FastAPI应用
app = FastAPI(
    title="企业级 RAG 智能知识库 API",
    description="提供文档上传、管理和智能问答功能的RESTful API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化RAG处理器
rag_processor = RAGProcessor()

# 请求模型
class QuestionRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = None

class DeleteDocumentRequest(BaseModel):
    file_id: str

# 响应模型
class Response(BaseModel):
    success: bool
    message: str

class ProcessFileResponse(Response):
    file_name: str
    document_count: Optional[int] = None
    chunk_count: Optional[int] = None

class AnswerResponse(Response):
    answer: str
    sources: List[dict]
    question: str

class DocumentInfo(BaseModel):
    file_id: str
    file_name: str
    page_count: int
    chunk_count: int

class DocumentListResponse(Response):
    documents: List[DocumentInfo]

# 健康检查端点
@app.get("/health", tags=["健康检查"])
def health_check():
    """检查API服务是否正常运行"""
    return {
        "status": "ok",
        "message": "企业级 RAG 智能知识库 API 服务正常运行",
        "version": "1.0.0"
    }

# 上传文件端点
@app.post("/api/v1/documents/upload", response_model=ProcessFileResponse, tags=["文档管理"])
async def upload_file(file: UploadFile = File(...)):
    """上传并处理文档"""
    # 验证文件类型
    allowed_extensions = {"pdf", "md", "docx", "txt"}
    file_extension = os.path.splitext(file.filename)[1].lower().lstrip(".")
    
    if file_extension not in allowed_extensions:
        return ProcessFileResponse(
            success=False,
            message=f"不支持的文件类型，仅支持 {', '.join(allowed_extensions)} 格式",
            file_name=file.filename
        )
    
    try:
        # 保存文件到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        # 处理文件
        result = rag_processor.process_file(temp_file_path)
        
        # 删除临时文件
        os.unlink(temp_file_path)
        
        return ProcessFileResponse(
            success=result["success"],
            message=result["message"],
            file_name=result["file_name"],
            document_count=result.get("document_count"),
            chunk_count=result.get("chunk_count")
        )
    except Exception as e:
        error_msg = str(e)
        print(f"API文件处理错误详细信息: {error_msg}")
        
        # 优化错误信息，使其更用户友好
        user_friendly_msg = "文件处理失败"
        
        # 处理OpenAI API密钥错误
        if "Error code: 401" in error_msg or "invalid_api_key" in error_msg:
            user_friendly_msg = "文件处理失败：无效的OpenAI API密钥，请检查.env文件中的配置"
        # 处理其他可能的API错误
        elif "Error code:" in error_msg:
            user_friendly_msg = "文件处理失败：API请求错误，请检查网络连接和配置"
        # 处理其他未知错误
        else:
            user_friendly_msg = f"文件处理失败：{str(e)[:100]}..."  # 限制错误信息长度
        
        return ProcessFileResponse(
            success=False,
            message=user_friendly_msg,
            file_name=file.filename
        )

# 获取文档列表端点
@app.get("/api/v1/documents", response_model=DocumentListResponse, tags=["文档管理"])
def get_documents():
    """获取知识库中的所有文档"""
    try:
        documents = rag_processor.list_documents()
        
        return DocumentListResponse(
            success=True,
            message="获取文档列表成功",
            documents=[
                DocumentInfo(
                    file_id=doc["file_id"],
                    file_name=doc["file_name"],
                    page_count=doc["page_count"],
                    chunk_count=doc["chunk_count"]
                ) for doc in documents
            ]
        )
    except Exception as e:
        print(f"获取文档列表错误详细信息: {str(e)}")
        return DocumentListResponse(
            success=False,
            message="获取文档列表失败，请稍后重试",
            documents=[]
        )

# 删除文档端点
@app.delete("/api/v1/documents", response_model=Response, tags=["文档管理"])
def delete_document(request: DeleteDocumentRequest):
    """从知识库中删除文档"""
    try:
        success = rag_processor.delete_document(request.file_id)
        
        if success:
            return Response(
                success=True,
                message="文档删除成功"
            )
        else:
            return Response(
                success=False,
                message="文档删除失败，文件ID不存在"
            )
    except Exception as e:
        print(f"删除文档错误详细信息: {str(e)}")
        return Response(
            success=False,
            message="文档删除失败，请稍后重试"
        )

# 智能问答端点
@app.post("/api/v1/qa", response_model=AnswerResponse, tags=["智能问答"])
def ask_question(request: QuestionRequest):
    """基于知识库进行智能问答"""
    try:
        result = rag_processor.answer_question(request.question)
        
        return AnswerResponse(
            success=result["success"],
            message="问答成功" if result["success"] else "问答失败",
            answer=result["answer"],
            sources=result["sources"],
            question=result["question"]
        )
    except Exception as e:
        error_msg = str(e)
        print(f"问答API错误详细信息: {error_msg}")
        
        # 优化错误信息，使其更用户友好
        if "Error code: 401" in error_msg or "invalid_api_key" in error_msg:
            return AnswerResponse(
                success=False,
                message="API密钥配置错误",
                answer="抱歉，API密钥配置错误，请检查.env文件中的OpenAI API密钥",
                sources=[],
                question=request.question
            )
        else:
            return AnswerResponse(
                success=False,
                message="问答失败",
                answer="抱歉，当前无法处理您的请求，请稍后重试",
                sources=[],
                question=request.question
            )

# 搜索文档端点
@app.get("/api/v1/search", tags=["文档搜索"])
def search_documents(query: str, k: int = 3):
    """搜索相关文档"""
    try:
        results = rag_processor.search_documents(query, k=k)
        
        # 格式化搜索结果
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "success": True,
            "message": "搜索成功",
            "query": query,
            "results": formatted_results
        }
    except Exception as e:
        error_msg = str(e)
        print(f"搜索API错误详细信息: {error_msg}")
        
        return {
            "success": False,
            "message": "搜索失败，请稍后重试",
            "query": query,
            "results": []
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )