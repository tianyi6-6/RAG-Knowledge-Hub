import os
import uuid
from typing import List, Dict, Optional
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

# 加载环境变量
load_dotenv()

class RAGProcessor:
    """RAG智能知识库核心处理器"""
    
    def __init__(self):
        """初始化RAG处理器"""
        # 配置参数
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 600))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 60))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", 0.1))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", 2000))
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
        
        # 初始化组件
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 初始化向量存储
        self.vector_store = None
        self._load_vector_store()
        
        # 初始化对话记忆
        self.memory = ConversationBufferWindowMemory(
            k=5,  # 保持最近5轮对话
            return_messages=True
        )
    
    def _load_vector_store(self):
        """加载已存在的向量存储"""
        if os.path.exists(self.vector_store_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"成功加载向量存储: {self.vector_store_path}")
            except Exception as e:
                print(f"加载向量存储失败: {e}")
                self.vector_store = None
        else:
            self.vector_store = None
    
    def _save_vector_store(self):
        """保存向量存储到本地"""
        if self.vector_store:
            self.vector_store.save_local(self.vector_store_path)
            print(f"向量存储已保存到: {self.vector_store_path}")
    
    def load_document(self, file_path: str) -> List[Document]:
        """加载并解析文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析后的文档列表
        """
        file_extension = Path(file_path).suffix.lower()
        loader = None
        
        # 根据文件类型选择合适的加载器
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif file_extension == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")
        
        # 加载文档
        documents = loader.load()
        
        # 添加元数据
        file_name = Path(file_path).name
        file_id = str(uuid.uuid4())
        
        for doc in documents:
            # 确保元数据存在
            if not doc.metadata:
                doc.metadata = {}
            
            # 添加文件名和文件ID
            doc.metadata["file_name"] = file_name
            doc.metadata["file_id"] = file_id
            
            # 确保页码存在
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1
            
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分割成文本切片
        
        Args:
            documents: 原始文档列表
            
        Returns:
            分割后的文档切片列表
        """
        return self.text_splitter.split_documents(documents)
    
    def add_documents_to_vector_store(self, documents: List[Document]):
        """将文档添加到向量存储
        
        Args:
            documents: 文档列表
        """
        if not self.vector_store:
            # 如果向量存储不存在，创建新的
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        else:
            # 否则，添加到现有向量存储
            self.vector_store.add_documents(documents)
        
        # 保存向量存储
        self._save_vector_store()
    
    def process_file(self, file_path: str) -> Dict[str, any]:
        """处理文件并添加到知识库
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理结果信息
        """
        try:
            # 1. 加载文档
            documents = self.load_document(file_path)
            
            # 2. 分割文档
            chunks = self.split_documents(documents)
            
            # 3. 添加到向量存储
            self.add_documents_to_vector_store(chunks)
            
            return {
                "success": True,
                "message": f"文件处理成功",
                "file_name": Path(file_path).name,
                "document_count": len(documents),
                "chunk_count": len(chunks)
            }
        except Exception as e:
            error_msg = str(e).lower()  # 转换为小写，提高匹配成功率
            print(f"文件处理错误详细信息: {str(e)}")
            
            # 优化错误信息，使其更用户友好
            user_friendly_msg = "文件处理失败"
            
            # 处理OpenAI API密钥错误 - 使用更灵活的匹配
            if "401" in error_msg or "invalid_api_key" in error_msg or "incorrect api key" in error_msg:
                user_friendly_msg = "文件处理失败：无效的OpenAI API密钥，请检查.env文件中的配置"
            # 处理其他可能的API错误
            elif "error code:" in error_msg or "api" in error_msg:
                user_friendly_msg = "文件处理失败：API请求错误，请检查网络连接和配置"
            # 处理文件格式错误
            elif "不支持的文件类型" in str(e):  # 保持原语言，因为这是我们自己抛出的错误
                user_friendly_msg = str(e)
            # 处理其他未知错误
            else:
                user_friendly_msg = "文件处理失败：请检查配置和网络连接"
            
            return {
                "success": False,
                "message": user_friendly_msg,
                "file_name": Path(file_path).name
            }
    
    def search_documents(self, query: str, k: int = 3) -> List[Document]:
        """搜索相关文档
        
        Args:
            query: 查询语句
            k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        if not self.vector_store:
            return []
        
        # 搜索相似文档
        results = self.vector_store.similarity_search(
            query=query,
            k=k
        )
        
        return results
    
    def get_qa_chain(self):
        """获取问答链
        
        Returns:
            问答链实例
        """
        if not self.vector_store:
            raise ValueError("向量存储未初始化，请先添加文档")
        
        # 创建检索器
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # 先获取前10个结果
        )
        
        # 创建LLM实例
        llm = ChatOpenAI(
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens
        )
        
        # 创建问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        
        return qa_chain
    
    def answer_question(self, question: str) -> Dict[str, any]:
        """回答问题
        
        Args:
            question: 用户问题
            
        Returns:
            回答结果，包含答案和参考来源
        """
        try:
            # 获取问答链
            qa_chain = self.get_qa_chain()
            
            # 执行问答
            result = qa_chain({
                "query": question,
                "chat_history": self.memory.load_memory_variables({})
            })
            
            # 更新对话记忆
            self.memory.save_context(
                {"input": question},
                {"output": result["result"]}
            )
            
            # 提取参考来源
            sources = []
            for doc in result["source_documents"]:
                sources.append({
                    "file_name": doc.metadata.get("file_name", "未知文件"),
                    "page": doc.metadata.get("page", 1),
                    "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                })
            
            return {
                "success": True,
                "answer": result["result"],
                "sources": sources,
                "question": question
            }
        except Exception as e:
            error_msg = str(e).lower()  # 转换为小写，提高匹配成功率
            print(f"问答错误详细信息: {str(e)}")
            
            # 优化错误信息，使其更用户友好
            # 对于API密钥错误或无法连接到API的情况
            if "401" in error_msg or "invalid_api_key" in error_msg or "incorrect api key" in error_msg:
                return {
                    "success": False,
                    "answer": "抱歉，API密钥配置错误，请检查.env文件中的OpenAI API密钥",
                    "sources": [],
                    "question": question,
                    "error": "API密钥配置错误"
                }
            elif "error code:" in error_msg or "api" in error_msg or "connection" in error_msg:
                return {
                    "success": False,
                    "answer": "抱歉，当前无法连接到AI服务，请检查网络连接或API配置",
                    "sources": [],
                    "question": question,
                    "error": "AI服务连接错误"
                }
            # 对于向量存储未初始化的情况
            elif "向量存储未初始化" in str(e):
                return {
                    "success": False,
                    "answer": "抱歉，知识库尚未初始化，请先上传文档",
                    "sources": [],
                    "question": question,
                    "error": "知识库未初始化"
                }
            # 其他情况，返回默认的未找到内容
            else:
                return {
                    "success": False,
                    "answer": "抱歉，在知识库中未找到相关内容",
                    "sources": [],
                    "question": question,
                    "error": "未找到相关内容"
                }
    
    def list_documents(self) -> List[Dict[str, any]]:
        """列出知识库中的所有文档
        
        Returns:
            文档列表
        """
        if not self.vector_store:
            return []
        
        # 获取所有文档的元数据
        all_docs = self.vector_store.docstore._dict.values()
        
        # 去重并整理文档信息
        document_map = {}
        for doc in all_docs:
            file_id = doc.metadata.get("file_id")
            file_name = doc.metadata.get("file_name")
            
            if file_id and file_name:
                if file_id not in document_map:
                    document_map[file_id] = {
                        "file_id": file_id,
                        "file_name": file_name,
                        "page_count": 0,
                        "chunk_count": 0
                    }
                
                # 更新文档信息
                document_map[file_id]["chunk_count"] += 1
                current_page = doc.metadata.get("page", 1)
                if current_page > document_map[file_id]["page_count"]:
                    document_map[file_id]["page_count"] = current_page
        
        return list(document_map.values())
    
    def delete_document(self, file_id: str) -> bool:
        """从知识库中删除文档
        
        Args:
            file_id: 文件唯一标识符
            
        Returns:
            删除是否成功
        """
        if not self.vector_store:
            return False
        
        try:
            # 获取所有文档
            all_docs = self.vector_store.docstore._dict.values()
            
            # 找到要删除的文档ID
            doc_ids_to_delete = []
            for doc_id, doc in self.vector_store.docstore._dict.items():
                if doc.metadata.get("file_id") == file_id:
                    doc_ids_to_delete.append(doc_id)
            
            # 删除文档
            if doc_ids_to_delete:
                self.vector_store.delete(doc_ids_to_delete)
                self._save_vector_store()
                return True
            
            return False
        except Exception as e:
            print(f"删除文档失败: {e}")
            return False