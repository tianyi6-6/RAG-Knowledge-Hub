import pytest
import os
import tempfile
from src.core.rag_processor import RAGProcessor

# 创建测试用的临时文本文件
def create_temp_test_file(content: str, extension: str = ".txt") -> str:
    """创建临时测试文件"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as f:
        f.write(content.encode("utf-8"))
        return f.name

class TestRAGProcessor:
    """测试RAGProcessor核心功能"""
    
    def setup_method(self):
        """测试前初始化"""
        self.rag_processor = RAGProcessor()
    
    def teardown_method(self):
        """测试后清理"""
        # 清理向量存储
        vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
        if os.path.exists(vector_store_path):
            import shutil
            shutil.rmtree(vector_store_path)
    
    def test_text_splitting(self):
        """测试文本分割功能"""
        # 创建测试文本
        test_content = "这是一个测试文本。" * 200  # 生成较长的文本
        temp_file = create_temp_test_file(test_content)
        
        try:
            # 加载并分割文档
            documents = self.rag_processor.load_document(temp_file)
            chunks = self.rag_processor.split_documents(documents)
            
            # 验证分割结果
            assert len(chunks) > 1, "文本应该被分割成多个片段"
            
            # 验证每个片段的大小
            for chunk in chunks:
                assert len(chunk.page_content) <= self.rag_processor.chunk_size,\
                    f"片段大小不应超过 {self.rag_processor.chunk_size}"
        finally:
            # 清理临时文件
            os.unlink(temp_file)
    
    def test_document_processing(self):
        """测试文档处理功能"""
        # 创建测试文本
        test_content = "这是一个测试文档，用于验证RAG处理器的文档处理功能。"
        temp_file = create_temp_test_file(test_content)
        
        try:
            # 处理文档
            result = self.rag_processor.process_file(temp_file)
            
            # 验证处理结果
            assert result["success"] == True, "文档处理应该成功"
            assert result["file_name"] == os.path.basename(temp_file), "文件名应该匹配"
            assert result["document_count"] >= 1, "应该至少处理一个文档"
            assert result["chunk_count"] >= 1, "应该至少生成一个文本切片"
        finally:
            # 清理临时文件
            os.unlink(temp_file)
    
    def test_document_listing(self):
        """测试文档列表功能"""
        # 初始状态下应该没有文档
        initial_docs = self.rag_processor.list_documents()
        assert len(initial_docs) == 0, "初始状态下知识库应该为空"
        
        # 创建并处理两个测试文档
        test_content1 = "这是第一个测试文档。"
        test_content2 = "这是第二个测试文档。"
        
        temp_file1 = create_temp_test_file(test_content1, ".txt")
        temp_file2 = create_temp_test_file(test_content2, ".txt")
        
        try:
            # 处理文档
            self.rag_processor.process_file(temp_file1)
            self.rag_processor.process_file(temp_file2)
            
            # 获取文档列表
            docs = self.rag_processor.list_documents()
            
            # 验证文档列表
            assert len(docs) == 2, "知识库中应该有两个文档"
        finally:
            # 清理临时文件
            os.unlink(temp_file1)
            os.unlink(temp_file2)
    
    def test_document_deletion(self):
        """测试文档删除功能"""
        # 创建并处理测试文档
        test_content = "这是一个用于测试删除功能的文档。"
        temp_file = create_temp_test_file(test_content)
        
        try:
            # 处理文档
            result = self.rag_processor.process_file(temp_file)
            assert result["success"] == True, "文档处理应该成功"
            
            # 获取文档列表
            docs = self.rag_processor.list_documents()
            assert len(docs) == 1, "知识库中应该有一个文档"
            
            # 删除文档
            file_id = docs[0]["file_id"]
            delete_result = self.rag_processor.delete_document(file_id)
            assert delete_result == True, "文档删除应该成功"
            
            # 验证文档已删除
            docs_after_delete = self.rag_processor.list_documents()
            assert len(docs_after_delete) == 0, "知识库中应该没有文档"
        finally:
            # 清理临时文件
            os.unlink(temp_file)
    
    def test_search_functionality(self):
        """测试搜索功能"""
        # 创建测试文档
        test_content = "这是一个关于人工智能和机器学习的测试文档。人工智能是计算机科学的一个分支，机器学习是人工智能的一个子领域。"
        temp_file = create_temp_test_file(test_content)
        
        try:
            # 处理文档
            self.rag_processor.process_file(temp_file)
            
            # 搜索相关内容
            results = self.rag_processor.search_documents("人工智能")
            
            # 验证搜索结果
            assert len(results) > 0, "应该返回相关搜索结果"
            
            # 验证结果相关性
            for result in results:
                assert "人工智能" in result.page_content, "搜索结果应该包含关键词"
        finally:
            # 清理临时文件
            os.unlink(temp_file)
    
    def test_qa_functionality(self):
        """测试问答功能"""
        # 创建测试文档
        test_content = "LangChain是一个用于构建大语言模型应用的框架。它提供了多种组件，包括文档加载器、文本分割器、向量存储和检索器等。"
        temp_file = create_temp_test_file(test_content)
        
        try:
            # 处理文档
            self.rag_processor.process_file(temp_file)
            
            # 提问
            result = self.rag_processor.answer_question("LangChain是什么？")
            
            # 验证回答结果
            assert result["success"] == True, "问答应该成功"
            assert "LangChain" in result["answer"], "回答应该包含LangChain"
            assert len(result["sources"]) > 0, "回答应该包含参考来源"
        finally:
            # 清理临时文件
            os.unlink(temp_file)
    
    def test_unknown_question(self):
        """测试未知问题的处理"""
        # 创建测试文档（仅包含特定内容）
        test_content = "这是一个关于Python编程语言的测试文档。Python是一种高级编程语言，用于Web开发、数据分析和人工智能等领域。"
        temp_file = create_temp_test_file(test_content)
        
        try:
            # 处理文档
            self.rag_processor.process_file(temp_file)
            
            # 提问无关问题
            result = self.rag_processor.answer_question("什么是Java？")
            
            # 验证回答结果
            assert result["success"] == True, "问答应该成功"
            assert "抱歉，在知识库中未找到相关内容" in result["answer"],\
                "对于未知问题，应该返回特定提示"
        finally:
            # 清理临时文件
            os.unlink(temp_file)