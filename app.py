import streamlit as st
import os
import tempfile
from pathlib import Path
from src.core.rag_processor import RAGProcessor

# 设置页面配置
st.set_page_config(
    page_title="企业级 RAG 智能知识库",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化RAG处理器
# 移除cache_resource装饰器，确保每次运行都使用最新的代码
# @st.cache_resource
def init_rag_processor():
    return RAGProcessor()

rag_processor = init_rag_processor()

# 创建临时目录用于文件上传
temp_dir = tempfile.mkdtemp()

# 侧边栏
with st.sidebar:
    st.title("📁 文档管理")
    
    # 文件上传区域
    st.header("上传文件")
    uploaded_files = st.file_uploader(
        "支持 .pdf, .md, .docx, .txt 格式",
        type=["pdf", "md", "docx", "txt"],
        accept_multiple_files=True,
        help="选择要上传到知识库的文件"
    )
    
    # 文件上传处理
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # 保存文件到临时目录
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 处理文件
            with st.spinner(f"正在处理 {uploaded_file.name}..."):
                result = rag_processor.process_file(file_path)
                
                if result["success"]:
                    st.success(f"✅ {uploaded_file.name} 处理成功")
                    st.info(f"文档数: {result['document_count']}, 切片数: {result['chunk_count']}")
                else:
                    st.error(f"❌ {uploaded_file.name} 处理失败")
                    st.error(result["message"])
    
    # 分隔线
    st.divider()
    
    # 已上传文件列表
    st.header("已上传文件")
    documents = rag_processor.list_documents()
    
    if documents:
        for doc in documents:
            with st.expander(f"📄 {doc['file_name']}"):
                st.write(f"**文件ID**: {doc['file_id']}")
                st.write(f"**页数**: {doc['page_count']}")
                st.write(f"**切片数**: {doc['chunk_count']}")
                
                if st.button(f"删除 {doc['file_name']}", key=f"delete_{doc['file_id']}"):
                    if rag_processor.delete_document(doc['file_id']):
                        st.success(f"✅ {doc['file_name']} 已删除")
                        st.rerun()
                    else:
                        st.error(f"❌ {doc['file_name']} 删除失败")
    else:
        st.info("📭 知识库中暂无文档")
    
    # 分隔线
    st.divider()
    
    # 系统信息
    st.header("系统信息")
    st.write("🤖 企业级 RAG 智能知识库")
    st.write("📚 基于 LangChain + FAISS + OpenAI")
    st.write("🔒 私有数据，安全可靠")

# 主界面
st.title("🤖 企业级 RAG 智能知识库")
st.subheader("基于私有文档的智能问答系统")

# 聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # 显示引用来源
        if message.get("sources"):
            with st.expander("📚 参考来源"):
                for source in message["sources"]:
                    st.write(f"**{source['file_name']}** (第 {source['page']} 页)")
                    st.caption(source['content'])

# 聊天输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息到聊天历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 生成AI回答
    with st.chat_message("assistant"):
        # 使用流式输出
        message_placeholder = st.empty()
        full_response = ""
        
        # 获取回答结果
        result = rag_processor.answer_question(prompt)
        
        # 显示回答内容
        full_response = result["answer"]
        message_placeholder.markdown(full_response)
        
        # 显示引用来源
        if result.get("sources"):
            with st.expander("📚 参考来源"):
                for source in result["sources"]:
                    st.write(f"**{source['file_name']}** (第 {source['page']} 页)")
                    st.caption(source['content'])
    
    # 添加AI消息到聊天历史
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": result.get("sources", [])
    })

# 页脚
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #666;
        border-top: 1px solid #e0e0e0;
        z-index: 1000;
    }
    </style>
    <div class="footer">
        <p>企业级 RAG 智能知识库 | 基于 LangChain 构建</p>
    </div>
    """,
    unsafe_allow_html=True
)