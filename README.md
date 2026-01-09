# 企业级 RAG 智能知识库 (RAG-Knowledge-Hub)

## 项目概述

构建一个企业级专业 RAG (检索增强生成) 智能知识库系统，实现私有文档的安全上传、智能解析与精准问答功能。系统确保 AI 回答严格基于用户提供的私有数据，有效消除大语言模型幻觉问题，并全面保障企业数据隐私安全。

## 技术栈

- **核心框架**: LangChain (最新稳定版)
- **大语言模型接口**: DeepSeek-V3 (兼容 OpenAI API 格式) 或 Llama-3 (通过 Ollama 本地部署)
- **向量数据库**: FAISS (高效的相似性搜索库)
- **向量嵌入模型**: OpenAI text-embedding-3-small
- **后端接口**: FastAPI (RESTful API 设计)
- **用户前端**: Streamlit (简洁直观、现代美观且响应式)

## 核心功能

### 模块 A：数据接入与处理 (Data Ingestion)
- **多格式文件加载**: 支持 .pdf, .md, .docx, .txt 四种文件格式的完整解析
- **语义切片处理**: 采用 RecursiveCharacterTextSplitter 进行文本分割，参数设置为 chunk_size=600, chunk_overlap=60
- **元数据提取与存储**: 每个文本切片包含并记录来源文件名、原始页码及文件唯一标识符

### 模块 B：检索引擎 (Retrieval Engine)
- **文本向量化**: 将文本切片转换为 1536 维向量表示
- **相似性搜索**: 实现高效的向量相似度搜索
- **结果优化**: 精选最相关的 Top-3 片段作为 LLM 输入

### 模块 C：逻辑推理与安全机制
- **CoT 思维链推理**: 强制模型在生成最终回答前执行"思维链推理"过程
- **事实约束机制**: 当检索上下文中无法找到问题答案时，模型固定输出："抱歉，在知识库中未找到相关内容"
- **对话记忆管理**: 集成 ConversationBufferWindowMemory，保持最近 5 轮对话的上下文连贯性

### 模块 D：用户交互界面 (Streamlit)
- **文档管理功能**: 侧边栏实现文件上传区域、已解析文件列表展示及文件删除功能
- **智能聊天窗口**: 实现标准对话流交互，支持回答内容的流式输出
- **引用溯源展示**: 回答内容底部清晰列出所有参考的文档名称及具体页码信息

## 项目结构

```
RAG-Knowledge-Hub/
├── src/
│   ├── core/              # 核心功能模块
│   │   └── rag_processor.py  # RAG 处理器实现
│   ├── api/               # API 相关代码
│   ├── utils/             # 工具函数
│   └── tests/             # 单元测试
├── data/                  # 数据目录
│   └── vector_store/      # 向量存储
├── .env                   # 环境配置文件
├── requirements.txt        # 依赖库列表
├── main.py                # FastAPI 后端入口
├── app.py                 # Streamlit 前端入口
└── README.md              # 项目说明文档
```

## 环境配置

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **配置环境变量**
   复制 `.env` 文件并修改以下配置：
   ```dotenv
   # OpenAI API 配置
   OPENAI_API_KEY=your_openai_api_key_here
   EMBEDDING_MODEL=text-embedding-3-small
   
   # 应用配置
   CHUNK_SIZE=600
   CHUNK_OVERLAP=60
   VECTOR_STORE_PATH=./data/vector_store
   ```

## 运行项目

### 方式一：Streamlit 前端 (推荐)
```bash
streamlit run app.py
```

访问 http://localhost:8501 即可使用

### 方式二：FastAPI 后端
```bash
uvicorn main:app --reload
```

API 文档访问地址：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 端点

- **GET /health**: 健康检查
- **POST /api/v1/documents/upload**: 上传并处理文档
- **GET /api/v1/documents**: 获取文档列表
- **DELETE /api/v1/documents**: 删除文档
- **POST /api/v1/qa**: 智能问答
- **GET /api/v1/search**: 文档搜索

## 使用示例

1. **上传文档**
   - 在 Streamlit 界面的侧边栏选择要上传的文件
   - 支持批量上传，系统会自动处理并添加到知识库

2. **智能问答**
   - 在主界面的聊天输入框中输入问题
   - 系统会基于知识库内容生成回答，并在回答底部显示参考来源

3. **文档管理**
   - 在侧边栏的"已上传文件"列表中查看已添加的文档
   - 可以删除不需要的文档，系统会自动更新知识库

## 开发说明

### 单元测试
```bash
python -m pytest src/tests/ -v
```

### 代码规范
- 遵循 PEP 8 规范
- 实现模块化与组件化设计
- 关键功能编写单元测试，测试覆盖率不低于 70%
- 所有用户输入进行验证与清洗，防止注入攻击

## 部署建议

1. **生产环境配置**
   - 使用 gunicorn + uvicorn 部署 FastAPI
   - 配置 Nginx 作为反向代理
   - 设置适当的日志级别和监控

2. **数据安全**
   - 定期备份向量存储和原始文档
   - 配置适当的访问控制
   - 考虑使用私有部署的嵌入模型和大语言模型

## 注意事项

1. 首次运行时，系统会自动创建向量存储目录
2. 确保环境变量中的 API 密钥有效且具有相应权限
3. 处理大型文件时，可能需要较长时间，请耐心等待
4. 建议定期清理不需要的文档，以保持知识库的高效性

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。