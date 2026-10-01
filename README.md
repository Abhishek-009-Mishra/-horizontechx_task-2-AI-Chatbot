# AWS Padho ☁️🤖

### Agentic AWS Q&A Assistant using LangGraph, RAG, FAISS, Ollama & Groq

**AWS Padho** is an AI-powered command-line Q&A assistant designed to help users learn AWS concepts from authoritative AWS certification study material.

The project uses **Retrieval-Augmented Generation (RAG)** combined with an **agentic workflow built using LangGraph**. It retrieves relevant information from AWS certification books, evaluates the retrieved context, optionally rewrites the user's query when the context is insufficient, and generates a grounded answer using the Groq LLM API.

This project was developed as part of an **AI/ML internship task** to demonstrate practical implementation of **Agentic AI, RAG pipelines, vector databases, LLM integration, and document retrieval**.

---

## 🚀 Features

- 📚 Supports AWS certification study material
- 🔎 Retrieval-Augmented Generation (RAG)
- 🤖 Agentic workflow using **LangGraph**
- 🧠 LLM-powered answers using **Groq**
- 🔢 Local embeddings using **Ollama**
- 🗂️ FAISS vector database for efficient similarity search
- 🔄 Query rewriting when retrieved context is insufficient
- 📖 Source/page references for retrieved information
- 💻 Lightweight command-line interface
- ⚡ Streaming AI responses in the terminal
- 🔐 API keys managed securely using environment variables
- 💾 Persistent FAISS vector database
- 🛡️ Error handling for missing files, APIs, models, and vector databases

---

# 📚 Supported Books

The current project is designed to work with the following AWS certification study guides:

### 1. AWS Certified Cloud Practitioner — CLF-C02

**AWS Certified Cloud Practitioner Foundational (CLF-C02) Exam Study Guide**

Authors: Ben Piper & David Clinton

### 2. AWS Certified Solutions Architect — SAA-C02

**AWS Certified Solutions Architect Study Guide, 3rd Edition**

Authors: Ben Piper & David Clinton

> The PDF files are not included in the repository. Users should provide their own legally obtained copies.

---

# 🏗️ System Architecture

```text
                    AWS Study Books
                         │
                         ▼
                  PDF Document Loader
                         │
                         ▼
                    Text Extraction
                         │
                         ▼
                  Document Chunking
                         │
                         ▼
                Ollama Embeddings
                (nomic-embed-text)
                         │
                         ▼
                    FAISS Vector DB
                         │
                         ▼
                 ┌─────────────────┐
                 │    LangGraph    │
                 │  Agentic Q&A    │
                 └────────┬────────┘
                          │
                          ▼
                   Query Analysis
                          │
                          ▼
                  Document Retrieval
                          │
                          ▼
                  Context Evaluation
                          │
                 ┌────────┴─────────┐
                 │                  │
              Sufficient         Insufficient
                 │                  │
                 │                  ▼
                 │            Query Rewriting
                 │                  │
                 │                  ▼
                 │            Re-retrieval
                 │                  │
                 └────────┬─────────┘
                          ▼
                    Groq LLM
                          │
                          ▼
                    Final Answer
                          │
                          ▼
                  Sources & Pages
```

---

# 🧠 Agentic Workflow

Unlike a traditional RAG pipeline, AWS Padho uses **LangGraph** to control the retrieval and answer-generation process.

The workflow consists of the following nodes:

```text
START
  │
  ▼
Analyze Question
  │
  ▼
Retrieve Documents
  │
  ▼
Evaluate Context
  │
  ├─────────────── Context Sufficient ───────────────┐
  │                                                  │
  ▼                                                  ▼
Rewrite Query                                  Generate Answer
  │                                                  │
  ▼                                                  ▼
Retrieve Documents                                 END
  │
  ▼
Evaluate Context
```

### 1. Analyze Question

The user's question is analyzed to determine what information needs to be retrieved.

### 2. Retrieve Documents

Relevant chunks are retrieved from the FAISS vector database using semantic similarity search.

### 3. Evaluate Context

The retrieved information is evaluated to determine whether it is sufficient to answer the question.

### 4. Rewrite Query

If the retrieved context is insufficient, the agent reformulates the query and performs another retrieval attempt.

### 5. Generate Answer

Once sufficient context is available, the retrieved information is passed to the Groq-powered LLM to generate the final answer.

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **LangGraph** | Agentic workflow orchestration |
| **LangChain** | RAG and document-processing framework |
| **FAISS** | Vector similarity search |
| **Ollama** | Local embedding generation |
| **nomic-embed-text** | Embedding model |
| **Groq** | LLM inference |
| **PyPDF** | PDF text extraction |
| **python-dotenv** | Environment variable management |

---

# 📁 Project Structure

```text
aws-padho/
│
├── data/
│   ├── clf_c02/
│   │   └── clf_c02.pdf
│   │
│   └── saa_c02/
│       └── saa_c02.pdf
│
├── vector_db/
│
├── graph/
│   ├── __init__.py
│   ├── state.py
│   ├── nodes.py
│   └── workflow.py
│
├── retrieval/
│   ├── __init__.py
│   ├── loader.py
│   ├── embeddings.py
│   └── retriever.py
│
├── prompts/
│   ├── __init__.py
│   └── prompts.py
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── ingest.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd aws-padho
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 📦 Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The major dependencies include:

```text
langchain
langchain-community
langchain-text-splitters
langgraph
langchain-ollama
langchain-groq
pypdf
faiss-cpu
ollama
groq
python-dotenv
```

---

# 🧠 Configure Ollama

AWS Padho uses Ollama for generating document embeddings locally.

Install Ollama and then download the embedding model:

```bash
ollama pull nomic-embed-text
```

Verify that the model is installed:

```bash
ollama list
```

Make sure Ollama is running before starting document ingestion or the chatbot.

---

# 🔑 Configure Groq API

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=your_groq_model_here
```

The API key should **never be hardcoded into the source code**.

A `.env.example` file is included for reference.

---

# 📄 Add the AWS Books

Place the PDF files in the following locations:

```text
data/
├── clf_c02/
│   └── clf_c02.pdf
│
└── saa_c02/
    └── saa_c02.pdf
```

The application checks whether the required PDFs exist before starting ingestion.

If the PDFs are missing, the application displays the expected file locations instead of creating fake data.

---

# 🗃️ Build the Vector Database

After placing the PDFs in the correct directories, run:

```bash
python ingest.py
```

The ingestion pipeline performs the following operations:

```text
PDF Files
   ↓
PDF Text Extraction
   ↓
Document Chunking
   ↓
Metadata Assignment
   ↓
Ollama Embeddings
   ↓
FAISS Index
   ↓
Persistent Vector Database
```

Metadata such as the following is maintained where available:

```text
Book
Source
Page
```

This allows AWS Padho to display source information with the generated answer.

---

# ▶️ Run AWS Padho

Start the chatbot using:

```bash
python main.py
```

You should see:

```text
==================================================
AWS PADHO
Agentic AWS Q&A Assistant
==================================================

Books:
  • AWS Certified Cloud Practitioner — CLF-C02
  • AWS Certified Solutions Architect — SAA-C02

Powered by:
  LangGraph + FAISS + Ollama + Groq

Commands:
  /help
  /source
  /clear
  /exit
==================================================
```

---

# 💬 Example

```text
You: give me the steps for creating an EC2 instance
```

AWS Padho retrieves relevant information from the AWS study material and generates a grounded response.

Example:

```text
AWS Padho:

1. Open the EC2 console.
2. Start the instance launch wizard.
3. Choose an AMI.
4. Select an instance type.
5. Configure instance details.
6. Configure storage.
7. Add tags.
8. Configure the security group.
9. Review and launch.
10. Configure the key pair if required.

Sources:
  • SAA-C02 — Page 60
  • SAA-C02 — Page 65
  • SAA-C02 — Page 73
```

---

# 🖥️ CLI Commands

AWS Padho supports the following commands:

| Command | Description |
|---|---|
| `/help` | Display available commands |
| `/source` | Display source information |
| `/clear` | Clear the current conversation/context |
| `/exit` | Exit the application |

---

# 🔄 RAG Pipeline

AWS Padho follows a standard Retrieval-Augmented Generation pipeline with an agentic control layer.

### Document Processing

```text
AWS PDF
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS
```

### Question Answering

```text
User Question
      ↓
Question Analysis
      ↓
FAISS Retrieval
      ↓
Context Evaluation
      ↓
Query Rewriting if Required
      ↓
Relevant Context
      ↓
Groq LLM
      ↓
Grounded Answer
      ↓
Sources
```

This approach reduces the need for the LLM to rely entirely on its pretrained knowledge and allows answers to be grounded in the provided study material.

---

# ⚡ Streaming Responses

AWS Padho uses LLM streaming so that generated responses can appear progressively in the terminal rather than waiting for the complete response before displaying it.

Conceptually:

```text
User Question
      ↓
Groq
      ↓
Token Stream
      ↓
Terminal
      ↓
Progressive Answer
```

This provides a more interactive CLI experience.

---

# 🛡️ Error Handling

AWS Padho handles common configuration and runtime problems, including:

- Missing PDF files
- Invalid or corrupted PDFs
- Empty PDF documents
- Missing FAISS database
- Corrupted FAISS database
- Ollama not running
- Missing `nomic-embed-text`
- Missing `GROQ_API_KEY`
- Invalid Groq model
- Groq API errors
- Network/API failures
- Retrieval failures
- Empty user questions

The application provides user-friendly error messages instead of exposing unnecessary technical tracebacks during normal operation.

---

# 🔐 Security

Sensitive credentials are stored in environment variables.

The project uses:

```text
.env
```

for local configuration.

The `.env` file should be added to `.gitignore:

```gitignore
.env
venv/
__pycache__/
*.pyc
vector_db/
```

**Never commit your Groq API key to GitHub.**

---

# 🎯 Project Objectives

The main objectives of AWS Padho are:

1. Build a practical **Agentic AI application**.
2. Implement a complete **RAG pipeline**.
3. Use **LangGraph** for agentic workflow orchestration.
4. Implement semantic document retrieval using **FAISS**.
5. Generate embeddings locally using **Ollama**.
6. Integrate an LLM using the **Groq API**.
7. Provide grounded answers based on AWS certification material.
8. Implement source/page tracking for retrieved information.
9. Build a lightweight and interactive **CLI-based AI assistant**.
10. Demonstrate practical understanding of modern **Generative AI architecture**.

---

# 🧪 Example Questions

The chatbot can be used to ask questions such as:

```text
What is Amazon EC2?
```

```text
What is Amazon S3?
```

```text
What is the difference between EBS and S3?
```

```text
Explain IAM roles.
```

```text
What are security groups?
```

```text
How does Auto Scaling work?
```

```text
What is a VPC?
```

```text
Explain the difference between a public and private subnet.
```

```text
What are the different EC2 instance types?
```

The quality and scope of answers depend on the information available in the indexed study material.

---

# 📊 Why Agentic RAG?

A traditional RAG pipeline can be represented as:

```text
Question
   ↓
Retrieve
   ↓
Generate
```

AWS Padho introduces an additional decision-making layer:

```text
Question
   ↓
Analyze
   ↓
Retrieve
   ↓
Evaluate
   ↓
 ┌──────────────────────┐
 │ Is context sufficient?│
 └──────────┬───────────┘
            │
       ┌────┴────┐
       │         │
      YES        NO
       │         │
       ▼         ▼
   Generate   Rewrite
       │         │
       │         ▼
       │      Retrieve
       │         │
       └────┬────┘
            ▼
          Answer
```

This allows the system to make decisions about the retrieval process instead of blindly generating an answer after a single retrieval step.

---

# 📌 Limitations

- The chatbot depends on the quality of the provided study material.
- It is primarily designed for AWS certification learning.
- Answers may not reflect AWS documentation released after the indexed books.
- The application requires Ollama to be available locally for embeddings.
- Groq API access is required for answer generation.
- The current interface is command-line based.

---

# 🚀 Future Improvements

Potential future enhancements include:

- Support for additional AWS certification books
- Hybrid keyword + semantic retrieval
- Improved reranking of retrieved documents
- Conversation memory
- More advanced agent routing
- AWS documentation integration
- Evaluation framework for RAG accuracy
- Retrieval quality metrics
- Automated citation validation
- Multi-document comparison
- Web-based interface
- MCP integration
- Cloud deployment
- Authentication and multi-user support

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

- Agentic AI
- Generative AI
- Retrieval-Augmented Generation
- LangGraph
- LangChain
- Vector databases
- FAISS
- Embeddings
- Ollama
- LLM APIs
- Prompt engineering
- Document processing
- Semantic search
- Python application development
- Environment and API-key management
- CLI application development

---

# 👨‍💻 Author

**Abhishek Mishra**

B.Tech — Computer Science & Engineering (AI & ML)

Vellore Institute of Technology, Bhopal

---

# ⭐ Internship Project

**AWS Padho** was developed as an **AI and Data Science internship task** to demonstrate the practical implementation of an Agentic RAG-based application using modern Generative AI technologies.

The project focuses on applying concepts such as **LLM integration, document retrieval, vector search, embeddings, and agentic workflows** to a practical AWS learning use case.

---

## 📄 License

This project is intended for educational and internship evaluation purposes.

The AWS certification books used as knowledge sources are copyrighted materials and are **not included in this repository**. Users should provide and use legally obtained copies of the relevant books.