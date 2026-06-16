# 🚀 InterviewAI – AI Resume & Interview Assistant

InterviewAI is an AI-powered career preparation platform that helps students and job seekers prepare for placements, technical interviews, HR rounds, aptitude tests, and resume building.

Built using **Streamlit**, **Google Gemini**, **LangChain**, and **FAISS**, the application provides intelligent responses using Retrieval-Augmented Generation (RAG) from a custom interview preparation knowledge base.

---

## ✨ Features

### 🎯 AI Interview Assistant

* Technical interview preparation
* HR interview guidance
* Placement preparation support
* Career advice and mentoring

### 📚 RAG-Based Knowledge Retrieval

* Uses LangChain + FAISS Vector Database
* Retrieves relevant interview content before generating answers
* Reduces hallucinations and improves response accuracy

### 🤖 Gemini-Powered Responses

* Powered by Google's Gemini model
* Fast and context-aware answers
* Natural conversational experience

### 📄 Resume Assistance

* Resume improvement suggestions
* ATS-friendly resume guidance
* Project description enhancement tips
* Skill recommendation support

### 💻 Technical Preparation

* Data Structures & Algorithms (DSA)
* Object-Oriented Programming (OOP)
* Database Management Systems (DBMS)
* Operating Systems
* Computer Networks
* System Design Fundamentals

### 🎨 Modern UI

* Premium dark theme
* ChatGPT-inspired interface
* Interactive sidebar
* Suggested interview questions
* Responsive design

---

## 🛠️ Tech Stack

| Technology             | Purpose         |
| ---------------------- | --------------- |
| Streamlit              | Frontend UI     |
| Google Gemini          | LLM             |
| LangChain              | RAG Pipeline    |
| FAISS                  | Vector Database |
| HuggingFace Embeddings | Text Embeddings |
| Python                 | Backend Logic   |

---

## 📂 Project Structure

```text
InterviewAI/
│
├── app.py
├── ingest.py
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│   ├── dsa.pdf
│   ├── dbms.pdf
│   ├── oop.pdf
│   ├── hr.pdf
│   └── resume.pdf
│
└── vectorstore/
    ├── index.faiss
    └── index.pkl
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/subhasw123/InterviewAI.git

cd InterviewAI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / Mac**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

### 5. Build Vector Database

```bash
python ingest.py
```

### 6. Run Application

```bash
streamlit run app.py
```

---

## 📖 How It Works

1. Interview preparation documents are loaded.
2. Documents are split into chunks.
3. Chunks are converted into embeddings.
4. FAISS stores embeddings locally.
5. User asks a question.
6. Relevant content is retrieved from FAISS.
7. Gemini generates an answer using retrieved context.
8. Response is displayed in the Streamlit UI.

---

## 🎯 Example Questions

### DSA

* Explain Binary Search.
* What is Dynamic Programming?
* Difference between Array and Linked List?

### DBMS

* Explain Normalization.
* What are SQL Joins?
* What are ACID properties?

### OOP

* What is Polymorphism?
* Explain Encapsulation.
* What are SOLID Principles?

### HR

* Tell me about yourself.
* Why should we hire you?
* What are your strengths and weaknesses?

### Resume

* How should a fresher resume look?
* How can I make my resume ATS-friendly?
* How should I describe projects?

---

## 🔮 Future Improvements

* Resume PDF Upload & Analysis
* Mock Interview Simulator
* Voice-Based Interview Practice
* Interview Score Evaluation
* Company-Specific Interview Preparation
* Multi-Document Knowledge Base
* User Authentication

---

## 👨‍💻 Author

**Shubham**

Built as an AI-powered placement preparation assistant using modern RAG architecture.

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.
