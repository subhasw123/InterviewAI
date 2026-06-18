"""
InterviewAI — AI Resume & Interview Assistant
A premium Streamlit UI inspired by ChatGPT, Notion, Linear and Perplexity.

Run:
    pip install streamlit
    streamlit run app.py
"""

import time
from datetime import datetime
from pypdf import PdfReader

import bcrypt


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os
import streamlit as st



os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_vectorstore():
    embeddings = get_embeddings()

    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )


st.set_page_config(
    page_title="InterviewAI — AI Resume & Interview Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────  CUSTOM CSS  ───────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root{
  --bg:#0A0B1E; --bg-2:#0F1130; --panel:rgba(255,255,255,0.04);
  --border:rgba(255,255,255,0.08); --text:#F5F7FF; --muted:#9AA3C7;
  --grad:linear-gradient(135deg,#7C3AED 0%,#3B82F6 50%,#06B6D4 100%);
  --grad-soft:linear-gradient(135deg,rgba(124,58,237,.18),rgba(59,130,246,.12));
}
html, body, [class*="css"], .stApp, .main, section.main{
  font-family:'Inter',system-ui,sans-serif !important; color:var(--text) !important;
}
.stApp{
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(124,58,237,.25), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(59,130,246,.20), transparent 60%),
    radial-gradient(800px 600px at 50% 100%, rgba(6,182,212,.12), transparent 60%),
    linear-gradient(180deg,#07081A 0%, #0A0B1E 60%, #07081A 100%) !important;
}
#MainMenu, footer{visibility:hidden;}
[data-testid="stToolbar"]{display:none !important;}
[data-testid="stDecoration"]{display:none !important;}
[data-testid="stHeader"]{
  background:rgba(7,8,26,0.95) !important;
  border-bottom:1px solid var(--border) !important;
}
/* Force sidebar always open and visible */
section[data-testid="stSidebar"]{
  display:flex !important;
  visibility:visible !important;
  opacity:1 !important;
  min-width:244px !important;
  max-width:320px !important;
  transform:translateX(0) !important;
}
[data-testid="stSidebarContent"]{
  display:flex !important;
  flex-direction:column !important;
  visibility:visible !important;
}
.block-container{padding-top:1.2rem; padding-bottom:6rem; max-width:1400px;}

/* ───── Sidebar ───── */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0B0D24 0%,#0A0B1E 100%) !important;
  border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] *{color:var(--text) !important;}
.sb-logo{display:flex;align-items:center;gap:.65rem;padding:.4rem .2rem 1rem;}
.sb-logo .mark{
  width:36px;height:36px;border-radius:10px;background:var(--grad);
  display:grid;place-items:center;font-weight:800;color:#fff;
  box-shadow:0 8px 24px rgba(124,58,237,.45);
}
.sb-logo .name{font-weight:700;font-size:1.05rem;letter-spacing:.2px;}
.sb-logo .name span{background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.new-chat .stButton>button{
  width:100%; border:1px solid transparent; border-radius:12px; padding:.7rem 1rem;
  background:var(--grad); color:#fff !important; font-weight:600;
  box-shadow:0 10px 30px rgba(59,130,246,.35); transition:.25s ease;
}
.new-chat .stButton>button:hover{transform:translateY(-1px); filter:brightness(1.08);}
.sb-label{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:1rem .25rem .4rem;}

/* ── Category nav buttons — look like sidebar nav items ── */
section[data-testid="stSidebar"] .element-container .stButton>button{
  background:transparent !important;
  border:1px solid transparent !important;
  border-radius:10px !important;
  color:var(--text) !important;
  font-size:.9rem !important;
  font-weight:400 !important;
  text-align:left !important;
  justify-content:flex-start !important;
  padding:.5rem .75rem !important;
  width:100% !important;
  transition:background .2s, border-color .2s, transform .2s !important;
  box-shadow:none !important;
  letter-spacing:.01em !important;
}
section[data-testid="stSidebar"] .element-container .stButton>button:hover{
  background:rgba(255,255,255,.06) !important;
  border-color:rgba(255,255,255,.1) !important;
  transform:translateX(4px) !important;
}
/* New Chat overrides – must be more specific */
section[data-testid="stSidebar"] .new-chat .element-container .stButton>button,
section[data-testid="stSidebar"] .new-chat .stButton>button{
  background:var(--grad) !important;
  border:1px solid transparent !important;
  color:#fff !important;
  font-weight:600 !important;
  text-align:center !important;
  justify-content:center !important;
  box-shadow:0 10px 30px rgba(59,130,246,.35) !important;
  transform:none !important;
}
section[data-testid="stSidebar"] .new-chat .stButton>button:hover{
  filter:brightness(1.1) !important;
  transform:translateY(-1px) !important;
  border-color:transparent !important;
}
.sb-history{font-size:.85rem;color:var(--muted);padding:.45rem .7rem;border-radius:8px;}
.sb-history:hover{background:rgba(255,255,255,.04); color:var(--text);}

/* ───── Hero ───── */
.hero{
  padding:2.2rem 2rem; border-radius:24px; margin-bottom:1.4rem;
  background:var(--grad-soft); border:1px solid var(--border);
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
  position:relative; overflow:hidden;
}
.hero::after{
  content:""; position:absolute; inset:-2px; border-radius:24px; pointer-events:none;
  background:linear-gradient(120deg,rgba(124,58,237,.5),rgba(6,182,212,.4));
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite:xor; mask-composite:exclude; padding:1px; opacity:.35;
}
.hero .badge{
  display:inline-flex;align-items:center;gap:.4rem;padding:.3rem .7rem;border-radius:999px;
  background:rgba(255,255,255,.06);border:1px solid var(--border);font-size:.75rem;color:var(--muted);
  margin-bottom:.9rem;
}
.hero h1{
  font-size:clamp(1.9rem,3.4vw,3rem); font-weight:800; line-height:1.1; margin:0 0 .6rem;
  background:linear-gradient(120deg,#fff 10%, #C9D2FF 50%, #9AB4FF 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.hero p{color:var(--muted); font-size:1.02rem; max-width:720px; margin:0;}

/* ───── Quick cards ───── */
.qa-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:1rem 0 1.6rem;}
@media (max-width:900px){.qa-grid{grid-template-columns:repeat(2,1fr);}}
@media (max-width:560px){.qa-grid{grid-template-columns:1fr;}}
.qa-card{
  padding:1rem 1.1rem; border-radius:18px; border:1px solid var(--border);
  background:rgba(255,255,255,.03); backdrop-filter:blur(12px);
  transition:.25s ease; cursor:pointer; min-height:118px;
}
.qa-card:hover{transform:translateY(-3px); border-color:rgba(124,58,237,.55);
  background:linear-gradient(180deg,rgba(124,58,237,.10),rgba(59,130,246,.06));
  box-shadow:0 18px 40px -18px rgba(124,58,237,.55);}
.qa-card .ico{
  width:36px;height:36px;border-radius:10px;display:grid;place-items:center;font-size:1.1rem;
  background:var(--grad); color:#fff; box-shadow:0 8px 20px rgba(59,130,246,.35); margin-bottom:.6rem;
}
.qa-card h4{margin:0 0 .25rem; font-size:.98rem; font-weight:600;}
.qa-card p{margin:0;color:var(--muted);font-size:.82rem;line-height:1.4;}

/* ───── Chat ───── */
.chat-wrap{display:flex;flex-direction:column;gap:.9rem;margin-top:.4rem;}
.msg-row{display:flex;gap:.7rem;align-items:flex-end;}
.msg-row.user{justify-content:flex-end;}
.avatar{
  width:32px;height:32px;border-radius:10px;display:grid;place-items:center;
  font-size:.85rem;font-weight:700;color:#fff;flex-shrink:0;
}
.avatar.ai{background:var(--grad);box-shadow:0 6px 16px rgba(124,58,237,.45);}
.avatar.user{background:linear-gradient(135deg,#1f2547,#2a3068);border:1px solid var(--border);}
.bubble{
  max-width:72%; padding:.85rem 1rem; border-radius:16px; font-size:.94rem; line-height:1.55;
  border:1px solid var(--border);
}
.bubble.ai{background:rgba(255,255,255,.04); border-top-left-radius:4px;}
.bubble.user{
  background:var(--grad); color:#fff; border:none; border-top-right-radius:4px;
  box-shadow:0 10px 24px -10px rgba(59,130,246,.55);
}
.sources{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.55rem;}
.src{
  font-size:.72rem;color:var(--muted);padding:.3rem .55rem;border-radius:8px;
  background:rgba(255,255,255,.04);border:1px solid var(--border);
}
.src b{color:var(--text);font-weight:600;}
.typing{display:inline-flex;gap:4px;align-items:center;}
.typing span{width:6px;height:6px;border-radius:50%;background:#9AB4FF;animation:blink 1.2s infinite;}
.typing span:nth-child(2){animation-delay:.2s;} .typing span:nth-child(3){animation-delay:.4s;}
@keyframes blink{0%,80%,100%{opacity:.25;transform:translateY(0);}40%{opacity:1;transform:translateY(-2px);}}

/* ───── Right panel ───── */
.panel{
  padding:1.1rem; border-radius:18px; border:1px solid var(--border);
  background:rgba(255,255,255,.03); backdrop-filter:blur(14px); margin-bottom:1rem;
}
.panel h5{margin:0 0 .8rem;font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);}
.stat{display:flex;justify-content:space-between;align-items:baseline;padding:.55rem 0;border-bottom:1px dashed var(--border);}
.stat:last-child{border-bottom:none;}
.stat .v{font-weight:700;font-size:1.15rem;
  background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.stat .l{color:var(--muted);font-size:.85rem;}
.tag{display:inline-block;font-size:.7rem;padding:.25rem .55rem;border-radius:999px;
  background:rgba(124,58,237,.15);border:1px solid rgba(124,58,237,.3);color:#C9B8FF;margin:.15rem .2rem 0 0;}

/* ───── Composer ───── */
.stChatInput, [data-testid="stChatInput"]{
  background:rgba(255,255,255,.04) !important; border:1px solid var(--border) !important;
  border-radius:16px !important; backdrop-filter:blur(14px);
}
.stChatInput textarea{color:var(--text) !important;}

/* ───── Footer ───── */
.footer{
  margin-top:2rem;padding:1rem 1.2rem;border-radius:16px;border:1px solid var(--border);
  background:rgba(255,255,255,.03);display:flex;justify-content:space-between;align-items:center;
  flex-wrap:wrap;gap:.6rem;color:var(--muted);font-size:.82rem;
}
.footer b{color:var(--text);}
.dot{width:6px;height:6px;border-radius:50%;background:#22C55E;display:inline-block;margin-right:.4rem;
  box-shadow:0 0 0 4px rgba(34,197,94,.15);}
</style>
""",
    unsafe_allow_html=True,
)

db = load_vectorstore()
st.sidebar.success(
    f"Knowledge Base Loaded: {len(db.index_to_docstore_id)} docs"
)
llm = get_llm()

def extract_resume_text(uploaded_file):

    pdf_reader = PdfReader(uploaded_file)

    text = ""

    for page in pdf_reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

def register_user(
    name,
    email,
    password,
    desired_role,
    skills
):
    conn = get_connection()
    cursor = conn.cursor()

    hashed_password = bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    )

    query = """
    INSERT INTO users
    (
        name,
        email,
        password,
        desired_role,
        skills
    )
    VALUES (%s,%s,%s,%s,%s)
    """

    cursor.execute(
        query,
        (
            name,
            email,
            hashed_password.decode(),
            desired_role,
            skills
        )
    )

    conn.commit()

    cursor.close()
    conn.close()


def login_user(email, password):

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        "SELECT * FROM users WHERE email=%s",
        (email,)
    )

    user = cursor.fetchone()

    cursor.close()
    conn.close()

    if user:

        if bcrypt.checkpw(
            password.encode(),
            user["password"].encode()
        ):

            return user

    return None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "👋 Hey! I'm **InterviewAI** — your personal mentor for placements. "
                    "Ask me anything about DSA, DBMS, OOP, HR rounds, or upload your resume for a review.",
         "sources": [{"title": "Placement Handbook", "kind": "PDF"},
                     {"title": "DSA Cheatsheet", "kind": "Notes"}]},
    ]
if "asked" not in st.session_state:
    st.session_state.asked = 128

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

if "page" not in st.session_state:
    st.session_state.page = "home"




CATEGORIES = [
    ("📄", "Resume Review", "resume_analyzer"),
    ("💼", "HR Interview",  "chat_hr"),
    ("🧠", "Technical Interview", "chat_technical"),
    ("🧮", "DSA",  "chat_dsa"),
    ("🗄️", "DBMS", "chat_dbms"),
    ("🧩", "OOP",  "chat_oop"),
    ("📊", "Aptitude", "chat_aptitude"),
]

CATEGORY_STARTERS = {
    "chat_hr":        "Give me a common HR interview question and a sample answer.",
    "chat_technical": "Give me a common technical interview question for software engineers.",
    "chat_dsa":       "Explain the Two Sum problem and its optimal solution.",
    "chat_dbms":      "Explain ACID properties in DBMS.",
    "chat_oop":       "Explain the four pillars of Object-Oriented Programming.",
    "chat_aptitude":  "Give me a common aptitude question asked in placement exams with solution.",
}
HISTORY = [
    "Two Sum — optimal approach", "SQL joins explained",
    "Tell me about yourself", "Resume bullet rewrite",
    "Polymorphism vs Overloading", "TCP vs UDP",
]
CATEGORY_QUESTIONS = {
    "DSA": [
        "Explain Two Sum problem",
        "What is Binary Search?",
        "Difference between Array and Linked List",
        "What is Time Complexity?",
        "Explain Dynamic Programming"
    ],

    "DBMS": [
        "What is Normalization?",
        "Difference between SQL and NoSQL",
        "Explain ACID properties",
        "What is Indexing in DBMS?",
        "What are Joins?"
    ],

    "OOP": [
        "What is Inheritance?",
        "Explain Polymorphism",
        "What is Encapsulation?",
        "Difference between Abstract Class and Interface",
        "Explain SOLID Principles"
    ],

    "HR": [
        "Tell me about yourself",
        "Why should we hire you?",
        "What are your strengths?",
        "What are your weaknesses?",
        "Where do you see yourself in 5 years?"
    ],

    "Resume": [
        "How should a fresher resume look?",
        "Best resume format for placements",
        "How to write project descriptions?",
        "What skills should I include?",
        "How to make ATS-friendly resume?"
    ]
}
QUICK = [
    ("📝", "Resume Tips", "ATS-friendly bullets, action verbs, metrics."),
    ("💬", "HR Questions", "Behavioral answers using the STAR method."),
    ("⚡", "DSA Practice", "Pattern-based problems with complexity."),
    ("🗃️", "DBMS Questions", "Normalization, indexes, transactions."),
    ("🧬", "OOP Concepts", "SOLID, design patterns, inheritance."),
    ("🎙️", "Mock Interview", "Live simulated technical round."),
]

# ──────────────────────────────  SIDEBAR  ──────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div class="sb-logo"><div class="mark">ai</div>
        <div class="name">Interview<span>AI</span></div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="new-chat">', unsafe_allow_html=True)
    
    if st.button("✨ New Chat"):


        st.session_state.messages = [
            {
                "role": "assistant",
                "content":
                "👋 Hey! I'm InterviewAI. Ask me anything about Resume, DSA, DBMS, OOP, HR Interviews and Placements."
            }
        ]

        st.rerun()
        
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Categories</div>', unsafe_allow_html=True)
    for ico, name, page_key in CATEGORIES:
        if st.button(f"{ico}  {name}", key=f"cat_{page_key}", use_container_width=True):
            if page_key == "resume_analyzer":
                st.session_state.page = "resume_analyzer"
            else:
                # Pre-load a starter question and go to home/chat
                st.session_state.page = "home"
                st.session_state.selected_question = CATEGORY_STARTERS[page_key]
            st.rerun()
    st.markdown('<div class="sb-label">Chat History</div>', unsafe_allow_html=True)
    st.markdown(
        "".join(f'<div class="sb-history">💭 {h}</div>' for h in HISTORY),
        unsafe_allow_html=True,
    )

# ──────────────────────────────  LAYOUT  ───────────────────────────────────
main, side = st.columns([2.4, 1], gap="large")

if st.session_state.page == "home":
    with main:
        st.markdown(
            """
            <div class="hero">
            <div class="badge">✦ Powered by Gemini · RAG + FAISS + LangChain</div>
            <h1>AI Resume &amp; Interview Assistant</h1>
            <p>Your personal AI mentor for interviews, placements and career growth —
            built for students, freshers and software engineering candidates.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cards = "".join(
            f"""<div class="qa-card"><div class="ico">{i}</div>
            <h4>{t}</h4><p>{d}</p></div>""" for i, t, d in QUICK
        )
        st.markdown(f'<div class="qa-grid">{cards}</div>', unsafe_allow_html=True)

        # Chat transcript
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for m in st.session_state.messages:
            if m["role"] == "user":
                st.markdown(
                    f"""<div class="msg-row user">
                      <div class="bubble user">{m["content"]}</div>
                      <div class="avatar user">U</div></div>""",
                    unsafe_allow_html=True,
                )
            else:
                srcs = ""
                if m.get("sources"):
                    chips = "".join(
                        f'<div class="src"><b>{s["title"]}</b> · {s["kind"]}</div>'
                        for s in m["sources"]
                    )
                    srcs = f'<div class="sources">{chips}</div>'
                st.markdown(
                    f"""<div class="msg-row">
                      <div class="avatar ai">ai</div>
                      <div class="bubble ai">{m["content"]}{srcs}</div></div>""",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

        chat_input = st.chat_input("Ask InterviewAI anything — DSA, HR, resume, OOP…")
        prompt = chat_input

        if st.session_state.selected_question:
            prompt = st.session_state.selected_question
            st.session_state.selected_question = None
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.asked += 1
            placeholder = st.empty()
            placeholder.markdown(
                """<div class="msg-row"><div class="avatar ai">ai</div>
                <div class="bubble ai"><span class="typing">
                <span></span><span></span><span></span></span></div></div>""",
                unsafe_allow_html=True,
            )
            time.sleep(1.0)

            with st.spinner("Searching knowledge base..."):
                retriever = db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                docs = retriever.invoke(prompt)

            if docs:
                context = "\n\n".join([doc.page_content for doc in docs])
            else:
                context = ""

            rag_prompt = f"""
      You are InterviewAI.

      You help users with:
      - Resume Building
      - DSA
      - DBMS
      - OOP
      - HR Interviews
      - Technical Interviews
      - Placement Preparation

      Use ONLY the information provided below.
      If the answer is unavailable, say:
      'I could not find this information in the knowledge base.'

      Context:
      {context}

      Question:
      {prompt}
      """

            start = time.time()

            if context:
                response = llm.invoke(rag_prompt)
                answer = response.content
            else:
                answer = """
        I could not find this information in the knowledge base.

        Try asking:
        • Resume questions
        • DSA questions
        • DBMS questions
        • OOP concepts
        • HR interview questions
        """

            elapsed = round(time.time() - start, 2)
            placeholder.empty()

            sources = []
            for doc in docs:
                source_name = doc.metadata.get("source", "Knowledge Base")
                sources.append({"title": source_name.split("\\")[-1], "kind": "PDF"})

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"{answer}\n\n⏱ Response Time: {elapsed}s",
                "sources": sources
            })
            st.rerun()

elif st.session_state.page == "resume_analyzer":
    with main:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
            
        st.markdown("## 📄 Resume ATS Analyzer")
        st.write("Upload your resume to get an AI-powered ATS analysis, feedback, and improvement suggestions.")

        uploaded_resume = st.file_uploader(
            "Upload Resume (PDF)",
            type=["pdf"]
        )

        if uploaded_resume:
            if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
                with st.spinner("Analyzing Resume..."):
                    resume_text = extract_resume_text(uploaded_resume)

                    ats_prompt = f"""
    You are an expert ATS Resume Analyzer.

    Analyze this resume and provide:

    1. ATS Score out of 100
    2. Resume Summary
    3. Strengths
    4. Weaknesses
    5. Missing Skills
    6. Suggestions for Improvement

    Resume:

    {resume_text}

    Format your response clearly using headings.
    """
                    ats_response = llm.invoke(ats_prompt)

                st.markdown("## 📊 ATS Analysis Report")
                st.markdown(ats_response.content)

with side:

    st.markdown(
        '<div class="panel"><h5>Session Stats</h5>',
        unsafe_allow_html=True
    )

    doc_count = len(db.index_to_docstore_id)

    st.markdown(
        f"""
        <div class="stat">
            <span class="l">Questions Asked</span>
            <span class="v">{st.session_state.asked}</span>
        </div>

        <div class="stat">
            <span class="l">Documents Indexed</span>
            <span class="v">{doc_count}</span>
        </div>

        <div class="stat">
            <span class="l">Model</span>
            <span class="v">Gemini</span>
        </div>

        <div class="stat">
            <span class="l">Vector DB</span>
            <span class="v">FAISS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><h5>Focus Areas</h5>', unsafe_allow_html=True)
    st.markdown(
        '<span class="tag">DSA</span> '
        '<span class="tag">DBMS</span> '
        '<span class="tag">OOP</span> '
        '<span class="tag">HR</span> '
        '<span class="tag">Resume</span> '
        '<span class="tag">Placement</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="panel"><h5>Tip of the Day</h5>'
        '<p style="margin:0;color:var(--muted);font-size:.88rem;line-height:1.55;">'
        'Quantify every resume bullet. Example: '
        '"Improved API latency by 38%" instead of "Improved API performance".'
        '</p></div>',
        unsafe_allow_html=True,
    )