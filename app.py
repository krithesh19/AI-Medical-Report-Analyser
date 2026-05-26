# ===========================================
# Streamlit app: upload, chat, purge
# ===========================================
import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from agent import create_agent


from ingest import ingest_helpbook_pdf, ingest_patient_files
from vectorstore import (
    ensure_indexes,
    drop_patient_index,
)

import time
from rag import get_last_context, get_last_metrics
from metrics import summarize_session, append_session_summary

load_dotenv()

# Define tab name and logo 
st.set_page_config(page_title="Medical Report Assistant", page_icon="🧠",layout="wide")

# Design the webpage
st.markdown("""
<style>
:root{
  --primary:#475569;   /* slate */
  --border:#E5E7EB;
  --text:#0F172A;
  --panel:#F8FAFC;
  --gap-main:0.9rem;   /* tune these if needed */
  --gap-side:0.8rem;
}

/* Header + main container */
header[data-testid="stHeader"]{ padding: 6px 0 !important; background: transparent; }
.appview-container .main .block-container{
  max-width: 1080px;
  padding-top: .6rem !important;      /* reduced but not cramped */
  padding-bottom: 1.2rem !important;
}

/* Global block gaps (default is too large) */
div[data-testid="stVerticalBlock"]{ gap: var(--gap-main) !important; }
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{ gap: var(--gap-side) !important; }
div.element-container{ margin-bottom: .55rem !important; }  /* per element spacing */

/* Sidebar: remove extra “Setup” top whitespace, but keep a little padding */
section[data-testid="stSidebar"] > div{
  width: 330px;
  padding: .3rem .8rem 1rem .8rem !important;
}
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]{ padding-top: 0 !important; }
section[data-testid="stSidebar"] h2{ margin-top: .25rem !important; }

/* Title & subheaders (smaller, not tiny) */
h1, .stMarkdown h1{
  color: var(--text);
  font-weight: 800;
  font-size: clamp(22px, 2.2vw, 28px);
  line-height: 1.15;
  margin: .35rem 0 .55rem 0 !important;
}
h2, .stMarkdown h2{
  color: var(--text);
  border-bottom: 1px solid var(--border);
  padding-bottom: .2rem;
  margin-top: .6rem !important;
  margin-bottom: .4rem !important;
}

/* Info alerts: slimmer but readable */
div[role="alert"]{
  padding: .6rem .85rem !important;
  border-radius: 10px !important;
  margin: .35rem 0 !important;
}

/* Buttons */
.stButton>button{
  border: 1px solid var(--border);
  background: var(--primary);
  color: #fff;
  border-radius: 10px;
  padding: .55rem .9rem;
  font-weight: 600;
  box-shadow: 0 2px 6px rgba(15,23,42,.06);
}

/* Uploader */
.stFileUploader{
  border:1px solid var(--border);
  border-radius:12px;
  background:#fff;
  padding:.45rem .6rem;
}

/* Chat bubbles */
[data-testid="stChatMessage"]{
  border:1px solid var(--border);
  border-radius: 12px;
  background:#fff;
  padding:.55rem .8rem;
  margin-bottom:.6rem;
  box-shadow: 0 2px 6px rgba(15,23,42,.04);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"]):nth-of-type(odd){
  background: var(--panel);
}

/* Chat input */
[data-testid="stChatInput"] textarea{
  border-radius: 10px !important;
  border:1px solid var(--border);
  background:#fff;
}

/* Keep base font normal (no clumping) */
html, body, [class*="css"]{ font-size: 16px; }
</style>
""", unsafe_allow_html=True)


# Page title
st.title(" Medical Report Analyser - Get instant report summaries")

# Session bootstrap
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant","content": "..."}]

if "turn_log" not in st.session_state:
    st.session_state.turn_log = []  # list of {"q","answer","context","ts"}

# NEW: gate chat until patient docs are embedded
if "patient_ingested" not in st.session_state:
    st.session_state.patient_ingested = False


# App constants
GENERAL_INDEX_NAME = os.getenv("GENERAL_INDEX_NAME", "medical-helpbook") # Pinecone index name 1
PATIENT_INDEX_NAME = os.getenv("PATIENT_INDEX_NAME", "patient-reports") # Pinecone index name 2
region = os.getenv("PINECONE_REGION", "us-east-1")

# Create agent for each session
if "agent" not in st.session_state or st.session_state.get("_agent_session_id") != st.session_state.session_id:
    st.session_state.agent, st.session_state.agent_memory = create_agent(
        general_index=GENERAL_INDEX_NAME,
        patient_index=PATIENT_INDEX_NAME,
        session_id=st.session_state.session_id,
    )
    st.session_state._agent_session_id = st.session_state.session_id

# Sidebar: File Uploads and session reset
with st.sidebar:

    agent, agent_memory = create_agent(
    general_index=GENERAL_INDEX_NAME,
    patient_index=PATIENT_INDEX_NAME,
    session_id=st.session_state.session_id,
)
    # Tabs in sidebar
    patientReports,GeneralDocument=st.tabs(["Patient reports","General Helper documents"])
    # Tab 1: To upload and process patient report
    with patientReports:
        st.header("Upload Patient Report")
        files = st.file_uploader(
            "Upload a patient document (PDF/TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="patient_files",
        )
        # Process file control
        if st.button("Process File",key="patient report file"):
            if not files:
                st.warning("Please upload at least one patient file.")
            else:
                try:
                    # Ensure index and ingest files into pinecone
                    ensure_indexes(GENERAL_INDEX_NAME, PATIENT_INDEX_NAME, region=region)
                    count = ingest_patient_files(
                        files,
                        PATIENT_INDEX_NAME,
                        session_id=st.session_state.session_id,
                    )
                    if count > 0:
                        st.session_state.patient_ingested = True  # allow chat

                        # Reset chat + agent memory so next turn uses the fresh docs
                        st.session_state.messages = []
                        try:
                            agent_memory.clear()  # local variable from create_agent(...)
                        except Exception:
                            pass

                        st.success(f"Embedded {count} chunks for this session. Chat is now enabled.")
                        st.rerun()  # refresh UI immediately
                    else:
                        st.warning("No content was ingested. Please check the files and try again.")

                except Exception as e:
                    st.error(f"Failed to embed patient files: {e}")
    # Tab 1: To upload and process general helper documents
    with GeneralDocument:
        st.header("Upload Helpbook")
        help_pdf = st.file_uploader("Upload helpbook PDF", type=["pdf"], key="helpbook_pdf")

        # File process handle
        if st.button("Process file",key="Process general report"):
            if not help_pdf:
                st.warning("Please upload a helpbook PDF first.")
            else:
                try:
                    # Ensure index and ingest documents
                    ensure_indexes(GENERAL_INDEX_NAME, PATIENT_INDEX_NAME, region=region)
                    pages = ingest_helpbook_pdf(help_pdf, GENERAL_INDEX_NAME)
                    st.success(f"Embedded {pages} pages into '{GENERAL_INDEX_NAME}'.")
                except Exception as e:
                    st.error(f"Failed to embed helpbook: {e}")
    st.markdown("---")
    if st.button("Process new report"):
        try:
            # Extract performance metrics after every run      
            summary = summarize_session(st.session_state.turn_log)
            append_session_summary(
                csv_path="session_metrics.csv",
                session_id=st.session_state.session_id,
                summary=summary,
                extra={"patient_index": PATIENT_INDEX_NAME, "general_index": GENERAL_INDEX_NAME},
            )  

            # Refresh patient index and refresh session    
            drop_patient_index(PATIENT_INDEX_NAME)
            ensure_indexes(GENERAL_INDEX_NAME, PATIENT_INDEX_NAME, region=region)

            # Clear chat + agent memory, rotate session, and lock chat until new upload
            st.session_state.messages = []
            try:
                agent_memory.clear()  # clear the agent's ConversationBufferMemory
            except Exception:
                pass

            st.session_state.turn_log = []       
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.patient_ingested = False  # ⬅️ gate chat again

            st.success(f"Patient index '{PATIENT_INDEX_NAME}' deleted and recreated. New conversation started.")
            st.info(" All patient data has been permanently deleted.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to reset patient index: {e}")

# Main page with chat window
with st.expander("About the app.."):  
    st.info(
        "Upload a patient PDF/TXT to embed it in Pinecone for this session, then chat with an agent that uses RAG to answer from your report (and an optional helpbook). It remembers the conversation, cites its sources, and you can reset or delete the patient index anytime. For educational purpose only and not medical advice.",
        icon="ℹ️",
    )
st.subheader("Chat")

# Block chat until patient docs are ingested
if not st.session_state.patient_ingested:
    st.info("Please upload and embed patient reports first (see sidebar). Chat will unlock afterwards.")
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Ask about the patient's condition...")
    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run the agent to get the response to user query
                    response = st.session_state.agent.run(user_msg)
                    ctx = ""
                    try: ctx = get_last_context()
                    except: pass
                    m = {}
                    try: m = get_last_metrics() or {}
                    except: pass

                    st.session_state.turn_log.append({
                        "q": user_msg,
                        "answer": response,
                        "context": ctx,
                        "metrics": m,
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
                except Exception as e:
                    response = f"Sorry, something went wrong: {e}"
            # Display the response
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Footer
st.markdown("---")
st.markdown("<center>Built by <b>Kritheshvar Vinothkumar</b> & Sushmitha B | MSc Data & Computational Science, UCD 2025 | For educational purposes only</center>", unsafe_allow_html=True)

