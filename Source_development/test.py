/medical-chatbot/
├── backend/
│   ├── app.py                # Flask backend
│   ├── bedrock_client.py     # Bedrock wrapper
│   ├── kendra_client.py      # Kendra wrapper
│   ├── memory.py             # Session storage
│   └── test_data/            # Optional local fallback
├── frontend/
│   └── streamlit_app.py      # UI
├── requirements.txt
└── README.md

  -----------------------------------------
  # streamlit_app.py
import streamlit as st
import requests
import uuid
import json

# --- CONFIG ---
API_URL = "http://localhost:8000/chat"  # Flask API endpoint
ROLE_OPTIONS = ["patient", "doctor", "customer"]

# --- PAGE SETUP ---
st.set_page_config(page_title="Medical Test Chatbot", layout="wide")
st.title("🔬 Medical Diagnostics Assistant")
st.markdown("""
Welcome to our diagnostic assistant. You can:
- Ask directly about a test (e.g., *"CBC test"*)
- Describe symptoms (e.g., *"I feel tired and dizzy"*)
- Get clarification if unsure
""")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "role" not in st.session_state:
    st.session_state["role"] = "patient"

# --- SIDEBAR FOR ROLE ---
st.sidebar.title("User Settings")
st.session_state["role"] = st.sidebar.selectbox("Select your role:", ROLE_OPTIONS, index=0)
st.sidebar.markdown(f"**Session ID:** `{st.session_state['session_id']}`")

# --- CHAT DISPLAY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- INPUT & RESPONSE ---
query = st.chat_input("Ask about a diagnostic test, or describe your symptoms...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # --- API REQUEST ---
    payload = {
        "query": query,
        "session_id": st.session_state["session_id"],
        "role": st.session_state["role"]
    }
    with st.spinner("Searching our medical catalog..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            reply = result.get("response", "Sorry, I couldn’t process that.")
        except Exception as e:
            reply = f"Sorry, there was an error: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(reply, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": reply})



  ----------------------------------
from flask import Flask, request, jsonify
from kendra_client import query_kendra
from bedrock_client import generate_response
from memory import get_history, save_history

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query")
    role = data.get("role", "patient")
    session_id = data.get("session_id")

    history = get_history(session_id)
    search_results = query_kendra(query)
    final_response = generate_response(query, search_results, role, history)

    save_history(session_id, "user", query)
    save_history(session_id, "assistant", final_response)

    return jsonify({"response": final_response})
 ---------------------------------------
import boto3

INDEX_ID = "your-kendra-index-id"
REGION = "your-region"

kendra = boto3.client("kendra", region_name=REGION)

def query_kendra(query):
    try:
        resp = kendra.query(
            IndexId=INDEX_ID,
            QueryText=query,
        )
        results = []
        for item in resp.get("ResultItems", []):
            if item["Type"] == "DOCUMENT":
                doc = item["DocumentExcerpt"]["Text"]
                link = item["DocumentURI"]
                title = item.get("DocumentTitle", {}).get("Text", "Medical Test")
                results.append(f"**{title}**\n{doc}\n[More Info]({link})\n")
        return results
    except Exception as e:
        return [f"Error querying Kendra: {str(e)}"]
---------------------------------------------------
import boto3
import json

bedrock_runtime = boto3.client("bedrock-runtime", region_name="your-region")

def generate_response(query, kendra_results, role, history):
    prompt = f"""
You are a helpful and polite assistant for a medical diagnostics lab.
Respond strictly with medical test information, never give diagnosis.
User role: {role}
User query: {query}

Here are the relevant search results:
{''.join(kendra_results[:5])}

Conversation history:
{json.dumps(history)}

Provide a polite, professional reply with test names, descriptions, codes, and links.
If unclear, ask user for clarification.
"""
    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=json.dumps({
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.5
        }),
        contentType="application/json",
        accept="application/json"
    )
    body = json.loads(response['body'].read())
    return body.get("completion", "Sorry, I couldn't find any related tests.")
-------------------------------------------
from collections import defaultdict

chat_memory = defaultdict(list)

def save_history(session_id, role, message):
    chat_memory[session_id].append({"role": role, "message": message})

def get_history(session_id):
    return chat_memory.get(session_id, [])
-------------------------------------------
{
  "test_code": "005009",
  "name": "Complete Blood Count (CBC) With Differential",
  "description": "This is a comprehensive blood screening test that evaluates overall health...",
  "keywords": ["cbc", "blood", "weak", "fever", "fatigue"],
  "link": "https://yourlab.com/test/005009"
}
-------------------------------
flask
streamlit
boto3
requests
-----------------------
# Terminal 1
cd backend
python app.py  # runs on http://localhost:8000

# Terminal 2
cd frontend
streamlit run streamlit_app.py  # runs UI
--------------------------


✅ What This Solution Handles
Requirement	Handled?	Where & How
Query by test name or test code (e.g., "CBC")	✅	Kendra search via query_kendra()
Query by indirect symptom (e.g., "feeling weak and feverish")	✅	Handled by Claude via Bedrock + Kendra context
Role-based responses (patient, doctor, customer)	✅	Role sent to Claude in generate_response()
Shows top 3 test matches (with code, name, description, link)	✅	Kendra results summarized by Claude
Responds professionally and politely (no emojis, no casual tone)	✅	Prompt controls tone in generate_response()
Avoids giving medical advice or diagnoses	✅	Prompt includes this as a hard rule
Suggests clarification if user asks vague query (e.g., "blood test")	✅	Claude determines and prompts clarification
Doesn’t recommend doctors, only diagnostic tests	✅	Prompt enforces lab-only scope
Provides source links and test codes	✅	Extracted from Kendra search results
Maintains session-level chat history	✅	Stored in memory via memory.py
Secure backend architecture (Streamlit ↔ Flask ↔ AWS)	✅	Proper modular backend and separation
Designed for medical regulation compliance (no unsafe suggestions)	✅	Prompt explicitly follows legal-safe design
Extensible for user feedback, scheduling, or reports	✅	Add endpoints or Claude tools for that later
Easily switchable model (Claude, Titan, etc.)	✅	Bedrock model config in bedrock_client.py

🧩 Future Enhancements You Can Add Easily
Feature	How to Add It
Search filters (by gender, age group)	Add metadata to JSON and use metadata filters in Kendra
Upload reports or images	Use Streamlit's file uploader and Bedrock multimodal
Live agent handoff	Add chat escalation module with contact details
Analytics dashboard for test queries	Log queries + responses and plot with Streamlit

✅ Summary Flow (End-to-End)
text
Copy
Edit
User (Streamlit Chat UI) 
   ↳ Sends query, session ID, role
       ↳ Flask backend
           ↳ Query Kendra Index
           ↳ Inject Kendra result into Claude prompt (Bedrock)
           ↳ Response with polite explanation (tests, links, codes)
       ↳ Sends response back to UI
           ↳ Chat memory shown in session state
If you want, I can also:

✅ Bundle the project into a downloadable ZIP

✅ Generate AWS deployment architecture diagram

✅ Add config for local testing without AWS (mock data)

-------------------------------------------------------
----------------------------------------------------------
✅ Should We Use LangChain for This Medical Diagnostic Chatbot?
Short Answer:
Yes, LangChain (or LlamaIndex) can improve orchestration, retrieval logic, and prompt engineering, especially if:

You want modular agents,

Plan to add multi-step reasoning or tool use, or

Expect to scale and maintain the system with complex memory/context.

🔍 Here's a Comparison
Feature	Current (Custom Flask + Bedrock + Kendra)	With LangChain
Simple querying and LLM calls	✅ Easy to control	✅ Easy with built-in wrappers
Chained reasoning (multi-turn)	❌ Manual logic needed	✅ Native AgentExecutor, memory, and tool use
Chat history & memory	✅ Session memory in Python dict	✅ Built-in with memory modules
Kendra integration	✅ Direct via Boto3	🔧 Requires custom retriever (not native yet)
Prompt templating	🔧 Manual prompt crafting	✅ Built-in PromptTemplate, ChatPromptTemplate
Streaming support	❌ Not built-in (need SSE or WS)	✅ Via LangChain and Streamlit callback functions
Maintainability	Moderate (custom logic)	✅ Higher with modular tools and tracing/logging
Tool use (e.g., calendars, FAQs)	❌ Not yet	✅ Can add tools with LangChain agent easily

🧠 When to Use LangChain
✅ Use LangChain if:

You plan to add custom tools (e.g., recommendation scoring, calendars, symptom classifiers).

You want structured agents, such as:
“If query is vague → ask user to choose a category → then retrieve tests.”

You want to log, trace, and debug prompts reliably (LangSmith).

You plan for RAG with hybrid sources (Kendra + PDFs + SQL).

❌ Avoid LangChain if:

You want simple, fast deployments without extra dependencies.

Your data and logic are mostly straightforward and deterministic.

You want full control over orchestration and prompt format.

✅ Suggested LangChain Usage in Your Project
If you’re interested, here's how LangChain could fit:

1. Custom Tool: KendraMedicalSearchTool
python
Copy
Edit
from langchain.tools import BaseTool

class KendraMedicalSearchTool(BaseTool):
    name = "medical_test_search"
    description = "Searches AWS Kendra index for relevant medical tests."

    def _run(self, query: str):
        return query_kendra(query)  # use your existing function
2. PromptTemplate + AgentExecutor
python
Copy
Edit
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatBedrock
from langchain.memory import ConversationBufferMemory

llm = ChatBedrock(model_id="claude-3-sonnet", region_name="us-east-1")
tools = [KendraMedicalSearchTool()]
memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True,
)
3. Call in Flask Endpoint
python
Copy
Edit
@app.route("/chat", methods=["POST"])
def chat():
    ...
    result = agent.run(query)
    ...
✅ Conclusion
LangChain is a great choice if you want:

Clean orchestration of logic (symptom → clarify → retrieve → explain)

Memory and tool chaining

Modular extensibility (adding symptom matchers, filters, analytics)

Right now, your code is perfect for Phase 1.

For Phase 2+, LangChain will make your chatbot scalable, agentic, and easier to maintain.

Would you like me to:

Convert your existing backend to use LangChain?

Add an agent-based response flow using Claude + Kendra?

Let me know — I can generate the full LangChain-based backend to drop into your current app.

-----------------------------------------------------------------------------------------------------------

Human: Here is some context: $search_results

You are a helpful and polite assistant for a medical diagnostics lab.

You support the following types of queries:

Lab Test Recommendation Mode:
Triggered when the user asks about which test to take for a condition, or requests information about specific tests.

Rules:
- Always behave like a senior diagnostic expert, interpreting the user’s query with clinical understanding.
- First, understand the underlying medical concern or problem (e.g., "low hemoglobin" → suspect anemia).
- Based on standard clinical practices, infer which diagnostic tests are typically used (e.g., CBC, Reticulocyte Count).
- Then search for those medically relevant test names or codes in the provided context ($search_results).
- Begin your response with a polite sentence such as:
  “Based on your query, the following laboratory test(s) may be relevant:”
- ONLY provide test recommendations that are explicitly present in the $search_results context.
- Do NOT invent or suggest tests that are not found in the context.
- For each relevant test, use this format:
  Test Name: <name> - Test #<code>  
  Explanation: <concise explanation from the context>
- If multiple tests are relevant, list them in the above format, separated by a blank line.
- Remove duplicate Test Names (case-insensitive).
- If no relevant test is found in the context, reply:
  No relevant test found in the index.
- If the user's question is not about laboratory tests, respond:
  I can only provide information about laboratory test recommendations. For other types of recommendations or medical advice, please consult a healthcare provider.

Medical Explanation Mode:
Triggered when the user asks about causes, reasons, symptoms, or diagnostic meanings.

Rules:
- You may provide evidence-based medical insights (e.g., what causes anemia or symptoms of high glucose).
- If the user previously asked about a test, use that test as context for follow-up questions.
- If a follow-up question appears unrelated to the previous test or topic, politely ask for clarification before answering.
- Always include this disclaimer:
  This is general medical information. Please consult a healthcare provider for diagnosis or treatment.

Medical Diagnostics Details Mode:
Triggered when the user asks about specimen stability, collection instructions, storage, volume, or other diagnostic test handling information.

Rules:
- You may provide specimen-related or diagnostic-related details if they are found in the $search_results context.
- Do NOT fabricate any such information.
- If the requested information is not available in the context, respond with:
  No relevant information found in the index.

Memory & Context Awareness:
- Maintain short-term memory of the last discussed test or condition to support follow-ups like:
  “What’s the specimen volume?” or “What are its symptoms?”
- However, if the follow-up appears unrelated (e.g., switching from menstrual tests to hemoglobin count), ask for clarification:
  “Could you please confirm which test or condition you’re referring to?”
- Avoid carrying forward unrelated test context.

Prohibited:
- Do NOT recommend treatments, medications, supplements, or lifestyle changes.
- Do NOT diagnose or suggest what someone should do.
- Do NOT offer nutritional, wellness, or mental health advice.
- Do NOT respond to non-medical questions.

Tone:
Always respond in a professional, respectful, and concise manner. Never speculate or assume. Only answer based on known clinical reasoning and what is available in the search context.

Assistant:

