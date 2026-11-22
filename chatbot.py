"""
Groq AI Studio
A premium, fully-functional AI chatbot built with Streamlit and the official Groq SDK.

Run:
    pip install -r requirements.txt
    streamlit run chatbot.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import streamlit as st

# Official Groq Python SDK
from groq import Groq
from groq import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

APP_TITLE = "Groq AI Studio"
APP_TAGLINE = "Premium AI Chatbot powered by Groq"

# Curated list of current models (id -> friendly label).
# Retired model IDs (llama-3.3-70b-versatile, llama-3.1-8b-instant,
# llama3-70b-8192, llama3-8b-8192, mixtral-8x7b-32768, gemma2-9b-it)
# have been removed. Use the custom model ID input for any other model.
AVAILABLE_MODELS: dict[str, str] = {
    "openai/gpt-oss-120b": "OpenAI GPT-OSS 120B",
}

# Predefined AI roles with rich system prompts.
AI_ROLES: dict[str, str] = {
    "General Assistant": (
        "You are Groq AI Studio, a knowledgeable, friendly, and concise general-purpose "
        "assistant. Provide accurate, well-structured answers. Use Markdown formatting, "
        "code blocks with language tags, and lists where helpful. Ask clarifying questions "
        "only when essential."
    ),
    "Python Tutor": (
        "You are an expert Python tutor. Explain concepts clearly with runnable code "
        "examples, best practices, and common pitfalls. Adapt your explanations to the "
        "learner's level. Always use fenced code blocks tagged with `python`."
    ),
    "Machine Learning Expert": (
        "You are a senior machine learning engineer. Give rigorous, practical guidance on "
        "model architecture, training, evaluation, and deployment. Include concise code "
        "snippets (PyTorch / scikit-learn / TensorFlow as appropriate) and justify design "
        "choices with theory where relevant."
    ),
    "AI & Robotics Trainer": (
        "You are an AI and robotics trainer. Cover perception, control, reinforcement "
        "learning, SLAM, and ROS. Provide actionable, safety-conscious advice with code "
        "examples and references to real-world robotics stacks."
    ),
    "Data Analyst": (
        "You are a meticulous data analyst. Help with data cleaning, exploratory analysis, "
        "statistics, SQL, and visualization (pandas, NumPy, matplotlib, seaborn, Plotly). "
        "Prefer reproducible, well-commented code and interpret results in plain language."
    ),
    "Custom System Prompt": "",
}

DEFAULT_ROLE = "General Assistant"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TOP_P = 1.0
DEFAULT_STREAMING = True
DEFAULT_SHOW_HISTORY = True

SUGGESTED_PROMPTS = [
    "Explain quantum computing in simple terms",
    "Write a Python function to detect palindromes",
    "Design a neural network for image classification",
    "Summarize the key ideas behind reinforcement learning",
    "Give me a SQL query to find the top 5 customers by revenue",
]

# --------------------------------------------------------------------------- #
# Premium dark-theme styling
# --------------------------------------------------------------------------- #

CUSTOM_CSS = """
<style>
/* ---- Root / global ---- */
:root {
    --bg-primary: #0b0e14;
    --bg-secondary: #11151f;
    --bg-card: #161b27;
    --bg-elevated: #1c2230;
    --border: #262d3d;
    --text-primary: #e6e9ef;
    --text-secondary: #9aa3b2;
    --text-muted: #6b7280;
    --accent: #f55036;
    --accent-soft: #ff7a5c;
    --accent-glow: rgba(245, 80, 54, 0.35);
    --success: #2ecc71;
    --warning: #f1c40f;
    --danger: #e74c3c;
    --radius: 14px;
    --shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* Subtle accent gradient backdrop */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1200px 600px at 85% -10%, rgba(245, 80, 54, 0.10), transparent 60%),
        radial-gradient(900px 500px at -10% 110%, rgba(46, 204, 113, 0.06), transparent 55%),
        var(--bg-primary);
}

/* ---- Header ---- */
.studio-header {
    background: linear-gradient(135deg, rgba(245,80,54,0.18), rgba(245,80,54,0.04));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: var(--shadow);
}
.studio-header h1 {
    margin: 0;
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #ffffff, #ff7a5c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.studio-header p {
    margin: 6px 0 0;
    color: var(--text-secondary);
    font-size: 0.95rem;
}
.studio-badge {
    display: inline-block;
    margin-top: 12px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(46, 204, 113, 0.12);
    color: var(--success);
    border: 1px solid rgba(46, 204, 113, 0.3);
}
.studio-badge.offline {
    background: rgba(231, 76, 60, 0.12);
    color: var(--danger);
    border-color: rgba(231, 76, 60, 0.3);
}

/* ---- Cards ---- */
.studio-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    box-shadow: var(--shadow);
}
.welcome-card {
    text-align: center;
    padding: 34px 26px;
}
.welcome-card h2 {
    margin: 0 0 8px;
    font-size: 1.4rem;
    font-weight: 700;
}
.welcome-card p {
    color: var(--text-secondary);
    margin: 0 0 22px;
}
.prompt-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-top: 8px;
}
.prompt-chip {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    text-align: left;
    color: var(--text-primary);
    font-size: 0.88rem;
    cursor: pointer;
    transition: all 0.18s ease;
}
.prompt-chip:hover {
    border-color: var(--accent);
    background: rgba(245, 80, 54, 0.08);
    transform: translateY(-1px);
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 18px;
}
.sidebar-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 4px 0 2px;
}
.sidebar-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-bottom: 14px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ---- Streamlit widget theming ---- */
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stTextArea label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stToggle label,
.stTextInput label, .stTextArea label, .stSelectbox label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}

.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    background: rgba(245, 80, 54, 0.10) !important;
}
.stButton > button[kind="primary"], .stButton > button[data-testid*="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-soft)) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 14px var(--accent-glow) !important;
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.08);
}

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25) !important;
}
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #2ecc71, #27ae60) !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-soft)) !important;
}

/* ---- Chat input ---- */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stChatInputTextArea"] {
    color: var(--text-primary) !important;
}

/* ---- Status / alerts ---- */
.stAlert {
    border-radius: 10px !important;
}

/* ---- Metric chips in header ---- */
.metric-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 14px;
}
.metric-pill {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: var(--text-secondary);
}
.metric-pill strong {
    color: var(--text-primary);
}

/* ---- Divider ---- */
.studio-divider {
    height: 1px;
    background: var(--border);
    margin: 16px 0;
    border: none;
}

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb {
    background: #2a3344;
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover { background: #364154; }

/* ---- Footer ---- */
.studio-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.75rem;
    margin-top: 26px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
}
</style>
"""


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

def init_session_state() -> None:
    """Initialise all session-state keys with sensible defaults."""
    defaults: dict[str, Any] = {
        "messages": [],                 # list[dict] — conversation history
        "api_key": "",                  # str — Groq API key (kept only in memory)
        "api_valid": False,             # bool — last validation result
        "selected_role": DEFAULT_ROLE,
        "custom_prompt": "",
        "selected_model": DEFAULT_MODEL,
        "custom_model": "",
        "use_custom_model": False,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
        "streaming": DEFAULT_STREAMING,
        "show_history": DEFAULT_SHOW_HISTORY,
        "request_count": 0,
        "total_tokens": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def get_system_prompt() -> str:
    """Return the active system prompt based on the chosen role."""
    role = st.session_state.selected_role
    if role == "Custom System Prompt":
        return st.session_state.custom_prompt.strip() or (
            "You are a helpful AI assistant."
        )
    return AI_ROLES.get(role, AI_ROLES[DEFAULT_ROLE])


def get_active_model() -> str:
    """Return the model id to send to the API."""
    if st.session_state.use_custom_model and st.session_state.custom_model.strip():
        return st.session_state.custom_model.strip()
    return st.session_state.selected_model


def get_client() -> Groq | None:
    """Build a Groq client from the stored API key. Returns None if no key."""
    key = st.session_state.api_key.strip()
    if not key:
        return None
    return Groq(api_key=key)


def validate_api_connection() -> tuple[bool, str]:
    """
    Validate the API key and the currently selected model by issuing a
    minimal chat completion. Uses the exact same model ID (get_active_model)
    as chat completion, so both paths are guaranteed to agree.
    Returns (success, message). Never echoes the key.
    """
    client = get_client()
    if client is None:
        return False, "Please enter an API key first."
    model_id = get_active_model()
    try:
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            stream=False,
        )
        return True, f"Connection successful — model '{model_id}' is reachable."
    except AuthenticationError:
        return False, "Authentication failed: the API key is invalid or revoked."
    except RateLimitError:
        return False, "Rate limit reached while validating. Try again shortly."
    except (APIConnectionError, APITimeoutError):
        return False, "Network error: could not reach Groq. Check your connection."
    except APIError as exc:
        return False, f"Groq API error: {exc.message if hasattr(exc, 'message') else exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Unexpected error during validation: {exc}"


def build_messages() -> list[dict[str, str]]:
    """Assemble the full message list including the system prompt."""
    system_prompt = get_system_prompt()
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in st.session_state.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})
    return messages


def stream_completion(client: Groq, messages: list[dict[str, str]]) -> Any:
    """Create a streaming chat completion and return the stream iterator."""
    return client.chat.completions.create(
        model=get_active_model(),
        messages=messages,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
        top_p=st.session_state.top_p,
        stream=True,
    )


def non_stream_completion(client: Groq, messages: list[dict[str, str]]) -> str:
    """Create a non-streaming chat completion and return the text."""
    response = client.chat.completions.create(
        model=get_active_model(),
        messages=messages,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens,
        top_p=st.session_state.top_p,
        stream=False,
    )
    return response.choices[0].message.content or ""


def handle_api_error(exc: Exception) -> str:
    """Map Groq SDK exceptions to safe, user-friendly messages (no key leakage)."""
    if isinstance(exc, AuthenticationError):
        return "Authentication failed. Please verify your API key in the sidebar."
    if isinstance(exc, RateLimitError):
        return "Rate limit exceeded. Please wait a moment and try again."
    if isinstance(exc, APITimeoutError):
        return "The request timed out. Try reducing max tokens or retry."
    if isinstance(exc, APIConnectionError):
        return "Connection error: could not reach the Groq API. Check your network."
    if isinstance(exc, APIError):
        msg = getattr(exc, "message", None) or str(exc)
        return f"Groq API error: {msg}"
    return f"An unexpected error occurred: {exc}"


def export_as_text(messages: list[dict[str, str]]) -> str:
    """Render the conversation as human-readable text. Excludes API keys."""
    lines = [
        f"Groq AI Studio — Conversation Export",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {get_active_model()}",
        f"Role: {st.session_state.selected_role}",
        "-" * 60,
        "",
    ]
    for msg in messages:
        speaker = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{speaker}]")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def export_as_json(messages: list[dict[str, str]]) -> str:
    """Render the conversation as JSON. Excludes API keys."""
    payload = {
        "app": APP_TITLE,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "model": get_active_model(),
        "role": st.session_state.selected_role,
        "temperature": st.session_state.temperature,
        "max_tokens": st.session_state.max_tokens,
        "top_p": st.session_state.top_p,
        "messages": messages,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_header() -> None:
    """Render the premium dashboard header."""
    status = "online" if st.session_state.api_valid else "offline"
    badge_class = "" if st.session_state.api_valid else "offline"
    badge_text = "API Connected" if st.session_state.api_valid else "API Not Connected"

    model_label = AVAILABLE_MODELS.get(get_active_model(), get_active_model())
    msg_count = len(st.session_state.messages)
    role = st.session_state.selected_role

    st.markdown(
        f"""
        <div class="studio-header">
            <h1>{APP_TITLE}</h1>
            <p>{APP_TAGLINE}</p>
            <span class="studio-badge {badge_class}">{badge_text}</span>
            <div class="metric-row">
                <span class="metric-pill">Model: <strong>{model_label}</strong></span>
                <span class="metric-pill">Role: <strong>{role}</strong></span>
                <span class="metric-pill">Messages: <strong>{msg_count}</strong></span>
                <span class="metric-pill">Requests: <strong>{st.session_state.request_count}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    """Render the welcome screen with suggested prompts."""
    st.markdown(
        """
        <div class="studio-card welcome-card">
            <h2>Welcome to Groq AI Studio</h2>
            <p>Configure your API key in the sidebar, then start chatting or pick a prompt below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="prompt-grid">', unsafe_allow_html=True)
    cols = st.columns(len(SUGGESTED_PROMPTS))
    for col, prompt in zip(cols, SUGGESTED_PROMPTS):
        with col:
            if st.button(prompt, key=f"prompt_{hash(prompt)}", use_container_width=True):
                st.session_state["pending_prompt"] = prompt
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def render_sidebar() -> None:
    """Render the full configuration sidebar."""
    with st.sidebar:
        st.markdown(
            f'<p class="sidebar-title">{APP_TITLE}</p>'
            '<p class="sidebar-sub">Configuration</p>',
            unsafe_allow_html=True,
        )

        # --- API key ----------------------------------------------------- #
        st.markdown("#### 🔑 API Connection")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="gsk_...",
            help="Your key is stored only in this browser session and never logged.",
        )
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.api_valid = False

        if st.button("Validate Connection", use_container_width=True, type="primary"):
            with st.spinner("Validating…"):
                ok, msg = validate_api_connection()
            st.session_state.api_valid = ok
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        st.markdown('<hr class="studio-divider">', unsafe_allow_html=True)

        # --- Model ------------------------------------------------------- #
        st.markdown("#### 🧠 Model")
        model_choice = st.selectbox(
            "Pretrained Model",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda m: AVAILABLE_MODELS[m],
            index=list(AVAILABLE_MODELS).index(st.session_state.selected_model)
            if st.session_state.selected_model in AVAILABLE_MODELS
            else 0,
        )
        st.session_state.selected_model = model_choice

        use_custom = st.checkbox("Use custom model ID", value=st.session_state.use_custom_model)
        st.session_state.use_custom_model = use_custom
        if use_custom:
            st.session_state.custom_model = st.text_input(
                "Custom Model ID",
                value=st.session_state.custom_model,
                placeholder="e.g. openai/gpt-oss-120b",
            )

        st.markdown('<hr class="studio-divider">', unsafe_allow_html=True)

        # --- Role -------------------------------------------------------- #
        st.markdown("#### 🎭 AI Role")
        role_choice = st.selectbox(
            "Select Role",
            options=list(AI_ROLES.keys()),
            index=list(AI_ROLES).index(st.session_state.selected_role)
            if st.session_state.selected_role in AI_ROLES
            else 0,
        )
        st.session_state.selected_role = role_choice

        if role_choice == "Custom System Prompt":
            st.session_state.custom_prompt = st.text_area(
                "System Prompt",
                value=st.session_state.custom_prompt,
                height=120,
                placeholder="Describe the assistant's behaviour…",
            )

        st.markdown('<hr class="studio-divider">', unsafe_allow_html=True)

        # --- Parameters -------------------------------------------------- #
        st.markdown("#### ⚙️ Parameters")
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(st.session_state.temperature),
            step=0.05,
            help="Lower = focused & deterministic. Higher = creative & varied.",
        )
        st.session_state.max_tokens = st.number_input(
            "Max Response Tokens",
            min_value=256,
            max_value=32768,
            value=int(st.session_state.max_tokens),
            step=256,
            help="Maximum tokens to generate in the response.",
        )
        st.session_state.top_p = st.slider(
            "Top-P (nucleus sampling)",
            min_value=0.1,
            max_value=1.0,
            value=float(st.session_state.top_p),
            step=0.05,
            help="Controls diversity via nucleus sampling.",
        )

        st.markdown('<hr class="studio-divider">', unsafe_allow_html=True)

        # --- Toggles ----------------------------------------------------- #
        st.markdown("#### 🔄 Behaviour")
        st.session_state.streaming = st.toggle(
            "Stream responses",
            value=st.session_state.streaming,
            help="Show tokens in real time as they arrive.",
        )
        st.session_state.show_history = st.toggle(
            "Show conversation history",
            value=st.session_state.show_history,
            help="Render previous messages in the chat window.",
        )

        st.markdown('<hr class="studio-divider">', unsafe_allow_html=True)

        # --- Conversation actions --------------------------------------- #
        st.markdown("#### 💬 Conversation")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🆕 New Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_b:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        if st.button("♻️ Reset Settings", use_container_width=True):
            _reset_settings()

        st.markdown('<hr class="studio-divider">', unsafe_allow_html=True)

        # --- Export ------------------------------------------------------ #
        st.markdown("#### 📤 Export")
        has_messages = len(st.session_state.messages) > 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.download_button(
            "Export as TXT",
            data=export_as_text(st.session_state.messages),
            file_name=f"groq_conversation_{timestamp}.txt",
            mime="text/plain",
            disabled=not has_messages,
            use_container_width=True,
        )
        st.download_button(
            "Export as JSON",
            data=export_as_json(st.session_state.messages),
            file_name=f"groq_conversation_{timestamp}.json",
            mime="application/json",
            disabled=not has_messages,
            use_container_width=True,
        )


def _reset_settings() -> None:
    """Restore all configurable settings to defaults (keeps API key)."""
    st.session_state.selected_role = DEFAULT_ROLE
    st.session_state.custom_prompt = ""
    st.session_state.selected_model = DEFAULT_MODEL
    st.session_state.custom_model = ""
    st.session_state.use_custom_model = False
    st.session_state.temperature = DEFAULT_TEMPERATURE
    st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    st.session_state.top_p = DEFAULT_TOP_P
    st.session_state.streaming = DEFAULT_STREAMING
    st.session_state.show_history = DEFAULT_SHOW_HISTORY
    st.success("Settings reset to defaults.")


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #

def render_chat() -> None:
    """Render the main chat interface and handle message flow."""
    messages = st.session_state.messages

    # Show history if enabled.
    if st.session_state.show_history:
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    else:
        # When history is hidden, only show the most recent exchange.
        if messages:
            last = messages[-1]
            with st.chat_message(last["role"]):
                st.markdown(last["content"])

    # Welcome screen when empty.
    if not messages:
        render_welcome()

    # Handle a suggested-prompt selection.
    pending = st.session_state.pop("pending_prompt", None)

    # Chat input.
    user_input = st.chat_input("Ask anything…")

    if pending and not user_input:
        user_input = pending

    if not user_input:
        return

    # Need a valid API key to proceed.
    if not st.session_state.api_key.strip():
        st.error("Please enter your Groq API key in the sidebar to start chatting.")
        return

    # Append user message and render it.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response.
    client = get_client()
    api_messages = build_messages()

    with st.chat_message("assistant"):
        try:
            if st.session_state.streaming:
                _render_streaming_response(client, api_messages)
            else:
                _render_non_streaming_response(client, api_messages)
        except Exception as exc:  # noqa: BLE001
            error_msg = handle_api_error(exc)
            st.error(error_msg)
            # Remove the user message so the user can retry cleanly.
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()


def _render_streaming_response(client: Groq, api_messages: list[dict[str, str]]) -> None:
    """Stream tokens into a placeholder and persist the full response."""
    placeholder = st.empty()
    full_response = ""
    token_count = 0

    with st.spinner("Thinking…"):
        stream = stream_completion(client, api_messages)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_response += delta.content
                token_count += 1
                placeholder.markdown(full_response + "▌")

    placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.request_count += 1
    st.session_state.total_tokens += token_count


def _render_non_streaming_response(client: Groq, api_messages: list[dict[str, str]]) -> None:
    """Fetch a complete response and render it."""
    with st.spinner("Generating response…"):
        response_text = non_stream_completion(client, api_messages)

    st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.request_count += 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    init_session_state()
    render_sidebar()
    render_header()
    render_chat()

    st.markdown(
        f'<div class="studio-footer">'
        f'{APP_TITLE} · Built with Streamlit & the official Groq SDK · '
        f"Your API key never leaves this browser session."
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()