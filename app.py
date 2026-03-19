import json
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import requests
import streamlit as st

st.set_page_config(page_title="My AI Chat", layout="wide")

st.title("My AI Chat")
st.caption("Ask a question and get a reply powered by Hugging Face Inference.")

# Load token from Streamlit secrets.
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    hf_token = ""

if not hf_token:
    st.error(
        "Hugging Face token is missing. Add it to .streamlit/secrets.toml as HF_TOKEN."
    )
    st.stop()

model_id = "katanemo/Arch-Router-1.5B:hf-inference"
api_url = "https://router.huggingface.co/v1/chat/completions"
STREAM_HEADERS = {
    "Authorization": f"Bearer {hf_token}",
    "Accept": "text/event-stream",
}
JSON_HEADERS = {
    "Authorization": f"Bearer {hf_token}",
    "Accept": "application/json",
}


def _new_chat():
    return {
        "id": str(uuid4()),
        "title": "New chat",
        "created_at": datetime.now().isoformat(timespec="minutes"),
        "messages": [],
    }


BASE_DIR = Path(__file__).resolve().parent
CHATS_DIR = BASE_DIR / "chats"
MEMORY_FILE = BASE_DIR / "memory.json"


def _chat_path(chat_id):
    return CHATS_DIR / f"{chat_id}.json"


def _save_chat(chat):
    CHATS_DIR.mkdir(exist_ok=True)
    payload = {
        "id": chat["id"],
        "title": chat.get("title", "New chat"),
        "created_at": chat.get("created_at", datetime.now().isoformat(timespec="minutes")),
        "messages": chat.get("messages", []),
    }
    _chat_path(chat["id"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_chats():
    if not CHATS_DIR.exists():
        return []
    chats = []
    for path in sorted(CHATS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        chat_id = data.get("id") or path.stem
        title = data.get("title") or "New chat"
        created_at = data.get("created_at")
        if not created_at:
            created_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="minutes")
        messages = data.get("messages")
        if not isinstance(messages, list):
            messages = []
        chats.append(
            {
                "id": chat_id,
                "title": title,
                "created_at": created_at,
                "messages": messages,
            }
        )
    return chats


def _load_memory():
    if not MEMORY_FILE.exists():
        return {}
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data.get("traits", data)
    return {}


def _save_memory(traits):
    payload = {
        "traits": traits,
        "updated_at": datetime.now().isoformat(timespec="minutes"),
    }
    MEMORY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _memory_system_prompt(traits):
    if not traits:
        return None
    lines = []
    for key, value in traits.items():
        if isinstance(value, list):
            cleaned = [str(v) for v in value if str(v).strip()]
            if cleaned:
                lines.append(f"- {key}: {', '.join(cleaned)}")
        elif value:
            lines.append(f"- {key}: {value}")
    if not lines:
        return None
    return (
        "You are a helpful assistant. Use the following user memory to personalize "
        "responses when relevant. If it is unrelated, ignore it.\n"
        + "\n".join(lines)
    )


def _messages_with_memory(history, traits):
    system_prompt = _memory_system_prompt(traits)
    if system_prompt:
        return [{"role": "system", "content": system_prompt}] + history
    return history


def _build_api_error_message(resp):
    msg = f"Hugging Face API error ({resp.status_code})."
    if resp.status_code in (401, 403):
        msg += " Invalid or missing token."
    elif resp.status_code == 410:
        msg += " Legacy endpoint is deprecated. Use router.huggingface.co."
    elif resp.status_code == 404:
        msg += " Model or endpoint not found."
    elif resp.status_code == 429:
        msg += " Rate limit exceeded."
    elif resp.status_code >= 500:
        msg += " Server error from Hugging Face."

    # Try to extract error details from the response body.
    try:
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            msg += f" Details: {data['error']}"
    except json.JSONDecodeError:
        pass
    return msg


def _extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _split_items(text):
    parts = []
    for chunk in text.replace(" and ", ", ").split(","):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts


def _looks_like_list(text):
    if "," in text or " and " in text:
        return True
    return False


def _infer_followup_category(last_assistant_message):
    if not last_assistant_message:
        return None
    text = last_assistant_message.lower()
    if "color" in text:
        return "favorite_colors"
    if "music" in text:
        return "music"
    if ("genre" in text or "genres" in text) and ("book" in text or "read" in text or "reading" in text):
        return "book_genres"
    if "author" in text:
        return "favorite_authors"
    if "hobbies" in text or "hobby" in text:
        return "hobbies"
    if "movie" in text or "film" in text:
        return "movie_genres"
    if "language" in text:
        return "preferred_language"
    if "communication style" in text or "tone" in text:
        return "communication_style"
    if "name" in text:
        return "name"
    return None


def _is_request_like(text):
    lower = text.lower().strip()
    if lower.endswith("?"):
        return True
    for token in [
        "ask me",
        "ask about",
        "ask questions",
        "questions",
        "question",
        "can you",
        "could you",
        "would you",
        "please",
        "tell me",
        "you did not",
        "you didn't",
        "do you",
        "did you",
    ]:
        if token in lower:
            return True
    return False


def _is_preference_question(text):
    lower = text.lower()
    if "like to know" in lower or "would you like to know" in lower:
        return False
    keywords = [
        "preferences",
        "preference",
        "favorite",
        "favourite",
        "what do you like",
        "what do you enjoy",
        "what kind of",
        "which hobbies",
        "what type of music",
        "what colors",
        "what colours",
        "book genre",
        "authors",
        "dislike",
        "avoid",
        "not a fan",
    ]
    return any(k in lower for k in keywords)


def _is_explicit_preference_statement(text):
    lower = text.lower()
    explicit_markers = [
        "my name is ",
        "call me ",
        "i'm ",
        "i am ",
        "i like ",
        "i love ",
        "i prefer ",
        "my favorite ",
        "i enjoy ",
        "i dislike ",
        "i don't like ",
        "i hate ",
        "i prefer not to ",
        "not a fan of ",
    ]
    return any(m in lower for m in explicit_markers)


def _heuristic_memory_from_message(user_message, last_assistant_message=None):
    lower = user_message.lower()
    traits = {}
    for prefix in ["my name is ", "call me "]:
        if prefix in lower:
            start = lower.find(prefix) + len(prefix)
            segment = user_message[start:].strip().split(".")[0]
            name = segment.split(",")[0].split(" and ")[0]
            if name:
                traits["name"] = name.strip()
            break
    if "i'm " in lower or "i am " in lower:
        for prefix in ["i'm ", "i am "]:
            if prefix in lower:
                start = lower.find(prefix) + len(prefix)
                segment = user_message[start:].strip().split(".")[0]
                if " and " in segment:
                    name = segment.split(" and ")[0].split(",")[0]
                    if name:
                        traits["name"] = name.strip()
                break
    for prefix in ["i like ", "i love ", "i prefer ", "my favorite ", "i enjoy "]:
        if prefix in lower:
            start = lower.find(prefix) + len(prefix)
            pref_text = user_message[start:].strip().split(".")[0]
            items = _split_items(pref_text)
            if items:
                traits["likes"] = items
            break
    if " and like " in lower or " and love " in lower or " and prefer " in lower or " and enjoy " in lower:
        for prefix in [" and like ", " and love ", " and prefer ", " and enjoy "]:
            if prefix in lower:
                start = lower.find(prefix) + len(prefix)
                pref_text = user_message[start:].strip().split(".")[0]
                items = _split_items(pref_text)
                if items:
                    traits["likes"] = items
                break
    for prefix in ["i dislike ", "i don't like ", "i hate ", "i prefer not to ", "not a fan of "]:
        if prefix in lower:
            start = lower.find(prefix) + len(prefix)
            dislike_text = user_message[start:].strip().split(".")[0]
            items = _split_items(dislike_text)
            if items:
                traits["dislikes"] = items
            break
    if "concise" in lower or "brief" in lower or "short" in lower:
        traits["communication_style"] = "concise"
    if "detailed" in lower or "step-by-step" in lower or "long" in lower:
        traits["communication_style"] = "detailed"

    if not traits and last_assistant_message:
        last_lower = last_assistant_message.lower()
        if _is_preference_question(last_assistant_message) and not _is_request_like(
            user_message
        ):
            if "dislike" in last_lower or "don't like" in last_lower or "avoid" in last_lower:
                traits["dislikes"] = _split_items(user_message)
            elif "like" in last_lower or "enjoy" in last_lower or "favorite" in last_lower:
                traits["likes"] = _split_items(user_message)
            else:
                category = _infer_followup_category(last_assistant_message)
                if category:
                    items = _split_items(user_message)
                    if items:
                        traits[category] = items if len(items) > 1 else items[0]
    return traits


def _normalize_memory_value(key, value):
    list_keys = {
        "likes",
        "dislikes",
        "interests",
        "favorite_topics",
        "hobbies",
        "favorite_colors",
        "music",
        "book_genres",
        "movie_genres",
        "favorite_authors",
    }
    if isinstance(value, list) or key in list_keys:
        items = value if isinstance(value, list) else _split_items(str(value))
        return [str(v).strip() for v in items if str(v).strip()]
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return ""


def _merge_memory(existing, new_data):
    merged = dict(existing)
    for key, value in new_data.items():
        if value is None or value == "":
            continue
        normalized = _normalize_memory_value(str(key), value)
        if isinstance(normalized, list):
            current = merged.get(key, [])
            if not isinstance(current, list):
                current = _normalize_memory_value(str(key), current)
            seen = {str(v).lower() for v in current}
            for item in normalized:
                if item.lower() not in seen:
                    current.append(item)
                    seen.add(item.lower())
            merged[key] = current
        else:
            merged[key] = normalized
    return merged


def _extract_memory_from_message(user_message, existing_traits, last_assistant_message=None):
    should_consider = _is_explicit_preference_statement(user_message) or (
        last_assistant_message
        and _is_preference_question(last_assistant_message)
        and not _is_request_like(user_message)
    )
    if not should_consider:
        return {}

    prompt = (
        "Extract any personal traits or preferences stated by the user in the message. "
        "Return a JSON object of key-value pairs. If none, return {}. "
        "Use lists for multiple items (e.g., likes, dislikes, interests, hobbies, "
        "favorite_colors, music, book_genres, movie_genres, favorite_authors). "
        "Only include facts explicitly stated by the user and avoid guesses."
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"User message: {user_message}\n"
                f"Last assistant message: {last_assistant_message or ''}\n"
                f"Existing memory: {json.dumps(existing_traits)}\n"
                "Return JSON only."
            ),
        },
    ]
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "max_tokens": 120,
        "temperature": 0,
    }
    try:
        resp = requests.post(api_url, headers=JSON_HEADERS, json=payload, timeout=20)
    except requests.exceptions.RequestException:
        return {}
    if resp.status_code != 200:
        return {}
    try:
        data = resp.json()
    except json.JSONDecodeError:
        return {}
    content = ""
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
    extracted = _extract_json(content)
    if isinstance(extracted, dict) and "name" in extracted and "likes" not in extracted:
        heuristic_check = _heuristic_memory_from_message(
            user_message, last_assistant_message
        )
        if "likes" in heuristic_check:
            extracted["likes"] = heuristic_check["likes"]
    if isinstance(extracted, dict):
        heuristic = _heuristic_memory_from_message(
            user_message, last_assistant_message
        )
        if extracted:
            for key, value in heuristic.items():
                if key not in extracted:
                    extracted[key] = value
            return extracted
        return heuristic
    return _heuristic_memory_from_message(user_message, last_assistant_message)


def _stream_chat_completion(messages, placeholder):
    payload = {"model": model_id, "messages": messages, "stream": True}
    try:
        resp = requests.post(
            api_url, headers=STREAM_HEADERS, json=payload, timeout=30, stream=True
        )
    except requests.exceptions.RequestException as exc:
        msg = f"Network error while contacting Hugging Face: {exc}"
        placeholder.error(msg)
        return "", msg

    if resp.status_code != 200:
        msg = _build_api_error_message(resp)
        placeholder.error(msg)
        return "", msg

    collected = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data:"):
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("error"):
                msg = f"Hugging Face API error. Details: {data['error']}"
                placeholder.error(msg)
                return collected, msg
            choices = data.get("choices") if isinstance(data, dict) else None
            if isinstance(choices, list) and choices:
                delta = choices[0].get("delta", {})
                if isinstance(delta, dict):
                    chunk = delta.get("content")
                    if chunk:
                        collected += chunk
                        placeholder.markdown(collected)
                        # Tiny delay to make streaming visible.
                        time.sleep(0.02)

    if not collected:
        collected = "No text was generated by the model."
        placeholder.markdown(collected)
    return collected, None


if "memory" not in st.session_state:
    st.session_state.memory = _load_memory()

if "chats" not in st.session_state:
    loaded = _load_chats()
    if loaded:
        st.session_state.chats = loaded
        st.session_state.active_chat_id = loaded[0]["id"]
    else:
        st.session_state.chats = [_new_chat()]
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]
        _save_chat(st.session_state.chats[0])

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = (
        st.session_state.chats[0]["id"] if st.session_state.chats else None
    )


def _get_active_chat():
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.active_chat_id:
            return chat
    return None


def _last_assistant_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


st.sidebar.header("Chats")
with st.sidebar.expander("User Memory", expanded=False):
    if st.session_state.memory:
        st.json(st.session_state.memory)
    else:
        st.caption("No saved preferences yet.")
    if st.button("Clear Memory", use_container_width=True):
        st.session_state.memory = {}
        _save_memory({})
        st.rerun()
if st.sidebar.button("New Chat"):
    new_chat = _new_chat()
    st.session_state.chats.insert(0, new_chat)
    st.session_state.active_chat_id = new_chat["id"]
    _save_chat(new_chat)
    st.rerun()

chat_list = st.sidebar.container(height=400)
with chat_list:
    for chat in st.session_state.chats:
        is_active = chat["id"] == st.session_state.active_chat_id
        label = f"{chat['title']} · {chat['created_at']}"
        cols = st.columns([0.88, 0.12])
        with cols[0]:
            if st.button(
                label,
                key=f"open_{chat['id']}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.active_chat_id = chat["id"]
                st.rerun()
        with cols[1]:
            if st.button("✕", key=f"del_{chat['id']}", use_container_width=True):
                st.session_state.chats = [
                    c for c in st.session_state.chats if c["id"] != chat["id"]
                ]
                try:
                    _chat_path(chat["id"]).unlink(missing_ok=True)
                except Exception:
                    pass
                if st.session_state.active_chat_id == chat["id"]:
                    st.session_state.active_chat_id = (
                        st.session_state.chats[0]["id"]
                        if st.session_state.chats
                        else None
                    )
                st.rerun()

active_chat = _get_active_chat()
if active_chat is None:
    st.info("No chats yet. Click New Chat to start a conversation.")
    st.stop()

history = st.container(height=500)
with history:
    for msg in active_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

prompt = st.chat_input("Type your message")
if prompt:
    active_chat["messages"].append({"role": "user", "content": prompt})
    if active_chat["title"] == "New chat":
        active_chat["title"] = prompt[:40].strip() or "New chat"
    _save_chat(active_chat)
    last_assistant = _last_assistant_message(active_chat["messages"][:-1])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        api_messages = _messages_with_memory(
            active_chat["messages"], st.session_state.memory
        )
        generated, error_msg = _stream_chat_completion(api_messages, placeholder)

    final_text = error_msg or generated
    active_chat["messages"].append({"role": "assistant", "content": final_text})
    _save_chat(active_chat)

    extracted = _extract_memory_from_message(
        prompt, st.session_state.memory, last_assistant
    )
    if extracted:
        st.session_state.memory = _merge_memory(st.session_state.memory, extracted)
        _save_memory(st.session_state.memory)

    st.rerun()
