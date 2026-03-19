### Task 1A: Page Setup & API Connection
**Prompt:** 

I'm building an app: using python

Requirements:

Use st.set_page_config(page_title="My AI Chat", layout="wide").
Load your Hugging Face token using st.secrets["HF_TOKEN"]. The token must never be hardcoded in app.py.
If the token is missing or empty, display a clear error message in the app. The app must not crash.
Send a single hardcoded test message (e.g. "Hello!") to the Hugging Face API using the loaded token and display the model’s response in the main area.
Handle API errors gracefully (missing token, invalid token, rate limit, network failure) with a user-visible message rather than a crash.

Success criteria (Part A): Running streamlit run app.py with a valid .streamlit/secrets.toml sends a test message and displays the model’s reply. Running it without the secrets file shows an error message instead of crashing.


**AI Suggestion:** 
Created the project structure, set up a Python virtual environment, installed streamlit and requests, and added them to requirements.txt. Also created the Streamlit app entry point (app.py), initialized git, committed the initial files, and connected/pushed the repo to GitHub.

**My Modifications & Reflections:** 
There was an error message intially. The venv was making some error so it had to be rebuilt. The error message kept showing. So we redid some Hugging Face API error. There was also an error in the secrets file. Finally it showed the prewritten message.


### Task 1B: Multi-Turn Conversation UI
**Prompt:** 

Thank you, for part B:

Requirements:

Extend Part A to replace the hardcoded test message with a real input interface.
Use native Streamlit chat UI elements. Render messages with st.chat_message(...) and collect user input with st.chat_input(...).
Add a fixed input bar at the bottom of the main area.
Store the full conversation history in st.session_state. After each exchange, append both the user message and the assistant response to the history.
Send the full message history with each API request so the model maintains context.
Render the conversation history above the input bar using default Streamlit UI elements rather than CSS-based custom chat bubbles.
The message history must scroll independently of the input bar — the input bar stays visible at all times.

Success criteria (Part B): Sending multiple messages in a row produces context-aware replies (e.g. the model remembers the user’s name from an earlier message). Messages are displayed with correct styling and the input bar remains fixed.


**AI Suggestion:** 
Replaced the hardcoded test message with a real chat UI using st.chat_input and st.chat_message, stored full conversation history in st.session_state, and sent that history with each API request to maintain context. Also rendered the history in a scrollable container above a fixed input bar, and added graceful error handling so the app never crashes.

**My Modifications & Reflections:** 
This time things looked more like I needed it to. I ran some tests by greeting it giving it my name. Then sent it multiple messages about random topics then asked my name again. It remebered and the formating was correct throughout.

### Task 1C: Chat Management
**Prompt:** 

Great, for part C:

Requirements:

Add a New Chat button to the sidebar that creates a fresh, empty conversation and adds it to the sidebar chat list.
Use the native Streamlit sidebar (st.sidebar) for chat navigation.
The sidebar shows a scrollable list of all current chats, each displaying a title and timestamp.
The currently active chat must be visually highlighted in the sidebar.
Clicking a chat in the sidebar switches to it without deleting or overwriting any other chats.
Each chat entry must have a ✕ delete button. Clicking it removes the chat from the list. If the deleted chat was active, the app must switch to another chat or show an empty state.

Success criteria (Part C): Multiple chats can be created, switched between, and deleted independently. The active chat is always visually distinct.



**AI Suggestion:** 
Added a sidebar chat manager with a New Chat button, a scrollable list of chats showing titles and timestamps, and per‑chat ✕ delete buttons. The active chat is visually highlighted, clicking a chat switches without losing history, and deleting the active one automatically selects another or shows an empty state

**My Modifications & Reflections:** 
Chatted, made another chat & chatted more. Then made a new one and deleted one. Throughout all things were correctly formatted. The chat was saved until I deleted it. All things worked as needed.

### Task 1D: Chat Persistence
**Prompt:** 

Thank you, part D is:

Requirements:

Each chat session is saved as a separate JSON file inside a chats/ directory. Each file must store at minimum: a chat ID, a title or timestamp, and the full message history.
On app startup, all existing files in chats/ are loaded and shown in the sidebar automatically.
Returning to a previous chat and continuing the conversation must work correctly.
Deleting a chat (✕ button) must also delete the corresponding JSON file from chats/.
A generated or summarized chat title is acceptable and encouraged. The title does not need to be identical to the first user message.

Success criteria (Part D): Closing and reopening the app shows all previous chats intact in the sidebar. Continuing a loaded chat works correctly. Deleting a chat removes its file from disk.


**AI Suggestion:** 
Each chat is saved as its own JSON file in chats/ (including id, title/timestamp, and full message history), and all existing files are loaded into the sidebar on startup so chats survive restarts. Deleting a chat now removes its JSON file from disk, and continuing a loaded chat works normally.

**My Modifications & Reflections:** 
I made some chats and reloaded. It worked. I closed and reopened, it didn't work just didn't open the page at all. Told the Ai and he reworked the code. All worked as needed this time.

### Task 2: Response Streaming
**Prompt:** 

That finished task 1. For task 2:

Requirements

Use the stream=True parameter in your API request and handle the server-sent event stream.
In Streamlit, use native Streamlit methods such as st.write_stream() or manually update a placeholder with st.empty() as chunks arrive.
The full streamed response must be saved to the chat history once streaming is complete.
Hint: Add stream=True to your request payload and set stream=True on the requests.post() call. The response body will be a series of data: lines in SSE format.

Note: Very small models such as meta-llama/Llama-3.2-1B-Instruct may stream so quickly that the output appears to arrive all at once. If your app is correctly receiving multiple streamed chunks but the effect is too fast to notice, you are required to add a very short delay between rendering chunks so the streaming behavior is visible in the UI.

Success criteria: Responses appear incrementally in the chat interface and are correctly saved to history.


**AI Suggestion:** 
Enabled streaming responses by sending stream=True in the API payload and using stream=True on the requests.post() call, then parsed SSE data: lines to build the reply incrementally. Rendered the stream with a Streamlit placeholder (st.empty()), added a tiny delay so the streaming is visible, and saved the full assembled response back into chat history once streaming finished.

**My Modifications & Reflections:** 
Everything worked as desired. I messaged it and it types out its' response and saved it to history.

### Task 3: User Memory
**Prompt:** 

Thank you for the second task. Now for task three, the final task: 

Goal: Extract and store user preferences from conversations, then use them to personalize future responses.

Requirements

After each assistant response, make a second lightweight API call asking the model to extract any personal traits or preferences mentioned by the user in that message.
Extracted traits are stored in a memory.json file. Example categories might include name, preferred language, interests, communication style, favorite topics, or other useful personal preferences.
The sidebar displays a User Memory expander panel showing the currently stored traits.
Include a native Streamlit control to clear/reset the saved memory.
Stored memory is injected into the system prompt of future conversations so the model can personalize responses.
Implementation note: The categories above are only examples for reference. It is up to you to decide what traits to store, how to structure your memory.json, how to merge or update existing memory, and how to incorporate that memory into future prompts, as long as the final app clearly demonstrates persistent user memory and personalization.

Hint: A simple memory extraction prompt might look like: “Given this user message, extract any personal facts or preferences as a JSON object. If none, return {}”

Success criteria: User traits are extracted, displayed in the sidebar, and used to personalize subsequent responses.


**AI Suggestion:** 
Added persistent user memory: after each assistant reply, the app makes a second API call to extract user traits/preferences, stores them in memory.json, and shows them in a User Memory expander in the sidebar with a Clear Memory button. That memory is injected into future system prompts so replies are personalized, and it persists across app restarts.

**My Modifications & Reflections:** 
It intially had the user memory tab but didn't save perferences. I raised this problem and then it workedd but only saved one preference at a time and no dislikes. I told it and it was fixed but did not save further specific preferences that the chat asked me for. Then it saved any response to questions as a "like". Then it did not distinguish multiple preferences and different types of informations. Finally, after all those errors, it worked well enough to work, remember, and distinguish preferences.