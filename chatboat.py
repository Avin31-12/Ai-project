#!/usr/bin/env python3
r"""
Local AI Chatbot — llama.cpp Server Backend (Windows-Friendly)
==============================================================
No llama-cpp-python package. No Ollama app. No registry edits.
Auto-downloads llama-server.exe + model to D:\Ai_chatboat.
Everything (venv, cache, temp, model, binaries) stays on D: — nothing
is written to C:.

Run:
    D:\Ai_chatboat\venv\Scripts\python.exe -m pip install flask requests
    D:\Ai_chatboat\venv\Scripts\python.exe chatboat.py

Open: http://localhost:5000
"""

import atexit
import json
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

import requests
from flask import Flask, Response, jsonify, render_template_string, request

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
HOST = "127.0.0.1"
PORT = 5000

BASE_DIR = Path("D:/Ai_chatboat")

MODEL_URL = (
    "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/"
    "resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)
MODEL_PATH = BASE_DIR / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

LLAMA_DIR = BASE_DIR / "llama_cpp"
SERVER_PORT = 8081
SERVER_EXE: Optional[Path] = None
SERVER_PROC: Optional[subprocess.Popen] = None

SYSTEM_PROMPT = (
    "You are a friendly, helpful local assistant. "
    "Be warm, clear, and concise. If the user asks for code, explain it simply. "
    "If the request is vague, ask a short clarifying question."
)
MAX_CHAT_HISTORY = 8
CHAT_TIMEOUT_SECONDS = 20


def trim_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"system", "user", "assistant"} and isinstance(content, str):
            cleaned.append({"role": role, "content": content})

    if not cleaned:
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    kept = []
    system_prompt = None
    for message in cleaned:
        if message.get("role") == "system":
            if system_prompt is None:
                system_prompt = message
            continue
        kept.append(message)

    if len(kept) > MAX_CHAT_HISTORY * 2:
        kept = kept[-(MAX_CHAT_HISTORY * 2) :]

    result = []
    if system_prompt is not None:
        result.append(system_prompt)
    else:
        result.append({"role": "system", "content": SYSTEM_PROMPT})
    result.extend(kept)
    return result

# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOAD MODEL
# ═════════════════════════════════════════════════════════════════════════════
def ensure_model() -> None:
    if MODEL_PATH.exists():
        print(f"[✓] Model found: {MODEL_PATH}")
        return

    print(f"[↓] Downloading model (~2 GB)...\n    {MODEL_URL}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".part")
    with requests.get(MODEL_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb = downloaded / 1024 / 1024
                    tot_mb = total / 1024 / 1024
                    print(f"\r    {pct:.1f}%  ({mb:.0f}/{tot_mb:.0f} MB)", end="", flush=True)
    tmp_path.rename(MODEL_PATH)
    print(f"\n[✓] Saved to {MODEL_PATH}")

# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOAD & FIND LLAMA-SERVER.EXE
# ═════════════════════════════════════════════════════════════════════════════
def _pick_windows_asset(assets: list) -> Optional[dict]:
    """
    llama.cpp's release asset naming changes over time
    (win-avx2-x64, win-cpu-x64, win-x64, etc). Instead of hardcoding one
    exact filename, search the actual asset list for the best Windows
    CPU build so this keeps working as releases change.
    """
    candidates_priority = [
        lambda n: "win" in n and "avx2" in n and "x64" in n and n.endswith(".zip"),
        lambda n: "win" in n and "x64" in n and "cuda" not in n and n.endswith(".zip"),
        lambda n: "win" in n and n.endswith(".zip"),
    ]
    for check in candidates_priority:
        for a in assets:
            name = a["name"].lower()
            if check(name):
                return a
    return None

def ensure_llama_server() -> Path:
    global SERVER_EXE

    # Already discovered?
    if SERVER_EXE and SERVER_EXE.exists():
        return SERVER_EXE

    # Search existing extract folder
    if LLAMA_DIR.exists():
        for p in LLAMA_DIR.rglob("llama-server.exe"):
            SERVER_EXE = p
            print(f"[✓] Found llama-server: {SERVER_EXE}")
            return SERVER_EXE

    # Download latest release from GitHub
    print("[↓] Looking up latest llama.cpp release...")
    api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    release = resp.json()
    tag = release["tag_name"]
    assets = release.get("assets", [])

    asset = _pick_windows_asset(assets)
    if not asset:
        raise FileNotFoundError(
            f"Could not find a Windows x64 build in release {tag}. "
            "Download manually from https://github.com/ggerganov/llama.cpp/releases "
            f"and unzip it into {LLAMA_DIR}"
        )

    download_url = asset["browser_download_url"]
    print(f"[↓] Downloading {asset['name']} ({tag})...")
    zip_data = requests.get(download_url, timeout=120).content
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = LLAMA_DIR / "server.zip"

    with open(zip_path, "wb") as f:
        f.write(zip_data)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(LLAMA_DIR)
    zip_path.unlink()

    for p in LLAMA_DIR.rglob("llama-server.exe"):
        SERVER_EXE = p
        print(f"[✓] Extracted llama-server: {SERVER_EXE}")
        return SERVER_EXE

    raise FileNotFoundError(
        "llama-server.exe not found in the downloaded archive. "
        "Try manually downloading from https://github.com/ggerganov/llama.cpp/releases"
    )

# ═════════════════════════════════════════════════════════════════════════════
# START / STOP BACKEND SERVER
# ═════════════════════════════════════════════════════════════════════════════
def is_server_running() -> bool:
    try:
        response = requests.get(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2)
        return response.status_code < 500
    except Exception:
        return False

def start_llama_server() -> None:
    global SERVER_PROC

    if is_server_running():
        print("[✓] llama.cpp server already running.")
        return

    exe = ensure_llama_server()
    ensure_model()

    print(f"[⏳] Starting llama.cpp server on port {SERVER_PORT}...")
    SERVER_PROC = subprocess.Popen(
        [
            str(exe),
            "-m", str(MODEL_PATH),
            "-c", "4096",
            "--port", str(SERVER_PORT),
            "--host", "127.0.0.1",
        ],
        cwd=str(exe.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait until the port responds (max 120s — first load can be slow)
    for _ in range(120):
        if is_server_running():
            print("[✓] Server ready!")
            return
        if SERVER_PROC.poll() is not None:
            raise RuntimeError(
                "llama-server process exited early. Run the exe manually from a "
                "terminal to see the error, e.g.:\n"
                f'  "{exe}" -m "{MODEL_PATH}" -c 4096 --port {SERVER_PORT}'
            )
        time.sleep(1)

    raise TimeoutError("llama.cpp server failed to start within 120s.")

def stop_llama_server() -> None:
    global SERVER_PROC
    if SERVER_PROC:
        SERVER_PROC.terminate()
        try:
            SERVER_PROC.wait(timeout=5)
        except Exception:
            SERVER_PROC.kill()

atexit.register(stop_llama_server)

def _signal_handler(sig, frame):
    stop_llama_server()
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)

# ═════════════════════════════════════════════════════════════════════════════
# FLASK APP + INLINE FRONTEND
# ═════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Local AI Chatbot</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#0d1117;color:#c9d1d9;height:100vh;display:flex;flex-direction:column}
header{background:#161b22;padding:1rem 1.5rem;border-bottom:1px solid #30363d;
       display:flex;justify-content:space-between;align-items:center}
header h1{font-size:1.1rem;color:#58a6ff}
.badge{background:#238636;color:#fff;padding:.2rem .7rem;border-radius:999px;font-size:.7rem}
#chat{flex:1;overflow-y:auto;padding:1.5rem;display:flex;flex-direction:column;gap:1rem;
      max-width:800px;width:100%;margin:0 auto}
.msg{max-width:85%;padding:.9rem 1.1rem;border-radius:12px;line-height:1.55;font-size:.95rem;
     animation:pop .25s ease}
@keyframes pop{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.user{align-self:flex-end;background:#1f6feb;color:#fff}
.bot{align-self:flex-start;background:#21262d;border:1px solid #30363d}
.bot .name{font-size:.72rem;color:#58a6ff;font-weight:700;margin-bottom:.3rem}
#input-area{background:#161b22;border-top:1px solid #30363d;padding:1rem}
#wrap{max-width:800px;margin:0 auto;display:flex;gap:.6rem}
#msgInput{flex:1;background:#0d1117;border:1px solid #30363d;border-radius:10px;
          padding:.8rem 1rem;color:inherit;font-size:1rem;outline:none}
#msgInput:focus{border-color:#58a6ff}
#sendBtn{background:#1f6feb;color:#fff;border:none;border-radius:10px;
         padding:.8rem 1.4rem;cursor:pointer;font-size:1rem}
#sendBtn:disabled{background:#30363d;cursor:not-allowed}
#voiceBtn{background:#21262d;border:1px solid #30363d;color:#c9d1d9;border-radius:10px;
          padding:.8rem .95rem;cursor:pointer;font-size:1rem}
#voiceBtn.listening{background:#f85149;color:#fff;border-color:#f85149}
#clearBtn{background:transparent;border:1px solid #484f58;color:#8b949e;
         padding:.25rem .7rem;border-radius:6px;cursor:pointer;font-size:.75rem}
#clearBtn:hover{border-color:#f85149;color:#f85149}
.info{text-align:center;color:#484f58;font-size:.8rem;margin:.3rem 0}
</style>
</head>
<body>
<header>
  <h1>AI_Avinash</h1>
  <div style="display:flex;gap:.6rem;align-items:center">
    <span id="statusBadge" class="badge offline">● Offline</span>
    <button id="clearBtn" onclick="clearChat()">Clear</button>
  </div>
</header>

<div id="chat"></div>

<div id="input-area">
  <div id="wrap">
    <input type="text" id="msgInput" placeholder="Type a message…" autocomplete="off">
    <button id="voiceBtn" type="button" onclick="toggleVoice()">🎤</button>
    <button id="sendBtn" onclick="send()">Send</button>
  </div>
</div>

<script>
const chat=document.getElementById('chat');
const input=document.getElementById('msgInput');
const btn=document.getElementById('sendBtn');
const voiceBtn=document.getElementById('voiceBtn');
const statusBadge=document.getElementById('statusBadge');
let history=[{role:'system',content:{{ SYSTEM_PROMPT|tojson }} }];
let recognition=null;
let isListening=false;

input.addEventListener('keypress',e=>{if(e.key==='Enter')send()});

function setStatus(text, online){
  statusBadge.textContent=text;
  statusBadge.className='badge ' + (online ? 'online' : 'offline');
}

function setVoiceButton(listening){
  isListening=listening;
  voiceBtn.classList.toggle('listening', listening);
  voiceBtn.textContent=listening ? '●' : '🎤';
}

function speakText(text){
  if(!text || typeof window.speechSynthesis==='undefined') return;
  window.speechSynthesis.cancel();
  const utter=new SpeechSynthesisUtterance(text);
  utter.lang='en-US';
  utter.rate=1;
  window.speechSynthesis.speak(utter);
}

function addMsg(role,text){
  const d=document.createElement('div');
  d.className='msg '+role;
  if(role==='bot') d.innerHTML='<div class="name">Assistant</div><div class="txt"></div>';
  else d.textContent=text;
  chat.appendChild(d);
  chat.scrollTop=chat.scrollHeight;
  return d;
}

function showWelcome(){
  chat.innerHTML='<div class="info">Hello! I am your local assistant. Ask me anything, and I will reply naturally and helpfully.</div>';
}

function clearChat(){
  history=[{role:'system',content:{{ SYSTEM_PROMPT|tojson }} }];
  showWelcome();
  setStatus('● Offline', false);
  setVoiceButton(false);
}

function scrollBottom(){
  chat.scrollTop=chat.scrollHeight;
}

function toggleVoice(){
  const SpeechRecognitionCtor=window.SpeechRecognition || window.webkitSpeechRecognition;
  if(!SpeechRecognitionCtor){
    alert('Voice input is not supported in this browser. Please use Chrome or Edge.');
    return;
  }

  if(!recognition){
    recognition=new SpeechRecognitionCtor();
    recognition.continuous=false;
    recognition.interimResults=true;
    recognition.lang='en-US';
    recognition.onresult=(event)=>{
      let transcript='';
      for(let i=event.resultIndex;i<event.results.length;i++){
        const result=event.results[i];
        if(result.isFinal){
          transcript += result[0].transcript.trim() + ' ';
        }
      }
      input.value=transcript.trim();
      if(event.results[event.results.length - 1].isFinal){
        setVoiceButton(false);
        if(input.value.trim()){
          setTimeout(()=>{ if(!btn.disabled) send(); }, 400);
        }
      }
    };
    recognition.onerror=(event)=>{
      console.error('Speech recognition error:', event.error);
      setVoiceButton(false);
      setStatus('● Offline', false);
      if(event.error==='not-allowed' || event.error==='service-not-allowed'){
        alert('Microphone permission was blocked. Please allow microphone access in your browser and try again.');
      }
    };
    recognition.onend=()=>{
      setVoiceButton(false);
    };
  }

  if(isListening){
    recognition.stop();
    return;
  }

  input.value='';
  setVoiceButton(true);
  setStatus('● Listening…', true);
  recognition.start();
}

async function send(){
  const text=input.value.trim();
  if(!text) return;
  input.value='';
  addMsg('user',text);
  history.push({role:'user',content:text});
  btn.disabled=true;
  setStatus('● Thinking…', true);

  const botDiv=addMsg('bot','');
  const txtDiv=botDiv.querySelector('.txt');
  txtDiv.textContent='⏳ Starting the local model…';
  let dotCount=0;
  const typingTimer=setInterval(()=>{
    txtDiv.textContent='⏳ Starting the local model' + '.'.repeat(dotCount % 4);
    dotCount += 1;
    scrollBottom();
  }, 400);

  try{
    const controller=new AbortController();
    const timeoutId=setTimeout(()=>controller.abort(), 25000);
    const res=await fetch('/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:history}),
      signal:controller.signal
    });
    clearTimeout(timeoutId);
    if(!res.ok){
      throw new Error('The assistant could not respond right now.');
    }
    const reader=res.body.getReader();
    const dec=new TextDecoder();
    let full='';
    txtDiv.textContent='';

    while(true){
      const {done,value}=await reader.read();
      if(done) break;
      const lines=dec.decode(value).split('\\n');
      for(const line of lines){
        if(!line.startsWith('data: ')) continue;
        const data=line.slice(6);
        if(data==='[DONE]') continue;
        try{
          const j=JSON.parse(data);
          const c=j.content||'';
          full+=c;
          txtDiv.textContent+=c;
          chat.scrollTop=chat.scrollHeight;
        }catch(_){}
      }
    }
    history.push({role:'assistant',content:full});
    setStatus('● Online', true);
    speakText(full);
  }catch(err){
    if(err.name==='AbortError'){
      txtDiv.textContent='The model is taking longer than expected. Please wait a moment and try again.';
    } else {
      txtDiv.textContent='Sorry — ' + err.message;
    }
    setStatus('● Offline', false);
  }finally{
    clearInterval(typingTimer);
    btn.disabled=false;
    input.focus();
  }
}

showWelcome();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, SYSTEM_PROMPT=SYSTEM_PROMPT)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "backend": is_server_running()})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    messages = trim_messages(data.get("messages", []))

    def stream():
        try:
            r = requests.post(
                f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
                json={
                    "model": "local",
                    "messages": messages,
                    "stream": True,
                    "temperature": 0.7,
                    "max_tokens": 400,
                },
                stream=True,
                timeout=CHAT_TIMEOUT_SECONDS,
            )
            if not r.ok:
                payload = r.text or "Backend error"
                yield f"data: {json.dumps({'content': f'Sorry — {payload}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            for line in r.iter_lines():
                if not line:
                    continue
                txt = line.decode("utf-8")
                if not txt.startswith("data: "):
                    continue
                payload = txt[6:]
                if payload == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                try:
                    obj = json.loads(payload)
                    content = obj["choices"][0]["delta"].get("content", "")
                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"
                except Exception:
                    pass
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'content': f'Sorry — {e}'})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(stream(), mimetype="text/event-stream")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    start_llama_server()
    print(f" Open http://{HOST}:{PORT} in your browser\n")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)