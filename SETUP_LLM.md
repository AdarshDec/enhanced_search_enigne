# Setting Up LLM (FREE Options)

## 🆓 Free LLM Option: Ollama (Recommended)

**Ollama is 100% FREE and runs on your computer!**

### Step 1: Install Ollama

1. Download from: **https://ollama.ai**
2. Install it (Windows/Mac/Linux supported)
3. It runs in the background automatically

### Step 2: Download a Model

Open terminal/command prompt and run:

```bash
# Small, fast model (recommended to start)
ollama pull llama2

# Or try other free models:
ollama pull mistral        # Another popular option
ollama pull codellama      # Better for code questions
ollama pull phi            # Very small, very fast
```

**Model sizes:**
- `llama2` - ~3.8 GB (good balance)
- `mistral` - ~4.1 GB (high quality)
- `phi` - ~1.6 GB (fastest, smallest)

### Step 3: Start Your Server

```bash
cd C:\Users\adars\enhanced-search-engine
python main.py
```

**That's it!** The server will automatically detect Ollama and use it.

### Verify It's Working

1. Check terminal output - should see:
   ```
   ✓ Initialized Ollama with model: llama2
   ✓ LLM enabled: ollama (llama2)
   ```

2. Check stats in browser:
   - Go to http://localhost:8000
   - Scroll to "System Statistics"
   - Should show "LLM: ✓ Available" with "ollama"

3. Test Q&A:
   - Go to "Ask Questions" tab
   - Ask a question
   - Should get real answer (not mock response)

---

## 🔄 Alternative: Mock Mode (For Testing UI)

**If you don't want to install Ollama:**

The system automatically uses "mock mode" which:
- ✅ Works without any setup
- ✅ Allows you to test the Q&A interface
- ✅ Returns sample responses (not real answers)
- ⚠️ Answers are just placeholders

**No action needed** - mock mode is automatic fallback!

---

## 💰 Paid Option: OpenAI (Optional)

If you want to use OpenAI GPT models instead:

### Setup:

1. Get API key: https://platform.openai.com/api-keys
2. Set environment variable:

   **Windows PowerShell:**
   ```powershell
   $env:OPENAI_API_KEY="sk-your-key-here"
   ```

   **Windows CMD:**
   ```cmd
   set OPENAI_API_KEY=sk-your-key-here
   ```

   **Linux/Mac:**
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   ```

3. Restart server - OpenAI will be auto-detected

**Cost:** ~$0.002 per question (very cheap, but not free)

---

## 🎯 Which Should You Use?

| Option | Cost | Setup | Quality | Speed |
|--------|------|-------|---------|-------|
| **Ollama** | 🆓 Free | Easy | Good | Fast (local) |
| **Mock** | 🆓 Free | None | Sample only | Instant |
| **OpenAI** | 💰 Paid | Easy | Excellent | Fast (cloud) |

**Recommendation:** Use **Ollama** - it's free, private, and works great!

---

## 🔍 Troubleshooting

### "Ollama not available"

**Problem:** Server says Ollama not found

**Fix:**
1. Make sure Ollama is installed
2. Make sure Ollama is running (it should start automatically)
3. Check if port 11434 is accessible:
   ```bash
   # Test connection
   curl http://localhost:11434/api/tags
   ```
4. Restart your server

### "Model not found"

**Problem:** Ollama is running but model missing

**Fix:**
```bash
# Download the model
ollama pull llama2

# Verify it's downloaded
ollama list
```

### Slow Responses

**Problem:** Q&A takes too long

**Fixes:**
1. Use a smaller model: `ollama pull phi`
2. Use OpenAI (faster, but paid)
3. Reduce `top_k_docs` in question (fewer documents = faster)

---

## 📊 Current Status

Check your LLM status:

1. **In Terminal:** Look for startup messages
2. **In Browser:** Check `/stats` endpoint or stats section
3. **Via API:**
   ```bash
   curl http://localhost:8000/stats
   ```

Look for:
- `"llm_available": true` ✅
- `"llm_provider": "ollama"` ✅
- `"llm_model": "llama2"` ✅

---

## 🚀 Quick Start (Ollama)

```bash
# 1. Install Ollama from https://ollama.ai
# 2. Download model
ollama pull llama2

# 3. Run your server (in enhanced-search-engine folder)
python main.py

# 4. Check terminal - should see:
# ✓ Initialized Ollama with model: llama2
# ✓ LLM enabled: ollama (llama2)

# 5. Open browser: http://localhost:8000
# 6. Test Q&A tab - ask a question!
```

---

**That's it! Ollama is completely free and works offline. 🎉**


