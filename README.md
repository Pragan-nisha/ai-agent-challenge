

````markdown
# Karbon AI Challenge – Agent-as-Coder

This repository contains an autonomous **Agent-as-Coder** that can generate, test, and debug PDF parsers for financial statements.  
The agent uses **Gemini** (for code generation) and **DeepSeek via OpenRouter** (for self-debugging).  
It follows a closed loop: generate → run → debug → retry (≤3 iterations).

---

## 🚀 Quickstart (5 Steps)

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/karbon-agent.git
   cd karbon-agent
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```



3. **Run the agent on a target dataset**

   ```bash
   python try.py --target icici
   ```

   This will:

   * Read the sample PDF from `data/icici/icici_sample.pdf`
   * Generate a parser under `custom_parsers/icici_parser.py`
   * Create a test at `tests/test_icici_parser.py`
   * Run pytest to validate the parser

4. **Check results**
   If successful, you’ll see ✅ pytest passing.
   The parsed output is returned as a Pandas DataFrame.

---

## 🛠️ How It Works

The agent follows a **closed feedback loop**:

```
┌───────────┐      ┌───────────┐      ┌─────────────┐
│  Gemini   │ ---> │  Parser   │ ---> │   Pytest    │
│ Generator │      │ Candidate │      │   Results   │
└─────┬─────┘      └─────┬─────┘      └─────┬──────┘
      │                  │                 │
      │            Fails │                 │
      │                  ▼                 │
      │           ┌───────────┐            │
      └────────── │ DeepSeek  │ <──────────┘
                  │ Debugger  │
                  └───────────┘
```

* **Gemini** generates initial parser code.
* The code is tested via **pytest** against the provided CSV.
* On failure, errors are passed to **DeepSeek** for patching.
* This repeats up to **3 times** (configurable).

---

## 📂 Project Structure

```
.
├── agent.py                     # Main agent
├── custom_parsers/              # Generated parsers
├── tests/                       # Auto-generated pytest files
├── data/                        # Contains PDF + CSV samples per target
├── requirements.txt
└── README.md
```

---

## ⚠️ Notes & Limitations

* The agent depends on **external APIs** (Gemini + OpenRouter).
* After 3 failed attempts, the agent will return the **last candidate parser** for inspection.
* Works best with well-structured financial PDFs (tables, statements).

---

