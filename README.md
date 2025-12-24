# Starship Coffee Co. — Candidate Instructions (Python + Streamlit, 3 simple tasks / ~12h)

**You build everything in Python** and demo each task with a tiny **Streamlit** app.
**Use hosted LLM/VLM APIs only** (e.g., OpenAI). No local/base models.

You will receive small sample docs, two CSV files, and a few receipt images. Your apps should run locally with your `OPENAI_API_KEY`.

Deliver three runnable apps:

* `part1_rag/app_rag.py`
* `part2_funcs/app_funcs.py`
* `part3_receipts/app_receipts.py`

Keep code simple, readable, and short.

---

## Task 1 — Simple RAG with Citations (+ injection handling)

**Goal:** Ask a question about the provided docs and get a short answer **with citations** to the files that supported it.

### What your app must do

* Embed the provided markdown docs (chunk if you like). Store vectors in **one** of: **Qdrant**, **SQLite**, or **JSON**.
* Retrieve top‑k passages (default **k=5**) by cosine similarity.
* Call a hosted LLM to write a **≤100‑word** answer **using only retrieved text**.
* Return **citations** as the file names used to answer (no character ranges).
* Handle **prompt injection**: if asked to reveal file contents or anything in a `secrets/` folder, **refuse with one sentence** and suggest a safe alternative. Do not print any secret content.

### Streamlit UI (required)

* **Input:** a single text box: "Ask a question about the docs".
* **Sidebar:** selector for storage backend (**qdrant/sqlite/json**) and a numeric input for **k**.
* **Output area:**

  * The final answer as plain text.
  * A **Citations** table with a single column: `doc_id`.
  * A small **Debug** expander listing the top‑k filenames and short snippets.

### Input → Output (contract)

* **Input:** user question (string).
* **Output (you also print this JSON to the console):**

```json
{
  "answer": "...",
  "citations": [
    "<filename>.md"
  ]
}
```

### Acceptance

* Each preset produces a clear `final_answer` and shows **at least one** tool call.
* **Preset 1:** returns a numeric total spend for the period.
* **Preset 2:** calls `refund_order` and returns `{ok:true}` if rules allow; otherwise `{ok:false, reason:"..."}`.
* **Preset 3:** normalizes the order id to `C9` and returns the order `status` and a **masked** email.
* All emails shown are masked.

---

## Task 2 — Function Calling over Customer Data (Streamlit)

**Goal:** Use LLM **function/tool calling** to answer simple questions about customers and orders from two CSV files.

### Data you load

* `customers.csv` with columns: `customer_id, name, email, tier, credits`.
* `orders.csv` with columns: `order_id, customer_id, status, item, qty, unit_price, total, created_at`.

### Functions you expose to the model

Implement these Python functions and wire them to your LLM's function/tool calling. Keep names exactly as below; keep arguments simple.

* `get_order(order_id)` → returns: `status`, `total`, and a **masked email** (mask like `l***@domain`).
* `refund_order(order_id, amount)` → returns: `ok` and optional `reason`. Allow refunds only when status is `settled` or `prepping`, and only up to `total`.
* `spend_in_period(customer_id, start, end)` → returns: `total_spend`.

### Streamlit UI (required)

* **Left column:** a textarea for a question and **three preset buttons** that fill the textarea:

  1. "Total spend for **customer C‑101** from **2025‑09‑01** to **2025‑09‑30**."
  2. "Refund **5.40** credits for **order B77**."
  3. "What is the **status** and **masked email** for order `  c9  `?" (normalize to `C9` before lookup)
* **Right column:**

  * The **final answer** as plain text.
  * A **Tool calls** table with columns: `tool`, `args`, `result`.

### Input → Output (contract)

* **Input:** user question (string) or a preset.
* **Output:** a Python dict (also render as JSON in the UI) with keys:

  * `final_answer` (string)
  * `tool_calls` (list of objects with `tool`, `args`, `result`)

### Acceptance

* Presets **1–4** produce a sensible `final_answer` and **at least one** tool call is shown.
* **Preset 2** refuses the refund with a clear **reason** (amount exceeds total) if it violates your rule.
* All emails shown are **masked**.
* **Preset 5** is refused with **one sentence** and **no tool calls**.

---

## Task 3 — Receipt OCR with a Vision Model (Streamlit)

**Goal:** Upload a receipt image and get back **what was ordered** and the **total** as structured JSON.

### What your app must do

* Accept a single PNG/JPG upload.
* Call a hosted **vision** model to read the image.
* Return structured data with the line items and the total. Keep names and numbers simple.
* Display results as both JSON and a small table. Save each prediction to a local `.jsonl` file.

### Streamlit UI (required)

* **Input:** one file uploader.
* **Output:**

  * A JSON box with keys `items` and `total`.
  * A table with columns `name`, `qty`, `unit_price`, `line_total`.

### Input → Output (contract)

* **Output JSON shape:**

```json
{
  "items": [
    {"name": "...", "qty": 1, "unit_price": "0.00", "line_total": "0.00"}
  ],
  "total": "0.00"
}
```

### Acceptance

* A known demo receipt returns the correct **total** and the expected **item list**.
* If both a crossed‑out total and a current total appear, you return the **current** one.

---

## Setup Instructions

### Prerequisites

1. Python 3.8 or higher
2. OpenAI API key

### Installation

```bash
pip install -r requirements.txt
```

### Environment Setup

1. Copy `env.example` to `.env`
2. Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```

### Running the Apps

**Note:** before starting Task1, please pull qdrant service with below command. You should have docker and docker compose on your machine to get and run  service

```bash

docker compose pull

```
Run manually qdrant service

```bash
docker compose up -d 
```


```bash
# Task 1 - RAG with Citations
streamlit run part1_rag/app_rag.py

# Task 2 - Function Calling
streamlit run part2_funcs/app_funcs.py

# Task 3 - Receipt OCR
streamlit run part3_receipts/app_receipts.py
```

### Data Locations

* **Task 1:** Documentation files in `/part1_rag/docs/`
* **Task 2:** CSV data in `/part2_funcs/data/`
* **Task 3:** Demo receipts in `/part3_receipts/receipts/`

