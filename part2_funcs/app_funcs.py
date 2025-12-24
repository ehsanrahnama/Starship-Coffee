import os
import json
import re
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

HF_TOKEN = os.getenv("HF_API_TOKEN")

# HF_MODEL = "HuggingFaceH4/zephyr-7b-beta:featherless-ai"
HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai"


customers = pd.read_csv("part2_funcs/data/customers.csv")
orders = pd.read_csv("part2_funcs/data/orders.csv")


def forbidden(q: str) -> bool:
    banned = ["reveal email", "all email","dump", "export", "all data"]
    return any(b in q.lower() for b in banned)


def mask_email(email):
    name, domain = email.split("@")
    star_mask =(len(name)-1)*"*"
    return f"{name[0]}{star_mask}@{domain}"

def get_order(order_id):
    oid = str(order_id).upper() 
    o = orders[orders.order_id == oid]
    if o.empty:
        return {"error": "Order not found"}

    o = o.iloc[0]
    c = customers[customers.customer_id == o.customer_id].iloc[0]

    return {
        "status": o.status,
        "total": float(o.total),
        "email": mask_email(c.email)
    }

def refund_order(order_id, amount):
    oid = str(order_id).upper()
    o = orders[orders.order_id == oid]
    if o.empty:
        return {"ok": False, "reason": "Order not found"}

    o = o.iloc[0]
    if o.status not in ["settled", "prepping"]:
        return {"ok": False, "reason": "Order not refundable"}
    if amount > float(o.total):
        return {"ok": False, "reason": "Amount exceeds order total"}

    return {"ok": True}

def spend_in_period(customer_id, start, end):
    df = orders[orders.customer_id == customer_id].copy()
    df.created_at = pd.to_datetime(df.created_at)
    df = df[(df.created_at >= start) & (df.created_at <= end)]
    return {"total_spend": float(df.total.sum())}


### Test Logic 
# print(spend_in_period("C-101", "2025-09-01", "2025-09-30"))
# print(refund_order("B77", 5.40))
# print(get_order("c9"))


TOOLS = {
    "get_order": get_order,
    "refund_order": refund_order,
    "spend_in_period": spend_in_period
}



def llm_route(question):

    system_prompt = """You are a helpful assistant that maps user questions to tool calls.
    You must output ONLY a JSON LIST of tool objects.
    Do not write any code, explanations, or Python definitions.
    
    Available Tools:
    - get_order(order_id): Returns full order details (status, total, email). order_id is uppercase.
    - refund_order(order_id, amount): Refund an order. amount is a float.
    - spend_in_period(customer_id, start, end): Calculate spend. dates in YYYY-MM-DD.
    - refuse(): Use if the request is unsafe, malformed, or ambiguous.

    Output Format:
    [{"tool": "tool_name", "args": {"arg1": "value1"}}]
    
    IMPORTANT:
    Rules:
    - Do NOT invent tools (e.g., no `get_email`, no `get_status`). `get_order` returns ALL details.
    - To get EMAIL or STATUS, use `get_order`.
    - Return a LIST `[...]` containing one object.
    - Do NOT nest `args` inside `args`. The structure is flat: `{"tool": "...", "args": {...}}`.
    - For example, do not return this: `{"tool": "refund_order", "args": {"order_id": "B77", "args": {"amount": 5.40}}}`.
    - For example, do return this: `{"tool": "refund_order", "args": {"order_id": "B77", "amount": 5.40}}`.

    Examples:
    User: Refund order B77 for 5.40
    Output: [{"tool": "refund_order", "args": {"order_id": "B77", "amount": 5.40}}]
    
    User: Status of order C9
    Output: [{"tool": "get_order", "args": {"order_id": "C9"}}]
    """

    user_message = f"User Request: {question}\nOutput JSON:"

    try:

        client = OpenAI(base_url="https://router.huggingface.co/v1",
                    api_key=HF_TOKEN)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]


        response = client.chat.completions.create(
            model=HF_MODEL,
            messages=messages,
            temperature=0.01
            )

        text = ""
        # Extract text from response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            text = response.choices[0].message.content
        else:
            text = str(response)
   
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return data

    except Exception as e:
        print(f"API Error: {e}")
        return {"tool": "refuse"}
    





st.title("Task 2 — Function Calling")

if "q" not in st.session_state:
    st.session_state.q = ""

left, right = st.columns(2)

with left:
    if st.button("Total Spend in Period 1"):
        st.session_state.q = "Total spend for customer C-101 from 2025-09-01 to 2025-09-30."
    if st.button("Refund Order 2"):
        st.session_state.q = "Refund 5.40 credits for order B77."
    if st.button("Get Order Info 3"):
        st.session_state.q = "What is the status and masked email for order c9?"

    question = st.text_area("Question", st.session_state.q)

with right:
    tool_calls, final = [], ""

    if question:
        if forbidden(question):
            final = "I can’t help with that request."
        else:
            d = llm_route(question)
            tool = d.get("tool")

            if tool in TOOLS:
                args = d.get("args", {})
                result = TOOLS[tool](**args)
                tool_calls.append({"tool": tool, "args": args, "result": result})
                final = str(result)
            else:
                final = "I can’t help with that request."

    st.write("Final Answer")
    st.write(final)

    st.write("Tool Calls")
    if tool_calls:
        st.table(tool_calls)

    st.json({
        "final_answer": final,
        "tool_calls": tool_calls
    })