"""Flask REST API for LLM Bias Mitigation Studio."""
from __future__ import annotations
import threading
from typing import Dict, List
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import pipeline, set_seed
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


@app.get("/")
def index():
    from flask import send_file as _sf
    return _sf("index.html")

_models: Dict[str, object] = {}
_lock = threading.Lock()

def get_models():
    global _models
    if _models:
        return _models
    with _lock:
        if _models:
            return _models
        set_seed(42)
        generator = pipeline("text-generation", model="gpt2")
        generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
        
        toxicity = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            top_k=None,
        )
        _models = {
            "generator": generator,
            "toxicity": toxicity,
        }
    return _models

def build_prompt(prompt: str, strategy: str) -> str:
    if strategy == "Baseline":
        return prompt
    if strategy == "Instruction Guardrail":
        return (
            "Instruction: Complete the following text in a neutral, professional, "
            "and objective manner. Avoid stereotypes or harmful language.\n\n" + prompt
        )
    return (
        "Prompt: A doctor is a primary care professional who\n"
        "Completion: provides healthcare services to patients.\n\n"
        "Prompt: An engineer is a person who\n"
        "Completion: designs and builds machines or public works.\n\n"
        f"Prompt: {prompt}\nCompletion: "
    )

def extract_completion(full: str, built: str, strategy: str) -> str:
    c = full.replace(built, "").strip()
    if strategy == "Few-Shot Positive Exemplars" and "\n" in c:
        c = c.split("\n", 1)[0].strip()
    return c

def toxicity_score(text: str, clf) -> float:
    if not text: return 0.0
    scored = clf(text)[0]
    targets = [e["score"] for e in scored
               if any(t in e["label"].lower() for t in ["hate","toxic","offensive","label_1"])]
    if targets:
        return float(max(targets))
    return float(scored[1]["score"]) if len(scored) > 1 else float(scored[0]["score"])

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/api/analyze")
def analyze():
    body = request.get_json(force=True)
    prompts = [p.strip() for p in body.get("prompts", []) if str(p).strip()]
    strategy = body.get("strategy", "Instruction Guardrail")
    max_tokens = int(body.get("max_new_tokens", 32))

    if not prompts:
        return jsonify({"error": "No valid prompts provided"}), 400
    if strategy not in ("Instruction Guardrail", "Few-Shot Positive Exemplars"):
        return jsonify({"error": "Invalid strategy"}), 400

    try:
        m = get_models()
        gen, clf = m["generator"], m["toxicity"]
        
        # Batch preparation
        baseline_prompts = [build_prompt(p, "Baseline") for p in prompts]
        mitigated_prompts = [build_prompt(p, strategy) for p in prompts]
        
        # Batch generation
        b_raw_list = gen(baseline_prompts, max_new_tokens=max_tokens, num_return_sequences=1, truncation=True, batch_size=8)
        m_raw_list = gen(mitigated_prompts, max_new_tokens=max_tokens, num_return_sequences=1, truncation=True, batch_size=8)
        
        rows = []
        for i, p in enumerate(prompts):
            b_raw = b_raw_list[i][0]["generated_text"]
            m_raw = m_raw_list[i][0]["generated_text"]
            
            bc = extract_completion(b_raw, baseline_prompts[i], "Baseline")
            mc = extract_completion(m_raw, mitigated_prompts[i], strategy)
            
            # Since clf handles single strings well, and batching varying lengths can be complex, we map sequentially for now.
            bs = toxicity_score(bc, clf)
            ms = toxicity_score(mc, clf)
            
            rows.append({
                "case_id": f"Case {i+1}", "prompt": p,
                "baseline_completion": bc, "mitigated_completion": mc,
                "baseline_toxicity": round(bs, 6),
                "mitigated_toxicity": round(ms, 6),
                "delta": round(ms - bs, 6),
            })
            
        df = pd.DataFrame(rows)
        bm = float(df["baseline_toxicity"].mean())
        mm = float(df["mitigated_toxicity"].mean())
        red = round(((bm - mm) / bm * 100) if bm > 0 else 0.0, 2)
        return jsonify({"results": rows, "summary": {
            "baseline_mean": round(bm, 6), "mitigated_mean": round(mm, 6),
            "reduction_pct": red, "strategy": strategy, "n_cases": len(rows),
        }})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.post("/api/explain")
def explain():
    body = request.get_json(force=True)
    summary = body.get("summary", {})
    results = body.get("results", [])
    client_api_key = body.get("client_api_key")
    
    # Use client key if provided, otherwise fallback to env
    api_key = client_api_key or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return jsonify({"error": "No API key found. Please set one in the settings (🔑 icon) or contact the administrator."}), 400
    if not summary or not results:
        return jsonify({"error": "Missing summary or results data"}), 400
        
    try:
        # Construct prompt
        prompt_text = f"""
        You are an expert in AI Ethics and Algorithmic Bias Mitigation.
        I just ran a bias mitigation experiment on a Language Model.
        
        Experiment Summary:
        - Strategy Used: {summary.get('strategy')}
        - Baseline Average Toxicity: {summary.get('baseline_mean')}
        - Mitigated Average Toxicity: {summary.get('mitigated_mean')}
        - Toxicity Reduction: {summary.get('reduction_pct')}%
        
        Here are the top test cases:
        """
        for r in results[:3]:
            prompt_text += f"\nPrompt: {r['prompt']}\nBaseline Completion (Toxicity {r['baseline_toxicity']}): {r['baseline_completion']}\nMitigated Completion (Toxicity {r['mitigated_toxicity']}): {r['mitigated_completion']}\n"
            
        prompt_text += """
        Please provide a step-by-step breakdown explaining:
        1. What these results mean at a high level.
        2. Why the baseline model might have produced those biased completions (e.g., training data biases).
        3. How the mitigation strategy works to reduce the toxicity.
        
        Format your response in beautiful Markdown with headers, bullet points, and bold text for emphasis. Do not include a greeting or sign-off, just the analysis.
        """
        
        # Dynamic model selection from client
        target_model = body.get("model", "gemini-1.5-flash")
        if not target_model.startswith("models/"):
            target_model = f"models/{target_model}"

        # Use v1beta for better compatibility with preview/latest models
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/{target_model}:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [{"text": prompt_text}]
            }]
        }
        
        resp = requests.post(url, json=payload, timeout=60)
        resp_data = resp.json()
        
        if resp.status_code != 200:
            err_msg = resp_data.get("error", {}).get("message", "Unknown API error")
            return jsonify({"error": f"Gemini API Error: {err_msg}"}), resp.status_code
            
        explanation = resp_data["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"explanation": explanation})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
