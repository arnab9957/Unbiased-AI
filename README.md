<div align="center">
  
# ⚖️ LLM Bias Mitigation Studio
**A powerful tool to detect, analyze, and mitigate algorithmic bias in Large Language Models.**

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.40+-orange.svg)
![Gemini AI](https://img.shields.io/badge/Gemini_AI-2.5_Flash-purple.svg)

</div>

## 📖 Overview

The **LLM Bias Mitigation Studio** is an interactive web-based toolkit designed to audit and reduce biased outputs in LLM generation. It allows researchers, developers, and AI ethicists to run controlled experiments by comparing a model's baseline behavior against prompt-based mitigation strategies. 

The application utilizes **Hugging Face Transformers** for text generation (`gpt2`) and toxicity classification (`roberta-hate-speech`), presenting the results through a sleek, glassmorphism-inspired dashboard. Additionally, it integrates **Google's Gemini 2.5 Flash** to provide expert-level, step-by-step Explanatory AI (XAI) insights into *why* the bias occurred and *how* it was mitigated.

---

## ✨ Key Features

- **🎯 Interactive Test Cases:** Input custom prompts to evaluate demographic bias across categories like gender, race, nationality, age, disability, and politics.
- **🛡️ Mitigation Strategies:** Apply real-time mitigation guardrails including:
  - *Instruction Guardrail:* Prepending strict behavioral guidelines to the prompt.
  - *Few-Shot Positive Exemplars:* Providing unbiased context examples to guide the model.
- **📊 Real-time Toxicity Scoring:** Uses Facebook's `roberta-hate-speech-dynabench-r4-target` pipeline to score the generated text for toxicity and hate speech.
- **📈 Advanced Visualizations:** Side-by-side performance metrics, interactive bar charts for case-by-case analysis, and scatter plots comparing baseline vs. mitigated toxicity.
- **✨ Gemini AI Explainer:** Automatically connects to the `gemini-2.5-flash` model to break down complex biases into digestible, step-by-step Markdown reports.
- **📥 Data Export:** Download comprehensive CSV reports of your experiment's prompts, completions, and delta metrics.

---

## 🛠️ Technology Stack

| Category | Technologies |
| --- | --- |
| **Frontend** | HTML5, Vanilla JS, CSS3 (Glassmorphism UI), Chart.js, Marked.js |
| **Backend API** | Python, Flask, Flask-CORS, Pandas |
| **Machine Learning** | PyTorch, Hugging Face `transformers` |
| **GenAI / XAI** | Google Generative AI SDK (`gemini-2.5-flash`) |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Clone and Install Dependencies
Navigate to the project directory and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Setup API Keys
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 4. Run the Application
Start the Flask backend server:
```bash
python api.py
```
The server will start on `http://localhost:5050`.

### 5. Access the Dashboard
Open the `index.html` file in your preferred web browser, or navigate directly to `http://localhost:5050/` since the Flask app serves the static frontend as well!

---

## 💡 How It Works (Architecture Flow)

1. **User Input:** The user provides a set of potentially triggering prompts and selects a mitigation strategy in the frontend UI.
2. **API Request:** The data is sent to the `/api/analyze` endpoint.
3. **Generation:** The backend formats the baseline prompts and mitigated prompts. It queries `gpt2` to generate completions for both.
4. **Scoring:** The raw completions are cleaned and passed to the `roberta-hate-speech` classifier to extract a normalized toxicity score (0 to 1).
5. **UI Rendering:** The results are sent back to the frontend, calculating the relative reduction and plotting the data using `Chart.js`.
6. **AI Explanation:** When the user clicks "Generate Explanation", the payload is sent to `/api/explain`. The Gemini 2.5 Flash model processes the statistics and generates an educational breakdown of the training data biases and mitigation efficacy.

---

## 📂 Project Structure

```text
📦 LLM Bias Mitigation Studio
 ┣ 📂 static/
 ┃ ┣ 📜 chart.umd.min.js      # Local dependency for chart visualizations
 ┃ ┗ 📜 marked.min.js         # Local dependency for rendering Gemini Markdown
 ┣ 📜 .env                    # Environment variables (API Keys)
 ┣ 📜 .gitignore              # Git ignore file
 ┣ 📜 api.py                  # Core Flask backend, LLM pipelines, and routing
 ┣ 📜 index.html              # Frontend UI dashboard
 ┣ 📜 requirements.txt        # Python dependency manifest
 ┗ 📜 README.md               # Project documentation
```

---

<div align="center">
  <i>Built for advancing AI Ethics and building safer Language Models.</i>
</div>
