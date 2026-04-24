# LLM Bias Mitigation Frontend

This frontend converts your notebook workflow into an interactive web UI where a user can:

- Enter custom test cases
- Run baseline and mitigated text generation
- Score outputs with a toxicity model
- View visual comparisons (bar chart + scatter chart)
- Export results as CSV

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run app.py
```

3. Open the local URL shown in terminal (usually `http://localhost:8501`).

## Notes

- First run will download model weights (`gpt2` and `facebook/roberta-hate-speech-dynabench-r4-target`).
- In low-memory systems, first load can take a while.
- This UI is an experimentation dashboard, not a production moderation system.
