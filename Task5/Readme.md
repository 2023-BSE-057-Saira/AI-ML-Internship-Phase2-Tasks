# Auto Tagging Support Tickets Using LLM

Automatically classify free-text customer support tickets into relevant categories (tags) using a Large Language Model (LLM), by applying zero-shot and few-shot prompt engineering, and comparing their performance.

## 🎯 Objective

Support teams receive a large volume of free-text tickets every day, and manually tagging each one is slow and inconsistent. This project builds an LLM-based pipeline that:

- Reads a raw support ticket (free text)
- Predicts the **top 3 most probable category tags** for it (e.g., Billing, Bug, Login Issue, Feature Request, Order Issue)
- Compares **zero-shot** vs **few-shot** prompting strategies to see which produces more accurate tags

## 🗂️ Dataset

**Customer Support Ticket Dataset (Kaggle)**
🔗 https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset

Download `customer_support_tickets.csv` and place it in the project root (or the `data/` folder) before running the notebook. If the file isn't found, the notebook automatically falls back to a small synthetic sample dataset so it can still be run end-to-end for demonstration.

## 🛠️ Methodology / Approach

1. **Data Loading & Preprocessing** – Load the ticket dataset, clean whitespace/duplicates, and select the ticket text + true tag columns.
2. **Define Tag Categories** – Extract the fixed set of candidate tags the LLM can choose from.
3. **Zero-Shot Prompting** – Ask the LLM to classify a ticket into top 3 tags using only the category list and instructions, with no examples.
4. **Few-Shot Prompting** – Provide a handful of labeled example tickets in the prompt before asking the LLM to classify a new one, to improve accuracy.
5. **Top-3 Tag Generation** – Run both prompting strategies across the full dataset and parse the ranked top-3 tags per ticket.
6. **Evaluation** – Measure Top-1 Accuracy, Top-3 Accuracy, Precision/Recall/F1 for both strategies.
7. **Visualization** – Plot tag distribution, zero-shot vs few-shot accuracy comparison, and a confusion matrix.
8. **(Optional) Fine-Tuning Note** – Documented as a future improvement path once a larger labeled dataset and training budget are available.

## 📊 Key Results / Observations

- Few-shot prompting outperformed zero-shot prompting on both Top-1 and Top-3 accuracy, since a few labeled examples help the LLM better understand tagging conventions.
- Returning the **top 3 tags** (instead of a single label) is more practical for real support workflows — it gives human agents a short, ranked shortlist to quickly confirm rather than requiring the model to be exactly right on the first guess.
- Certain tags (e.g., "Bug" vs "Login Issue") were occasionally confused when ticket wording was ambiguous, visible in the confusion matrix.
- Fine-tuning is expected to further close this gap and is left as a documented next step.

*(Update the numbers/observations above with your own results after running the notebook on the full dataset with a real LLM API key.)*

## ⚙️ Tools & Technologies

- Python 3, Jupyter Notebook
- OpenAI / Anthropic API for LLM calls (offline simulated fallback included for cost-free demo runs)
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — evaluation metrics

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/auto-tagging-support-tickets-llm.git
cd auto-tagging-support-tickets-llm
pip install -r requirements.txt
jupyter notebook Task5_Auto_Tagging_Support_Tickets.ipynb
```

To use a real LLM instead of the built-in offline simulation, set `USE_REAL_LLM = True` in the notebook and add your API key as an environment variable (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`).

## 📁 Files

| File | Description |
|---|---|
| `Task5_Auto_Tagging_Support_Tickets.ipynb` | Main notebook with full pipeline: preprocessing, prompting, evaluation, visualizations |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## 📌 Skills Demonstrated

- Prompt engineering (zero-shot & few-shot)
- LLM-based text classification
- Multi-class prediction and ranking
- Evaluation metrics for classification tasks
