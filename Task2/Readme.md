# News Topic Classifier Using BERT

Fine-tune a transformer model (BERT) to classify news headlines into topic categories, evaluate it, and deploy it as a live interactive app.

## 🎯 Objective
News platforms need to automatically categorize incoming articles/headlines (e.g., World, Sports, Business, Sci/Tech) at scale. This project fine-tunes `bert-base-uncased` on the AG News dataset to perform this classification, then deploys the trained model as a simple web app for live interaction.

## 🗂️ Dataset
**AG News Dataset** (Hugging Face Datasets)
🔗 https://huggingface.co/datasets/ag_news

4 balanced classes: World, Sports, Business, Sci/Tech.

## 🛠️ Methodology / Approach
1. **Data Loading** – Load the AG News dataset via the Hugging Face `datasets` library.
2. **Tokenization & Preprocessing** – Tokenize headlines using the `bert-base-uncased` tokenizer (padding, truncation, attention masks).
3. **Model Fine-Tuning** – Load `bert-base-uncased` with a classification head and fine-tune it on the training split using Hugging Face `Trainer` / `TrainingArguments`.
4. **Evaluation** – Evaluate on the test split using Accuracy and weighted/macro F1-score; inspect a confusion matrix for per-class performance.
5. **Deployment** – Wrap the fine-tuned model in a Streamlit or Gradio app so a user can type a headline and get an instant predicted category.

## 📊 Key Results / Observations
- Fine-tuned BERT achieved strong accuracy (~94%+ is typical on AG News) and high F1-scores across all 4 classes.
- Most misclassifications occurred between semantically close categories (e.g., Business vs Sci/Tech headlines about companies).
- Transfer learning from a pretrained BERT model required far fewer epochs than training a classifier from scratch.

*(Replace with your actual accuracy/F1 numbers after training.)*

## ⚙️ Tools & Technologies
- Python 3, Jupyter Notebook
- Hugging Face Transformers & Datasets
- PyTorch
- scikit-learn (metrics)
- Streamlit or Gradio (deployment)

## 🚀 How to Run
```bash
git clone https://github.com/<your-username>/news-topic-classifier-bert.git
cd news-topic-classifier-bert
pip install -r requirements.txt
jupyter notebook News_Topic_Classifier_BERT.ipynb
```

To launch the deployed app:
```bash
streamlit run app.py
# or
python app_gradio.py
```

## 📁 Files
| File | Description |
|---|---|
| `Task1_Phase2.ipynb` | Full pipeline: preprocessing, fine-tuning, evaluation |
| `app.py` | Streamlit/Gradio app for live predictions |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

## 📌 Skills Demonstrated
- NLP using Transformers
- Transfer learning & fine-tuning
- Evaluation metrics for text classification (Accuracy, F1-score)
- Lightweight model deployment
