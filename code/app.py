from flask import Flask, render_template, request
import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from collections import Counter

app = Flask(__name__)

# ========== Custom Ensemble Model for Medical Category ==========
class CustomModel(torch.nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base_model = base_model
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ========== Load Ensemble Category Models ==========
model_paths = [
    "models/AraBERT",
    "models/BioBert (2)",
    "models/distilBert",
    "models/multiBert",
    "models/xlmRoBERTaa"
]

models, tokenizers = [], []

# Load class labels
with open(f"{model_paths[0]}/category_mapping.pkl", "rb") as f:
    category_mapping = pickle.load(f)
label_to_category = {v: k for k, v in category_mapping.items()}

# Load all models + tokenizers
for path in model_paths:
    tokenizer = AutoTokenizer.from_pretrained(path)
    base_model = AutoModel.from_pretrained(path)

    with open(f"{path}/classifier_state.pt", "rb") as f:
        state = torch.load(f, map_location=torch.device('cpu'))

    model = CustomModel(base_model, num_labels=state['num_labels'])
    model.classifier.load_state_dict(state['classifier_state'])
    model.eval()

    models.append(model)
    tokenizers.append(tokenizer)

# ========== Load Severity Model (no ensemble) ==========
severity_model_path = r"C:\Users\alie0\Downloads\saved_model_full20 (1)"
severity_model = AutoModelForSequenceClassification.from_pretrained(severity_model_path)
severity_tokenizer = AutoTokenizer.from_pretrained(severity_model_path)

# Severity labels
severity_class_labels = {
    0: "غير حرج",
    1: "حرج"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
severity_model = severity_model.to(device).eval()

# ========== Routes ==========

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    input_text = ""  # this keeps the user's question

    if request.method == 'POST':
        input_text = request.form['text']
        action = request.form['action']

        if not input_text.strip():
            prediction = "⚠️ الرجاء إدخال النص"
        elif action == "category":
            prediction = predict_category(input_text)
        elif action == "severity":
            prediction = predict_severity(input_text)

    return render_template("index.html", prediction=prediction, input_text=input_text)


# ========== Prediction Functions ==========

def predict_category(text):
    votes = []
    for model, tokenizer in zip(models, tokenizers):
        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs)
            pred_idx = int(torch.argmax(logits, dim=1))
            category = label_to_category[pred_idx]
            votes.append(category)

    most_common = Counter(votes).most_common(1)[0][0]
    return f"📋 التصنيف الطبي: {most_common}"

def predict_severity(text):
    inputs = severity_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = severity_model(**inputs)
        logits = outputs.logits
        class_id = torch.argmax(logits, dim=-1).item()
        return f"🚨 درجة الخطورة: {severity_class_labels.get(class_id, 'غير معروف')}"

# ========== Run ==========

if __name__ == '__main__':
    app.run(debug=True)
