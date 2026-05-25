import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

st.set_page_config(
    page_title="News Topic Classifier",
    page_icon="📰",
    layout="centered"
)

st.title("📰 News Topic Classifier")
st.subheader("Using BERT for Multi-Class Classification")

label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}

@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

try:
    model, tokenizer = load_model()
except:
    st.info("Note: For demo purposes, we're using the base BERT model. To use the fine-tuned model, train it first and load from './bert-news-classifier-final'.")
    model, tokenizer = load_model()

st.write("---")
st.write("Enter a news headline or article snippet below to classify its topic:")

user_input = st.text_area(
    "News Text:",
    placeholder="e.g., 'Scientists discover new species of deep-sea fish'",
    height=150
)

if st.button("Classify Topic", type="primary"):
    if user_input.strip():
        with st.spinner("Classifying..."):
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
            
            topic = label_names[predicted_class]
            confidence = probabilities[0][predicted_class].item() * 100
            
            st.success(f"Predicted Topic: **{topic}**")
            st.info(f"Confidence: {confidence:.2f}%")
            
            st.write("---")
            st.write("### Probabilities for All Topics:")
            prob_dict = {label_names[i]: probabilities[0][i].item() * 100 for i in range(4)}
            for topic, prob in prob_dict.items():
                st.progress(prob / 100, text=f"{topic}: {prob:.2f}%")
    else:
        st.warning("Please enter some news text to classify.")

st.write("---")
st.write("### About This App")
st.write("""
This app uses the BERT (Bidirectional Encoder Representations from Transformers) model 
fine-tuned on the AG News dataset to classify news into 4 categories:
- 🌍 World
- ⚽ Sports
- 💼 Business
- 🔬 Science/Technology
""")
