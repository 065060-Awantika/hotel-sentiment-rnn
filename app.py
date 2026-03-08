import streamlit as st
import numpy as np
import re
import joblib
import time
import os

st.set_page_config(page_title="Hotel Sentiment Analyzer", page_icon="🏨", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2rem; font-weight:700; color:#2C3E50; text-align:center; }
.sub-header  { font-size:1rem; color:#7F8C8D; text-align:center; margin-bottom:2rem; }
.positive { background:#D5F5E3; color:#1E8449; border:2px solid #27AE60;
            padding:1.5rem; border-radius:12px; text-align:center; font-size:1.4rem; font-weight:700; }
.neutral  { background:#FEF9E7; color:#B7950B; border:2px solid #F1C40F;
            padding:1.5rem; border-radius:12px; text-align:center; font-size:1.4rem; font-weight:700; }
.negative { background:#FADBD8; color:#CB4335; border:2px solid #E74C3C;
            padding:1.5rem; border-radius:12px; text-align:center; font-size:1.4rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        model     = tf.keras.models.load_model('lstm_hotel_sentiment.h5')
        tokenizer = joblib.load('tokenizer.pkl')
        le        = joblib.load('label_encoder.pkl')
        config    = joblib.load('config.pkl')
        return model, tokenizer, le, config, True
    except Exception as e:
        return None, None, None, None, False

model, tokenizer, le, config, model_loaded = load_model()

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_text(text):
    text = str(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    fixes = {"won't":"will not","can't":"cannot","n't":" not",
             "'re":" are","'ve":" have","'ll":" will","'d":" would","'m":" am"}
    for k, v in fixes.items():
        text = text.replace(k, v)
    return text

# ── Predict ───────────────────────────────────────────────────────────────────
def predict_sentiment(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    cleaned  = preprocess_text(text)
    seq      = tokenizer.texts_to_sequences([cleaned])
    padded   = pad_sequences(seq, maxlen=config['MAX_LEN'], padding='post', truncating='post')
    t0       = time.time()
    probs    = model.predict(padded, verbose=0)[0]
    latency  = (time.time() - t0) * 1000
    idx      = np.argmax(probs)
    label    = le.inverse_transform([idx])[0]
    conf     = probs[idx] * 100
    all_p    = {le.inverse_transform([i])[0]: float(probs[i])*100 for i in range(len(probs))}
    return label, conf, all_p, latency

# ── VADER fallback ────────────────────────────────────────────────────────────
def vader_predict(text):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    score = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
    if score >= 0.05:    return 'Positive', abs(score)*100, {'Positive':abs(score)*100,'Neutral':0,'Negative':0}, 0
    elif score <= -0.05: return 'Negative', abs(score)*100, {'Positive':0,'Neutral':0,'Negative':abs(score)*100}, 0
    else:                return 'Neutral',  50.0,           {'Positive':0,'Neutral':50,'Negative':0}, 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏨 About")
    st.markdown("""
**Model:** Stacked LSTM
**Task:** Multi-Class Sentiment
**Classes:** Positive | Neutral | Negative
**Dataset:** TripAdvisor Reviews
**Embeddings:** GloVe 100d
    """)
    st.divider()
    st.markdown("### 🧠 Architecture")
    st.code("""GloVe Embedding (100d)
       ↓
LSTM Layer 1 (128 units)
       ↓
Dropout (0.4)
       ↓
LSTM Layer 2 (64 units)
       ↓
Dense (64, ReLU)
       ↓
Dropout (0.3)
       ↓
Softmax (3 classes)""")
    st.divider()
    st.markdown("### ⚙️ Hyperparameters")
    st.markdown("""
| Param | Value |
|---|---|
| Max Seq Len | 200 |
| Vocab Size | 15,000 |
| Embed Dim | 100 |
| LSTM-1 | 128 units |
| LSTM-2 | 64 units |
| Batch Size | 64 |
| Optimizer | Adam |
| LR | 0.001 |
    """)
    st.divider()
    if model_loaded:
        st.success("✅ LSTM Model Loaded")
    else:
        st.warning("⚠️ Using VADER fallback")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🏨 Hotel Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">LSTM Multi-Class Sentiment Analysis | Business Decision Support</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Review", "📋 Batch Analysis", "📊 Project Info"])

# ─ Tab 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Enter Review Text")
        review_input = st.text_area("Paste a hotel or restaurant review:",
            height=200,
            placeholder="e.g. The room was clean and staff was very helpful...")

        examples = {
            "😊 Positive": "Absolutely loved our stay! Room spotless, staff incredibly friendly. Breakfast was amazing. Will definitely return!",
            "😐 Neutral":  "Hotel was okay. Room decent but nothing special. Location good. Service average.",
            "😞 Negative": "Terrible experience. Room dirty, AC broken, staff rude. Waited 30 mins for check-in. Would not recommend."
        }
        st.markdown("**Quick Examples:**")
        c1, c2, c3 = st.columns(3)
        for btn_col, (lbl, txt) in zip([c1,c2,c3], examples.items()):
            if btn_col.button(lbl, use_container_width=True):
                review_input = txt

        analyze = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    with col2:
        if analyze and review_input.strip():
            with st.spinner("Analyzing..."):
                if model_loaded:
                    label, conf, all_p, latency = predict_sentiment(review_input)
                else:
                    label, conf, all_p, latency = vader_predict(review_input)

            emoji = {'Positive':'😊','Neutral':'😐','Negative':'😞'}
            css   = {'Positive':'positive','Neutral':'neutral','Negative':'negative'}
            st.markdown(f'<div class="{css[label]}">{emoji[label]} {label}<br><span style="font-size:1rem">Confidence: {conf:.1f}%</span></div>', unsafe_allow_html=True)

            st.markdown("**Class Probabilities:**")
            for cls in ['Positive','Neutral','Negative']:
                p = all_p.get(cls, 0)
                st.markdown(f"**{cls}** — {p:.1f}%")
                st.progress(p/100)

            st.divider()
            actions = {
                'Positive': "✅ Share in marketing. Invite to loyalty program.",
                'Neutral':  "📧 Send personalized follow-up. Offer upgrade.",
                'Negative': "🚨 Escalate to manager immediately within 2 hours."
            }
            st.info(f"💼 **Action:** {actions[label]}")
            if latency > 0:
                st.caption(f"⚡ Latency: {latency:.1f}ms")

        elif analyze:
            st.warning("Please enter a review first.")
        else:
            st.info("👈 Enter a review and click Analyze")

# ─ Tab 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Review Analysis")
    batch_input = st.text_area("Enter reviews (one per line):", height=200,
        placeholder="The room was excellent.\nService was okay.\nHorrible experience!")

    if st.button("🔍 Analyze All", type="primary"):
        if batch_input.strip():
            reviews  = [r.strip() for r in batch_input.split('\n') if r.strip()]
            results  = []
            progress = st.progress(0)
            for i, rev in enumerate(reviews):
                fn = predict_sentiment if model_loaded else vader_predict
                lbl, conf, _, _ = fn(rev)
                results.append({
                    'Review':    rev[:80]+'...' if len(rev)>80 else rev,
                    'Sentiment': lbl,
                    'Confidence':f"{conf:.1f}%",
                    'Action':    {'Positive':'Market ✅','Neutral':'Follow-up 📧','Negative':'Escalate 🚨'}[lbl]
                })
                progress.progress((i+1)/len(reviews))

            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True, hide_index=True)

            counts = df['Sentiment'].value_counts()
            c1,c2,c3 = st.columns(3)
            c1.metric("😊 Positive", counts.get('Positive',0))
            c2.metric("😐 Neutral",  counts.get('Neutral',0))
            c3.metric("😞 Negative", counts.get('Negative',0))

            st.download_button("⬇️ Download CSV", df.to_csv(index=False), "results.csv", "text/csv")

# ─ Tab 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📘 Project 3 — RNN with Text Datasets")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### 🎯 Problem Statement
Classify hotel & restaurant reviews into **Positive / Neutral / Negative**
to support hospitality managers in:
- Service recovery decisions
- Marketing content curation
- Quality benchmarking

### 📦 Dataset
- **Source:** TripAdvisor Hotel Reviews (Kaggle)
- **Size:** 20,000+ reviews
- **Labels:** 1–2★ Negative | 3★ Neutral | 4–5★ Positive
- **License:** CC BY 4.0

### 🧹 Data Pipeline
1. HTML/URL/special char removal
2. Lowercasing + contraction expansion
3. Negation handling (NOT_ prefix)
4. Lemmatization (WordNet POS-aware)
5. Stopword removal (negations kept)
6. TF-IDF (Unigrams + Bigrams + Char)
7. Lexicon scores (VADER, AFINN, SentiWordNet)
8. GloVe 100d embeddings
9. Sequence padding (max_len=200)
        """)
    with c2:
        st.markdown("""
### 📊 Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC Curve & AUC (per class)
- Cohen's Kappa & Log Loss
- Inference Latency

### 🔒 Ethical Practices
- ✅ PII removed (emails, phones, names)
- ✅ Class imbalance corrected (class weights)
- ✅ Bias audit conducted
- ✅ LIME + SHAP explainability applied
- ✅ Academic use only (CC BY 4.0)

### 💼 Managerial Actions
| Sentiment | Action | Priority |
|---|---|---|
| 🔴 Negative | Escalate within 2 hours | HIGH |
| 🟡 Neutral | Follow-up + upgrade offer | MEDIUM |
| 🟢 Positive | Marketing + loyalty invite | LOW |
        """)
