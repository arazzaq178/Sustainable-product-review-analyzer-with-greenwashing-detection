# ============================================================
# STREAMLIT APP — Greenwashing Detector
# Run karne ka tarika Colab mein:
#
# !pip install streamlit pyngrok -q
# from pyngrok import ngrok
# ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # ngrok.com se free token lo
# !streamlit run streamlit_app.py &
# public_url = ngrok.connect(8501)
# print(public_url)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import sys
import pickle
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🌿 GreenCheck — Greenwashing Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        padding: 2rem; border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .header-box h1 { color: white !important; font-size: 2.2rem !important; margin: 0 !important; }
    .header-box p  { color: #d8f3dc !important; margin: 0.3rem 0 0 0 !important; }

    .score-card {
        background: white; border-radius: 10px;
        padding: 1.2rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .verdict-eco      { background: #d4edda; color: #155724; padding: 0.8rem 1.2rem; border-radius: 8px; font-weight: 600; text-align: center; }
    .verdict-greenwash{ background: #f8d7da; color: #721c24; padding: 0.8rem 1.2rem; border-radius: 8px; font-weight: 600; text-align: center; }
    .verdict-warning  { background: #fff3cd; color: #856404; padding: 0.8rem 1.2rem; border-radius: 8px; font-weight: 600; text-align: center; }

    .eco-badge  { display: inline-block; background: #d4edda; color: #155724; padding: 3px 10px; border-radius: 12px; margin: 2px; font-size: 0.82rem; font-weight: 600; }
    .gw-badge   { display: inline-block; background: #f8d7da; color: #721c24; padding: 3px 10px; border-radius: 12px; margin: 2px; font-size: 0.82rem; font-weight: 600; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_all_models():
    """Models load karo — sirf ek baar"""

    # Import all classes
    # Note: Colab mein cells ke baad yeh classes available hoti hain
    # Agar standalone run karo to import karo

    try:
        preprocessor_obj = TextPreprocessor()
    except NameError:
        # Fallback minimal preprocessor
        class MinimalPreprocessor:
            def process(self, text):
                import re
                cleaned = re.sub(r'[^\w\s]', ' ', text.lower()).strip()
                eco_kws = ['certified', 'organic', 'biodegradable', 'recycled', 'compostable', 'solar', 'renewable']
                gw_kws  = ['all natural', 'chemical free', 'pure', 'earth friendly']
                eco_score   = sum(2 for k in eco_kws if k in text.lower())
                gw_score    = sum(1 for k in gw_kws if k in text.lower())
                return {
                    'cleaned_text'     : cleaned,
                    'eco_score'        : eco_score - gw_score,
                    'keyword_count'    : eco_score,
                    'has_certification': int('certified' in text.lower()),
                    'has_contradiction': 0,
                    'vague_claim_count': gw_score
                }
        preprocessor_obj = MinimalPreprocessor()

    # Load NLP model
    nlp_trainer_obj = None
    nlp_path = Path("models/nlp/nlp_model.pkl")
    if nlp_path.exists():
        try:
            nlp_trainer_obj = NLPTrainer()
            nlp_trainer_obj.load_model(str(nlp_path))
        except:
            pass

    # Load CNN model
    cnn_trainer_obj = None
    cnn_path = Path("models/cnn/resnet_packaging_model.pt")
    if cnn_path.exists():
        try:
            cnn_trainer_obj = CNNTrainer()
            cnn_trainer_obj.load_model(str(cnn_path))
        except:
            pass

    # Load Ensemble model
    ens_obj  = None
    ens_path = Path("models/final/greenwash_ensemble.pkl")
    if ens_path.exists():
        try:
            ens_obj = GreenwashEnsemble()
            ens_obj.load(str(ens_path))
        except:
            pass

    return preprocessor_obj, nlp_trainer_obj, cnn_trainer_obj, ens_obj


# ─────────────────────────────────────────────────────────────
# HELPER: RUN FULL PREDICTION
# ─────────────────────────────────────────────────────────────

def run_prediction(text, preprocessor_obj, nlp_trainer_obj, ens_obj, image=None, cnn_trainer_obj=None):
    """Full prediction pipeline"""

    processed = preprocessor_obj.process(text)

    # NLP prediction
    if nlp_trainer_obj:
        nlp_out = nlp_trainer_obj.predict([text])[0]
    else:
        # Simple fallback
        eco_s = processed['eco_score']
        prob  = max(0, min(1, 0.5 - eco_s * 0.1))
        nlp_out = {
            'greenwashing_probability': prob,
            'sentiment_score'         : 0.0,
            'confidence'              : 0.6,
            'is_greenwashing'         : int(prob > 0.5)
        }

    nlp_out.update({
        'eco_score'         : processed['eco_score'],
        'keyword_count'     : processed['keyword_count'],
        'has_certification' : processed['has_certification'],
        'has_contradiction' : processed['has_contradiction'],
        'vague_claim_count' : processed['vague_claim_count'],
    })

    # CNN prediction
    cnn_out = None
    if image and cnn_trainer_obj:
        cnn_out = cnn_trainer_obj.predict_image(image)

    # Final ensemble prediction
    if ens_obj:
        result = ens_obj.predict(nlp_output=nlp_out, cnn_output=cnn_out)
    else:
        # Fallback scoring
        score   = float(nlp_out['greenwashing_probability'])
        if score < 0.2:   risk, verdict = "Very Low",  "Genuinely Eco-Friendly"
        elif score < 0.4: risk, verdict = "Low",       "Likely Eco-Friendly"
        elif score < 0.6: risk, verdict = "Moderate",  "Uncertain"
        elif score < 0.8: risk, verdict = "High",      "Likely Greenwashing"
        else:             risk, verdict = "Critical",  "Strong Greenwashing Indicators"
        result = {
            'greenwashing_score'      : round(score, 4),
            'greenwashing_percentage' : round(score * 100, 1),
            'classification'          : "Greenwashing" if score > 0.5 else "Eco-Friendly",
            'verdict'                 : verdict,
            'risk_level'              : risk,
            'confidence'              : 0.6,
            'eco_score'               : round(1 - score, 4),
            'module_breakdown'        : {
                'nlp_contribution': round(score, 3),
                'cnn_contribution': 0.5,
                'rnn_contribution': 0.5
            }
        }

    return result, nlp_out, processed, cnn_out


# ─────────────────────────────────────────────────────────────
# GAUGE CHART
# ─────────────────────────────────────────────────────────────

def make_gauge(score, title="Greenwashing Risk"):
    color = "#2ecc71" if score < 0.35 else ("#f39c12" if score < 0.65 else "#e74c3c")
    fig   = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score * 100,
        title = {'text': title, 'font': {'size': 13}},
        number= {'suffix': "%", 'font': {'size': 26, 'color': color}},
        gauge = {
            'axis'     : {'range': [0, 100]},
            'bar'      : {'color': color, 'thickness': 0.28},
            'steps'    : [
                {'range': [0, 30],  'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100],'color': '#f8d7da'}
            ],
            'threshold': {'line': {'color': '#2c3e50', 'width': 3}, 'value': 50}
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0">
        <div style="font-size:2.5rem">🌿</div>
        <h2 style="color:#2d6a4f; margin:0">GreenCheck</h2>
        <p style="color:#666; font-size:0.85rem">AI Greenwashing Detector</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        ["🔍 Analyze Review", "🖼️ Image Analysis", "📊 Dashboard", "ℹ️ How It Works"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Model Status**")
    for name, path in [
        ("NLP Model",      "models/nlp/nlp_model.pkl"),
        ("CNN Model",      "models/cnn/resnet_packaging_model.pt"),
        ("RNN Model",      "models/rnn/lstm_trust_model.pt"),
        ("Ensemble Model", "models/final/greenwash_ensemble.pkl")
    ]:
        exists = Path(path).exists()
        st.markdown(f"{'✅' if exists else '⚠️'} {name}")


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────

with st.spinner("Loading AI models..."):
    try:
        preprocessor_m, nlp_trainer_m, cnn_trainer_m, ensemble_m = load_all_models()
        models_ok = True
    except Exception as e:
        st.warning(f"Model loading issue: {e} — running in demo mode")
        preprocessor_m = None
        nlp_trainer_m  = None
        cnn_trainer_m  = None
        ensemble_m     = None
        models_ok      = False


# ─────────────────────────────────────────────────────────────
# PAGE 1 — ANALYZE REVIEW
# ─────────────────────────────────────────────────────────────

if page == "🔍 Analyze Review":

    st.markdown("""
    <div class="header-box">
        <h1>🔍 Greenwashing Detector</h1>
        <p>Paste any product review — AI will detect greenwashing in seconds</p>
    </div>
    """, unsafe_allow_html=True)

    samples = {
        "Select a sample..."             : "",
        "✅ Genuine Eco (Good example)"  : "This bamboo toothbrush is USDA certified organic. B-Corp certified company, truly sustainable. Packaging is 100% compostable and plastic-free.",
        "🚨 Greenwashing (Bad example)"  : "All natural and chemical free! Our pure earth-friendly product is eco and bio with green packaging. Going green with our sustainable choice!",
        "⚠️  Ambiguous (Mixed signals)"  : "Claims to be sustainable and uses organic ingredients. Packaging says eco-friendly but no certifications visible anywhere on product."
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        sample  = st.selectbox("Try a sample:", list(samples.keys()))
        review  = st.text_area(
            "Product review:",
            value   = samples[sample],
            height  = 140,
            placeholder="Paste product review here..."
        )
        analyze = st.button("🔍 Analyze for Greenwashing", type="primary", use_container_width=True)

    with col2:
        st.markdown("**Optional: Upload Product Image**")
        uploaded = st.file_uploader(
            "Upload packaging image:",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_column_width=True, caption="Uploaded Image")
        else:
            st.info("📷 No image uploaded\nText-only analysis will run")

    if analyze and review and len(review) > 10:

        if preprocessor_m is None:
            st.error("Models not loaded. Please run training cells first.")
        else:
            with st.spinner("🤖 Running AI analysis..."):
                image_obj = Image.open(uploaded) if uploaded else None
                result, nlp_out, processed, cnn_out = run_prediction(
                    review, preprocessor_m, nlp_trainer_m,
                    ensemble_m, image_obj, cnn_trainer_m
                )

            st.divider()
            score = result['greenwashing_score']

            # Verdict Banner
            if score < 0.35:
                st.markdown(f'<div class="verdict-eco">✅ {result["verdict"].upper()} — {result["risk_level"]} RISK</div>', unsafe_allow_html=True)
            elif score < 0.65:
                st.markdown(f'<div class="verdict-warning">⚠️ {result["verdict"].upper()} — {result["risk_level"]} RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-greenwash">🚨 {result["verdict"].upper()} — {result["risk_level"]} RISK</div>', unsafe_allow_html=True)

            st.markdown("")

            # Scores row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.plotly_chart(make_gauge(score), use_container_width=True)
            with c2:
                eco = result['eco_score']
                color = "#2ecc71" if eco > 0.6 else ("#f39c12" if eco > 0.4 else "#e74c3c")
                st.markdown(f"""
                <div class="score-card" style="margin-top:0.5rem">
                    <div style="font-size:0.85rem;color:#888">Eco Score</div>
                    <div style="font-size:2.2rem;color:{color};font-weight:800">{eco*100:.0f}%</div>
                    <div style="font-size:0.75rem;color:#aaa">Authenticity</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                sent = nlp_out.get('sentiment_score', 0)
                icon = "😊" if sent > 0.2 else ("😐" if sent > -0.2 else "😞")
                st.markdown(f"""
                <div class="score-card" style="margin-top:0.5rem">
                    <div style="font-size:0.85rem;color:#888">Sentiment</div>
                    <div style="font-size:2rem">{icon}</div>
                    <div style="font-size:1rem;font-weight:700">{sent:+.2f}</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                kw = processed['keyword_count']
                kw_color = "#2ecc71" if kw > 2 else ("#f39c12" if kw > 0 else "#e74c3c")
                st.markdown(f"""
                <div class="score-card" style="margin-top:0.5rem">
                    <div style="font-size:0.85rem;color:#888">Eco Keywords</div>
                    <div style="font-size:2.2rem;color:{kw_color};font-weight:800">{kw}</div>
                    <div style="font-size:0.75rem;color:#aaa">Found</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # Module Breakdown
            bd = result['module_breakdown']
            fig = go.Figure(go.Bar(
                x  = [v * 100 for v in bd.values()],
                y  = ['🧠 NLP (BERT)', '🖼️ CNN (ResNet)', '🔄 RNN (LSTM)'],
                orientation = 'h',
                marker_color = ['#3498db', '#9b59b6', '#e67e22'],
                text = [f"{v*100:.1f}%" for v in bd.values()],
                textposition = 'outside'
            ))
            fig.update_layout(
                title="Module Contribution to Risk Score",
                xaxis=dict(range=[0, 115], title="Greenwashing Risk (%)"),
                height=200,
                margin=dict(l=10, r=60, t=40, b=10),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Keywords
            with st.expander("🔑 Keyword Analysis", expanded=True):
                eco_kws = ['certified organic', 'usda organic', 'b corp', 'biodegradable',
                           'compostable', 'recycled', 'renewable', 'fair trade', 'plastic free']
                gw_kws  = ['all natural', 'chemical free', 'pure', 'earth friendly', 'eco', 'bio', 'going green']

                found_eco = [k for k in eco_kws if k in review.lower()]
                found_gw  = [k for k in gw_kws  if k in review.lower()]

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**✅ Eco Keywords:**")
                    if found_eco:
                        st.markdown(' '.join([f'<span class="eco-badge">{k}</span>' for k in found_eco]), unsafe_allow_html=True)
                    else:
                        st.caption("None found")
                with col_b:
                    st.markdown("**⚠️ Vague Claims:**")
                    if found_gw:
                        st.markdown(' '.join([f'<span class="gw-badge">{k}</span>' for k in found_gw]), unsafe_allow_html=True)
                    else:
                        st.caption("None found")
                with col_c:
                    st.markdown("**📋 Flags:**")
                    if processed['has_certification']:
                        st.markdown("✅ Has eco certification")
                    if processed['has_contradiction']:
                        st.markdown("🚨 Contradictory claims!")
                    if processed['vague_claim_count'] > 2:
                        st.markdown("⚠️ Many vague claims")
                    if not any([processed['has_certification'], processed['has_contradiction'], processed['vague_claim_count'] > 2]):
                        st.caption("No specific flags")

    elif analyze:
        st.warning("Please enter at least 10 characters.")


# ─────────────────────────────────────────────────────────────
# PAGE 2 — IMAGE ANALYSIS
# ─────────────────────────────────────────────────────────────

elif page == "🖼️ Image Analysis":

    st.markdown("""
    <div class="header-box">
        <h1>🖼️ Packaging Image Analyzer</h1>
        <p>Upload a product image — CNN will classify packaging as eco-friendly or non-eco</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_img = st.file_uploader("Upload packaging image", type=['jpg', 'jpeg', 'png'])
        if uploaded_img:
            img = Image.open(uploaded_img)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            if st.button("🖼️ Classify Packaging", type="primary", use_container_width=True):
                if cnn_trainer_m:
                    with st.spinner("Running CNN..."):
                        res = cnn_trainer_m.predict_image(img)
                    st.session_state['cnn_result'] = res
                else:
                    st.error("CNN model not loaded. Run cell6_cnn_module.py first.")

    with col2:
        if 'cnn_result' in st.session_state:
            res = st.session_state['cnn_result']
            if res['is_eco_friendly']:
                st.success("✅ ECO-FRIENDLY PACKAGING")
            else:
                st.error("🚨 NON-ECO PACKAGING")

            fig = go.Figure(go.Bar(
                x=['🌿 Eco-Friendly', '🗑️ Non-Eco'],
                y=[res['eco_probability']*100, res['non_eco_probability']*100],
                marker_color=['#2ecc71', '#e74c3c'],
                text=[f"{res['eco_probability']*100:.1f}%", f"{res['non_eco_probability']*100:.1f}%"],
                textposition='outside'
            ))
            fig.update_layout(
                yaxis=dict(range=[0, 120], title="Probability (%)"),
                height=280, showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Confidence", f"{res['confidence']*100:.1f}%")
        else:
            st.info("Upload an image and click classify to see results.")


# ─────────────────────────────────────────────────────────────
# PAGE 3 — DASHBOARD
# ─────────────────────────────────────────────────────────────

elif page == "📊 Dashboard":

    st.markdown("""
    <div class="header-box">
        <h1>📊 ESG Analytics Dashboard</h1>
        <p>Product sustainability performance and greenwashing trends</p>
    </div>
    """, unsafe_allow_html=True)

    # Generate sample dashboard data
    np.random.seed(42)
    n   = 200
    ddf = pd.DataFrame({
        'product'            : [f'Product {i}' for i in range(n)],
        'category'           : np.random.choice(['Skincare', 'Food', 'Cleaning', 'Clothing', 'Electronics'], n),
        'greenwashing_score' : np.random.beta(2, 3, n),
        'eco_score'          : np.random.beta(3, 2, n),
        'trust_score'        : np.random.beta(4, 2, n),
        'has_certification'  : np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'review_count'       : np.random.randint(10, 500, n),
    })
    ddf['classification'] = ddf['greenwashing_score'].apply(
        lambda x: 'Greenwashing' if x > 0.5 else 'Eco-Friendly'
    )

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Products Analyzed", n)
    c2.metric("Greenwashing Rate", f"{(ddf['greenwashing_score']>0.5).mean()*100:.1f}%")
    c3.metric("Avg Eco Score",     f"{ddf['eco_score'].mean()*100:.1f}%")
    c4.metric("Certified Products",f"{ddf['has_certification'].mean()*100:.1f}%")
    c5.metric("Avg Trust Score",   f"{ddf['trust_score'].mean()*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        cat = ddf.groupby('category')['greenwashing_score'].mean().sort_values()
        fig = px.bar(x=cat.values*100, y=cat.index, orientation='h',
                     title="Greenwashing Rate by Category",
                     labels={'x': 'Score (%)', 'y': 'Category'},
                     color=cat.values, color_continuous_scale='RdYlGn_r')
        fig.update_layout(height=300, coloraxis_showscale=False,
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cc = ddf['classification'].value_counts()
        fig = go.Figure(go.Pie(
            labels=cc.index, values=cc.values, hole=0.5,
            marker_colors=['#2ecc71', '#e74c3c']
        ))
        fig.update_layout(title="Product Split",
                          height=300, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # Scatter
    fig = px.scatter(ddf, x='eco_score', y='trust_score',
                     color='greenwashing_score',
                     color_continuous_scale='RdYlGn_r',
                     size='review_count', hover_data=['category'],
                     title="Eco Score vs Trust Score")
    fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PAGE 4 — HOW IT WORKS
# ─────────────────────────────────────────────────────────────

elif page == "ℹ️ How It Works":

    st.markdown("""
    <div class="header-box">
        <h1>ℹ️ How GreenCheck Works</h1>
        <p>Three AI models working together to detect greenwashing</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🧠 NLP Module
        **Model:** TF-IDF + Logistic Regression (BERT optional)
        - Cleans and tokenizes text
        - Detects 75+ eco keywords
        - Identifies certifications
        - Flags contradictory claims
        - Sentiment analysis
        - **Output:** Greenwashing probability 0–1
        """)

    with col2:
        st.markdown("""
        ### 🖼️ CNN Module
        **Model:** ResNet50 Transfer Learning
        - ImageNet pretrained weights
        - layer4 fine-tuned
        - Image augmentation
        - Custom classification head
        - **Output:** Eco vs Non-Eco packaging probability
        """)

    with col3:
        st.markdown("""
        ### 🔄 RNN Module
        **Model:** Bidirectional LSTM + Attention
        - Review sequences over time
        - Detects rating drift patterns
        - Astroturfing detection
        - Temporal attention weights
        - **Output:** Trust Score 0–1
        """)

    st.divider()

    st.markdown("""
    ### 📊 Greenwashing Score Interpretation

    | Score | Risk | Verdict |
    |-------|------|---------|
    | 0 – 20% | 🟢 Very Low | Genuinely Eco-Friendly |
    | 20 – 40% | 🟡 Low | Likely Eco-Friendly |
    | 40 – 60% | 🟠 Moderate | Needs More Analysis |
    | 60 – 80% | 🔴 High | Likely Greenwashing |
    | 80 – 100% | ⛔ Critical | Strong Greenwashing Indicators |
    """)


# ─────────────────────────────────────────────────────────────
# RUN INSTRUCTIONS (shown at bottom)
# ─────────────────────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.caption("""
**Run in Colab:**
```python
!pip install streamlit pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
!streamlit run streamlit_app.py &
url = ngrok.connect(8501)
print(url)
```
""")