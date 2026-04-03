    col1, col2 = st.columns(2)
    
    with col1:
        # Greenwashing by Category
        cat_greenwash = dashboard_data.groupby('category')['greenwashing_score'].mean().sort_values(ascending=True)
        fig = px.bar(
            x=cat_greenwash.values * 100,
            y=cat_greenwash.index,
            orientation='h',
            title="🏭 Greenwashing Rate by Category",
            labels={'x': 'Greenwashing Score (%)', 'y': 'Category'},
            color=cat_greenwash.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Classification Donut
        class_counts = dashboard_data['classification'].value_counts()
        fig = go.Figure(go.Pie(
            labels=class_counts.index,
            values=class_counts.values,
            hole=0.55,
            marker_colors=['#2ecc71', '#e74c3c']
        ))
        fig.update_layout(
            title="🍩 Product Classification Split",
            height=320, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text=f"{total}<br>Products", x=0.5, y=0.5,
                             font_size=14, showarrow=False)]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ── Charts Row 2 ──────────────────────────────────────────
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter: Eco Score vs Trust Score
        fig = px.scatter(
            dashboard_data,
            x='eco_score',
            y='trust_score',
            color='greenwashing_score',
            color_continuous_scale='RdYlGn_r',
            size='review_count',
            hover_data=['category'],
            title="🎯 Eco Score vs Trust Score",
            labels={'eco_score': 'Eco Score', 'trust_score': 'Trust Score',
                    'greenwashing_score': 'Greenwash Risk'}
        )
        fig.add_hline(y=0.5, line_dash='dash', line_color='gray', opacity=0.5)
        fig.add_vline(x=0.5, line_dash='dash', line_color='gray', opacity=0.5)
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly trend
        monthly = dashboard_data.groupby('month').agg(
            greenwashing_score=('greenwashing_score', 'mean'),
            eco_score=('eco_score', 'mean'),
            trust_score=('trust_score', 'mean')
        ).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['eco_score']*100,
                                 mode='lines+markers', name='Eco Score',
                                 line=dict(color='#2ecc71', width=2)))
        fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['greenwashing_score']*100,
                                 mode='lines+markers', name='Greenwash Risk',
                                 line=dict(color='#e74c3c', width=2)))
        fig.add_trace(go.Scatter(x=monthly['month'], y=monthly['trust_score']*100,
                                 mode='lines+markers', name='Trust Score',
                                 line=dict(color='#3498db', width=2)))
        fig.update_layout(
            title="📅 Monthly ESG Trends",
            xaxis_title="Month", yaxis_title="Score (%)",
            height=320, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ── ESG Radar Chart ────────────────────────────────────────
    st.markdown("### 🌐 ESG Performance Radar")
    
    filter_cat = st.selectbox("Filter by category:", 
                               ["All"] + list(dashboard_data['category'].unique()))
    
    if filter_cat != "All":
        subset = dashboard_data[dashboard_data['category'] == filter_cat]
    else:
        subset = dashboard_data
    
    categories_radar = ['Eco Authenticity', 'Certification Rate', 
                         'Trust Score', 'Sentiment Quality', 'Transparency']
    values_radar = [
        (1 - subset['greenwashing_score'].mean()) * 100,
        subset['has_certification'].mean() * 100,
        subset['trust_score'].mean() * 100,
        (subset['sentiment_score'].mean() + 1) / 2 * 100,
        (1 - subset['greenwashing_score'].std()) * 100
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_radar + [values_radar[0]],
        theta=categories_radar + [categories_radar[0]],
        fill='toself',
        fillcolor='rgba(46,204,113,0.2)',
        line=dict(color='#2ecc71', width=2),
        name=filter_cat
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=400, margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ── Top/Bottom Products Table ──────────────────────────────
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Most Trustworthy Products")
        top_products = dashboard_data.nsmallest(5, 'greenwashing_score')[
            ['product', 'category', 'greenwashing_score', 'eco_score']
        ].copy()
        top_products['greenwashing_score'] = (top_products['greenwashing_score']*100).round(1).astype(str) + '%'
        top_products['eco_score'] = (top_products['eco_score']*100).round(1).astype(str) + '%'
        top_products.columns = ['Product', 'Category', 'GW Risk', 'Eco Score']
        st.dataframe(top_products, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### 🚨 Highest Greenwashing Risk")
        bottom_products = dashboard_data.nlargest(5, 'greenwashing_score')[
            ['product', 'category', 'greenwashing_score', 'trust_score']
        ].copy()
        bottom_products['greenwashing_score'] = (bottom_products['greenwashing_score']*100).round(1).astype(str) + '%'
        bottom_products['trust_score'] = (bottom_products['trust_score']*100).round(1).astype(str) + '%'
        bottom_products.columns = ['Product', 'Category', 'GW Risk', 'Trust Score']
        st.dataframe(bottom_products, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: IMAGE ANALYSIS
# ============================================================

def render_image_page(cnn_trainer):
    """Image classification page."""
    
    st.markdown("""
    <div class="main-header">
        <h1>🖼️ Packaging Image Analyzer</h1>
        <p>Upload a product image to classify its packaging as eco-friendly or non-eco</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded = st.file_uploader(
            "Upload product/packaging image",
            type=['jpg', 'jpeg', 'png', 'webp'],
            label_visibility="visible"
        )
        
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🖼️ Classify Packaging", type="primary", use_container_width=True):
                with st.spinner("Running CNN analysis..."):
                    result = cnn_trainer.predict_image(img)
                
                st.session_state['cnn_result'] = result
    
    with col2:
        if 'cnn_result' in st.session_state:
            result = st.session_state['cnn_result']
            eco_prob = result['eco_probability']
            non_eco_prob = result['non_eco_probability']
            
            # Main verdict
            if result['is_eco_friendly']:
                st.success(f"✅ **ECO-FRIENDLY PACKAGING**")
                badge_color = "#2ecc71"
            else:
                st.error(f"🚨 **NON-ECO PACKAGING DETECTED**")
                badge_color = "#e74c3c"
            
            # Probability bars
            st.markdown("**Classification Probabilities:**")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Probability',
                x=['🌿 Eco-Friendly', '🗑️ Non-Eco'],
                y=[eco_prob * 100, non_eco_prob * 100],
                marker_color=['#2ecc71', '#e74c3c'],
                text=[f"{eco_prob*100:.1f}%", f"{non_eco_prob*100:.1f}%"],
                textposition='outside'
            ))
            fig.update_layout(
                yaxis=dict(range=[0, 115], title='Probability (%)'),
                height=300,
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col_a, col_b = st.columns(2)
            col_a.metric("Model Confidence", f"{result['confidence']*100:.1f}%")
            col_b.metric("Predicted Class", result['class'].replace('_', ' ').title())
            
            st.info("""
            **CNN Model Info:**
            - Architecture: ResNet50 (Transfer Learning)  
            - Training: Waste Classification Dataset
            - Classes: Eco-Friendly vs Non-Eco Packaging
            """)
        else:
            st.markdown("""
            <div style="
                border: 2px dashed #ccc; border-radius:10px;
                padding: 3rem; text-align:center; color:#888;
            ">
                <div style="font-size:4rem">🖼️</div>
                <h3>Upload an image to analyze</h3>
                <p>The CNN model will classify the packaging as<br>
                eco-friendly or non-eco-friendly</p>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# PAGE: HOW IT WORKS
# ============================================================

def render_how_it_works():
    """Explains the system architecture."""
    
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ How GreenCheck Works</h1>
        <p>Three AI models working together to detect greenwashing</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🏗️ System Architecture
    
    GreenCheck uses an **ensemble of three specialized AI models**, each analyzing a different aspect of a product's sustainability claims.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <h3>🧠 NLP Module</h3>
            <b>Model:</b> BERT (fine-tuned)<br>
            <b>Input:</b> Review text<br>
            <b>Output:</b><br>
            • Greenwashing probability<br>
            • Sentiment score<br>
            • Eco keyword detection<br>
            • Certification detection<br>
            • Contradiction flags
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card" style="border-left-color: #9b59b6">
            <h3>🖼️ CNN Module</h3>
            <b>Model:</b> ResNet50 (Transfer Learning)<br>
            <b>Input:</b> Product image<br>
            <b>Output:</b><br>
            • Packaging classification<br>
            • Eco probability<br>
            • Non-eco probability<br>
            • Model confidence<br>
            • Grad-CAM heatmap
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="module-card" style="border-left-color: #e67e22">
            <h3>🔄 RNN Module</h3>
            <b>Model:</b> Bidirectional LSTM<br>
            <b>Input:</b> Review sequence<br>
            <b>Output:</b><br>
            • Trust consistency score<br>
            • Rating trend analysis<br>
            • Suspicious timestamps<br>
            • Attention weights<br>
            • Risk classification
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ## 🔍 What is Greenwashing?
    
    **Greenwashing** is when a company makes misleading or unsubstantiated claims about the environmental benefits of their products.
    
    ### Common Greenwashing Tactics:
    | Tactic | Example | Red Flag |
    |--------|---------|----------|
    | Vague Claims | "Natural", "Pure", "Green" | No certification |
    | Irrelevant Claims | "CFC-free" (already banned) | Misdirection |
    | Hidden Trade-offs | "Recycled packaging" but toxic product | Partial truth |
    | False Certifications | Made-up eco logos | Self-certified |
    | Exaggerated Claims | "100% sustainable" with no proof | No evidence |
    | Contradictions | Eco label + plastic wrap | Inconsistency |
    
    ### ✅ Signs of Genuine Eco-Friendliness:
    - Third-party certifications (USDA Organic, B-Corp, Green Seal)
    - Transparent supply chain information
    - Specific, measurable environmental claims
    - Consistent positive reviews over time
    - Detailed sustainability reports
    
    ---
    
    ## 📊 Greenwashing Score Interpretation
    
    | Score Range | Risk Level | Interpretation |
    |-------------|------------|----------------|
    | 0 – 20% | 🟢 Very Low | Genuinely eco-friendly |
    | 20 – 40% | 🟡 Low | Likely eco-friendly |
    | 40 – 60% | 🟠 Moderate | Uncertain, needs more analysis |
    | 60 – 80% | 🔴 High | Likely greenwashing |
    | 80 – 100% | ⛔ Critical | Strong greenwashing indicators |
    """)


# ============================================================
# MAIN APP
# ============================================================

def main():
    """Main application entry point."""
    
    # Load models
    try:
        preprocessor, nlp_trainer, cnn_trainer, rnn_trainer, ensemble = load_models()
        models_ok = True
    except Exception as e:
        st.error(f"⚠️ Model loading error: {e}. Running in demo mode.")
        preprocessor = TextPreprocessor()
        nlp_trainer = NLPTrainer()
        cnn_trainer = CNNTrainer()
        cnn_trainer.setup_model()
        rnn_trainer = RNNTrainer()
        ensemble = GreenwashEnsemble()
        models_ok = False
    
    # Render sidebar and get page
    page = render_sidebar()
    
    # Route to pages
    if page == "🔍 Analyze Review":
        render_analyze_page(preprocessor, nlp_trainer, cnn_trainer, rnn_trainer, ensemble)
    
    elif page == "🖼️ Analyze Image":
        render_image_page(cnn_trainer)
    
    elif page == "📊 Dashboard":
        render_dashboard()
    
    elif page == "ℹ️ How It Works":
        render_how_it_works()


if __name__ == "__main__":
    main()
