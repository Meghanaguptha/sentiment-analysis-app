import streamlit as st
from view_utils import main_page_styles

# Configure the page
st.set_page_config(page_title="About the App", page_icon="ℹ️", layout="centered")

def about_page():
    # Apply shared styles
    main_page_styles()

    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("ℹ️ About the Sentiment Analyzer")
    st.subheader("Understand the science and technology behind our sentiment predictions.")

    st.divider()

    # Introduction
    st.markdown("""
    ### 🌟 Overview
    This web application uses **supervised Machine Learning** to automatically categorize the emotional tone of text. 
    It is specifically optimized for analyzing short-form social media comments, feedback, and customer reviews, 
    classifying them into three distinct sentiments:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**😄 Positive**")
        st.caption("Favorable, appreciative, or excited remarks.")
    with col2:
        st.info("**😐 Neutral**")
        st.caption("Factual, objective, or query-based text.")
    with col3:
        st.warning("**😞 Negative**")
        st.caption("Disappointed, critical, or frustrated feedback.")

    st.divider()

    # ML Pipeline Section
    st.markdown("### 🧠 The Machine Learning Pipeline")
    st.markdown("Here is how your text is processed and predicted under the hood:")

    # Using streamlit columns to present a visual pipeline
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
        #### 1. Preprocessing 🧹
        - Text is converted to lowercase.
        - Punctuation & special chars are stripped.
        - Whitespace is normalized.
        """)
    with p2:
        st.markdown("""
        #### 2. Vectorization 🔢
        - **TF-IDF** (Term Frequency-Inverse Document Frequency) is applied.
        - Words are converted to mathematical vectors based on their importance in the corpus.
        """)
    with p3:
        st.markdown("""
        #### 3. Classification 🎯
        - A trained **Logistic Regression** model classifies the vectorized text.
        - The model computes confidence scores for all three classes.
        """)

    st.divider()

    # Performance Stats
    st.markdown("### 📊 Model Performance")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric(label="Validation Accuracy", value="82.4%")
    with col_stat2:
        st.metric(label="TF-IDF Features", value="15,000+")
    with col_stat3:
        st.metric(label="Training Dataset Size", value="10k+ Samples")

    st.divider()

    # Tech Stack
    st.markdown("### 🛠️ Built With")
    
    # We will lay out the tech stack in a nice grid
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("💻 **Python**\n\nLanguage")
    with t2:
        st.markdown("⚡ **Streamlit**\n\nWeb Interface")
    with t3:
        st.markdown("🤖 **Scikit-Learn**\n\nML Framework")
    with t4:
        st.markdown("📂 **Pandas & NumPy**\n\nData Processing")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    about_page()
