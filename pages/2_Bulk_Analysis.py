import streamlit as st
import pandas as pd
import numpy as np
import io
from utils import load_model_and_vectorizer, softmax
from view_utils import main_page_styles

# Configure the page
st.set_page_config(page_title="Bulk Analysis", page_icon="📂", layout="centered")

def bulk_analysis_page():
    # Apply shared styles
    main_page_styles()

    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("📂 Bulk Sentiment Analysis")
    st.subheader("Upload a CSV file to analyze multiple comments at once.")

    st.divider()

    model, vectorizer = load_model_and_vectorizer()

    if model is None or vectorizer is None:
        st.error("🔴 **Error:** Model or vectorizer files not found. Please verify they are present in the project directory.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # File uploader
    uploaded_file = st.file_uploader("📤 Choose a CSV file", type=["csv"], help="Upload a CSV file containing text feedback or comments.")

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded '{uploaded_file.name}' with {len(df)} rows and {len(df.columns)} columns.")

            # Column selector
            text_column = st.selectbox(
                "💬 Select the column containing text to analyze:",
                options=df.columns,
                index=0
            )

            # Analyze button
            if st.button("🔍 Run Bulk Prediction", use_container_width=True):
                with st.spinner("Processing sentiments... Please wait."):
                    # Clean and fill NaN values in selected column
                    text_data = df[text_column].fillna("").astype(str).tolist()
                    
                    # Vectorize and Predict
                    X = vectorizer.transform(text_data)
                    predictions = model.predict(X)
                    
                    # Capitalize classes for presentation
                    df['Predicted_Sentiment'] = [pred.capitalize() for pred in predictions]

                    # Predict probabilities / confidence
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X)
                        # Map to class indices
                        class_indices = [list(model.classes_).index(pred.lower()) for pred in predictions]
                        df['Confidence'] = [float(probabilities[i][class_indices[i]]) for i in range(len(predictions))]
                    else:
                        decision_scores = model.decision_function(X)
                        # Handle binary vs multiclass decision scores
                        if len(decision_scores.shape) > 1:
                            probs = np.apply_along_axis(softmax, 1, decision_scores)
                            class_indices = [list(model.classes_).index(pred.lower()) for pred in predictions]
                            df['Confidence'] = [float(probs[i][class_indices[i]]) for i in range(len(predictions))]
                        else:
                            probs = softmax(decision_scores)
                            df['Confidence'] = [float(probs[i]) for i in range(len(predictions))]

                st.success("✅ Bulk prediction completed!")

                # --- Visualizations & Metrics ---
                st.subheader("📊 Sentiment Distribution")
                
                counts = df['Predicted_Sentiment'].value_counts()
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    pos_count = counts.get('Positive', 0)
                    st.metric(label="😄 Positive", value=f"{pos_count} ({pos_count/len(df):.1%})")
                with col_m2:
                    neu_count = counts.get('Neutral', 0)
                    st.metric(label="😐 Neutral", value=f"{neu_count} ({neu_count/len(df):.1%})")
                with col_m3:
                    neg_count = counts.get('Negative', 0)
                    st.metric(label="😞 Negative", value=f"{neg_count} ({neg_count/len(df):.1%})")

                # Display simple bar chart
                # Format counts dataframe for Streamlit charting
                chart_df = pd.DataFrame({'Count': counts.values}, index=counts.index)
                st.bar_chart(chart_df, color="#14B8A6")

                # Show preview table
                st.subheader("🔍 Analysis Preview (First 50 Rows)")
                preview_cols = [text_column, 'Predicted_Sentiment', 'Confidence'] if 'Confidence' in df.columns else [text_column, 'Predicted_Sentiment']
                st.dataframe(df[preview_cols].head(50), use_container_width=True)

                # Download button
                # Convert back to CSV
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue().encode('utf-8')
                
                st.divider()
                st.download_button(
                    label="📥 Download Analyzed CSV Results",
                    data=csv_data,
                    file_name=f"analyzed_{uploaded_file.name}",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error parsing file: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    bulk_analysis_page()
