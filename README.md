# 📊 Sentiment Analysis App
A machine learning powered web application that classifies text sentiment (Positive, Negative, Neutral) using TF-IDF features and supervised ML models.  
The app is built with Streamlit and supports **single text prediction** as well as **bulk sentiment analysis**.

---

## 🚀 Features
- 🔍 **Single Text Sentiment Prediction**
- 📂 **Bulk Analysis** using CSV files
- ⚡ **Fast & Accurate ML Model**
- 🧹 **Automated Text Preprocessing**
- 📈 **Interactive Streamlit UI**
- 💡 Easy to run and extend

---

## 🧠 Technologies Used
- **Python**
- **Streamlit**
- **scikit-learn**
- **NumPy**
- **Pandas**
- **TF-IDF Vectorizer**

---

## 📁 Project Structure

```
Sentiment-Analysis-App/
│
├── app.py # Main Streamlit application
├── pages/
│ ├── 1_About.py
│ └── 2_Bulk_Analysis.py
│
├── utils.py # Text preprocessing utilities
├── view_utils.py # UI helper functions
├── sentiment_models.pkl # Trained ML model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # All dependencies
└── README.md# Additional notes
```
---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```
git clone https://github.com/<your-username>/Sentiment-Analysis-App.git
cd Sentiment-Analysis-App
```
2️⃣ Install required dependencies
```
pip install -r requirements.txt
```

3️⃣ Run the Streamlit app
```
streamlit run app.py
```
🎯 How It Works

User inputs text (or uploads a CSV).
The system preprocesses text: cleaning, normalization, tokenizing.
TF-IDF converts text to numerical vectors.
A trained ML model predicts Positive, Negative, or Neutral.
Streamlit displays results with a clean and interactive layout.

🤝 Contributing

Contributions are welcome!
Feel free to submit issues or pull requests.

📜 License

This project is open-source under the MIT License.


