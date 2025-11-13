# Fake Job Posting Detection

This project is a **Streamlit web application** designed to detect whether a job posting is **real or fake** using a **Logistic Regression** model trained on textual job data.  
It allows users to test job descriptions, provide feedback, and even retrain the model using the collected feedback for continuous improvement.

---

## 🚀 Features

- Detects **real** or **fake** job postings using NLP and machine learning  
- Interactive **Streamlit web interface**  
- Accepts **user feedback** on predictions (Correct / Incorrect)  
- Allows **model retraining** using feedback data combined with the original dataset  
- Displays **performance metrics** (Accuracy, Precision, Recall, F1 Score)  
- Supports **real-time prediction** of custom job descriptions  

---

## 🧠 How It Works

1. **Data Loading:** Reads the dataset (`Fakejob_dataset.xlsx`)  
2. **Text Preprocessing:** Cleans and tokenizes text (removes HTML, punctuation, stopwords, etc.)  
3. **Feature Extraction:** Uses TF-IDF vectorization to transform text into numerical features  
4. **Model Training:** Trains a Logistic Regression classifier  
5. **Prediction:** Predicts if a given job posting is real (0) or fake (1)  
6. **Feedback System:** Users can mark predictions as “Correct” or “Incorrect”  
7. **Retraining:** Combines feedback data with the original dataset for retraining  
8. **Cache Control:** Allows clearing cached data and restarting the app  

---

## 💡 Example Test Sentences

Real (Correct) Job Postings:

"Looking for a data analyst to join our finance team in New York."
"Hiring a software engineer with experience in Python and Django."

Fake (Incorrect) Job Postings:

"Work from home and earn $5000 weekly with no experience required."
"Urgent! Deposit $50 to secure your data entry job offer today."
