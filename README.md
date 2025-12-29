# Fake Job Posting Detection

A **Streamlit app** that detects whether a job posting is **real or fake** using a **Random Forest classifier**.  

You can also give **feedback** to improve the model over time.  

---

## Features

- Test job postings for **real (0)** or **fake (1)**.  
- View a **preview of the dataset**.  
- See **model performance metrics** (Accuracy, Precision, Recall, F1).  
- Provide **feedback** (Correct / Incorrect).  
- **Retrain the model** using feedback.  
- Uses **Streamlit caching** for faster loading and training.  

---

## Tools & Technologies

- **Python 3.x**  
- **Streamlit**  
- **pandas**  
- **scikit-learn** (Random Forest, TF-IDF)  
- **NLTK** (tokenization, stopwords, lemmatization)  
- **joblib** (save/load model and vectorizer)  

---

## Dataset

Contains job posting information like:  

- Title, Location, Department  
- Company profile, Description, Requirements  
- Employment type, Experience, Education  
- Industry, Function  
- Label: `fraudulent` (0 = real, 1 = fake)  

---

## How It Works

1. Job postings are **preprocessed** (cleaning, tokenization, stopwords removal, lemmatization).  
2. Features are extracted using **TF-IDF**.  
3. **Random Forest classifier** predicts whether the job is fake or real.  
4. Users can provide feedback which is saved to `feedback.csv`.  
5. Model can be **retrained with new feedback**.  
6. Uses **cache** to avoid retraining or reloading unnecessarily.  

---

## Screenshots

<img width="1344" height="598" alt="Screenshot 2025-12-29 164011" src="https://github.com/user-attachments/assets/2a39dc89-5805-4326-b22e-c731e183d107" />
<img width="1336" height="602" alt="Screenshot 2025-12-29 164048" src="https://github.com/user-attachments/assets/af7534a8-0e45-4957-8031-647fbca6e71c" />


