# recommend_book_sysyem
# 📚 Book Recommendation System

A content-based and collaborative filtering book recommendation web app built with Python and Streamlit.



---



---

## 📌 Features

- 🔍 Search any book and get instant recommendations
- 📊 Popularity-based recommendations on the home page
- 🤝 Collaborative filtering using cosine similarity
- 🌐 Clean and interactive web interface

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Streamlit | Web app framework |
| Pandas & NumPy | Data processing |
| Scikit-learn | Cosine similarity / ML |
| Pickle | Model serialization |

---

## 📂 Project Structure

```
├── app7.py               # Main Streamlit app
├── reco23.py             # Recommendation logic
├── popular.pkl           # Popularity-based model
├── popular_dict.pkl      # Popular books dictionary
├── pt.pkl                # Pivot table for collaborative filtering
├── similarity.pkl        # Precomputed similarity scores
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/amitkumarjy/recommend_web_apm.git

# 2. Navigate to project folder
cd recommend_web_apm

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app7.py
```

---

## 🧠 How It Works

1. **Popularity-based filtering** — Shows top-rated books based on number of ratings and average score
2. **Collaborative filtering** — Uses cosine similarity on a user-book pivot table to find books similar to the selected one
3. **Pickle files** — Preprocessed models loaded at runtime for fast recommendations

---



---

## 👨‍💻 Author

**Amit Mutyalwar**  
Data Scientist | ML Engineer  
[LinkedIn](https://www.linkedin.com/in/amitkumar-mutyalwar-56519723b/) | [GitHub](https://github.com/amitkumarjy)

---

## ⭐ If you found this useful, give it a star!
