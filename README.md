# [Data_Project_Master -> Main AI Project About Heart Failure Prediction - Kalp Yetmezliği Risk Tahmini İçeren Yapay Zeka Projesi]

## 📌 Overview - Genel Bakış
- This repository features an AI project which includes graphical depictions of heart failure in terms of genders (male/female) with other factors and prediction of heart failure according to criterias such as gender, age, living standards and more.
- Bu kod tabanı, kalp yetmezliği riskinin cinsiyet ile diğer faktörlerle ilişkisini grafiklerle gösteren ve kalp yetmezliği riskini cinsiyet, yaş, yaşama koşulları ve daha fazla kritere göre tahmin eden bir yapay zeka projesidir. 

## 🚀 Key Features - Anahtar Özellikler
- **Custom Model Training - Özel Model Eğitimi:** Custom model is trained for the dataset (heart.csv) and in this project the best performed model weights dct.pkl (Decision Tree) and knn.pkl (k-Nearest Neighbor) are used.
- Veri setine (heart.csv) özel model eğitimi gerçekleştirilmiştir ve bu projede en yüksek performanslı dct.pkl (Karar Ağaçları) ile knn (En Yakın Komşuluk) knn.pkl model ağırlıkları kullanılmıştır. 
- **Data Science - Veri Bilimi:** Veri biliminin yarmıdıyla (pandas, matplotlib, numpy) veri setinin karakteristikleri kalp yetmezliği riskine kıyasla yansıtılabilmiştir. With help of data science characteristics of dataset are reflected with heart failure stats. 

## 🛠️ Tech Stack & Architecture - Teknik Özellikler ve Mimari
- **Code Area - Kod Alanı:** Python
- **Main Libraries - Temel Kütüphaneler:** Pandas, NumPy, Matpolib, Scikit-learn  
- **DevOps & Infrastructure - Geliştirme Çerçevesi ve Altyapı:** Streamlit, Git, Github

## 💻 Installation & Setup - Yükleme ve Kurulum

Follow these steps to run the project locally - Lokal kurulum için aşağıdaki adımları takip edebilirsiniz:

```bash
# Clone the repository
git clone [https://github.com/cerenimoon/data_project_master.git](https://github.com/cerenimoon/data_project_master.git)
cd your-repo-name

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
