import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier   

st.set_page_config(
    page_title="AI Project"
)

st.title("Main AI Page - Data Exploration - DL")
st.sidebar.success("Heart Disease AI Project") 

df = pd.read_csv("heart.csv") #raw data

st.subheader('Heart Disease Public Data set :hospital:')
st.write(df)

#st.subheader('Age Histogram') #subheader
fig_age, (ax_age, ax_sex) = plt.subplots(
    nrows = 1,
    ncols = 2,
    figsize = (6, 4)
)

sns.histplot(data = df, x = 'age', ax = ax_age) #histogram plot age 
df_positive = df[df["target"] == 1] #positive disease 
sns.barplot(data = df_positive, x = 'sex', y = 'target', ax = ax_sex) #bar plot
ax_age.set_title('age histogram')
ax_sex.set_title('gender bar graphic')
fig_age.set_tight_layout(True)
st.pyplot(fig_age)
#description text
st.write("Age histogram grafiği veri setindeki yaş dağılımlarını göstermektedir ve yaş dağılımlarının orta yaş aksanında birleştiği gözlemlenmektedir ")
st.write('Gender Bar Grafiği veri setindeki kalp hastalığı riski pozitif bireylerin cinsiyet dağılımını göstermektedir ve erkek cinsiyeti daha yüksek orana sahiptir :mens:')

#pie chart in disease target data
st.subheader('Target data distribution pie graphic')
fig_pie = plt.figure(figsize=(8, 6))
with_heart_disease = df.loc[df['target'] == 1].shape[0] #disease positive
without_heart_disease = df.loc[df['target'] == 0].shape[0] #disease negative
disease = [with_heart_disease, without_heart_disease] #data
labels = ["Heart Disease positive", "Heart Disease negative"] #labels
colors = sns.color_palette('pastel')[0:5]
plt.pie(disease, labels = labels, colors = colors, autopct = "%.0f%%")
st.pyplot(fig_pie)

st.write("Pie Grafiği kalp hastağını sınıflarının veri setindeki dağılımını göstermektedir ve kalp hastalığı sınıflarının dağılımı veri setinde dengelidir")

#seaborn chart
st.subheader('Target Hue Boxplots (1 = disease; 0 = no disease)')

fig, (ax_trest, ax_max) = plt.subplots(
    nrows = 1,
    ncols = 2,
    figsize = (6, 4)
) 
#fig = plt.figure(figsize=(8, 6))
sns.boxplot(data = df, y = 'trestbps', x = 'sex', hue = 'target', color = 'red', ax=ax_trest)
sns.boxplot(data = df, y = 'thalach', x = 'sex', hue = 'target', color = 'green', ax=ax_max)
ax_trest.set_title('trestbps and sex')
ax_max.set_title('thalach and sex')
fig.set_tight_layout(True)
st.pyplot(fig)

#chol serum choleterol
fig2, (ax_serum, ax_chest) = plt.subplots(
    nrows = 1,
    ncols = 2,
    figsize = (6, 4)
)
#fig3 = plt.figure(figsize=(8, 6))
sns.boxplot(data = df, y = 'chol', x = 'sex', hue = 'target', color = 'blue', ax = ax_serum)
sns.boxplot(data = df, y = 'cp', x = 'sex', hue = 'target', color = 'pink', ax = ax_chest)
ax_serum.set_title('chol and sex')
ax_chest.set_title('cp and sex')
fig2.set_tight_layout(True)
st.pyplot(fig2)

#slope 
fig5, (ax_age_se, ax_thal) = plt.subplots(
    nrows = 1,
    ncols = 2,
    figsize = (6, 4)
) 
sns.boxplot(data = df, y = 'age', x = 'sex', hue = 'target', color = 'magenta', ax = ax_age_se)
sns.boxplot(data = df, y = 'thal', x = 'sex', hue = 'target', color = 'orange', ax = ax_thal)
ax_age_se.set_title('age and sex')
ax_thal.set_title('thal and sex')
fig5.set_tight_layout(True)
st.pyplot(fig5)

#Heart Disease deployment - uı
st.subheader('Heart Disease Dataset AI Deployment')

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "DecisionTree"))

#scale data 
scaler = MinMaxScaler()
scaler.fit(df)
scaled_data = scaler.fit_transform(df) 
scaled_heart = pd.DataFrame(scaled_data, columns = df.columns)
#parse the data
scaled_heart['target'] = scaled_heart['target'].astype(int)
scaled_heart['fbs'] = scaled_heart['fbs'].astype(int) 
scaled_heart['sex'] = scaled_heart['sex'].astype(int) 
scaled_heart['exang'] = scaled_heart['exang'].astype(int) 
#KNN algorithm hold out
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = scaled_heart.loc[:, features]
#Decision Tree 
redu_features = ["cp", "trestbps", "chol", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
redu_data = scaled_heart.loc[:, redu_features]
features_t = ["cp", "trestbps", "chol", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
X = redu_data.loc[:, features_t]
X_train, X_test, y_train, y_test = train_test_split(X, redu_data.target, random_state=42, train_size= .70) #holdout 

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        age_v = st.sidebar.slider("Age", 8, 76)
        params["age"] = age_v
        sex_v = st.sidebar.slider("Sex (1 = male; 0 = female)", 0, 1)
        params["sex"] = sex_v
        cp = st.sidebar.slider("cp (chest pain type)", 0.0, 3.0)
        params["cp"] = cp 
        trestbps = st.sidebar.slider("trestbps (resting blood pressure on admission to hospital)", 94.0, 200.0)
        params["trestbps"] = trestbps
        chol = st.sidebar.slider("chol (serum cholestoral in mg/dl)", 126.0, 564.0)
        params["chol"] = chol 
        fbs = st.sidebar.slider("fbs (fasting blood sugar gt 120 mg/dl)", 0, 1)
        params["fbs"] = fbs 
        restecg = st.sidebar.slider("restecg (resting electrocardiographic results)", 0.0, 2.0)
        params["restecg"] = restecg
        thalach = st.sidebar.slider("thalach (max heart rate achieved)", 71.0, 202.0)
        params["thalach"] = thalach
        exang = st.sidebar.slider("exang (exercise induced angina)", 0, 1)
        params["exang"] = exang
        oldpeak = st.sidebar.slider("oldpeak (ST depression exercise)", 0.0, 6.2)
        params["oldpeak"] = oldpeak
        slope = st.sidebar.slider("slope (the slope of peak exercise ST segment)", 0.0, 2.0)
        params["slope"] = slope 
        ca = st.sidebar.slider("ca (number of major vessels)", 0.0, 4.0)
        params["ca"] = ca 
        thal = st.sidebar.slider("thal (defect type 1 = normal; 2 = fixed defect; 3 = reversable defect)", 1.0, 3.0)
        params["thal"] = thal 
    elif clf_name == "DecisionTree":
        cp = st.sidebar.slider("cp (chest pain type)", 0.0, 3.0)
        params["cp"] = cp 
        trestbps = st.sidebar.slider("trestbps (resting blood pressure on admission to hospital)", 94.0, 200.0)
        params["trestbps"] = trestbps
        chol = st.sidebar.slider("chol (serum cholestoral in mg/dl)", 126.0, 564.0)
        params["chol"] = chol 
        thalach = st.sidebar.slider("thalach (the max heart rate achieved)", 71.0, 202.0) 
        params["thalach"] = thalach
        exang = st.sidebar.slider("exang (exercise induced angina)", 0, 1)
        params["exang"] = exang 
        oldpeak = st.sidebar.slider("oldpeak (ST depression exercise)", 0.0, 2.0) 
        params["oldpeak"] = oldpeak
        slope = st.sidebar.slider("slope (slope of peak exercise ST segment)", 0.0, 2.0)
        params["slope"] = slope
        ca = st.sidebar.slider("ca (number of major vessels)", 0.0, 4.0)
        params["ca"] = ca 
        thal = st.sidebar.slider("thal (the defect type 1 = normal; 2 = fixed defect; 3 = reversable defect)", 1.0, 3.0)
        params["thal"] = thal
    return params 

params = add_parameter_ui(classifier_name) #get params

def get_classifier_heart(clf_name):
    if clf_name == "KNN":
        pickle_in = open("knn.pkl", "rb")
        clf = pickle.load(pickle_in)
    elif clf_name == "DecisionTree":
        pickle_in = open("dct.pkl", "rb")
        clf = pickle.load(pickle_in)
    return clf 

clf = get_classifier_heart(classifier_name) #get classifier

#train test split KNN 
#clf2 = DecisionTreeClassifier(max_depth=11, random_state=0) #KNeighborsClassifier(n_neighbors = 3, metric = "manhattan")
#X_train, X_test, y_train, y_test = train_test_split(X, scaled_heart.target, random_state = 42, train_size = .70) #holdout
#clf2.fit(X_train, y_train)
#pickle_out = open("dct.pkl", "wb")
#pickle.dump(clf2, pickle_out)
#pickle_out.close()

#y_pred = clf2.predict(X_test)
#score = accuracy_score(y_test, y_pred)
#st.write(score)

def get_predict_heart(classifier, name):
    if name == "KNN":
        value = classifier.predict([[params["age"], params["sex"], params["cp"], params["trestbps"], params["chol"], params["fbs"], params["restecg"], params["thalach"], params["exang"], params["oldpeak"], params["slope"], params["ca"], params["thal"]]])
        return value
    elif name == "DecisionTree":
        value = classifier.predict([[params["cp"], params["trestbps"], params["chol"], params["thalach"], params["exang"], params["oldpeak"], params["slope"], params["ca"], params["thal"]]])
        return value

result = ""
if st.button("Click here to predict"):
    with st.spinner("Please wait ..."):
        result = get_predict_heart(clf, classifier_name) #get result
    #st.balloons()

if result == "":
    st.info("The output is {}".format(result))
else:
    if  result[0] == 0:
        st.success('The output is {} patient heart situation is stable'.format(result))
    elif result[0] == 1:
        st.warning('The output is {} patient heart situation is critic :hospital:'.format(result))

