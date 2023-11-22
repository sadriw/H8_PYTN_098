#!/usr/bin/env python
# coding: utf-8

# # Assigment 3 Batch 98 Sadri Wahyudi

# ## Tujuan utama Project
# - mengoptimalkan kampanye pemasaran guna menarik lebih banyak pelanggan untuk memiliki deposit
# 
# ## Tujuan analisa data
# - Prediksi hasil kampanye pemasaran untuk setiap pelanggan dan klarifikasi faktor-faktor yang memengaruhi hasil kampanye.
# - Mengidentifikasi segmen pelanggan dengan menggunakan data pelanggan yang berlangganan deposit.
# 
# ## Langkah yang akan dilakukan dalam menganalisa data ini
# - Memahami Struktur Data
# - Explor Data
# - Visualisasi Data
# - Pengolahan Data
# - Pemodelan Machine Learning
# - Kesimpulan singkat dan Rekomendasi

# #### Penjelasan Data
# 1. Age: Numerik, merepresentasikan usia individu.
# 2. Job: Kategorikal, jenis pekerjaan individu ('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown').
# 3. Marital: Kategorikal, status pernikahan individu ('divorced','married','single','unknown'; catatan: 'divorced' berarti bercerai atau duda/janda).
# 4. Education: Kategorikal, tingkat pendidikan individu ('primary', 'secondary', 'tertiary', dan 'unknown').
# 5. Default: Kategorikal, apakah individu memiliki kredit dalam keadaan gagal ('no','yes','unknown').
# 6. Housing: Kategorikal, apakah individu memiliki pinjaman perumahan ('no','yes','unknown').
# 7. Loan: Kategorikal, apakah individu memiliki pinjaman pribadi ('no','yes','unknown').
# 
# -----------------------------------------------
# #### Terkait dengan Kontak Terakhir Kampanye Saat Ini:
# 8. Contact: Kategorikal, jenis komunikasi kontak ('cellular','telephone').
# 9. Month: Kategorikal, bulan terakhir kontak dalam setahun ('jan', 'feb', 'mar', ..., 'nov', 'dec').
# 10. Day: Kategorikal, hari terakhir kontak dalam seminggu ('mon','tue','wed','thu','fri').
# 11. Duration: Numerik, durasi kontak terakhir dalam detik. Catatan penting: atribut ini sangat mempengaruhi target output (misalnya, jika duration=0, maka y='no'). Durasi tidak diketahui sebelum panggilan dilakukan dan hanya diketahui setelah panggilan berakhir. Sebaiknya hanya digunakan untuk tujuan benchmark dan dihapus jika tujuannya adalah memiliki model prediktif yang realistis.
# 
# #### Atribut Lainnya:
# 12. Campaign: Numerik, jumlah kontak yang dilakukan selama kampanye ini dan untuk klien ini (termasuk kontak terakhir).
# 13. Pdays: Numerik, jumlah hari yang telah berlalu setelah klien terakhir dihubungi dari kampanye sebelumnya (999 berarti klien tidak pernah dihubungi sebelumnya).
# 14. Previous: Numerik, jumlah kontak yang dilakukan sebelum kampanye ini dan untuk klien ini.
# 15. Poutcome: Kategorikal, hasil dari kampanye pemasaran sebelumnya ('failure','nonexistent','success').
# 
# #### Output Variabel (Desired Target):
# 16. y: Biner, apakah klien berlangganan deposito berjangka ('yes','no').

# In[65]:


# Import library untuk manipulasi dan analisis data
import pandas as pd
import numpy as np

# Import library visualisasi data standar
import matplotlib.pyplot as plt
import seaborn as sns

# Import library preprocessing untuk skala data
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Import library visualisasi data khusus (berbasis plotly)
import squarify
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

# Import library untuk pemodelan Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Import library untuk evaluasi model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import library untuk pembagian data menjadi data pelatihan dan pengujian
from sklearn.model_selection import train_test_split

# Import library untuk pemodelan regresi logistik
from sklearn.linear_model import LogisticRegression

#Import library SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV


# In[3]:


#import dataset

bank = pd.read_csv('bank-additional-full.csv',  sep=';')
bank.head()


# In[4]:


# Melihat jumlah row
print("Panjang data {rows} rows.".format(rows = len(bank)))


# In[5]:


bank.info()


# In[6]:


#Melihat missing value
missing_values = bank.isnull()

missing_values.sum()


# Karna tidak ada nata kosong mari kita explore isi dari datanya

# In[7]:


#Lihat colum dengan format categorical

cat_columns = ['month', 'marital', 'contact', 'default', 'housing', 'loan', 'education', 'job','poutcome']

fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(20, 20))

counter = 0
for cat_column in cat_columns:
    value_counts = bank[cat_column].value_counts()
    
    trace_x = counter // 3
    trace_y = counter % 3
    x_pos = np.arange(0, len(value_counts))
    
    axs[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label = value_counts.index)
    
    axs[trace_x, trace_y].set_title(cat_column)
    
    for tick in axs[trace_x, trace_y].get_xticklabels():
        tick.set_rotation(90)
    
    counter += 1

plt.show()


# In[9]:


num_columns = ['age', 'pdays', 'duration', 'campaign', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

fig, axs = plt.subplots(3, 4, sharex=False, sharey=False, figsize=(20, 15))

counter = 0
for num_column in num_columns:
    
    trace_x = counter // 4
    trace_y = counter % 4
    
    axs[trace_x, trace_y].hist(bank[num_column])
    
    axs[trace_x, trace_y].set_title(num_column)
    
    counter += 1

plt.show()


# Mari kita lihat lebih dala nilai-nilai dalam kolom-kolom 'campaign', 'pdays', dan 'previous'data -data ini memiliki distribusi yang terlihat aneh

# In[10]:


# Menampilkan distribusi nilai dalam kolom 'pdays'
print(bank['pdays'].value_counts())


# In[11]:


# Menampilkan distribusi nilai dalam kolom 'campaign'
print(bank['campaign'].value_counts())


# In[12]:


# Menampilkan distribusi nilai dalam kolom 'previous'
print(bank['previous'].value_counts())


# In[13]:


bank[['pdays','campaign','previous']].describe()


# ### Menganalisa nasabah memiliki deposit atau tidak
# 
# Ini penting dilakukan untuk melihat hubungan antara variabel lain. Sebelum itu kita lihat dulu sebaran deposit

# In[14]:


# Mengganti nama kolom 'y' menjadi 'depositt'
bank = bank.rename(columns={'y': 'deposit'})
bank.head()


# In[15]:


bank.info()


# In[16]:


# Melihat sebaran nilai
value_counts = bank['deposit'].value_counts()

# Membuat diagram batang

value_counts.plot.bar(title='Deposit Value Counts')

# Menampilkan nilai pada setiap batang
for i, value in enumerate(value_counts):
    plt.text(i, value + 50, str(value), ha='center', va='bottom')

# Menampilkan diagram
plt.show()
print(value_counts)


# Selanjutnya kita lihat hubungan antar colum dengan deposit

# In[17]:


# Membuat DataFrame untuk 'job' dan 'deposit'
j_bank = pd.DataFrame()

# Mengisi kolom 'yes' dan 'no'
j_bank['yes'] = bank[bank['deposit'] == 'yes']['job'].value_counts()
j_bank['no'] = bank[bank['deposit'] == 'no']['job'].value_counts()

# Menambahkan kolom total
j_bank['total'] = j_bank['yes'] + j_bank['no']

# Mengurutkan DataFrame berdasarkan jumlah 'yes' secara terurut
j_bank = j_bank.sort_values(by='no', ascending=False)

# Membuat bar chart
j_bank[['yes', 'no']].plot.bar(title='Job and Deposit')

# Menampilkan diagram
plt.show()
print(j_bank['total'])


# In[18]:


# Membuat DataFrame untuk 'marital' dan 'deposit'
j_bank = pd.DataFrame()

# Mengisi kolom 'yes' dan 'no'
j_bank['yes'] = bank[bank['deposit'] == 'yes']['marital'].value_counts()
j_bank['no'] = bank[bank['deposit'] == 'no']['marital'].value_counts()

# Menambahkan kolom total
j_bank['total'] = j_bank['yes'] + j_bank['no']

# Membuat bar chart
ax = j_bank[['yes', 'no']].plot(kind='bar', title='Marital and Deposit')

# Menambahkan nilai pada setiap bar
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline')

# Menampilkan diagram
plt.show()


# In[101]:


#education and deposit
j_bank = pd.DataFrame()

j_bank['yes'] = bank[bank['deposit'] == 'yes']['education'].value_counts()
j_bank['no'] = bank[bank['deposit'] == 'no']['education'].value_counts()

j_bank.plot.bar(title = 'Education and deposit')


# In[20]:


#contact and deposit
j_bank = pd.DataFrame()

j_bank['yes'] = bank[bank['deposit'] == 'yes']['contact'].value_counts()
j_bank['no'] = bank[bank['deposit'] == 'no']['contact'].value_counts()

j_bank.plot.bar(title = 'contact and deposit')


# ### Dari data di atas dapat kita lihat beberapa hal:
# - Orang yang di bidang Blue-Collar & Technician cendrung tidak memiliki deposit
# - Orang yang sudah menikah juga cendrung tidak memilikideposit
# - Serta orang pada contact cellular cendrung diak memiliki deposit
# 
# ### Selanjutnya kita lihat data yang bersifat numerik terhadap nasabah deposit
# 

# In[21]:


bank.info()


# In[22]:


#age and deposit

a_bank = pd.DataFrame()
a_bank['age_yes'] = (bank[bank['deposit'] == 'yes'][['deposit','age']].describe())['age']
a_bank['age_no'] = (bank[bank['deposit'] == 'no'][['deposit','age']].describe())['age']

a_bank


# In[23]:


# Menghapus kolom count, 25%, 50%, dan 75%
a_bank_cleaned = a_bank.drop(['count', '25%', '50%', '75%'])

# Membuat batang diagram horizontal
ax = a_bank_cleaned.plot.barh(title='Age and Deposit Statistics', legend=False)

plt.show()


# In[24]:


#age and deposit

a_bank = pd.DataFrame()
a_bank['age_yes'] = (bank[bank['deposit'] == 'yes'][['deposit','age']].describe())['age']
a_bank['age_no'] = (bank[bank['deposit'] == 'no'][['deposit','age']].describe())['age']

a_bank


# In[25]:


# Hapus the Job Occupations jika datanya "Unknown"
bank = bank.drop(bank.loc[bank["job"] == "unknown"].index)

# Admin dan manajemen pada dasarnya sama, kita letakkan di bawah nilai kategori yang sama
lst = [bank]

for col in lst:
    col.loc[col["job"] == "admin.", "job"] = "management"


# In[26]:


bank.columns


# In[27]:


# Filter data untuk yang berlangganan deposit ('yes')
subscribed_bank = bank.loc[bank["deposit"] == "yes"]

# Dapatkan daftar pekerjaan unik
occupations = bank["job"].unique().tolist()

# Dapatkan usia berdasarkan pekerjaan
management = subscribed_bank["age"].loc[subscribed_bank["job"] == "management"].values
technician = subscribed_bank["age"].loc[subscribed_bank["job"] == "technician"].values
services = subscribed_bank["age"].loc[subscribed_bank["job"] == "services"].values
retired = subscribed_bank["age"].loc[subscribed_bank["job"] == "retired"].values
blue_collar = subscribed_bank["age"].loc[subscribed_bank["job"] == "blue-collar"].values
unemployed = subscribed_bank["age"].loc[subscribed_bank["job"] == "unemployed"].values
entrepreneur = subscribed_bank["age"].loc[subscribed_bank["job"] == "entrepreneur"].values
housemaid = subscribed_bank["age"].loc[subscribed_bank["job"] == "housemaid"].values
self_employed = subscribed_bank["age"].loc[subscribed_bank["job"] == "self-employed"].values
student = subscribed_bank["age"].loc[subscribed_bank["job"] == "student"].values

ages = [management, technician, services, retired, blue_collar, unemployed,
        entrepreneur, housemaid, self_employed, student]

# Warna
colors = ['rgba(255, 102, 102, 0.7)', 'rgba(255, 178, 102, 0.7)',
          'rgba(255, 255, 102, 0.7)', 'rgba(178, 255, 102, 0.7)',
          'rgba(102, 255, 178, 0.7)', 'rgba(102, 229, 255, 0.7)',
          'rgba(102, 153, 255, 0.7)', 'rgba(178, 102, 255, 0.7)',
          'rgba(255, 102, 255, 0.7)', 'rgba(255, 102, 178, 0.7)']

traces = []

for xd, yd, cls in zip(occupations, ages, colors):
    traces.append(go.Box(
        y=yd,
        name=xd,
        boxpoints='all',
        jitter=0.5,
        whiskerwidth=0.2,
        fillcolor=cls,
        marker=dict(
            size=2,
        ),
        line=dict(width=1),
    ))

layout = go.Layout(
    title='Distribusi Usia Berdasarkan Pekerjaan',
    yaxis=dict(
        autorange=True,
        showgrid=True,
        zeroline=True,
        dtick=5,
        gridcolor='rgb(255, 255, 255)',
        gridwidth=1,
        zerolinecolor='rgb(255, 255, 255)',
        zerolinewidth=2,
    ),
    margin=dict(
        l=40,
        r=30,
        b=80,
        t=100,
    ),
    paper_bgcolor='rgb(224,255,246)',
    plot_bgcolor='rgb(224,255,246)'
)

fig = go.Figure(data=traces, layout=layout)

# Tampilkan plot
pyo.iplot(fig)


# diagram boxplot ini yang menunjukkan distribusi usia (age) berdasarkan pekerjaan (job) pada dataset. Diagram boxplot memberikan informasi tentang sebaran dan kecenderungan data, serta dapat membantu mengidentifikasi adanya pencilan (outliers) dalam distribusi usia di setiap pekerjaan.

# In[30]:


sns.set(rc={'figure.figsize':(10,6)})
sns.set_style('whitegrid')
avg_duration = bank['duration'].mean()

lst = [bank]
bank["duration_status"] = np.nan

for col in lst:
    col.loc[col["duration"] < avg_duration, "duration_status"] = "below_average"
    col.loc[col["duration"] > avg_duration, "duration_status"] = "above_average"
    
pct_term = pd.crosstab(bank['duration_status'], bank['deposit']).apply(lambda r: round(r/r.sum(), 2) * 100, axis=1)

# Ganti warna menggunakan palet 'Set2'
ax = pct_term.plot(kind='bar', stacked=False, cmap='Set2')
plt.title("Dampak Durasi (kontak terakhir) \n dalam Membuka Deposito Berjangka \n", fontsize=10)
plt.xlabel("Status Durasi", fontsize=10)
plt.ylabel("Persentase (%)", fontsize=10)

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.02, p.get_height() * 1.02))

plt.show()


# - Durasi di Bawah Rata-rata: Dapat dilihat bahwa lebih banyak pelanggan yang berlangganan deposito ketika durasi kontak terakhirnya di bawah rata-rata. Persentase pelanggan yang tidak berlangganan deposito lebih tinggi pada kelompok ini (96%) dibandingkan dengan yang berlangganan (4%).
# - Durasi di Atas Rata-rata: Sebaliknya, pada kelompok durasi di atas rata-rata, terlihat bahwa persentase pelanggan yang berlangganan deposito lebih tinggi (25%) dibandingkan dengan yang tidak berlangganan (75%).
# 
# - Pelanggan dengan durasi kontak terakhir di atas rata-rata lebih cenderung untuk berlangganan deposito dibandingkan dengan pelanggan yang durasi kontaknya di bawah rata-rata.
# 
# - Faktor durasi kontak terakhir bisa menjadi indikator penting dalam meramalkan keputusan pelanggan untuk berlangganan deposito.

# In[38]:


# Encoding Variabel Kategorikal
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder()

categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

for col in categorical_columns:
    bank[col] = label_encoder.fit_transform(bank[col])


# In[39]:


# Feature Scaling
numeric_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
scaler = StandardScaler()

bank[numeric_columns] = scaler.fit_transform(bank[numeric_columns])


# In[40]:


# skala nilai numerik dan kategorikal
# Kemudian mari gunakan matriks korelasi
# Dengan itu kita dapat menentukan apakah durasi berpengaruh pada deposito berjangka

fig = plt.figure(figsize=(12, 8))

# dataframe
bank['deposit'] = LabelEncoder().fit_transform(bank['deposit'])

# Pisahkan kedua data frame
numeric_bank = bank.select_dtypes(exclude="object")
# categorical_bank = bank.select_dtypes(include="object")

korelasi_numeric = numeric_bank.corr()

#Color
sns.heatmap(korelasi_numeric, cbar=True, cmap="viridis")
plt.title("Matriks Korelasi \n", fontsize=12)
plt.show()


# In[42]:


# DataFrame
bank['deposit'] = LabelEncoder().fit_transform(bank['deposit'])

# Pisahkan kedua data frame
numeric_bank = bank.select_dtypes(exclude="object")

# Matriks Korelasi
korelasi_numeric = numeric_bank.corr()

# Print semua skor korelasi
print("Skor Korelasi untuk Variabel Numerik:")
print(korelasi_numeric)


# - Terlihat bahwa umur (age) memiliki korelasi positif yang lemah dengan deposit, sementara pekerjaan (job) dan status perkawinan (marital) menunjukkan korelasi positif yang sedang. 
# - Variabel durasi panggilan (duration) memiliki korelasi positif yang cukup kuat dengan deposit, yang dapat diartikan bahwa durasi panggilan yang lebih lama cenderung berkorelasi positif dengan keputusan deposit yang positif. 
# - Selain itu, beberapa variabel seperti hari sejak kontak sebelumnya (pdays) dan jumlah kontak sebelumnya (previous) menunjukkan korelasi negatif yang signifikan dengan deposit, menunjukkan bahwa semakin lama waktu sejak kontak sebelumnya atau semakin sedikit jumlah kontak sebelumnya, semakin tinggi kemungkinan keputusan deposit yang positif.

# Variabel yang dipilih : 
# 
# 1. Umur (age):
# - Memiliki korelasi positif yang lemah dengan deposit, dapat menjadi fitur yang relevan untuk memahami bagaimana umur nasabah berpengaruh terhadap keputusan deposit.
# 2. Durasi Panggilan (duration):
# - Memiliki korelasi positif yang cukup kuat dengan deposit, sehingga dapat menjadi indikator penting dalam memprediksi keputusan deposit.
# 3. Pekerjaan (job):
# - Memiliki korelasi positif yang sedang dengan deposit, menunjukkan bahwa jenis pekerjaan nasabah dapat mempengaruhi keputusan deposit.
# 4. Status Pernikahan (marital):
# - Memiliki korelasi positif yang sedang dengan deposit, memberikan informasi tentang bagaimana status pernikahan dapat berdampak pada keputusan deposit.
# 5. Pendidikan (education):
# - Memiliki korelasi positif yang lemah dengan deposit, dapat memberikan wawasan tentang sejauh mana tingkat pendidikan berpengaruh terhadap keputusan deposit.
# 6. Hari Sejak Kontak Sebelumnya (pdays):
# - Memiliki korelasi negatif yang kuat dengan deposit, menunjukkan bahwa semakin lama waktu sejak kontak sebelumnya, semakin rendah kemungkinan keputusan deposit yang positif.
# 7. Jumlah Kontak Sebelumnya (previous):
# - Memiliki korelasi positif yang sedang dengan deposit, mengindikasikan bahwa jumlah kontak sebelumnya dapat menjadi faktor yang relevan dalam memprediksi keputusan deposit.
# 
# 
# ###### Pemilihan variabel-variabel ini didasarkan pada korelasi mereka dengan variabel target (deposit) dan pertimbangan bisnis yang mendasari. Namun, perlu dicatat bahwa pemilihan variabel juga dapat melibatkan analisis lebih lanjut, seperti uji signifikansi statistik dan evaluasi fitur secara holistik.

# # Logistic Regression

# In[43]:


# Pisahkan variabel independen (fitur) dan variabel dependen (target)
X = bank[['age', 'duration', 'job', 'marital', 'education', 'pdays', 'previous']]
y = bank['deposit']


# In[44]:


# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[46]:


# Inisialisasi model Logistic Regression
model = LogisticRegression()
model


# In[47]:


# Latih model pada set pelatihan
model.fit(X_train, y_train)


# In[48]:


# Lakukan prediksi pada set pengujian
y_pred = model.predict(X_test)


# In[49]:


# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Tampilkan hasil evaluasi
print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)


# In[50]:


# Fungsi untuk menampilkan confusion matrix dalam bentuk heatmap
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Deposit', 'Deposit'], 
                yticklabels=['No Deposit', 'Deposit'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Panggil fungsi plot_confusion_matrix dengan menggunakan confusion matrix yang sudah dihitung sebelumnya
plot_confusion_matrix(conf_matrix)


# Berdasarkan analisis ini, meskipun akurasi cukup tinggi, performa model terutama terpengaruh oleh masalah ketidakseimbangan kelas. Oleh karena itu, untuk menguji model lain atau melakukan penyesuaian terhadap model ini. 
# Model lain seperti Decision Tree, Random Forest, atau Gradient Boosting dapat diuji untuk melihat apakah mereka dapat memberikan hasil yang lebih baik dalam mengatasi masalah ketidakseimbangan kelas. Selain itu, teknik penanganan ketidakseimbangan kelas seperti oversampling atau undersampling dapat dieksplorasi untuk meningkatkan kinerja model terutama pada kelas minoritas.

# # Decision Tree Classifier

# In[52]:


from sklearn.tree import DecisionTreeClassifier

# Inisialisasi model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)


# In[55]:


# Latih model pada set pelatihan
dt_model.fit(X_train, y_train)


# In[56]:


# Lakukan prediksi pada set pengujian
y_pred_dt = dt_model.predict(X_test)


# In[57]:


# Evaluasi performa model Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
class_report_dt = classification_report(y_test, y_pred_dt)


# In[58]:


# hasil evaluasi
print(f'Decision Tree Accuracy: {accuracy_dt:.4f}')
print('Decision Tree Confusion Matrix:\n', conf_matrix_dt)
print('Decision Tree Classification Report:\n', class_report_dt)

# chart dari confusion matrix
plot_confusion_matrix(conf_matrix_dt)


# Model Decision Tree memiliki akurasi yang cukup baik, namun performanya lebih rendah dibandingkan dengan Logistic Regression pada kelas minoritas (kelas 1, Deposit). Meskipun Decision Tree dapat dengan baik mengidentifikasi kelas mayoritas (No Deposit), recall dan f1-score pada kelas minoritas lebih rendah. Oleh karena itu, pemilihan model tergantung pada kepentingan relatif antara kelas mayoritas dan minoritas. Jika deteksi Deposit (kelas minoritas) penting, mungkin diperlukan pendekatan lebih lanjut untuk menangani ketidakseimbangan kelas, seperti oversampling atau menggunakan model lain yang lebih dapat menangani masalah ini.

# # Penanganan ketidakseimbangan kelas dengan SMOTE dan penyetelan hyperparameter

# In[64]:


# 1. Penanganan Ketidakseimbangan Kelas dengan SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# In[62]:


# 2. Penyetelan Hyperparameter dengan Grid Search pada Decision Tree
param_grid = {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_resampled, y_resampled)


# In[63]:


# 3. Evaluasi model dengan hyperparameter terbaik
best_dt_model = grid_search.best_estimator_
y_pred_best_dt = best_dt_model.predict(X_test)
accuracy_best_dt = accuracy_score(y_test, y_pred_best_dt)
conf_matrix_best_dt = confusion_matrix(y_test, y_pred_best_dt)
class_report_best_dt = classification_report(y_test, y_pred_best_dt)

# Tampilkan hasil evaluasi model terbaik
print(f'Best Decision Tree Accuracy: {accuracy_best_dt:.4f}')
print('Best Decision Tree Confusion Matrix:\n', conf_matrix_best_dt)
print('Best Decision Tree Classification Report:\n', class_report_best_dt)


# Setelah penanganan ketidakseimbangan kelas dengan SMOTE dan penyetelan hyperparameter, model Decision Tree menunjukkan peningkatan performa dengan akurasi yang lebih baik. Meskipun recall untuk kelas minoritas (Deposit) meningkat, masih ada ruang untuk perbaikan, terutama dalam meningkatkan presisi dan f1-score. Pemilihan model ini dapat dianggap sebagai sebuah trade-off antara akurasi dan performa kelas minoritas, dan dapat menjadi langkah awal untuk eksplorasi model dan teknik lainnya. Apabila meningkatkan performa kelas minoritas sangat penting, model atau teknik lain mungkin perlu dieksplorasi lebih lanjut.

# # Berdasarkan hasil analisis dan pengujian model, dapat disimpulkan beberapa hal dari proyek ini:
# 
# 1. Akurasi Model:
# - Model Decision Tree yang telah diatur dengan SMOTE dan penyetelan hyperparameter mencapai akurasi sekitar 83.15%.
# 
# 2. Confusion Matrix dan Classification Report:
# - Meskipun akurasi telah meningkat, terdapat beberapa keterbatasan. Model memiliki kinerja baik dalam memprediksi kelas mayoritas (No Deposit), tetapi masih mengalami kesulitan dalam memprediksi kelas minoritas (Deposit).
# - Recall untuk kelas Deposit meningkat setelah penanganan ketidakseimbangan kelas, tetapi masih cukup rendah (50%). Precision dan f1-score untuk kelas Deposit juga belum optimal.
# 
# ### Kesimpulan Utama:
# Model yang dikembangkan mampu memprediksi deposit dengan tingkat akurasi yang memadai, tetapi masih ada ruang untuk perbaikan, khususnya dalam memprediksi kelas minoritas. Penting untuk terus mengevaluasi dan meningkatkan model, serta mempertimbangkan metode dan teknik lainnya untuk menangani ketidakseimbangan kelas. 
# Proses ini dapat melibatkan eksplorasi model yang lebih kompleks, penyetelan lebih lanjut, atau penerapan teknik ensemble untuk meningkatkan kinerja prediktif terutama pada kelas minoritas.
# 
# 
# 
# 
# 
# 
# 
