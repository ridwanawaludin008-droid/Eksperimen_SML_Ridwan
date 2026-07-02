# Eksperimen_SML_Nama-siswa

Repository eksperimen preprocessing dataset Iris untuk submission Sistem ML.

## Struktur Folder

```
Eksperimen_SML_Nama-siswa/
├── .github/
│   └── workflows/
│       └── preprocessing.yml
├── iris_raw/
│   └── iris_raw.csv
├── preprocessing/
│   ├── Eksperimen_Nama-siswa.ipynb
│   ├── automate_Nama-siswa.py
│   └── iris_preprocessing/
│       ├── iris_train.csv
│       └── iris_test.csv
└── README.md
```

## Cara Menjalankan

```bash
cd preprocessing
pip install pandas numpy scikit-learn
python automate_Nama-siswa.py
```

## Dataset

- **Sumber**: Iris Dataset (sklearn / UCI ML Repository)
- **Fitur**: sepal length, sepal width, petal length, petal width
- **Target**: species (setosa, versicolor, virginica)
- **Jumlah sampel**: 150

## Preprocessing Steps

1. Load dataset dari sklearn
2. Hapus duplikat
3. Hapus missing values
4. Label encoding pada kolom target
5. Standarisasi fitur dengan StandardScaler
6. Train-test split (80:20)
