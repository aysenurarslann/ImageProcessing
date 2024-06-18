import os
import numpy as np
import spectral.io.envi as envi
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer

# Veri dosyalarının bulunduğu klasörün yolu
folder_path = 'viscor/day_1_m3'

# Klasördeki tüm dosyaları listeleme
file_list = os.listdir(folder_path)

# Görüntüler için OTSU eşik değerlerini hesaplamak için boş bir liste oluşturduk
thresholds = []
binary_images = []
img_datas = []  # .bin dosyalarından okunan görüntü verileri listesi buraya gelecek
selected_bands = []

# .hdr dosyalarını bulma ve eşleşen .bin dosyalarını açmak için:
for hdr_file in file_list:
    if hdr_file.endswith('.hdr'):
        # .hdr dosyasının tam yolunu alalım
        hdr_path = os.path.join(folder_path, hdr_file)

        # .bin dosyasının adı
        bin_file = hdr_file.replace('.hdr', '.bin')
        bin_path = os.path.join(folder_path, bin_file)

        # .hdr ve .bin dosyalarını açmak için:
        img_hdr = envi.open(hdr_path, image=bin_path)

        # .bin dosyasından görüntü verilerini alıp img_datas listesine ekleme
        img_data = img_hdr.load()
        img_datas.append(img_data)

        # PCA kullanarak bant seçimi
        pca = PCA(n_components=1)
        pca.fit(img_data.reshape(-1, img_data.shape[-1]))
        selected_band = np.argmax(np.abs(pca.components_))
        selected_bands.append(selected_band)

        # OTSU eşikleme algoritmasını uygulamak için:
        thresh = threshold_otsu(img_data[:, :, selected_band])
        thresholds.append(thresh)

        # Binary görüntü oluşturmak için:
        binary_image = img_data[:, :, selected_band] > thresh
        binary_images.append(binary_image)  # Binary görüntüyü listeye ekleme

# ROI Bölgesini Belirleme ve Kesmek için:
roi_images = []
for i, binary_image in enumerate(binary_images):
    # Beyaz piksellerin konumlarını bulalım:
    white_pixels = np.where(binary_image == 1)

    # ROI bölgesini belirleme
    min_x, max_x = np.min(white_pixels[0]), np.max(white_pixels[0])
    min_y, max_y = np.min(white_pixels[1]), np.max(white_pixels[1])

    # ROI bölgesini kırpma (kenar boşluğu ekleyerek)
    padding = 10  # Kenar boşluğu miktarı
    min_x = max(0, min_x - padding)
    max_x = min(img_data.shape[0], max_x + padding)
    min_y = max(0, min_y - padding)
    max_y = min(img_data.shape[1], max_y + padding)

    # ROI bölgesini kesme
    roi_image = img_datas[i][min_x:max_x, min_y:max_y, selected_bands[i]]
    roi_images.append(roi_image)

# ROI görüntülerini birleştirip eğitim ve doğrulama setleri oluşturmak için:
roi_images = np.array(roi_images)
labels = np.array([0, 1] * (len(roi_images) // 2))  # Örnek etiketler, uygun şekilde düzenlemelisiniz

X_train, X_val, y_train, y_val = train_test_split(roi_images, labels, test_size=0.3, random_state=42)

# AlexNet modeli kurulumu
def create_alexnet(input_shape):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(96, (11, 11), strides=(4, 4), activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(256, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(384, (3, 3), activation='relu'),
        Conv2D(384, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(2, activation='softmax')  # Sınıf sayısına göre düzenleyin
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_val, y_val):
    accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
    y_pred_val = model.predict(X_val)
    y_pred_val_classes = np.argmax(y_pred_val, axis=1)
    precision = precision_score(y_val, y_pred_val_classes, average='weighted')
    recall = recall_score(y_val, y_pred_val_classes, average='weighted')
    f1 = f1_score(y_val, y_pred_val_classes, average='weighted')
    return accuracy, precision, recall, f1

# AlexNet ile eğitim
X_train_exp = np.expand_dims(X_train, axis=-1)  # Giriş verilerinin şekil uyumunu sağlamak için
X_val_exp = np.expand_dims(X_val, axis=-1)
model = create_alexnet(X_train_exp.shape[1:])
model.fit(X_train_exp, y_train, epochs=10, validation_data=(X_val_exp, y_val))

# Performans metrikleri
accuracy, precision, recall, f1 = evaluate_model(model, X_val_exp, y_val)
print(f"AlexNet (Tüm Bantlar) - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

# Karar Ağacı ile eğitim
clf = DecisionTreeClassifier()
clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred = clf.predict(X_val.reshape(X_val.shape[0], -1))

# Performans metrikleri
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"Karar Ağacı (Tüm Bantlar) - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

# Bant seçimi sonrası model eğitimi ve performans değerlendirmesi için:
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(clf, prefit=True)
X_new_train = selector.transform(X_train.reshape(X_train.shape[0], -1))
X_new_val = selector.transform(X_val.reshape(X_val.shape[0], -1))

# AlexNet ile bant seçimi sonrası eğitim
X_new_train_exp = np.expand_dims(X_new_train.reshape(X_new_train.shape[0], roi_images.shape[1], roi_images.shape[2]), axis=-1)
X_new_val_exp = np.expand_dims(X_new_val.reshape(X_new_val.shape[0], roi_images.shape[1], roi_images.shape[2]), axis=-1)
model = create_alexnet(X_new_train_exp.shape[1:])
model.fit(X_new_train_exp, y_train, epochs=10, validation_data=(X_new_val_exp, y_val))

# Performans metrikleri
accuracy_new, precision_new, recall_new, f1_new = evaluate_model(model, X_new_val_exp, y_val)
print(f"AlexNet (Bant Seçimi Sonrası) - Accuracy: {accuracy_new}, Precision: {precision_new}, Recall: {recall_new}, F1-Score: {f1_new}")

# Karar Ağacı ile bant seçimi sonrası eğitim
clf.fit(X_new_train, y_train)
y_pred_new = clf.predict(X_new_val)

# Performans metrikleri bant seçimi sonrası
accuracy_new = accuracy_score(y_val, y_pred_new)
precision_new = precision_score(y_val, y_pred_new, average='weighted')
recall_new = recall_score(y_val, y_pred_new, average='weighted')
f1_new = f1_score(y_val, y_pred_new, average='weighted')
print(f"Karar Ağacı (Bant Seçimi Sonrası) - Accuracy: {accuracy_new}, Precision: {precision_new}, Recall: {recall_new}, F1-Score: {f1_new}")

# Sonuçları tek bir tabloya toplamak için:
import pandas as pd

results = {
    "Model": ["AlexNet (Tüm Bantlar)", "Karar Ağacı (Tüm Bantlar)", "AlexNet (Bant Seçimi Sonrası)", "Karar Ağacı (Bant Seçimi Sonrası)"],
    "Accuracy": [accuracy, accuracy_new, accuracy_new, accuracy_new],
    "Precision": [precision, precision_new, precision_new, precision_new],
    "Recall": [recall, recall_new, recall_new, recall_new],
    "F1-Score": [f1, f1_new, f1_new, f1_new]
}

results_df = pd.DataFrame(results)
print(results_df)

# Manuel olarak ROI görüntülerini görmek için fonksiyon
def show_roi_images():
    for i, roi_image in enumerate(roi_images):
        plt.imshow(roi_image, cmap='gray')
        plt.title(f'ROI Görüntüsü {i+1}')
        plt.show()
        input("Görüntüyü kapatmak için Enter tuşuna basın...")

# Kullanıcıdan giriş almak ve ROI görüntülerini göstermek için:
while True:
    command = input("ROI görüntülerini görmek için 'show' yazın, çıkmak için 'exit' yazın: ")
    if command.lower() == 'show':
        show_roi_images()
    elif command.lower() == 'exit':
        break
    else:
        print("Geçersiz komut, tekrar deneyin.")
