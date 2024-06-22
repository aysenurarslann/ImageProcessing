import os
import numpy as np
import spectral.io.envi as envi
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, Dropout
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Ana veri klasörünün yolu (tüm günlerin bulunduğu klasör)
main_folder_path = 'VIS_COR'

# Gün klasörlerini listeleme
day_folders = sorted([f for f in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, f))])

# Görüntüler ve etiketler için boş listeler oluşturma
all_img_datas = []
all_labels = []
file_names = []  # Dosya adlarını kaydetmek için boş liste

# Her gün klasörü için verileri okuma ve işleme
for day_folder in day_folders:
    folder_path = os.path.join(main_folder_path, day_folder)
    file_list = sorted(os.listdir(folder_path))

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
            if selected_band < img_data.shape[2]:
                selected_bands.append(selected_band)
            else:
                print(f"Uyarı: Seçilen bant ({selected_band}) img.shape[2] değerinden büyük.")

            # OTSU eşikleme algoritmasını uygulamak için:
            thresh = threshold_otsu(img_data[:, :, selected_band])
            thresholds.append(thresh)

            # Binary görüntü oluşturmak için:
            binary_image = img_data[:, :, selected_band] > thresh
            binary_images.append(binary_image)  # Binary görüntüyü listeye ekleme

            # Dosya adını kaydetme
            file_names.append(f"{day_folder}/{hdr_file}")

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
        if i < len(selected_bands):
            roi_image = img_datas[i][min_x:max_x, min_y:max_y, selected_bands[i]]
            roi_images.append(roi_image)

    # ROI görüntülerini ve etiketleri birleştirme
    roi_images = np.array(roi_images)
    labels = np.array([0, 1] * (len(roi_images) // 2))  # Örnek etiketler, uygun şekilde düzenlemelisiniz

    # Tüm günlerin verilerini birleştirme
    all_img_datas.extend(roi_images)
    all_labels.extend(labels)

# Tüm günlerin verilerini birleştirme
all_img_datas = np.array(all_img_datas)
all_labels = np.array(all_labels)
print("all_img_datas shape:", all_img_datas.shape)
print("all_labels shape:", all_labels.shape)

# En küçük boyutu belirleme
min_height = min([img.shape[0] for img in all_img_datas])
min_width = min([img.shape[1] for img in all_img_datas])
print("Minimum yükseklik:", min_height)
print("Minimum genişlik:", min_width)

# Tüm görüntüleri yeniden boyutlandırma
resized_img_datas = np.array([resize(img, (min_height, min_width), anti_aliasing=True) for img in all_img_datas])
print("Resized all_img_datas shape:", resized_img_datas.shape)

# Eğitim ve doğrulama setlerini ayırma (PCA öncesi)
X_train, X_val, y_train, y_val = train_test_split(resized_img_datas, all_labels, test_size=0.3, random_state=42)
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)

# Veriyi 4D hale getirme
X_train_exp = X_train.reshape(X_train.shape[0], min_height, min_width, 1)
X_val_exp = X_val.reshape(X_val.shape[0], min_height, min_width, 1)

print("Reshaped X_train shape:", X_train_exp.shape)
print("Reshaped X_val shape:", X_val_exp.shape)

# Etiketleri one-hot encoded hale getirme
y_train_one_hot = to_categorical(y_train, num_classes=2)
y_val_one_hot = to_categorical(y_val, num_classes=2)

# Veri artırımı
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train_exp)

# AlexNet modeli kurulumu
def create_alexnet(input_shape):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(96, (11, 11), strides=(4, 4), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        Dropout(0.25),
        Conv2D(256, (5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        Dropout(0.25),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
        Dropout(0.25),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_val, y_val):
    accuracy = model.evaluate(X_val, y_val, verbose=0)[1]
    y_pred_val = model.predict(X_val)
    y_pred_val_classes = np.argmax(y_pred_val, axis=1)
    precision = precision_score(np.argmax(y_val, axis=1), y_pred_val_classes, average='weighted', zero_division=0)
    recall = recall_score(np.argmax(y_val, axis=1), y_pred_val_classes, average='weighted')
    f1 = f1_score(np.argmax(y_val, axis=1), y_pred_val_classes, average='weighted')
    return accuracy, precision, recall, f1

# AlexNet ile eğitim (PCA öncesi)
model = create_alexnet(X_train_exp.shape[1:])
history = model.fit(datagen.flow(X_train_exp, y_train_one_hot, batch_size=32),
                    epochs=10,
                    validation_data=(X_val_exp, y_val_one_hot))

# Performans metrikleri (PCA öncesi)
accuracy_before, precision_before, recall_before, f1_before = evaluate_model(model, X_val_exp, y_val_one_hot)
print(f"AlexNet (PCA Öncesi) - Accuracy: {accuracy_before}, Precision: {precision_before}, Recall: {recall_before}, F1-Score: {f1_before}")

# RandomForestClassifier kullanma (PCA öncesi)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred = clf.predict(X_val.reshape(X_val.shape[0], -1))

# Performans metrikleri (PCA öncesi)
accuracy_rf_before = accuracy_score(y_val, y_pred)
precision_rf_before = precision_score(y_val, y_pred, average='weighted', zero_division=0)
recall_rf_before = recall_score(y_val, y_pred, average='weighted')
f1_rf_before = f1_score(y_val, y_pred, average='weighted')
print(f"Random Forest (PCA Öncesi) - Accuracy: {accuracy_rf_before}, Precision: {precision_rf_before}, Recall: {recall_rf_before}, F1-Score: {f1_rf_before}")

# PCA ile bant seçimi sonrası veriyi hazırlama
valid_selected_bands = []
for i, img in enumerate(X_train):
    if i < len(selected_bands) and selected_bands[i] < img.shape[2]:
        valid_selected_bands.append(selected_bands[i])

# Eğer geçerli bant yoksa hata ver
if len(valid_selected_bands) == 0:
    raise ValueError("No valid bands selected. Please check the PCA band selection process.")

# Bant seçimi sonrası veriyi hazırlama
X_train_pca = np.array([img[:, :, valid_selected_bands[i]] for i, img in enumerate(X_train) if i < len(valid_selected_bands)])
X_val_pca = np.array([img[:, :, valid_selected_bands[i]] for i, img in enumerate(X_val) if i < len(valid_selected_bands)])

# Veriyi 4D hale getirme (PCA sonrası)
X_train_pca_exp = X_train_pca.reshape(X_train_pca.shape[0], min_height, min_width, 1)
X_val_pca_exp = X_val_pca.reshape(X_val_pca.shape[0], min_height, min_width, 1)

# AlexNet ile eğitim (PCA sonrası)
model = create_alexnet(X_train_pca_exp.shape[1:])
history = model.fit(datagen.flow(X_train_pca_exp, y_train_one_hot, batch_size=32),
                    epochs=10,
                    validation_data=(X_val_pca_exp, y_val_one_hot))

# Performans metrikleri (PCA sonrası)
accuracy_after, precision_after, recall_after, f1_after = evaluate_model(model, X_val_pca_exp, y_val_one_hot)
print(f"AlexNet (PCA Sonrası) - Accuracy: {accuracy_after}, Precision: {precision_after}, Recall: {recall_after}, F1-Score: {f1_after}")

# RandomForestClassifier kullanma (PCA sonrası)
clf.fit(X_train_pca.reshape(X_train_pca.shape[0], -1), y_train)
y_pred_pca = clf.predict(X_val_pca.reshape(X_val_pca.shape[0], -1))

# Performans metrikleri (PCA sonrası)
accuracy_rf_after = accuracy_score(y_val, y_pred_pca)
precision_rf_after = precision_score(y_val, y_pred_pca, average='weighted', zero_division=0)
recall_rf_after = recall_score(y_val, y_pred_pca, average='weighted')
f1_rf_after = f1_score(y_val, y_pred_pca, average='weighted')
print(f"Random Forest (PCA Sonrası) - Accuracy: {accuracy_rf_after}, Precision: {precision_rf_after}, Recall: {recall_rf_after}, F1-Score: {f1_rf_after}")

# Sonuçları tek bir tabloya toplamak için:
results = {
    "Model": ["AlexNet (PCA Öncesi)", "Random Forest (PCA Öncesi)", "AlexNet (PCA Sonrası)", "Random Forest (PCA Sonrası)"],
    "Accuracy": [accuracy_before, accuracy_rf_before, accuracy_after, accuracy_rf_after],
    "Precision": [precision_before, precision_rf_before, precision_after, precision_rf_after],
    "Recall": [recall_before, recall_rf_before, recall_after, recall_rf_after],
    "F1-Score": [f1_before, f1_rf_before, f1_after, f1_rf_after]
}

results_df = pd.DataFrame(results)
print(results_df)
