import os
import numpy as np
import spectral.io.envi as envi
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

            # Bantların ortalamasını kullanarak bant seçimi
            avg_band = np.mean(img_data, axis=2)
            selected_bands.append(avg_band)

            # OTSU eşikleme algoritmasını uygulamak için:
            thresh = threshold_otsu(avg_band)
            thresholds.append(thresh)

            # Binary görüntü oluşturmak için:
            binary_image = avg_band > thresh
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
        roi_image = img_datas[i][min_x:max_x, min_y:max_y]
        roi_images.append(roi_image)

    # ROI görüntülerini ve etiketleri birleştirme
    roi_images = np.array(roi_images)
    labels = np.array([0 if 'back' in name else 1 for name in file_names])  # back=0, front=1

    # Tüm günlerin verilerini birleştirme
    all_img_datas.extend(roi_images)
    all_labels.extend(labels)

# Tüm günlerin verilerini birleştirme
all_img_datas = np.array(all_img_datas)
all_labels = np.array(all_labels)
print("all_img_datas shape:", all_img_datas.shape)
print("all_labels shape:", all_labels.shape)

# Etiketlerin şekli tüm görüntülerle eşleşmelidir
all_labels = all_labels[:len(all_img_datas)]

# Veri madenciliği adımları
# Her bir görüntü verisini np.float32'ye dönüştür
all_img_datas = np.array([img.astype(np.float32) for img in all_img_datas])
print("Eksik veri sayısı:", np.sum([np.isnan(img).sum() for img in all_img_datas]))

# Eksik verileri doldurma (örneğin, medyan ile)
all_img_datas = np.array([np.nan_to_num(img, nan=np.nanmedian(img)) for img in all_img_datas])

# Standardizasyon
scaler = StandardScaler()
all_img_datas = np.array([scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) for img in all_img_datas])

# Normalizasyon
scaler = MinMaxScaler()
all_img_datas = np.array([scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) for img in all_img_datas])

# En küçük boyutu belirleme
min_height = min([img.shape[0] for img in all_img_datas])
min_width = min([img.shape[1] for img in all_img_datas])
print("Minimum yükseklik:", min_height)
print("Minimum genişlik:", min_width)

# Tüm görüntüleri yeniden boyutlandırma
resized_img_datas = np.array([resize(img, (min_height, min_width), anti_aliasing=True) for img in all_img_datas])
print("Resized all_img_datas shape:", resized_img_datas.shape)

# Eğitim ve doğrulama setlerini ayırma
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

# Bant seçimi sonrası veriyi hazırlama
X_train_pca = np.array([img[:, :, np.newaxis] for img in X_train])
X_val_pca = np.array([img[:, :, np.newaxis] for img in X_val])

# Veriyi 4D hale getirme (bant seçimi sonrası)
X_train_pca_exp = X_train_pca.reshape(X_train_pca.shape[0], min_height, min_width, 1)
X_val_pca_exp = X_val_pca.reshape(X_val_pca.shape[0], min_height, min_width, 1)

# AlexNet ile eğitim (bant seçimi sonrası)
model = create_alexnet(X_train_pca_exp.shape[1:])
history = model.fit(datagen.flow(X_train_pca_exp, y_train_one_hot, batch_size=32),
                    epochs=10,
                    validation_data=(X_val_pca_exp, y_val_one_hot))

# Performans metrikleri (bant seçimi sonrası)
accuracy_after, precision_after, recall_after, f1_after = evaluate_model(model, X_val_pca_exp, y_val_one_hot)
print(f"AlexNet (Bant Seçimi Sonrası) - Accuracy: {accuracy_after}, Precision: {precision_after}, Recall: {recall_after}, F1-Score: {f1_after}")

# RandomForestClassifier kullanma (bant seçimi sonrası)
clf.fit(X_train_pca.reshape(X_train_pca.shape[0], -1), y_train)
y_pred_pca = clf.predict(X_val_pca.reshape(X_val_pca.shape[0], -1))

# Performans metrikleri (bant seçimi sonrası)
accuracy_rf_after = accuracy_score(y_val, y_pred_pca)
precision_rf_after = precision_score(y_val, y_pred_pca, average='weighted', zero_division=0)
recall_rf_after = recall_score(y_val, y_pred_pca, average='weighted')
f1_rf_after = f1_score(y_val, y_pred_pca, average='weighted')
print(f"Random Forest (Bant Seçimi Sonrası) - Accuracy: {accuracy_rf_after}, Precision: {precision_rf_after}, Recall: {recall_rf_after}, F1-Score: {f1_rf_after}")

# Sonuçları tek bir tabloya toplamak için:
results = {
    "Model": ["AlexNet (PCA Öncesi)", "Random Forest (PCA Öncesi)", "AlexNet (Bant Seçimi Sonrası)", "Random Forest (Bant Seçimi Sonrası)"],
    "Accuracy": [accuracy_before, accuracy_rf_before, accuracy_after, accuracy_rf_after],
    "Precision": [precision_before, precision_rf_before, precision_after, precision_rf_after],
    "Recall": [recall_before, recall_rf_before, recall_after, recall_rf_after],
    "F1-Score": [f1_before, f1_rf_before, f1_after, f1_rf_after]
}

results_df = pd.DataFrame(results)
print(results_df)
