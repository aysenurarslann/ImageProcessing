import os
import spectral.io.envi as envi
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np


# Veri dosyalarının bulunduğu klasörün yolu
folder_path = 'viscor/day_1_m3'

# Klasördeki tüm dosyaları listeleme
file_list = os.listdir(folder_path)

# Görüntüler için OTSU eşik değerlerini hesaplamak için boş bir liste oluşturduk
thresholds = []
binary_images = []
img_datas = []  # .bin dosyalarından okunan görüntü verileri listesi buraya gelecek

# .hdr dosyalarını bulma ve eşleşen .bin dosyalarını açmak için :
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

        # Tüm bantların ortalamasını aldık:
        mean_spectrum = np.mean(img_data, axis=(0, 1))

        # Histogramı çizdirmek için:
        plt.plot(mean_spectrum, label=hdr_file)

        # Hangi bantın seçileceğine karar verin ve seçin
        selected_band = np.argmax(mean_spectrum)  # Ortalama spektrumun maksimum değerine sahip olan bantı seçtik
        plt.axvline(x=selected_band, color='r', linestyle='--', label='Seçilen Bant')  # Seçilen bantı göster

        # OTSU eşikleme algoritmasını uygulamak için:
        thresh = threshold_otsu(img_data[:, :, selected_band])
        thresholds.append(thresh)

        # Binary görüntü oluşturmak için:
        binary_image = img_data[:, :, selected_band] > thresh
        binary_images.append(binary_image)  # Binary görüntüyü listeye ekleme

# Histogramı göster
plt.xlabel('Bantlar')
plt.ylabel('Ortalama Değerler')
plt.title('Bantların Ortalama Değerlerinin Histogramı')
plt.legend()
plt.show()




# ROI Bölgesini Belirleme ve Kesmek için:
for i, binary_image in enumerate(binary_images):
    # Beyaz piksellerin konumlarını bulalılm:
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
    roi_image = img_datas[i][min_x:max_x, min_y:max_y, selected_band]

    # ROI görüntüsünü görselleştirmek için:
    plt.imshow(roi_image, cmap='gray')
    plt.title('ROI Görüntüsü')
    plt.show()