# SkinImageClassifier
Bu proje Deep Learning Türkiye'nin Proje Odaklı Derin Öğrenme Eğitimi'ne başvuru için yükledim. Github'a yüklediğim ilk poje. IDE olarak PyCharm kullandım.

## 1-Gerekli Kütüphaneler
* tensorFlow
* keras
* sklearn

```
pip install tensorFlow
pip install keras
pip install sklearn
```
 
## 2-HAM10000 Dataset
Yaklaşık 10000 farklı boyutlarda dermoscopic resim içeren bir dataset. [HAM10000](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) .

## Kurulum
1. Yeni bir proje oluşturun.
2. Proje dosyalarını indirip yeni projenin içine atın. 
3. Dataseti kodların bulunduğu proje dosyasına indirip  sıkıştırılmış dosyaları açın. Tüm sıkıştırılmışları açtıktan sıkıştırılmış klasörleri silebilirsiniz.  

## 3-PreProcessor.py
Gerekli klasör düzenini yaratmak için. Dosyayı birkere çalıştırmanız yeterli.

```
base
└── train
    └── akiec
    └── bcc
    └── bkl
    └── df
    └── mel
    └── nv
    └── vasc
└── validation
    └── akiec
    └── bcc
    └── bkl
    └── df
    └── mel
    └── nv
    └── vasc
        
```
## 4-PreProcessor2.py
1. Resimleri %90 train %10 validation olmak üzere 2 parçaya ayırır.
2. Sonra tüm resimleri HAMdataset.csv dosyasındaki kategorisine göre ilgili klasöre taşır.

## 5-PreProcessor.py
