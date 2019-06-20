### Başlamadan önce
Bu proje Deep Learning Türkiye'nin Proje Odaklı Derin Öğrenme Eğitimi'ne başvuru için yükledim. Geçmişte yaptığım projelerden biriydi. Kendi bilgisayarımda tekrardan çalıştırdım. Takıldığınız yer olursa bana ilkanerensuslu@gmail.com adresinden yazabilirsiniz.  Github'a yüklediğim ilk poje olduğu için eksikler olabilir. IDE olarak PyCharm kullandım.

# SkinImageClassifier

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

## 5-model1.py
Keras Sequential Model (4 Convolutional Layer, 1 Dense Layer). Training sonucunda sonuçları plot eder.

## 6-ImageAugmentation.py
Her klasör faklı sayıda resimler içermekte bu nedenle ilk modelimiz training de iyi sonuç verse de test de kötü sonuç veriyor. Bu nedenle orjinal resimlere rastgele rotation, zoom, flip, shift... işlemleri uygulayarak yeni resimler oluşturuyoruz.

## 7-ImageAugmentationVal.py
Aynı işlemleri validation klasörü için yapıyoruz.

## 8-Count Images.py
Klasörlerde anlık kaç resim olduğunu yazdırır.

## 9-model2.py
model1'e çok benzer fakat trainingte çok daha fazla resim işleyecek ve daha doğru sonuçlar verecek.

## 10-test.py
modelleri sonradan test ettirmek için oluşturduğum kod. "test2" adında bir klasör oluşturup, içine alt klasörlerini oluşturdum ve her bir alt klasöre 25'er tane resim koydum. 
