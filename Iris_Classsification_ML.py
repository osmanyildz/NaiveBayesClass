# Iris çiçeği ile ilgili supervised(denetimli) öğrenme örneği yapılacak. Tür tespit edileceği için
# classification işlemi yapılacak. KNN(K-En yakın komşu) algoritması ile model oluşturulacak ve eğitilip test edilecek.
# Aşamalar:
# 1.Veriyi tanıma, 2.Veriyi parçalama (eğitim-test), 3.Veriyi görselleştirme, 4.Model oluşturma,
# 5.Oluşturulan model ile yeni bir verinin türünü tahmin etme, 6.Model değerlendirme

#Iris data set scikit-learn kütüphanesinin içinde var bunu bu şekilde kullanabiliriz indirmeden. sklearn'ü kullanabilmek
# için scikit-learn kütüphanesini indirdik ve sklearn içindekileri de kullanabiliyoruz.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = load_iris()

# 1.Veri setini tanıma
#print(iris.keys()) # dataseti sözlük yapısında tutuluyor.(key-value)
#print(iris['DESCR'])
#print(iris["target_names"])
#print(iris['data'])
#print(iris['target'])
#print(iris['feature_names'])
#print(iris["data"].shape) #(150,4) -> 150 satır ve 4 sütun iki boyutlu bir matris. 150 kayıt 4 features. Sklearn'de birinci
# değer örneklem sayısını ikinci değer ise öznitelik sayısını gösterir
#print(iris["data"][:5])
#print(iris["target"][:5])
#print(iris["target_names"])

# 2.Veri setini parçalama Training-Test (ML modelini eğitmek için, ML modelini test etmek için)
# sklearn'de veri(data) X harfi ile, etiket(target) ise y harfi ile gösterilir(F(X) = y fnk yazım tipinden geliyor).
# sklearn.model_selection paketi içinde train_test_split adındaki fonksiyon tüm kayıtları karıştırıyor ve %75 eğitim %25
# test olarak iki ayrı dizi matris oluşturuyor

X_egitim, X_test, y_egitim, y_test = train_test_split(iris['data'], iris['target'], random_state=1)

#print(X_egitim.shape) #iki boyutlu dizi
#print(y_egitim.shape) #tek boyutlu dizi
#print(X_test.shape) #iki boyutlu dizi
#print(y_test.shape) #tek boyutlu dizi

# 3.Veriyi görselleştirme: Görselleştirme için saçılım grafiğini kullanalım. Öncelikle Numpy dizisini Pandas data frame
# yapısına çevirmek gerekir.

iris_df = pd.DataFrame(X_egitim, columns=iris.feature_names)
#pandas'ın scatter matris adında ikili grafikleri gösteren bir fonksiyonu var

#scatter_matrix(iris_df, c=y_egitim, figsize=(15,15),marker='o', hist_kwds={'bins':20},s=80,alpha=0.8)
#plt.show() # bir scatter plot veya bir histogram oluşturduğunuzda, bu grafikler plt.show() çağrısı yapılmadan önce biriktirilir.
# plt.show() çağrısı yapıldığında, bu grafikler ekranda görüntülenir.
# 4.Model Oluşturma: skitlearn'de kullanabileceğimiz birçok sınıflandırma alg. vardır. Burada biz K-En yakın komşu(KNN) alg.
# kullanacağız. from sklearn.neighbors import KNeighborsClassifier ile KNN algoritmasını kullanacağımız metodu import ettik.

knn = KNeighborsClassifier(n_neighbors=1) #burada n_neighbors=1 demek en yakın komşu sayısını 1 olarak belirledik demek.
# Modeli test ederken en yakın 1 tane komşusuna göre sınıflandıracak. Burada knn bir objedir.
knn.fit(X_egitim, y_egitim) #modeli eğitmek için knn nesnesinin fit metodunu çağırıyoruz
X_yeni = np.array([[5,2.9,1,0.2]]) #numpy arrayi oluşuturuyoruz ve numpy arrayi mutlaka iki boyutlu olmak zorundadır.
tahmin = knn.predict(X_yeni)
print('tahmin sınıfı: ',tahmin)
print("tahmin türü: ", iris['target_names'][tahmin])

# 5.Tahmin: modeli eğitim verileri ile eğittikten sonra test verileri ile de test edip çıkan sonucun doğru olup olmadığının
# doğruluk skoruna bakabiliriz.
y_tahmin = knn.predict(X_test) #burada X_test yani test verilerini gönderiyoruz ve tahmin ettiği çıktıları y_tahmin
# değişkenine atıyoruz. Daha sonra bu tahmin ettikleri ile gerçek etiketleri ne kadar tutturmuş diye bakacağız.
print(y_tahmin)
print(np.mean(y_tahmin==y_test)) #tahmin ettiği etiketler ile gerçek etiketlerin ne kadar tutturduğunun ortalamasını buluyoruz
# 0.9736842105263158 çıktı. Yani model içine verdiğimiz bir iris kaydını(iki boyutlu numpy dizisi olmalı) bu oranda doğru tahmin edecek
#ya da knn.score ile de bulabiliriz:
print(knn.score(X_test,y_test))
