import pandas
import numpy
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self):
        self.olasiliklar = {}

    def fit(self, X, y):
        self.olasiliklar['class_sayisi'] = len(set(y))

        for sinif in set(y):
            sinif_olasilik = len(y[y == sinif]) / len(y)
            self.olasiliklar[sinif] = {'olasilik': sinif_olasilik, 'veri': {}}

            for sutun in X.columns:
                self.olasiliklar[sinif]['veri'][sutun] = {}

                for deger in set(X[sutun]):
                    veri_olasilik = (len(X[(X[sutun] == deger) & (y == sinif)]) + 1) / (len(y[y == sinif]) + len(set(X[sutun])))
                    self.olasiliklar[sinif]['veri'][sutun][deger] = veri_olasilik

    def predict(self, X):
        tahminler = []
        for _, veri in X.iterrows():
            tahmin = self._sinif_tahmini(veri)
            tahminler.append(tahmin)
        return tahminler

    def _sinif_tahmini(self, veri):
        en_yuksek_olasilik = -numpy.inf
        secili_sinif = None
        for sinif in range(self.olasiliklar['class_sayisi']):
            sinif_olasilik = numpy.log(self.olasiliklar[sinif]['olasilik'])
            for sutun, deger_olasiklikleri in self.olasiliklar[sinif]['veri'].items():
                deger = veri[sutun]
                if deger in deger_olasiklikleri:
                    sinif_olasilik += numpy.log(deger_olasiklikleri[deger])
                else:
                    sinif_olasilik += numpy.log(1e-5)
            if sinif_olasilik > en_yuksek_olasilik:
                en_yuksek_olasilik = sinif_olasilik
                secili_sinif = sinif
        return secili_sinif


def eksikVerileriTamamla(veriseti):
    verisetiKopyasi = veriseti.copy()

    for sutun in verisetiKopyasi.columns:
        if verisetiKopyasi[sutun].isnull().any():
            if set(verisetiKopyasi[sutun]) == {0, 1}:
                enFazlaTekrarEdenDeger = verisetiKopyasi[sutun].mode()[0]
                verisetiKopyasi[sutun].fillna(enFazlaTekrarEdenDeger, inplace=True)
            else:
                verisetiKopyasi[sutun].fillna(verisetiKopyasi[sutun].mean(), inplace=True)

    return verisetiKopyasi


def verileriNormalizeEt(veriseti):
    verisetiKopyasi = veriseti.copy()
    for sutun in verisetiKopyasi.columns:
        min_deger = verisetiKopyasi[sutun].min()
        max_deger = verisetiKopyasi[sutun].max()

        verisetiKopyasi[sutun] = (verisetiKopyasi[sutun] - min_deger) / (max_deger - min_deger)

    return verisetiKopyasi


def kfoldAyirma(veriseti, k):
    veriseti = veriseti.sample(frac=1).reset_index(drop=True)
    veri_uzunlugu = len(veriseti)

    kat_uzunlugu = veri_uzunlugu // k

    egitim_verileri = []
    test_verileri = []

    for i in range(k):
        test_baslangic = i * kat_uzunlugu
        test_bitis = (i + 1) * kat_uzunlugu

        test_verileri.append(veriseti.iloc[test_baslangic:test_bitis, :])

        if i == 0:
            egitim_verileri.append(veriseti.iloc[test_bitis:, :])
        elif i == k - 1:
            egitim_verileri.append(veriseti.iloc[:test_baslangic, :])
        else:
            egitim_verileri.append(pandas.concat([veriseti.iloc[:test_baslangic, :], veriseti.iloc[test_bitis:, :]]))

    return egitim_verileri, test_verileri


def siniflandir(egitim,test):
    naive_bayes_model = NaiveBayes()

    dogruluk_skorlari = []
    stSapmalar = []
    for egitim_verileri, test_verileri in zip(egitim, test):
        egitim_X = egitim_verileri.iloc[:, :-1]
        egitim_y = egitim_verileri.iloc[:, -1]

        naive_bayes_model.fit(egitim_X, egitim_y)

        tahminler = naive_bayes_model.predict(test_verileri.iloc[:, :-1])
        dogruluk_skor = numpy.sum(tahminler == test_verileri.iloc[:, -1]) / len(test_verileri)
        dogruluk_skorlari.append(dogruluk_skor)
        standart_sapma = numpy.std(test_verileri,axis=0)
        stSapmalar.append(standart_sapma)

    print("K-Fold Cross-Validation Doğruluk Skorları:", dogruluk_skorlari)
    print("Ortalama Doğruluk Skoru:", numpy.mean(dogruluk_skorlari))
    print("Ortalama Standart Sapma:", numpy.mean(stSapmalar))


    return dogruluk_skorlari



veriYolu = "./diabetes1.xlsx"
veriseti = pandas.read_excel(veriYolu)
etiketSayisi = len(veriseti.columns) - 1
veriSayisi = len(veriseti)
k_degeri = 10

veriseti = eksikVerileriTamamla(veriseti)

normalizeVeriseti = verileriNormalizeEt(veriseti)

NormalizeOlmayanEgitim,NormalizeOlmayanTest = kfoldAyirma(veriseti,k_degeri)

NormalizeEdilmisEgitim,NormalizeEdilmisTest = kfoldAyirma(normalizeVeriseti,k_degeri)


# K-fold cross-validation ile Naive Bayes modelini eğit ve değerlendir

print("Min-Max Normalizasyonu Kullanılmadan Önceki Sınıflandırma Sonuçları: ")
dogruluk_skorlari_normalize_olmayan = siniflandir(NormalizeOlmayanEgitim,NormalizeOlmayanTest)

print("Min-Max Normalizasyonu Kullanıldıktan Sonraki Sınıflandırma Sonuçları: ")
dogruluk_skorlari_normalize_edilmis = siniflandir(NormalizeEdilmisEgitim,NormalizeEdilmisTest)


plt.figure(figsize=(12, 7))


plt.plot(range(1, k_degeri + 1), dogruluk_skorlari_normalize_olmayan, marker='o', linestyle='-', color='b', label='Normalleştirilmemiş Veriler')


plt.plot(range(1, k_degeri + 1), dogruluk_skorlari_normalize_edilmis, marker='o', linestyle='-', color='r', label='Normalleştirilmiş Veriler')

plt.title('K-Fold Cross-Validation Doğruluk Skorları')
plt.xlabel('Fold Numarası')
plt.ylabel('Doğruluk Skoru')
plt.legend()
plt.grid(True)
plt.show()