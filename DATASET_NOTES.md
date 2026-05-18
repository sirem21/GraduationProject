# Veri Seti Varsayılan (Default) Değerleri

`sample_dataset_cleared.csv` dosyası oluşturulurken veride karşılığı bulunmayan veya dönüştürülmesi gereken özellikler (features) için aşağıdaki varsayılan (default) değerler kullanılmıştır:

1. **glucose_lvl:** Veri setinde sayısal glikoz değeri bulunmadığından, modeli saptırmaması adına ortalama bir değer olan **`100`** atanmıştır.
2. **blood_p:** Veri setinde sayısal tansiyon (kan basıncı) değeri bulunmadığından, yine modeli saptırmaması adına standart sistolik tansiyon ortalaması olan **`120`** atanmıştır.

**Not:** `bmi` özelliği ise "Boyunuz (cm)" ve "Kilonuz (kg)" sütunları kullanılarak matematiksel formülle (Kilo / Boy²) hesaplanmış ve veriye eklenmiştir. Evet/Hayır gibi tüm Türkçe metin yanıtları ise algoritmamız tarafından `1` ve `0` formatına çevrilerek kaydedilmiştir.
