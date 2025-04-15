# DDPM (Denoising Diffusion Probabilistic Model)

DDPM iki temel aşamadan oluşur:

**İleri Süreç (Forward Process):** Gerçek veriye gürültü ekleyerek tamamen rastgele bir hale getiririz.

**Geri Süreç (Reverse Process):** Rastgele gürültüden başlayarak, bu gürültüyü adım adım temizleyip orijinal veriyi geri oluştururuz.

Bu iki süreçte, UNet kullanılır. UNet, her adımda gürültüyü ne kadar temizleyeceğimizi tahmin eder.

### İleri Süreç (Forward Process)

İleri süreçte, orijinal veriye ($x_0$) adım adım gürültü eklenir ve $t$ adımında $x_t$ elde edilir. Bu süreç bir Markov zinciridir ve her adımda Gaussian gürültü eklenir.

Formül:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

Burada:
- $q(x_t | x_{t-1})$: $t-1$ adımından $t$ adımına geçiş olasılığı.
- $\mathcal{N}$: Gaussian (normal) dağılım.
- $\beta_t$: $t$ adımında eklenen gürültü miktarı.
- $I$: Birim matris.

### İleri Süreç Kısayolu

Her adımı tek tek hesaplamak yerine, $x_0$'dan direkt $x_t$'ye şu formülle geçebiliriz:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

Burada:
- $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$: Tüm adımlardaki gürültü birikimini temsil eder.
- $\epsilon \sim \mathcal{N}(0, I)$: Rastgele Gaussian gürültü.


### Geri Süreç (Reverse Process)

Geri süreçte, rastgele gürültüden ($x_T$) başlayarak orijinal veriye ($x_0$) ulaşılır. Bu süreç, sinir ağı tarafından tahmin edilir.

Formül:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Burada:
- $p_\theta$: Sinir ağı parametreleri $\theta$ ile tanımlı olasılık.
- $\mu_\theta(x_t, t)$: Sinir ağının tahmin ettiği ortalama.
- $\Sigma_\theta(x_t, t)$: Varyans (genelde $\beta_t I$ olarak alınır) veya bir ölçeklendirilmiş versiyonu:

$$\Sigma_\theta(x_t, t) = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t I$$


### Geri Süreç Ortalaması

Geri süreçte sinir ağı, gürültüyü ($\epsilon$) tahmin eder ve ortalama şu şekilde hesaplanır:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{1-\beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

Burada:
- $\epsilon_\theta(x_t, t)$: Sinir ağının $t$ adımında tahmin ettiği gürültü.

### Kayıp Fonksiyonu

DDPM'yi eğitmek için kullanılan kayıp fonksiyonu, gerçek gürültü ile tahmin edilen gürültü arasındaki farkı minimize eder.

Formül:

$$L = \mathbb{E}_{x_0, \epsilon, t} \left[ || \epsilon - \epsilon_\theta(x_t, t) ||^2 \right]$$

Burada:
- $\mathbb{E}$: Beklenen değer (ortalama).
- $|| \cdot ||^2$: Mean Squared Error (MSE).

## UNet Mimarisi

UNet, şu temel bölümlerden oluşur:

**Down Bloklar:** Görüntüyü sıkıştırır (örneğin, 1024x1024'den 512x512'ye, sonra 256x256'ya). ResNet + Self Attention and Down Sample blok

**Mid Blok:** En küçük boyutta (örneğin, 64x64) yoğun özellik işleme yapar.

**Up Bloklar:** Sıkıştırılmış veriyi tekrar genişletir (64x64'ten 1024x1024'e geri döner).

Her blokta genellikle konvolüsyon katmanları, normalizasyon (örneğin grup normalizasyonu), aktifleştirme fonksiyonları (ReLU veya SiLU) ve dikkat mekanizmaları (self-attention veya cross-attention) bulunur.

- **Time Embedding:** DDPM'de, modelin her zaman adımı ($t$) için hangi gürültü seviyesinde çalıştığını bilmesi gerekir. Örneğin, t = 1 (az gürültü) ile t=1000 (çok gürültü) farklı durumları temsil eder. Zaman embedding, $t$'yi bir skaler sayıdan (örneğin, 500) sabit boyutta bir vektöre dönüştürür. Bu vektör, modelin (genellikle bir U-Net) $t$'ye bağlı olarak gürültü tahmini yapmasını sağlar. Bu fonksiyon, sinüzoidal embedding'ler kullanarak her $t$ için benzersiz bir vektör üretir, tıpkı Transformer'lardaki pozisyonel embedding'ler gibi. Sinüs ve kosinüs fonksiyonları, farklı frekanslarla $t$'yi temsil eder.

- **Cross-Attention:** Stable Diffusion'ın metin yönlendirmeli çalışmasını sağlar. Örneğin, "bir kedi" prompt'u, mid blokta görüntüye entegre edilir.

![UNet Architecture](https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs11554-021-01166-z/MediaObjects/11554_2021_1166_Fig7_HTML.png)
