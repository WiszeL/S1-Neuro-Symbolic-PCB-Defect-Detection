**Skripsi**

**DETEKSI CACAT PCB DENGAN PENJELASAN YANG FAITHFUL MELALUI ARSITEKTUR NEURO-SYMBOLIC FASTER R-CNN DAN SPARSE OBLIQUE DECISION TREE**

**[Logo UNS]**

**Disusun Oleh :**

**HANDI DWI CAHYO**

**L0122072**

**FAKULTAS TEKNOLOGI INFORMASI DAN SAINS DATA**

**UNIVERSITAS SEBELAS MARET**

**SURAKARTA**

**2026**

**[Lembar Persetujuan]**

# PENGESAHAN

Proposal Skripsi Mahasiswa :

HANDI DWI CAHYO

L0122072

Dengan Judul

**DETEKSI CACAT PCB DENGAN PENJELASAN YANG FAITHFUL MELALUI ARSITEKTUR NEURO-SYMBOLIC FASTER R-CNN DAN SPARSE OBLIQUE DECISION TREE**

Telah diseminarkan pada tanggal, ...

Susunan Penguji Seminar Proposal

1. Nama Pembimbing 1				(                ttd                 )

NIP .....

2. Nama Pembimbing 2				(                ttd                 )

NIP .....

3. Nama Penguji Utama				(                ttd                 )

NIP .....

Mengetahui,

| Koordinator Skripsi |  | Ketua Jurusan .... |
| --- | --- | --- |
|  |  |  |
| NIP .... |  | NIP .... |

# BAB I PENDAHULUAN

## 1.1 Latar Belakang

Perkembangan teknologi elektronik meningkatkan kebutuhan terhadap komponen yang andal dan berkualitas tinggi, khususnya *Printed Circuit Board* (PCB). PCB berfungsi sebagai penopang sekaligus penghubung antar-komponen yang menentukan keandalan operasional sistem elektronik. Untuk memenuhi tuntutan keandalan tersebut, PCB kemudian dikembangkan dengan arsitektur yang semakin kompleks. Namun, kompleksitas ini justru meningkatkan risiko munculnya cacat pada PCB akibat kesalahan manusia maupun gangguan mesin selama proses produksi (Chen et al., 2023; Fung et al., 2024). Keberadaan cacat tersebut dapat berdampak pada penurunan kualitas produk, peningkatan biaya produksi, hingga risiko keselamatan manusia (Ali et al., 2024; Fung et al., 2024). Dengan demikian, metode inspeksi menjadi tahapan krusial untuk mendeteksi keberadaan cacat pada PCB.

Tahap inspeksi PCB selama ini bertumpu pada metode konvensional seperti pemeriksaan visual manual dan pengujian kelistrikan. Namun, metode-metode tersebut kerap dinilai tidak efisien, memakan biaya tinggi, dan rentan terhadap kesalahan akibat faktor manusia (Ali et al., 2024; Chen et al., 2023; Fung et al., 2024). Keterbatasan tersebut mendorong pengembangan solusi otomatis berbasis *Artificial Intelligence* (AI), khususnya pendekatan *deep learning*. Berbagai model *deep learning* seperti *You Only Look Once* (YOLO), *Single Shot MultiBox Detector* (SSD), dan *Faster Region-based Convolutional Neural Network* (R-CNN) telah banyak diterapkan untuk deteksi cacat PCB dan menunjukkan performa yang tinggi dalam berbagai studi (Chen et al., 2023). Sebagai contoh, Wang et al. (2024) mengembangkan YOLO-RLC, yaitu YOLOv5 dengan jaringan konvolusi kernel besar residual yang meningkatkan akurasi deteksi cacat PCB. Bahkan, penelitian oleh Fung et al. (2024) menunjukkan bahwa optimasi pada arsitektur *Faster* R-CNN dapat meningkatkan performa deteksi cacat PCB, terutama pada cacat yang berukuran kecil. Hasil beberapa penelitian tersebut menunjukkan bahwa pendekatan berbasis *deep learning* merupakan pendekatan yang dapat diandalkan untuk tahap inspeksi.

Meskipun pendekatan *deep learning* menunjukkan performa deteksi tinggi, model ini memiliki kelemahan berupa sifat *black-box,* yakni ketidakmampuan model dalam memberikan penjelasan atas setiap keputusan yang dihasilkannya. Padahal, dalam industri manufaktur berisiko tinggi seperti produksi PCB, penjelasan tersebut diperlukan teknisi untuk memvalidasi dan mempertanggungjawabkan hasil deteksi  (Tzionis et al., 2026). Tanpa penjelasan yang memadai, proses validasi menjadi sulit dilakukan sehingga berpotensi menurunkan kepercayaan teknisi terhadap model (Tziolas et al., 2025). Oleh karena itu, kemampuan model dalam menjelaskan keputusan (*explainability*) menjadi kebutuhan esensial guna mendukung proses validasi teknisi dalam inspeksi cacat PCB (Saadallah et al., 2022; Chen et al., 2023).

Untuk merespons kebutuhan *explainability* tersebut, *Explainable Artificial Intelligence* (XAI) menjadi pendekatan yang dominan diterapkan dalam sistem inspeksi manufaktur (Tzionis et al., 2026). Pendekatan XAI menghadirkan *explainability* dengan membuat proses pengambilan keputusan model AI dapat ditelusuri dan dipahami. Dalam inspeksi PCB, penelitian Tziolas et al. (2025) menunjukkan bahwa *Deep Shapley Additive Explanations* (SHAP) dan *Gradient-weighted Class Activation Mapping* (Grad-CAM) dapat menyoroti area visual yang memengaruhi keputusan model CNN, sehingga meningkatkan interpretabilitas model bagi teknisi. Penelitian lain oleh Saadallah et al. (2022) menggunakan penjelasan berbasis *heatmap* yang mengungkap fitur-fitur penting pada citra PCB, sehingga membantu teknisi dalam memvalidasi relevansi fitur tersebut terhadap jenis cacat yang terdeteksi. Dengan demikian, XAI mampu memenuhi kebutuhan *explainability* pada inspeksi PCB melalui justifikasi visual atas keputusan model.

Meskipun informatif, metode XAI Grad-CAM dan SHAP pada studi tersebut, masih memiliki keterbatasan berupa sifat *post-hoc*, yaitu penjelasan yang baru diberikan setelah model *black-box* menghasilkan prediksinya. Mekanisme ini menyebabkan penjelasan yang dihasilkan tidak memenuhi *faithfulness*, yaitu kemampuannya dalam merepresentasikan keputusan model yang sesungguhnya, sehingga penjelasan tersebut hanyalah berupa perkiraan (Rudin, 2019). Pada domain berisiko tinggi, peta *saliency post-hoc*, termasuk Grad-CAM, terbukti belum sepenuhnya dapat diandalkan (Arun et al., 2021), sehingga validasi teknisi yang bertumpu padanya dalam inspeksi PCB berpotensi mengarah pada keputusan yang salah. Dengan demikian, ketiadaan *faithfulness* dalam penjelasan *post-hoc* justru mengurangi kualitas *explainability* yang dibutuhkan untuk validasi teknisi (Rudin, 2019).

Untuk mengatasi keterbatasan *faithfulness* dalam penjelasan model, arsitektur *neuro-symbolic* hadir dengan menggabungkan ekstraksi fitur dari *deep learning* dan penalaran transparan berbasis simbolik. Dalam arsitektur ini, komponen simbolik terintegrasi langsung ke dalam mekanisme pengambilan keputusan sehingga penjelasan yang dihasilkan bersifat *faithful* (d'Avila Garcez & Lamb, 2023). Arsitektur ini telah ditunjukkan oleh model *Faster*-LTN yang mengintegrasikan *Faster* R-CNN dengan *Logic Tensor Network* (LTN) untuk mempertahankan performa deteksi sekaligus memberikan penalaran terstruktur (Manigrasso et al., 2021). Sejalan dengan arah tersebut, Hada et al. (2024) serta Kairgeldin dan Carreira-Perpiñán (2025) mengembangkan integrasi CNN dengan *sparse oblique decision tree* (SODT) yang membuat proses pengambilan keputusan model lebih dapat diinterpretasikan dan divisualisasikan. Dengan demikian, ketiga penelitian tersebut berpotensi menghadirkan *faithfulness* sehingga meningkatkan kualitas *explainability* pada inspeksi PCB.

Untuk mewujudkan *explainability* yang *faithfulness*, penelitian ini mengusulkan integrasi *Faster* R‑CNN dengan SODT sebagai sistem deteksi cacat berbasis arsitektur *neuro‑symbolic*. *Faster* R‑CNN dipilih karena terbukti efektif mendeteksi cacat berukuran kecil pada PCB melalui optimalisasi SF‑PSPyramid (Fung et al., 2024), sehingga berperan sebagai komponen ekstraksi fitur visual dan deteksi objek. Sementara itu, SODT dipilih karena kemampuannya dalam meniru (*mimic)* keputusan jaringan saraf *teacher* dengan akurasi tinggi, namun dengan mekanisme eliminasi fitur (*sparsity*) yang menghasilkan struktur pohon lebih sederhana (Hada et al., 2024; Kairgeldin & Carreira‑Perpiñán, 2025). Integrasi ini dirancang untuk mempertahankan performa deteksi tinggi dari *Faster* R‑CNN sekaligus menghadirkan penjelasan yang *faithful*. Dengan demikian, sistem ini ditujukan untuk mendukung validasi teknisi dalam inspeksi cacat PCB.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, rumusan masalah dalam penelitian ini adalah sebagai berikut.

1. Bagaimana model *deep learning* dapat mempertahankan performa deteksi tinggi pada inspeksi PCB sekaligus mengatasi sifat *black‑box* yang menghambat validasi teknisi?
2. Bagaimana arsitektur *neuro‑symbolic* yang mengintegrasikan *Faster* R‑CNN dengan *Sparse Oblique Decision Tree* (SODT) dapat dikembangkan untuk menghadirkan sistem deteksi cacat PCB yang akurat sekaligus menyediakan *explainability* yang *faithful*?

## 1.3 Batasan Masalah

Agar penelitian ini tetap terarah dan fokus sesuai dengan tujuan yang telah ditetapkan, ruang lingkup permasalahan dibatasi pada hal-hal sebagai berikut.

1. Objek Penelitian

Penelitian difokuskan pada deteksi enam jenis cacat visual PCB, yaitu *open*, *short*, *mousebite*, *spur*, *pinhole*, dan *spurious copper*, menggunakan pendekatan *deep learning* dan *neuro‑symbolic*.

2. Arsitektur Model

Model merupakan integrasi Faster R‑CNN (mengacu pada implementasi Fung et al., 2024) dengan *Sparse Oblique Decision Tree* (SODT) tanpa modifikasi terhadap struktur internal kedua komponen.

3. Sumber Data

Data yang digunakan berasal dari dataset publik DeepPCB yang memuat 1.500 pasang citra beserta anotasi posisi dan kelas cacat.

4. Metode Evaluasi *Explainability*

Evaluasi kualitas *explainability* dilakukan secara kuantitatif melalui perbandingan dengan Grad‑CAM sebagai *baseline post‑hoc* yang telah teruji, tanpa melibatkan studi pengguna atau wawancara teknisi.

## 1.4 Tujuan Penelitian

1. Mengembangkan sistem deteksi cacat PCB berbasis integrasi Faster R CNN dan Sparse Oblique Decision Tree (SODT) yang mampu mempertahankan performa deteksi tinggi sekaligus menyediakan penjelasan yang faithful guna mendukung proses validasi teknisi.
2. Mengevaluasi apakah sistem yang diusulkan mampu menghasilkan penjelasan yang *faithful* dan lebih baik dibandingkan pendekatan *post-hoc* Grad-CAM.

## 1.5 Manfaat Penelitian

1. Mendukung proses validasi teknisi di industri manufaktur elektronik melalui sistem deteksi cacat PCB yang tidak hanya akurat, tetapi juga menyediakan penjelasan yang *faithful* dan dapat dipertanggungjawabkan.
2. Memberikan bukti empiris bahwa integrasi Faster R CNN dan *Sparse Oblique Decision Tree* (SODT) mampu mengatasi keterbatasan *faithfulness* yang melekat pada pendekatan XAI *post-hoc*.

# BAB II TINJAUAN PUSTAKA

## 2.1 Dasar Teori

### 2.1.1 *Printed Circuit Board* (PCB)

*Printed Circuit Board* (PCB) adalah papan dari bahan isolator yang dilapisi tembaga. PCB berfungsi sebagai jalur penghubung listrik sekaligus penyangga antarkomponen elektronik (Coombs & Holden, 2016; Khandpur, 2005).

Kriteria inspeksi PCB di industri mengacu pada standar *Association Connecting Electronics Industries* (IPC), khususnya IPC-A-600 dan IPC-6012 (IPC, 2015, 2020). Standar ini menyatakan suatu kondisi sebagai cacat apabila melanggar batas toleransi, misalnya jalur konduktor yang terlalu sempit atau jarak antarjalur yang terlalu dekat. Dalam *Computer Vision*, pelanggaran tersebut dipandang sebagai cacat visual yang polanya dapat dipelajari oleh model *deep learning* (Tang et al., 2019; Chen et al., 2023). Enam jenis cacat yang umum dipakai pada penelitian deteksi PCB dirangkum pada Tabel 2.1.

**Tabel 2.1 Kategori Cacat Visual Umum pada PCB.**

| Jenis Cacat | Definisi & Karakteristik Visual |
| --- | --- |
| *Open* | Jalur tembaga terputus, sehingga muncul celah yang memutus jalur. |
| *Short* | Dua jalur yang seharusnya terpisah tetapi menjadi tersambung, membentuk jembatan tembaga. |
| *Mousebite* | Tepi jalur tergerus seperti gigitan, sehingga lebar jalur berkurang secara tidak merata. |
| *Spur* | Tonjolan tembaga kecil yang keluar dari tepi jalur menuju area yang seharusnya kosong. |
| *Pinhole* | Tembaga liar yang muncul terpisah di area non-konduktif, tidak terhubung ke jalur utama. |
| *Spurious Copper* | Lubang kecil berbentuk lingkaran di dalam area tembaga yang seharusnya padat. |

Keenam cacat tersebut terbagi menjadi dua kelompok pola. Kelompok pertama adalah pengurangan material (*open*, *mousebite*, *pinhole*), yaitu hilangnya sebagian tembaga. Kelompok kedua adalah penambahan material liar (*short*, *spur*, *spurious copper*), yaitu munculnya tembaga di tempat yang salah.

### 2.1.2 *Computer Vision*

*Computer Vision* (CV) adalah cabang ilmu komputer yang membuat komputer mampu memahami isi gambar secara otomatis (Szeliski, 2022). Berbeda dengan pengolahan citra yang hanya memanipulasi piksel, CV bertujuan menghasilkan makna atau keputusan dari isi gambar (Prince, 2023). CV mencakup tiga tugas utama yang dibedakan berdasarkan kedetailan keluarannya.

1. **Klasifikasi Citra** (*Image Classification*) memberikan satu label kelas untuk keseluruhan gambar, tanpa memperhatikan letak objek (Prince, 2023).
2. **Deteksi Objek** (*Object Detection*) mengenali objek, menentukan kelasnya, sekaligus menunjukkan posisinya menggunakan kotak pembatas (*bounding box*) (Szeliski, 2022).
3. **Segmentasi Citra** (*Image Segmentation*) memberi label pada setiap piksel, sehingga batas objek tergambar lebih presisi (Minaee et al., 2021).

Perbandingan ketiga tugas tersebut ditunjukkan pada Gambar 2.1.

[Gambar 2.1]

**Gambar 2.1 Perbandingan Klasifikasi, Deteksi Objek, dan Segmentasi**

Deteksi objek merupakan tugas yang relevan untuk inspeksi visual, karena selain mengenali jenis cacat juga menunjukkan lokasinya. Representasi dan sistem koordinat *bounding box* ditunjukkan pada Gambar 2.2.

[Gambar 2.2]

**Gambar 2.2 Anatomi dan Representasi Koordinat Bounding Box**

### 2.1.3 *Deep Learning*

*Deep learning* adalah cabang *machine learning* yang mempelajari representasi data secara bertingkat melalui banyak lapisan pemrosesan (Goodfellow et al., 2016). *Machine learning* konvensional bergantung pada fitur rancangan manusia, sedangkan *deep learning* memperoleh fitur secara otomatis dari data mentah (Han et al., 2017). Perbedaan alur kerja keduanya ditunjukkan pada Gambar 2.3.

[Gambar 2.3]

**Gambar 2.3 Perbedaan cara kerja Machine Learning dan Deep Learning**

Pelatihan jaringan saraf tiruan terdiri atas empat komponen utama.

1. ***Forward Propagation***

Propagasi maju menghasilkan prediksi dari data masukan melalui operasi berlapis. Pada setiap lapisan, keluaran lapisan sebelumnya dikalikan dengan matriks bobot, ditambah bias, lalu dilewatkan ke fungsi aktivasi (Goodfellow et al., 2016).

2. **Fungsi Aktivasi**

Fungsi aktivasi memberikan sifat non-linear pada jaringan sehingga model mampu memodelkan hubungan yang kompleks. Fungsi yang umum digunakan adalah *Rectified Linear Unit* (ReLU) pada Persamaan 2.1, karena ringan secara komputasi dan mengurangi masalah *vanishing gradient* (LeCun et al., 2015).

$$
f\left(x\right)=\max{\left(0,x\right)}
$$

(2.1)

dengan $x$ nilai masukan dan $f\left(x\right)$ nilai keluaran *neuron*.

3. ***Loss Function***

Fungsi kerugian mengukur selisih antara prediksi model dan nilai sebenarnya (*ground truth*). Untuk klasifikasi banyak kelas, fungsi kerugian yang umum digunakan adalah *Cross-Entropy Loss* pada Persamaan 2.2 (Goodfellow et al., 2016).

$$
L=-\sum_{i=1}^{C}{{y}_{i}}\log{\left({p}_{i}\right)}
$$

(2.2)

dengan $C$ jumlah kelas, ${y}_{i}$ label sebenarnya kelas ke-*i* dalam bentuk *one-hot encoding*, dan ${p}_{i}$ probabilitas prediksi kelas ke-*i*.

4. ***Back propagation***

Propagasi mundur menghitung gradien fungsi kerugian terhadap setiap parameter menggunakan aturan rantai (*chain rule*), lapisan demi lapisan dari keluaran menuju masukan (LeCun et al., 2015). Gradien terhadap parameter dipakai oleh *Stochastic Gradient Descent* (SGD) untuk memperbarui parameter, sebagaimana dinyatakan pada Persamaan 2.3 (Goodfellow et al., 2016).

$$
w\leftarrow w-η\frac{\partial L}{\partial w}
$$

(2.3)

dengan $w$ parameter bobot, $η$ laju pembelajaran (*learning rate*), dan $\frac{\partial L}{\partial w}$ gradien fungsi kerugian terhadap bobot.

Siklus propagasi maju dan mundur diulang hingga model konvergen, sebagaimana diilustrasikan pada Gambar 2.4.

[Gambar 2.4]

**Gambar 2.4 Diagram Alur Kerja Siklus Pelatihan Jaringan Saraf Tiruan**

### 2.1.4 *Convolution Neural Network* (CNN)

*Convolutional Neural Network* (CNN) adalah jaringan saraf tiruan yang dirancang untuk data berbentuk *grid*, seperti citra (Goodfellow et al., 2016; LeCun et al., 2015). Efisiensinya bertumpu pada konektivitas lokal, yaitu setiap neuron hanya melihat sebagian kecil citra, dan berbagi parameter, yaitu filter yang sama dipakai di seluruh citra (LeCun et al., 2015). Arsitektur umum CNN ditunjukkan pada Gambar 2.5.

[Gambar 2.5]

**Gambar 2.5 Ilustrasi Arsitektur Convolutional Neural Network (CNN)**

Pada Gambar 2.5, arsitektur CNN terdiri atas tiga jenis lapisan utama.

1. ***Convolutional Layer***

*Layer* ini mengekstraksi fitur lokal dengan menggeser filter (*kernel*) terpelajar di atas citra atau peta fitur (*feature map*). Setiap filter mengenali pola tertentu seperti tepi, sudut, atau tekstur (Zhao et al., 2024; Goodfellow et al., 2016). Keluarannya dilewatkan ke fungsi aktivasi seperti ReLU pada Persamaan 2.1.

2. ***Pooling Layer***

Berfungsi untuk memperkecil dimensi spasial *feature map* (*downsampling*), misalnya dengan *max pooling* yang mengambil nilai terbesar pada setiap jendela, sehingga beban komputasi berkurang dan model lebih tahan terhadap pergeseran kecil.

3. ***Fully Connected Layer***

Berupa *Multi-Layer Perceptron* (MLP) yang menggabungkan fitur tingkat tinggi menjadi keputusan akhir (*classifier*). Masukannya berupa *flatten vector* dari *feature map*. Skor mentah (*logit*) keluaran kepala klasifikasi diubah menjadi probabilitas oleh fungsi *softmax* pada Persamaan 2.4.

$$
{p}_{i}=\frac{{e}^{{z}_{i}}}{\sum_{j=1}^{C}{{e}^{{z}_{j}}}}
$$

(2.4)

dengan ${p}_{i}$ probabilitas kelas ke-*i*, ${z}_{i}$ *logit* kelas ke-*i*, dan $C$ jumlah kelas.

Lapisan konvolusi dan *pooling* menghasilkan *feature map* berukuran $C\times H\times W$. Setiap kanal merupakan respons satu filter, sedangkan posisi $\left(h \middle| w\right)$ menyatakan lokasi respons tersebut (Goodfellow et al., 2016). *Stride* adalah besar langkah pergeseran filter. *Stride* total lapisan ke-*l* dan posisi citra yang berkorespondensi dengan sel $\left(h \middle| w\right)$ dinyatakan pada Persamaan 2.5 dan 2.6 (Goodfellow et al., 2016).

$$
{S}_{l}=\prod_{i=1}^{l}{{s}_{i}}
$$

(2.5)

$$
\left(u,v\right)=\left({S}_{l}⋅w,{S}_{l}⋅h\right)
$$

(2.6)

dengan ${S}_{l}$ *stride* total lapisan ke-*l*, ${s}_{i}$ *stride* lapisan ke-*i*, $\left(h \middle| w\right)$ indeks baris dan kolom pada *feature map*, dan $\left(u \middle| v\right)$ koordinat pada citra.

Persamaan 2.6 baru menetapkan titik koordinat korespondensi, belum luas *region* citra yang benar-benar memengaruhi sel tersebut. Luas tersebut dinyatakan oleh *Receptive Field*, yaitu *region* citra yang memengaruhi nilai satu sel *feature map* (Goodfellow et al., 2016; Luo et al., 2016). Ukurannya membesar seiring kedalaman jaringan, sebagaimana dinyatakan pada Persamaan 2.7.

$$
{r}_{l}={r}_{l-1}+\left({k}_{l}-1\right)\prod_{i=1}^{l-1}{{s}_{i}}, {r}_{0}=1
$$

(2.7)

dengan ${r}_{l}$ ukuran *receptive field* lapisan ke-*l* dalam piksel, ${k}_{l}$ ukuran kernel lapisan ke-*l*, dan ${s}_{i}$ *stride* lapisan ke-*i*.

*Stride* total pada Persamaan 2.5 menentukan letak pusat daerah citra yang diwakili satu sel *feature map*, sedangkan *Receptive Field* pada Persamaan 2.7 menentukan luas daerah tersebut. Keduanya membuat setiap sel dapat dipetakan kembali ke *region* citranya sendiri, sehingga bobot atau atribusi pada sel *feature map* dapat diterjemahkan menjadi penjelasan spasial pada citra di Subbab 2.1.8. Namun, *Receptive Field* antarsel saling tumpang-tindih dan pengaruh piksel di dalamnya menurun dari pusat ke tepi (Luo et al., 2016), sehingga penjelasan tersebut berlaku pada tingkat *regional*, bukan pada piksel. Ilustrasi stride dan receptive field ditunjukkan pada Gambar 2.6.

[Gambar 2.6]

**Gambar 2.6 Ilustrasi *Stride* dan *Receptive Field***

### 2.1.5 *Faster* R-CNN

*Faster* R-CNN adalah arsitektur deteksi objek dua tahap (*two-stage*) yang menyatukan pengusulan area dan klasifikasi dalam satu jaringan yang dilatih secara *end-to-end* (Ren et al., 2017). Keunggulannya terletak pada *Region Proposal Network* (RPN) yang menggantikan metode pencarian area eksternal, sehingga seluruh komponen berbagi *feature map* yang sama. Arsitektur standarnya ditunjukkan pada Gambar 2.7.

[Gambar 2.7]

**Gambar 2.7 Arsitektur Faster R-CNN Standar**

Penelitian ini menggunakan varian SF-PSPyramid (Fung et al., 2024), yaitu *Faster* R-CNN dengan *neck* yang dirancang untuk cacat berukuran mikro pada PCB, sebagaimana ditunjukkan pada Gambar 2.8.

[Gambar 2.8]

**Gambar 2.8 Arsitektur modifikasi Faster R-CNN dengan SF-PSPyramid**

1. **Backbone** 

*Backbone* mengubah citra menjadi *feature map*. Penelitian ini menggunakan ResNet-50, yang terdiri atas empat kelompok lapisan (${C}_{2}$, ${C}_{3}$, ${C}_{4}$, ${C}_{5}$) dengan resolusi menurun dan makna semantik meningkat (He et al., 2016). *Stride* total keempat kelompok tersebut berturut-turut 4, 8, 16, dan 32 piksel.

2. ***Neck (SF-PSPyramid)***

*Neck* menggabungkan fitur dari berbagai skala backbone menjadi piramida fitur. Dasarnya adalah *Feature Pyramid Network (FPN)*, yang menggabungkan jalur *bottom-up*, jalur *top-down*, dan koneksi lateral antartingkat (Lin et al., 2017). SF-PSPyramid menyempurnakan FPN dengan tiga perbedaan (Fung et al., 2024).

a. ***CP Block***

 Resolusi diperbesar melalui penataan ulang kanal (*pixel shuffle*), bukan interpolasi, sehingga bersifat terpelajar yang ditunjukkan pada Persamaan 2.8.

$$
PS{\left(T\right)}_{c,h,w}={T}_{c⋅{r}^{2}+r⋅mod\left(h,r\right)+mod\left(w,r\right),\left⌊h/r\right⌋,\left⌊w/r\right⌋}
$$

(2.8)

dengan $T$ tensor masukan, $r$ faktor pembesaran, serta $c,h,w$ indeks kanal, baris, dan kolom keluaran.

b. ***Selective Feature Attention***

Dua tingkat fitur digabungkan dengan bobot terpelajar yang dinormalisasi *softmax*, mengadaptasi *Selective Kernel Network* (Li et al., 2019), sebagaimana dinyatakan pada Persamaan 2.9.

$$
P'={\alpha}_{1}⊙U+{\alpha}_{2}⊙V, \left[{\alpha}_{1},{\alpha}_{2}\right]=softmax\left({W}_{2}\delta \left({W}_{1}z\right)\right)
$$

(2.9)

dengan $U$ *feature map* dari tingkat yang lebih dalam, $V$ *feature map* beresolusi lebih tinggi, $z$ vektor hasil *global average pooling*, ${W}_{1},{W}_{2}$ bobot lapisan kompresi dan perluasan, $\delta$ ReLU, dan $\operatorname{⊙}{}$ perkalian per kanal.

c. **Susunan piramida tanpa koneksi lateral**

SF-PSPyramid membentuk ${P}_{2}'$ dan ${P}_{3}'$ langsung dari gabungan ${C}_{2}$ hingga $C_5$ melalui CP Block pada Persamaan 2.8, bukan dari koneksi lateral seperti FPN standar, sehingga tetap beresolusi tinggi namun memuat informasi semantik penuh yang membantu deteksi cacat sangat kecil (Fung et al., 2024), sebagaimana ditunjukkan pada Gambar 2.8.

Keluaran neck adalah P2′, P3′, P4, P5, dan P6 (*max pooling* dari P5) dengan *stride* 4, 8, 16, 32, dan 64 piksel dan jumlah kanal *C* yang sama.

3. ***Region Proposal Network* (RPN)**

RPN menghasilkan usulan area kandidat objek. Pada setiap posisi *feature map* disiapkan sejumlah *anchor box* dengan beragam ukuran dan rasio, lalu setiap *anchor* memperoleh skor objektivitas dan empat nilai penyesuaian koordinat (Ren et al., 2015) pada Persamaan 2.10 sampai 2.13.

$$
{t}_{x}=\frac{x-{x}_{a}}{{w}_{a}}
$$

(2.10)

$$
{t}_{y}=\frac{y-{y}_{a}}{{h}_{a}}
$$

(2.11)

$$
{t}_{w}=\log{\left(\frac{w}{{w}_{a}}\right)}
$$

(2.12)

$$
{t}_{h}=\log{\left(\frac{h}{{h}_{a}}\right)}
$$

(2.13)

dengan $x,y,w,h$ pusat, lebar, dan tinggi kotak prediksi, serta ${x}_{a},{y}_{a},{w}_{a},{h}_{a}$ milik *anchor*. Pelatihan memakai *multi-task loss* pada Persamaan 2.14 (Ren et al., 2017).

$$
L=\frac{1}{{N}_{cls}}\sum_{i}{{L}_{cls}}\left({p}_{i},{p}_{i}^{*}\right)+\lambda \frac{1}{{N}_{reg}}\sum_{i}{{p}_{i}^{*}}{L}_{reg}\left({t}_{i},{t}_{i}^{*}\right)
$$

(2.14)

dengan ${p}_{i}$ probabilitas *anchor* ke-*i* memuat objek, ${p}_{i}^{*}$ label sebenarnya (1 positif, 0 negatif), ${N}_{cls},{N}_{reg}$ jumlah sampel tiap suku, ${L}_{cls}$ *binary cross-entropy*, ${L}_{reg}$ kerugian regresi, dan $\lambda$ faktor penyeimbang. Kerugian regresi yang digunakan adalah L1 pada Persamaan 2.15 (Fung et al., 2024).

$$
{L}_{reg}\left(t,{t}^{*}\right)=\sum_{j}{\left|{t}_{j}-{t}_{j}^{*}\right|}
$$

(2.15)

dengan *j* merentang pada keempat komponen koordinat, yaitu *x*, *y*, *w*, dan *h*.

4. ***RoI Align***

*RoI Align* menyeragamkan setiap proposal menjadi *tensor* berukuran tetap tanpa pembulatan koordinat (He et al., 2020). Nilai *feature map* pada koordinat pecahan diperoleh dengan interpolasi *bilinear* pada Persamaan 2.16.

$$
f\left(x,y\right)=\sum_{i=1}^{4}{{w}_{i}}{f}_{i}
$$

(2.16)

dengan ${f}_{i}$ nilai fitur pada empat titik grid terdekat dan ${w}_{i}$ bobot interpolasi yang berbanding terbalik dengan jarak.

Setiap proposal dibagi menjadi grid $G\times G$ *bin*. Pada setiap *bin* diambil beberapa titik sampel, dan nilai *bin* adalah rata-rata titik sampelnya (Persamaan 2.17) (He et al., 2020).

$$
{x}_{c,p,q}=\frac{1}{N}\sum_{n=1}^{N}{\sum_{i=1}^{4}{{w}_{n,i} {F}_{c}\left({h}_{n,i},{v}_{n,i}\right)}}
$$

(2.17)

dengan ${x}_{c,p,q}$ keluaran kanal ke-*c* pada *bin* $\left(p \middle| q\right)$, $N$ jumlah titik sampel per *bin*, ${w}_{n,i}$ bobot interpolasi titik sampel ke-*n*, dan ${F}_{c}\left({h}_{n,i} \middle| {v}_{n,i}\right)$ nilai *feature map* kanal ke-*c* pada titik grid terdekat. Dalam bentuk matriks, Persamaan 2.17 dapat ditulis sebagai Persamaan 2.18.

$$
{x}_{c}=A {F}_{c}
$$

(2.18)

dengan ${x}_{c}$ vektor keluaran kanal ke-*c* ($G\times G$ nilai), ${F}_{c}$ vektor nilai *feature map* kanal ke-*c*, dan $A$ matriks koefisien interpolasi yang hanya bergantung pada proposal dan sama untuk setiap kanal.

Pada piramida fitur, tingkat yang dipakai untuk setiap proposal dipilih dengan Persamaan 2.19 (Lin et al., 2017).

$$
k=\left⌊{k}_{0}+{log}_{2}\left(\frac{\sqrt{wh}}{224}\right)\right⌋
$$

(2.19)

dengan $k$ tingkat terpilih, ${k}_{0}=4$ tingkat acuan untuk proposal 224×224 piksel, serta $w,h$ lebar dan tinggi proposal.

5. ***Box Head***

Keluaran *RoI Align* setiap proposal diratakan menjadi vektor, lalu diproses oleh *Box Head* yang berupa MLP dengan dua lapisan terhubung penuh yang menghasilkan representasi RoI (Girshick, 2015; Ren et al., 2017).

Representasi RoI diteruskan ke dua lapisan keluaran yang bekerja berdampingan (*two sibling output layers*) (Girshick, 2015; Ren et al., 2017). Cabang pertama adalah *classifier* yang menghasilkan skor untuk $C+1$ kelas (termasuk background) melalui *softmax* pada Persamaan 2.4 dan dilatih dengan *cross-entropy* pada Persamaan 2.2. Cabang kedua adalah *regressor* yang menghasilkan empat nilai koreksi koordinat untuk setiap kelas pada Persamaan 2.10 sampai 2.13 dan dilatih dengan kerugian L1 (Persamaan 2.15).

6. ***Soft*-NMS**

Kotak dan skor keluaran *box head* kemudian disaring dengan *Soft*-NMS, yang menurunkan skor kandidat yang tumpang-tindih secara bertahap, bukan menghapusnya seketika seperti NMS konvensional (Bodla et al., 2017), sehingga cacat yang berdekatan tidak ikut terbuang dengan menggunakan Persamaan 2.20.

$$
{s}_{i}=\left\{\begin{bmatrix}{s}_{i}, & IoU\left(M,{b}_{i}\right)<{N}_{t} \\ {s}_{i}\left(1-IoU\left(M,{b}_{i}\right)\right), & IoU\left(M,{b}_{i}\right)\ge {N}_{t}\end{bmatrix}\right)
$$

(2.20)

dengan ${s}_{i}$ dan ${b}_{i}$ skor dan kotak kandidat ke-*i*, $M$ kandidat berskor tertinggi, dan ${N}_{t}$ ambang IoU.

### 2.1.6 *Explainable Artificial Intelligence* (XAI)

*Explainable Artificial Intelligence* (XAI) adalah bidang yang mengembangkan cara agar keputusan model kecerdasan buatan dapat dipahami manusia (Barredo Arrieta et al., 2020). XAI dibutuhkan karena model *deep learning* bersifat *black-box* (Samek et al., 2019). Berdasarkan waktu penjelasan dibentuk, XAI terbagi atas dua pendekatan.

**1. Pendekatan *post-hoc***

Pendekatan yang menghasilkan penjelasan setelah model memprediksi tanpa mengubah struktur model, misalnya LIME, SHAP, dan Grad-CAM. Penjelasannya bersifat aproksimasi dan tidak dijamin mencerminkan keputusan internal model (Guidotti et al., 2018; Rudin, 2019).

**2. Pendekatan *ante-hoc***

Pendekatan yang menanamkan kemampuan menjelaskan ke dalam struktur model, misalnya pohon keputusan, model linear, dan arsitektur *neuro-symbolic* (Arrieta et al., 2020).

Penjelasan untuk citra umumnya berupa peta atribusi (*attribution map*), yaitu peta besar kontribusi setiap posisi terhadap skor kelas (Bach et al., 2015). *Attribution map* yang divisualisasikan dengan skala warna di atas citra disebut *heatmap*. Dua metode atribusi yang relevan dijelaskan berikut.

1. ***Gradient-weighted Class Activation Mapping* (Grad-CAM)**

Grad-CAM menghasilkan *attribution map* spesifik-kelas tanpa mengubah maupun melatih ulang model (Selvaraju et al., 2017). Bobot kepentingan *feature map* ke-*k* terhadap kelas *c* dan *attribution map* dinyatakan pada Persamaan 2.21 dan 2.22.

$$
{\alpha}_{k}^{c}=\frac{1}{Z}\sum_{i}{\sum_{j}{\frac{\partial {y}^{c}}{\partial {A}_{ij}^{k}}}}
$$

(2.21)

$$
{L}_{Grad-CAM}^{c}=ReLU\left(\sum_{k}{{\alpha}_{k}^{c}{A}^{k}}\right)
$$

(2.22)

dengan ${y}^{c}$ skor kelas *c* sebelum *softmax*, ${A}^{k}$ feature map ke-*k* pada lapisan konvolusi acuan, ${A}_{ij}^{k}$ nilainya pada posisi $\left(i \middle| j\right)$, dan $Z$ jumlah posisi spasial. Contoh hasil Grad-CAM ditunjukkan pada Gambar 2.9.

[Gambar 2.9]

**Gambar 2.9 Hasil penjelasan dari Grad-CAM (Selvaraju et al., 2017).**

2. ***Gradient × Input***

*Gradient × Input* menghitung kontribusi setiap elemen masukan sebagai hasil kali nilai elemen dan gradien keluaran terhadap elemen tersebut (Shrikumar et al., 2017; Ancona et al., 2019), sebagaimana dinyatakan pada Persamaan 2.23.

$$
{R}_{j}\left(x\right)={x}_{j}⋅\frac{\partial f\left(x\right)}{\partial {x}_{j}}
$$

(2.23)

dengan ${R}_{j}$ kontribusi elemen ke-*j*, ${x}_{j}$ nilai elemen ke-*j*, dan $f\left(x\right)$ keluaran model.

Metode atribusi yang baik diharapkan memenuhi sifat *completeness*, yaitu jumlah seluruh atribusi sama dengan selisih keluaran model pada masukan dan pada masukan acuan (*baseline*) $\overset{̅}{x}$ (Sundararajan et al., 2017), sebagaimana dinyatakan pada Persamaan 2.24.

$$
\sum_{j}{{R}_{j}\left(x\right)}=f\left(x\right)-f\left(\overset{̅}{x}\right)
$$

(2.24)

*Gradient × Input* umumnya tidak memenuhi sifat ini pada model non-linear, tetapi memenuhinya secara eksak pada model linear dengan *baseline* nol (Ancona et al., 2019), sebagaimana dinyatakan pada Persamaan 2.25.

$$
\sum_{j}{{w}_{j}{x}_{j}}={w}^{T}x=f\left(x\right)-f\left(0\right)
$$

(2.25)

Dengan demikian, *Gradient × Input* merupakan dekomposisi eksak, bukan aproksimasi.

### 2.1.7 *Neuro-Symbolic* AI (NeSy)

*Neuro-Symbolic* AI (NeSy) menggabungkan pembelajaran statistik jaringan saraf dengan penalaran berbasis aturan pada sistem simbolik (d'Avila Garcez & Lamb, 2023), untuk menyatukan kemampuan mengenali pola dari data mentah dengan transparansi penalaran (Kautz, 2022). Arsitektur NeSy ditunjukkan pada Gambar 2.10.

[Gambar 2.10]

**Gambar 2.10 Arsitektur Neuro-Symbolic**

Arsitektur NeSy terdiri atas dua komponen utama.

**1. Komponen *neural*,** berfungsi untuk mengubah data mentah menjadi representasi fitur numerik (d'Avila Garcez and Lamb, 2023).

**2. Komponen simbolik,** mengolah representasi fitur melalui aturan yang dapat ditelusuri. Penjelasannya *faithful*, yaitu merupakan proses keputusan model itu sendiri, bukan aproksimasi (Rudin, 2019). Komponen ini dapat dibentuk melalui *model mimicking*, yaitu melatih model sederhana untuk meniru keluaran model kompleks (*teacher*), dengan label pelatihan berupa prediksi *teacher*, bukan label sebenarnya (Buciluǎ et al., 2006).

Pengalihan keputusan dari komponen *neural* ke komponen simbolik membawa konsekuensi pada bentuk keluarannya. Keluaran komponen *neural* bersifat kontinu, sedangkan keluaran komponen simbolik bersifat diskrit karena hanya berupa label kelas. Proses diskritisasi ini memetakan banyak nilai kontinu yang berbeda ke label yang sama, sehingga informasi tingkat keyakinan turut hilang (Provost & Domingos, 2003). Informasi tersebut masih tersimpan pada nilai keputusan sebelum diubah menjadi label, dan dapat dinyatakan melalui dua konsep berikut.

1. *Margin*

Pada fungsi keputusan bernilai riil, tanda keluaran menyatakan label, sedangkan besarnya menyatakan keyakinan. Keluaran yang dekat dengan nol berarti keyakinan rendah, dan yang jauh dari nol berarti keyakinan tinggi (Schapire & Singer, 1999). Besar *margin* dinyatakan pada Persamaan 2.26.

$$
m\left(x\right)=\left|f\left(x\right)\right|
$$

(2.26)

dengan $f\left(x\right)$ fungsi keputusan linear.

2. Fungsi *Sigmoid*

Fungsi *sigmoid* adalah fungsi monoton naik yang memetakan nilai riil ke selang 0 dan 1 (Platt, 1999), sebagaimana dinyatakan pada Persamaan 2.27.

$$
\sigma \left(z\right)=\frac{1}{1+{e}^{-z}}
$$

(2.27)

dengan $z$ nilai masukan dan $e$ bilangan Euler.

### 2.1.8 *Sparse Oblique Decision Tree* (SODT)

*Sparse Oblique Decision Tree* (SODT) adalah pohon keputusan yang melakukan pemisahan linear multivariat (*oblique split*) pada setiap *node* internal, berbeda dengan pohon *axis-aligned* yang hanya memakai satu fitur per pemisahan (Hada et al., 2024). Setiap *node* internal meneruskan masukan ke salah satu dari dua anaknya, dan label pada *leaf*  yang dicapai menjadi prediksi pohon. Fungsi keputusan *node* internal ke-*i* dinyatakan pada Persamaan 2.28.

$$
{w}_{i}^{T}x+{b}_{i}\ge 0
$$

(2.28)

dengan $x$ vektor fitur berdimensi $D$, ${w}_{i}$ vektor bobot, dan ${b}_{i}$ bias *node* ke-*i*.

Arah percabangan ditentukan oleh tanda ${f}_{i}\left(x\right)$, dinotasikan ${d}_{i}=+1$ untuk ke anak kiri dan ${d}_{i}=-1$ untuk ke anak kanan. Rangkaian *node* dari *root* hingga *leaf* membentuk jalur keputusan (*decision path*) (Hada et al., 2024; Kairgeldin & Carreira-Perpiñán, 2025). Ilustrasinya ditunjukkan pada Gambar 2.11.

[Gambar 2.11]

**Gambar 2.11 Ilustrasi SODT dan perbedaan dengan *Decision  Tree* biasa**

Apabila $x$ berasal dari perataan peta fitur $C\times H\times W$ (Subbab 2.1.4), setiap elemen $x$ berkorespondensi satu-satu dengan satu kanal dan posisi spasial tertentu pada peta fitur. Korespondensi yang sama berlaku pada bobot ${w}_{i}$ karena dikalikan pada indeks yang sama (Persamaan 2.28), sehingga vektor ${w}_{i}$ dapat disusun ulang ke bentuk grid $C\times H\times W$, yang menyatakan kanal dan posisi tertentu yang dipakai *node* (Hada et al., 2024; Kairgeldin & Carreira-Perpiñán, 2025).

1. **Regularisasi L1 dan Sparsitas**

Sifat *sparse* diperoleh melalui regularisasi L1 pada fungsi tujuan pelatihan pada Persamaan 2.29, yang membuat bobot fitur tidak relevan bernilai tepat nol (Hada et al., 2024).

$$
E\left(\Theta \right)=\sum_{n=1}^{N}{L}\left({y}_{n},T\left({x}_{n};\Theta \right)\right)+\lambda \sum_{i\in D}{{\left‖{w}_{i}\right‖}_{1}}
$$

(2.29)

dengan $\Theta$ parameter pohon, $N$ jumlah sampel, ${x}_{n},{y}_{n}$ fitur dan label sampel ke-*n*, $T\left({x}_{n} \middle| \Theta \right)$ prediksi pohon, $L$ fungsi kerugian klasifikasi, $D$ himpunan *node* internal, dan $\lambda$ pengontrol sparsitas. *Node* yang seluruh bobotnya nol hanya ditentukan oleh bias, sehingga selalu mengarahkan masukan ke sisi yang sama.

2. ***Tree Alternating Optimization (TAO)***

Persamaan 2.29 tidak dapat dioptimasi dengan metode berbasis gradien karena keputusan pohon diskrit, dan tidak didukung oleh metode pembentukan pohon konvensional. *Tree Alternating Optimization* (TAO) memecahnya menjadi masalah klasifikasi biner yang diselesaikan terpisah untuk setiap *node* (Carreira-Perpiñán & Tavallali, 2018), sebagaimana dinyatakan pada Persamaan 2.30.

$$
{E}_{i}({w}_{i},{b}_{i})=\sum_{n\in {R}_{i}}{{L}^{‾}}({\overset{‾}{y}}_{n},{g}_{i}({x}_{n};{w}_{i},{b}_{i}))+\lambda ∥{w}_{i}{∥}_{1}
$$

(2.30)

dengan ${R}_{i}$ sampel yang mencapai *node* ke-*i* (*reduced set*), ${\overset{̅}{y}}_{n}$ label semu (*pseudo-label*) arah kiri atau kanan, ${g}_{i}$ keputusan biner *node*, dan $\overset{̅}{L}$ kerugian 0/1.

Tidak semua sampel dalam ${R}_{i}$ berpengaruh, sampel yang prediksinya sama pada kedua arah diabaikan, sedangkan sisanya (*care set*) memperoleh label semu berupa arah yang menghasilkan prediksi benar. Masalah biner tersebut diselesaikan dengan regresi logistik berregularisasi L1 sebagai pengganti (*surrogate*) kerugian 0/1, dan setiap *leaf* berlabel kelas mayoritas sampel yang mencapainya. Penyelesaian ini dilakukan bergantian antar-*node* dari yang terdalam menuju *root*, dan karena tidak pernah menaikkan nilai fungsi tujuan, TAO dijamin konvergen (Carreira-Perpiñán & Tavallali, 2018; Hada et al., 2024).

Karena penalti $\lambda$ sama untuk semua *node*, *node* dengan banyak data cenderung kurang *sparse*. Kairgeldin dan Carreira-Perpiñán (2025) mengatasinya dengan membobot penalti berdasarkan jumlah sampel *node* melalui parameter $\alpha$, sebagaimana dinyatakan pada Persamaan 2.31 dan 2.32.

$$
E(\Theta )=\sum_{n=1}^{N}{L(}{y}_{n},T({x}_{n};\Theta ))+\lambda \sum_{i\in D}{{h}_{\alpha}}(∣{R}_{i}∣)∥{w}_{i}{∥}_{1}
$$

(2.31)

$$
{h}_{\alpha}(t)=\left\{\begin{bmatrix}1, & t=0 \\ {t}^{\alpha}, & t>0\end{bmatrix}\right)
$$

(2.32)

dengan $\left∣{R}_{i}\right∣$ jumlah sampel yang mencapai *node* ke-*i*. Nilai $\alpha >0$ membuat eliminasi fitur lebih agresif pada *node* yang menangani banyak data.

3. ***Class Weighting***

Pada *model mimicking* (Subbab 2.1.7), label pelatihan SODT adalah prediksi *teacher*. Apabila sebagian kelas jauh lebih jarang, pohon cenderung mengabaikannya. *Cost-sensitive learning* mengatasinya dengan memberi biaya kesalahan yang berbeda antarkelas (Elkan, 2001; He & Garcia, 2009), sebagaimana dinyatakan pada Persamaan 2.33.

$$
{E}_{\omega}\left(\Theta \right)=\sum_{n=1}^{N}{{\omega}_{{y}_{n}}L\left({y}_{n},T\left({x}_{n};\Theta \right)\right)}
$$

(2.33)

dengan ${\omega}_{{y}_{n}}\ge 0$ biaya kesalahan kelas ${y}_{n}$.

4. ***Explanation* melalui *Weight Node***

Setiap *node* bersifat linear dan *sparse*, sehingga bobot ${w}_{i}$ langsung menunjukkan fitur yang dipakai *node*. Bobot nol berarti fitur tidak dipakai, tandanya menyatakan arah dorongan, dan besarnya menyatakan kekuatan pengaruh. Hada et al. (2024) memvisualisasikan bobot ini untuk menelusuri fitur yang memisahkan antarkelas.

Kairgeldin dan Carreira-Perpiñán (2025) memperluasnya pada model hibrida CNN dan SODT. Karena setiap fitur berasal dari sel *feature map* CNN yang memiliki *Receptive Field* (Subbab 2.1.4), mereka menyusun peta kepadatan *Receptive Field* (RF *density map*) untuk setiap *node*, sebagaimana dinyatakan pada Persamaan 2.34.

$$
{D}_{i}\left(u\right)=\sum_{j=1}^{D}{\left|{w}_{ij}\right|⋅1\left[u\in {RF}_{j}\right]}
$$

(2.34)

dengan ${D}_{i}\left(u\right)$ kepadatan pada posisi citra $u$ untuk *node* ke-*i*, ${w}_{ij}$ bobot fitur ke-*j*, ${RF}_{j}$ *Receptive Field* fitur ke-*j*, dan $1\left[⋅\right]$ fungsi indikator. Kepadatan nol berarti daerah tersebut tidak dipakai *node* (Kairgeldin & Carreira-Perpiñán, 2025). Peta ini disusun per *node* dan hanya bergantung pada bobot, sehingga bersifat statis.

### 2.1.9 Metrik Evaluasi

Evaluasi dibagi menjadi evaluasi kinerja deteksi objek dan evaluasi kualitas penjelasan (*explainability*).

1. **Metrik Evaluasi Deteksi Objek**

*Intersection over Union* (IoU) mengukur rasio luas irisan terhadap luas gabungan *bounding box* prediksi dan *ground truth* pada Persamaan 2.35. Berdasarkan ambang IoU, prediksi dikelompokkan menjadi *True Positive* (TP), *False Positive* (FP), dan *False Negative* (FN), yang menjadi dasar *Precision* dan *Recall* yang didefiniskan pada Persamaan 2.36 dan 2.37.

$$
IoU=\frac{Area Prediksi∩Area Ground Truth}{Area Prediksi∪Area Ground Truth}
$$

(2.35)

$$
Precision=\frac{TP}{TP+FP}
$$

(2.36)

$$
Recall=\frac{TP}{TP+FN}
$$

(2.37)

*Precision* mengukur ketepatan prediksi positif, sedangkan *Recall* mengukur kelengkapan deteksi. F1-*score* adalah rata-rata harmonik *Precision* dan *Recall* (Persamaan 2.38) (Sokolova & Lapalme, 2009).

$$
F1=\frac{2⋅Precision⋅Recall}{Precision+Recall}
$$

(2.38)

*Average Precision* (AP) adalah luas di bawah kurva *Precision-Recall*pada Persamaan 2.39, dan *mean Average Precision* (mAP) adalah rata-rata AP seluruh kelas pada Persamaan 2.40 (Everingham et al., 2010).

$$
AP=\int_{0}^{1}{Precision\left(Recall\right) d\left(Recall\right)}
$$

(2.39)

$$
mAP=\frac{1}{K}\sum_{i=1}^{K}{{AP}_{i}}
$$

(2.40)

dengan $K$ jumlah kelas dan ${AP}_{i}$ AP kelas ke-*i*. mAP@0,5 dihitung pada ambang IoU 0,5, sedangkan mAP@0,5:0,95 merupakan rata-rata mAP pada ambang IoU 0,5 hingga 0,95 dengan kenaikan 0,05 (Everingham et al., 2010; Lin et al., 2014).

2. **Metrik Evaluasi *Explainability***

Evaluasi XAI dapat melibatkan pengguna (*application-grounded* dan *human-grounded*) atau tanpa pengguna melalui ukuran kuantitatif (*functionally-grounded*) (Nauta et al., 2023). Penelitian ini berfokus pada evaluasi *functionally-grounded*, khususnya *faithfulness* dan *localization*.

a. *Faithfulness*

*Faithfulness* umumnya diuji melalui perturbasi, yaitu menghapus (*masking*) atau mempertahankan bagian masukan yang disorot penjelasan lalu mengamati perubahan keluaran model (Petsiuk et al., 2018).

*Necessity* menguji apakah fitur yang disorot memang diperlukan, antara lain fitur dihapus, lalu diperiksa apakah prediksi berubah sebagaimana didefinisikan pada Persamaan 2.41. *Sufficiency* menguji apakah fitur yang disorot sudah cukup: hanya fitur tersebut yang dipertahankan, lalu diperiksa apakah prediksi tetap sehingga dapat definisikan dengan Persamaan 2.42.

$$
Necessity Flip Rate=\frac{1}{N}\sum_{n=1}^{N}{1\left[\overset{ˆ}{y}\left({x}_{n}^{-S}\right)\ne \overset{ˆ}{y}\left({x}_{n}\right)\right]}
$$

(2.41)

$$
Sufficiency Preservation=\frac{1}{N}\sum_{n=1}^{N}{1\left[\overset{ˆ}{y}\left({x}_{n}^{S}\right)=\overset{ˆ}{y}\left({x}_{n}\right)\right]}
$$

(2.42)

dengan $S$ himpunan fitur atau daerah yang disorot paling penting oleh penjelasan, ${x}_{n}^{-S}$ masukan tanpa $S$ (dinolkan), ${x}_{n}^{S}$ masukan yang hanya mempertahankan $S$, $\overset{ˆ}{y}$ keluaran model, dan $1\left[⋅\right]$ fungsi indikator. Nilai yang tinggi pada kedua metrik menandakan penjelasan yang *faithful*.

*Faithfulness* juga dapat diukur secara bertahap dengan *Deletion* AUC dan *Insertion* AUC (Petsiuk et al., 2018), yaitu luas di bawah kurva skor seiring fitur dihapus atau ditambahkan dalam $K$ tahap seperti pada Persamaan 2.43.

$$
AUC=\sum_{k=1}^{K-1}{\frac{{s}_{k}+{s}_{k+1}}{2}⋅\Delta {x}_{k}}
$$

(2.43)

dengan ${s}_{k}$ skor ternormalisasi pada tahap ke-*k* dan $\Delta {x}_{k}$ proporsi area yang dimodifikasi antartahap.

Pada *Deletion*, fitur terpenting dihapus lebih dulu, sehingga AUC rendah menandakan *necessity* yang baik. Pada *Insertion*, fitur terpenting ditambahkan lebih dulu, sehingga AUC tinggi menandakan *sufficiency* yang baik.

b. *Localization*

*Localization* mengukur kesesuaian area yang disorot *heatmap* dengan lokasi objek. *Pointing Game* menghitung proporsi kasus ketika titik tertinggi *heatmap* jatuh di dalam *bounding box ground truth* seperti pada Persamaan 2.44 (Zhang et al., 2018), sedangkan IoU *Heatmap* mengukur IoU antara *heatmap* terbinerisasi dan *bounding box ground truth* yang didefinisikan pada Persamaan 2.45.

$$
{Acc}_{PG}=\frac{Hits}{Hits+Misses}
$$

(2.44)

$$
{Acc}_{PG}=\frac{Hits}{Hits+Misses}
$$

(2.44)

$$
{IoU}_{Heatmap}=\frac{\left|{H}_{bin}∩{B}_{gt}\right|}{\left|{H}_{bin}∪{B}_{gt}\right|}
$$

(2.45)

dengan $Hits$ dan $Misses$ jumlah kasus yang titik maksimumnya berada di dalam dan di luar *ground truth*, ${H}_{bin}$ *heatmap* yang hanya mempertahankan sejumlah posisi bernilai tertinggi, dan ${B}_{gt}$ daerah *ground truth*.

## 2.2 Penelitian Terkait

**Tabel 2.2 Ringkasan dan Perbandingan Penelitian Terkait**

| No. | Judul / Penulis | Masalah | Tujuan | Metode | Hasil | Keterkaitan |
| --- | --- | --- | --- | --- | --- | --- |
| 1. | *Improving PCB defect detection using selective feature attention and pixel shuffle pyramid (Fung et al., 2024)* | Deteksi cacat mikroskopis pada sirkuit PCB memiliki tingkat *false negative* yang tinggi pada model deteksi standar. | Meningkatkan kemampuan model melokalisasi target berukuran kecil pada citra PCB. | Faster R-CNN dengan *Feature Pyramid Network* (FPN), *Pixel Shuffle Pyramid* (PSPyramid), *Selective Feature Attention,* dan *Soft-NMS*. | Terjadi peningkatan *Mean Average Precision* (mAP) yang signifikan pada pengujian dataset DeepPCB. | Penelitian ini menjadi referensi utama arsitektur dasar (*baseline*) komponen Neuro (ekstraktor fitur dan lokalisasi) yang digunakan dalam tugas akhir ini. |
| 2. | *Faster-LTN: a neuro-symbolic, end-to-end object detection architecture (Manigrasso et al., 2023)* | Model *deep learning* konvensional tidak mampu mengintegrasikan pengetahuan relasional dan penalaran logis ke dalam proses deteksi objek, sehingga kurang transparan dalam pengambilan keputusan. | Mengintegrasikan kemampuan penalaran logis dengan jaringan saraf konvolusional ke dalam arsitektur deteksi objek *end-to-end* untuk meningkatkan transparansi. | Faster R-CNN dengan penggantian kepala klasifikasi menjadi *Logic Tensor Networks* (LTN). | Arsitektur *end-to-end* berhasil dilatih dan mencapai performa kompetitif pada dataset PASCAL VOC. | Memberikan landasan konseptual integrasi pendekatan Neuro-Symbolic ke dalam arsitektur deteksi objek dua tahap (Faster R-CNN). |
| 3. | *Sparse oblique decision trees: a tool to understand and manipulate neural net features (Hada et al., 2024)* | Lapisan MLP pada jaringan *deep learning* tidak dapat dijelaskan secara komputasional (*black-box*). | Menciptakan proyektor linear pengganti lapisan MLP tanpa menurunkan akurasi. | *Sparse Oblique Decision Tree* (SODT) dengan penerapan regularisasi L1. | Menghasilkan struktur pohon yang sangat ramping dengan performa setara MLP. | Menjadi landasan teoritis komponen **Symbolic** untuk menggantikan MLP pada *RoI head* jaringan Faster R-CNN. |
| 4. | *Neurosymbolic models based on hybrids of convolutional neural networks and decision trees (Kairgeldin & Carreira-Perpiñán, 2025)* | Melatih struktur pohon secara global adalah masalah *NP-Hard*. | Mengoptimalkan bobot pohon secara efisien untuk data representasi dimensi tinggi. | *Algoritma iteratif Tree Alternating Optimization (TAO).* | Pohon mencapai konvergensi lebih stabil dengan efisiensi komputasi tinggi. | Merupakan algoritma komputasi utama yang digunakan untuk proses *fitting* model SODT di penelitian ini. |
| 5. | *Explainable Predictive Quality Inspection using Deep Learning in Electronics Manufacturing* (Saadallah et al., 2022) | Model deep learning untuk prediksi kualitas bersifat *black-box* sehingga menyulitkan teknisi memahami fitur mana yang paling berpengaruh terhadap keputusan prediksi. | Menyediakan penjelasan visual atas prediksi kualitas PCB menggunakan *heatmap* untuk membantu teknisi mengidentifikasi fitur global (kuantitas fisik SPI) dan lokal (pin) yang paling menentukan. | *1D-CNN untuk prediksi kualitas biner (OK/NOK) dan Grad-CAM untuk menghasilkan peta panas penjelasan.* | Grad-CAM berhasil menyoroti fitur SPI (DX, DY, DVolume) dan pin spesifik yang paling diskriminatif untuk kelas "NOK", membantu teknisi melacak penyebab deviasi kualitas. | Menunjukkan aplikasi Grad-CAM sebagai metode *post-hoc* untuk inspeksi kualitas PCB, |
| 6. | *Explainable AI Methods for Identification of Glue Volume Deficiencies in Printed Circuit Boards* (Tziolas et al., 2025) | Inspeksi volume lem pada PCB sulit dilakukan secara manual dan model *deep learning* yang digunakan tidak memberikan penjelasan atas deteksi defisiensi. | Mengidentifikasi defisiensi volume lem pada PCB menggunakan model *deep learning* dan menyediakan penjelasan visual atas prediksi model. | *CNN (ResNet-50 dan Vision Transformer) untuk klasifikasi defisiensi lem, dengan Grad-CAM dan Deep SHAP untuk menghasilkan peta panas penjelasan.* | CNN mencapai akurasi tinggi dalam mendeteksi defisiensi lem, dan Grad-CAM/Deep SHAP berhasil menyoroti area dengan volume lem tidak memadai yang menjadi dasar keputusan model. | Memperkuat justifikasi penggunaan Grad-CAM sebagai metode *post-hoc* yang telah teruji dalam inspeksi visual PCB, serta menunjukkan keterbatasan *post-hoc* yang mendorong kebutuhan pendekatan *faithful*. |
| 7. | *Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging (Arun et al., 2021)* | Peta *saliency* banyak dipakai untuk menjelaskan dan melokalisasi keputusan CNN pada domain berisiko tinggi, tetapi keandalannya belum teruji secara sistematis. | Mengevaluasi keandalan (*trustworthiness*) peta *saliency* untuk lokalisasi kelainan pada citra medis. | Delapan metode *saliency*, termasuk Grad-CAM, diuji pada dua *dataset* radiologi berdasarkan utilitas lokalisasi, sensitivitas terhadap pengacakan bobot model, *repeatability*, dan *reproducibility*, lalu dibandingkan dengan jaringan lokalisasi (U-Net dan RetinaNet). | *Seluruh metode gagal pada minimal satu kriteria dan kalah dari jaringan lokalisasi. Grad-CAM lolos uji pengacakan bobot, tetapi AUPRC lokalisasinya (0,41) tetap di bawah RetinaNet (0,596).* | Menunjukkan bahwa penjelasan *post-hoc*, termasuk Grad-CAM, belum dapat diandalkan pada domain berisiko tinggi, serta menjadi dasar kontrol pengacakan bobot yang juga digunakan dalam penelitian ini. |

# BAB III METODOLOGI PENELITIAN

Penelitian ini membangun, mengintegrasikan, dan mengevaluasi sistem deteksi cacat PCB berbasis arsitektur *neuro-symbolic*. Alur penelitian ditunjukkan pada Gambar 3.1.

[Gambar 3.1]

**Gambar 3.1 Diagram Alur Penelitian**

Penelitian dimulai dengan persiapan dan pra-pemrosesan *dataset* DeepPCB, dilanjutkan pelatihan *Faster* R-CNN, ekstraksi *dataset* simbolik, pelatihan SODT, integrasi keduanya, lalu evaluasi akhir.

## 3.1 Persiapan *Dataset*

Dataset DeepPCB (Tang et al., 2019) dibagi menjadi data latih dan data uji berdasarkan berkas indeks bawaannya yang ditunjukkan pada Tabel 3.1, sehingga tidak ada kebocoran data antarfase.

**Tabel 3.1 Pembagian dataset berdasarkan kategori**

| Kategori | Berkas Indeks | Jumlah Citra | Persentase (%) |
| --- | --- | --- | --- |
| Data Pelatihan (Training) | *trainval.txt* | 1.000 | 66,67% |
| Data Pengujian (Testing) | *test.txt* | 500 | 33,33% |

Penelitian ini bersifat non-referensial, yaitu model mendeteksi cacat tanpa membandingkan citra uji dengan citra *template*, sesuai kondisi inspeksi nyata ketika citra referensi sering tidak tersedia. Oleh karena itu, dari setiap pasangan citra hanya citra target (*\_test.jpg*) yang dimuat, sedangkan citra templat (*\_temp.jpg*) diabaikan.

Koordinat cacat pada berkas anotasi diubah menjadi *bounding box*, sedangkan ID kelas 1–6 dipakai langsung sebagai indeks kelas, yaitu *open*, *short*, *mousebite*, *spur*, *spurious copper*, dan *pinhole*. Contoh sampel beserta anotasinya ditunjukkan pada Gambar 3.2.

[Gambar 3.2]

**Gambar 3.2 Contoh sampel dataset PCB beserta anotasinya**

## 3.2 Pra-pemrosesan *Dataset*

Citra diproses melalui dua *pipeline*. *Pipeline* di luar model mengubah citra menjadi *tensor* dan menerapkan augmentasi saat pelatihan. *Pipeline* di dalam model menormalisasi citra dan mengatur resolusi, yaitu dengan acak multi-resolusi saat latih, dan tetap saat inferensi maupun ekstraksi fitur agar fitur bagi komponen simbolik stabil. Konfigurasinya dirangkum pada Tabel 3.2 dan Tabel 3.3.

**Tabel 3.2 Parameter Normalisasi dan Standardisasi Input**

| Parameter | Nilai Konfigurasi | Tujuan |
| --- | --- | --- |
| Rentang Intensitas | \[0.0, 1.0] | Penyeragaman *dynamic range* piksel |
| Rata-rata (*Mean*) | \[0.485, 0.456, 0.406] | Penyelarasan distribusi ImageNet |
| Deviasi Standar (*Std*) | \[0.229, 0.224, 0.225] | Penyelarasan distribusi ImageNet |

**Tabel 3.3 Konfigurasi Augmentasi Data Pelatihan**

| Jenis Augmentasi | Parameter | Dekripsi |
| --- | --- | --- |
| *Random Horizontal Flip* | Probabilitas: 0,5 | Variasi Arah |
| *Multi-resolution Scaling* | Sisi terpendek: {480, 560, 640, 720, 800, 880}; sisi terpanjang maks. 880 | Ketahanan terhadap skala |
| *Coordinate Sync* | Enabled | Penyesuaian otomatis lokasi *box* saat gambar berubah |

Secara berurutan, citra diubah menjadi *tensor* berintensitas \[0,0; 1,0], dibalik horizontal secara acak pada fase latih beserta koordinat *bounding box*-nya, lalu di dalam model dinormalisasi per kanal sesuai Tabel 3.2, diskalakan sesuai fase pada Tabel 3.3, dan disamakan ukurannya dalam satu *batch* melalui *padding*.

## 3.3 Model Neuro (*Faster* R-CNN)

*Faster* R-CNN berperan sebagai ekstraktor fitur, pengusul area, dan model *teacher*. Arsitekturnya mengikuti SF-PSPyramid dari Fung et al. (2024) yang dijelaskan pada Subbab 2.1.5, dengan satu perbedaan, yaitu jumlah kanal *neck* 64, bukan 256, agar dimensi masukan SODT tidak terlalu besar sehingga hanya menjadi ($64\times 7\times 7=3.136$).

Alur modelnya mengikuti Subbab 2.1.5. *Backbone* ResNet-50 mengekstrak *feature map* C2–C5, lalu *neck* membentuk piramida P2′–P6 melalui *pixel shuffle* pada Persamaan 2.8 dan SF *Attention* pada Persamaan 2.9. RPN menghasilkan proposal dari *anchor* sesuai Persamaan 2.10 sampai 2.13, lalu *RoI Align* mengubah setiap proposal menjadi *tensor* 64×7×7 pada tingkat piramida terpilih dengan Persamaan 2.17 dan 2.19. Selanjutnya, *box head* menghasilkan skor kelas dan koordinat *bounding box*, dan Soft-NMS menyaring deteksi yang tumpang-tindih dengan Persamaan 2.20. Rincian konfigurasi tiap modul dirangkum pada Tabel 3.4.

**Tabel 3.4 Parameter utama Faster R-CNN**

| Modul Fungsional | Parameter Utama | Nilai / Konfigurasi |
| --- | --- | --- |
| *Backbone & Neck* | Inisialisasi Bobot | *ResNet-50, Pre-trained* ImageNet |
|  | Batch Normalization | *Frozen* |
| *Neck* (SF-PSPyramid) | Feature map FPN | P2, P3, P4, P5, P6 |
|  | Jumlah Kanal (C) | 64 |
| *RPN* | Ukuran (*Anchor*) | \[16, 32, 64, 128, 256] piksel |
|  | Rasio Aspek | \[0.5, 1.0, 2.0] |
|  | Ambang Batas IoU | *Foreground*: 0.7<br>*Background*: 0.3 |
|  | Batas Proposal (NMS) | *Train: 2000 \| Test: 1000* |
| *RoI Align* | Dimensi *RoI Align* | *7x7 piksel (Rasio sampling: 2)* |
| *Soft-NMS* | Metode Penurunan (*Decay*) | *Linier (Linear)* |
|  | Parameter Sigma & IoU | *Sigma: 0.5*<br>*IoU Threshold: 0.5* |
|  | Pemotongan Skor (*Score Thresh*) | *0.001* |

## 3.4 Pelatihan dan Evaluasi Model Neuro

*Faster* R-CNN dilatih selama 15 *epoch* (Fung et al. (2024) memakai 12) dengan *Automatic Mixed Precision* (AMP) dan *gradient accumulation* untuk mengatasi keterbatasan memori GPU. Kerugian regresi memakai L1 murni pada Persamaan 2.15 mengikuti Fung et al. (2024) karena memberi penalti lebih tegas pada *cacat* kecil. Laju pembelajaran dinaikkan secara linear pada awal pelatihan (*warmup*), lalu diturunkan bertahap. *Hyperparameter* dirangkum pada Tabel 3.5.

**Tabel 3.5 Hyperparameter pelatihan model Faster R-CNN**

| *Hyperparameter* | Parameter Utama | Nilai / Konfigurasi |
| --- | --- | --- |
| Siklus Pelatihan | Total *Epoch* | 15 |
|  | Gradient Accumulation | 4 langkah (*Effective Batch*: 4) |
|  | Presisi Komputasi | *AMP: True*) |
| Fungsi Pengoptimal | Optimizer | SGD |
|  | Laju Pembelajaran (*Learning Rate*) | 0,02 |
|  | Momentum & *Weight Decay* | Momentum: 0.9 \| *Decay*: 0.0001 |
| Penjadwalan (Scheduler) | Tipe *Scheduler* | Milestone / Step Decay |
|  | *Milestones* | *Epoch* ke-8 dan ke-11 |
|  | Faktor Penurunan (*Gamma*) | 0.1 (Penurunan 10%) |
| Pemanasan (Warmup) | Iterasi *Warmup* | 500 batch |
|  | Rasio *Warmup* Awal | 0.001 |

Pada setiap *batch*, *multi-task loss* pada Persamaan 2.14 dengan kerugian L1 pada Persamaan 2.15 dihitung, lalu gradiennya diakumulasikan selama empat iterasi sebelum SGD memperbarui bobot sesuai Persamaan 2.3. Setelah pelatihan, model dievaluasi pada data uji dengan mAP@0,5 dan mAP@0,5:0,95, serta *Precision*, *Recall*, dan F1 pada IoU 0,5 dan skor 0,5 sesuai Persamaan 2.35 sampai 2.40, termasuk per kelas dan *confusion matrix*. Bobot model kemudian dibekukan dan disimpan sebagai model *teacher*.

## 3.5 Ekstraksi Fitur RoI dan Hasil Klasifikasi *Faster* R-CNN

Tahap ini membentuk dataset simbolik sesuai skema *model mimicking* di Subbab 2.1.7. Fitur diambil tepat setelah *RoI Align*, sebelum masuk MLP pada *box head*, sehingga bentuk grid 7×7 tetap terjaga. Yang diekstrak adalah seluruh proposal RPN pada mode inferensi, bukan hanya deteksi akhir, agar SODT dilatih pada populasi proposal yang sama dengan saat inferensi.

Label setiap proposal adalah prediksi *teacher*, termasuk *background*, bukan *ground truth*. *Ground truth* hanya disimpan sebagai data pendamping untuk metrik lokalisasi. Ekstraksi dilakukan terpisah untuk data latih dan data uji. Mekanismenya diilustrasikan pada Gambar 3.3.

[Gambar 3.3]

**Gambar 3.3 Pengambilan RoI Align dan Hasil dari Faster R-CNN**

Berdasarkan mekanisme tersebut, pembentukan *dataset* simbolik dirangkum pada Algoritma 3.1.

**Algoritma 3.1 Pembentukan *Dataset* Simbolik**

**Input:** Model *teacher* (*frozen*) serta citra latih dan uji beserta anotasinya.

**Output:** *Dataset* simbolik latih dan uji berisi fitur $64\times 7\times 7$, koordinat proposal, label *teacher*, dan data pendamping *ground truth*.

**Langkah-langkah:**

1. Setiap citra diproses dengan resolusi tetap sesuai Tabel 3.3, lalu RPN menghasilkan hingga 1.000 proposal.
2. *RoI Align* mengubah setiap proposal menjadi *tensor* 64×7×7 dengan Persamaan 2.17 dan 2.19.
3. *Tensor* diteruskan ke *box head teacher*, dan kelas dengan *softmax* tertinggi pada Persamaan 2.4 diambil sebagai label.
4. Setiap proposal dicocokkan dengan *ground truth* ber-IoU tertinggi sebagai data pendamping.
5. Fitur, label, dan data pendamping disimpan sebagai *dataset* simbolik.

## 3.6 Model Simbolik (SODT)

SODT menggantikan *classifier* pada *box head*, sehingga setiap keputusan kelas berasal dari rangkaian keputusan linear yang dapat ditelusuri seperti dijelaskan pada Subbab 2.1.8. Masukannya adalah *tensor* $64\times 7\times 7$ yang diratakan dengan urutan kanal, baris, lalu kolom menjadi vektor berdimensi 3.136. Urutan ini dicatat agar setiap bobot dapat dikembalikan ke kanal dan posisi grid asalnya.

SODT dibentuk sebagai pohon biner lengkap dengan fungsi keputusan linear pada setiap *node* internal sesuai Persamaan 2.28 dan satu label kelas pada setiap *leaf*. Bobot dan bias setiap *node* diinisialisasi dari distribusi normal baku, sedangkan label *leaf* diinisialisasi secara acak, bukan dengan kelas mayoritas, agar TAO tidak terjebak pada kelas *background* yang dominan (Hada et al., 2024; Kairgeldin & Carreira-Perpiñán, 2025). Konfigurasinya dirangkum pada Tabel 3.6.

**Tabel 3.6 Konfigurasi SODT**

| Parameter | Nilai | Keterangan |
| --- | --- | --- |
| Kedalaman pohon | 6 | Batas maksimum tingkat hierarki *node* |
| Inisialisasi bobot dan bias | Distribusi normal baku N(0, 1) | Titik awal optimasi TAO |
| Inisialisasi label *leaf* | Acak | Mencegah TAO terjebak pada kelas *background* yang dominan |

## 3.7 Pelatihan dan Evaluasi Model Simbolik

SODT dilatih dengan TAO pada *dataset* simbolik latih untuk meminimalkan fungsi tujuan yang terdiri atas kerugian berbobot kelas pada Persamaan 2.33 dan penalti L1 berbobot ukuran *reduced set* pada Persamaan 2.31 dan 2.32. Karena sebagian besar proposal berlabel *background*, distribusi kelasnya sangat timpang sehingga diperlukan *negative sampling*.

Pembobotan kelas diterapkan di dua tempat. Pada masalah tereduksi setiap *node* sesuai Persamaan 2.30, setiap sampel diberi bobot pada Persamaan 3.1. Pada *leaf*, label ditentukan dengan mayoritas berbobot pada Persamaan 3.2.

$$
{u}_{n}=\left|{l}_{L}\left(n\right)-{l}_{R}\left(n\right)\right|⋅{\omega}_{{y}_{n}}
$$

(3.1)

$$
{\overset{ˆ}{y}}_{l}=\operatorname{arg}{\operatorname{{max}_{c}}{{\omega}_{c}⋅{N}_{l,c}}}
$$

(3.2)

dengan ${u}_{n}$ bobot sampel ke-*n*, ${l}_{L}\left(n\right) ,{l}_{R}\left(n\right)$ kerugian 0/1 sampel ke-*n* bila diarahkan ke kiri dan ke kanan, ${\omega}_{c}$ bobot kelas $c$ pada Persamaan 2.33, ${\overset{ˆ}{y}}_{l}$ label *leaf* ke-$l$, dan ${N}_{l,c}$ jumlah sampel berlabel $c$ pada *leaf* tersebut. Faktor $\left|{l}_{L}-{l}_{R}\right|$ bernilai 1 hanya untuk *care set*, sehingga kesalahan pengarahan pada kelas berbobot besar menjadi lebih mahal. Bobot *background* dibiarkan netral agar *false positive* tidak meningkat.

Setelah TAO, pohon dipangkas tanpa mengubah keputusannya, sehingga penjelasan yang dihasilkan menjadi lebih ringkas tanpa mengorbankan fidelitas. *Hyperparameter* pelatihan dirangkum pada Tabel 3.7.

**Tabel 3.7 *Hyperparameter* Pelatihan SODT**

| Hyperparameter | Nilai | Keterangan |
| --- | --- | --- |
| Lambda (*λ*) | 20 | Kekuatan penalti L1 pada Persamaan 2.31 |
| Alpha (*α*) | 0,15 | Eksponen pembobot penalti pada Persamaan 2.32 |
| Bobot Kelas (*ω*) | *Short*: 2<br>*Spur*: 1,5<br>*Open*: 1,5<br>*Pinhole*: 1,25<br>*Spurious copper*: 1,25<br>*Mousebite* dan *background*: 1 | Biaya kesalahan kelas pada Persamaan 2.33, lebih besar untuk kelas dengan *recall* terlemah |
| Rasio Negatif | 2 | Sampel *background* terhadap sampel cacat |
| Iterasi TAO Maksimum | 15 | Batas jumlah siklus pembaruan seluruh *node* |
| Toleransi Konvergensi | 10⁻⁶ | Pelatihan berhenti jika penurunan relatif fungsi tujuan di bawah nilai ini |

Fidelitas SODT diukur terhadap label *teacher* pada *dataset* simbolik uji dengan akurasi (*mimic accuracy*) dan *macro*-F1, serta kesesuaian per kelas, disertai jumlah *node* aktif dan bobot bukan nol. Seluruh tahapan pelatihan, mulai dari *negative sampling* hingga pemangkasan, dirangkum pada Algoritma 3.2.

**Algoritma 3.2 Pelatihan SODT dengan TAO**

**Input:** *Dataset* simbolik latih dan uji, SODT terinisialisasi sesuai Tabel 3.6, dan *hyperparameter* pelatihan pada Tabel 3.7.

**Output:** SODT terlatih dan hasil evaluasi fidelitasnya.

**Langkah-langkah:**

1. Seluruh proposal berlabel cacat dipertahankan, sedangkan proposal *background* diambil secara acak sesuai rasio negatif.
2. Ulangi hingga batas iterasi TAO tercapai atau penurunan relatif fungsi tujuan berada di bawah toleransi konvergensi:
   1. Setiap *node* internal, dari yang terdalam menuju *root*, diperbarui dengan menyelesaikan masalah tereduksinya pada Persamaan 2.30 menggunakan bobot sampel Persamaan 3.1.
   2. Label setiap *leaf* diperbarui dengan Persamaan 3.2.
3. *Node* yang tidak dilalui sampel (*dead branch*) dan *node* yang seluruh *leaf* di bawahnya berlabel sama (*pure subtree*) dinolkan bersama sub-pohonnya, sedangkan *leaf* yang tidak dicapai sampel diberi label *background*.
4. Fidelitas diukur pada *dataset* simbolik uji.

## 3.8 Integrasi *Faster* R-CNN dan SODT

Tahap ini menyatukan *Faster* R-CNN dan SODT menjadi satu alur inferensi, ditambah dua komponen, yaitu skor deteksi berbasis *routing margin* dan *heatmap* per *node*.

1. ***Hybrid Inference***

SODT hanya menggantikan kepala klasifikasi. Koordinat *bounding box* akhir tetap dihitung oleh kepala regresi *Faster* R-CNN, sehingga sifat *faithful* berlaku pada keputusan kelas, bukan pada penyesuaian lokasi kotak. Alurnya ditunjukkan pada Gambar 3.4.

[Gambar 3.4]

**Gambar 3.4 Diagram integrasi Neuro-Symbolic**

Penggantian kepala klasifikasi tersebut menimbulkan satu persoalan pada skor deteksi. Karena setiap *leaf*  hanya menyimpan satu label, semua deteksi yang mencapai *leaf*  berkelas sama akan memiliki skor identik di Subbab 2.1.7. Akibatnya, AP dan *Soft*-NMS kehilangan urutan skor. Oleh karena itu, skor dibentuk dari *margin* pada Persamaan 2.26 dan *sigmoid* pada Persamaan 2.27 setiap *node* pada jalur keputusan, yang selanjutnya disebut skor deteksi berbasis *routing margin*, sebagaimana dinyatakan pada Persamaan 3.3.

$$
s\left(x\right)=\prod_{i\in P\left(x\right), {w}_{i}\ne 0}{\sigma \left(\left|{f}_{i}\left(x\right)\right|\right)}
$$

(3.3)

dengan $s\left(x\right)$ skor untuk RoI $x$ pada kelas *leaf* yang dicapai (kelas lain bernilai 0), $P\left(x\right)$ *node* pada jalur keputusan, dan ${f}_{i}$ fungsi keputusan *node* ke-*i* pada Persamaan 2.28. *Node* yang telah dinolkan dilewati. Skor ini hanya berasal dari parameter pohon dan tidak mengubah jalur maupun label. Ilustrasinya ditunjukkan pada Gambar 3.5.

[Gambar 3.5]

**Gambar 3.5 Ilustasi *Routing Margin* pada SODT.**

Berbeda dengan Platt (1999), parameter *sigmoid* bernilai tetap, sehingga $s\left(x\right)$ merupakan skor *confidence*, bukan probabilitas kelas, dan setiap faktornya berada pada rentang \[0,5; 1).

Dengan skor tersebut, alur inferensi model hibrida dari citra uji hingga deteksi akhir dirangkum pada Algoritma 3.3.

**Algoritma 3.3 *Hybrid Inference* dengan *Routing Margin***

**Input:** Citra uji, *Faster* R-CNN (*frozen*), dan SODT terlatih.

**Output:** *Bounding box*, label kelas, skor deteksi, dan jalur keputusan setiap deteksi.

**Langkah-langkah:**

1. *Faster* R-CNN menghasilkan proposal dan *tensor* $64\times 7\times 7$ untuk setiap proposal.
2. SODT menentukan label dari *leaf* yang dicapai, dan skor deteksi berbasis *routing margin* dihitung dengan Persamaan 3.3.
3. Kepala regresi *Faster* R-CNN menghitung koordinat *bounding box* akhir.
4. Deteksi berlabel *background* atau berskor di bawah ambang skor minimum sesuai Tabel 3.4 dibuang, lalu *Soft*-NMS diterapkan dengan Persamaan 2.20.
5. Jalur keputusan, tingkat piramida sumber, dan *feature map neck* setiap deteksi disimpan untuk pembentukan *heatmap*.

2. ***Heatmap* per *Node***

Setiap *node* pada jalur keputusan memperoleh satu *heatmap* yang menunjukkan daerah yang dipertimbangkan oleh *node* tersebut untuk RoI yang dijelaskan. Berbeda dengan peta Kairgeldin dan Carreira-Perpiñán (2025) pada Persamaan 2.34 yang statis, *heatmap* ini dinamis karena memakai nilai fitur RoI, dan dihitung pada feature map *neck*, bukan grid $7\times 7$.

Fungsi keputusan *node* linear terhadap fitur RoI pada Persamaan 2.28, dan *RoI Align* linear terhadap feature map *neck* sesuai Persamaan 2.18. Dengan bobot *node* per kanal ${w}_{i,c}$ dan feature map kanal ke-*c* pada tingkat terpilih ${F}_{c}$, bagian linear fungsi keputusan dapat ditulis sebagai Persamaan 3.4.

$$
{f}_{i}\left(x\right)-{b}_{i}=\sum_{c=1}^{C}{{w}_{i,c}^{T} A {F}_{c}}
$$

(3.4)

dengan $A$ matriks koefisien *RoI Align* dan $C=64$. Gradien Persamaan 3.4 terhadap ${F}_{c}$ adalah ${A}^{T}{w}_{i,c}$, yaitu bobot *node* yang disebar kembali ke posisi asalnya pada feature map. Gradien ini dikalikan dengan nilai feature map dengan menggunakan *Gradient × Input* pada Persamaan 2.23, sebagaimana dinyatakan pada Persamaan 3.5.

$$
{c}_{i}\left[c,p\right]={d}_{i}⋅{\left({A}^{T}{w}_{i,c}\right)}_{p}⋅{F}_{c}\left(p\right)
$$

(3.5)

dengan ${c}_{i}\left[c,p\right]$ kontribusi kanal ke-*c* posisi $p$ terhadap *node* ke-*i*, dan ${d}_{i}$ arah keputusan *node*, sehingga kontribusi positif berarti mendukung arah yang diambil. Karena seluruh operasinya linear, *completeness* yang disebutkan pada Persamaan 2.24 dan 2.25 berlaku secara eksak sebagaimana Persamaan 3.6.

$$
\sum_{c=1}^{C}{\sum_{p}{{c}_{i}\left[c,p\right]}}={d}_{i}⋅\left({f}_{i}\left(x\right)-{b}_{i}\right)
$$

(3.6)

Untuk ditampilkan, kontribusi seluruh kanal diringkas menjadi satu peta dengan Persamaan 3.7.

$$
{H}_{i}\left(p\right)=\sum_{c=1}^{C}{\left|{c}_{i}\left[c,p\right]\right|}
$$

(3.7)

Nilai mutlak membuat ${H}_{i}$ menunjukkan besar pengaruh, baik yang mendukung maupun yang menentang, sedangkan arah keputusan ditampilkan pada jalur pohon.

Ketelitian ${H}_{i}$ berada pada tingkat daerah karena perhitungan eksak berhenti pada *feature map neck*, tetapi tetap lebih halus daripada *grid* $7\times 7$. Alurnya ditunjukkan pada Gambar 3.6.

[Gambar 3.6]

**Gambar 3.6 Alur Pembentukan *Heatmap* per *Node***

Berdasarkan Persamaan 3.4 hingga 3.7, pembentukan *heatmap* untuk satu deteksi dirangkum pada Algoritma 3.4.

**Algoritma 3.4 Pembentukan *Heatmap* per *Node***

**Input:** Satu deteksi beserta *tensor* RoI, jalur keputusan, tingkat piramida, dan *feature map neck*.

**Output:** Satu *heatmap* untuk setiap *node* aktif pada jalur keputusan.

**Langkah-langkah:**

Untuk setiap *node* aktif pada jalur keputusan:

1. Bobot ${w}_{i}$ dikembalikan ke grid $64\times 7\times 7$ dan dikalikan dengan ${d}_{i}$.
2. *RoI Align* dijalankan ulang pada *feature map neck*, lalu gradien Persamaan 3.4 dihitung dengan propagasi mundur.
3. Gradien dikalikan dengan *feature map* sesuai Persamaan 3.5 dan diringkas menjadi ${H}_{i}$ dengan Persamaan 3.7.
4. ${H}_{i}$ dipotong sesuai letak proposal pada tingkat piramida menggunakan *stride* pada Persamaan 2.5 dan 2.6, dinormalisasi, lalu diperbesar dengan interpolasi bilinear sesuai Persamaan 2.16 ke ukuran proposal.
5. *Heatmap* ditumpangkan pada citra.

## 3.9 Evaluasi Model *Neuro-Symbolic*

Evaluasi mencakup dua aspek, yaitu kinerja deteksi model hibrida dan kualitas penjelasan yang dihasilkannya. Seluruh pengujian dilakukan pada data uji.

1. **Evaluasi Deteksi**

Model hibrida dibandingkan dengan *Faster* R-CNN menggunakan metrik pada Subbab 3.4. Untuk mengukur peran *Routing Margin*, metrik yang sama dihitung ulang dengan skor seluruh deteksi diganti menjadi 1, tanpa mengubah jalur maupun label.

Waktu rata-rata per citra uji juga diukur untuk *Faster* R-CNN, model hibrida, dan Grad-CAM (termasuk pembentukan petanya).

2. **Evaluasi Penjelasan**

Grad-CAM menghasilkan satu peta per deteksi, sedangkan SODT satu peta per *node*. Untuk perbandingan, *heatmap* per *node* ditumpuk menjadi satu peta dengan Persamaan 3.8.

$$
M\left(p\right)=\sum_{i\in P\left(x\right)}{{H}_{i}\left(p\right)}
$$

(3.8)

dengan $M\left(p\right)$ peta gabungan pada posisi $p$, $P\left(x\right)$ *node* pada jalur keputusan RoI $x$ pada Persamaan 3.3, dan ${H}_{i}$ *heatmap node* ke-*i* pada Persamaan 3.7. Peta $M$ hanya dipakai untuk perbandingan dengan Grad-CAM.

Penjelasan diuji pada tiga hal, yaitu *faithfulness* tingkat jalur yang memeriksa apakah daerah yang disorot menentukan label, lokalisasi yang memeriksa apakah daerah tersebut jatuh pada cacat sebenarnya, dan *faithfulness* per *node* yang menguji klaim bahwa setiap *heatmap* menentukan keputusan *node*-nya. Perbandingan kualitatif melengkapinya dengan telaah per deteksi.

Seluruhnya dibandingkan dengan Grad-CAM, yang dihitung pada *feature map neck* tempat proposal di-*pool* sesuai anjuran Selvaraju et al. (2017), serta dengan kontrol acak (Adebayo et al., 2018) agar hasilnya tidak dapat dijelaskan oleh pola aktivasi semata. Prosedur pengujiannya dirangkum pada Algoritma 3.5.

**Algoritma 3.5 Evaluasi Penjelasan**

**Input:** Model hibrida, Grad-CAM, data uji, dan *dataset* simbolik uji.

**Output:** Metrik *faithfulness* tingkat jalur dan per *node*, metrik lokalisasi, serta perbandingan visual.

**Langkah-langkah:**

1. Deteksi berskor ≥ 0,5 dipilih dari 500 citra uji.
2. Peta M pada Persamaan 3.8 dan peta Grad-CAM pada Persamaan 2.21 dan 2.22 dihitung untuk deteksi yang sama.
3. Pada *feature map neck* di dalam proposal (diperluas 2 posisi), 50% posisi tertinggi tiap peta dipilih.
4. Posisi tersebut dinolkan untuk *Necessity* sesuai Persamaan 2.41 dan disisakan untuk *Sufficiency* pada Persamaan 2.42, lalu *RoI Align* dijalankan ulang dan label diperiksa.
5. Langkah 3 dan 4 diulang untuk tiga kontrol, yaitu posisi acak, bobot *node* diacak, dan pengurutan tanpa bobot pohon.
6. Setiap peta di-*resample* ke grid 7×7, lalu Pointing Game dan IoU Heatmap dihitung pada proposal longgar (IoU 0,05–0,35).
7. Untuk setiap node aktif, 50% posisi tertinggi ${H}_{i}$ dihapus dan pembalikan tanda ${f}_{i}\left(x\right)$ diperiksa, lalu *Deletion* dan *Insertion* AUC pada Persamaan 2.43 dihitung dalam lima tahap.
8. Deteksi kedua model dan *ground truth* ditampilkan berdampingan, dengan jalur keputusan dan *heatmap* setiap *node* di samping peta Grad-CAM.

# BAB IV HASIL DAN PEMBAHASAN

## 4.1 Hasil Persiapan *Dataset*

Dataset DeepPCB dipartisi menjadi 1.000 citra latih dan 500 citra uji sesuai berkas indeks pada Subbab 3.1, tanpa ada citra yang muncul di kedua himpunan. Statistik anotasi pada kedua himpunan dirangkum dalam Tabel 4.1.

**Tabel 4.1 Statistik Partisi Dataset DeepPCB**

| Metrik | *Train Set* | *Test Set* |
| --- | --- | --- |
| Jumlah Citra | 1.000 | 500 |
| Total Anotasi | 6.873 | 3.140 |
| Rata-rata Anotasi/Citra | 6,87 | 6,28 |
| Minimum Anotasi Citra | 1 | 2 |
| Maximum Anotasi Citra | 15 | 13 |
| Citra Kosong | 0 | 0 |

Setiap citra memuat 1 hingga 15 cacat dengan rata-rata 6,87 pada data latih dan 6,28 pada data uji, sehingga tidak ada citra kosong dan setiap citra merupakan kasus deteksi multi-objek. Distribusi kelas juga relatif seimbang, seperti terlihat pada Gambar 4.1 dan Gambar 4.2.

[Gambar 4.1]

**Gambar 4.1 Distribusi anotasi per kelas pada *training set.***

[Gambar 4.2]

**Gambar 4.2 Distribusi anotasi per kelas pada *test set.***

Kelas terbanyak hanya sekitar 1,4 kali kelas tersedikit, yaitu *mousebite* (1.379) terhadap *spurious copper* (1.010) pada data latih, serta *open* (659) terhadap *spurious copper* (464) pada data uji. Dengan demikian, perbedaan kinerja antarkelas pada pembahasan berikutnya tidak dapat dijelaskan terutama oleh ketimpangan jumlah data.

Dari sisi Anotasi, seluruh *bounding box* berada di dalam batas citra 640×640 piksel dan tidak ada yang berdimensi nol. Gambar 4.3 menunjukkan enam sampel acak dengan 5 hingga 9 anotasi per citra. Setiap *bounding box* menutupi area cacat dengan label yang sesuai, sehingga anotasi dapat langsung digunakan sebagai *ground truth*.

[Gambar 4.3]

**Gambar 4.3 Sampel acak citra PCB dengan anotasi *bounding box ground truth***

## 4.2 Hasil Pra-Pemrosesan *Dataset*

Kedua transformasi pada Tabel 3.3 divisualisasikan untuk memastikan *bounding box* tetap sesuai dengan citranya. Setiap citra diproses dua kali agar pengaruh pemilihan acak dapat terlihat. Hasil *horizontal flip* ditunjukkan pada Gambar 4.4.

[Gambar 4.4]

**Gambar 4.4 Hasil visualisasi augmentasi pembalikan horizontal pada citra PCB**

Pada Gambar 4.4, citra 00041069 dan 20085089 terbalik pada pemrosesan pertama tetapi tidak pada pemrosesan kedua, sedangkan citra 50600013 tidak terbalik sama sekali. Hal ini menunjukkan bahwa *flip* terjadi secara acak, sehingga model menerima orientasi yang berbeda setiap kali citra dimuat. Saat citra terbalik, *bounding box* ikut berpindah ke posisi cerminnya dan tetap menutupi cacat yang sama. Hasil *multi-resolution scaling* ditunjukkan pada Gambar 4.5.

[Gambar 4.5]

**Gambar 4.5 Hasil visualisasi penskalaan multi-resolusi dalam arsitektur model**

Pada Gambar 4.5, resolusi juga dipilih secara acak. Contohnya, citra 13000141 diskalakan menjadi 880×880 piksel pada pemrosesan pertama dan 720×720 piksel pada pemrosesan kedua. Citra kemudian diberi *padding* hingga ukurannya kelipatan 32, yaitu 880 menjadi 896 dan 720 menjadi 736, sedangkan 800 tidak berubah. Karena penskalaan bersifat proporsional, bentuk cacat tidak terdistorsi dan *bounding box* tetap berada pada area cacat.

Selain diskalakan, citra juga dinormalisasi. Normalisasi tidak mengubah tampilan citra, tetapi nilai pikselnya berubah menjadi −2,12 hingga 2,64 (Gambar 4.5), sesuai dengan *mean* dan *std* pada Tabel 3.2. Dengan demikian, pra-pemrosesan menghasilkan variasi orientasi dan skala tanpa merusak kesesuaian anotasi, sehingga data siap digunakan untuk pelatihan *Faster* R-CNN.

## 4.3 Hasil dan Evaluasi *Faster* R-CNN

*Faster* R-CNN dilatih selama 15 *epoch* dengan *hyperparameter* pada Tabel 3.5. Perkembangan *loss* total dan setiap komponennya ditunjukkan pada Gambar 4.6.

[Gambar 4.6]

**Gambar 4.6 *Training Loss Faster* R-CNN selama 15 *Epoch***

*Loss* total turun dari 0,817 menjadi 0,267 dan mendatar setelah epoch ke-12, sehingga model telah konvergen. Penurunan tajam pada epoch ke-9 dan ke-12 terjadi tepat setelah laju pembelajaran diturunkan (Tabel 3.5). Di antara komponennya, *loss regresi bounding box* tetap menjadi yang terbesar, yaitu 0,193, sedangkan loss RPN sudah berada di bawah 0,02 sejak epoch ke-2. Artinya, kesulitan utama model bukan menemukan cacat, melainkan menentukan batas bounding box secara presisi.

Kinerja model pada data uji dirangkum dalam Tabel 4.2, bersama hasil yang dilaporkan Fung et al. (2024) pada *dataset* DeepPCB non-referensial.

**Tabel 4.2 Kinerja Deteksi *Faster* R-CNN pada Data Uji**

| Model | AP50 | AP75 | AP@50:5:85 | mAP@0,5:0,95 | *Precision* | *Recall* | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| *Faster* R-CNN (Fung et al., 2024) | 0,970 | 0,900 | 0,888 | \- | \- | \- | \- |
| *Faster* R-CNN + SF-PSPyramid (Fung et al., 2024) | 0,986 | 0,946 | 0,932 | \- | \- | \- | \- |
| *Faster* R-CNN + SF-PSPyramid (penelitian ini) | 0,979 | 0,920 | 0,901 | 0,759 | 0,910 | 0,982 | 0,945 |

*Recall* sebesar 0,982 menunjukkan bahwa hampir seluruh cacat terdeteksi. Namun, AP turun dari 0,979 (AP50) menjadi 0,759 (mAP@0,5:0,95) saat ambang IoU dinaikkan, sejalan dengan *loss* regresi *bounding box* yang tetap tinggi. Terhadap Fung et al. (2024), AP@50:5:85 model ini berada di antara *Faster* R-CNN standar dan SF-PSPyramid. Selisih 3,1 poin dari SF-PSPyramid diduga berasal dari kanal *neck* yang dikurangi menjadi 64 di Subbab 3.3.

Kinerja per kelas cacat ditunjukkan pada Tabel 4.3.

**Tabel 4.3 Kinerja *Faster* R-CNN per Kelas Cacat**

| Kelas | AP@0,5:0,95 | *Precision* | *Recall* |
| --- | --- | --- | --- |
| *Spurious copper* | 0,889 | 0,933 | 0,991 |
| *Pinhole* | 0,862 | 0,799 | 1,000 |
| *Mousebite* | 0,743 | 0,948 | 0,986 |
| *Spur* | 0,726 | 0,961 | 0,977 |
| *Open* | 0,684 | 0,960 | 0,979 |
| *Short* | 0,651 | 0,859 | 0,956 |

AP tertinggi dicapai *spurious copper* dan *pinhole* yang berbentuk gumpalan atau lubang, sedangkan AP terendah terdapat pada *short* dan *open* yang berada pada jalur konduktor. Hal ini diduga karena batas cacat pada jalur lebih ambigu. Karena distribusi kelas relatif seimbang di Subbab 4.1, perbedaan ini tidak berasal dari jumlah data. Adapun *precision* terendah terdapat pada *pinhole* dengan 0,799 dan *short* dengan 0,859, yang penyebabnya terlihat pada Gambar 4.7.

[Gambar 4.7]

**Gambar 4.7 *Confusion Matrix Faster* R-CNN pada Data Uji**

Sebanyak 97,8% cacat terklasifikasi benar dan hanya 0,7% tertukar antarkelas. Kesalahan utama justru berasal dari 292 area *background* yang terdeteksi sebagai cacat, terutama sebagai *pinhole* dengan 118 dan *short* dengan 73, sehingga *precision* kedua kelas tersebut rendah. Sebaliknya, cacat yang terlewat hanya 46, sehingga model lebih cenderung mendeteksi berlebih daripada melewatkan cacat. Karena SODT dilatih meniru label *teacher*, termasuk *background* di Subbab 3.5, label yang diterimanya hampir tidak tertukar antarkelas, tetapi turut membawa kecenderungan deteksi berlebih pada *pinhole* dan *short*.

## 4.4 Hasil Ekstraksi Fitur *Teacher*

Model *teacher* dari Subbab 4.3 selanjutnya digunakan untuk mengekstrak fitur dan label seluruh proposal RPN di Subbab 3.5, yaitu 1.000 RoI per citra. Distribusi label yang dihasilkan ditunjukkan pada Tabel 4.4.

**Tabel 4.4 Distribusi Label *Teacher* pada RoI Hasil Ekstraksi**

| Label *Teacher* | Data Latih | Data Uji |
| --- | --- | --- |
| *Background* | 854.164 | 431.344 |
| *Open* | 29.288 | 15.116 |
| *Short* | 20.214 | 10.037 |
| *Mousebite* | 29.329 | 12.619 |
| *Spur* | 23.213 | 9.577 |
| *Spurious copper* | 20.964 | 9.894 |
| *Pinhole* | 22.828 | 11.413 |
| Total | 1.000.000 | 500.000 |

Tabel 4.4 menunjukkan bahwa sebagian besar RoI dilabeli *background* oleh *teacher*, yaitu 85,4% pada data latih dan 86,3% pada data uji. Hal ini karena setiap citra hanya memuat sekitar 6 sampai 7 cacat, sedangkan proposalnya berjumlah 1.000. RoI berlabel cacat pun jauh lebih banyak daripada jumlah cacat sebenarnya, yaitu sekitar 21 RoI per cacat, karena satu cacat tertangkap oleh beberapa proposal yang saling tumpang-tindih. Dengan demikian, data yang akan dipelajari SODT didominasi *background*, sedangkan setiap cacat terwakili dari berbagai posisi proposal.

Isi fitur tersebut divisualisasikan dengan merata-ratakan 64 kanal setiap RoI menjadi *grid* 7×7, seperti pada Gambar 4.8.

[Gambar 4.8]

**Gambar 4.8 Visualisasi Fitur RoI 7×7 per Kelas**

Pada kelas cacat, aktivasi tinggi umumnya terkumpul di tengah grid, sedangkan pada *background* aktivasi berada di tepi atau sudut. Hal ini menunjukkan bahwa *RoI Align* mempertahankan letak cacat di dalam proposal. Namun, pola antarkelas sulit dibedakan, misalnya *open*, *short*, dan *pinhole* sama-sama tampak sebagai area terang di tengah *grid*. Hal ini terjadi karena 64 kanal tersebut merupakan hasil pembelajaran *backbone* dan *neck* yang maknanya tidak diketahui, sehingga fitur ini tetap bersifat *black-box*. Dengan demikian, fitur RoI menyimpan informasi letak, tetapi dasar keputusannya tidak dapat dibaca langsung. Hal inilah yang ditangani SODT, karena setiap bobot *node*-nya terikat pada kanal dan posisi grid tertentu sehingga dapat dipetakan kembali menjadi *heatmap* di Subbab 3.8.2.

## 4.5 Evaluasi Model Simbolik (*Sparse Oblique Decision Tree*)

1. **Pelatihan TAO**

Proses optimasi parameter SODT menggunakan *Tree Alternating Optimization* (TAO) menunjukkan pola konvergensi yang khas, sebagaimana terlihat pada Gambar 4.10. Kurva pelatihan menggambarkan dua aspek kritis, yaitu peningkatan fidelitas model terhadap *teacher* (*Faster* R-CNN) dan penurunan jumlah bobot non-nol yang terjadi secara progresif sepanjang iterasi.

[Gambar 4.10]

**Gambar 4.10 Grafik TAO Training History**

Pada iterasi awal (0–5), terjadi peningkatan fidelitas yang tajam dari 75% menjadi 92%, diikuti oleh konvergensi bertahap menuju 96,7% pada iterasi ke-15. Pola ini mengindikasikan bahwa algoritma TAO berhasil menemukan parameter optimal yang mempertahankan kesetiaan terhadap *teacher* sambil meningkatkan sparsitas melalui penalti L₁.

Pada iterasi terakhir, terjadi penurunan drastis jumlah bobot non-nol dari sekitar 60.000 menjadi 3.326. Fenomena ini disebabkan oleh *post-processing* berupa *node pruning* yang tidak memberikan kontribusi signifikan terhadap keputusan klasifikasi. Proses ini memastikan struktur pohon tetap ramping tanpa mengorbankan fidelitas model, sehingga menghasilkan representasi logika yang lebih mudah diinterpretasikan oleh manusia.

2. **Fidelitas Terhadap *Teacher***

Evaluasi fidelitas dilakukan melalui metrik *Teacher-Student Agreement*, yang mengukur kesesuaian prediksi SODT dengan model *teacher* (Faster R-CNN). Seperti ditunjukkan pada Gambar 4.11, model mencapai tingkat kesesuaian sebesar 96,71% pada data pelatihan dan 96,77% pada data uji. Stabilitas metrik ini antara dua himpunan data membuktikan bahwa SODT berhasil meniru logika *teacher* tanpa mengalami *overfitting*.

[Gambar 4.11]

**Gambar 4.11 Grafik *Teacher-Student Agreement* pada Data Pelatihan dan Data Uji**

Analisis lebih detail pada tingkat kelas cacat pada Gambar 4.12 menunjukkan variasi fidelitas yang moderat di antara keenam kategori. Kelas *spur* dan *background* mencatatkan kesesuaian tertinggi (97,0%), sedangkan kelas *short* menunjukkan nilai terendah (91,1%) akibat variasi geometris yang lebih kompleks. Meskipun demikian, seluruh kelas mempertahankan fidelitas di atas 90%, mengonfirmasi bahwa SODT mampu mereplikasi keputusan *teacher* secara konsisten.

[Gambar 4.12]

**Gambar 4.12 *Per-Class Agreement* antara SODT dan *Teacher* pada Data *Held-out***

3. ***Sparsity* Model**

Sparsitas menjadi faktor krusial dalam meningkatkan interpretabilitas model. Gambar 4.13 memvisualisasikan distribusi bobot pada seluruh simpul internal pohon, menunjukkan bahwa dari total 97.216 parameter, hanya 3.326 bobot yang bernilai non-nol (96,6% sparsitas). Tingkat sparsitas ini memiliki implikasi langsung terhadap kualitas penjelasan visual yang dihasilkan.

[Gambar 4.13]

**Gambar 4.13 Visualisasi *Global SODT Sparsity***

Sparsitas tinggi memaksa setiap *node* keputusan hanya bergantung pada subset fitur spasial yang paling informatif. Hal ini menghasilkan heatmap yang lebih bersih dan terlokalisasi dengan presisi, sehingga teknisi dapat dengan mudah mengidentifikasi wilayah kritis yang menjadi dasar keputusan klasifikasi. Struktur pohon yang *terpruning* seperti pada Gambar 4.14 hanya terdiri dari 12 simpul aktif, yang berada dalam batas kapasitas kognitif manusia untuk melacak jalur penalaran secara manual.

[Gambar 4.14]

**Gambar 4.14 Diagram Struktur Pohon SODT**

Kombinasi antara fidelitas tinggi dan sparsitas ekstrem membuktikan bahwa SODT tidak hanya mampu mereplikasi keputusan *teacher*, tetapi juga menghasilkan penjelasan yang *faithful* dan mudah dipahami. Dengan mengeliminasi fitur yang tidak relevan, model secara alami mengarahkan perhatian teknisi ke wilayah spasial yang paling kritis untuk validasi, sehingga memenuhi tujuan utama arsitektur neuro-simbolik dalam mendukung proses inspeksi PCB yang transparan dan akuntabel.

## 4.6 Evaluasi Kinerja Deteksi *Neuro-Symbolic*

1. **Hasil Keseluruhan**

Evaluasi kinerja deteksi model Neuro-Symbolic (NeSy) secara keseluruhan diukur menggunakan berbagai metrik standar untuk melihat seberapa baik model terintegrasi ini melokalisasi dan mengklasifikasikan cacat PCB. Hasil metrik deteksi secara agregat disajikan pada Gambar 4.15.

[Gambar 4.15]

**Gambar 4.15 Metriks deteksi NeSy.**

Model NeSy mencatatkan nilai *Recall* yang sangat tinggi, yaitu 0,985, serta *F1-Score* sebesar 0,897. Nilai *Recall* yang mendekati 1,0 ini mengindikasikan bahwa model hampir tidak melewatkan cacat yang ada pada citra uji (*low false negative*). Namun, terdapat kesenjangan yang cukup signifikan pada nilai *Precision* yang bernilai 0,823 dan *mAP@0.5 yang bernilai* 0,87. Penurunan pada metrik *Precision* dan *mAP* ini tidak disebabkan oleh kegagalan model dalam mendeteksi cacat, melainkan akibat tingginya tingkat *False Positive* yang dihasilkan oleh model. Fenomena penurunan presisi dan *mAP* ini terkonfirmasi secara visual melalui Matriks Konfusi pada Gambar 4.16.

[Gambar 4.16]

**Gambar 4.16 *Confusion Matrix* pada Neuro-Symbolic**

Pada baris *background* di Gambar 4.16, terdapat total 667 *False Positive* di mana area sirkuit normal (*background*) diklasifikasikan secara keliru sebagai salah satu dari enam kelas cacat. Sebaliknya, pada baris kelas cacat, hanya terdapat 47 *False Negative* (cacat yang terlewat dan dikira background). Dominasi *False Positive* dari background inilah yang secara matematis memberikan penalti besar pada perhitungan *Precision* dan *mAP*, sehingga menyebabkan nilai keduanya lebih rendah dibandingkan *Recall*.

Meskipun menghasilkan banyak Fal*se Positive* dari *background*, *Confusion Matrix* NeSy pada Gambar 4.16 menunjukkan jumlah True Positive pada beberapa kelas lebih tinggi dibanding model *teacher* (Faster R-CNN). Misalnya pada kelas open yang di mana pada NeSy sebanyak 650 sementara Faster RCNN sebanyak 647. Hal ini menunjukkan bahwa SODT mampu mengklasifikasikan ulang RoI yang sebelumnya salah dikategorikan oleh MLP Faster R-CNN. Namun, hal ini membuat SODT cenderung *over-sensitif* terhadap pola *background* yang kompleks dan mirip dengan cacat yang asli. Oleh karena itu, SODT berhasil memperbaiki batas keputusan untuk kasus-kasus ambigu, tetapi belum cukup selektif dalam membedakan background normal dari cacat mikro

Gambar 4.17 menyajikan rincian kinerja deteksi pada tingkat per-kelas. Nilai Recall untuk seluruh kelas cacat berada di atas 0,96, namun nilai Precision bervariasi. Kelas short (0,740) dan spurious\_copper (0,760) memiliki presisi terendah, yang mengindikasikan bahwa kedua kelas ini paling sering memicu False Positive dari area background.

[Gambar 4.17]

**Gambar 4.17 Rincian *Precision* dan *Recall* per kelas cacat.**

Untuk memvalidasi temuan kuantitatif tersebut secara visual, Gambar 4.18 menyajikan perbandingan hasil inferensi antara Faster R-CNN, NeSy, dan *Ground Truth* (GT). Berdasarkan gambar tersebut, terlihat bahwa NeSy secara umum berhasil mendeteksi mayoritas cacat yang sejalan dengan GT, membuktikan bahwa substitusi MLP dengan SODT tidak mengganggu kemampuan deteksi spasial. Namun, pada beberapa sampel seperti pada Gambar 4.19, NeSy menghasilkan *bounding box* tambahan pada area sirkuit normal, yang secara visual mengonfirmasi bahwa penurunan *Precision* murni disebabkan oleh sensitivitas berlebih terhadap *background*, bukan karena kegagalan melokalisasi cacat yang sebenarnya.

[Gambar 4.18]

**Gambar 4.18 Perbandingan keberhasilan deteksi pada Neuro-Symbolic dengan Faster RCNN dan *Ground Truth*.**

[Gambar 4.19]

**Gambar 4.19 Perbandingan kegagalan deteksi pada Neuro-Symbolic dengan Faster RCNN dan *Ground Truth*.**

2. **Analisis Kegagalan Deteksi NeSy**

Berdasarkan evaluasi pada subbab sebelumnya, penurunan *Precision* dan *mAP* secara fundamental disebabkan oleh tingginya *False Positive* yang berasal dari area *background*. Kesalahan ini tidak terjadi secara acak, melainkan dipicu oleh pola-pola spesifik pada area sirkuit normal. Analisis difokuskan pada empat kelas dengan tingkat kesalahan signifikan, yaitu *short*, *spur*, *pinhole*, dan *spurious\_copper*. Untuk mengungkap akar masalah ini, komparasi visual antara prediksi NeSy, *Ground Truth*, dan pola unik pada sirkuit yang disajikan pada gambar-gambar berikut.

a. Kelas *Short*

Tingginya tingkat *False Positive* pada kelas *short* berbanding lurus dengan rendahnya jumlah RoI untuk kelas ini, yang hanya mencapai 1,87% pada data pelatihan dan merupakan yang paling rendah dibandingkan kelas lainnya sebagaimana dirangkum pada Tabel 4.3. Ketidakseimbangan data ini menyebabkan SODT kesulitan mempelajari variasi pola *short* secara komprehensif. Untuk memvisualisasikan dampak keterbatasan ini, Gambar 4.20 menyajikan komparasi visual deteksi pada kelas *short*.

*Gambar 4.20 Komparasi visual False Positive pada kelas short.*

Berdasarkan Gambar 4.20, SODT cenderung gagal mendeteksi *short* yang memiliki benjolan atau terlalu pendek. Sebaliknya, SODT lebih mampu mendeteksi *short* yang relatif panjang dan tidak memiliki timbulan.

b. Kelas *Spur*

Meskipun jumlah *Region of Interest* untuk kelas *spur* tergolong seimbang dengan persentase 2,15% pada data pelatihan seperti pada Tabel 4.3, SODT menunjukkan keterbatasan dalam menangani variasi orientasi geometris cacat ini. Pola kesalahan deteksi pada kelas *spur* diilustrasikan secara visual pada Gambar 4.21.

## 4.7 Hasil dan Evaluasi *Explanation* pada *Neuro-Symbolic*

Evaluasi Explanation pada Neuro-Symbolic menggunakan metrik untuk mengukur *faithfulness* dan kualitas heatmap. Kemudian akan dibandingkan dengan GradCAM yang di mana telah disajikan pada Gambar 4.18.

[Gambar 4.18]

**Gambar 4.18 Perbandingan *Faithfulness* dan kualitas *Heatmap* pada NeuroSymbolic dan GradCAM**

1. **Faithfulness**

Neuro-Symbolic (SODT) mencapai Sufficiency Preservation 1,000 dan Necessity Flip Rate 0,971, menunjukkan penjelasan yang benar-benar merepresentasikan keputusan internal model. Sementara itu, Grad-CAM hanya mencapai Sufficiency 0,956 dan Necessity 0,047, mengindikasikan penjelasan yang tidak memenuhi prinsip *necessity* (hanya 4,7% kasus di mana penghapusan area penting mengubah prediksi).

2. **Spatial Grounding**

Grad-CAM memiliki IoU Heatmap (0,994) sedikit lebih tinggi dari SODT (0,920) karena *bleeding effect* (area heatmap melebar ke luar bounding box). Sementara itu, untuk Pointing Game, Kedua metode hampir identik (0,994 vs 0,983), membuktikan bahwa keduanya mampu mengidentifikasi wilayah cacat. Namun, SODT lebih *faithful* karena heatmapnya **terlokalisasi presisi** pada fitur yang secara eksplisit digunakan dalam keputusan.

3. **Waktu komputasi**

Grad-CAM membutuhkan 12,6× lebih lama dibanding Neuro-Symbolic (5045,4 ms vs 426,2 ms) karena mekanisme komputasi yang tidak efisien seperti ditunjukkan pada Gambar 4.19.

[Gambar 4.19]

**Gambar 4.19 Perbandingan *inference time* pada ketiga model.**

Hal ini bisa terjadi karena pada Faster R-CNN, Grad-CAM harus:

1. Menjalankan satu kali ***backward pass* penuh** melalui seluruh arsitektur dua-tahap (RPN + RoI Head) untuk **setiap kotak deteksi (RoI)** yang dihasilkan.
2. Menghitung gradien dari skor kelas yang diprediksi hingga ke lapisan konvolusional terakhir.

Proses komputasi berulang untuk setiap RoI ini sangat tidak efisien dan membebani komputasi, sehingga tidak cocok untuk skenario *real-time* seperti inspeksi PCB.

# DAFTAR PUSTAKA

Ali, A.M.M., Ziyi, X., Sahlan, S., Khamis, N., Nor Rashid, F.’A., 2023. Transparency in Detecting Defects of a Printed Circuit Board: Harnessing XAI for Improved Quality Control in Electronic Manufacturing. In: 2023 IEEE 9th International Conference on Smart Instrumentation, Measurement and Applications (ICSIMA), pp. 1-6.

Arrighi, L., Pintus, F., Giuliani, L., Vantini, S., & Bacciu, D., 2023. A study on the faithfulness of feature attribution explanations in pruned vision-based multi-task learning. *CEUR Workshop Proceedings*, 4017.

Bodla, N., Singh, B., Chellappa, R., Davis, L.S., 2017. Soft-NMS — Improving object detection with one line of code. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 5561–5569.

Carreira Perpiñán, M.Á., Tavallali, P., 2018. Alternating optimization of decision trees, with application to learning sparse oblique trees. In: *Advances in Neural Information Processing Systems (NeurIPS)*, 31, pp. 1211–1221.

Canha, D., Kubler, S., Främling, K., & Fagherazzi, G. (2025). A Functionally-Grounded Benchmark Framework for XAI Methods: Insights and Foundations from a Systematic Literature Review. *ACM Computing Surveys*, *57*(12), Article 320.

Chen, X., Wu, Y., He, X., Ming, W., 2023. A Comprehensive Review of Deep Learning-Based PCB Defect Detection. IEEE Access 11, 139017-139036.

Coombs, C.F., Holden, H.T., 2016. *Printed Circuits Handbook*, 7th ed. McGraw-Hill Education, New York.

Cortes, C., Vapnik, V., 1995. Support-vector networks. Machine Learning, 20(3), pp. 273–297.

d'Avila Garcez, A. and Lamb, L.C., 2023. Neurosymbolic AI: the 3rd wave. *Artificial Intelligence Review*, 56(11), pp.12387-12406.

Elkan, C., 2001. The foundations of cost-sensitive learning. In: Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI), pp. 973–978.

Fung, K.C., Xue, K.-W., Lai, C.-M., Lin, K.-H., Lam, K.-M., 2024. Improving PCB defect detection using selective feature attention and pixel shuffle pyramid. Results in Engineering 21, 101992.

Girshick, R. (2015). Fast R-CNN. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 1440-1448). IEEE.

Goodfellow, I., Bengio, Y., Courville, A., 2016. *Deep Learning*. MIT Press, Cambridge, MA.

Hada, S.S., Carreira-Perpiñán, M.Á. and Zharmagambetov, A., 2024. Sparse oblique decision trees: a tool to understand and manipulate neural net features. *Data Mining and Knowledge Discovery*, 38(5), pp.2863-2902.

Han, Z., Hong, M., Wang, D., 2017. Deep learning and applications. In: *Signal Processing and Networking for Big Data Applications*. Cambridge University Press, Cambridge, pp. 203-228.

He, H., Garcia, E.A., 2009. Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), pp. 1263–1284.

Hinton, G., Vinyals, O., Dean, J., 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

IPC, 2015. IPC-6012D: Qualification and Performance Specification for Rigid Printed Boards. Association Connecting Electronics Industries.

IPC, 2020. *IPC-A-600K: Acceptability of Printed Boards*. Association Connecting Electronics Industries.

Kairgeldin, R. and Carreira-Perpiñán, M.Á., 2025. Neurosymbolic models based on hybrids of convolutional neural networks and decision trees. *Proceedings of Machine Learning Research (NeSy 2025)*, 284, pp.796-813.

Khandpur, R.S., 2005. *Printed Circuit Boards: Design, Fabrication, and Assembly*. McGraw-Hill Education.

Klette, R., 2014. *Concise Computer Vision: An Introduction into Theory and Algorithms*. Springer, London.

LeCun, Y., Bengio, Y., Hinton, G., 2015. Deep learning. *Nature*, 521(7553), pp. 436-444.

Manigrasso, F., Miro, F.D., Morra, L. and Lamberti, F., 2021. Faster-LTN: a neuro-symbolic, end-to-end object detection architecture. In: *International Conference on Artificial Neural Networks (ICANN)*. Springer, pp.40-52.

Minaee, S., Boykov, Y., Porikli, F., Plaza, A., Kehtarnavaz, N., Terzopoulos, D., 2021. Image segmentation using deep learning: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(7), pp.3523-3542.

Petsiuk, V., Das, A., & Saenko, K. (2018). RISE: Randomized Input Sampling for Explanation of Black-box Models. In *Proceedings of the British Machine Vision Conference (BMVC)* (pp. 1-13). BMVA Press.

Platt, J.C., 1999. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: Advances in Large Margin Classifiers. MIT Press, Cambridge, MA, pp. 61–74.

Prince, S. J. D. (2023). *Understanding Deep Learning*. The MIT Press

Rudin, C., 2019. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), pp.206-215.

Ren, M., Zeng, W., Yang, B., Urtasun, R., 2018. Learning to reweight examples for robust deep learning. In: Proceedings of the 35th International Conference on Machine Learning (ICML), PMLR 80, pp. 4334–4343.

Saadallah, A., Büscher, J., Abdulaaty, O., Panusch, T., Deuse, J., Morik, K., 2022. Explainable Predictive Quality Inspection using Deep Learning in Electronics Manufacturing. Procedia CIRP 107, 594-599.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 618-626). IEEE.

Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A. P., Bishop, R., ... & Wang, Z. (2016). Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 1874-1883). IEEE

Szeliski, R., 2022. *Computer Vision: Algorithms and Applications*, 2nd ed. Springer, Cham.

Tang, S., He, F., Huang, X., Yang, J., 2019. Online PCB Defect Detector On A New PCB Defect Dataset. arXiv preprint arXiv:1902.06197.

Tziolas, T., Papageorgiou, K., Theodosiou, T., Ioannidis, D., Dimitriou, N., Tinker, G., Papageorgiou, E., 2025. Explainable AI Methods for Identification of Glue Volume Deficiencies in Printed Circuit Boards. Applied Sciences 15(16), 9061.

Tzionis, G., Mouratidis, P., Kougka, G., Gialampoukidis, I., Vrochidis, S., Kompatsiaris, I., Vlachopoulou, M., 2026. A review of explainable AI methods and their application in manufacturing systems. Discover Applied Sciences 8, 52.

Wang, Y., Huang, J., Dipu, M.S.K., Zhao, H., Gao, S., Zhang, H., Lv, P., 2024. YOLO-RLC: An Advanced Target-Detection Algorithm for Surface Defects of Printed Circuit Boards Based on YOLOv5. Computers, Materials & Continua 80(3), 4973-4995.

Zhao, X., Wang, L., Zhang, Y., Han, X., Deveci, M., Parmar, M., 2024. A review of convolutional neural networks in computer vision. *Artificial Intelligence Review* 57(4), 99.

Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., Torralba, A., 2016. Learning deep features for discriminative localization. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2921–2929.
