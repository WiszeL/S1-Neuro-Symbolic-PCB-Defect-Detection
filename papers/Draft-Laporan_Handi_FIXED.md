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

1.  Nama Pembimbing 1 ( <u>ttd</u> )

NIP .....

2.  Nama Pembimbing 2 ( <u>ttd</u> )

NIP .....

3.  Nama Penguji Utama ( <u>ttd</u> )

NIP .....

Mengetahui,

| Koordinator Skripsi |     | Ketua Jurusan .... |
|---------------------|-----|--------------------|
|                     |     |                    |
| NIP ....            |     | NIP ....           |

# BAB I PENDAHULUAN

## 1.1 Latar Belakang

Perkembangan teknologi elektronik meningkatkan kebutuhan terhadap komponen yang andal dan berkualitas tinggi, khususnya *Printed Circuit Board* (PCB). PCB berfungsi sebagai penopang sekaligus penghubung antar-komponen yang menentukan keandalan operasional sistem elektronik. Untuk memenuhi tuntutan keandalan tersebut, PCB kemudian dikembangkan dengan arsitektur yang semakin kompleks. Namun, kompleksitas ini justru meningkatkan risiko munculnya cacat pada PCB akibat kesalahan manusia maupun gangguan mesin selama proses produksi (Chen et al., 2023; Fung et al., 2024). Keberadaan cacat tersebut dapat berdampak pada penurunan kualitas produk, peningkatan biaya produksi, hingga risiko keselamatan manusia (Ali et al., 2024; Fung et al., 2024). Dengan demikian, metode inspeksi menjadi tahapan krusial untuk mendeteksi keberadaan cacat pada PCB.

Tahap inspeksi PCB selama ini bertumpu pada metode konvensional seperti pemeriksaan visual manual dan pengujian kelistrikan. Namun, metode-metode tersebut kerap dinilai tidak efisien, memakan biaya tinggi, dan rentan terhadap kesalahan akibat faktor manusia (Ali et al., 2024; Chen et al., 2023; Fung et al., 2024). Keterbatasan tersebut mendorong pengembangan solusi otomatis berbasis *Artificial Intelligence* (AI), khususnya pendekatan *deep learning*. Berbagai model *deep learning* seperti *You Only Look Once* (YOLO), *Single Shot MultiBox Detector* (SSD), dan *Faster* *Region-based Convolutional Neural Network* (R-CNN) telah banyak diterapkan untuk deteksi cacat PCB dan menunjukkan performa yang tinggi dalam berbagai studi (Chen et al., 2023). Sebagai contoh, Wang et al. (2024) mengembangkan YOLO-RLC, yaitu YOLOv5 dengan jaringan konvolusi kernel besar residual yang meningkatkan akurasi deteksi cacat PCB. Bahkan, penelitian oleh Fung et al. (2024) menunjukkan bahwa optimasi pada arsitektur *Faster* R-CNN dapat meningkatkan performa deteksi cacat PCB, terutama pada cacat yang berukuran kecil. Hasil beberapa penelitian tersebut menunjukkan bahwa pendekatan berbasis *deep learning* merupakan pendekatan yang dapat diandalkan untuk tahap inspeksi.

Meskipun pendekatan *deep learning* menunjukkan performa deteksi tinggi, model ini memiliki kelemahan berupa sifat *black-box,* yakni ketidakmampuan model dalam memberikan penjelasan atas setiap keputusan yang dihasilkannya. Padahal, dalam industri manufaktur berisiko tinggi seperti produksi PCB, penjelasan tersebut diperlukan teknisi untuk memvalidasi dan mempertanggungjawabkan hasil deteksi  (Tzionis et al., 2026). Tanpa penjelasan yang memadai, proses validasi menjadi sulit dilakukan sehingga berpotensi menurunkan kepercayaan teknisi terhadap model (Tziolas et al., 2025). Oleh karena itu, kemampuan model dalam menjelaskan keputusan (*explainability*) menjadi kebutuhan esensial guna mendukung proses validasi teknisi dalam inspeksi cacat PCB (Saadallah et al., 2022; Chen et al., 2023).

Untuk merespons kebutuhan *explainability* tersebut, *Explainable Artificial Intelligence* (XAI) menjadi pendekatan yang dominan diterapkan dalam sistem inspeksi manufaktur (Tzionis et al., 2026). Pendekatan XAI menghadirkan *explainability* dengan membuat proses pengambilan keputusan model AI dapat ditelusuri dan dipahami. Dalam inspeksi PCB, penelitian Tziolas et al. (2025) menunjukkan bahwa *Deep* *Shapley Additive Explanations* (SHAP) dan *Gradient-weighted Class Activation Mapping* (Grad-CAM) dapat menyoroti area visual yang memengaruhi keputusan model CNN, sehingga meningkatkan interpretabilitas model bagi teknisi. Penelitian lain oleh Saadallah et al. (2022) menggunakan penjelasan berbasis *heatmap* yang mengungkap fitur-fitur penting pada data *Solder Paste Inspection* (SPI) PCB, sehingga membantu teknisi dalam memvalidasi relevansi fitur tersebut terhadap prediksi kualitas produk. Dengan demikian, XAI mampu memenuhi kebutuhan *explainability* pada inspeksi PCB melalui justifikasi visual atas keputusan model.

Meskipun informatif, metode XAI Grad-CAM dan SHAP pada studi tersebut, masih memiliki keterbatasan berupa sifat *post-hoc*, yaitu penjelasan yang baru diberikan setelah model *black-box* menghasilkan prediksinya. Mekanisme ini menyebabkan penjelasan yang dihasilkan tidak memenuhi *faithfulness*, yaitu kemampuannya dalam merepresentasikan keputusan model yang sesungguhnya, sehingga penjelasan tersebut hanyalah berupa perkiraan (Rudin, 2019). Pada domain berisiko tinggi, peta *saliency* *post-hoc*, termasuk Grad-CAM, terbukti belum sepenuhnya dapat diandalkan (Arun et al., 2021), sehingga validasi teknisi yang bertumpu padanya dalam inspeksi PCB berpotensi mengarah pada keputusan yang salah. Dengan demikian, ketiadaan *faithfulness* dalam penjelasan *post-hoc* justru mengurangi kualitas *explainability* yang dibutuhkan untuk validasi teknisi (Rudin, 2019).

Untuk mengatasi keterbatasan *faithfulness* dalam penjelasan model, arsitektur *neuro-symbolic* hadir dengan menggabungkan ekstraksi fitur dari *deep learning* dan penalaran transparan berbasis simbolik. Dalam arsitektur ini, komponen simbolik terintegrasi langsung ke dalam mekanisme pengambilan keputusan sehingga penjelasan yang dihasilkan bersifat *faithful* (d'Avila Garcez & Lamb, 2023). Arsitektur ini telah ditunjukkan oleh model *Faster*-LTN yang mengintegrasikan *Faster* R-CNN dengan *Logic Tensor Network* (LTN) untuk mempertahankan performa deteksi sekaligus memberikan penalaran terstruktur (Manigrasso et al., 2021). Sejalan dengan arah tersebut, Hada et al. (2024) serta Kairgeldin dan Carreira-Perpiñán (2025) mengembangkan integrasi CNN dengan *sparse oblique decision tree* (SODT) yang membuat proses pengambilan keputusan model lebih dapat diinterpretasikan dan divisualisasikan. Dengan demikian, ketiga penelitian tersebut berpotensi menghadirkan *faithfulness* sehingga meningkatkan kualitas *explainability* pada inspeksi PCB.

Untuk mewujudkan *explainability* yang *faithfulness*, penelitian ini mengusulkan integrasi *Faster* R‑CNN dengan SODT sebagai sistem deteksi cacat berbasis arsitektur *neuro‑symbolic*. *Faster* R‑CNN dipilih karena terbukti efektif mendeteksi cacat berukuran kecil pada PCB melalui optimalisasi SF‑PSPyramid (Fung et al., 2024), sehingga berperan sebagai komponen ekstraksi fitur visual dan deteksi objek. Sementara itu, SODT dipilih karena kemampuannya dalam meniru (*mimic)* keputusan jaringan saraf *teacher* dengan akurasi tinggi, namun dengan mekanisme eliminasi fitur (*sparsity*) yang menghasilkan struktur pohon lebih sederhana (Hada et al., 2024; Kairgeldin & Carreira‑Perpiñán, 2025). Integrasi ini dirancang untuk mempertahankan performa deteksi tinggi dari *Faster* R‑CNN sekaligus menghadirkan penjelasan yang *faithful*. Dengan demikian, sistem ini ditujukan untuk mendukung validasi teknisi dalam inspeksi cacat PCB.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang yang telah diuraikan, rumusan masalah dalam penelitian ini adalah sebagai berikut.

1.  Bagaimana model *deep learning* dapat mempertahankan performa deteksi tinggi pada inspeksi PCB sekaligus mengatasi sifat *black‑box* yang menghambat validasi teknisi?

2.  Bagaimana arsitektur *neuro‑symbolic* yang mengintegrasikan *Faster* R‑CNN dengan *Sparse Oblique Decision Tree* (SODT) dapat dikembangkan untuk menghadirkan sistem deteksi cacat PCB yang akurat sekaligus menyediakan *explainability* yang *faithful*?

## 1.3 Batasan Masalah

Agar penelitian ini tetap terarah dan fokus sesuai dengan tujuan yang telah ditetapkan, ruang lingkup permasalahan dibatasi pada hal-hal sebagai berikut.

1.  Objek Penelitian

Penelitian difokuskan pada deteksi enam jenis cacat visual PCB, yaitu *open*, *short*, *mousebite*, *spur*, *pinhole*, dan *spurious copper*, menggunakan pendekatan *deep learning* dan *neuro‑symbolic*.

2.  Arsitektur Model

Model merupakan integrasi Faster R‑CNN (mengacu pada implementasi Fung et al., 2024) dengan *Sparse Oblique Decision Tree* (SODT) tanpa modifikasi terhadap struktur internal kedua komponen.

3.  Sumber Data

Data yang digunakan berasal dari dataset publik DeepPCB yang memuat 1.500 pasang citra beserta anotasi posisi dan kelas cacat.

4.  Metode Evaluasi *Explainability*

Evaluasi kualitas *explainability* dilakukan secara kuantitatif melalui perbandingan dengan Grad‑CAM sebagai *baseline post‑hoc* yang telah teruji, tanpa melibatkan studi pengguna atau wawancara teknisi.

## 1.4 Tujuan Penelitian

1.  Mengembangkan sistem deteksi cacat PCB berbasis integrasi Faster R CNN dan Sparse Oblique Decision Tree (SODT) yang mampu mempertahankan performa deteksi tinggi sekaligus menyediakan penjelasan yang faithful guna mendukung proses validasi teknisi.

2.  Mengevaluasi apakah sistem yang diusulkan mampu menghasilkan penjelasan yang *faithful* dan lebih baik dibandingkan pendekatan *post-hoc* Grad-CAM.

## 1.5 Manfaat Penelitian

1.  Mendukung proses validasi teknisi di industri manufaktur elektronik melalui sistem deteksi cacat PCB yang tidak hanya akurat, tetapi juga menyediakan penjelasan yang *faithful* dan dapat dipertanggungjawabkan.

2.  Memberikan bukti empiris bahwa integrasi Faster R CNN dan *Sparse Oblique Decision Tree* (SODT) mampu mengatasi keterbatasan *faithfulness* yang melekat pada pendekatan XAI *post-hoc*.

**\**

# BAB II TINJAUAN PUSTAKA

## 2.1 Dasar Teori

### 2.1.1 *Printed Circuit Board* (PCB)

*Printed Circuit Board *(PCB) adalah papan dari bahan isolator yang dilapisi tembaga. PCB berfungsi sebagai jalur penghubung listrik sekaligus penyangga antarkomponen elektronik (Coombs & Holden, 2016; Khandpur, 2005).

Kriteria inspeksi PCB di industri mengacu pada standar *Association Connecting Electronics Industries* (IPC), khususnya IPC-A-600 dan IPC-6012 (IPC, 2015, 2020). Standar ini menyatakan suatu kondisi sebagai cacat apabila melanggar batas toleransi, misalnya jalur konduktor yang terlalu sempit atau jarak antarjalur yang terlalu dekat. Dalam *Computer Vision*, pelanggaran tersebut dipandang sebagai cacat visual yang polanya dapat dipelajari oleh model *deep learning* (Tang et al., 2019; Chen et al., 2023). Enam jenis cacat yang umum dipakai pada penelitian deteksi PCB dirangkum pada Tabel 2.1.

**Tabel 2.1 Kategori Cacat Visual Umum pada PCB.**

| **Jenis Cacat** | **Definisi & Karakteristik Visual** |
|:----------:|-----------------------------------------------------------|
| *Open* | Jalur tembaga terputus, sehingga muncul celah yang memutus jalur. |
| *Short* | Dua jalur yang seharusnya terpisah tetapi menjadi tersambung, membentuk jembatan tembaga. |
| *Mousebite* | Tepi jalur tergerus seperti gigitan, sehingga lebar jalur berkurang secara tidak merata. |
| *Spur* | Tonjolan tembaga kecil yang keluar dari tepi jalur menuju area yang seharusnya kosong. |
| *Pinhole* | Lubang kecil berbentuk lingkaran di dalam area tembaga yang seharusnya padat. |
| *Spurious Copper* | Tembaga liar yang muncul terpisah di area non-konduktif, tidak terhubung ke jalur utama. |

Keenam cacat tersebut terbagi menjadi dua kelompok pola. Kelompok pertama adalah pengurangan material (*open*, *mousebite*, *pinhole*), yaitu hilangnya sebagian tembaga. Kelompok kedua adalah penambahan material liar (*short*, *spur*, *spurious copper*), yaitu munculnya tembaga di tempat yang salah.

### 2.1.2 Computer Vision

*Computer Vision *(CV) adalah cabang ilmu komputer yang membuat komputer mampu memahami isi gambar secara otomatis (Szeliski, 2022). Berbeda dengan pengolahan citra yang hanya memanipulasi piksel, CV bertujuan menghasilkan makna atau keputusan dari isi gambar (Prince, 2023). CV mencakup tiga tugas utama yang dibedakan berdasarkan kedetailan keluarannya.

1.  **Klasifikasi Citra** (*Image Classification*) memberikan satu label kelas untuk keseluruhan gambar, tanpa memperhatikan letak objek (Prince, 2023).

2.  **Deteksi Objek** (*Object Detection*) mengenali objek, menentukan kelasnya, sekaligus menunjukkan posisinya menggunakan kotak pembatas (*bounding box*) (Szeliski, 2022).

3.  **Segmentasi Citra** (*Image Segmentation*) memberi label pada setiap piksel, sehingga batas objek tergambar lebih presisi (Minaee et al., 2021).

Perbandingan ketiga tugas tersebut ditunjukkan pada Gambar 2.1.

[Gambar 2.1]

**Gambar 2.1 Perbandingan Klasifikasi, Deteksi Objek, dan Segmentasi**

Deteksi objek merupakan tugas yang relevan untuk inspeksi visual, karena selain mengenali jenis cacat juga menunjukkan lokasinya. Representasi dan sistem koordinat *bounding box* ditunjukkan pada Gambar 2.2.

[Gambar 2.2]

**Gambar 2.2 Anatomi dan Representasi Koordinat Bounding Box**

### 2.1.3 Deep Learning

*Deep learning *adalah cabang *machine learning* yang mempelajari representasi data secara bertingkat melalui banyak lapisan pemrosesan (Goodfellow et al., 2016). *Machine learning* konvensional bergantung pada fitur rancangan manusia, sedangkan *deep learning* memperoleh fitur secara otomatis dari data mentah (Han et al., 2017). Perbedaan alur kerja keduanya ditunjukkan pada Gambar 2.3.

[Gambar 2.3]

**Gambar 2.3 Perbedaan cara kerja Machine Learning dan Deep Learning**

Pelatihan jaringan saraf tiruan terdiri atas empat komponen utama.

1.  ***Forward Propagation***

Propagasi maju menghasilkan prediksi dari data masukan melalui operasi berlapis. Pada setiap lapisan, keluaran lapisan sebelumnya dikalikan dengan matriks bobot, ditambah bias, lalu dilewatkan ke fungsi aktivasi (Goodfellow et al., 2016).

2.  **Fungsi Aktivasi**

Fungsi aktivasi memberikan sifat non-linear pada jaringan sehingga model mampu memodelkan hubungan yang kompleks. Fungsi yang umum digunakan adalah *Rectified Linear Unit* (ReLU) pada Persamaan 2.1, karena ringan secara komputasi dan mengurangi masalah *vanishing gradient* (LeCun et al., 2015).

$$
f(x) = \max(0,x)
$$

(2.1)

dengan $x$ nilai masukan dan $f(x)$ nilai keluaran *neuron*.

3.  ***Loss Function***

Fungsi kerugian mengukur selisih antara prediksi model dan nilai sebenarnya (*ground truth*). Untuk klasifikasi banyak kelas, fungsi kerugian yang umum digunakan adalah *Cross-Entropy Loss* pada Persamaan 2.2 (Goodfellow et al., 2016).

$$
L = - \sum_{i = 1}^{C}y_{i}\log\left( p_{i} \right)
$$

(2.2)

dengan $C$ jumlah kelas, $y_{i}$ label sebenarnya kelas ke-*i* dalam bentuk *one-hot encoding*, dan $p_{i}$ probabilitas prediksi kelas ke-*i*.

4.  ***Back propagation***

Propagasi mundur menghitung gradien fungsi kerugian terhadap setiap parameter menggunakan aturan rantai (*chain rule*), lapisan demi lapisan dari keluaran menuju masukan (LeCun et al., 2015). Gradien terhadap parameter dipakai oleh *Stochastic Gradient Descent* (SGD) untuk memperbarui parameter, sebagaimana dinyatakan pada Persamaan 2.3 (Goodfellow et al., 2016).

$$
w \leftarrow w - \eta\frac{\partial L}{\partial w}
$$

(2.3)

dengan $w$ parameter bobot, $\eta$ laju pembelajaran (*learning rate*), dan $\frac{\partial L}{\partial w}$ gradien fungsi kerugian terhadap bobot.

Siklus propagasi maju dan mundur diulang hingga model konvergen, sebagaimana diilustrasikan pada Gambar 2.4.

[Gambar 2.4]

**Gambar 2.4 Diagram Alur Kerja Siklus Pelatihan Jaringan Saraf Tiruan**

### 2.1.4 Convolution Neural Network (CNN)

*Convolutional Neural Network *(CNN) adalah jaringan saraf tiruan yang dirancang untuk data berbentuk *grid*, seperti citra (Goodfellow et al., 2016; LeCun et al., 2015). Efisiensinya bertumpu pada konektivitas lokal, yaitu setiap neuron hanya melihat sebagian kecil citra, dan berbagi parameter, yaitu filter yang sama dipakai di seluruh citra (LeCun et al., 2015). Arsitektur umum CNN ditunjukkan pada Gambar 2.5.

[Gambar 2.5]

**Gambar 2.5 Ilustrasi Arsitektur Convolutional Neural Network (CNN)**

Pada Gambar 2.5, arsitektur CNN terdiri atas tiga jenis lapisan utama.

1.  ***Convolutional Layer***

*Layer* ini mengekstraksi fitur lokal dengan menggeser filter (*kernel*) terpelajar di atas citra atau peta fitur (*feature map*). Setiap filter mengenali pola tertentu seperti tepi, sudut, atau tekstur (Zhao et al., 2024; Goodfellow et al., 2016). Keluarannya dilewatkan ke fungsi aktivasi seperti ReLU pada Persamaan 2.1.

2.  ***Pooling Layer***

Berfungsi untuk memperkecil dimensi spasial *feature map* (*downsampling*), misalnya dengan *max pooling* yang mengambil nilai terbesar pada setiap jendela, sehingga beban komputasi berkurang dan model lebih tahan terhadap pergeseran kecil.

3.  ***Fully Connected Layer***

Berupa *Multi-Layer Perceptron* (MLP) yang menggabungkan fitur tingkat tinggi menjadi keputusan akhir (*classifier*). Masukannya berupa *flatten vector* dari *feature map*. Skor mentah (*logit*) keluaran kepala klasifikasi diubah menjadi probabilitas oleh fungsi *softmax* pada Persamaan 2.4.

$$
p_{i} = \frac{e^{z_{i}}}{\sum_{j = 1}^{C}e^{z_{j}}}
$$

(2.4)

dengan $p_{i}$ probabilitas kelas ke-*i*, $z_{i}$ *logit* kelas ke-*i*, dan $C$ jumlah kelas.

Lapisan konvolusi dan *pooling* menghasilkan *feature map* berukuran $C \times H \times W$. Setiap kanal merupakan respons satu filter, sedangkan posisi $\left( h,w \right)$ menyatakan lokasi respons tersebut (Goodfellow et al., 2016). *Stride* adalah besar langkah pergeseran filter. *Stride* total lapisan ke-*l* dan posisi citra yang berkorespondensi dengan sel $\left( h,w \right)$ dinyatakan pada Persamaan 2.5 dan 2.6 (Araujo et al., 2019).

$$
S_{l} = \prod_{i = 1}^{l}s_{i}
$$

(2.5)

$$
(u,v) = \left( S_{l} \cdot w,S_{l} \cdot h \right)
$$

(2.6)

dengan $S_{l}$ *stride* total lapisan ke-*l*, $s_{i}$ *stride* lapisan ke-*i*, $\left( h,w \right)$ indeks baris dan kolom pada *feature map*, dan $\left( u,v \right)$ koordinat pada citra.

Persamaan 2.6 baru menetapkan titik koordinat korespondensi, belum luas *region* citra yang benar-benar memengaruhi sel tersebut. Luas tersebut dinyatakan oleh *Receptive Field*, yaitu *region* citra yang memengaruhi nilai satu sel *feature map* (Goodfellow et al., 2016; Luo et al., 2016). Ukurannya membesar seiring kedalaman jaringan, sebagaimana dinyatakan pada Persamaan 2.7 (Araujo et al., 2019).

$$
r_{l} = r_{l - 1} + \left( k_{l} - 1 \right)\prod_{i = 1}^{l - 1}s_{i},\quad r_{0} = 1
$$

(2.7)

dengan $r_{l}$ ukuran *receptive field* lapisan ke-*l* dalam piksel, $k_{l}$ ukuran kernel lapisan ke-*l*, dan $s_{i}$ *stride* lapisan ke-*i*.

*Stride *total pada Persamaan 2.5 menentukan letak pusat daerah citra yang diwakili satu sel *feature map*, sedangkan *Receptive Field* pada Persamaan 2.7 menentukan luas daerah tersebut. Keduanya membuat setiap sel dapat dipetakan kembali ke *region* citranya sendiri, sehingga bobot atau atribusi pada sel *feature map* dapat diterjemahkan menjadi penjelasan spasial pada citra di Subbab 2.1.8. Namun, *Receptive Field* antarsel saling tumpang-tindih dan pengaruh piksel di dalamnya menurun dari pusat ke tepi (Luo et al., 2016), sehingga penjelasan tersebut berlaku pada tingkat *regional*, bukan pada piksel. Ilustrasi stride dan receptive field ditunjukkan pada Gambar 2.6.

[Gambar 2.6]

**Gambar 2.6 Ilustrasi *Stride* dan *Receptive Field***

### 2.1.5 *Faster* R-CNN

*Faster *R-CNN adalah arsitektur deteksi objek dua tahap (*two-stage*) yang menyatukan pengusulan area dan klasifikasi dalam satu jaringan yang dilatih secara *end-to-end* (Ren et al., 2017). Keunggulannya terletak pada *Region Proposal Network* (RPN) yang menggantikan metode pencarian area eksternal, sehingga seluruh komponen berbagi *feature map* yang sama. Arsitektur standarnya ditunjukkan pada Gambar 2.7.

[Gambar 2.7]

**Gambar 2.7 Arsitektur Faster R-CNN Standar**

Penelitian ini menggunakan varian SF-PSPyramid (Fung et al., 2024), yaitu *Faster* R-CNN dengan *neck* yang dirancang untuk cacat berukuran mikro pada PCB, sebagaimana ditunjukkan pada Gambar 2.8.

[Gambar 2.8]

**Gambar 2.8 Arsitektur modifikasi Faster R-CNN dengan SF-PSPyramid**

1.  **Backbone** 

*Backbone *mengubah citra menjadi *feature map*. Penelitian ini menggunakan ResNet-50, yang terdiri atas empat kelompok lapisan ($C_{2}$, $C_{3}$, $C_{4}$, $C_{5}$) dengan resolusi menurun dan makna semantik meningkat (He et al., 2016). *Stride* total keempat kelompok tersebut berturut-turut 4, 8, 16, dan 32 piksel.

2.  ***Neck (SF-PSPyramid)***

*Neck *menggabungkan fitur dari berbagai skala backbone menjadi piramida fitur. Dasarnya adalah *Feature Pyramid Network (FPN)*, yang menggabungkan jalur *bottom-up*, jalur *top-down*, dan koneksi lateral antartingkat (Lin et al., 2017). SF-PSPyramid menyempurnakan FPN dengan tiga perbedaan (Fung et al., 2024).

a.  ***CP Block***

 Resolusi diperbesar melalui penataan ulang kanal (*pixel shuffle*) (Shi et al., 2016), bukan interpolasi, sehingga bersifat terpelajar yang ditunjukkan pada Persamaan 2.8.

$$
PS(T)_{c,h,w} = T_{c \cdot r^{2} + r \cdot \text{mod}(h,r) + \text{mod}(w,r),\left\lfloor h/r \right\rfloor,\left\lfloor w/r \right\rfloor}
$$

(2.8)

dengan $T$ tensor masukan, $r$ faktor pembesaran, serta $c,h,w$ indeks kanal, baris, dan kolom keluaran.

b.  ***Selective Feature Attention***

Dua tingkat fitur digabungkan dengan bobot terpelajar yang dinormalisasi *softmax*, mengadaptasi *Selective Kernel Network* (Li et al., 2019), sebagaimana dinyatakan pada Persamaan 2.9.

$$
P' = \alpha_{1} \odot U + \alpha_{2} \odot V,\quad\left\lbrack \alpha_{1},\alpha_{2} \right\rbrack = \text{softmax}\left( W_{2}\delta\left( W_{1}z \right) \right)
$$

(2.9)

dengan $U$ *feature map* dari tingkat yang lebih dalam, $V$ *feature map* beresolusi lebih tinggi, $z$ vektor hasil *global average pooling*, $W_{1},W_{2}$ bobot lapisan kompresi dan perluasan, $\delta$ ReLU, dan $\odot_{}^{}{}$ perkalian per kanal.

c.  **Susunan piramida tanpa koneksi lateral**

SF-PSPyramid membentuk $P_{2}'$ dan $P_{3}'$ langsung dari gabungan $C_{2}$ hingga \$C_5\$ melalui CP Block pada Persamaan 2.8, bukan dari koneksi lateral seperti FPN standar, sehingga tetap beresolusi tinggi namun memuat informasi semantik penuh yang membantu deteksi cacat sangat kecil (Fung et al., 2024), sebagaimana ditunjukkan pada Gambar 2.8.

Keluaran neck adalah P2′, P3′, P4, P5, dan P6 (*max pooling* dari P5) dengan *stride* 4, 8, 16, 32, dan 64 piksel dan jumlah kanal *C* yang sama.

3.  ***Region Proposal Network* (RPN)**

RPN menghasilkan usulan area kandidat objek. Pada setiap posisi *feature map* disiapkan sejumlah *anchor box* dengan beragam ukuran dan rasio, lalu setiap *anchor* memperoleh skor objektivitas dan empat nilai penyesuaian koordinat (Ren et al., 2017) pada Persamaan 2.10 sampai 2.13.

$$
t_{x} = \frac{x - x_{a}}{w_{a}}
$$

(2.10)

$$
t_{y} = \frac{y - y_{a}}{h_{a}}
$$

(2.11)

$$
t_{w} = \log\left( \frac{w}{w_{a}} \right)
$$

(2.12)

$$
t_{h} = \log\left( \frac{h}{h_{a}} \right)
$$

(2.13)

dengan $x,y,w,h$ pusat, lebar, dan tinggi kotak prediksi, serta $x_{a},y_{a},w_{a},h_{a}$ milik *anchor*. Pelatihan memakai *multi-task loss* pada Persamaan 2.14 (Ren et al., 2017).

$$
L = \frac{1}{N_{cls}}\sum_{i}^{}L_{cls}\left( p_{i},p_{i}^{*} \right) + \lambda\frac{1}{N_{reg}}\sum_{i}^{}p_{i}^{*}L_{reg}\left( t_{i},t_{i}^{*} \right)
$$

(2.14)

dengan $p_{i}$ probabilitas *anchor* ke-*i* memuat objek, $p_{i}^{*}$ label sebenarnya (1 positif, 0 negatif), $N_{cls},N_{reg}$ jumlah sampel tiap suku, $L_{cls}$ *binary cross-entropy*, $L_{reg}$ kerugian regresi, dan $\lambda$ faktor penyeimbang. Kerugian regresi yang digunakan adalah L1 pada Persamaan 2.15 (Fung et al., 2024).

$$
L_{reg}\left( t,t^{*} \right) = \sum_{j}^{}\left| t_{j} - t_{j}^{*} \right|
$$

(2.15)

dengan *j* merentang pada keempat komponen koordinat, yaitu *x*, *y*, *w*, dan *h*.

4.  ***RoI Align***

*RoI Align *menyeragamkan setiap proposal menjadi *tensor* berukuran tetap tanpa pembulatan koordinat (He et al., 2020). Nilai *feature map* pada koordinat pecahan diperoleh dengan interpolasi *bilinear* pada Persamaan 2.16.

$$
f(x,y) = \sum_{i = 1}^{4}w_{i}f_{i}
$$

(2.16)

dengan $f_{i}$ nilai fitur pada empat titik grid terdekat dan $w_{i}$ bobot interpolasi yang berbanding terbalik dengan jarak.

Setiap proposal dibagi menjadi grid $G \times G$ *bin*. Pada setiap *bin* diambil beberapa titik sampel, dan nilai *bin* adalah rata-rata titik sampelnya (Persamaan 2.17) (He et al., 2020).

$$
x_{c,p,q} = \frac{1}{N}\sum_{n = 1}^{N}{\sum_{i = 1}^{4}{w_{n,i}\, F_{c}\left( h_{n,i},v_{n,i} \right)}}
$$

(2.17)

dengan $x_{c,p,q}$ keluaran kanal ke-*c* pada *bin* $\left( p,q \right)$, $N$ jumlah titik sampel per *bin*, $w_{n,i}$ bobot interpolasi titik sampel ke-*n*, dan $F_{c}\left( h_{n,i},v_{n,i} \right)$ nilai *feature map* kanal ke-*c* pada titik grid terdekat. Dalam bentuk matriks, Persamaan 2.17 dapat ditulis sebagai Persamaan 2.18.

$$
x_{c} = A\, F_{c}
$$

(2.18)

dengan $x_{c}$ vektor keluaran kanal ke-*c* ($G \times G$ nilai), $F_{c}$ vektor nilai *feature map* kanal ke-*c*, dan $A$ matriks koefisien interpolasi yang hanya bergantung pada proposal dan sama untuk setiap kanal.

Pada piramida fitur, tingkat yang dipakai untuk setiap proposal dipilih dengan Persamaan 2.19 (Lin et al., 2017).

$$
k = \left\lfloor k_{0} + \log_{2}\left( \frac{\sqrt{wh}}{224} \right) \right\rfloor
$$

(2.19)

dengan $k$ tingkat terpilih, $k_{0} = 4$ tingkat acuan untuk proposal 224×224 piksel, serta $w,h$ lebar dan tinggi proposal.

5.  ***Box Head***

Keluaran *RoI Align* setiap proposal diratakan menjadi vektor, lalu diproses oleh *Box Head* yang berupa MLP dengan dua lapisan terhubung penuh yang menghasilkan representasi RoI (Girshick, 2015; Ren et al., 2017).

Representasi RoI diteruskan ke dua lapisan keluaran yang bekerja berdampingan (*two sibling output layers*) (Girshick, 2015; Ren et al., 2017). Cabang pertama adalah *classifier* yang menghasilkan skor untuk $C + 1$ kelas (termasuk background) melalui *softmax* pada Persamaan 2.4 dan dilatih dengan *cross-entropy* pada Persamaan 2.2. Cabang kedua adalah *regressor* yang menghasilkan empat nilai koreksi koordinat untuk setiap kelas pada Persamaan 2.10 sampai 2.13 dan dilatih dengan kerugian L1 (Persamaan 2.15).

6.  ***Soft-*NMS**

Kotak dan skor keluaran *box head* kemudian disaring dengan *Soft*-NMS, yang menurunkan skor kandidat yang tumpang-tindih secara bertahap, bukan menghapusnya seketika seperti NMS konvensional (Bodla et al., 2017), sehingga cacat yang berdekatan tidak ikut terbuang dengan menggunakan Persamaan 2.20.

$$
s_{i} = \left\{ \begin{matrix}
s_{i}, & IoU\left( M,b_{i} \right) < N_{t} \\
s_{i}\left( 1 - IoU\left( M,b_{i} \right) \right), & IoU\left( M,b_{i} \right) \geq N_{t}
\end{matrix} \right.\
$$

(2.20)

dengan $s_{i}$ dan $b_{i}$ skor dan kotak kandidat ke-*i*, $M$ kandidat berskor tertinggi, dan $N_{t}$ ambang IoU.

### 2.1.6 *Explainable Artificial Intelligence* (XAI)

*Explainable Artificial Intelligence *(XAI) adalah bidang yang mengembangkan cara agar keputusan model kecerdasan buatan dapat dipahami manusia (Barredo Arrieta et al., 2020). XAI dibutuhkan karena model *deep learning* bersifat *black-box* (Samek et al., 2019). Berdasarkan waktu penjelasan dibentuk, XAI terbagi atas dua pendekatan.

**1. Pendekatan *post-hoc***

Pendekatan yang menghasilkan penjelasan setelah model memprediksi tanpa mengubah struktur model, misalnya LIME, SHAP, dan Grad-CAM. Penjelasannya bersifat aproksimasi dan tidak dijamin mencerminkan keputusan internal model (Guidotti et al., 2018; Rudin, 2019).

**2. Pendekatan *ante-hoc***

Pendekatan yang menanamkan kemampuan menjelaskan ke dalam struktur model, misalnya pohon keputusan, model linear, dan arsitektur *neuro-symbolic* (Barredo Arrieta et al., 2020).

Penjelasan untuk citra umumnya berupa peta atribusi (*attribution map*), yaitu peta besar kontribusi setiap posisi terhadap skor kelas (Bach et al., 2015). *Attribution map* yang divisualisasikan dengan skala warna di atas citra disebut *heatmap*. Dua metode atribusi yang relevan dijelaskan berikut.

1.  ***Gradient-weighted Class Activation Mapping* (Grad-CAM)**

Grad-CAM menghasilkan *attribution map* spesifik-kelas tanpa mengubah maupun melatih ulang model (Selvaraju et al., 2017). Bobot kepentingan *feature map* ke-*k* terhadap kelas *c* dan *attribution map* dinyatakan pada Persamaan 2.21 dan 2.22.

$$
\alpha_{k}^{c} = \frac{1}{Z}\sum_{i}^{}{\sum_{j}^{}\frac{\partial y^{c}}{\partial A_{ij}^{k}}}
$$

(2.21)

$$
L_{Grad\text{-}CAM}^{c} = ReLU\left( \sum_{k}^{}{\alpha_{k}^{c}A^{k}} \right)
$$

(2.22)

dengan $y^{c}$ skor kelas *c* sebelum *softmax*, $A^{k}$ feature map ke-*k* pada lapisan konvolusi acuan, $A_{ij}^{k}$ nilainya pada posisi $\left( i,j \right)$, dan $Z$ jumlah posisi spasial. Contoh hasil Grad-CAM ditunjukkan pada Gambar 2.9.

[Gambar 2.9]

**Gambar 2.9 Hasil penjelasan dari Grad-CAM (Selvaraju et al., 2017).**

2.  **Gradient *×* Input**

*Gradient × Input* menghitung kontribusi setiap elemen masukan sebagai hasil kali nilai elemen dan gradien keluaran terhadap elemen tersebut (Shrikumar et al., 2017; Ancona et al., 2019), sebagaimana dinyatakan pada Persamaan 2.23.

$$
R_{j}(x) = x_{j} \cdot \frac{\partial f(x)}{\partial x_{j}}
$$

(2.23)

dengan $R_{j}$ kontribusi elemen ke-*j*, $x_{j}$ nilai elemen ke-*j*, dan $f(x)$ keluaran model.

Metode atribusi yang baik diharapkan memenuhi sifat *completeness*, yaitu jumlah seluruh atribusi sama dengan selisih keluaran model pada masukan dan pada masukan acuan (*baseline*) $\overline{x}$ (Sundararajan et al., 2017), sebagaimana dinyatakan pada Persamaan 2.24.

$$
\sum_{j}^{}{R_{j}(x)} = f(x) - f\left( \overline{x} \right)
$$

(2.24)

*Gradient × Input* umumnya tidak memenuhi sifat ini pada model non-linear, tetapi memenuhinya secara eksak pada model linear dengan *baseline* nol (Ancona et al., 2019), sebagaimana dinyatakan pada Persamaan 2.25.

$$
\sum_{j}^{}{w_{j}x_{j}} = w^{T}x = f(x) - f(0)
$$

(2.25)

Dengan demikian, *Gradient × Input* merupakan dekomposisi eksak, bukan aproksimasi.

### 2.1.7 *Neuro-Symbolic* AI (NeSy)

*Neuro-Symbolic* AI (NeSy) menggabungkan pembelajaran statistik jaringan saraf dengan penalaran berbasis aturan pada sistem simbolik (d'Avila Garcez & Lamb, 2023), untuk menyatukan kemampuan mengenali pola dari data mentah dengan transparansi penalaran (Kautz, 2022). Arsitektur NeSy ditunjukkan pada Gambar 2.10.

[Gambar 2.10]

**Gambar 2.10 Arsitektur Neuro-Symbolic**

Arsitektur NeSy terdiri atas dua komponen utama.

**1. Komponen *neural*,** berfungsi untuk mengubah data mentah menjadi representasi fitur numerik (d'Avila Garcez & Lamb, 2023).

**2. Komponen simbolik,** mengolah representasi fitur melalui aturan yang dapat ditelusuri. Penjelasannya *faithful*, yaitu merupakan proses keputusan model itu sendiri, bukan aproksimasi (Rudin, 2019). Komponen ini dapat dibentuk melalui *model mimicking*, yaitu melatih model sederhana untuk meniru keluaran model kompleks (*teacher*), dengan label pelatihan berupa prediksi *teacher*, bukan label sebenarnya (Buciluǎ et al., 2006; Hinton et al., 2015).

Pengalihan keputusan dari komponen *neural* ke komponen simbolik membawa konsekuensi pada bentuk keluarannya. Keluaran komponen *neural* bersifat kontinu, sedangkan keluaran komponen simbolik bersifat diskrit karena hanya berupa label kelas. Proses diskritisasi ini memetakan banyak nilai kontinu yang berbeda ke label yang sama, sehingga informasi tingkat keyakinan turut hilang (Provost & Domingos, 2003). Informasi tersebut masih tersimpan pada nilai keputusan sebelum diubah menjadi label, dan dapat dinyatakan melalui dua konsep berikut.

1.  *Margin*

Pada fungsi keputusan bernilai riil, tanda keluaran menyatakan label, sedangkan besarnya menyatakan keyakinan. Keluaran yang dekat dengan nol berarti keyakinan rendah, dan yang jauh dari nol berarti keyakinan tinggi (Schapire & Singer, 1999). Besar *margin* dinyatakan pada Persamaan 2.26.

$$
m(x) = \left| f(x) \right|
$$

(2.26)

dengan $f(x)$ fungsi keputusan linear.

2.  Fungsi *Sigmoid*

Fungsi *sigmoid* adalah fungsi monoton naik yang memetakan nilai riil ke selang 0 dan 1 (Platt, 1999), sebagaimana dinyatakan pada Persamaan 2.27.

$$
\sigma(z) = \frac{1}{1 + e^{- z}}
$$

(2.27)

dengan $z$ nilai masukan dan $e$ bilangan Euler.

### 2.1.8 *Sparse Oblique Decision Tree* (SODT)

*Sparse Oblique Decision Tree *(SODT) adalah pohon keputusan yang melakukan pemisahan linear multivariat (*oblique split*) pada setiap *node* internal, berbeda dengan pohon *axis-aligned* yang hanya memakai satu fitur per pemisahan (Hada et al., 2024). Setiap *node* internal meneruskan masukan ke salah satu dari dua anaknya, dan label pada *leaf*  yang dicapai menjadi prediksi pohon. Fungsi keputusan *node* internal ke-*i* dinyatakan pada Persamaan 2.28.

$$
w_{i}^{T}x + b_{i} \geq 0
$$

(2.28)

dengan $x$ vektor fitur berdimensi $D$, $w_{i}$ vektor bobot, dan $b_{i}$ bias *node* ke-*i*.

Arah percabangan ditentukan oleh tanda $f_{i}(x)$, dinotasikan $d_{i} = + 1$ untuk ke anak kiri dan $d_{i} = - 1$ untuk ke anak kanan. Rangkaian *node* dari *root* hingga *leaf* membentuk jalur keputusan (*decision path*) (Hada et al., 2024; Kairgeldin & Carreira-Perpiñán, 2025). Ilustrasinya ditunjukkan pada Gambar 2.11.

[Gambar 2.11]

**Gambar 2.11 Ilustrasi SODT dan perbedaan dengan *Decision Tree* biasa**

Apabila $x$ berasal dari perataan peta fitur $C \times H \times W$ (Subbab 2.1.4), setiap elemen $x$ berkorespondensi satu-satu dengan satu kanal dan posisi spasial tertentu pada peta fitur. Korespondensi yang sama berlaku pada bobot $w_{i}$ karena dikalikan pada indeks yang sama (Persamaan 2.28), sehingga vektor $w_{i}$ dapat disusun ulang ke bentuk grid $C \times H \times W$, yang menyatakan kanal dan posisi tertentu yang dipakai *node* (Hada et al., 2024; Kairgeldin & Carreira-Perpiñán, 2025).

1.  **Regularisasi L1 dan Sparsitas**

Sifat *sparse* diperoleh melalui regularisasi L1 pada fungsi tujuan pelatihan pada Persamaan 2.29, yang membuat bobot fitur tidak relevan bernilai tepat nol (Hada et al., 2024).

$$
E(\Theta) = \sum_{n = 1}^{N}L\left( \mathbf{y}_{n},T\left( \mathbf{x}_{n};\Theta \right) \right) + \lambda\sum_{i \in \mathcal{D}}^{}\left\| \mathbf{w}_{i} \right\|_{1}
$$

(2.29)

dengan $\Theta$ parameter pohon, $N$ jumlah sampel, $x_{n},y_{n}$ fitur dan label sampel ke-*n*, $T\left( x_{n};\Theta \right)$ prediksi pohon, $L$ fungsi kerugian klasifikasi, $\mathcal{D}$ himpunan *node* internal, dan $\lambda$ pengontrol sparsitas. *Node* yang seluruh bobotnya nol hanya ditentukan oleh bias, sehingga selalu mengarahkan masukan ke sisi yang sama. Bobot yang tidak bernilai nol disebut *nonzero weight*, dan jumlahnya digunakan sebagai ukuran kompleksitas pohon, yaitu banyaknya fitur yang benar-benar dipakai dalam keputusan (Hada et al., 2024). Sparsitas pohon dinyatakan sebagai proporsi bobot yang bernilai nol terhadap seluruh bobot *node* internal.

2.  ***Tree Alternating Optimization (TAO)***

Persamaan 2.29 tidak dapat dioptimasi dengan metode berbasis gradien karena keputusan pohon diskrit, dan tidak didukung oleh metode pembentukan pohon konvensional. *Tree Alternating Optimization* (TAO) memecahnya menjadi masalah klasifikasi biner yang diselesaikan terpisah untuk setiap *node* (Carreira-Perpiñán & Tavallali, 2018), sebagaimana dinyatakan pada Persamaan 2.30.

$$
E_{i}(w_{i},b_{i}) = \sum_{n \in \mathcal{R}_{i}}^{}\overset{‾}{L}({\bar{y}}_{n},g_{i}(x_{n};w_{i},b_{i})) + \lambda \parallel w_{i} \parallel_{1}
$$

(2.30)

dengan $\mathcal{R}_{i}$ sampel yang mencapai *node* ke-*i* (*reduced set*), ${\overline{y}}_{n}$ label semu (*pseudo-label*) arah kiri atau kanan, $g_{i}$ keputusan biner *node*, dan $\overline{L}$ kerugian 0/1.

Tidak semua sampel dalam $\mathcal{R}_{i}$ berpengaruh, sampel yang prediksinya sama pada kedua arah diabaikan, sedangkan sisanya (*care set*) memperoleh label semu berupa arah yang menghasilkan prediksi benar. Masalah biner tersebut diselesaikan dengan regresi logistik berregularisasi L1 sebagai pengganti (*surrogate*) kerugian 0/1, dan setiap *leaf* berlabel kelas mayoritas sampel yang mencapainya. Penyelesaian ini dilakukan bergantian antar-*node* dari yang terdalam menuju *root*. Selama pembaruan suatu *node* hanya diterima apabila memperbaiki masalah tereduksinya, nilai fungsi tujuan tidak pernah naik, sehingga TAO dijamin menurunkan fungsi tujuan secara monoton (Carreira-Perpiñán & Tavallali, 2018; Hada et al., 2024; Kairgeldin & Carreira-Perpiñán, 2025).

Karena penalti $\lambda$ sama untuk semua *node*, *node* dengan banyak data cenderung kurang *sparse*. Kairgeldin dan Carreira-Perpiñán (2025) mengatasinya dengan membobot penalti berdasarkan jumlah sampel *node* melalui parameter $\alpha$, sebagaimana dinyatakan pada Persamaan 2.31 dan 2.32.

$$
E(\Theta) = \sum_{n = 1}^{N}{L(}\mathbf{y}_{n},T(\mathbf{x}_{n};\Theta)) + \lambda\sum_{i \in \mathcal{D}}^{}h_{\alpha}( \mid \mathcal{R}_{i} \mid ) \parallel \mathbf{w}_{i} \parallel_{1}
$$

(2.31)

$$
h_{\alpha}(t) = \left\{ \begin{matrix}
1, & t = 0 \\
t^{\alpha}, & t > 0
\end{matrix} \right.\
$$

(2.32)

dengan $\left. \mid\mathcal{R}_{i} \right.\mid$ jumlah sampel yang mencapai *node* ke-*i*. Nilai $\alpha > 0$ memperbesar penalti pada *node* yang menangani banyak data dibandingkan TAO standar (α = 0), sedangkan α = 1 menyamakan penalti per sampel antar-*node* (Kairgeldin & Carreira-Perpiñán, 2025).

3.  ***Class Weighting***

Pada *model mimicking* (Subbab 2.1.7), label pelatihan SODT adalah prediksi *teacher*. Apabila sebagian kelas jauh lebih jarang, pohon cenderung mengabaikannya. *Cost-sensitive learning* mengatasinya dengan memberi biaya kesalahan yang berbeda antarkelas (Elkan, 2001; He & Garcia, 2009), sebagaimana dinyatakan pada Persamaan 2.33.

$$
E_{\omega}(\Theta) = \sum_{n = 1}^{N}{\omega_{y_{n}}L\left( y_{n},T\left( x_{n};\Theta \right) \right)}
$$

(2.33)

dengan $\omega_{y_{n}} \geq 0$ biaya kesalahan kelas $y_{n}$.

5.  ***Explanation* melalui *Weight Node***

Setiap *node* bersifat linear dan *sparse*, sehingga bobot $w_{i}$ langsung menunjukkan fitur yang dipakai *node*. Bobot nol berarti fitur tidak dipakai, tandanya menyatakan arah dorongan, dan besarnya menyatakan kekuatan pengaruh. Hada et al. (2024) memvisualisasikan bobot ini untuk menelusuri fitur yang memisahkan antarkelas.

Kairgeldin dan Carreira-Perpiñán (2025) memperluasnya pada model hibrida CNN dan SODT. Karena setiap fitur berasal dari sel *feature map* CNN yang memiliki *Receptive Field* (Subbab 2.1.4), mereka menyusun peta kepadatan *Receptive Field* (RF *density map*) untuk setiap *node*, sebagaimana dinyatakan pada Persamaan 2.34.

$$
D_{i}(u) = \sum_{j = 1}^{D}{\left| w_{ij} \right| \cdot \mathbb{1}\left\lbrack u \in {RF}_{j} \right\rbrack}
$$

(2.34)

dengan $D_{i}(u)$ kepadatan pada posisi citra $u$ untuk *node* ke-*i*, $w_{ij}$ bobot fitur ke-*j*, ${RF}_{j}$ *Receptive Field* fitur ke-*j*, dan $\mathbb{1}\lbrack \cdot \rbrack$ fungsi indikator. Kepadatan nol berarti daerah tersebut tidak dipakai *node* (Kairgeldin & Carreira-Perpiñán, 2025). Peta ini disusun per *node* dan hanya bergantung pada bobot, sehingga bersifat statis.

### 2.1.9 Metrik Evaluasi

Evaluasi dibagi menjadi evaluasi kinerja deteksi objek dan evaluasi kualitas penjelasan (*explainability*).

1.  **Metrik Evaluasi Deteksi Objek**

*Intersection over Union* (IoU) mengukur rasio luas irisan terhadap luas gabungan *bounding box* prediksi dan *ground truth* pada Persamaan 2.35. Berdasarkan ambang IoU, prediksi dikelompokkan menjadi *True Positive* (TP), *False Positive* (FP), dan *False Negative* (FN), yang menjadi dasar *Precision* dan *Recall* yang didefiniskan pada Persamaan 2.36 dan 2.37.

$$
IoU = \frac{\text{Area~Prediksi} \cap \text{Area~Ground~Truth}}{\text{Area~Prediksi} \cup \text{Area~Ground~Truth}}
$$

(2.35)

$$
\text{Precision} = \frac{TP}{TP + FP}\text{ }
$$

(2.36)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

(2.37)

*Precision* mengukur ketepatan prediksi positif, sedangkan *Recall* mengukur kelengkapan deteksi. F1-*score* adalah rata-rata harmonik *Precision* dan *Recall* (Persamaan 2.38) (Sokolova & Lapalme, 2009).

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

(2.38)

*Average Precision* (AP) adalah luas di bawah kurva *Precision-Recall*pada Persamaan 2.39, dan *mean Average Precision* (mAP) adalah rata-rata AP seluruh kelas pada Persamaan 2.40 (Everingham et al., 2010).

$$
AP = \int_{0}^{1}{Precision(Recall)\, d(Recall)}
$$

(2.39)

$$
mAP = \frac{1}{K}\sum_{i = 1}^{K}{AP}_{i}
$$

(2.40)

dengan $K$ jumlah kelas dan ${AP}_{i}$ AP kelas ke-*i*. mAP@0,5 dihitung pada ambang IoU 0,5, sedangkan mAP@0,5:0,95 merupakan rata-rata mAP pada ambang IoU 0,5 hingga 0,95 dengan kenaikan 0,05 (Lin et al., 2014).

2.  **Metrik Evaluasi *Explainability***

Evaluasi XAI dapat melibatkan pengguna (*application-grounded* dan *human-grounded*) atau tanpa pengguna melalui ukuran kuantitatif (*functionally-grounded*) (Doshi-Velez & Kim, 2017; Nauta et al., 2023). Penelitian ini berfokus pada evaluasi *functionally-grounded*, khususnya *faithfulness* dan *localization*.

a.  *Faithfulness*

*Faithfulness* umumnya diuji melalui perturbasi, yaitu menghapus (*masking*) atau mempertahankan bagian masukan yang disorot penjelasan lalu mengamati perubahan keluaran model (Petsiuk et al., 2018).

*Necessity* (DeYoung et al., 2020; Canha et al., 2025) menguji apakah fitur yang disorot memang diperlukan, antara lain fitur dihapus, lalu diperiksa apakah prediksi berubah sebagaimana didefinisikan pada Persamaan 2.41. *Sufficiency* menguji apakah fitur yang disorot sudah cukup: hanya fitur tersebut yang dipertahankan, lalu diperiksa apakah prediksi tetap sehingga dapat definisikan dengan Persamaan 2.42.

$$
\text{Necessity~Flip~Rate} = \frac{1}{N}\sum_{n = 1}^{N}{\mathbb{1}\left\lbrack \widehat{y}\left( x_{n}^{- S} \right) \neq \widehat{y}\left( x_{n} \right) \right\rbrack}
$$

(2.41)

$$
\text{Sufficiency~Preservation} = \frac{1}{N}\sum_{n = 1}^{N}{\mathbb{1}\left\lbrack \widehat{y}\left( x_{n}^{S} \right) = \widehat{y}\left( x_{n} \right) \right\rbrack}
$$

(2.42)

dengan $S$ himpunan fitur atau daerah yang disorot paling penting oleh penjelasan, $x_{n}^{- S}$ masukan tanpa $S$ (dinolkan), $x_{n}^{S}$ masukan yang hanya mempertahankan $S$, $\widehat{y}$ keluaran model, dan $\mathbb{1}\lbrack \cdot \rbrack$ fungsi indikator. Nilai yang tinggi pada kedua metrik menandakan penjelasan yang *faithful*.

*Faithfulness* juga dapat diukur secara bertahap dengan *Deletion* AUC dan *Insertion* AUC (Petsiuk et al., 2018), yaitu luas di bawah kurva skor seiring fitur dihapus atau ditambahkan dalam $K$ tahap seperti pada Persamaan 2.43.

$$
AUC = \sum_{k = 1}^{K - 1}{\frac{s_{k} + s_{k + 1}}{2} \cdot \Delta x_{k}}
$$

(2.43)

dengan $s_{k}$ skor ternormalisasi pada tahap ke-*k* dan $\Delta x_{k}$ proporsi area yang dimodifikasi antartahap.

Pada *Deletion*, fitur terpenting dihapus lebih dulu, sehingga AUC rendah menandakan *necessity* yang baik. Pada *Insertion*, fitur terpenting ditambahkan lebih dulu, sehingga AUC tinggi menandakan *sufficiency* yang baik.

b.  *Localization*

*Localization* mengukur kesesuaian area yang disorot *heatmap* dengan lokasi objek. *Pointing Game* menghitung proporsi kasus ketika titik tertinggi *heatmap* jatuh di dalam *bounding box ground truth* seperti pada Persamaan 2.44 (Zhang et al., 2018), sedangkan IoU *Heatmap* mengukur IoU antara *heatmap* terbinerisasi dan *bounding box ground truth* (Zhou et al., 2016) yang didefinisikan pada Persamaan 2.45.

$$
{Acc}_{PG} = \frac{Hits}{Hits + Misses}
$$

(2.44)

$$
{Acc}_{PG} = \frac{Hits}{Hits + Misses}
$$

(2.44)

$$
{IoU}_{Heatmap} = \frac{\left| H_{bin} \cap B_{gt} \right|}{\left| H_{bin} \cup B_{gt} \right|}
$$

(2.45)

dengan $Hits$ dan $Misses$ jumlah kasus yang titik maksimumnya berada di dalam dan di luar *ground truth*, $H_{bin}$ *heatmap* yang hanya mempertahankan sejumlah posisi bernilai tertinggi, dan $B_{gt}$ daerah *ground truth*.

## 2.2 Penelitian Terkait

**Tabel 2.2 Ringkasan dan Perbandingan Penelitian Terkait**

| **No.** | **Judul / Penulis** | **Masalah** | **Tujuan** | **Metode** | **Hasil** | **Keterkaitan** |
|:--:|--------------|------------|-------------|------------|----------|---------|
| 1\. | *Improving PCB defect detection using selective feature attention and pixel shuffle pyramid (Fung et al., 2024)* | Deteksi cacat mikroskopis pada sirkuit PCB memiliki tingkat *false negative* yang tinggi pada model deteksi standar. | Meningkatkan kemampuan model melokalisasi target berukuran kecil pada citra PCB. | Faster R-CNN dengan *Feature Pyramid Network* (FPN), *Pixel Shuffle Pyramid* (PSPyramid), *Selective Feature Attention,* dan *Soft-NMS*. | Terjadi peningkatan *Mean Average Precision* (mAP) yang signifikan pada pengujian dataset DeepPCB. | Penelitian ini menjadi referensi utama arsitektur dasar (*baseline*) komponen Neuro (ekstraktor fitur dan lokalisasi) yang digunakan dalam tugas akhir ini. |
| 2\. | *Faster-LTN: a neuro-symbolic, end-to-end object detection architecture (Manigrasso et al., 2021)* | Model *deep learning* konvensional tidak mampu mengintegrasikan pengetahuan relasional dan penalaran logis ke dalam proses deteksi objek, sehingga kurang transparan dalam pengambilan keputusan. | Mengintegrasikan kemampuan penalaran logis dengan jaringan saraf konvolusional ke dalam arsitektur deteksi objek *end-to-end* untuk meningkatkan transparansi. | Faster R-CNN dengan penggantian kepala klasifikasi menjadi *Logic Tensor Networks* (LTN). | Arsitektur *end-to-end* berhasil dilatih dan mencapai performa kompetitif pada dataset PASCAL VOC. | Memberikan landasan konseptual integrasi pendekatan Neuro-Symbolic ke dalam arsitektur deteksi objek dua tahap (Faster R-CNN). |
| 3\. | *Sparse oblique decision trees: a tool to understand and manipulate neural net features (Hada et al., 2024)* | Fitur internal jaringan *deep learning* sulit diinterpretasikan sehingga tidak diketahui fitur mana yang menentukan suatu kelas (*black-box*). | Memahami dan memanipulasi fitur internal jaringan saraf dengan meniru (*mimic*) bagian *classifier*-nya menggunakan pohon keputusan yang akurat sekaligus *interpretable*. | *Sparse Oblique Decision Tree* (SODT) dengan regularisasi L1 yang dilatih menggunakan TAO untuk meniru *classifier* jaringan saraf. | Menghasilkan pohon yang ramping dengan akurasi mendekati jaringan saraf yang ditiru, serta mengungkap subset fitur yang menentukan kelas tertentu. | Menjadi landasan teoritis komponen **Symbolic** untuk menggantikan MLP pada *RoI head* jaringan Faster R-CNN. |
| 4\. | *Neurosymbolic models based on hybrids of convolutional neural networks and decision trees (Kairgeldin & Carreira-Perpiñán, 2025)* | Model *end-to-end* berbasis jaringan saraf tidak menyediakan penalaran yang dapat ditelusuri, sedangkan sparsitas SODT hasil TAO standar tidak merata antar-*node* (*node* dekat *root* kurang *sparse*). | Membangun model *neurosymbolic* berupa lapisan CNN yang dikomposisikan dengan SODT, serta mengatur distribusi sparsitas antar-*node*. | CNN (LeNet) dan SODT yang dilatih dengan TAO termodifikasi (hyperparameter α untuk membobot penalti L1 berdasarkan jumlah sampel *node*), serta *RF density map* per *node*. | SODT yang lebih *sparse* dengan akurasi kompetitif; sekelompok kecil neuron menentukan kelas tertentu dan *receptive field*-nya terpusat pada daerah citra yang diskriminatif. | Menjadi dasar arsitektur hibrida CNN–SODT, pembobotan penalti L1 (Persamaan 2.31 dan 2.32), dan pembanding heatmap per *node* (*RF density map*). |
| 5\. | *Explainable Predictive Quality Inspection using Deep Learning in Electronics Manufacturing *(Saadallah et al., 2022) | Model deep learning untuk prediksi kualitas bersifat *black-box* sehingga menyulitkan teknisi memahami fitur mana yang paling berpengaruh terhadap keputusan prediksi. | Menyediakan penjelasan visual atas prediksi kualitas PCB menggunakan *heatmap* untuk membantu teknisi mengidentifikasi fitur global (kuantitas fisik SPI) dan lokal (pin) yang paling menentukan. | *1D-CNN untuk prediksi kualitas biner (OK/NOK) dan Grad-CAM untuk menghasilkan peta panas penjelasan.* | Grad-CAM berhasil menyoroti fitur SPI (DX, DY, DVolume) dan pin spesifik yang paling diskriminatif untuk kelas "NOK", membantu teknisi melacak penyebab deviasi kualitas. | Menunjukkan aplikasi Grad-CAM sebagai metode *post-hoc* untuk inspeksi kualitas PCB berbasis data SPI. |
| 6\. | *Explainable AI Methods for Identification of Glue Volume Deficiencies in Printed Circuit Boards *(Tziolas et al., 2025) | Inspeksi volume lem pada PCB sulit dilakukan secara manual dan model *deep learning* yang digunakan tidak memberikan penjelasan atas deteksi defisiensi. | Mengidentifikasi defisiensi volume lem pada PCB menggunakan model *deep learning* dan menyediakan penjelasan visual atas prediksi model. | CenterNet-MobileNetV2 untuk lokalisasi PCB dan CNN *custom* (GlueVolNet) untuk klasifikasi volume lem ke dalam tiga kelas*, dengan Grad-CAM dan Deep SHAP untuk menghasilkan peta panas penjelasan.* | GlueVolNet mencapai akurasi 92,2% dalam mengklasifikasikan volume lem, dan Grad-CAM/Deep SHAP berhasil menyoroti area dengan volume lem tidak memadai yang menjadi dasar keputusan model. | Memperkuat justifikasi penggunaan Grad-CAM sebagai metode *post-hoc* yang telah teruji dalam inspeksi visual PCB, serta menunjukkan keterbatasan *post-hoc* yang mendorong kebutuhan pendekatan *faithful*. |
| 7\. | *Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging (Arun et al., 2021)* | Peta *saliency* banyak dipakai untuk menjelaskan dan melokalisasi keputusan CNN pada domain berisiko tinggi, tetapi keandalannya belum teruji secara sistematis. | Mengevaluasi keandalan (*trustworthiness*) peta *saliency* untuk lokalisasi kelainan pada citra medis. | Delapan metode *saliency*, termasuk Grad-CAM, diuji pada dua *dataset* radiologi berdasarkan utilitas lokalisasi, sensitivitas terhadap pengacakan bobot model, *repeatability*, dan *reproducibility*, lalu dibandingkan dengan jaringan lokalisasi (U-Net dan RetinaNet). | *Seluruh metode gagal pada minimal satu kriteria dan kalah dari jaringan lokalisasi. Grad-CAM lolos uji pengacakan bobot,* tetapi AUPRC lokalisasi seluruh metode *saliency* (0,160–0,519) tetap di bawah RetinaNet (0,596) pada deteksi pneumonia, begitu pula (0,024–0,224) terhadap U-Net (0,404) pada segmentasi pneumothorax. | Menunjukkan bahwa penjelasan *post-hoc*, termasuk Grad-CAM, belum dapat diandalkan pada domain berisiko tinggi, serta menjadi dasar kontrol pengacakan bobot yang juga digunakan dalam penelitian ini. |

# BAB III METODOLOGI PENELITIAN

Penelitian ini membangun, mengintegrasikan, dan mengevaluasi sistem deteksi cacat PCB berbasis arsitektur *neuro-symbolic*. Alur penelitian ditunjukkan pada Gambar 3.1.

[Gambar 3.1]

**Gambar 3.1 Diagram Alur Penelitian**

Penelitian dimulai dengan persiapan dan pra-pemrosesan *dataset* DeepPCB, dilanjutkan pelatihan *Faster* R-CNN, ekstraksi *dataset* simbolik, pelatihan SODT, integrasi keduanya, lalu evaluasi akhir.

## 3.1 Persiapan Dataset

Dataset DeepPCB (Tang et al., 2019) dibagi menjadi data latih dan data uji berdasarkan berkas indeks bawaannya yang ditunjukkan pada Tabel 3.1, sehingga tidak ada kebocoran data antarfase.

**Tabel 3.1 Pembagian dataset berdasarkan kategori**

| **Kategori** | **Berkas Indeks** | **Jumlah Citra** | **Persentase (%)** |
|:-------------------:|------------|----------------|-----------------|
| Data Pelatihan (Training) | *trainval.txt* | 1.000 | 66,67% |
| Data Pengujian (Testing) | *test.txt* | 500 | 33,33% |

Penelitian ini bersifat non-referensial, yaitu model mendeteksi cacat tanpa membandingkan citra uji dengan citra *template*, sesuai kondisi inspeksi nyata ketika citra referensi sering tidak tersedia. Oleh karena itu, dari setiap pasangan citra hanya citra target (*\_test.jpg*) yang dimuat, sedangkan citra templat (*\_temp.jpg*) diabaikan.

Koordinat cacat pada berkas anotasi diubah menjadi *bounding box*, sedangkan ID kelas 1–6 dipakai langsung sebagai indeks kelas, yaitu *open*, *short*, *mousebite*, *spur*, *spurious copper*, dan *pinhole*. Contoh sampel beserta anotasinya ditunjukkan pada Gambar 3.2.

[Gambar 3.2]

**Gambar 3.2 Contoh sampel dataset PCB beserta anotasinya**

## 3.2 Pra-pemrosesan *Dataset*

Citra diproses melalui dua *pipeline*. *Pipeline* di luar model mengubah citra menjadi *tensor* dan menerapkan augmentasi saat pelatihan. *Pipeline* di dalam model menormalisasi citra dan mengatur resolusi, yaitu dengan acak multi-resolusi saat latih, dan tetap saat inferensi maupun ekstraksi fitur agar fitur bagi komponen simbolik stabil. Konfigurasinya dirangkum pada Tabel 3.2 dan Tabel 3.3.

**Tabel 3.2 Parameter Normalisasi dan Standardisasi Input**

| **Parameter** | **Nilai Konfigurasi** | **Tujuan** |
|:----------------------:|------------------------|------------------------|
| Rentang Intensitas | \[0.0, 1.0\] | Penyeragaman *dynamic range* piksel |
| Rata-rata (*Mean*) | \[0.485, 0.456, 0.406\] | Penyelarasan distribusi ImageNet |
| Deviasi Standar (*Std*) | \[0.229, 0.224, 0.225\] | Penyelarasan distribusi ImageNet |

**Tabel 3.3 Konfigurasi Augmentasi Data Pelatihan**

| **Jenis Augmentasi** | **Parameter** | **Dekripsi** |
|:----------------------:|------------------------|------------------------|
| *Random Horizontal Flip* | Probabilitas: 0,5 | Variasi Arah |
| *Multi-resolution Scaling* | Sisi terpendek: {480, 560, 640, 720, 800, 880}; sisi terpanjang maks. 880 | Ketahanan terhadap skala |
| *Coordinate Sync* | Enabled | Penyesuaian otomatis lokasi *box* saat gambar berubah |

Secara berurutan, citra diubah menjadi *tensor* berintensitas \[0,0; 1,0\], dibalik horizontal secara acak pada fase latih beserta koordinat *bounding box*-nya, lalu di dalam model dinormalisasi per kanal sesuai Tabel 3.2, diskalakan sesuai fase pada Tabel 3.3, dan disamakan ukurannya dalam satu *batch* melalui *padding*.

## 3.3 Model Neuro (*Faster* R-CNN)

*Faster* R-CNN berperan sebagai ekstraktor fitur, pengusul area, dan model *teacher*. Arsitekturnya mengikuti SF-PSPyramid dari Fung et al. (2024) (Subbab 2.1.5), dengan satu perbedaan, yaitu jumlah kanal *neck* 64, bukan 256, agar dimensi masukan SODT tidak terlalu besar sehingga hanya menjadi ($64 \times 7 \times 7 = 3.136$).

Alur modelnya mengikuti Subbab 2.1.5. *Backbone* ResNet-50 mengekstrak *feature map* C2–C5, lalu *neck* membentuk piramida P2′–P6 melalui *pixel shuffle* pada Persamaan 2.8 dan SF *Attention* pada Persamaan 2.9. RPN menghasilkan proposal dari *anchor* sesuai Persamaan 2.10 sampai 2.13, lalu *RoI Align* mengubah setiap proposal menjadi *tensor* 64×7×7 pada tingkat piramida terpilih dengan Persamaan 2.17 dan 2.19. Selanjutnya, *box head* menghasilkan skor kelas dan koordinat *bounding box*, dan Soft-NMS menyaring deteksi yang tumpang-tindih dengan Persamaan 2.20. Rincian konfigurasi tiap modul dirangkum pada Tabel 3.4.

**Tabel 3.4 Parameter utama Faster R-CNN**

<table style="width:95%;">
<colgroup>
<col style="width: 29%" />
<col style="width: 30%" />
<col style="width: 35%" />
</colgroup>
<thead>
<tr>
<th style="text-align: center;"><strong>Modul Fungsional</strong></th>
<th style="text-align: center;"><strong>Parameter Utama</strong></th>
<th style="text-align: center;"><strong>Nilai / Konfigurasi</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" style="text-align: left;"><em>Backbone &amp; Neck</em></td>
<td style="text-align: left;">Inisialisasi Bobot</td>
<td style="text-align: left;"><em>ResNet-50, Pre-trained</em> ImageNet</td>
</tr>
<tr>
<td style="text-align: left;"><em>Batch Normalization</em></td>
<td style="text-align: left;"><em>Frozen</em></td>
</tr>
<tr>
<td rowspan="2" style="text-align: left;"><em>Neck</em> (SF-PSPyramid)</td>
<td style="text-align: left;"><em>Feature map</em> FPN</td>
<td style="text-align: left;">P2, P3, P4, P5, P6</td>
</tr>
<tr>
<td style="text-align: left;">Jumlah Kanal (C)</td>
<td style="text-align: left;">64</td>
</tr>
<tr>
<td rowspan="4" style="text-align: left;"><em>RPN</em></td>
<td style="text-align: left;">Ukuran (<em>Anchor</em>)</td>
<td style="text-align: left;">[16, 32, 64, 128, 256] piksel</td>
</tr>
<tr>
<td style="text-align: left;">Rasio Aspek</td>
<td style="text-align: left;">[0.5, 1.0, 2.0]</td>
</tr>
<tr>
<td style="text-align: left;">Ambang Batas IoU</td>
<td style="text-align: left;"><p><em>Foreground</em>: 0.7</p>
<p><em>Background</em>: 0.3</p></td>
</tr>
<tr>
<td style="text-align: left;">Batas Proposal (NMS)</td>
<td style="text-align: left;"><em>Train: 2000 | Test: 1000</em></td>
</tr>
<tr>
<td style="text-align: left;"><em>RoI Align</em></td>
<td style="text-align: left;">Dimensi <em>RoI Align</em></td>
<td style="text-align: left;"><em>7x7 piksel (Rasio sampling: 2)</em></td>
</tr>
<tr>
<td rowspan="3" style="text-align: left;"><em>Soft-NMS</em></td>
<td style="text-align: left;">Metode Penurunan (<em>Decay</em>)</td>
<td style="text-align: left;"><em>Linier (Linear)</em></td>
</tr>
<tr>
<td style="text-align: left;">Parameter Sigma &amp; IoU</td>
<td style="text-align: left;"><p><em>Sigma: 0.5</em></p>
<p><em>IoU Threshold: 0.5</em></p></td>
</tr>
<tr>
<td style="text-align: left;">Pemotongan Skor (<em>Score Thresh</em>)</td>
<td style="text-align: left;"><em>0.001</em></td>
</tr>
</tbody>
</table>

## 3.4 Pelatihan dan Evaluasi Model Neuro

*Faster* R-CNN dilatih selama 15 *epoch* (Fung et al. (2024) memakai 12) dengan *Automatic Mixed Precision* (AMP) dan *gradient accumulation* untuk mengatasi keterbatasan memori GPU. Kerugian regresi memakai L1 murni pada Persamaan 2.15 mengikuti Fung et al. (2024) karena memberi penalti lebih tegas pada *cacat* kecil. Laju pembelajaran dinaikkan secara linear pada awal pelatihan (*warmup*), lalu diturunkan bertahap. *Hyperparameter* dirangkum pada Tabel 3.5.

*\*

**Tabel 3. 5 Hyperparameter pelatihan model Faster R-CNN**

<table style="width:91%;">
<colgroup>
<col style="width: 31%" />
<col style="width: 30%" />
<col style="width: 29%" />
</colgroup>
<thead>
<tr>
<th style="text-align: center;"><em><strong>Hyperparameter</strong></em></th>
<th style="text-align: center;"><strong>Parameter Utama</strong></th>
<th style="text-align: center;"><strong>Nilai / Konfigurasi</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align: left;">Siklus Pelatihan</td>
<td style="text-align: left;">Total <em>Epoch</em></td>
<td style="text-align: left;">15</td>
</tr>
<tr>
<td style="text-align: left;">Gradient Accumulation</td>
<td style="text-align: left;">4 langkah (<em>Effective Batch</em>: 4)</td>
</tr>
<tr>
<td style="text-align: left;">Presisi Komputasi</td>
<td style="text-align: left;"><em>AMP: True</em>)</td>
</tr>
<tr>
<td rowspan="3" style="text-align: left;">Fungsi Pengoptimal</td>
<td style="text-align: left;">Optimizer</td>
<td style="text-align: left;">SGD</td>
</tr>
<tr>
<td style="text-align: left;">Laju Pembelajaran (<em>Learning Rate</em>)</td>
<td style="text-align: left;">0,02</td>
</tr>
<tr>
<td style="text-align: left;">Momentum &amp; <em>Weight Decay</em></td>
<td style="text-align: left;">Momentum: 0.9 | <em>Decay</em>: 0.0001</td>
</tr>
<tr>
<td rowspan="3" style="text-align: left;">Penjadwalan (Scheduler)</td>
<td style="text-align: left;">Tipe <em>Scheduler</em></td>
<td style="text-align: left;">Milestone / Step Decay</td>
</tr>
<tr>
<td style="text-align: left;"><em>Milestones</em></td>
<td style="text-align: left;"><em>Epoch</em> ke-8 dan ke-11</td>
</tr>
<tr>
<td style="text-align: left;">Faktor Penurunan (<em>Gamma</em>)</td>
<td style="text-align: left;">0.1 (Penurunan 10%)</td>
</tr>
<tr>
<td style="text-align: left;">Pemanasan (Warmup)</td>
<td style="text-align: left;">Iterasi <em>Warmup</em></td>
<td style="text-align: left;">500 batch</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: left;">Rasio <em>Warmup</em> Awal</td>
<td style="text-align: left;">0.001</td>
</tr>
</tbody>
</table>

Pada setiap *batch*, *multi-task loss* pada Persamaan 2.14 dengan kerugian L1 pada Persamaan 2.15 dihitung, lalu gradiennya diakumulasikan selama empat iterasi sebelum SGD memperbarui bobot sesuai Persamaan 2.3. Setelah pelatihan, model dievaluasi pada data uji dengan mAP@0,5 dan mAP@0,5:0,95, serta *Precision*, *Recall*, dan F1 pada IoU 0,5 dan skor 0,5 sesuai Persamaan 2.35 sampai 2.40, termasuk per kelas dan *confusion matrix*. Bobot model kemudian dibekukan dan disimpan sebagai model *teacher*.

## 3.5 Ekstraksi Fitur RoI dan Hasil Klasifikasi *Faster* R-CNN

Tahap ini membentuk dataset simbolik sesuai skema *model mimicking* di Subbab 2.1.7. Fitur diambil tepat setelah *RoI Align*, sebelum masuk MLP pada *box head*, sehingga bentuk grid 7×7 tetap terjaga. Yang diekstrak adalah seluruh proposal RPN pada mode inferensi, bukan hanya deteksi akhir, agar SODT dilatih pada populasi proposal yang sama dengan saat inferensi.

Label setiap proposal adalah prediksi *teacher*, termasuk *background*, bukan *ground truth*. *Ground truth* hanya disimpan sebagai data pendamping untuk metrik lokalisasi. Ekstraksi dilakukan terpisah untuk data latih dan data uji. Mekanismenya diilustrasikan pada Gambar 3.3.

[Gambar 3.3]

**Gambar 3.3 Pengambilan RoI Align dan Hasil dari Faster R-CNN**

Berdasarkan mekanisme tersebut, pembentukan *dataset* simbolik dirangkum pada Algoritma 3.1.

**Algoritma 3. 1 Pembentukan Dataset Simbolik**

**Input:** Model *teacher* (*frozen*) serta citra latih dan uji beserta anotasinya.

**Output:**  *Dataset* simbolik latih dan uji berisi fitur $64 \times 7 \times 7$, koordinat proposal, label *teacher*, dan data pendamping *ground truth*.

**Langkah-langkah:**

1. Setiap citra diproses dengan resolusi tetap di Tabel 3.3, lalu RPN menghasilkan hingga 1.000 proposal.
2. *RoI Align* mengubah setiap proposal menjadi tensor 64×7×7 pada Persamaan 2.17 dan 2.19.
3. *Tensor* diteruskan ke *box head teacher*, dan kelas dengan *softmax* tertinggi dengan Persamaan 2.4 diambil sebagai label.
4. Setiap proposal dicocokkan dengan *ground truth* ber-IoU tertinggi sebagai data pendamping.
5. Fitur, label, dan data pendamping disimpan sebagai dataset simbolik.

## 3.6 Model Simbolik (SODT)

SODT menggantikan *classifier* pada *box head*, sehingga setiap keputusan kelas berasal dari rangkaian keputusan linear yang dapat ditelusuri di Subbab 2.1.8. Masukannya adalah *tensor* $64 \times 7 \times 7$ yang diratakan menjadi vektor berdimensi 3.136, dengan urutan yang dicatat agar setiap bobot dapat dikembalikan ke kanal dan posisi grid asalnya.

SODT dibentuk sebagai pohon biner lengkap dengan fungsi keputusan linear pada setiap *node* internal sesuai Persamaan 2.28 dan satu label kelas pada setiap *leaf*. Bobot dan bias setiap *node* diinisialisasi dari distribusi normal baku (Hada et al., 2024), sedangkan label *leaf* diinisialisasi secara acak, bukan dengan kelas mayoritas, agar TAO tidak terjebak pada kelas *background* yang dominan. Konfigurasinya dirangkum pada Tabel 3.6.

**Tabel 3.6 Konfigurasi struktur dan inisialisasi SODT**

| **Parameter** | **Nilai** | **Keterangan** |
|:----------------:|------------------|-------------------------------------|
| Kedalaman pohon | 6 | Batas maksimum tingkat hierarki *node* |
| Inisialisasi bobot dan bias | Distribusi normal baku N(0, 1) | Titik awal optimasi TAO |
| Inisialisasi label *leaf* | Acak | Mencegah TAO terjebak pada kelas *background* yang dominan |

## 3.7 Pelatihan dan Evaluasi Model Simbolik

SODT dilatih dengan TAO pada *dataset* simbolik latih untuk meminimalkan fungsi tujuan yang terdiri atas kerugian berbobot kelas pada Persamaan 2.33 dan penalti L1 berbobot ukuran *reduced set* pada Persamaan 2.31 dan 2.32. Karena sebagian besar proposal berlabel *background*, distribusi kelasnya sangat timpang sehingga diperlukan *negative sampling*.

Pembobotan kelas diterapkan di dua tempat. Pada masalah tereduksi setiap node (Persamaan 2.30), setiap sampel diberi bobot pada Persamaan 3.1. Pada *leaf*, label ditentukan dengan mayoritas berbobot pada Persamaan 3.2.

$$
u_{n} = \left| \mathcal{l}_{L}(n) - \mathcal{l}_{R}(n) \right| \cdot \omega_{y_{n}}
$$

(3.1)

$$
{\widehat{y}}_{\mathcal{l}} = \arg{\max_{c}{\omega_{c} \cdot N_{\mathcal{l,}c}}}
$$

(3.2)

dengan $u_{n}$ bobot sampel ke-*n*, $\mathcal{l}_{L}(n)\ ,\mathcal{l}_{R}(n)$ kerugian 0/1 sampel ke-*n* bila diarahkan ke kiri dan ke kanan, $\omega_{c}$ bobot kelas $c$ pada Persamaan 2.33, ${\widehat{y}}_{\mathcal{l}}$ label *leaf* ke-$\mathcal{l}$, dan $N_{\mathcal{l,}c}$ jumlah sampel berlabel $c$ pada *leaf* tersebut. Faktor $\left| \mathcal{l}_{L} - \mathcal{l}_{R} \right|$ bernilai 1 hanya untuk *care set*, sehingga kesalahan pengarahan pada kelas berbobot besar menjadi lebih mahal. Bobot *background* dibiarkan netral agar *false positive* tidak meningkat.

Setelah TAO, pohon dipangkas dengan menghapus *dead branch* dan *pure subtree* (Carreira-Perpiñán & Tavallali, 2018; Hada et al., 2024) tanpa mengubah keputusannya, sehingga penjelasan yang dihasilkan menjadi lebih ringkas tanpa mengorbankan fidelitas. *Hyperparameter pelatihan dirangkum pada Tabel 3.7.*

**Tabel 3.7 Hyperparameter pelatihan SODT**

<table style="width:97%;">
<colgroup>
<col style="width: 31%" />
<col style="width: 26%" />
<col style="width: 38%" />
</colgroup>
<thead>
<tr>
<th style="text-align: center;"><strong>Tahapan/Komponen</strong></th>
<th style="text-align: center;"><strong>Nilai</strong></th>
<th style="text-align: center;"><strong>Deskripsi</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align: left;">Iterasi Maksimum TAO</td>
<td style="text-align: left;">15</td>
<td style="text-align: left;">Jumlah siklus pembaruan seluruh <em>node</em></td>
</tr>
<tr>
<td style="text-align: left;">Lambda (<em>λ</em>)</td>
<td style="text-align: left;">20</td>
<td style="text-align: left;">Kekuatan penalti L1 pada Persamaan 2.31</td>
</tr>
<tr>
<td style="text-align: left;">Alpha (<em>α)</em></td>
<td style="text-align: left;">0.15</td>
<td style="text-align: left;">Eksponen pembobot penalti pada Persamaan 2.32</td>
</tr>
<tr>
<td style="text-align: left;">Rasio Negatif</td>
<td style="text-align: left;">2</td>
<td style="text-align: left;">Sampel <em>background</em> terhadap sampel cacat</td>
</tr>
<tr>
<td style="text-align: left;">Bobot Kelas (<em>w</em>)</td>
<td><em>Short</em>: 2<br />
<em>Spur</em>: 1,5<br />
<em>Open</em>: 1,5<br />
<em>Pinhole</em>: 1,25<br />
<em>Spurious copper</em>: 1,25</td>
<td style="text-align: left;">Biaya lebih besar untuk kelas dengan <em>recall</em> terlemah</td>
</tr>
<tr>
<td style="text-align: left;">Toleransi Konvergensi</td>
<td>$10^{- 6}$</td>
<td style="text-align: left;">Pelatihan berhenti jika penurunan relatif fungsi tujuan di bawah nilai ini</td>
</tr>
</tbody>
</table>

Fidelitas SODT diukur terhadap label *teacher* pada *dataset* simbolik uji dengan akurasi (*mimic accuracy*) dan *macro*-F1, serta kesesuaian per kelas, disertai jumlah *node* aktif dan *nonzero weight*. Seluruh tahapan pelatihan, mulai dari *negative sampling* hingga pemangkasan, dirangkum pada Algoritma 3.2.

**Algoritma 3.2 Pelatihan SODT dengan TAO**

**Input:**  *Dataset* simbolik latih dan uji, serta SODT terinisialisasi.

**Output:** SODT terlatih dan hasil evaluasi fidelitasnya.

**Langkah-langkah:**

1. Sistem mempertahankan seluruh proposal berlabel cacat dan mengambil acak proposal *background* sesuai rasio negatif pada Tabel 3.7.
2. Pada setiap iterasi TAO, *node* diperbarui dari yang terdalam menuju *root* menggunakan bobot sampel Persamaan 3.1, lalu label *leasf* diperbarui dengan Persamaan 3.2.
3. Iterasi berhenti setelah mencapai batas iterasi TAO atau ketika penurunan relatif fungsi tujuan di bawah toleransi sesuai Tabel 3.7.
4. Node yang tidak dilalui sampel (*dead branch*) dan *node* yang seluruh *leaf* di bawahnya berlabel sama (*pure subtree*) dinolkan bersama sub-pohonnya, sedangkan *leaf* yang tidak dicapai sampel diberi label *background*.
5. Fidelitas diukur pada dataset simbolik uji.

## 3.8 Integrasi *Faster* R-CNN dan SODT

Tahap ini menyatukan *Faster* R-CNN dan SODT menjadi satu alur inferensi, ditambah dua komponen, yaitu skor deteksi berbasis *routing margin* dan *heatmap* per *node*.

1.  ***Hybrid* *Inference***

SODT hanya menggantikan kepala klasifikasi. Koordinat *bounding box* akhir tetap dihitung oleh kepala regresi *Faster* R-CNN, sehingga sifat *faithful* berlaku pada keputusan kelas, bukan pada penyesuaian lokasi kotak. Alurnya ditunjukkan pada Gambar 3.4.

[Gambar 3.4]

**Gambar 3.4 Diagram integrasi Neuro-Symbolic**

Penggantian kepala klasifikasi tersebut menimbulkan satu persoalan pada skor deteksi. Karena setiap *leaf*  hanya menyimpan satu label, semua deteksi yang mencapai *leaf*  berkelas sama akan memiliki skor identik di Subbab 2.1.7. Akibatnya, AP dan *Soft*-NMS kehilangan urutan skor. Oleh karena itu, skor dibentuk dari *margin* pada Persamaan 2.26 dan *sigmoid* pada Persamaan 2.27 setiap *node* pada jalur keputusan, yang selanjutnya disebut skor deteksi berbasis *routing margin*, sebagaimana dinyatakan pada Persamaan 3.3.

$$
s(x) = \prod_{i \in P(x),\, w_{i} \neq 0}^{}{\sigma\left( \left| f_{i}(x) \right| \right)}
$$

(3.3)

dengan $s(x)$ skor untuk RoI $x$ pada kelas *leaf* yang dicapai (kelas lain bernilai 0), $P(x)$ *node* pada jalur keputusan, dan $f_{i}$ fungsi keputusan *node* ke-*i* pada Persamaan 2.28. *Node* yang telah dinolkan dilewati. Bentuk perkalian ini setara dengan probabilitas *routing* pada *soft decision tree* (Kontschieder et al., 2015), namun pada penelitian ini pohon tetap dieksekusi secara *hard* sehingga jalur dan label tidak berubah. Skor ini hanya berasal dari parameter pohon. Ilustrasinya ditunjukkan pada Gambar 3.5.

[Gambar 3.5]

**Gambar 3.5 Ilustasi *Routing Margin* pada SODT.**

Berbeda dengan Platt (1999), parameter *sigmoid* bernilai tetap, sehingga $s(x)$ merupakan skor *confidence*, bukan probabilitas kelas, dan setiap faktornya berada pada rentang \[0,5; 1). Karena setiap *node* pada TAO diselesaikan dengan regresi logistik berregularisasi L1, fungsi keputusan *node* pada Persamaan 2.28 merupakan *logit* dari regresi logistik tersebut, sehingga setiap faktor sigmoid pada Persamaan 3.3 dapat dimaknai sebagai keyakinan lokal *node* atas arah yang diambil (Carreira-Perpiñán & Tavallali, 2018; Kairgeldin & Carreira-Perpiñán, 2025). Namun, karena setiap faktor bernilai kurang dari 1, *leaf* pada jalur yang lebih panjang cenderung memperoleh skor maksimum yang lebih rendah, sehingga skor ini memiliki bias terhadap panjang jalur.

Dengan skor tersebut, alur inferensi model hibrida dari citra uji hingga deteksi akhir dirangkum pada Algoritma 3.3.

**Algoritma 3.3 Hybrid Inference dengan Routing Margin**

**Input:** Citra uji, *Faster* R-CNN (*frozen*), dan SODT terlatih.

**Output:** *Bounding box, label kelas, skor deteksi, dan jalur keputusan setiap deteksi.*

**Langkah-langkah:**

1. *Faster* R-CNN menghasilkan proposal dan *tensor* $64 \times 7 \times 7$ untuk setiap proposal.
2. SODT menentukan label dari *leaf* yang dicapai, dan skor deteksi berbasis *routing margin* dihitung dengan Persamaan 3.3.
3. Kepala regresi *Faster* R-CNN menghitung koordinat *bounding box* akhir.
4. Deteksi berlabel *background* atau berskor di bawah ambang skor minimum sesuai pada Tabel 3.4 dibuang, lalu *Soft*-NMS diterapkan dari Persamaan 2.20.
5. Jalur keputusan, tingkat piramida sumber, dan feature map *neck* setiap deteksi disimpan untuk pembentukan *heatmap*.

2.  ***Heatmap* per *Node***

Setiap *node* pada jalur keputusan memperoleh satu *heatmap* yang menunjukkan daerah yang dipertimbangkan oleh *node* tersebut untuk RoI yang dijelaskan. Pendekatan ini serupa dengan *Class Activation Mapping* (CAM) yang mengalikan bobot *classifier* linear dengan *feature map* (Zhou et al., 2016), tetapi diterapkan pada setiap *node* SODT dan melalui RoI Align. Berbeda dengan peta Kairgeldin dan Carreira-Perpiñán (2025) pada Persamaan 2.34 yang statis, *heatmap* ini dinamis karena memakai nilai fitur RoI, dan dihitung pada feature map *neck*, bukan grid $7 \times 7$.

Fungsi keputusan *node* linear terhadap fitur RoI pada Persamaan 2.28, dan RoI Align, yang merata-ratakan titik sampel hasil interpolasi bilinear, linear terhadap feature map *neck* sesuai Persamaan 2.18. Dengan bobot *node* per kanal $w_{i,c}$ dan feature map kanal ke-*c* pada tingkat terpilih $F_{c}$, bagian linear fungsi keputusan dapat ditulis sebagai Persamaan 3.4.

$$
f_{i}(x) - b_{i} = \sum_{c = 1}^{C}{w_{i,c}^{T}\, A\, F_{c}}
$$

(3.4)

dengan $A$ matriks koefisien *RoI Align* dan $C = 64$. Gradien Persamaan 3.4 terhadap $F_{c}$ adalah $A^{T}w_{i,c}$, yaitu bobot *node* yang disebar kembali ke posisi asalnya pada feature map. Gradien ini dikalikan dengan nilai feature map dengan menggunakan *Gradient × Input* pada Persamaan 2.23, sebagaimana dinyatakan pada Persamaan 3.5.

$$
c_{i}\lbrack c,p\rbrack = d_{i} \cdot \left( A^{T}w_{i,c} \right)_{p} \cdot F_{c}(p)
$$

(3.5)

dengan $c_{i}\lbrack c,p\rbrack$ kontribusi kanal ke-*c* posisi $p$ terhadap *node* ke-*i*, dan $d_{i}$ arah keputusan *node*, sehingga kontribusi positif berarti mendukung arah yang diambil. Karena seluruh operasinya linear, *completeness* yang disebutkan pada Persamaan 2.24 dan 2.25 berlaku secara eksak sebagaimana Persamaan 3.6. Bias *node* tidak diatribusikan ke posisi mana pun, sehingga jumlah seluruh kontribusi sama dengan bagian linear fungsi keputusan pada Persamaan 3.4, bukan margin pada Persamaan 3.3.

$$
\sum_{c = 1}^{C}{\sum_{p}^{}{c_{i}\lbrack c,p\rbrack}} = d_{i} \cdot \left( f_{i}(x) - b_{i} \right)
$$

(3.6)

Untuk ditampilkan, kontribusi seluruh kanal diringkas menjadi satu peta dengan Persamaan 3.7.

$$
H_{i}(p) = \sum_{c = 1}^{C}\left| c_{i}\lbrack c,p\rbrack \right|
$$

(3.7)

Nilai mutlak membuat $H_{i}$ menunjukkan besar pengaruh, baik yang mendukung maupun yang menentang, sedangkan arah keputusan ditampilkan pada jalur pohon.

Ketelitian $H_{i}$ berada pada tingkat daerah karena perhitungan eksak berhenti pada *feature map* *neck*, tetapi tetap lebih halus daripada *grid* $7 \times 7$. Alurnya ditunjukkan pada Gambar 3.6.

[Gambar 3.6]

**Gambar 3.6 Alur Pembentukan *Heatmap* per *Node***

Berdasarkan Persamaan 3.4 hingga 3.7, pembentukan *heatmap* untuk satu deteksi dirangkum pada Algoritma 3.4.

**Algoritma 3.4 Pembentukan Heatmap per Node**

**Input:** Satu deteksi beserta *tensor* RoI, jalur keputusan, tingkat piramida, dan feature map *neck*.

**Output:** Satu heatmap untuk setiap node aktif pada jalur keputusan.

**Langkah-langkah:**

1. Bobot $w_{i}$ dikembalikan ke grid 64×7×7 dan dikalikan dengan $d_{i}$.
2. *RoI Align* dijalankan ulang pada *feature map neck*, lalu gradien Persamaan 3.4 dihitung dengan propagasi mundur.
3. Gradien dikalikan dengan *feature map* pada Persamaan 3.5 dan diringkas menjadi $H_{i}$ pada Persamaan 3.7.
4. $H_{i}$ dipotong sesuai letak proposal pada tingkat piramida menggunakan *stride* pada Persamaan 2.5 dan 2.6, dinormalisasi, lalu diperbesar dengan interpolasi bilinear sesuai Persamaan 2.16 ke ukuran proposal.
5. *Heatmap* ditumpangkan pada citra.

## 3.9 Evaluasi Model *Neuro-Symbolic*

Evaluasi mencakup dua aspek, yaitu kinerja deteksi model hibrida dan kualitas penjelasan yang dihasilkannya. Seluruh pengujian dilakukan pada data uji.

1.  **Evaluasi Deteksi**

Model hibrida dibandingkan dengan *Faster* R-CNN menggunakan metrik pada Subbab 3.5. Untuk mengukur peran *Routing Margin*, metrik yang sama dihitung ulang dengan skor seluruh deteksi diganti menjadi 1, tanpa mengubah jalur maupun label.

Waktu rata-rata per citra uji juga diukur untuk *Faster* R-CNN, model hibrida, dan Grad-CAM (termasuk pembentukan petanya).

2.  **Evaluasi Penjelasan**

Grad-CAM menghasilkan satu peta per deteksi, sedangkan SODT satu peta per *node*. Untuk perbandingan, *heatmap* per *node* ditumpuk menjadi satu peta dengan Persamaan 3.8.

$$
M(p) = \sum_{i \in P(x)}^{}{H_{i}(p)}
$$

(3.8)

dengan $M(p)$ peta gabungan pada posisi $p$, $P(x)$ *node* pada jalur keputusan RoI $x$ pada Persamaan 3.3, dan $H_{i}$ *heatmap* *node* ke-*i* pada Persamaan 3.7. Peta $M$ hanya dipakai untuk perbandingan dengan Grad-CAM.

Penjelasan diuji pada tiga hal, yaitu *faithfulness* tingkat jalur yang memeriksa apakah daerah yang disorot menentukan label, lokalisasi yang memeriksa apakah daerah tersebut jatuh pada cacat sebenarnya, dan *faithfulness* per *node* yang menguji klaim bahwa setiap *heatmap* menentukan keputusan *node*-nya. Perbandingan kualitatif melengkapinya dengan telaah per deteksi.

Seluruhnya dibandingkan dengan Grad-CAM, yang dihitung pada *feature map* *neck* tempat proposal di-pool, yaitu lapisan konvolusi terakhir sebelum kepala klasifikasi, sesuai anjuran Selvaraju et al. (2017), serta dengan kontrol acak (Adebayo et al., 2018) agar hasilnya tidak dapat dijelaskan oleh pola aktivasi semata. Prosedur pengujiannya dirangkum pada Algoritma 3.5.

**Algoritma 3. 5 Evaluasi Penjelasan**

**Input:** Model hibrida, Grad-CAM, data uji, dan *dataset* simbolik uji.

**Output:** Metrik *faithfulness* tingkat jalur dan per *node*, metrik lokalisasi, serta perbandingan visual.

**Langkah-langkah:**

1. Deteksi berskor ≥ 0,5 dipilih dari 500 citra uji.
2. Peta M pada Persamaan 3.8 dan peta Grad-CAM pada Persamaan 2.21 dan 2.22 dihitung untuk deteksi yang sama.
3. Pada *feature map neck* di dalam proposal (diperluas 2 posisi), 50% posisi tertinggi tiap peta dipilih.
4. Posisi tersebut dinolkan untuk *Necessity* sesuai Persamaan 2.41 dan disisakan untuk *Sufficiency* pada Persamaan 2.42, lalu *RoI Align* dijalankan ulang dan label diperiksa.
5. Langkah 3 dan 4 diulang untuk tiga kontrol, yaitu posisi acak, bobot *node* diacak, dan pengurutan tanpa bobot pohon.
6. Setiap peta di-*resample* ke grid 7×7, lalu Pointing Game dan IoU Heatmap dihitung pada proposal longgar (IoU 0,05–0,35).
7. Untuk setiap node aktif, 50% posisi tertinggi $H_{i}$ dihapus dan pembalikan tanda $f_{i}(x)$ diperiksa, lalu *Deletion* dan *Insertion* AUC (Persamaan 2.43) dihitung dalam lima tahap.
8. Deteksi kedua model dan *ground* *truth* ditampilkan berdampingan, dengan jalur keputusan dan *heatmap* setiap *node* di samping peta Grad-CAM.

# BAB IV HASIL DAN PEMBAHASAN

## 4.1 Hasil Persiapan *Dataset*

*Dataset* DeepPCB dipartisi menjadi 1.000 citra latih dan 500 citra uji sesuai Subbab 3.1 tanpa ada citra yang muncul di kedua himpunan. Statistik anotasinya dirangkum pada Tabel 4.1.

**Tabel 4.1 Statistik Partisi *Dataset* DeepPCB**

|       **Metrik**        | ***Train Set*** | ***Test Set*** |
|:-----------------------:|-----------------|----------------|
|      Jumlah Citra       | 1.000           | 500            |
|      Total Anotasi      | 6.873           | 3.140          |
| Rata-rata Anotasi/Citra | 6,87            | 6,28           |
|  Minimum Anotasi Citra  | 1               | 2              |
|  Maximum Anotasi Citra  | 15              | 13             |
|      Citra Kosong       | 0               | 0              |

Berdasarkan Tabel 4.1, tidak ada citra kosong dan setiap citra rata-rata memuat 6 hingga 7 cacat, sehingga seluruh data merupakan kasus deteksi multi-objek. Distribusi anotasi per kelas ditunjukkan pada Gambar 4.1 dan Gambar 4.2.

[Gambar 4.1]

**Gambar 4.1 Distribusi Anotasi per Kelas pada *Training Set***

[Gambar 4.2]

**Gambar 4.2 Distribusi Anotasi per Kelas pada *Test Set***

Berdasarkan Gambar 4.1 dan Gambar 4.2, kelas terbanyak hanya sekitar 1,4 kali kelas tersedikit pada kedua himpunan, sehingga perbedaan kinerja antarkelas pada pembahasan berikutnya tidak disebabkan oleh ketimpangan jumlah data.

Seluruh *bounding box* berada di dalam batas citra 640×640 piksel tanpa dimensi nol, dan kesesuaiannya dengan citra diperiksa melalui enam sampel acak pada Gambar 4.3.

[Gambar 4.3]

**Gambar 4.3 Sampel Acak Citra PCB dengan Anotasi *Bounding Box Ground Truth***

Berdasarkan Gambar 4.3, setiap *bounding box* pada keenam sampel menutupi area cacat dengan label yang sesuai, sehingga anotasi dapat langsung digunakan sebagai *ground truth*.

## 4.2 Hasil Pra-Pemrosesan *Dataset*

Kedua transformasi pada Tabel 3.3 diterapkan dua kali pada setiap citra untuk memperlihatkan keacakannya sekaligus memastikan *bounding box* tetap sesuai. Hasil *horizontal flip* ditunjukkan pada Gambar 4.4.

[Gambar 4.4]

**Gambar 4.4 Hasil Visualisasi Augmentasi Pembalikan Horizontal pada Citra PCB**

Berdasarkan Gambar 4.4, *flip* terjadi secara acak sehingga model menerima orientasi yang beragam, dan saat citra terbalik *bounding box* selalu ikut berpindah ke posisi cerminnya. Hasil *multi-resolution scaling* ditunjukkan pada Gambar 4.5.

[Gambar 4.5]

**Gambar 4.5 Hasil Visualisasi Penskalaan Multi-Resolusi dalam Arsitektur Model**

Berdasarkan Gambar 4.5, resolusi juga dipilih secara acak lalu diberi *padding* hingga kelipatan 32, tetapi penskalaan yang proporsional membuat bentuk cacat tidak terdistorsi dan *bounding box* tetap pada area cacat. Normalisasi hanya menggeser rentang nilai piksel sesuai *mean* dan *std* pada Tabel 3.2 tanpa mengubah tampilan citra. Dengan demikian, pra-pemrosesan menghasilkan variasi orientasi dan skala tanpa merusak kesesuaian anotasi.

## 4.3 Hasil dan Evaluasi *Faster* R-CNN

*Faster* R-CNN dilatih selama 15 *epoch* dengan *hyperparameter* pada Tabel 3.5. Perkembangan *loss* total dan setiap komponennya ditunjukkan pada Gambar 4.6.

[Gambar 4.6]

**Gambar 4.6 *Training Loss Faster* R-CNN selama 15 *Epoch***

Berdasarkan Gambar 4.6, *loss* total turun tajam setiap kali laju pembelajaran diturunkan sesuai Tabel 3.5 dan mendatar setelah *epoch* ke-12, menandakan model telah konvergen. *Loss* regresi *bounding box* tetap terbesar hingga akhir pelatihan, sedangkan *loss* RPN sudah sangat kecil sejak awal, sehingga kesulitan utama model terletak pada presisi batas *bounding box*, bukan pada menemukan cacat.

Kinerja model pada data uji dirangkum pada Tabel 4.2 bersama hasil Fung et al. (2024) pada *dataset* DeepPCB non-referensial.

**Tabel 4.2 Kinerja Deteksi *Faster* R-CNN pada Data Uji**

| **Model** | **AP50** | **AP75** | **AP@50:5:85** | **mAP@0,5:0,95** | ***Precision*** | ***Recall*** | **F1** |
|:------------|--------|--------|---------|----------|-----------|---------|--------|
| *Faster* R-CNN (Fung et al., 2024) | 0,970 | 0,900 | 0,888 | \- | \- | \- | \- |
| *Faster* R-CNN + SF-PSPyramid (Fung et al., 2024) | 0,986 | 0,946 | 0,932 | \- | \- | \- | \- |
| *Faster* R-CNN + SF-PSPyramid (penelitian ini) | 0,979 | 0,920 | 0,901 | 0,759 | 0,910 | 0,982 | 0,945 |

Berdasarkan Tabel 4.2, hampir seluruh cacat terdeteksi, tetapi AP turun sekitar 0,22 saat ambang IoU dinaikkan dari AP50 ke mAP@0,5:0,95, sejalan dengan *loss* regresi *bounding box* yang tetap tinggi. Terhadap Fung et al. (2024), AP@50:5:85 model ini berada di antara *Faster* R-CNN standar dan SF-PSPyramid, dengan selisih 0,031 dari SF-PSPyramid yang diduga berasal dari kanal *neck* yang dikurangi menjadi 64 pada Subbab 3.3.

Kinerja per kelas cacat dirangkum pada Tabel 4.3.

**Tabel 4.3 Kinerja *Faster* R-CNN per Kelas Cacat**

|     **Kelas**     | **AP@0,5**      | ***Precision*** | ***Recall*** |
|:-----------------:|-----------------|-----------------|--------------|
|     *Pinhole*     | 0,992           | 0,799           | 1,000        |
| *Spurious copper* | 0,988           | 0,933           | 0,991        |
|    *Mousebite*    | 0,983           | 0,948           | 0,986        |
|      *Open*       | 0,979           | 0,960           | 0,979        |
|      *Spur*       | 0,971           | 0,961           | 0,977        |
|      *Short*      | 0,963           | 0,859           | 0,956        |

Berdasarkan Tabel 4.3, seluruh kelas mencapai AP yang tinggi dengan selisih antarkelas yang kecil. AP tertinggi dicapai *pinhole* dan *spurious copper* yang berbentuk lubang atau gumpalan, sedangkan AP terendah dicapai *short* dan *spur* yang menempel pada jalur konduktor dan diduga lebih sulit dibedakan dari pola jalur normal. Karena distribusi kelas relatif seimbang, perbedaan ini tidak berasal dari jumlah data. *Precision* terendah terdapat pada *pinhole* dan *short*, yang penyebabnya ditelusuri melalui *confusion matrix* pada Gambar 4.7.

[Gambar 4.7]

**Gambar 4.7 *Confusion Matrix Faster* R-CNN pada Data Uji**

Berdasarkan Gambar 4.7, hampir seluruh cacat terklasifikasi benar dengan sangat sedikit yang tertukar antarkelas, sedangkan kesalahan utama berasal dari *background* yang terdeteksi sebagai cacat, terutama *pinhole* dan *short*, yang menjelaskan rendahnya *precision* kedua kelas tersebut. Kesalahan ini sekitar enam kali lebih banyak daripada cacat yang terlewat, sehingga model cenderung mendeteksi berlebih. Karena SODT meniru label *teacher* sesuai Subbab 3.5, label yang diterimanya hampir tidak tertukar antarkelas, tetapi membawa kecenderungan deteksi berlebih yang sama.

## 4.4 Hasil Ekstraksi Fitur *Teacher*

Model *teacher* tersebut digunakan untuk mengekstrak fitur dan label 1.000 proposal RPN per citra sesuai Subbab 3.5. Distribusi label hasil ekstraksi ditunjukkan pada Tabel 4.4.

**Tabel 4.4 Distribusi Label *Teacher* pada RoI Hasil Ekstraksi**

| **Label *Teacher*** | **Data Latih** | **Data Uji** |
|:-------------------:|----------------|--------------|
|    *Background*     | 854.164        | 431.344      |
|       *Open*        | 29.288         | 15.116       |
|       *Short*       | 20.214         | 10.037       |
|     *Mousebite*     | 29.329         | 12.619       |
|       *Spur*        | 23.213         | 9.577        |
|  *Spurious copper*  | 20.964         | 9.894        |
|      *Pinhole*      | 22.828         | 11.413       |
|        Total        | 1.000.000      | 500.000      |

Berdasarkan Tabel 4.4, lebih dari 85% RoI dilabeli *background* karena setiap citra hanya memuat 6 hingga 7 cacat dari 1.000 proposal. Selain itu, setiap cacat diwakili sekitar 21 proposal yang saling tumpang-tindih. Akibatnya, data SODT didominasi *background*, sedangkan setiap cacat terwakili dari berbagai posisi proposal.

Isi fitur tersebut divisualisasikan pada Gambar 4.8 dengan merata-ratakan 64 kanal setiap RoI menjadi *grid* 7×7.

[Gambar 4.8]

**Gambar 4.8 Visualisasi Fitur RoI 7×7 per Kelas**

Berdasarkan Gambar 4.8, aktivasi kelas cacat terkumpul di tengah *grid*, sedangkan aktivasi *background* berada di tepi atau sudut, sehingga *RoI Align* terbukti mempertahankan letak cacat. Namun, pola antarkelas sulit dibedakan, misalnya *open*, *short*, dan *pinhole* sama-sama terang di tengah *grid*, karena makna 64 kanal hasil pembelajaran *backbone* dan *neck* tidak diketahui. Fitur RoI menyimpan letak cacat, tetapi dasar keputusannya tetap *black-box*. Keterbatasan ini ditangani SODT karena setiap bobot *node*-nya terikat pada kanal dan posisi *grid* tertentu sehingga dapat dipetakan menjadi *heatmap* sesuai Subbab 3.8.

## 4.5 Hasil dan Evaluasi Model Simbolik (*Sparse Oblique Decision Tree*)

Subbab ini mengevaluasi fidelitas SODT hasil Subbab 3.7, yaitu kemampuannya meniru label *teacher* pada 500.000 RoI data uji, bukan kinerjanya terhadap *ground truth* yang dibahas pada Subbab 4.6.

### 4.5.1 Pelatihan dan Fidelitas SODT

SODT dilatih dengan rasio negatif 2 sesuai Tabel 3.7, sehingga *background* mencakup 66,7% dari 437.508 RoI data latih. Perkembangan *mimic accuracy* dan jumlah *nonzero weight* selama pelatihan TAO ditunjukkan pada Gambar 4.9.

[Gambar 4.9 — SISIPKAN: grafik "SODT Training Diagnostics – TAO Training History" (notebook 03, cell training-diagnostics), run terpilih NR 2 + CW]

**Gambar 4.9 Perkembangan *Mimic Accuracy* dan Jumlah *Nonzero Weight* selama Pelatihan TAO**

Berdasarkan Gambar 4.9, *mimic accuracy* sudah di atas 95% sejak iterasi pertama dan hanya naik sekitar 1,5 poin, sedangkan *nonzero weight* berkurang hampir separuhnya, sehingga TAO lebih banyak menyederhanakan pohon daripada menaikkan fidelitas. Penurunan tipis *mimic accuracy* pada iterasi ke-5 wajar karena TAO hanya menjamin penurunan fungsi tujuan pada Persamaan 2.31 dan 2.33, bukan kenaikan akurasi. Pelatihan berhenti pada iterasi ke-8 setelah penurunan fungsi tujuan di bawah toleransi 10⁻⁶, dan titik ke-9 merupakan pemangkasan 14 *node* yang tidak mengubah keputusan pohon pada data latih.

Struktur dan fidelitas agregat SODT dirangkum pada Tabel 4.5. *Mimic accuracy* adalah proporsi RoI yang labelnya sama dengan *teacher*, sedangkan *agreement* per kelas adalah proporsi RoI suatu kelas menurut *teacher* yang juga diberi label tersebut oleh SODT.

**Tabel 4.5 Ringkasan Struktur dan Fidelitas SODT Terpilih**

|                **Metrik**                | **Nilai**                     |
|:----------------------------------------:|-------------------------------|
|             Kedalaman pohon              | 6                             |
|           *Node* internal aktif          | 32 dari 63                    |
|             *Nonzero weight*             | 4.057 dari 197.568            |
|                Sparsitas                 | 97,9%                         |
| Rata-rata *nonzero weight* per *node* aktif | 126,8 (4,0% dari 3.136 fitur) |
|       *Mimic accuracy* data latih        | 96,51%                        |
|        *Mimic accuracy* data uji         | 96,26%                        |
|           *Macro*-F1 data uji            | 0,889                         |

Berdasarkan Tabel 4.5, SODT meniru *teacher* dengan fidelitas di atas 96% meskipun setiap *node* hanya memakai sekitar 4% fitur. *Agreement* per kelasnya seimbang, yaitu 95,8% pada *spur* hingga 97,2% pada *short*, dengan *short* yang tersulit bagi *teacher* pada Tabel 4.3 justru tertinggi karena memperoleh bobot kelas terbesar. Namun, *macro*-F1 jauh lebih rendah karena *agreement* hanya setara *recall* terhadap *teacher*, sedangkan F1 juga memperhitungkan *precision* yang turun akibat sekitar 16.400 RoI *background* dilabeli cacat, lebih banyak daripada RoI kelas cacat mana pun pada Tabel 4.4. Dampak kebocoran ini terhadap deteksi dievaluasi pada Subbab 4.6.

### 4.5.2 Pengaruh *Negative Ratio* dan *Class Weighting*

Konfigurasi pada Tabel 3.7 dipilih melalui ablasi satu faktor dengan memvariasikan rasio negatif (NR) menjadi 1, 2, 4, dan tanpa *sampling* (*full*) dengan *class weighting* (CW) aktif, lalu menonaktifkan CW pada NR 2. Konfigurasi terbaik adalah yang paling seimbang antarkelas, ditandai oleh rentang *agreement* terkecil. Hasilnya ditunjukkan pada Tabel 4.6, dengan bobot ω yang hanya berlaku pada konfigurasi dengan CW dan konfigurasi terpilih dicetak tebal.

**Tabel 4.6 Pengaruh *Negative Ratio* dan *Class Weighting* terhadap Fidelitas SODT (*Agreement* dalam %)**

| **Kelas (ω) / Metrik** | **NR 1 + CW** | **NR 2 + CW (terpilih)** | **NR 4 + CW** | **NR *full* + CW** | **NR 2 tanpa CW** |
|:-----------------------|------|------|------|------|------|
| *Background* (1) | 94,3 | **96,2** | 97,7 | 98,3 | 96,9 |
| *Open* (1,5) | 98,0 | **96,1** | 93,9 | 91,9 | 93,3 |
| *Short* (2) | 98,3 | **97,2** | 92,8 | 90,9 | 94,1 |
| *Mousebite* (1) | 98,3 | **96,9** | 92,7 | 91,4 | 96,3 |
| *Spur* (1,5) | 97,2 | **95,8** | 92,9 | 91,5 | 93,5 |
| *Spurious copper* (1,25) | 97,4 | **96,0** | 92,3 | 90,5 | 94,6 |
| *Pinhole* (1,25) | 97,7 | **96,2** | 91,9 | 89,8 | 94,3 |
| **Rentang** | 4,0 | **1,4** | 5,8 | 8,5 | 3,6 |
| ***Mimic accuracy* (%)** | 94,77 | **96,26** | 97,03 | 97,34 | 96,56 |
| ***Macro*-F1 (0–1)** | 0,855 | **0,889** | 0,905 | 0,913 | 0,894 |

Berdasarkan Tabel 4.6, kenaikan NR dari NR 1 hingga NR *full* menaikkan *agreement background* sekitar 4 poin, tetapi menurunkan *agreement* kelas cacat terendah lebih dari 7 poin. Dalam jumlah RoI uji, RoI *background* yang dilabeli cacat turun dari sekitar 24.600 menjadi 7.300, sedangkan RoI cacat yang tidak ditiru naik dari sekitar 1.500 menjadi 6.100. *Mimic accuracy* dan *macro*-F1 ikut naik hanya karena *background* mengisi 86,3% data uji. Kecenderungan ini diduga terjadi karena semakin banyaknya sampel *background* pada *care set* di Persamaan 3.1 menggeser batas keputusan *node* agar menyaring *background* lebih ketat tanpa mengubah struktur pohon, seperti terlihat pada Subbab 4.5.3. Di antara keempat nilai, NR 2 menghasilkan rentang terkecil sehingga paling seimbang.

Tanpa CW, rentang pada NR 2 melebar lebih dari dua kali lipat karena kelas cacat tertinggal dari *background*. CW menaikkan *agreement* kelas cacat sebanding dengan bobotnya, dari 3,1 poin pada *short* dengan ω = 2 hingga 0,6 poin pada *mousebite* dengan ω = 1, karena Persamaan 3.1 dan 3.2 membuat kesalahan pada kelas berbobot besar lebih mahal. Sebagai *trade-off* umum *cost-sensitive learning* (Elkan, 2001), *agreement background* turun 0,7 poin atau sekitar 3.000 RoI, sehingga *mimic accuracy* dan *macro*-F1 sedikit lebih rendah. Karena kriteria penelitian ini adalah keseimbangan antarkelas, NR 2 dengan CW dipilih sebagai SODT akhir, dengan catatan CW hanya diuji pada NR terpilih.

### 4.5.3 Struktur Pohon SODT

Struktur SODT terpilih setelah pemangkasan ditunjukkan pada Gambar 4.10. Angka pada *node* menyatakan jumlah *nonzero weight*, label pada *leaf* menyatakan kelas keputusan, dan *node* dinomori per tingkat dari kiri ke kanan mulai dari N0 sebagai *root*, sehingga anak kiri dan kanan N*i* adalah N(2*i*+1) dan N(2*i*+2).

[Gambar 4.10 — SISIPKAN: "Pruned SODT Tree (32 active nodes)" (notebook 03, cell tree-comparison), run terpilih NR 2 + CW]

**Gambar 4.10 Struktur SODT Terpilih setelah Pemangkasan**

Berdasarkan Gambar 4.10, pintu keluar *background* tersebar pada kedalaman 3 hingga 6, sedangkan seluruh *leaf* cacat berada pada kedalaman 6 dan 10 dari 13 pasangan *leaf* terbawah berisi dua kelas cacat berbeda, sehingga *node* tingkat atas hingga tengah berperan menyaring *background* dan *node* terbawah menentukan jenis cacat. Empat *node* berbobot nol hanya menjadi jalur lewat, dan sebagian *leaf background* merupakan pengganti cabang yang tidak dilalui sampel. Pohon ketiga konfigurasi NR lainnya memiliki topologi serupa dengan 12 hingga 14 *leaf background*, karena *background* tetap menjadi kelas terbesar bahkan pada NR 1.

Sebaran sparsitas pohon ini sesuai dengan prediksi Kairgeldin dan Carreira-Perpiñán (2025). Dengan α bernilai kecil, yaitu 0,15, penalti efektif per sampel $\lambda\left| \mathcal{R}_{i} \right|^{\alpha - 1}$ pada Persamaan 2.31 dan 2.32 mengecil untuk *node* yang menerima banyak sampel, sehingga *root* dan *node* di jalur utama memakai 249 hingga 534 *nonzero weight*, sedangkan *node* pada kedalaman 5 hanya 8 hingga 131. Jumlah *nonzero weight* pada *root* juga naik seiring NR, dari 340 pada NR 1 menjadi 488 pada NR *full*, sedangkan λ yang besar, yaitu 20, menjaga sparsitas keseluruhan tetap 97,9%.

Namun, sub-pohon di bawah N19 dan N22 hanya berisi *leaf* cacat tanpa satu pun pintu keluar *background*. Karena *routing* bersifat *hard* tanpa mekanisme koreksi, RoI *background* yang lolos ke kedua sub-pohon tersebut pasti dilabeli cacat. Sub-pohon seperti ini muncul pada keempat konfigurasi NR, sehingga diduga menjadi salah satu sumber *false positive* tambahan di luar yang diwarisi dari *teacher*, dan satu contohnya ditunjukkan pada Subbab 4.6.2.

Secara keseluruhan, SODT terpilih meniru *teacher* secara seimbang antarkelas dengan hanya 2,1% *nonzero weight*. Kebocoran sekitar 16.400 RoI *background* dan jalur tanpa pintu keluar *background* diuji dampaknya terhadap deteksi pada Subbab 4.6, sedangkan *nonzero weight* setiap *node* menjadi dasar pembentukan *heatmap* per *node* pada Subbab 4.7.

## 4.6 Hasil dan Evaluasi Deteksi *Neuro-Symbolic*

Subbab ini mengevaluasi model hibrida, selanjutnya disebut NeSy, terhadap *ground truth* pada 500 citra uji dengan metrik pada Subbab 3.5 dan membandingkannya dengan *Faster* R-CNN. Karena *backbone*, RPN, dan kepala regresi keduanya sama, selisih kinerja sepenuhnya berasal dari penggantian kepala klasifikasi oleh SODT sesuai Subbab 3.8.

### 4.6.1 Perbandingan dengan *Faster* R-CNN

Kinerja deteksi kedua model dirangkum pada Tabel 4.7, bersama waktu inferensi rata-rata per citra yang diukur per *batch* pada data uji.

**Tabel 4.7 Kinerja Deteksi dan Waktu Inferensi *Faster* R-CNN dan *Neuro-Symbolic***

| **Model** | **mAP@0,5:0,95** | **mAP@0,5** | ***Precision*** | ***Recall*** | **F1** | **Waktu (ms/citra)** |
|:---------------|------|------|------|------|------|------|
| *Faster* R-CNN | 0,759 | 0,979 | 0,910 | 0,982 | 0,945 | 85,8 |
| NeSy | 0,757 | 0,974 | 0,923 | 0,973 | 0,947 | 68,6 |

Berdasarkan Tabel 4.7, mAP NeSy hampir sama dengan *Faster* R-CNN, dengan *precision* sedikit lebih tinggi dan *recall* sedikit lebih rendah. Fidelitas SODT terhadap *teacher* pada Subbab 4.5 terbawa ke tingkat deteksi, sehingga penggantian kepala klasifikasi hanya sedikit menurunkan akurasi. NeSy juga lebih cepat, bukan karena SODT lebih ringan, sebab MLP tetap dijalankan untuk kepala regresi, melainkan diduga karena jumlah kandidat *Soft*-NMS. Setiap RoI pada NeSy hanya menghasilkan satu kandidat kelas dan RoI yang berakhir pada *leaf background* tidak diteruskan, sedangkan pada *Faster* R-CNN hampir seluruh kelas hasil *softmax* lolos ambang skor dan diproses oleh *Soft*-NMS yang berjalan sekuensial.

Rincian kinerja per kelas cacat kedua model dirangkum pada Tabel 4.8 dalam bentuk AP@0,5, *precision*, dan *recall*.

**Tabel 4.8 Kinerja per Kelas Cacat *Faster* R-CNN dan *Neuro-Symbolic***

| **Kelas** | **AP@0,5 *Faster* R-CNN** | **AP@0,5 NeSy** | ***Precision Faster* R-CNN** | ***Precision* NeSy** | ***Recall Faster* R-CNN** | ***Recall* NeSy** |
|:------------------|------|------|------|------|------|------|
| *Pinhole* | 0,992 | 0,993 | 0,799 | 0,858 | 1,000 | 0,987 |
| *Spurious copper* | 0,988 | 0,978 | 0,933 | 0,983 | 0,991 | 0,981 |
| *Mousebite* | 0,983 | 0,979 | 0,948 | 0,947 | 0,986 | 0,978 |
| *Open* | 0,979 | 0,977 | 0,960 | 0,964 | 0,979 | 0,979 |
| *Spur* | 0,971 | 0,968 | 0,961 | 0,973 | 0,977 | 0,957 |
| *Short* | 0,963 | 0,950 | 0,859 | 0,817 | 0,956 | 0,954 |

Sumber perubahan *precision* dan *recall* tersebut ditelusuri melalui *confusion matrix* pada Gambar 4.11. Baris menyatakan kelas *ground truth* dan kolom menyatakan prediksi, sehingga baris *background* berisi *false positive* (FP), yaitu deteksi cacat pada daerah tanpa cacat, sedangkan kolom *background* berisi *false negative* (FN), yaitu cacat yang tidak terdeteksi.

[Gambar 4.11 — SISIPKAN: confusion matrix FRCNN | NeSy (notebook 06, sel "Detection Metrics Bar Chart + Confusion Matrix", panel 2–3), run NR 2 + CW]

**Gambar 4.11 *Confusion Matrix Faster* R-CNN dan *Neuro-Symbolic***

Berdasarkan Tabel 4.8 dan Gambar 4.11, penurunan *recall* NeSy berasal dari FN yang bertambah dari 46 menjadi 65, sedangkan FP *background* justru turun dari 292 menjadi 236 dan menjelaskan kenaikan *precision*. Penurunan ini diduga berasal dari kecenderungan SODT melabeli *background*, karena SODT dilatih dengan dua RoI *background* untuk setiap RoI cacat sesuai Subbab 4.5.1. Gejalanya sudah tampak pada *agreement* kelas cacat pada Tabel 4.5 yang tidak mencapai 100%, dan Subbab 4.5.2 menunjukkan bahwa makin besar porsi *background*, makin banyak RoI cacat yang tidak ditiru. RoI cacat yang tidak ditiru inilah yang muncul sebagai FN pada tingkat deteksi.

Pada tingkat kelas, pola tersebut mengikuti *agreement* pada Tabel 4.5, meskipun selisih FN setiap kelas paling banyak enam deteksi. FN tidak bertambah pada *short*, kelas dengan bobot dan *agreement* tertinggi, maupun pada *open*, tetapi bertambah pada keempat kelas lainnya dan paling banyak pada *spur* yang *agreement*-nya terendah. Sebaliknya, kenaikan FP *background* hanya terjadi pada *short*, sesuai dengan *class weighting* yang membuat SODT lebih mudah memberi label kelas berbobot terbesar, sehingga *precision short* turun paling besar. *Precision spurious copper* dan *pinhole* justru naik karena FP *background* keduanya berkurang. Kecenderungan SODT pada Subbab 4.5, yaitu condong ke *background* dan diimbangi *class weighting* pada *short*, terbawa hingga tingkat deteksi.

### 4.6.2 Pengaruh *Routing Margin*

Peran skor berbasis *routing margin* diuji dengan mengganti skor seluruh deteksi menjadi 1 sesuai Subbab 3.9, tanpa mengubah jalur maupun label. Hasilnya ditunjukkan pada Tabel 4.9, dengan Δ sebagai selisih antara kondisi dengan dan tanpa *routing margin*.
**Tabel 4.9 Pengaruh *Routing Margin* terhadap Kinerja Deteksi *Neuro-Symbolic***

| **Metrik** | **Dengan *Routing Margin*** | **Tanpa *Routing Margin*** | **Δ** |
|:------------------------|------|------|------|
| mAP@0,5 | 0,974 | 0,821 | +0,153 |
| *Precision* | 0,923 | 0,787 | +0,135 |
| *Recall* | 0,973 | 0,983 | −0,010 |
| F1 | 0,947 | 0,874 | +0,073 |
| AP@0,5 *open* | 0,977 | 0,855 | +0,122 |
| AP@0,5 *short* | 0,950 | 0,706 | +0,244 |
| AP@0,5 *mousebite* | 0,979 | 0,873 | +0,106 |
| AP@0,5 *spur* | 0,968 | 0,842 | +0,126 |
| AP@0,5 *spurious copper* | 0,978 | 0,847 | +0,131 |
| AP@0,5 *pinhole* | 0,993 | 0,801 | +0,192 |

*Confusion matrix* NeSy tanpa *routing margin* ditunjukkan pada Gambar 4.12 dengan cara baca yang sama seperti Gambar 4.11.

[Gambar 4.12 — SISIPKAN: confusion matrix "NeSy — margin OFF" (notebook 06, sel ablation routing-margin, panel 3), run NR 2 + CW]

**Gambar 4.12 *Confusion Matrix Neuro-Symbolic* tanpa *Routing Margin***

Berdasarkan Tabel 4.9 dan Gambar 4.12, tanpa *routing margin* mAP dan *precision* turun tajam, sedangkan *recall* sedikit naik. FP *background* naik dari 236 menjadi 817, atau sekitar 3,5 kali, dan salah kelas antarcacat naik dari 27 menjadi 127. Dengan skor yang seragam, deteksi yang ragu tidak lagi tersaring oleh ambang skor dan AP kehilangan urutan antara deteksi yang yakin dan yang ragu, sedangkan *recall* naik karena lebih sedikit deteksi yang jatuh di bawah ambang. Penurunan AP terbesar terjadi pada *short* dan *pinhole*, dua kelas dengan FP *background* terbanyak. Kebocoran RoI *background* pada Subbab 4.5.1 memang terbawa ke deteksi, dan *routing margin* yang menekannya.

Mekanisme penyaringan tersebut diperlihatkan pada Gambar 4.13 untuk dua deteksi *pinhole*, yaitu deteksi A berupa *true positive* dan deteksi B berupa *false positive* pada daerah *background*.

[Gambar 4.13 — SISIPKAN: panel "Routing margin per node" saja, tanpa pohon; baris A = deteksi #7 pinhole 0,91 (TP, jalur N0–N1–N4–N10–N22–N46), baris B = deteksi #15 pinhole 0,55 (FP background, jalur N0–N1–N4–N9–N19–N40); skor dan jalur ditulis pada judul tiap baris panel (notebook 06, sel 5 per-image explanation)]

**Gambar 4.13 *Routing Margin* per *Node* pada Deteksi *True Positive* dan *False Positive***

Setiap panel pada Gambar 4.13 mewakili satu *node*, dengan garis diagonal sebagai *split* dan garis putus-putus sebagai jarak RoI ke *split* yang sebanding dengan $\left| f_{i}(x) \right|$ pada Persamaan 3.3. Berdasarkan Gambar 4.13, A relatif jauh dari *split* pada seluruh *node*, sedangkan B mendekati *split* pada N4, N9, dan N19. Setelah N9, *node* terakhir yang masih memiliki jalan menuju *background*, B masuk ke sub-pohon N19 dan pasti dilabeli cacat sesuai dugaan pada Subbab 4.5.3. *Routing margin* tidak mengubah label ini, tetapi menempatkan B di bawah A meskipun skornya masih lolos ambang sebagai FP. Sementara itu, A melewati sub-pohon N22 dengan yakin, jadi sub-pohon tanpa pintu keluar *background* tetap sah untuk cacat sebenarnya. Secara keseluruhan, NeSy mempertahankan akurasi *Faster* R-CNN dengan *routing margin* sebagai penyaring deteksi yang ragu, sedangkan kualitas penjelasannya dievaluasi pada Subbab 4.7.

## 4.7 Hasil dan Evaluasi *Explanation* pada Neuro-Symbolic

Evaluasi Explanation pada Neuro-Symbolic menggunakan metrik untuk mengukur *faithfulness* dan kualitas heatmap. Kemudian akan dibandingkan dengan GradCAM yang di mana telah disajikan pada Gambar 4.22.

[Gambar 4.22]

**Gambar 4.22 Perbandingan Faithfulness dan kualitas *Heatmap* pada NeuroSymbolic dan GradCAM**

1.  **Faithfulness**

Neuro-Symbolic (SODT) mencapai Sufficiency Preservation 1,000 dan Necessity Flip Rate 0,971, menunjukkan penjelasan yang benar-benar merepresentasikan keputusan internal model. Sementara itu, Grad-CAM hanya mencapai Sufficiency 0,956 dan Necessity 0,047, mengindikasikan penjelasan yang tidak memenuhi prinsip *necessity* (hanya 4,7% kasus di mana penghapusan area penting mengubah prediksi).

2.  **Spatial Grounding**

Grad-CAM memiliki IoU Heatmap (0,994) sedikit lebih tinggi dari SODT (0,920) karena *bleeding effect* (area heatmap melebar ke luar bounding box). Sementara itu, untuk Pointing Game, Kedua metode hampir identik (0,994 vs 0,983), membuktikan bahwa keduanya mampu mengidentifikasi wilayah cacat. Namun, SODT lebih *faithful* karena heatmapnya **terlokalisasi presisi** pada fitur yang secara eksplisit digunakan dalam keputusan.

3.  **Waktu komputasi**

Grad-CAM membutuhkan 12,6× lebih lama dibanding Neuro-Symbolic (5045,4 ms vs 426,2 ms) karena mekanisme komputasi yang tidak efisien seperti ditunjukkan pada Gambar 4.23.

[Gambar 4.23]

**Gambar 4.23 Perbandingan inference time pada ketiga model.**

Hal ini bisa terjadi karena pada Faster R-CNN, Grad-CAM harus:

1.  Menjalankan satu kali ***backward pass* penuh** melalui seluruh arsitektur dua-tahap (RPN + RoI Head) untuk **setiap kotak deteksi (RoI)** yang dihasilkan.

2.  Menghitung gradien dari skor kelas yang diprediksi hingga ke lapisan konvolusional terakhir.

Proses komputasi berulang untuk setiap RoI ini sangat tidak efisien dan membebani komputasi, sehingga tidak cocok untuk skenario *real-time* seperti inspeksi PCB.

# DAFTAR PUSTAKA

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B., 2018. Sanity checks for saliency maps. In: *Advances in Neural Information Processing Systems (NeurIPS)*, 31, pp. 9525–9536.

Ali, A.M.M., Ziyi, X., Sahlan, S., Khamis, N., Nor Rashid, F.’A., 2024. Transparency in Detecting Defects of a Printed Circuit Board: Harnessing XAI for Improved Quality Control in Electronic Manufacturing Industries. In: 2024 IEEE 10th International Conference on Smart Instrumentation, Measurement and Applications (ICSIMA), pp. 1-6.

Ancona, M., Ceolini, E., Öztireli, C., Gross, M., 2019. Gradient-based attribution methods. In: Samek, W., Montavon, G., Vedaldi, A., Hansen, L.K., Müller, K.-R. (Eds.), *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning*, LNCS 11700. Springer, Cham, pp. 169–191.

Araujo, A., Norris, W., Sim, J., 2019. Computing receptive fields of convolutional neural networks. *Distill*. https://doi.org/10.23915/distill.00021

Arun, N., Gaw, N., Singh, P., Chang, K., Aggarwal, M., Chen, B., Hoebel, K., Gupta, S., Patel, J., Gidwani, M., Adebayo, J., Li, M.D., Kalpathy-Cramer, J., 2021. Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging. *Radiology: Artificial Intelligence* 3(6), e200267.

Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., Samek, W., 2015. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. *PLoS ONE* 10(7), e0130140.

Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., García, S., Gil-López, S., Molina, D., Benjamins, R., Chatila, R., Herrera, F., 2020. Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion* 58, 82–115.

Bodla, N., Singh, B., Chellappa, R., Davis, L.S., 2017. Soft-NMS — Improving object detection with one line of code. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 5561–5569.

Buciluǎ, C., Caruana, R., Niculescu-Mizil, A., 2006. Model compression. In: *Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, pp. 535–541.

Canha, D., Kubler, S., Främling, K., Fagherazzi, G., 2025. A functionally-grounded benchmark framework for XAI methods: Insights and foundations from a systematic literature review. *ACM Computing Surveys* 57(12), 320.

Carreira-Perpiñán, M.Á., Tavallali, P., 2018. Alternating optimization of decision trees, with application to learning sparse oblique trees. In: *Advances in Neural Information Processing Systems (NeurIPS)*, 31, pp. 1211–1221.

Chen, X., Wu, Y., He, X., Ming, W., 2023. A Comprehensive Review of Deep Learning-Based PCB Defect Detection. IEEE Access 11, 139017-139036.

Coombs, C.F., Holden, H.T., 2016. *Printed Circuits Handbook*, 7th ed. McGraw-Hill Education, New York.

d'Avila Garcez, A., Lamb, L.C., 2023. Neurosymbolic AI: the 3rd wave. *Artificial Intelligence Review*, 56(11), pp.12387-12406.

DeYoung, J., Jain, S., Rajani, N.F., Lehman, E., Xiong, C., Socher, R., Wallace, B.C., 2020. ERASER: A benchmark to evaluate rationalized NLP models. In: *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 4443–4458.

Doshi-Velez, F., Kim, B., 2017. Towards a rigorous science of interpretable machine learning. *arXiv preprint* arXiv:1702.08608.

Elkan, C., 2001. The foundations of cost-sensitive learning. In: Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI), pp. 973–978.

Everingham, M., Van Gool, L., Williams, C.K.I., Winn, J., Zisserman, A., 2010. The PASCAL Visual Object Classes (VOC) challenge. *International Journal of Computer Vision* 88(2), 303–338.

Fung, K.C., Xue, K.-W., Lai, C.-M., Lin, K.-H., Lam, K.-M., 2024. Improving PCB defect detection using selective feature attention and pixel shuffle pyramid. Results in Engineering 21, 101992.

Girshick, R., 2015. Fast R-CNN. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 1440–1448.

Goodfellow, I., Bengio, Y., Courville, A., 2016. *Deep Learning*. MIT Press, Cambridge, MA.

Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., Pedreschi, D., 2018. A survey of methods for explaining black box models. *ACM Computing Surveys* 51(5), 93.

Hada, S.S., Carreira-Perpiñán, M.Á., Zharmagambetov, A., 2024. Sparse oblique decision trees: a tool to understand and manipulate neural net features. *Data Mining and Knowledge Discovery*, 38(5), pp.2863-2902.

Han, Z., Hong, M., Wang, D., 2017. Deep learning and applications. In: *Signal Processing and Networking for Big Data Applications*. Cambridge University Press, Cambridge, pp. 126-168.

He, H., Garcia, E.A., 2009. Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), pp. 1263–1284.

He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778.

He, K., Gkioxari, G., Dollár, P., Girshick, R., 2020. Mask R-CNN. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 42(2), 386–397.

Hinton, G., Vinyals, O., Dean, J., 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.

IPC, 2015. IPC-6012D: Qualification and Performance Specification for Rigid Printed Boards. Association Connecting Electronics Industries.

IPC, 2020. *IPC-A-600K: Acceptability of Printed Boards*. Association Connecting Electronics Industries.

Kairgeldin, R., Carreira-Perpiñán, M.Á., 2025. Neurosymbolic models based on hybrids of convolutional neural networks and decision trees. *Proceedings of Machine Learning Research (NeSy 2025)*, 284, pp.796-813.

Kautz, H., 2022. The third AI summer: AAAI Robert S. Engelmore Memorial Lecture. *AI Magazine* 43(1), 105–125.

Khandpur, R.S., 2005. *Printed Circuit Boards: Design, Fabrication, and Assembly*. McGraw-Hill Education.

Kontschieder, P., Fiterau, M., Criminisi, A., Rota Bulò, S., 2015. Deep neural decision forests. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 1467–1475.

LeCun, Y., Bengio, Y., Hinton, G., 2015. Deep learning. *Nature*, 521(7553), pp. 436-444.

Li, X., Wang, W., Hu, X., Yang, J., 2019. Selective kernel networks. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 510–519.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L., 2014. Microsoft COCO: Common objects in context. In: *European Conference on Computer Vision (ECCV)*, LNCS 8693. Springer, Cham, pp. 740–755.

Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S., 2017. Feature pyramid networks for object detection. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2117–2125.

Luo, W., Li, Y., Urtasun, R., Zemel, R., 2016. Understanding the effective receptive field in deep convolutional neural networks. In: *Advances in Neural Information Processing Systems (NeurIPS)*, 29, pp. 4898–4906.

Manigrasso, F., Miro, F.D., Morra, L., Lamberti, F., 2021. Faster-LTN: a neuro-symbolic, end-to-end object detection architecture. In: *International Conference on Artificial Neural Networks (ICANN)*. Springer, pp.40-52.

Minaee, S., Boykov, Y., Porikli, F., Plaza, A., Kehtarnavaz, N., Terzopoulos, D., 2021. Image segmentation using deep learning: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(7), pp.3523-3542.

Nauta, M., Trienes, J., Pathak, S., Nguyen, E., Peters, M., Schmitt, Y., Schlötterer, J., van Keulen, M., Seifert, C., 2023. From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable AI. *ACM Computing Surveys* 55(13s), 295.

Petsiuk, V., Das, A., Saenko, K., 2018. RISE: Randomized input sampling for explanation of black-box models. In: *Proceedings of the British Machine Vision Conference (BMVC)*, pp. 1–13.

Platt, J.C., 1999. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: Advances in Large Margin Classifiers. MIT Press, Cambridge, MA, pp. 61–74.

Prince, S.J.D., 2023. *Understanding Deep Learning*. MIT Press, Cambridge, MA.

Provost, F., Domingos, P., 2003. Tree induction for probability-based ranking. *Machine Learning* 52(3), 199–215.

Ren, S., He, K., Girshick, R., Sun, J., 2017. Faster R-CNN: Towards real-time object detection with region proposal networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 39(6), 1137–1149.

Rudin, C., 2019. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), pp.206-215.

Saadallah, A., Büscher, J., Abdulaaty, O., Panusch, T., Deuse, J., Morik, K., 2022. Explainable Predictive Quality Inspection using Deep Learning in Electronics Manufacturing. Procedia CIRP 107, 594-599.

Samek, W., Montavon, G., Vedaldi, A., Hansen, L.K., Müller, K.-R. (Eds.), 2019. *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning*. LNCS 11700. Springer, Cham.

Schapire, R.E., Singer, Y., 1999. Improved boosting algorithms using confidence-rated predictions. *Machine Learning* 37(3), 297–336.

Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D., 2017. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, pp. 618–626.

Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A.P., Bishop, R., Rueckert, D., Wang, Z., 2016. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1874–1883.

Shrikumar, A., Greenside, P., Kundaje, A., 2017. Learning important features through propagating activation differences. In: *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, pp. 3145–3153.

Sokolova, M., Lapalme, G., 2009. A systematic analysis of performance measures for classification tasks. *Information Processing & Management* 45(4), 427–437.

Sundararajan, M., Taly, A., Yan, Q., 2017. Axiomatic attribution for deep networks. In: *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, pp. 3319–3328.

Szeliski, R., 2022. *Computer Vision: Algorithms and Applications*, 2nd ed. Springer, Cham.

Tang, S., He, F., Huang, X., Yang, J., 2019. Online PCB Defect Detector On A New PCB Defect Dataset. arXiv preprint arXiv:1902.06197.

Tziolas, T., Papageorgiou, K., Theodosiou, T., Ioannidis, D., Dimitriou, N., Tinker, G., Papageorgiou, E., 2025. Explainable AI Methods for Identification of Glue Volume Deficiencies in Printed Circuit Boards. Applied Sciences 15(16), 9061.

Tzionis, G., Mouratidis, P., Kougka, G., Gialampoukidis, I., Vrochidis, S., Kompatsiaris, I., Vlachopoulou, M., 2026. A review of explainable AI methods and their application in manufacturing systems. Discover Applied Sciences 8, 52.

Wang, Y., Huang, J., Dipu, M.S.K., Zhao, H., Gao, S., Zhang, H., Lv, P., 2024. YOLO-RLC: An Advanced Target-Detection Algorithm for Surface Defects of Printed Circuit Boards Based on YOLOv5. Computers, Materials & Continua 80(3), 4973-4995.

Zhang, J., Bargal, S.A., Lin, Z., Brandt, J., Shen, X., Sclaroff, S., 2018. Top-down neural attention by excitation backprop. *International Journal of Computer Vision* 126(10), 1084–1102.

Zhao, X., Wang, L., Zhang, Y., Han, X., Deveci, M., Parmar, M., 2024. A review of convolutional neural networks in computer vision. *Artificial Intelligence Review* 57(4), 99.

Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., Torralba, A., 2016. Learning deep features for discriminative localization. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2921–2929.
