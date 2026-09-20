**Proposal Skripsi**
 
**DETEKSI CACAT PCB DENGAN PENJELASAN YANG FAITHFUL MELALUI ARSITEKTUR NEURO-SYMBOLIC FASTER R-CNN DAN SPARSE OBLIQUE DECISION TREE**
 
**![unsbw](asset:sha256:db793cb0e28d4e64bb5a76831c14d3d3bf3a08b8a42b6a4281c239352bf5bacd)**
 
**Disusun Oleh :**
 
**HANDI DWI CAHYO**
 
**L0122072**
 
**FAKULTAS TEKNOLOGI INFORMASI DAN SAINS DATA**
 
**UNIVERSITAS SEBELAS MARET**
 
**SURAKARTA**
 
**2026**
 
**![](asset:sha256:2a9f5e63ababb9ef749d1f36c8982df5d2906f5ebb1b8a8b90052e8642f8bcc4)**
 
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
 
# BAB I
 
PENDAHULUAN
 
1. Latar Belakang
Perkembangan teknologi elektronik meningkatkan kebutuhan terhadap komponen yang andal dan berkualitas tinggi, khususnya \*Printed Circuit Board \*(PCB). PCB berfungsi sebagai penopang sekaligus penghubung antar-komponen yang menentukan keandalan operasional sistem elektronik. Untuk memenuhi tuntutan keandalan tersebut, PCB kemudian dikembangkan dengan arsitektur yang semakin kompleks. Namun, kompleksitas ini justru meningkatkan risiko munculnya cacat pada PCB akibat kesalahan manusia maupun gangguan mesin selama proses produksi (Chen et al., 2023; Fung et al., 2024). Keberadaan cacat tersebut dapat berdampak pada penurunan kualitas produk, peningkatan biaya produksi, hingga risiko keselamatan manusia (Ali et al., 2024; Fung et al., 2024). Dengan demikian, metode inspeksi menjadi tahapan krusial untuk mendeteksi keberadaan cacat pada PCB.
 
Tahap inspeksi PCB selama ini bertumpu pada metode konvensional seperti pemeriksaan visual manual dan pengujian kelistrikan. Namun, metode-metode tersebut kerap dinilai tidak efisien, memakan biaya tinggi, dan rentan terhadap kesalahan akibat faktor manusia (Ali et al., 2024; Chen et al., 2023; Fung et al., 2024). Keterbatasan tersebut mendorong pengembangan solusi otomatis berbasis *Artificial Intelligence* (AI), khususnya pendekatan *deep learning*. Berbagai model *deep learning* seperti \*You Only Look Once (\*YOLO), *Single Shot MultiBox Detector* (SSD), dan *Faster* \*Region-based Convolutional Neural Network \*(R-CNN) telah banyak diterapkan untuk deteksi cacat PCB dan menunjukkan performa yang tinggi dalam berbagai studi (Chen et al., 2023). Sebagai contoh, Wang et al. (2024) mengembangkan YOLO-RLC, yaitu YOLOv5 dengan jaringan konvolusi kernel besar residual yang meningkatkan akurasi deteksi cacat PCB. Bahkan, penelitian oleh Fung et al. (2024) menunjukkan bahwa optimasi pada arsitektur *Faster* R-CNN dapat meningkatkan performa deteksi cacat PCB, terutama pada cacat yang berukuran kecil. Hasil beberapa penelitian tersebut menunjukkan bahwa pendekatan berbasis *deep learning* merupakan pendekatan yang dapat diandalkan untuk tahap inspeksi.
 
Meskipun pendekatan *deep learning* menunjukkan performa deteksi tinggi, model ini memiliki kelemahan berupa sifat *black-box,* yakni ketidakmampuan model dalam memberikan penjelasan atas setiap keputusan yang dihasilkannya\*. \*Padahal, dalam industri manufaktur berisiko tinggi seperti produksi PCB, penjelasan tersebut diperlukan teknisi untuk memvalidasi dan mempertanggungjawabkan hasil deteksi  (Tzionis et al., 2026). Tanpa penjelasan yang memadai, proses validasi menjadi sulit dilakukan sehingga berpotensi menurunkan kepercayaan teknisi terhadap model (Tziolas et al., 2025). Oleh karena itu, kemampuan model dalam menjelaskan keputusan (*explainability*) menjadi kebutuhan esensial guna mendukung proses validasi teknisi dalam inspeksi cacat PCB (Saadallah et al., 2022; Chen et al., 2023).
 
Untuk merespons kebutuhan *explainability* tersebut, *Explainable Artificial Intelligence* (XAI) menjadi pendekatan yang dominan diterapkan dalam sistem inspeksi manufaktur (Tzionis et al., 2026). Pendekatan XAI menghadirkan *explainability* dengan membuat proses pengambilan keputusan model AI dapat ditelusuri dan dipahami. Dalam inspeksi PCB, penelitian Tziolas et al. (2025) menunjukkan bahwa *Deep* *Shapley Additive Explanations* (SHAP) dan *Gradient-weighted Class Activation Mapping* (Grad-CAM\*)\* dapat menyoroti area visual yang memengaruhi keputusan model CNN, sehingga meningkatkan interpretabilitas model bagi teknisi. Penelitian lain oleh Saadallah et al. (2022) menggunakan penjelasan berbasis *heatmap* yang mengungkap fitur-fitur penting pada citra PCB, sehingga membantu teknisi dalam memvalidasi relevansi fitur tersebut terhadap jenis cacat yang terdeteksi. Dengan demikian, XAI mampu memenuhi kebutuhan *explainability* pada inspeksi PCB melalui justifikasi visual atas keputusan model.
 
Meskipun informatif, metode XAI Grad-CAM dan SHAP pada studi tersebut, masih memiliki keterbatasan berupa sifat *post-hoc*, yaitu penjelasan yang baru diberikan setelah model *black-box* menghasilkan prediksinya. Mekanisme ini menyebabkan penjelasan yang dihasilkan tidak memenuhi *faithfulness*, yaitu kemampuannya dalam merepresentasikan keputusan model yang sesungguhnya, sehingga penjelasan tersebut hanyalah berupa perkiraan (Rudin, 2019). Pada domain berisiko tinggi, peta *saliency* *post-hoc*, termasuk Grad-CAM, terbukti belum sepenuhnya dapat diandalkan (Arun et al., 2021), sehingga validasi teknisi yang bertumpu padanya dalam inspeksi PCB berpotensi mengarah pada keputusan yang salah. Dengan demikian, ketiadaan *faithfulness* dalam penjelasan *post-hoc* justru mengurangi kualitas *explainability* yang dibutuhkan untuk validasi teknisi (Rudin, 2019).
 
Untuk mengatasi keterbatasan *faithfulness* dalam penjelasan model, arsitektur *neuro-symbolic* hadir dengan menggabungkan ekstraksi fitur dari *deep learning* dan penalaran transparan berbasis simbolik. Dalam arsitektur ini, komponen simbolik terintegrasi langsung ke dalam mekanisme pengambilan keputusan sehingga penjelasan yang dihasilkan bersifat *faithful* (d'Avila Garcez & Lamb, 2023). Arsitektur ini telah ditunjukkan oleh model *Faster*-LTN yang mengintegrasikan *Faster* R-CNN dengan *Logic Tensor Network* (LTN) untuk mempertahankan performa deteksi sekaligus memberikan penalaran terstruktur (Manigrasso et al., 2021). Sejalan dengan arah tersebut, Hada et al. (2024) serta Kairgeldin dan Carreira-Perpiñán (2025) mengembangkan integrasi CNN dengan *sparse oblique decision tree* (SODT) yang membuat proses pengambilan keputusan model lebih dapat diinterpretasikan dan divisualisasikan. Dengan demikian, ketiga penelitian tersebut berpotensi menghadirkan *faithfulness* sehingga meningkatkan kualitas *explainability* pada inspeksi PCB.
 
Untuk mewujudkan *explainability* yang *faithfulness*, penelitian ini mengusulkan integrasi *Faster* R‑CNN dengan SODT sebagai sistem deteksi cacat berbasis arsitektur *neuro‑symbolic*. *Faster* R‑CNN dipilih karena terbukti efektif mendeteksi cacat berukuran kecil pada PCB melalui optimalisasi SF‑PSPyramid (Fung et al., 2024), sehingga berperan sebagai komponen ekstraksi fitur visual dan deteksi objek. Sementara itu, SODT dipilih karena kemampuannya dalam meniru (*mimic)* keputusan jaringan saraf *teacher* dengan akurasi tinggi, namun dengan mekanisme eliminasi fitur (*sparsity*) yang menghasilkan struktur pohon lebih sederhana (Hada et al., 2024; Kairgeldin & Carreira‑Perpiñán, 2025). Integrasi ini dirancang untuk mempertahankan performa deteksi tinggi dari *Faster* R‑CNN sekaligus menghadirkan penjelasan yang *faithful*. Dengan demikian, sistem ini ditujukan untuk mendukung validasi teknisi dalam inspeksi cacat PCB.
 
2. Rumusan Masalah
Berdasarkan latar belakang yang telah diuraikan, rumusan masalah dalam penelitian ini adalah sebagai berikut.
 
1. Bagaimana model *deep learning* dapat mempertahankan performa deteksi tinggi pada inspeksi PCB sekaligus mengatasi sifat *black‑box* yang menghambat validasi teknisi?
2. Bagaimana arsitektur *neuro‑symbolic* yang mengintegrasikan *Faster* R‑CNN dengan *Sparse Oblique Decision Tree* (SODT) dapat dikembangkan untuk menghadirkan sistem deteksi cacat PCB yang akurat sekaligus menyediakan *explainability* yang *faithful*?
## Batasan Masalah
 
Agar penelitian ini tetap terarah dan fokus sesuai dengan tujuan yang telah ditetapkan, ruang lingkup permasalahan dibatasi pada hal-hal sebagai berikut.
 
1. Objek Penelitian
Penelitian difokuskan pada deteksi enam jenis cacat visual PCB, yaitu *open*, *short*, *mousebite*, *spur*, *pinhole*, dan *spurious copper*, menggunakan pendekatan *deep learning* dan *neuro‑symbolic*.
 
2. Arsitektur Model
Model merupakan integrasi Faster R‑CNN (mengacu pada implementasi Fung et al., 2024) dengan *Sparse Oblique Decision Tree* (SODT) tanpa modifikasi terhadap struktur internal kedua komponen.
 
3. Sumber Data
Data yang digunakan berasal dari dataset publik DeepPCB yang memuat 1.500 pasang citra beserta anotasi posisi dan kelas cacat.
 
1. Metode Evaluasi *Explainability*
Evaluasi kualitas *explainability* dilakukan secara kuantitatif melalui perbandingan dengan Grad‑CAM sebagai *baseline post‑hoc* yang telah teruji, tanpa melibatkan studi pengguna atau wawancara teknisi.
 
## Tujuan Penelitian
 
1. Mengembangkan sistem deteksi cacat PCB berbasis integrasi Faster R CNN dan Sparse Oblique Decision Tree (SODT) yang mampu mempertahankan performa deteksi tinggi sekaligus menyediakan penjelasan yang faithful guna mendukung proses validasi teknisi.
2. Mengevaluasi apakah sistem yang diusulkan mampu menghasilkan penjelasan yang *faithful* dan lebih baik dibandingkan pendekatan *post-hoc* Grad-CAM.

## Manfaat Penelitian
 
1. Mendukung proses validasi teknisi di industri manufaktur elektronik melalui sistem deteksi cacat PCB yang tidak hanya akurat, tetapi juga menyediakan penjelasan yang *faithful* dan dapat dipertanggungjawabkan.
2. Memberikan bukti empiris bahwa integrasi Faster R CNN dan *Sparse Oblique Decision Tree* (SODT) mampu mengatasi keterbatasan *faithfulness* yang melekat pada pendekatan XAI *post-hoc*.
# BAB II

# TINJAUAN PUSTAKA

## 2.1 Dasar Teori

### 2.1.1 *Printed Circuit Board* (PCB)

*Printed Circuit Board* (PCB) adalah papan dari bahan isolator yang dilapisi tembaga. PCB berfungsi sebagai jalur penghubung listrik sekaligus penyangga antarkomponen elektronik (Coombs & Holden, 2016; Khandpur, 2005).

Kriteria inspeksi PCB di industri mengacu pada standar *Association Connecting Electronics Industries* (IPC), khususnya IPC-A-600 dan IPC-6012 (IPC, 2015, 2020). Standar ini menyatakan suatu kondisi sebagai cacat apabila melanggar batas toleransi, misalnya jalur konduktor yang terlalu sempit atau jarak antarjalur yang terlalu dekat. Dalam *Computer Vision*, pelanggaran tersebut dipandang sebagai cacat visual yang polanya dapat dipelajari oleh model *deep learning* (Tang et al., 2019; Chen et al., 2023). Enam jenis cacat yang umum dipakai pada penelitian deteksi PCB dirangkum pada Tabel 2.1.

Tabel 2.1 Kategori Cacat Visual Umum pada PCB

| Jenis Cacat | Definisi & Karakteristik Visual |
| --- | --- |
| *Open* | Jalur tembaga terputus, sehingga muncul celah yang memutus jalur. |
| *Short* | Dua jalur yang seharusnya terpisah menjadi tersambung, membentuk jembatan tembaga. |
| *Mousebite* | Tepi jalur tergerus seperti gigitan, sehingga lebar jalur berkurang secara tidak merata. |
| *Spur* | Tonjolan tembaga kecil yang keluar dari tepi jalur menuju area yang seharusnya kosong. |
| *Pinhole* | Lubang kecil berbentuk lingkaran di dalam area tembaga yang seharusnya padat. |
| *Spurious Copper* | Tembaga liar yang muncul terpisah di area non-konduktif, tidak terhubung ke jalur utama. |

Keenam cacat tersebut terbagi menjadi dua kelompok pola. Kelompok pertama adalah pengurangan material (*open*, *mousebite*, *pinhole*), yaitu hilangnya sebagian tembaga. Kelompok kedua adalah penambahan material liar (*short*, *spur*, *spurious copper*), yaitu munculnya tembaga di tempat yang salah.

### 2.1.2 *Computer Vision*

*Computer Vision* (CV) adalah cabang ilmu komputer yang membuat komputer mampu memahami isi gambar secara otomatis (Szeliski, 2022). Berbeda dengan pengolahan citra yang hanya memanipulasi piksel, CV bertujuan menghasilkan makna atau keputusan dari isi gambar (Prince, 2023). CV mencakup tiga tugas utama yang dibedakan berdasarkan kedetailan keluarannya.

1. **Klasifikasi Citra** (*Image Classification*) memberikan satu label kelas untuk keseluruhan gambar (Prince, 2023).
2. **Deteksi Objek** (*Object Detection*) mengenali objek, menentukan kelasnya, dan menunjukkan posisinya dengan kotak pembatas (*bounding box*) (Szeliski, 2022).
3. **Segmentasi Citra** (*Image Segmentation*) memberi label pada setiap piksel, sehingga batas objek tergambar lebih presisi (Minaee et al., 2022).

Perbandingan ketiga tugas tersebut ditunjukkan pada Gambar 2.1.

![Gambar 2.1](asset:sha256:84fdf18c7d575a11c676bdf35cf7f7774f80493c98a8fc18a65985515c823ad6)

Gambar 2.1 Perbandingan Klasifikasi, Deteksi Objek, dan Segmentasi

Deteksi objek merupakan tugas yang relevan untuk inspeksi visual, karena selain mengenali jenis cacat juga menunjukkan lokasinya. Representasi dan sistem koordinat *bounding box* ditunjukkan pada Gambar 2.2.

![Gambar 2.2](asset:sha256:27deefeb1e31bd72f42d29eeefbc61c0e0a95508b5f607e03bbca086fa9ec2bc)

Gambar 2.2 Anatomi dan Representasi Koordinat *Bounding Box*

### 2.1.3 *Deep Learning*

*Deep learning* adalah cabang *machine learning* yang mempelajari representasi data secara bertingkat melalui banyak lapisan pemrosesan (Goodfellow et al., 2016). *Machine learning* konvensional bergantung pada fitur rancangan manusia, sedangkan *deep learning* memperoleh fitur secara otomatis dari data mentah (Han et al., 2017). Perbedaan alur kerja keduanya ditunjukkan pada Gambar 2.3.

![Gambar 2.3](asset:sha256:49413e3bab1c8a3956aeab46285530cdd02bab7e2cc1dd99671d1a088d7a15f4)

Gambar 2.3 Perbedaan Cara Kerja *Machine Learning* dan *Deep Learning*

Pelatihan jaringan saraf tiruan terdiri atas empat komponen utama.

**1. *Forward Propagation***

Propagasi maju menghasilkan prediksi dari data masukan melalui operasi berlapis. Pada setiap lapisan, keluaran lapisan sebelumnya dikalikan dengan matriks bobot, ditambah bias, lalu dilewatkan ke fungsi aktivasi (Goodfellow et al., 2016).

**2. Fungsi Aktivasi**

Fungsi aktivasi memberikan sifat non-linear pada jaringan sehingga model mampu memodelkan hubungan yang kompleks. Fungsi yang umum digunakan adalah *Rectified Linear Unit* (ReLU) pada Persamaan 2.1, karena ringan secara komputasi dan mengurangi masalah gradien menghilang (*vanishing gradient*) (LeCun et al., 2015).

$$
f\left(x\right)=\max{\left(0,x\right)}
$$

(2.1)

dengan $x$ nilai masukan dan $f\left(x\right)$ nilai keluaran *neuron*.

**3. *Loss Function***

Fungsi kerugian mengukur selisih antara prediksi model dan nilai sebenarnya (*ground truth*). Untuk klasifikasi banyak kelas, fungsi kerugian yang umum digunakan adalah *Cross-Entropy Loss* pada Persamaan 2.2 (Goodfellow et al., 2016).

$$
L=-\sum_{i=1}^{C}{{y}_{i}}\log{\left({p}_{i}\right)}
$$

(2.2)

dengan $C$ jumlah kelas, ${y}_{i}$ label sebenarnya kelas ke-*i* dalam bentuk *one-hot encoding*, dan ${p}_{i}$ probabilitas prediksi kelas ke-*i*.

**4. *Backpropagation***

Propagasi mundur menghitung gradien fungsi kerugian terhadap setiap parameter menggunakan aturan rantai (*chain rule*), lapisan demi lapisan dari keluaran menuju masukan (LeCun et al., 2015). Gradien terhadap parameter dipakai oleh *Stochastic Gradient Descent* (SGD) untuk memperbarui parameter, sebagaimana dinyatakan pada Persamaan 2.3 (Goodfellow et al., 2016).

$$
w\leftarrow w-\eta \frac{\partial L}{\partial w}
$$

(2.3)

dengan $w$ parameter bobot, $\eta$ laju pembelajaran (*learning rate*), dan $\frac{\partial L}{\partial w}$ gradien fungsi kerugian terhadap bobot.

Siklus propagasi maju dan mundur diulang hingga model konvergen, sebagaimana diilustrasikan pada Gambar 2.4.

![Gambar 2.4](asset:sha256:b79cf9490d1205c03846c19c7b03e6afd713a834cd06ad2b4268e18c6169ab4d)

Gambar 2.4 Diagram Alur Kerja Siklus Pelatihan Jaringan Saraf Tiruan

### 2.1.4 *Convolutional Neural Network* (CNN)

*Convolutional Neural Network* (CNN) adalah jaringan saraf tiruan yang dirancang untuk data berbentuk grid, seperti citra (Goodfellow et al., 2016; LeCun et al., 2015). Efisiensinya bertumpu pada konektivitas lokal, yaitu setiap *neuron* hanya melihat sebagian kecil citra, dan berbagi parameter, yaitu filter yang sama dipakai di seluruh citra (LeCun et al., 2015). Arsitektur umum CNN ditunjukkan pada Gambar 2.5.

![Gambar 2.5](asset:sha256:446485431913c5ed7c52f4f6057535edf806a51c8360f7efe5fe3f7eccce91a6)

Gambar 2.5 Ilustrasi Arsitektur *Convolutional Neural Network* (CNN)

Arsitektur CNN terdiri atas tiga jenis lapisan utama.

1. ***Convolutional Layer*** mengekstraksi fitur lokal dengan menggeser filter (*kernel*) terpelajar di atas citra atau peta fitur. Setiap filter mengenali pola tertentu seperti tepi, sudut, atau tekstur (Zhao et al., 2024; Goodfellow et al., 2016). Keluarannya dilewatkan ke fungsi aktivasi seperti ReLU (Persamaan 2.1).
2. ***Pooling Layer*** memperkecil dimensi spasial peta fitur (*downsampling*), misalnya dengan *max pooling* yang mengambil nilai terbesar pada setiap jendela, sehingga beban komputasi berkurang dan model lebih tahan terhadap pergeseran kecil.
3. ***Fully Connected Layer*** berupa *Multi-Layer Perceptron* (MLP) yang menggabungkan fitur tingkat tinggi menjadi keputusan akhir. Masukannya berupa vektor hasil perataan (*flatten*) peta fitur.

Bagian konvolusi berperan sebagai pengekstraksi fitur (*feature extractor*), sedangkan MLP berperan sebagai kepala klasifikasi (*classifier head*). Skor mentah (*logit*) keluaran kepala klasifikasi diubah menjadi probabilitas oleh fungsi *softmax* pada Persamaan 2.4.

$$
{p}_{i}=\frac{{e}^{{z}_{i}}}{\sum_{j=1}^{C}{{e}^{{z}_{j}}}}
$$

(2.4)

dengan ${p}_{i}$ probabilitas kelas ke-*i*, ${z}_{i}$ *logit* kelas ke-*i*, dan $C$ jumlah kelas.

Lapisan konvolusi dan *pooling* menghasilkan peta fitur (*feature map*) berukuran $C\times H\times W$. Setiap kanal merupakan respons satu filter, sedangkan posisi $\left(h,w\right)$ menyatakan lokasi respons tersebut (Goodfellow et al., 2016). *Stride* adalah besar langkah pergeseran filter. *Stride* total lapisan ke-*l* dan posisi citra yang berkorespondensi dengan sel $\left(h,w\right)$ dinyatakan pada Persamaan 2.5 dan 2.6 (Goodfellow et al., 2016).

$$
{S}_{l}=\prod_{i=1}^{l}{{s}_{i}}
$$

(2.5)

$$
\left(u,v\right)=\left({S}_{l}\cdot w,{S}_{l}\cdot h\right)
$$

(2.6)

dengan ${S}_{l}$ *stride* total lapisan ke-*l*, ${s}_{i}$ *stride* lapisan ke-*i*, $\left(h,w\right)$ indeks baris dan kolom pada peta fitur, dan $\left(u,v\right)$ koordinat pada citra.

Persamaan 2.6 baru menetapkan titik pusat korespondensi, belum luas daerah citra yang benar-benar dibaca oleh satu sel. Luas tersebut dinyatakan oleh *receptive field*, yaitu daerah citra yang memengaruhi nilai satu sel peta fitur (Goodfellow et al., 2016; Luo et al., 2016). Ukurannya membesar seiring kedalaman jaringan, sebagaimana dinyatakan pada Persamaan 2.7.

$$
{r}_{l}={r}_{l-1}+\left({k}_{l}-1\right)\prod_{i=1}^{l-1}{{s}_{i}},\quad {r}_{0}=1
$$

(2.7)

dengan ${r}_{l}$ ukuran *receptive field* lapisan ke-*l* dalam piksel, ${k}_{l}$ ukuran kernel lapisan ke-*l*, dan ${s}_{i}$ *stride* lapisan ke-*i*.

*Stride* total (Persamaan 2.5) menentukan letak pusat daerah citra yang diwakili satu sel peta fitur, sedangkan *receptive field* (Persamaan 2.7) menentukan luas daerah tersebut. Keduanya membuat setiap sel dapat dipetakan kembali ke daerah citranya sendiri, sehingga bobot atau atribusi pada sel peta fitur dapat diterjemahkan menjadi penjelasan spasial pada citra (Subbab 2.1.8). Namun, *receptive field* antarsel saling tumpang-tindih dan pengaruh piksel di dalamnya menurun dari pusat ke tepi (Luo et al., 2016), sehingga penjelasan tersebut berlaku pada tingkat daerah, bukan pada piksel tunggal. Ilustrasi *stride* dan *receptive field* ditunjukkan pada Gambar 2.6.

[SISIPKAN GAMBAR: satu sel peta fitur menempati petak ${S}_{l}\times {S}_{l}$ piksel pada citra, sedangkan *receptive field*-nya mencakup daerah citra yang lebih luas dan tumpang-tindih dengan *receptive field* sel tetangga]

Gambar 2.6 Ilustrasi *Stride* dan *Receptive Field*

### 2.1.5 *Faster* R-CNN

*Faster* R-CNN adalah arsitektur deteksi objek dua tahap (*two-stage*) yang menyatukan pengusulan area dan klasifikasi dalam satu jaringan yang dilatih secara *end-to-end* (Ren et al., 2017). Keunggulannya terletak pada *Region Proposal Network* (RPN) yang menggantikan metode pencarian area eksternal, sehingga seluruh komponen berbagi peta fitur yang sama. Arsitektur standarnya ditunjukkan pada Gambar 2.7.

![Gambar 2.7](asset:sha256:bdac8981ec742c01d0eaadc2a8feb3fc759bc3ee18000b0303f24901409bf56e)

Gambar 2.7 Arsitektur *Faster* R-CNN Standar

Penelitian ini menggunakan varian SF-PSPyramid (Fung et al., 2024), yaitu *Faster* R-CNN dengan *neck* yang dirancang untuk cacat berukuran mikro pada PCB, sebagaimana ditunjukkan pada Gambar 2.8.

![Gambar 2.8](asset:sha256:edc9dd932783a061b34bf4f25136acbe132e08ad8c0c6a2dcdbb34255633bf75)

Gambar 2.8 Arsitektur Modifikasi *Faster* R-CNN dengan SF-PSPyramid

**1. *Backbone***

*Backbone* mengubah citra menjadi peta fitur. Penelitian ini menggunakan ResNet-50, yang terdiri atas empat kelompok lapisan (C2, C3, C4, C5) dengan resolusi menurun dan makna semantik meningkat (He et al., 2016). *Stride* total (Persamaan 2.5) keempat kelompok tersebut berturut-turut 4, 8, 16, dan 32 piksel.

**2. *Neck* (SF-PSPyramid)**

*Neck* menggabungkan fitur dari berbagai skala *backbone* menjadi piramida fitur. Dasarnya adalah *Feature Pyramid Network* (FPN), yang menggabungkan jalur *bottom-up*, jalur *top-down*, dan koneksi lateral antartingkat (Lin et al., 2017). SF-PSPyramid menyempurnakan FPN dengan tiga perbedaan (Fung et al., 2024).

a. *CP Block*. Resolusi diperbesar melalui penataan ulang kanal (*pixel shuffle*) (Shi et al., 2016), bukan interpolasi, sehingga bersifat terpelajar (Persamaan 2.8).

$$
PS{\left(T\right)}_{c,h,w}={T}_{c\cdot {r}^{2}+r\cdot \mathrm{mod}\left(h,r\right)+\mathrm{mod}\left(w,r\right),\left\lfloor h/r\right\rfloor,\left\lfloor w/r\right\rfloor}
$$

(2.8)

dengan $T$ tensor masukan, $r$ faktor pembesaran, serta $c,h,w$ indeks kanal, baris, dan kolom keluaran.

b. *Selective Feature Attention*. Dua tingkat fitur digabungkan dengan bobot terpelajar yang dinormalisasi *softmax*, setelah tingkat yang lebih dalam diperbesar ke resolusi tingkat lainnya dengan interpolasi *nearest*, mengadaptasi *Selective Kernel Network* (Li et al., 2019), sebagaimana dinyatakan pada Persamaan 2.9.

$$
P'={\alpha}_{1}\odot U+{\alpha}_{2}\odot V,\quad \left[{\alpha}_{1},{\alpha}_{2}\right]=\mathrm{softmax}\left({W}_{2}\,\delta \left({W}_{1}z\right)\right)
$$

(2.9)

dengan $U$ peta fitur dari tingkat yang lebih dalam, $V$ peta fitur beresolusi lebih tinggi, $z$ vektor hasil *global average pooling*, ${W}_{1},{W}_{2}$ bobot lapisan kompresi dan perluasan, $\delta$ ReLU, dan $\odot$ perkalian per kanal.

c. Tanpa koneksi lateral konvensional. P2′ dan P3′ dibentuk dari tingkat yang lebih dalam, sehingga P2′ memuat informasi C2 hingga C5 yang membantu deteksi cacat sangat kecil (Fung et al., 2024), sebagaimana ditunjukkan pada Gambar 2.8.

Keluaran *neck* adalah P2′, P3′, P4, P5, dan P6 (*max pooling* dari P5) dengan *stride* 4, 8, 16, 32, dan 64 piksel dan jumlah kanal $C$ yang sama.

**3. *Region Proposal Network* (RPN)**

RPN menghasilkan usulan area kandidat objek (*proposal*). Pada setiap posisi peta fitur disiapkan sejumlah *anchor box* dengan beragam ukuran dan rasio, lalu setiap *anchor* memperoleh skor objektivitas dan empat nilai penyesuaian koordinat (Ren et al., 2017) pada Persamaan 2.10 sampai 2.13.

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

dengan $x,y,w,h$ pusat, lebar, dan tinggi kotak prediksi, serta ${x}_{a},{y}_{a},{w}_{a},{h}_{a}$ milik *anchor*. Pelatihan memakai *multi-task loss* pada Persamaan 2.14 (Ren et al., 2017).

$$
L=\frac{1}{{N}_{cls}}\sum_{i}{{L}_{cls}\left({p}_{i},{p}_{i}^{*}\right)}+\lambda \frac{1}{{N}_{reg}}\sum_{i}{{p}_{i}^{*}{L}_{reg}\left({t}_{i},{t}_{i}^{*}\right)}
$$

(2.14)

dengan ${p}_{i}$ probabilitas *anchor* ke-*i* memuat objek, ${p}_{i}^{*}$ label sebenarnya (1 positif, 0 negatif), ${N}_{cls},{N}_{reg}$ jumlah sampel tiap suku, ${L}_{cls}$ *binary cross-entropy*, ${L}_{reg}$ kerugian regresi, dan $\lambda$ faktor penyeimbang. Kerugian regresi yang digunakan adalah L1 pada Persamaan 2.15 (Fung et al., 2024).

$$
{L}_{reg}\left(t,{t}^{*}\right)=\sum_{j}{\left|{t}_{j}-{t}_{j}^{*}\right|}
$$

(2.15)

dengan $j$ merentang pada komponen $x$, $y$, $w$, dan $h$.

**4. *RoI Align***

*RoI Align* menyeragamkan setiap proposal menjadi tensor berukuran tetap tanpa pembulatan koordinat (He et al., 2020). Nilai peta fitur pada koordinat pecahan diperoleh dengan interpolasi bilinear pada Persamaan 2.16.

$$
f\left(x,y\right)=\sum_{i=1}^{4}{{w}_{i}{f}_{i}}
$$

(2.16)

dengan ${f}_{i}$ nilai fitur pada empat titik grid terdekat dan ${w}_{i}$ bobot interpolasi yang berbanding terbalik dengan jarak.

Setiap proposal dibagi menjadi grid $G\times G$ *bin*. Pada setiap *bin* diambil beberapa titik sampel, dan nilai *bin* adalah rata-rata titik sampelnya (Persamaan 2.17) (He et al., 2020).

$$
{x}_{c,p,q}=\frac{1}{N}\sum_{n=1}^{N}{\sum_{i=1}^{4}{{w}_{n,i}\,{F}_{c}\left({h}_{n,i},{v}_{n,i}\right)}}
$$

(2.17)

dengan ${x}_{c,p,q}$ keluaran kanal ke-*c* pada *bin* $\left(p,q\right)$, $N$ jumlah titik sampel per *bin*, ${w}_{n,i}$ bobot interpolasi titik sampel ke-*n*, dan ${F}_{c}\left({h}_{n,i},{v}_{n,i}\right)$ nilai peta fitur kanal ke-*c* pada titik grid terdekat. Dalam bentuk matriks, Persamaan 2.17 dapat ditulis sebagai Persamaan 2.18.

$$
{x}_{c}=A\,{F}_{c}
$$

(2.18)

dengan ${x}_{c}$ vektor keluaran kanal ke-*c* ($G\times G$ nilai), ${F}_{c}$ vektor nilai peta fitur kanal ke-*c*, dan $A$ matriks koefisien interpolasi yang hanya bergantung pada proposal dan sama untuk setiap kanal.

Pada piramida fitur, tingkat yang dipakai untuk setiap proposal dipilih dengan Persamaan 2.19 (Lin et al., 2017).

$$
k=\left\lfloor {k}_{0}+{\log}_{2}{\left(\frac{\sqrt{wh}}{224}\right)}\right\rfloor
$$

(2.19)

dengan $k$ tingkat terpilih, ${k}_{0}=4$ tingkat acuan untuk proposal 224×224 piksel, serta $w,h$ lebar dan tinggi proposal.

**5. *Box Head* (Klasifikasi dan Regresi *Bounding Box*)**

Keluaran *RoI Align* setiap proposal diratakan menjadi vektor, lalu diproses oleh *box head* berupa MLP dengan dua lapisan terhubung penuh yang menghasilkan representasi RoI (Girshick, 2015; Ren et al., 2017).

Representasi RoI diteruskan ke dua lapisan keluaran yang bekerja berdampingan (*two sibling output layers*) (Girshick, 2015; Ren et al., 2017). Cabang pertama adalah *classifier* yang menghasilkan skor untuk $C+1$ kelas (termasuk *background*) melalui *softmax* (Persamaan 2.4) dan dilatih dengan *cross-entropy* (Persamaan 2.2). Cabang kedua adalah *regressor* yang menghasilkan empat nilai koreksi koordinat untuk setiap kelas (Persamaan 2.10 sampai 2.13) dan dilatih dengan kerugian L1 (Persamaan 2.15).

**6. *Soft*-NMS**

Kotak dan skor keluaran *box head* kemudian disaring dengan *Soft*-NMS, yang menurunkan skor kandidat yang tumpang-tindih secara bertahap, bukan menghapusnya seketika seperti NMS konvensional (Bodla et al., 2017), sehingga cacat yang berdekatan tidak ikut terbuang (Persamaan 2.20).

$$
{s}_{i}=\begin{cases}{s}_{i}, & IoU\left(M,{b}_{i}\right)<{N}_{t}\\ {s}_{i}\left(1-IoU\left(M,{b}_{i}\right)\right), & IoU\left(M,{b}_{i}\right)\ge {N}_{t}\end{cases}
$$

(2.20)

dengan ${s}_{i}$ dan ${b}_{i}$ skor dan kotak kandidat ke-*i*, $M$ kandidat berskor tertinggi, dan ${N}_{t}$ ambang IoU.

### 2.1.6 *Explainable Artificial Intelligence* (XAI)

*Explainable Artificial Intelligence* (XAI) adalah bidang yang mengembangkan cara agar keputusan model kecerdasan buatan dapat dipahami manusia (Barredo Arrieta et al., 2020). XAI dibutuhkan karena model *deep learning* bersifat *black-box* (Samek et al., 2019). Berdasarkan waktu penjelasan dibentuk, XAI terbagi atas dua pendekatan.

1. **Pendekatan *post-hoc*** menghasilkan penjelasan setelah model memprediksi tanpa mengubah struktur model, misalnya LIME, SHAP, dan Grad-CAM. Penjelasannya bersifat aproksimasi dan tidak dijamin mencerminkan keputusan internal model (Guidotti et al., 2019; Rudin, 2019).
2. **Pendekatan *ante-hoc*** menanamkan kemampuan menjelaskan ke dalam struktur model, misalnya pohon keputusan, model linear, dan arsitektur *neuro-symbolic* (Barredo Arrieta et al., 2020).

Penjelasan untuk citra umumnya berupa peta atribusi (*attribution map*), yaitu peta besar kontribusi setiap posisi terhadap skor kelas (Bach et al., 2015). Peta atribusi yang divisualisasikan dengan skala warna di atas citra disebut *heatmap*. Dua metode atribusi yang relevan dijelaskan berikut.

**1. *Gradient-weighted Class Activation Mapping* (Grad-CAM)**

Grad-CAM menghasilkan peta atribusi spesifik-kelas tanpa mengubah maupun melatih ulang model (Selvaraju et al., 2017). Bobot kepentingan peta fitur ke-*k* terhadap kelas *c* dan peta atribusinya dinyatakan pada Persamaan 2.21 dan 2.22.

$$
{\alpha}_{k}^{c}=\frac{1}{Z}\sum_{i}{\sum_{j}{\frac{\partial {y}^{c}}{\partial {A}_{ij}^{k}}}}
$$

(2.21)

$$
{L}_{Grad\text{-}CAM}^{c}=\mathrm{ReLU}\left(\sum_{k}{{\alpha}_{k}^{c}{A}^{k}}\right)
$$

(2.22)

dengan ${y}^{c}$ skor kelas *c* sebelum *softmax*, ${A}^{k}$ peta fitur ke-*k* pada lapisan konvolusi acuan, ${A}_{ij}^{k}$ nilainya pada posisi $\left(i,j\right)$, dan $Z$ jumlah posisi spasial. Contoh hasil Grad-CAM ditunjukkan pada Gambar 2.9.

![Gambar 2.9](asset:sha256:8f273c9799298c27a9a31eddec7ec6083f75417c8b4d8898014f0387f0aa87e4)

[CEK: gambar ini dipasang berdasarkan pola pergeseran gambar hasil ekspor, tanpa teks OCR; pastikan isinya contoh Grad-CAM]

Gambar 2.9 Hasil Penjelasan dari Grad-CAM (Selvaraju et al., 2017)

**2. *Gradient × Input***

*Gradient × Input* menghitung kontribusi setiap elemen masukan sebagai hasil kali nilai elemen dan gradien keluaran terhadap elemen tersebut (Shrikumar et al., 2017; Ancona et al., 2019), sebagaimana dinyatakan pada Persamaan 2.23.

$$
{R}_{j}\left(x\right)={x}_{j}\cdot \frac{\partial f\left(x\right)}{\partial {x}_{j}}
$$

(2.23)

dengan ${R}_{j}$ kontribusi elemen ke-*j*, ${x}_{j}$ nilai elemen ke-*j*, dan $f\left(x\right)$ keluaran model.

Metode atribusi yang baik diharapkan memenuhi sifat *completeness*, yaitu jumlah seluruh atribusi sama dengan selisih keluaran model pada masukan dan pada masukan acuan (*baseline*) $\bar{x}$ (Sundararajan et al., 2017), sebagaimana dinyatakan pada Persamaan 2.24.

$$
\sum_{j}{{R}_{j}\left(x\right)}=f\left(x\right)-f\left(\bar{x}\right)
$$

(2.24)

*Gradient × Input* umumnya tidak memenuhi sifat ini pada model non-linear, tetapi memenuhinya secara eksak pada model linear $f\left(x\right)={w}^{T}x+b$ dengan *baseline* nol (Ancona et al., 2019), sebagaimana dinyatakan pada Persamaan 2.25.

$$
\sum_{j}{{w}_{j}{x}_{j}}={w}^{T}x=f\left(x\right)-f\left(0\right)
$$

(2.25)

### 2.1.7 *Neuro-Symbolic* AI (NeSy)

*Neuro-Symbolic* AI (NeSy) menggabungkan pembelajaran statistik jaringan saraf dengan penalaran berbasis aturan pada sistem simbolik (d'Avila Garcez & Lamb, 2023), untuk menyatukan kemampuan mengenali pola dari data mentah dengan transparansi penalaran. Arsitektur NeSy ditunjukkan pada Gambar 2.10.

![Gambar 2.10](asset:sha256:4c08b3e4ac581c8457efc8e8266669560070a370e66a079f18ff8becd945c206)

Gambar 2.10 Arsitektur *Neuro-Symbolic*

Arsitektur NeSy terdiri atas dua komponen utama.

1. **Komponen *neural*** mengubah data mentah menjadi representasi fitur numerik (d'Avila Garcez & Lamb, 2023).
2. **Komponen simbolik** mengolah representasi fitur melalui aturan yang dapat ditelusuri. Penjelasannya *faithful*, yaitu merupakan proses keputusan model itu sendiri, bukan aproksimasi (Rudin, 2019). Contohnya *Logic Tensor Network* dan model hibrida CNN dengan pohon keputusan (Manigrasso et al., 2021; Hada et al., 2024).

Komponen simbolik dapat dibentuk melalui ***model mimicking***, yaitu melatih model sederhana untuk meniru keluaran model kompleks (*teacher*), dengan label pelatihan berupa prediksi *teacher*, bukan label sebenarnya (Buciluǎ et al., 2006).

Komponen *neural* menghasilkan keluaran kontinu, sedangkan komponen simbolik menghasilkan keluaran diskrit, sehingga banyak masukan berbeda memperoleh keluaran identik dan informasi tingkat keyakinan hilang (Provost & Domingos, 2003). Pada komponen simbolik dengan fungsi keputusan bernilai riil, keyakinan dapat dinyatakan melalui dua konsep berikut.

**1. *Margin***

Pada fungsi keputusan bernilai riil, tanda keluaran menyatakan label, sedangkan besarnya menyatakan keyakinan. Keluaran yang dekat dengan nol berarti keyakinan rendah, dan yang jauh dari nol berarti keyakinan tinggi (Schapire & Singer, 1999). Besar *margin* dinyatakan pada Persamaan 2.26.

$$
m\left(x\right)=\left|f\left(x\right)\right|
$$

(2.26)

dengan $f\left(x\right)$ fungsi keputusan linear.

**2. Fungsi *Sigmoid***

Fungsi *sigmoid* adalah fungsi monoton naik yang memetakan nilai riil ke selang (0, 1) (Platt, 1999), sebagaimana dinyatakan pada Persamaan 2.27.

$$
\sigma \left(z\right)=\frac{1}{1+{e}^{-z}}
$$

(2.27)

dengan $z$ nilai masukan dan $e$ bilangan Euler.

### 2.1.8 *Sparse Oblique Decision Tree* (SODT)

*Sparse Oblique Decision Tree* (SODT) adalah pohon keputusan yang melakukan pemisahan linear multivariat (*oblique split*) pada setiap *node* internal, berbeda dengan pohon *axis-aligned* yang hanya memakai satu fitur per pemisahan (Hada et al., 2024). Fungsi keputusan *node* internal ke-*i* dinyatakan pada Persamaan 2.28.

$$
{f}_{i}\left(x\right)={w}_{i}^{T}x+{b}_{i}
$$

(2.28)

dengan $x$ vektor fitur berdimensi $D$, ${w}_{i}$ vektor bobot, dan ${b}_{i}$ bias *node* ke-*i*.

Tanda ${f}_{i}\left(x\right)$ menentukan arah percabangan, yaitu ${d}_{i}=+1$ ke anak kiri untuk ${f}_{i}\left(x\right)\ge 0$ dan ${d}_{i}=-1$ ke anak kanan untuk ${f}_{i}\left(x\right)<0$. Rangkaian *node* dari *root* hingga *leaf* membentuk jalur keputusan (*decision path*), dan label pada *leaf* yang dicapai menjadi prediksi pohon (Hada et al., 2024; Kairgeldin & Carreira-Perpiñán, 2025). Ilustrasinya ditunjukkan pada Gambar 2.11.

[SISIPKAN GAMBAR: ilustrasi SODT dengan *node* internal ${w}_{i}^{T}x+{b}_{i}\ge 0$, *leaf* berlabel kelas, dan satu jalur keputusan yang disorot dari *root* ke *leaf*]

Gambar 2.11 Ilustrasi SODT dan Jalur Keputusan

Apabila $x$ berasal dari perataan peta fitur $C\times H\times W$ (Subbab 2.1.4), setiap bobot berpasangan dengan satu elemen peta fitur. Vektor ${w}_{i}$ dengan demikian dapat dikembalikan ke bentuk grid $C\times H\times W$, sehingga setiap bobot menyatakan kanal dan posisi tertentu yang dipakai *node* (Hada et al., 2024; Kairgeldin & Carreira-Perpiñán, 2025).

**1. Regularisasi L1 dan Sparsitas**

Sifat *sparse* diperoleh melalui regularisasi L1 pada fungsi tujuan pelatihan (Persamaan 2.29), yang membuat bobot fitur tidak relevan bernilai tepat nol (Hada et al., 2024).

$$
E\left(\Theta \right)=\sum_{n=1}^{N}{L\left({y}_{n},T\left({x}_{n};\Theta \right)\right)}+\lambda \sum_{i\in \mathcal{D}}{{\left\Vert {w}_{i}\right\Vert }_{1}}
$$

(2.29)

dengan $\Theta$ parameter pohon, $N$ jumlah sampel, ${x}_{n},{y}_{n}$ fitur dan label sampel ke-*n*, $T\left({x}_{n};\Theta \right)$ prediksi pohon, $L$ fungsi kerugian klasifikasi, $\mathcal{D}$ himpunan *node* internal, dan $\lambda$ pengontrol sparsitas. *Node* yang seluruh bobotnya nol hanya ditentukan oleh bias, sehingga selalu mengarahkan masukan ke sisi yang sama.

**2. *Tree Alternating Optimization* (TAO)**

Persamaan 2.29 tidak dapat dioptimasi dengan metode berbasis gradien karena keputusan pohon diskrit, dan tidak didukung oleh metode pembentukan pohon konvensional. *Tree Alternating Optimization* (TAO) memecahnya menjadi masalah klasifikasi biner yang diselesaikan terpisah untuk setiap *node* (Carreira-Perpiñán & Tavallali, 2018), sebagaimana dinyatakan pada Persamaan 2.30.

$$
{E}_{i}\left({w}_{i},{b}_{i}\right)=\sum_{n\in {\mathcal{R}}_{i}}{\bar{L}\left({\bar{y}}_{n},{g}_{i}\left({x}_{n};{w}_{i},{b}_{i}\right)\right)}+\lambda {\left\Vert {w}_{i}\right\Vert }_{1}
$$

(2.30)

dengan ${\mathcal{R}}_{i}$ sampel yang mencapai *node* ke-*i* (*reduced set*), ${\bar{y}}_{n}$ label semu (*pseudo-label*) arah kiri atau kanan, ${g}_{i}$ keputusan biner *node*, dan $\bar{L}$ kerugian 0/1. Langkah TAO pada satu *node* adalah sebagai berikut (Carreira-Perpiñán & Tavallali, 2018; Hada et al., 2024).

1. Setiap sampel dalam ${\mathcal{R}}_{i}$ diprediksi dua kali, yaitu bila diarahkan ke kiri dan ke kanan, dengan sub-pohon di bawahnya tetap.
2. Sampel yang hasilnya sama pada kedua arah dikeluarkan; sisanya (*care set*) memperoleh label semu berupa arah yang menghasilkan prediksi benar.
3. Kerugian 0/1 diganti fungsi pengganti (*surrogate*) berupa regresi logistik dengan regularisasi L1.
4. Parameter baru hanya diterima apabila tidak memperburuk fungsi tujuan *node* tersebut.

Setiap *leaf* diberi label kelas mayoritas dari sampel yang mencapainya. TAO memperbarui *node* dari yang terdalam menuju *root* dan mengulangnya hingga konvergen.

Karena penalti $\lambda$ sama untuk semua *node*, *node* dengan banyak data cenderung kurang *sparse*. Kairgeldin dan Carreira-Perpiñán (2025) mengatasinya dengan membobot penalti berdasarkan jumlah sampel *node* melalui parameter $\alpha$, sebagaimana dinyatakan pada Persamaan 2.31 dan 2.32.

$$
E\left(\Theta \right)=\sum_{n=1}^{N}{L\left({y}_{n},T\left({x}_{n};\Theta \right)\right)}+\lambda \sum_{i\in \mathcal{D}}{{h}_{\alpha}\left(\left|{\mathcal{R}}_{i}\right|\right){\left\Vert {w}_{i}\right\Vert }_{1}}
$$

(2.31)

$$
{h}_{\alpha}\left(t\right)=\begin{cases}1, & t=0\\ {t}^{\alpha}, & t>0\end{cases}
$$

(2.32)

dengan $\left|{\mathcal{R}}_{i}\right|$ jumlah sampel yang mencapai *node* ke-*i*. Nilai $\alpha>0$ membuat eliminasi fitur lebih agresif pada *node* yang menangani banyak data.

**3. Pembobotan Kelas (*Class Weighting*)**

Pada *model mimicking* (Subbab 2.1.7), label pelatihan SODT adalah prediksi *teacher*. Apabila sebagian kelas jauh lebih jarang, pohon cenderung mengabaikannya. *Cost-sensitive learning* mengatasinya dengan memberi biaya kesalahan yang berbeda antarkelas (Elkan, 2001; He & Garcia, 2009), sebagaimana dinyatakan pada Persamaan 2.33.

$$
{E}_{\omega}\left(\Theta \right)=\sum_{n=1}^{N}{{\omega}_{{y}_{n}}L\left({y}_{n},T\left({x}_{n};\Theta \right)\right)}
$$

(2.33)

dengan ${\omega}_{{y}_{n}}\ge 0$ biaya kesalahan kelas ${y}_{n}$.
**4. Penjelasan melalui Bobot *Node***

Karena setiap *node* linear dan *sparse*, bobot ${w}_{i}$ langsung menunjukkan fitur yang dipakai *node*: bobot nol berarti tidak dipakai, tandanya menunjukkan arah dorongan, dan besarnya menunjukkan kekuatan pengaruh. Hada et al. (2024) memvisualisasikan bobot ini untuk menelusuri fitur yang memisahkan antarkelas.

Kairgeldin dan Carreira-Perpiñán (2025) memperluasnya pada model hibrida CNN dan SODT. Karena setiap fitur berasal dari sel peta fitur CNN yang memiliki *receptive field* (Subbab 2.1.4), mereka menyusun peta kepadatan *receptive field* (*RF density map*) untuk setiap *node*, sebagaimana dinyatakan pada Persamaan 2.34.

$$
{D}_{i}\left(u\right)=\sum_{j=1}^{D}{\left|{w}_{ij}\right|\cdot \mathbb{1}\left[u\in {RF}_{j}\right]}
$$

(2.34)

dengan ${D}_{i}\left(u\right)$ kepadatan pada posisi citra $u$ untuk *node* ke-*i*, ${w}_{ij}$ bobot fitur ke-*j*, ${RF}_{j}$ *receptive field* fitur ke-*j*, dan $\mathbb{1}\left[\cdot \right]$ fungsi indikator. Kepadatan nol berarti daerah tersebut tidak dipakai *node* (Kairgeldin & Carreira-Perpiñán, 2025). Peta ini disusun per *node* dan hanya bergantung pada bobot, sehingga bersifat statis.

### 2.1.9 Metrik Evaluasi

Evaluasi dibagi menjadi evaluasi kinerja deteksi objek dan evaluasi kualitas penjelasan (*explainability*).

**1. Metrik Evaluasi Deteksi Objek**

*Intersection over Union* (IoU) mengukur rasio luas irisan terhadap luas gabungan *bounding box* prediksi dan *ground truth* (Persamaan 2.35). Berdasarkan ambang IoU, prediksi dikelompokkan menjadi *True Positive* (TP), *False Positive* (FP), dan *False Negative* (FN), yang menjadi dasar *Precision* dan *Recall* (Persamaan 2.36 dan 2.37).

$$
IoU=\frac{\text{Area Prediksi}\cap \text{Area Ground Truth}}{\text{Area Prediksi}\cup \text{Area Ground Truth}}
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

*Precision* mengukur ketepatan prediksi positif, sedangkan *Recall* mengukur kelengkapan deteksi. *F1-score* adalah rata-rata harmonik *Precision* dan *Recall* (Persamaan 2.38) (Sokolova & Lapalme, 2009).

$$
F1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall}
$$

(2.38)

*Average Precision* (AP) adalah luas di bawah kurva *Precision-Recall* (Persamaan 2.39), dan *mean Average Precision* (mAP) adalah rata-rata AP seluruh kelas (Persamaan 2.40) (Everingham et al., 2010).

$$
AP=\int_{0}^{1}{Precision\left(Recall\right)\,d\left(Recall\right)}
$$

(2.39)

$$
mAP=\frac{1}{K}\sum_{i=1}^{K}{{AP}_{i}}
$$

(2.40)

dengan $K$ jumlah kelas dan ${AP}_{i}$ AP kelas ke-*i*. mAP@0,5 dihitung pada ambang IoU 0,5, sedangkan mAP@0,5:0,95 merupakan rata-rata mAP pada ambang IoU 0,5 hingga 0,95 dengan kenaikan 0,05 (Everingham et al., 2010; Lin et al., 2014).

**2. Metrik Evaluasi *Explainability***

Evaluasi XAI dapat melibatkan pengguna (*application-grounded* dan *human-grounded*) atau tanpa pengguna melalui ukuran kuantitatif (*functionally-grounded*) (Nauta et al., 2023). Penelitian ini berfokus pada evaluasi *functionally-grounded*, khususnya *faithfulness* dan *localization*.

a. *Faithfulness*

*Faithfulness* umumnya diuji melalui perturbasi, yaitu menghapus (*masking*) atau mempertahankan bagian masukan yang disorot penjelasan lalu mengamati perubahan keluaran model (Petsiuk et al., 2018).

*Necessity* menguji apakah fitur yang disorot memang diperlukan: fitur dihapus, lalu diperiksa apakah prediksi berubah (Persamaan 2.41). *Sufficiency* menguji apakah fitur yang disorot sudah cukup: hanya fitur tersebut yang dipertahankan, lalu diperiksa apakah prediksi tetap (Persamaan 2.42).

$$
\text{Necessity Flip Rate}=\frac{1}{N}\sum_{n=1}^{N}{\mathbb{1}\left[\hat{y}\left({x}_{n}^{-S}\right)\ne \hat{y}\left({x}_{n}\right)\right]}
$$

(2.41)

$$
\text{Sufficiency Preservation}=\frac{1}{N}\sum_{n=1}^{N}{\mathbb{1}\left[\hat{y}\left({x}_{n}^{S}\right)=\hat{y}\left({x}_{n}\right)\right]}
$$

(2.42)

dengan $S$ himpunan fitur atau daerah yang disorot paling penting oleh penjelasan, ${x}_{n}^{-S}$ masukan tanpa $S$ (dinolkan), ${x}_{n}^{S}$ masukan yang hanya mempertahankan $S$, $\hat{y}$ keluaran model, dan $\mathbb{1}\left[\cdot \right]$ fungsi indikator. Nilai yang tinggi pada kedua metrik menandakan penjelasan yang *faithful*.

*Faithfulness* juga dapat diukur secara bertahap dengan *Deletion* AUC dan *Insertion* AUC (Petsiuk et al., 2018), yaitu luas di bawah kurva skor seiring fitur dihapus atau ditambahkan dalam $K$ tahap (Persamaan 2.43).

$$
AUC=\sum_{k=1}^{K-1}{\frac{{s}_{k}+{s}_{k+1}}{2}\cdot \Delta {x}_{k}}
$$

(2.43)

dengan ${s}_{k}$ skor ternormalisasi pada tahap ke-*k* dan $\Delta {x}_{k}$ proporsi area yang dimodifikasi antartahap. Pada *Deletion*, fitur terpenting dihapus lebih dulu, sehingga AUC rendah menandakan *necessity* yang baik. Pada *Insertion*, fitur terpenting ditambahkan lebih dulu, sehingga AUC tinggi menandakan *sufficiency* yang baik.

b. *Localization*

*Localization* mengukur kesesuaian area yang disorot *heatmap* dengan lokasi objek. *Pointing Game* menghitung proporsi kasus ketika titik tertinggi *heatmap* jatuh di dalam *bounding box ground truth* (Persamaan 2.44) (Zhang et al., 2018), sedangkan IoU *Heatmap* mengukur IoU antara *heatmap* terbinerisasi dan *bounding box ground truth* (Persamaan 2.45).

$$
{Acc}_{PG}=\frac{Hits}{Hits+Misses}
$$

(2.44)

$$
{IoU}_{Heatmap}=\frac{\left|{H}_{bin}\cap {B}_{gt}\right|}{\left|{H}_{bin}\cup {B}_{gt}\right|}
$$

(2.45)

dengan $Hits$ dan $Misses$ jumlah kasus yang titik maksimumnya berada di dalam dan di luar *ground truth*, ${H}_{bin}$ *heatmap* yang hanya mempertahankan sejumlah posisi bernilai tertinggi, dan ${B}_{gt}$ daerah *ground truth*.

## 2.2 Penelitian Terkait

Ringkasan dan perbandingan penelitian terkait disajikan pada Tabel 2.2.

Tabel 2.2 Ringkasan dan Perbandingan Penelitian Terkait

| No. | Judul / Penulis | Masalah | Tujuan | Metode | Hasil | Keterkaitan |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | *Improving PCB defect detection using selective feature attention and pixel shuffle pyramid* (Fung et al., 2024) | Deteksi cacat mikroskopis pada sirkuit PCB memiliki tingkat *false negative* yang tinggi pada model deteksi standar. | Meningkatkan kemampuan model melokalisasi target berukuran kecil pada citra PCB. | *Faster* R-CNN dengan *Feature Pyramid Network* (FPN), *Pixel Shuffle Pyramid* (PSPyramid), *Selective Feature Attention*, dan *Soft*-NMS. | Terjadi peningkatan *Mean Average Precision* (mAP) yang signifikan pada pengujian dataset DeepPCB. | Menjadi referensi utama arsitektur dasar (*baseline*) komponen *neural* (ekstraktor fitur dan lokalisasi) dalam penelitian ini. |
| 2 | *Faster-LTN: a neuro-symbolic, end-to-end object detection architecture* (Manigrasso et al., 2021) | Model *deep learning* konvensional tidak mampu mengintegrasikan pengetahuan relasional dan penalaran logis ke dalam proses deteksi objek, sehingga kurang transparan dalam pengambilan keputusan. | Mengintegrasikan penalaran logis dengan jaringan saraf konvolusional ke dalam arsitektur deteksi objek *end-to-end* untuk meningkatkan transparansi. | *Faster* R-CNN dengan penggantian kepala klasifikasi menjadi *Logic Tensor Networks* (LTN). | Arsitektur *end-to-end* berhasil dilatih dan mencapai performa kompetitif pada dataset PASCAL VOC. | Memberikan landasan konseptual integrasi pendekatan *neuro-symbolic* ke dalam arsitektur deteksi objek dua tahap (*Faster* R-CNN). |
| 3 | *Sparse oblique decision trees: a tool to understand and manipulate neural net features* (Hada et al., 2024) | Lapisan MLP pada jaringan *deep learning* tidak dapat dijelaskan secara komputasional (*black-box*). | Menciptakan pengganti lapisan MLP yang dapat diinterpretasikan tanpa menurunkan akurasi. | *Sparse Oblique Decision Tree* (SODT) dengan regularisasi L1 yang dilatih menggunakan TAO. | Menghasilkan struktur pohon yang ramping dengan performa setara MLP; bobot tiap *node* dapat divisualisasikan untuk menelusuri fitur yang memisahkan antarkelas. | Menjadi landasan teoretis komponen simbolik yang menggantikan *classifier* pada *box head* *Faster* R-CNN, serta dasar pembacaan bobot *node* sebagai penjelasan. |
| 4 | *Neurosymbolic models based on hybrids of convolutional neural networks and decision trees* (Kairgeldin & Carreira-Perpiñán, 2025) | Interpretabilitas model hibrida CNN dan pohon keputusan masih terbatas, dan sebaran sparsitas antar-*node* pada TAO tidak dapat dikendalikan. | Membangun model *neuro-symbolic* hibrida CNN dan SODT yang keputusannya dapat dijelaskan. | Komposisi lapisan CNN dengan SODT yang dilatih menggunakan TAO termodifikasi (parameter $\alpha$), serta visualisasi *RF density map* per *node*. | Sebagian kecil *neuron* terbukti bertanggung jawab penuh atas klasifikasi kelas tertentu, dan *receptive field*-nya terpusat pada area citra yang paling membedakan antarkelas. | Menjadi dasar algoritma TAO dengan parameter $\alpha$ untuk pelatihan SODT, serta konsep peta berbasis bobot per *node* yang menjadi titik awal penjelasan *heatmap* pada penelitian ini. |
| 5 | *Explainable Predictive Quality Inspection using Deep Learning in Electronics Manufacturing* (Saadallah et al., 2022) | Model *deep learning* untuk prediksi kualitas bersifat *black-box* sehingga menyulitkan teknisi memahami fitur mana yang paling berpengaruh terhadap keputusan prediksi. | Menyediakan penjelasan visual atas prediksi kualitas PCB menggunakan *heatmap* untuk membantu teknisi mengidentifikasi fitur global (kuantitas fisik SPI) dan lokal (pin) yang paling menentukan. | 1D-CNN untuk prediksi kualitas biner (OK/NOK) dan Grad-CAM untuk menghasilkan *heatmap* penjelasan. | Grad-CAM berhasil menyoroti fitur SPI (DX, DY, DVolume) dan pin spesifik yang paling diskriminatif untuk kelas "NOK", membantu teknisi melacak penyebab deviasi kualitas. | Menunjukkan penerapan Grad-CAM sebagai metode *post-hoc* untuk inspeksi kualitas PCB. |
| 6 | *Explainable AI Methods for Identification of Glue Volume Deficiencies in Printed Circuit Boards* (Tziolas et al., 2025) | Inspeksi volume lem pada PCB sulit dilakukan secara manual dan model *deep learning* yang digunakan tidak memberikan penjelasan atas deteksi defisiensi. | Mengidentifikasi defisiensi volume lem pada PCB menggunakan model *deep learning* dan menyediakan penjelasan visual atas prediksi model. | CNN (ResNet-50 dan *Vision Transformer*) untuk klasifikasi defisiensi lem, dengan Grad-CAM dan Deep SHAP untuk menghasilkan *heatmap* penjelasan. | CNN mencapai akurasi tinggi dalam mendeteksi defisiensi lem, dan Grad-CAM/Deep SHAP berhasil menyoroti area dengan volume lem tidak memadai yang menjadi dasar keputusan model. | Memperkuat justifikasi penggunaan Grad-CAM sebagai metode *post-hoc* yang telah teruji dalam inspeksi visual PCB, serta menunjukkan keterbatasan *post-hoc* yang mendorong kebutuhan pendekatan *faithful*. |
| 7 | *Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging* (Arun et al., 2021) | Peta *saliency* banyak dipakai untuk menjelaskan dan melokalisasi keputusan CNN pada domain berisiko tinggi, tetapi keandalannya belum teruji secara sistematis. | Mengevaluasi keandalan (*trustworthiness*) peta *saliency* untuk lokalisasi kelainan pada citra medis. | Delapan metode *saliency*, termasuk Grad-CAM, diuji pada dua *dataset* radiologi berdasarkan utilitas lokalisasi, sensitivitas terhadap pengacakan bobot model, *repeatability*, dan *reproducibility*, lalu dibandingkan dengan jaringan lokalisasi (U-Net dan RetinaNet). | Seluruh metode gagal pada minimal satu kriteria dan kalah dari jaringan lokalisasi. Grad-CAM lolos uji pengacakan bobot, tetapi AUPRC lokalisasinya (0,41) tetap di bawah RetinaNet (0,596). | Menunjukkan bahwa penjelasan *post-hoc*, termasuk Grad-CAM, belum dapat diandalkan pada domain berisiko tinggi, serta menjadi dasar kontrol pengacakan bobot yang juga digunakan dalam penelitian ini. |


# BAB III

# METODOLOGI PENELITIAN

## 3.1 Diagram Alir Penelitian

Penelitian ini membangun, mengintegrasikan, dan mengevaluasi sistem deteksi cacat PCB berbasis arsitektur *neuro-symbolic*. Alur penelitian ditunjukkan pada Gambar 3.1.

![Gambar 3.1](asset:sha256:925eb4de0610231e1cd55ded23ce9a23651afa3d49641fe9b80d9a128bbae0ee)

Gambar 3.1 Diagram Alir Penelitian

Penelitian dimulai dengan persiapan dan pra-pemrosesan *dataset* DeepPCB. *Faster* R-CNN kemudian dilatih dan dibekukan sebagai model *teacher*, lalu dipakai untuk mengekstrak fitur RoI beserta label prediksinya sebagai *dataset* simbolik. *Dataset* ini melatih SODT yang meniru keputusan klasifikasi *teacher*. SODT kemudian menggantikan kepala klasifikasi *Faster* R-CNN, dilengkapi skor deteksi berbasis *routing margin* dan *heatmap* per *node*. Terakhir, model hibrida dievaluasi terhadap *Faster* R-CNN dari sisi deteksi dan terhadap Grad-CAM dari sisi penjelasan.

## 3.2 Persiapan *Dataset*

*Dataset* DeepPCB (Tang et al., 2019) dibagi menjadi data latih dan data uji berdasarkan berkas indeks bawaannya (Tabel 3.1), sehingga tidak ada kebocoran data antarfase. Penelitian ini bersifat non-referensial, yaitu model mendeteksi cacat tanpa membandingkan citra uji dengan citra templat, sesuai kondisi inspeksi nyata ketika citra referensi sering tidak tersedia.

Tabel 3.1 Pembagian *Dataset* Berdasarkan Kategori

| Kategori | Berkas Indeks | Jumlah Citra | Persentase (%) |
| --- | --- | --- | --- |
| Data Pelatihan (*Training*) | *trainval.txt* | 1.000 | 66,67 |
| Data Pengujian (*Testing*) | *test.txt* | 500 | 33,33 |

**Input:** Data mentah DeepPCB (1.500 pasang citra), berkas indeks partisi, dan berkas anotasi.

**Output:** Pasangan citra, koordinat *bounding box*, dan label kelas cacat.

**Langkah-langkah:**

1. Sistem membaca *trainval.txt* dan *test.txt* untuk membagi 1.000 citra latih dan 500 citra uji.
2. Sistem hanya memuat citra target (*\_test.jpg*) dan mengabaikan citra templat (*\_temp.jpg*).
3. Koordinat cacat pada berkas anotasi diubah menjadi *bounding box*.
4. ID kelas 1–6 pada berkas anotasi dipakai langsung sebagai indeks kelas, yaitu *open*, *short*, *mousebite*, *spur*, *spurious copper*, dan *pinhole*.

Contoh sampel beserta anotasinya ditunjukkan pada Gambar 3.2.

![Gambar 3.2](asset:sha256:df0f527618e8a09b155ae0bcbf4011b8f6bb4c89343f2136f8227cf57c2fc8a9)

Gambar 3.2 Contoh Sampel *Dataset* PCB beserta Anotasinya

## 3.3 Pra-pemrosesan *Dataset*

Citra diproses melalui dua *pipeline*. *Pipeline* di luar model mengubah citra menjadi *tensor* dan menerapkan augmentasi saat pelatihan. *Pipeline* di dalam model menormalisasi citra dan mengatur resolusi: acak multi-resolusi saat latih, dan tetap saat inferensi maupun ekstraksi fitur agar fitur bagi komponen simbolik stabil. Konfigurasinya dirangkum pada Tabel 3.2 dan Tabel 3.3.

Tabel 3.2 Parameter Normalisasi dan Standardisasi Masukan

| Parameter | Nilai Konfigurasi | Tujuan |
| --- | --- | --- |
| Rentang Intensitas | [0,0; 1,0] | Penyeragaman *dynamic range* piksel |
| Rata-rata (*Mean*) | [0,485; 0,456; 0,406] | Penyelarasan distribusi ImageNet |
| Deviasi Standar (*Std*) | [0,229; 0,224; 0,225] | Penyelarasan distribusi ImageNet |

Tabel 3.3 Konfigurasi Augmentasi dan Resolusi Masukan

| Jenis | Parameter | Deskripsi |
| --- | --- | --- |
| *Random Horizontal Flip* (latih) | Probabilitas: 0,5 | Variasi arah |
| *Multi-resolution Scaling* (latih) | Sisi terpendek: {480, 560, 640, 720, 800, 880}; sisi terpanjang maks. 880 | Ketahanan terhadap skala |
| Resolusi tetap (uji dan ekstraksi fitur) | Sisi terpendek: 640; sisi terpanjang maks. 880 | Kestabilan fitur bagi komponen simbolik |
| Sinkronisasi koordinat | Aktif | Lokasi *box* menyesuaikan perubahan citra |

**Input:** Citra mentah dan koordinat *bounding box* hasil partisi.

**Output:** *Tensor* citra ternormalisasi dan koordinat yang tersinkron.

**Langkah-langkah:**

1. Citra diubah menjadi *tensor* dengan intensitas [0,0; 1,0].
2. Pada fase latih, citra dibalik horizontal secara acak dan koordinat *bounding box* disesuaikan.
3. Di dalam model, citra dinormalisasi per kanal (Tabel 3.2).
4. Citra diskalakan sesuai fase (Tabel 3.3), lalu disamakan ukurannya dalam satu *batch* melalui *padding*.

## 3.4 Model *Neural* (*Faster* R-CNN)

*Faster* R-CNN berperan sebagai ekstraktor fitur, pengusul area, dan model *teacher*. Arsitekturnya mengikuti SF-PSPyramid dari Fung et al. (2024) (Subbab 2.1.5), dengan satu perbedaan: jumlah kanal *neck* 64, bukan 256, agar dimensi masukan SODT tidak terlalu besar ($64\times 7\times 7=3.136$). Konfigurasinya dirangkum pada Tabel 3.4.

Tabel 3.4 Parameter Utama *Faster* R-CNN

| Modul | Parameter | Nilai / Konfigurasi |
| --- | --- | --- |
| *Backbone* | Arsitektur dan inisialisasi | ResNet-50, *pre-trained* ImageNet |
| | *Batch Normalization* | Dibekukan (*frozen*) |
| *Neck* (SF-PSPyramid) | Tingkat peta fitur | P2′, P3′, P4, P5, P6 (*stride* 4, 8, 16, 32, 64) |
| | Jumlah kanal (*C*) | 64 |
| | *Bottleneck SF Attention* | 32 |
| RPN | Ukuran *anchor* | 16, 32, 64, 128, 256 piksel (satu ukuran per tingkat) |
| | Rasio aspek | 0,5; 1,0; 2,0 |
| | Ambang IoU *foreground* / *background* | 0,7 / 0,3 |
| | Jumlah proposal setelah NMS (latih / uji) | 2.000 / 1.000 |
| | Ambang NMS proposal | 0,7 |
| *RoI Align* | Ukuran keluaran | 7×7 *bin*, 2×2 titik sampel per *bin* |
| *Box Head* | Lapisan MLP | 2 lapisan terhubung penuh (1.024 *neuron*) |
| *Soft*-NMS | Metode penurunan skor | Linear (Persamaan 2.20) |
| | Ambang IoU | 0,5 |
| | Ambang skor minimum | 0,001 |
| | Deteksi maksimum per citra | 100 |

**Input:** *Tensor* citra hasil pra-pemrosesan.

**Output:** Model *Faster* R-CNN yang terkonfigurasi.

**Langkah-langkah:**

1. *Backbone* ResNet-50 mengekstrak peta fitur C2–C5.
2. *Neck* membentuk piramida P2′–P6 melalui *pixel shuffle* (Persamaan 2.8) dan *SF Attention* (Persamaan 2.9).
3. RPN menghasilkan proposal dari *anchor* (Persamaan 2.10 sampai 2.13) dan menyaringnya dengan NMS.
4. *RoI Align* memilih tingkat piramida (Persamaan 2.19) dan menghasilkan *tensor* $64\times 7\times 7$ untuk setiap proposal (Persamaan 2.17).
5. *Box Head* menghasilkan skor kelas (Persamaan 2.4) dan koordinat *bounding box* akhir.
6. *Soft*-NMS menyaring deteksi yang tumpang-tindih (Persamaan 2.20).

## 3.5 Pelatihan dan Evaluasi Model *Neural*

*Faster* R-CNN dilatih selama 15 *epoch* (Fung et al. (2024) memakai 12) dengan *Automatic Mixed Precision* (AMP) dan *gradient accumulation* untuk mengatasi keterbatasan memori GPU. Kerugian regresi memakai L1 murni (Persamaan 2.15) mengikuti Fung et al. (2024), alih-alih *smooth* L1 (Girshick, 2015), karena memberi penalti lebih tegas pada galat kecil. Laju pembelajaran dinaikkan secara linear pada awal pelatihan (*warmup*), lalu diturunkan bertahap. *Hyperparameter* dirangkum pada Tabel 3.5.

Tabel 3.5 *Hyperparameter* Pelatihan *Faster* R-CNN

| Kelompok | Parameter | Nilai / Konfigurasi |
| --- | --- | --- |
| Siklus pelatihan | Jumlah *epoch* | 15 |
| | Ukuran *batch* | 1 |
| | *Gradient accumulation* | 4 langkah (*batch* efektif 4) |
| | Presisi komputasi | AMP |
| | *Seed* | 42 |
| Pengoptimal | *Optimizer* | SGD |
| | Laju pembelajaran | 0,02 |
| | *Momentum* / *weight decay* | 0,9 / 0,0001 |
| Penjadwal | Tipe | *Milestone* / *step decay* |
| | *Milestone* | *Epoch* ke-8 dan ke-11 |
| | Faktor penurunan (*gamma*) | 0,1 |
| Pemanasan | Iterasi *warmup* | 500 (*batch*) |
| | Rasio *warmup* awal | 0,001 |

**Input:** Citra latih beserta anotasinya.

**Output:** Model *Faster* R-CNN terlatih dan hasil evaluasinya pada data uji.

**Langkah-langkah:**

1. Sistem menjalankan propagasi maju dengan AMP pada setiap *batch*.
2. *Multi-task loss* (Persamaan 2.14) dengan kerugian L1 (Persamaan 2.15) dihitung, lalu gradien empat iterasi diakumulasikan.
3. SGD memperbarui bobot (Persamaan 2.3) dengan *warmup* 500 *batch* dan penurunan ×0,1 pada *epoch* ke-8 dan ke-11.
4. Model dievaluasi pada data uji dengan mAP@0,5 dan mAP@0,5:0,95, serta *Precision*, *Recall*, dan F1 pada IoU 0,5 dan skor 0,5 (Persamaan 2.35 sampai 2.40), termasuk per kelas dan *confusion matrix*.
5. Bobot model dibekukan dan disimpan sebagai model *teacher*.

## 3.6 Ekstraksi Fitur RoI dan Label *Teacher*

Tahap ini membentuk *dataset* simbolik sesuai skema *model mimicking* (Subbab 2.1.7). Fitur diambil tepat setelah *RoI Align*, sebelum masuk MLP pada *box head*, sehingga bentuk grid $7\times 7$ tetap terjaga. Yang diekstrak adalah seluruh proposal RPN pada mode inferensi, bukan hanya deteksi akhir, agar SODT dilatih pada populasi proposal yang sama dengan saat inferensi.

Label setiap proposal adalah prediksi *teacher*, termasuk *background*, bukan *ground truth*. *Ground truth* hanya disimpan sebagai data pendamping untuk metrik lokalisasi. Ekstraksi dilakukan terpisah untuk data latih dan data uji. Mekanismenya diilustrasikan pada Gambar 3.3.

![Gambar 3.3](asset:sha256:336ec358889ca6a84ef8ca4fd0b1e985bcb11dd37e812d27a740ecda25205386)

Gambar 3.3 Pengambilan Fitur *RoI Align* dan Label dari *Faster* R-CNN

**Input:** Model *teacher* (*frozen*) serta citra latih dan uji beserta anotasinya.

**Output:** *Dataset* simbolik latih dan uji berisi fitur $64\times 7\times 7$, koordinat proposal, label *teacher*, dan data pendamping *ground truth*.

**Langkah-langkah:**

1. Setiap citra diproses dengan resolusi tetap (Tabel 3.3), lalu RPN menghasilkan hingga 1.000 proposal.
2. *RoI Align* mengubah setiap proposal menjadi *tensor* $64\times 7\times 7$ (Persamaan 2.17 dan 2.19).
3. *Tensor* diteruskan ke *box head* *teacher*, dan kelas dengan *softmax* tertinggi (Persamaan 2.4) diambil sebagai label.
4. Setiap proposal dicocokkan dengan *ground truth* ber-IoU tertinggi sebagai data pendamping.
5. Fitur, label, dan data pendamping disimpan sebagai *dataset* simbolik.

## 3.7 Model Simbolik (SODT)

SODT menggantikan *classifier* pada *box head*, sehingga setiap keputusan kelas berasal dari rangkaian keputusan linear yang dapat ditelusuri (Subbab 2.1.8). Masukannya adalah *tensor* $64\times 7\times 7$ yang diratakan menjadi vektor berdimensi 3.136, dengan urutan yang dicatat agar setiap bobot dapat dikembalikan ke kanal dan posisi grid asalnya.

**Input:** Kedalaman pohon dan dimensi *tensor* RoI.

**Output:** SODT terinisialisasi beserta fungsi tujuannya.

**Langkah-langkah:**

1. Sistem membentuk pohon biner lengkap berkedalaman 6 dengan fungsi keputusan linear pada setiap *node* (Persamaan 2.28) dan satu label pada setiap *leaf*.
2. *Tensor* $64\times 7\times 7$ diratakan dengan urutan kanal, baris, lalu kolom.
3. Bobot dan bias diinisialisasi dari distribusi normal baku, dan label *leaf* secara acak (Hada et al., 2024; Kairgeldin & Carreira-Perpiñán, 2025).
4. Fungsi tujuan disusun dari kerugian berbobot kelas (Persamaan 2.33) dan penalti L1 berbobot ukuran *reduced set* (Persamaan 2.31 dan 2.32).

## 3.8 Pelatihan dan Evaluasi Model Simbolik

SODT dilatih dengan TAO (Subbab 2.1.8) pada *dataset* simbolik latih. Karena sebagian besar proposal berlabel *background*, dilakukan *negative sampling*: semua proposal berlabel cacat dipertahankan, dan proposal *background* diambil acak sebanyak dua kali jumlahnya.

Pembobotan kelas diterapkan di dua tempat. Pada masalah tereduksi setiap *node* (Persamaan 2.30), setiap sampel diberi bobot pada Persamaan 3.1. Pada *leaf*, label ditentukan dengan mayoritas berbobot pada Persamaan 3.2.

$$
{u}_{n}=\left|{\ell}_{L}\left(n\right)-{\ell}_{R}\left(n\right)\right|\cdot {\omega}_{{y}_{n}}
$$

(3.1)

$$
{\hat{y}}_{\ell}=\arg\max_{c}{{\omega}_{c}\cdot {N}_{\ell,c}}
$$

(3.2)

dengan ${u}_{n}$ bobot sampel ke-*n*, ${\ell}_{L}\left(n\right),{\ell}_{R}\left(n\right)$ kerugian 0/1 sampel ke-*n* bila diarahkan ke kiri dan ke kanan, ${\omega}_{c}$ bobot kelas $c$ (Persamaan 2.33), ${\hat{y}}_{\ell}$ label *leaf* ke-$\ell$, dan ${N}_{\ell,c}$ jumlah sampel berlabel $c$ pada *leaf* tersebut. Faktor $\left|{\ell}_{L}-{\ell}_{R}\right|$ bernilai 1 hanya untuk *care set*, sehingga kesalahan pengarahan pada kelas berbobot besar menjadi lebih mahal. Bobot *background* dibiarkan 1,0 agar *false positive* tidak meningkat.

Setelah TAO, pohon dipangkas tanpa mengubah keputusannya. *Node* yang tidak dilalui sampel (*dead branch*) dan *node* yang seluruh *leaf* di bawahnya berlabel sama (*pure subtree*) dinolkan bersama sub-pohonnya, dan *leaf* yang tidak dicapai sampel diberi label *background*. *Hyperparameter* pelatihan dirangkum pada Tabel 3.6.

Tabel 3.6 *Hyperparameter* Pelatihan SODT

| Komponen | Nilai | Deskripsi |
| --- | --- | --- |
| Kedalaman pohon | 6 | 63 *node* internal, 64 *leaf* |
| Iterasi TAO | 15 | Jumlah siklus pembaruan seluruh *node* |
| Lambda ($\lambda$) | 10 | Kekuatan penalti L1 (Persamaan 2.31) |
| Alpha ($\alpha$) | 0,15 | Eksponen pembobot penalti (Persamaan 2.32) |
| Rasio negatif | 2 | Sampel *background* terhadap sampel cacat |
| Bobot kelas ($\omega$) | *short* 2,0; *spur* 1,5; *open* 1,5; *pinhole* 1,25; lainnya 1,0 | Biaya lebih besar untuk kelas dengan *recall* terlemah |
| Iterasi maksimum regresi logistik | 200 | Batas iterasi *solver* per masalah tereduksi |
| Toleransi *solver* | 0,0001 | Toleransi konvergensi regresi logistik |
| Toleransi konvergensi TAO | 0,000001 | Berhenti lebih awal bila penurunan relatif fungsi tujuan lebih kecil |
| Ambang nol bobot | 0,00001 | Bobot di bawah nilai ini dianggap nol |
| *Seed* | 112 | Inisialisasi dan *negative sampling* |

Fidelitas SODT diukur terhadap label *teacher* pada *dataset* simbolik uji dengan akurasi (*mimic accuracy*) dan *macro*-F1, serta kesesuaian per kelas, disertai jumlah *node* aktif dan bobot bukan nol.

**Input:** *Dataset* simbolik latih dan uji, serta SODT terinisialisasi.

**Output:** SODT terlatih dan hasil evaluasi fidelitasnya.

**Langkah-langkah:**

1. Sistem menerapkan *negative sampling* dengan rasio 2 pada *dataset* simbolik latih.
2. Pada setiap iterasi TAO, *node* diperbarui dari yang terdalam menuju *root* menggunakan bobot sampel Persamaan 3.1, lalu label *leaf* diperbarui dengan Persamaan 3.2.
3. Iterasi berhenti setelah 15 iterasi atau ketika penurunan relatif fungsi tujuan di bawah toleransi.
4. Pohon dipangkas dengan menolkan *dead branch* dan *pure subtree*.
5. Fidelitas diukur pada *dataset* simbolik uji.

## 3.9 Integrasi *Faster* R-CNN dan SODT

Tahap ini menyatukan *Faster* R-CNN dan SODT menjadi satu alur inferensi, ditambah dua komponen: skor deteksi berbasis *routing margin* dan *heatmap* per *node*.

### 3.9.1 Inferensi Hibrida

SODT hanya menggantikan kepala klasifikasi. Koordinat *bounding box* akhir tetap dihitung oleh kepala regresi *Faster* R-CNN, sehingga sifat *faithful* berlaku pada keputusan kelas, bukan pada penyesuaian lokasi kotak. Alurnya ditunjukkan pada Gambar 3.4.

![Gambar 3.4](asset:sha256:b6e08eb555049c63c5fbf1dd2ad76420f400a1f1bf5bc1df31542ab10cfecd23)

Gambar 3.4 Diagram Integrasi *Neuro-Symbolic*

**Input:** Citra uji, *Faster* R-CNN (*frozen*), dan SODT terlatih.

**Output:** *Bounding box*, label kelas, skor deteksi, dan jalur keputusan setiap deteksi.

**Langkah-langkah:**

1. *Faster* R-CNN menghasilkan proposal dan *tensor* $64\times 7\times 7$ untuk setiap proposal.
2. SODT menentukan label dari *leaf* yang dicapai, dan skor dihitung dengan Persamaan 3.3.
3. Kepala regresi *Faster* R-CNN menghitung koordinat *bounding box* akhir.
4. Deteksi berlabel *background* atau berskor di bawah 0,001 dibuang, lalu *Soft*-NMS diterapkan (Persamaan 2.20).
5. Jalur keputusan, tingkat piramida sumber, dan peta fitur *neck* setiap deteksi disimpan untuk pembentukan *heatmap*.

### 3.9.2 Skor Deteksi Berbasis *Routing Margin*

Karena setiap *leaf* hanya menyimpan satu label, semua deteksi yang mencapai *leaf* berkelas sama akan memiliki skor identik (Subbab 2.1.7). Akibatnya, AP (Persamaan 2.39) dan *Soft*-NMS kehilangan urutan skor. Oleh karena itu, skor dibentuk dari *margin* (Persamaan 2.26) dan *sigmoid* (Persamaan 2.27) setiap *node* pada jalur keputusan, sebagaimana dinyatakan pada Persamaan 3.3.

$$
s\left(x\right)=\prod_{i\in P\left(x\right),\, {w}_{i}\ne 0}{\sigma \left(\left|{f}_{i}\left(x\right)\right|\right)}
$$

(3.3)

dengan $s\left(x\right)$ skor untuk RoI $x$ pada kelas *leaf* yang dicapai (kelas lain bernilai 0), $P\left(x\right)$ *node* pada jalur keputusan, dan ${f}_{i}$ fungsi keputusan *node* ke-*i* (Persamaan 2.28). *Node* yang telah dinolkan dilewati. Skor ini hanya berasal dari parameter pohon dan tidak mengubah jalur maupun label.

Berbeda dengan Platt (1999), parameter *sigmoid* bernilai tetap, sehingga $s\left(x\right)$ merupakan skor keyakinan, bukan probabilitas kelas, dan setiap faktornya berada pada rentang [0,5; 1).

### 3.9.3 *Heatmap* Per *Node*

Setiap *node* pada jalur keputusan memperoleh satu *heatmap* yang menunjukkan daerah yang ditimbang *node* tersebut untuk RoI yang dijelaskan. Berbeda dengan peta Kairgeldin dan Carreira-Perpiñán (2025) (Persamaan 2.34) yang statis, *heatmap* ini dinamis karena memakai nilai fitur RoI, dan dihitung pada peta fitur *neck*, bukan grid $7\times 7$.

Fungsi keputusan *node* linear terhadap fitur RoI (Persamaan 2.28), dan *RoI Align* linear terhadap peta fitur *neck* (Persamaan 2.18). Dengan bobot *node* per kanal ${w}_{i,c}$ dan peta fitur kanal ke-*c* pada tingkat terpilih ${F}_{c}$, bagian linear fungsi keputusan dapat ditulis sebagai Persamaan 3.4.

$$
{f}_{i}\left(x\right)-{b}_{i}=\sum_{c=1}^{C}{{w}_{i,c}^{T}\,A\,{F}_{c}}
$$

(3.4)

dengan $A$ matriks koefisien *RoI Align* (Persamaan 2.18) dan $C=64$. Gradien Persamaan 3.4 terhadap ${F}_{c}$ adalah ${A}^{T}{w}_{i,c}$, yaitu bobot *node* yang disebar kembali ke posisi asalnya pada peta fitur. Gradien ini dikalikan dengan nilai peta fitur (*Gradient × Input*, Persamaan 2.23), sebagaimana dinyatakan pada Persamaan 3.5.

$$
{c}_{i}\left[c,p\right]={d}_{i}\cdot {\left({A}^{T}{w}_{i,c}\right)}_{p}\cdot {F}_{c}\left(p\right)
$$

(3.5)

dengan ${c}_{i}\left[c,p\right]$ kontribusi kanal ke-*c* posisi $p$ terhadap *node* ke-*i*, dan ${d}_{i}$ arah keputusan *node*, sehingga kontribusi positif berarti mendukung arah yang diambil. Karena seluruh operasinya linear, *completeness* (Persamaan 2.24 dan 2.25) berlaku secara eksak (Persamaan 3.6).

$$
\sum_{c=1}^{C}{\sum_{p}{{c}_{i}\left[c,p\right]}}={d}_{i}\cdot \left({f}_{i}\left(x\right)-{b}_{i}\right)
$$

(3.6)

Untuk ditampilkan, kontribusi seluruh kanal diringkas menjadi satu peta dengan Persamaan 3.7.

$$
{H}_{i}\left(p\right)=\sum_{c=1}^{C}{\left|{c}_{i}\left[c,p\right]\right|}
$$

(3.7)

Nilai mutlak membuat ${H}_{i}$ menunjukkan besar pengaruh, baik yang mendukung maupun yang menentang, sedangkan arah keputusan ditampilkan pada jalur pohon.

${H}_{i}$ kemudian dipotong sesuai letak proposal pada tingkat piramida menggunakan *stride* (Persamaan 2.5 dan 2.6), dinormalisasi, lalu diperbesar dengan interpolasi bilinear (Persamaan 2.16) ke ukuran proposal pada citra. Ketelitiannya berada pada tingkat daerah karena perhitungan eksak berhenti pada peta fitur *neck* (Persamaan 2.7), tetapi lebih halus daripada grid $7\times 7$. Alurnya ditunjukkan pada Gambar 3.5.

[SISIPKAN GAMBAR: alur *heatmap* per *node*: bobot *node* ${w}_{i}$ (grid $64\times 7\times 7$) → disebar kembali melalui koefisien *RoI Align* (${A}^{T}{w}_{i}$) → dikalikan peta fitur *neck* (*Gradient × Input*) → ${H}_{i}$ → dipotong, dinormalisasi, dan diperbesar ke proposal pada citra, untuk setiap *node* pada jalur keputusan]

Gambar 3.5 Alur Pembentukan *Heatmap* Per *Node*

**Input:** Satu deteksi beserta *tensor* RoI, jalur keputusan, tingkat piramida, dan peta fitur *neck*.

**Output:** Satu *heatmap* untuk setiap *node* aktif pada jalur keputusan.

**Langkah-langkah:**

1. Untuk setiap *node* pada jalur keputusan, bobot ${w}_{i}$ dikembalikan ke grid $64\times 7\times 7$ dan dikalikan dengan ${d}_{i}$.
2. *RoI Align* dijalankan ulang pada peta fitur *neck*, lalu gradien Persamaan 3.4 dihitung dengan propagasi mundur.
3. Gradien dikalikan dengan peta fitur (Persamaan 3.5) dan diringkas menjadi ${H}_{i}$ (Persamaan 3.7).
4. ${H}_{i}$ dipotong, dinormalisasi, diperbesar ke ukuran proposal, lalu ditumpangkan pada citra di samping jalur keputusan pohon.
## 3.10 Evaluasi Model *Neuro-Symbolic*

Seluruh pengujian dilakukan pada data uji. Ringkasan skenario evaluasi ditunjukkan pada Tabel 3.7.

Tabel 3.7 Ringkasan Skenario Evaluasi

| Aspek | Pembanding | Metrik |
| --- | --- | --- |
| Kinerja deteksi | *Faster* R-CNN vs model hibrida | mAP, *Precision*, *Recall*, F1, *confusion matrix* |
| Ablasi skor | Model hibrida dengan dan tanpa *routing margin* | mAP, *Precision*, *Recall*, F1 |
| Fidelitas | SODT terhadap label *teacher* | *Mimic accuracy*, *macro*-F1, kesesuaian per kelas (Subbab 3.8) |
| *Faithfulness* tingkat jalur | SODT vs Grad-CAM vs kontrol | *Necessity Flip Rate*, *Sufficiency Preservation* |
| Lokalisasi | SODT vs Grad-CAM vs acak | *Pointing Game*, IoU *Heatmap* |
| *Faithfulness* per *node* | SODT vs kontrol acak | *Necessity Flip Rate* per *node*, *Deletion*/*Insertion* AUC |
| Waktu inferensi | *Faster* R-CNN vs model hibrida vs Grad-CAM | Waktu rata-rata per citra |
| Kualitatif | SODT vs Grad-CAM | Perbandingan visual per deteksi |

### 3.10.1 Kinerja Deteksi dan Ablasi *Routing Margin*

Model hibrida dibandingkan dengan *Faster* R-CNN menggunakan metrik pada Subbab 3.5. Untuk mengukur peran *routing margin*, metrik yang sama dihitung ulang dengan skor seluruh deteksi diganti menjadi 1, tanpa mengubah jalur maupun label.

### 3.10.2 *Faithfulness* Tingkat Jalur terhadap Grad-CAM

Grad-CAM menghasilkan satu peta per deteksi, sedangkan SODT satu peta per *node*. Untuk perbandingan, *heatmap* per *node* ditumpuk menjadi satu peta dengan Persamaan 3.8.

$$
M\left(p\right)=\sum_{i\in P\left(x\right)}{{H}_{i}\left(p\right)}
$$

(3.8)

Penumpukan dilakukan setelah setiap *heatmap* dimutlakkan, bukan dengan menjumlahkan kontribusi bertanda. SODT tidak memiliki satu skor jalur, sehingga penjumlahan bertanda dapat menghapus daerah yang mendukung satu *node* tetapi menentang *node* lain. Peta $M$ hanya dipakai untuk perbandingan dengan Grad-CAM.

Grad-CAM (Persamaan 2.21 dan 2.22) dihitung pada tingkat peta fitur *neck* tempat proposal di-*pool*, yaitu lapisan konvolusi terakhir yang dibaca *box head* (Selvaraju et al., 2017), dari skor kelas hasil *pooling* pada proposal asal deteksi.

Kedua metode diuji dengan protokol yang sama pada seluruh 500 citra uji. Yang dijelaskan adalah seluruh deteksi masing-masing model dengan skor ≥ 0,5, yaitu titik operasi *Precision* dan *Recall*, pada proposal asalnya. Pada peta fitur *neck* di dalam proposal (diperluas 2 posisi), 50% posisi bernilai tertinggi menurut peta masing-masing metode dipilih. *Necessity* (Persamaan 2.41) menolkan posisi tersebut, menjalankan ulang *RoI Align*, lalu memeriksa apakah label berubah. *Sufficiency* (Persamaan 2.42) hanya mempertahankan posisi tersebut, lalu memeriksa apakah label tetap. Kedua metode dibandingkan dengan kontrol acak, dan SODT juga dengan dua kontrol tambahan (Adebayo et al., 2018):

1. **Acak**: posisi dipilih secara acak.
2. **Bobot diacak**: bobot setiap *node* diacak sebelum Persamaan 3.8 dihitung.
3. **Aktivasi saja**: posisi diurutkan berdasarkan besar peta fitur tanpa bobot pohon.

### 3.10.3 Lokalisasi

*Pointing Game* (Persamaan 2.44) dan IoU *Heatmap* (Persamaan 2.45) dihitung antara peta dan *ground truth* pada proposal longgar (IoU 0,05–0,35) dari seluruh 500 citra uji. Karena kedua model memakai *backbone* dan RPN yang sama, SODT (peta gabungan), Grad-CAM, dan peta acak dievaluasi pada himpunan proposal yang identik, yaitu proposal yang diprediksi sebagai cacat oleh kedua model. Setiap peta di-*resample* ke grid $7\times 7$, dan ${H}_{bin}$ berisi sel bernilai tertinggi sebanyak sel *ground truth*.

### 3.10.4 *Faithfulness* Per *Node*

Klaim utama penjelasan SODT, yaitu bahwa daerah pada setiap *heatmap* menentukan keputusan *node*-nya, diuji dengan dua cara. Pertama, 50% posisi tertinggi ${H}_{i}$ dihapus, lalu diperiksa apakah tanda ${f}_{i}\left(x\right)$ berbalik. Kedua, *Deletion* dan *Insertion* AUC (Persamaan 2.43) dihitung dalam lima tahap atas $\sigma \left(\left|{f}_{i}\left(x\right)\right|\right)$. Keduanya dihitung pada deteksi yang sama dengan Subbab 3.10.2 dan dibandingkan dengan kontrol acak per kedalaman *node*, dan *node* yang telah dinolkan dilewati.

### 3.10.5 Waktu Inferensi

Waktu rata-rata per citra uji diukur untuk *Faster* R-CNN, model hibrida, dan Grad-CAM (termasuk pembentukan petanya).

### 3.10.6 Perbandingan Kualitatif

Untuk citra uji terpilih, deteksi *Faster* R-CNN, deteksi model hibrida, dan *ground truth* ditampilkan berdampingan. Setiap deteksi model hibrida ditampilkan dengan jalur keputusan dan *heatmap* setiap *node*, di samping peta Grad-CAM untuk deteksi yang sama.

**Input:** Model hibrida, *Faster* R-CNN, Grad-CAM, data uji, dan *dataset* simbolik uji.

**Output:** Metrik deteksi, ablasi, fidelitas, *faithfulness*, lokalisasi, waktu inferensi, dan perbandingan visual.

**Langkah-langkah:**

1. Sistem menghitung metrik deteksi *Faster* R-CNN dan model hibrida, termasuk ablasi tanpa *routing margin*.
2. Sistem menghitung *necessity* dan *sufficiency* SODT dan Grad-CAM beserta kontrolnya pada deteksi masing-masing model.
3. Sistem menghitung *Pointing Game* dan IoU *Heatmap* untuk SODT, Grad-CAM, dan peta acak pada proposal longgar yang sama.
4. Sistem menghitung *faithfulness* per *node* beserta kontrol acaknya.
5. Sistem mencatat waktu inferensi dan menampilkan perbandingan visual.


#
 
3. HASIL DAN PEMBAHASAN
Bab ini membahas secara komprehensif hasil dari implementasi arsitektur Neuro-Symbolic dengan kombinasi Faster R-CNN dan SODT dibandingkan dengan metode penjelasan baseline berupa Grad-CAM dalam konteks deteksi cacat pada Printed Circuit Board (PCB). Pembahasan mencakup rincian lingkungan pengujian, evaluasi masing-masing model, komparasi visual, hingga analisis kegagalan (failure cases).
 
1. Lingkungan Implementasi dan Dataset
Eksperimen dalam penelitian ini dilakukan pada lingkungan komputasi dengan spesifikasi perangkat keras dan perangkat lunak yang mumpuni untuk melatih model *deep learning* dan *decision tree*.
 
Perangkat keras yang digunakan meliputi prosesor Intel Core i5-13500 dan unit pemroses grafis (GPU) NVIDIA RTX 3060. Lingkungan perangkat lunak dibangun di atas sistem bahasa pemrograman Python versi 3.12 dengan memanfaatkan *framework* PyTorch versi terbaru yang kompatibel. Seluruh tahapan pelatihan dan pengujian dijalankan sepenuhnya mengikuti skema pembagian dataset DeepPCB serta penetapan *hyperparameter* yang telah diuraikan pada BAB sebelumnya.
 
2. Hasil Persiapan Dataset
Tahap persiapan dataset berhasil mentransformasi data mentah DeepPCB menjadi representasi numerik yang terstruktur dan siap dikonsumsi oleh arsitektur deteksi objek. Melalui proses partisi, validasi, dan standardisasi anotasi, seluruh tahapan berjalan sesuai dengan rancangan metodologis yang telah ditetapkan.
 
1. **Partisi dan Distribusi Data**
Proses partisi deterministik berdasarkan indeks partisi yang telah ditetapkan berhasil mengalokasikan 1.000 citra untuk himpunan pelatihan dan 500 citra untuk himpunan pengujian, sesuai dengan proporsi 66,67% dan 33,33%. Tidak ditemukan *overlap* antara kedua himpunan data, sehingga objektivitas evaluasi pada fase pengujian tetap terjaga.
 
Tabel 4.1 merangkum statistik keseluruhan dari kedua himpunan data. Dataset pelatihan memuat total 6.873 anotasi cacat dengan rata-rata 6,87 kotak pembatas per citra. Sementara itu, dataset pengujian mengandung 3.140 anotasi dengan rata-rata 6,28 kotak per citra. Seluruh citra pada kedua himpunan data memiliki minimal satu anotasi cacat, mengonfirmasi bahwa tidak ada citra kosong yang lolos ke dalam pipeline eksperimen.
 
Tabel 4.1 Statistik Partisi Dataset DeepPCB
 
**Metrik**
 
***Train Set***
 
***Test Set***
 
Jumlah Citra
 
1.000
 
500
 
Total Anotasi
 
6.873
 
3.140
 
Rata-rata Anotasi/Citra
 
6,87
 
6,28
 
Minimum Anotasi Citra
 
1
 
2
 
Maximum Anotasi Citra
 
15
 
13
 
Citra Kosong
 
0
 
0
 
1. **Distribusi Kelas Cacat**
Distribusi anotasi per kelas cacat pada kedua himpunan data ditampilkan pada Gambar 4.1 dan Gambar 4.2. Pada dataset pelatihan, kelas *mouse\_bite* mendominasi dengan sekitar 1.380 anotasi (20,1%), diikuti oleh kelas *open* dengan 1.280 anotasi (18,6%). Kelas *pinhole* memiliki representasi terendah dengan sekitar 1.010 anotasi (14,7%). Pola distribusi serupa terlihat pada dataset pengujian, di mana kelas *open* memiliki proporsi tertinggi sekitar 650 anotasi atau 20,7% dan kelas *pinhole* serta *spurious\_copper* memiliki proporsi terendah sekitar 465 hingga 470 anotasi atau 14,8% hingga 15,0%.
 
![Text document](asset:sha256:df0f527618e8a09b155ae0bcbf4011b8f6bb4c89343f2136f8227cf57c2fc8a9)
 
*1
spur
pinhole
20085000_test.jpg
mousebite
•
spurious_copper
2
mouse_bite
「
spur*
 
Gambar 4.1 Distribusi anotasi per kelas pada *training set.*
 
![Text document](asset:sha256:336ec358889ca6a84ef8ca4fd0b1e985bcb11dd37e812d27a740ecda25205386)
 
*Model Faster RCNN
Input
Backbone
Lokalisasi
CNN
FPN
RPN
Rol Align
-
Klasifikasi
Dataset Simbolik
Fitur Spasial 7x7
Hasil Prediksi
class prediction
Koordinat Bbox
bbox prediction*
 
Gambar 4.2 Distribusi anotasi per kelas pada *test set.*
 
1. **Kualitas Anotasi**
Validasi visual terhadap sampel acak anotasi, sebagaimana ditunjukkan pada Gambar 4.3, mengonfirmasi bahwa seluruh kotak pembatas telah ter-petakan dengan presisi pada wilayah cacat yang sesuai. Format koordinat spasial yang diterapkan pada anotasi geometris berhasil mempertahankan korespondensi antara anotasi dan citra target berukuran 640×640 piksel.
 
![Text document](asset:sha256:b6e08eb555049c63c5fbf1dd2ad76420f400a1f1bf5bc1df31542ab10cfecd23)
 
*Model Faster RCNN
Input
Backbone
Lokalisasi
bbox prediction
CNN
FPN
RPN
RolAlign
)
)
Klasifikasi
class prediction
Model Neuro-Symbolic (Modelyang diusulkan)
Input
Backbone
FPN
RPN
RolAlign
Lokalisasi
bbox prediction
CNN
-)
Klasifikasi
oleh SODT
class prediction
penjelasan
heatmap
Proses Klasifikasi dapat
dilacak/interpretasi*
 
Gambar 4.3 Sampel acak citra PCB dengan anotasi *bounding box ground truth*
 
Seluruh enam kelas cacat terdeteksi dalam *dataset* dengan representasi visual yang jelas. Tidak ditemukan anotasi yang mengalami distorsi geometris, koordinat yang keluar dari batas kanvas, atau kotak dengan dimensi tidak valid.
 
3. Hasil Pra-Pemrosesan Dataset
Prapemrosesan dataset dilakukan melalui dua pipeline transformasi yang terpisah untuk memastikan konsistensi spasial antara matriks piksel dan anotasi geometris. Hasil visualisasi prapemrosesan menunjukkan bahwa sinkronisasi koordinat berjalan presisi selama transformasi geometris.
 
1. **Augmentasi**
Gambar 4.4 menampilkan hasil visualisasi augmentasi pembalikan horizontal pada citra PCB. Pada tahap ini, sistem menerapkan transformasi pembalikan horizontal secara acak dengan probabilitas 0,5. Hasil visualisasi mengonfirmasi bahwa augmentasi berjalan sesuai dengan konfigurasi yang ditetapkan.
 
![Text document](asset:sha256:40cbaecf11f791e8d5c72cd987b0a7f4e5d9874f400f7b6c46d5aeb0091ee073)
 
*Annotations per Class
1400
1200
1000
800-
600
400
200
0
open
short
mouse_bite
spur
pinhole
spurious_copper*
 
Gambar 4.4 Hasil visualisasi augmentasi pembalikan horizontal pada citra PCB
 
Pada Gambar 4.4, terlihat bahwa semua *bounding box* bergerak sinkron dengan transformasi citra tanpa terdapat *offset* atau *drift* spasial. Sistem secara otomatis menyesuaikan koordinat *bounding box* menggunakan matriks transformasi yang sama untuk menjamin korespondensi geometris yang presisi. Hasil ini mengonfirmasi bahwa augmentasi berhasil mempertahankan integritas anotasi geometris selama proses transformasi.
 
1. **Multi-Scale Resizing**
Gambar 4.5 menampilkan hasil visualisasi proses penskalaan multi-resolusi dalam arsitektur model. Sistem menerapkan penskalaan resolusi yang berbeda-beda sesuai konfigurasi yang telah ditetapkan, yaitu sisi terpendek {480, 560, 640, 720, 800, 880} piksel.
 
![Text document](asset:sha256:a8683ea7b3dd37ca268c87e8d1868abe04de255998b9e7a569d3051f371289ae)
 
*Annotations per Class
600
/
500
sr
400
300
200
100
0
open
short
mouse_bite
spur
pinhole
spurious_copper*
 
Gambar 4.5 Hasil visualisasi penskalaan multi-resolusi dalam arsitektur model
 
Terlihat bahwa sistem penerapan *padding* *internal* secara otomatis mempertahankan topologi spasial tanpa distorsi geometris. Hasil ini mengkonfirmasi bahwa parameter penskalaan multi-resolusi beroperasi sesuai ekspektasi dan tidak menyebabkan distorsi anotasi. Pada setiap tahap pra-pemrosesan, sistem berhasil mempertahankan konsistensi antara citra dan anotasi geometris
 
1. **Normalisasi Dataset**
Normalisasi statistik tidak menghasilkan perubahan visual yang terlihat pada citra *grayscale* DeepPCB. Hal ini disebabkan oleh sifat grayscale yang hanya memiliki satu saluran intensitas, sehingga transformasi normalisasi tidak mengubah kontras atau kecerahan citra.
 
Meskipun tidak terlihat secara visual, normalisasi ini tetap penting untuk keselarasan distribusi fitur dengan bobot pre-trained, yang memastikan ekstraksi fitur yang optimal pada tahap deteksi.
 
## Hasil dan Evaluasi Faster RCNN
 
1. **Analisis Konvergensi**
Proses pelatihan model *Faster R-CNN* dilakukan selama 12 *epoch* dengan memantau pergerakan nilai fungsi kerugian (*loss*) secara berkala. Tren penurunan *loss* selama masa pelatihan disajikan pada Gambar 4.6.
 
![Text document](asset:sha256:0333e4604aebefb62c6cb6835db8e3feb0bc984b5527c087d788ef2ab7692026)
 
*Random Samples (6)
50600075_test |5 annotations
00041219_test |5 annotations
12100190_test |6 annotations
mouse_bite
short
spurious_copper
mouse_bite
open
open
spur
7
spur
mouse_bite
2
7
7
spurious_copper
mouse_bite
pinhole
spurious_copper
N
open
open
…
2
short
S
00041200_test |5 annotations
12100159_test |9 annotations
12100192 test|7 annotations
mouse_bite
?
mouse bite Ise bite
mouse _bite
short
-
pinhole
mouse_bite
国
spurious_copper
spur
pinhole
open
pinhole
spur
open
2
□
open
spurious_copper
pinhole
spurious_copper
short
国
√
pinhole
mouse_bite
可*
 
Gambar 4.6 Grafik Total Training Loss Faster R-CNN selama 12 Epoch\_\_.\_\_
 
Berdasarkan grafik *Total Training Loss* pada Gambar 4.6, terlihat bahwa model mengalami konvergensi yang sangat stabil. Nilai *loss* total turun secara monoton dari angka 0,804 pada *epoch* pertama hingga mencapai 0,259 pada *epoch* terakhir. Pola penurunan yang mulus ini mengindikasikan bahwa model berhasil mengadaptasikan parameter bobotnya terhadap karakteristik citra cacat PCB tanpa menunjukkan gejala *overfitting* maupun *underfitting*.
 
Grafik *Detector Loss Breakdown* di Gambar 4.6 menunjukkan bahwa komponen *loss\_classifier* dan *loss\_box\_reg* merupakan penyumbang utama kerugian pada fase awal, sejalan dengan kompleksitas tugas regresi koordinat kontinu dan klasifikasi jenis cacat. Sementara itu, komponen *loss\_objectness* dan *loss\_rpn\_box\_reg* mencapai nilai yang sangat rendah (di bawah 0,02) sejak *epoch* ketiga. Hal ini membuktikan efisiensi *Region Proposal Network* (RPN) dalam memisahkan wilayah *foreground* cacat dari latar belakang sirkuit PCB sejak fase awal pelatihan.
 
1. **Kinerja Keseluruhan Kelas**
Model diuji menggunakan himpunan data uji untuk mengevaluasi kinerja deteksi secara agregat. Metrik kinerja pada Gambar 4.7 mencatat nilai *mean Average Precision* (mAP@0.5) sebesar 0,982, yang menunjukkan tingkat kesesuaian spasial tinggi dengan *ground truth*.
 
.
 
![Text document](asset:sha256:8af7600cd1f3a46a5d4d39c74d8799a3ef7e7dfbac4a4127f76aaf1323dedb8e)
 
*Dataset Preprocessing Preview
50600013_test |original
preprocess pass 1 | flipped: no
preprocess pass 2 | flipped: no
mouse_bite
mouse_bite
mouse_bite
pinhole
pinhole
pinhole
spurious copper
spurious copper
spurious copper
-
short
short
short
open
open
open
00041069_test |original
preprocess pass 1 | flipped: yes
preprocess pass 2 | flipped: no
spur
spur
spur
F
1
open
mouse_bite
mouse_bite
open
open
mouse_bite
N
pinhole
pinhole
pinhole
spur
spur
spur
spurious_copper
spurious_copper
spurious_copper
spur
spur
spur
mousebite
mouse_bite
mouse_bite
7
20085089_test | original
preprocess pass 1 | flipped: yes
preprocess pass 2 | flipped: no
57
5
short
short
short
F
-1
日
open
spur
spur
open
open
spur
Y
1
open
spur open
spur
open
Spur
mouse_bite
mouse_bite
mouse_bite
L
1
short
mouse_bite
short
short
mouse short
short
mouse_bite
short*
 
Gambar 4.7 Metrik Deteksi Keseluruhan Faster R-CNN pada Dataset Uji
 
Performa model yang sangat tinggi ini mereplikasi metodologi modifikasi pengekstrakan fitur yang diusulkan oleh (Fung et al., 2024). Integrasi komponen *SF-PSPyramid* (gabungan *Selective Feature Attention* dan *Pixel Shuffle Pyramid*) secara signifikan memperkuat kemampuan model dalam mendeteksi objek sirkuit mikro. Mekanisme atensi visual ini memungkinkan model menyalurkan informasi semantik bernilai tinggi dari lapisan terdalam ke peta fitur beresolusi tinggi tanpa memicu distorsi geometris. Hal inilah yang mendasari tingginya nilai *recall* model (0,980), yang memastikan detektor hampir tidak melewatkan satupun cacat mikro pada papan PCB uji (Fung et al., 2024).
 
1. **Analisis Kinerja Per Kelas Cacat**
Analisis detail performa deteksi pada tiap kategori cacat dilakukan dengan meninjau nilai *Precision* dan *Recall* sebagaimana disajikan pada Gambar 4.8.
 
![Text document](asset:sha256:f6bf1ba2b24bd932ec5da46753d79ddb1d4d860f0aa929f1fd32f0abed03a277)
 
*Train RCNN Preprocessing Preview
00041115_test | dataset preprocess
RCNN pass 1 |resized 800x800 | padded 800x800 | norm -2.12RΩN4pass 2 | resized 880x880 | padded 896x896 | norm -2.12..2.64
pinhole
pinhole
pinhole
-
mouse_bite
spurious_copper
?
mouse uite
。
mouse_bite
spurious_copper
mouse_bite
mouse_bite
spurious_copper
mouse_bite
open
short
pinhole
open
short
mouse_bite
spur
open
short
pinhole
厂
T
pinhole
mouse_bite
spur
7
mouse_bite
spur
13000141_test |dataset preprocess
RCNN pass 1 | resized 880x880 | padded 896x896 | norm -2.1R@Ni4pass 2 | resized 720x720 | padded 736x736 | norm -2.12..2.64
spur
pinhole
spur
pinhole
S
mouse_bite
pinhole
spur
/
spurious copper
Q
mouse_bite
4
open
mouse bite
spurious_copper
spurious_copper
open
open
spur
spur
spur
20085032_test | dataset preprocess
RCNN pass 1 | resized 720x720 | padded 736x736 | norm -2.1RΩNi4pass 2 |resized 800x800 | padded 800x800 | norm -2.12..2.64
spurious_copper
spurious_copper
spurious_copper
mouse_bite
mouse_bite
spurious_copper
mouse_bite
spurious_copper
open
spurious_copper
open
open
pinhole
.
spur
open
open
pinhole
openH
F
.
spur
pinhole
F
●
spur
●
•*
 
Gambar 4.8 Per-Class *Precision* dan *Recall* Faster R-CNN untuk Enam Kategori Cacat
 
Model menunjukkan kinerja yang sangat baik pada kelas *mouse\_bite* dan *open* karena pola geometrisnya yang konsisten dan kontras terhadap latar belakang. Sebaliknya, kinerja kelas *short* dan *spurious\_copper* berada di posisi terendah. Penurunan nilai pada kedua kelas ini disebabkan oleh karakteristik geometrisnya yang kompleks seperti contohnya kelas *short* memiliki kemiripan visual yang ekstrem dengan konduktor asli, sedangkan variasi skala dan orientasi yang sangat acak pada *spurious\_copper* membuat model kesulitan, meskipun keberadaan objeknya tetap berhasil diidentifikasi dengan tingkat *recall* yang tinggi.
 
1. **Analisis *Confusion Matrix***
Gambar 4.9 menyajikan matriks konfusi yang memvisualisasikan distribusi prediksi model terhadap *ground truth* untuk seluruh kategori, termasuk latar belakang (*background*). Diagonal utama matriks menunjukkan nilai yang dominan untuk setiap kelas aktual. Dominasi nilai pada diagonal ini mengonfirmasi bahwa model memiliki tingkat akurasi klasifikasi yang tinggi dengan minimal kesalahan kategorisasi antarjenis cacat.
 
![Text document](asset:sha256:545aca11eff007ddc7022e76daeb244a2b9d8282f8c3d1a043877547ad822f08)
 
*Total Training Loss
Detector Loss Breakdown
0.804
loss_classifier
0.4
0.391
0.390
0.8
中
•
loss_box_reg
中
0.351
?
loss_objectness
0
0.7
0.316
loss_rpn_box_reg
0634
0.3
0.285
0.27
G
0.274
•
0.265
价
•
0.249
0.6
0.216
9
SS0I
0526
SSOI
0.207
0.2
a
8
0.198
0.194
0.188
0.5
n
〜
0.470
0153
市
•
0
0.134
0.415
0.395
0.115
T
0.113
0.380
0.106
0.4
0.100
市
0.1
Q
0.093
市
*
0.356
市
4
0.072
t
1
.
0.067
0.065
0.062
市
•
节
0.3
0.289
0.274
0.021
0.268
8.月17
8.088
0.259
8.03
8.033
8.064
8.039
0.009
0.008
0.008
8.0
8.0
0
•
0.0
•
0
•
-
-
-
-
ü
-
0.2-
2
4
6
8
10
12
2
4
6
8
10
12
epoch
epoch*
 
Gambar 4.9 *Confusion Matrix* Deteksi Faster R-CNN pada *test set*
 
Baris *background* mengungkap adanya sejumlah *false positive* yang tersebar di seluruh kelas, dengan total 359 deteksi yang keliru diklasifikasikan sebagai cacat. Sebagian kecil dari cacat juga teridentifikasi sebagai *background*. Hal ini menunjukkan bahwa model masih berkecenderungan untuk mengidentifikasi pola tekstur sirkuit normal sebagai anomali, terutama untuk kelas *short* dan *spurious\_copper*. Fenomena ini disebabkan oleh tingginya kompleksitas pola jalur konduktor pada PCB yang terkadang memiliki kemiripan visual dengan cacat adisi material.
 
## Hasil Ekstraksi Fitur *Teacher*
 
1. **Statistik RoI yang ter-ekstrak**
Tabel 4.2 dan 4.3 merangkum statistik Region of Interest (RoI) yang berhasil diekstrak dari dataset.
 
Tabel 4.2 Statistik Ekstraksi RoI
 
***Split***
 
***Total RoI***
 
***Positive RoI***
 
***Background RoI***
 
Training
 
1.000.000
 
133.936 (13,4%)
 
866.064 (86,6%)
 
Test
 
500.000
 
437.148 (87,4%)
 
62.852 (12,6%)
 
Tabel 4.3 Distribusi RoI Per kelas cacat
 
***Split***
 
**Kelas**
 
**Jumlah\* RoI\***
 
**Persentase**
 
Training
 
*open*
 
27.020
 
2,70%
 
*Short*
 
18.704
 
1,87%
 
*mouse\_bite*
 
26.871
 
2,69%
 
*Spur*
 
21.519
 
2,15%
 
*Pinhole*
 
19.090
 
1,91%
 
*spurious\_copper*
 
20.732
 
2,07%
 
Test
 
*open*
 
13.707
 
2,74%
 
*Short*
 
9.387
 
1,88%
 
*mouse\_bite*
 
11.437
 
2,29%
 
*Spur*
 
8.792
 
1,76%
 
*Pinhole*
 
9.142
 
1,83%
 
*spurious\_copper*
 
10.387
 
2,08%
 
Meskipun dataset pelatihan hanya terdiri dari 1.000 citra, sistem berhasil mengekstrak total 1.500.000 RoI (rata-rata 1.000 RoI per citra). Volume yang besar ini berasal dari cara kerja *Region Proposal Network* (RPN). RPN secara otomatis mengevaluasi ratusan *anchor boxes* dengan berbagai skala dan rasio aspek pada setiap citra, yang kemudian dibatasi maksimal 1.000 proposal per citra melalui *Non-Maximum Suppression* (NMS).
 
Selain itu, dominasi kelas *background* yang mencapai 87% adalah hal yang wajar karena sebagian besar area PCB adalah sirkuit normal. RPN memang dirancang sangat sensitif (*high recall*) untuk meminimalkan risiko cacat yang terlewat (*false negative*). Kelebihan proporsi *background* ini tidak akan menjadi masalah karena akan diseimbangkan nanti melalui teknik *negative sampling* pada tahap pelatihan SODT.
 
1. **Fitur Spasial 7x7**
Untuk memvisualisasikan tensor $64\times 7\times 7$, sistem melakukan *mean pooling* pada dimensi kanal sehingga menghasilkan peta aktivasi spasial berukuran $7\times 7$. Tiga sampel representatif, yaitu kelas *background*, cacat "open", dan cacat "spur"—ditampilkan pada Gambar 4.20, 4.21, dan 4.22.
 
![Text document](asset:sha256:02aaf3add41ff953e5baa14c96c00c40afab687670ff93173ecd80b14064fcb9)
 
*Final Detection Metrics
1.0
0.982
0.980
0.924
0.907
0.936
0.896
0.853
0.8
0.764
0.6
0.4
0.2
0.0
mAP@0.5:0.95
mAP@0.5
AP75
AP@50:5:85
precision
recall
mar_100
fl_score*
 
Gambar 4.10 Peta aktivasi spasial 7×7 untuk kelas *background*
 
![Text document](asset:sha256:b446129dd272351e990ba625f6a74a8fb555f08bb5fafcb9eaaa67592b50313e)
 
*Per-Class Precision and Recal
1.0
0.953
0.982
0.964
0.983
0.969
0.989
0.994
0.950
0.955
0.896
0.822
0.8
0.794
0.6
0.4
0.2
Precision
Recall
0.0
open
short
spur
pinhole*
 
Gambar 4.11 Peta aktivasi spasial 7×7 untuk kelas *open*
 
![Text document](asset:sha256:55ad5cc745bf0364e7be9535a49ff6a14afb133ddda916e209359592b360b4f1)
 
*Detection Confusion Matrix
-600
background
0
32
100
30
22
53
121
-500
open
12
647
0
0
0
0
0
short
17
0
461
0
0
0
0
-400
anull
mouse_bite
10
0
0
576
0
0
0
-300
spur-
15
0
0
0
468
0
0
-200
pinhole
5
0
0
0
0
459
0
100
spurious_copper
3
0
0
0
0
0
467
open
short
mouse_bite
spur
pinhole
spurious_copper
0
Predicted*
 
Gambar 4.11 Peta aktivasi spasial 7×7 untuk kelas *spur*
 
Perbedaan pola aktivasi antara ketiga kelas tampak jelas. Pada kelas *background* di Gambar 4.10, aktivasi tersebar secara merata tanpa konsentrasi di area tertentu, yang mencerminkan tidak adanya anomali visual. Sebaliknya, pada kelas cacat *open* di Gambar 4.11, aktivasi terpusat di area tengah grid, menandakan bahwa fitur FPN berhasil menangkap celah pada jalur konduktor di lokasi spasial yang sesuai. Pada kelas cacat \*spur *di* \*Gambar 4.22, pola aktivasi menunjukkan konsentrasi yang lebih tajam dan terlokalisasi pada satu sisi grid, konsisten dengan karakteristik spur yang berupa tonjolan kecil di tepi jalur tembaga.
 
Ketiga visualisasi ini membuktikan bahwa RoI Align berhasil mempertahankan informasi spasial yang diskriminatif antar kelas. Mempertahankan struktur grid 7×7 ini sangat krusial bagi penelitian karena setiap sel grid mewakili area spasial spesifik pada citra asli. Ketika SODT membuat keputusan klasifikasi, bobot pada grid ini dapat dipetakan kembali menjadi *heatmap* yang *faithful*, yang menjamin bahwa area yang disorot pada penjelasan visual benar-benar merupakan dasar logis keputusan model, bukan sekadar perkiraan *post-hoc*.
 
## Evaluasi Model Simbolik (*Sparse Oblique Decision Tree*)
 
1. **Pelatihan TAO**
Proses optimasi parameter SODT menggunakan *Tree Alternating Optimization* (TAO) menunjukkan pola konvergensi yang khas, sebagaimana terlihat pada Gambar 4.10. Kurva pelatihan menggambarkan dua aspek kritis, yaitu peningkatan fidelitas model terhadap *teacher* (Faster R-CNN) dan penurunan jumlah bobot non-nol yang terjadi secara progresif sepanjang iterasi.
 
![Text document](asset:sha256:1370218f4626fd3da66a890266a42992b238fd9769994025b7ad64d8690c9d9e)
 
*Rol Align Spatial Topology (C = 64 → mean pool → 7 × 7)
Sample 1
Sample 2
score=1.0000
score=1.0000
_background_
score=1.0000
score=0.9995
open
score=0.9995
score=0.9995
short
score=1.0000
score=0.9995
mouse_bite
score=0.9995
score=0.9995
spur
score=1.0000
score=1.0000
pinhole
score=0.9995
score=1.0000
spurious_copper*
 
Gambar 4.10 Grafik TAO Training History
 
Pada iterasi awal (0–5), terjadi peningkatan fidelitas yang tajam dari 75% menjadi 92%, diikuti oleh konvergensi bertahap menuju 96,7% pada iterasi ke-15. Pola ini mengindikasikan bahwa algoritma TAO berhasil menemukan parameter optimal yang mempertahankan kesetiaan terhadap *teacher* sambil meningkatkan sparsitas melalui penalti L₁.
 
Pada iterasi terakhir, terjadi penurunan drastis jumlah bobot non-nol dari sekitar 60.000 menjadi 3.326. Fenomena ini disebabkan oleh *post-processing* berupa *node pruning* yang tidak memberikan kontribusi signifikan terhadap keputusan klasifikasi. Proses ini memastikan struktur pohon tetap ramping tanpa mengorbankan fidelitas model, sehingga menghasilkan representasi logika yang lebih mudah diinterpretasikan oleh manusia.
 
1. **Fidelitas Terhadap *Teacher***
Evaluasi fidelitas dilakukan melalui metrik *Teacher-Student Agreement*, yang mengukur kesesuaian prediksi SODT dengan model *teacher* (Faster R-CNN). Seperti ditunjukkan pada Gambar 4.11, model mencapai tingkat kesesuaian sebesar 96,71% pada data pelatihan dan 96,77% pada data uji. Stabilitas metrik ini antara dua himpunan data membuktikan bahwa SODT berhasil meniru logika *teacher* tanpa mengalami *overfitting*.
 
![Text document](asset:sha256:1370218f4626fd3da66a890266a42992b238fd9769994025b7ad64d8690c9d9e)
 
*Rol Align Spatial Topology (C = 64 → mean pool → 7 × 7)
Sample 1
Sample 2
score=1.0000
score=1.0000
_background_
score=1.0000
score=0.9995
open
score=0.9995
score=0.9995
short
score=1.0000
score=0.9995
mouse_bite
score=0.9995
score=0.9995
spur
score=1.0000
score=1.0000
pinhole
score=0.9995
score=1.0000
spurious_copper*
 
Gambar 4.11 Grafik *Teacher-Student Agreement* pada Data Pelatihan dan Data Uji
 
Analisis lebih detail pada tingkat kelas cacat pada Gambar 4.12 menunjukkan variasi fidelitas yang moderat di antara keenam kategori. Kelas *spur* dan *background* mencatatkan kesesuaian tertinggi (97,0%), sedangkan kelas *short* menunjukkan nilai terendah (91,1%) akibat variasi geometris yang lebih kompleks. Meskipun demikian, seluruh kelas mempertahankan fidelitas di atas 90%, mengonfirmasi bahwa SODT mampu mereplikasi keputusan *teacher* secara konsisten.
 
![Text document](asset:sha256:1370218f4626fd3da66a890266a42992b238fd9769994025b7ad64d8690c9d9e)
 
*Rol Align Spatial Topology (C = 64 → mean pool → 7 × 7)
Sample 1
Sample 2
score=1.0000
score=1.0000
_background_
score=1.0000
score=0.9995
open
score=0.9995
score=0.9995
short
score=1.0000
score=0.9995
mouse_bite
score=0.9995
score=0.9995
spur
score=1.0000
score=1.0000
pinhole
score=0.9995
score=1.0000
spurious_copper*
 
Gambar 4.12 *Per-Class Agreement* antara SODT dan *Teacher* pada Data *Held-out*
 
1. **\*Sparsity \*Model**
Sparsitas menjadi faktor krusial dalam meningkatkan interpretabilitas model. Gambar 4.13 memvisualisasikan distribusi bobot pada seluruh simpul internal pohon, menunjukkan bahwa dari total 97.216 parameter, hanya 3.326 bobot yang bernilai non-nol (96,6% sparsitas). Tingkat sparsitas ini memiliki implikasi langsung terhadap kualitas penjelasan visual yang dihasilkan.
 
![Text document](asset:sha256:87aa0213bf3a65a40a083ad29d65c5f8d6bc8d08b1588440f96d3c42a6821095)
 
*Single SODT Student Summary
Teacher-Student Agreement
Global SODT Sparsity
1.0
96.71%
96.77%
0.8
Koeunoe sis
0.6
Split weights
3,326 /97,216 active
96.6% sparse
0.4
0.2
Nonzero
Zero
0.0
Train
Held-out test
0
20000
40000
60000
80000
Weights across all internal nodes
Pruned Tree Structure (12 active nodes)
Held-out Per-Class Agreement
412
_background_
97.0%
399
background
open
94.7%
short
91.1%
619
317
mouse_bite
95.8%
380
78
461
0
spur
97.0%
background_
④
pinhole
24
476
48
short
?
pinhole-
96.0%
spurious_copper
95.9%
pinhole spur
open
p_backg spurious_copr spur
short
mouse_bite short
0.0
0.2
0.4
0.6
0.8
1.0
Agreement
Spatial Grounding (held-out)
TAO Training History
1.0
1.0
80000
0.818
70000
0.8
Koeunsse o ulerl
0.8
中
60000
0.643
0.6
0.6
50000
40000
0.4
0.4
30000
20000
0.2
0.2
Mimic
10000
Nonzero weigh
s
0.0
0.0
0
Box overlap
Pointing score
2
4
6
8
10
12
14
16
TAO iteration*
 
Gambar 4.13 Visualisasi *Global SODT Sparsity*
 
Sparsitas tinggi memaksa setiap *node* keputusan hanya bergantung pada subset fitur spasial yang paling informatif. Hal ini menghasilkan heatmap yang lebih bersih dan terlokalisasi dengan presisi, sehingga teknisi dapat dengan mudah mengidentifikasi wilayah kritis yang menjadi dasar keputusan klasifikasi. Struktur pohon yang *terpruning* seperti pada Gambar 4.14 hanya terdiri dari 12 simpul aktif, yang berada dalam batas kapasitas kognitif manusia untuk melacak jalur penalaran secara manual.
 
![Text document](asset:sha256:87aa0213bf3a65a40a083ad29d65c5f8d6bc8d08b1588440f96d3c42a6821095)
 
*Single SODT Student Summary
Teacher-Student Agreement
Global SODT Sparsity
1.0
96.71%
96.77%
0.8
Koeunoe sis
0.6
Split weights
3,326 /97,216 active
96.6% sparse
0.4
0.2
Nonzero
Zero
0.0
Train
Held-out test
0
20000
40000
60000
80000
Weights across all internal nodes
Pruned Tree Structure (12 active nodes)
Held-out Per-Class Agreement
412
_background_
97.0%
399
background
open
94.7%
short
91.1%
619
317
mouse_bite
95.8%
380
78
461
0
spur
97.0%
background_
④
pinhole
24
476
48
short
?
pinhole-
96.0%
spurious_copper
95.9%
pinhole spur
open
p_backg spurious_copr spur
short
mouse_bite short
0.0
0.2
0.4
0.6
0.8
1.0
Agreement
Spatial Grounding (held-out)
TAO Training History
1.0
1.0
80000
0.818
70000
0.8
Koeunsse o ulerl
0.8
中
60000
0.643
0.6
0.6
50000
40000
0.4
0.4
30000
20000
0.2
0.2
Mimic
10000
Nonzero weigh
s
0.0
0.0
0
Box overlap
Pointing score
2
4
6
8
10
12
14
16
TAO iteration*
 
Gambar 4.14 Diagram Struktur Pohon SODT
 
Kombinasi antara fidelitas tinggi dan sparsitas ekstrem membuktikan bahwa SODT tidak hanya mampu mereplikasi keputusan *teacher*, tetapi juga menghasilkan penjelasan yang *faithful* dan mudah dipahami. Dengan mengeliminasi fitur yang tidak relevan, model secara alami mengarahkan perhatian teknisi ke wilayah spasial yang paling kritis untuk validasi, sehingga memenuhi tujuan utama arsitektur neuro-simbolik dalam mendukung proses inspeksi PCB yang transparan dan akuntabel.
 
## Evaluasi Kinerja Deteksi Neuro-Symbolic
 
1. **Hasil Keseluruhan**
Evaluasi kinerja deteksi model Neuro-Symbolic (NeSy) secara keseluruhan diukur menggunakan berbagai metrik standar untuk melihat seberapa baik model terintegrasi ini melokalisasi dan mengklasifikasikan cacat PCB. Hasil metrik deteksi secara agregat disajikan pada Gambar 4.15.
 
![Text document](asset:sha256:87aa0213bf3a65a40a083ad29d65c5f8d6bc8d08b1588440f96d3c42a6821095)
 
*Single SODT Student Summary
Teacher-Student Agreement
Global SODT Sparsity
1.0
96.71%
96.77%
0.8
Koeunoe sis
0.6
Split weights
3,326 /97,216 active
96.6% sparse
0.4
0.2
Nonzero
Zero
0.0
Train
Held-out test
0
20000
40000
60000
80000
Weights across all internal nodes
Pruned Tree Structure (12 active nodes)
Held-out Per-Class Agreement
412
_background_
97.0%
399
background
open
94.7%
short
91.1%
619
317
mouse_bite
95.8%
380
78
461
0
spur
97.0%
background_
④
pinhole
24
476
48
short
?
pinhole-
96.0%
spurious_copper
95.9%
pinhole spur
open
p_backg spurious_copr spur
short
mouse_bite short
0.0
0.2
0.4
0.6
0.8
1.0
Agreement
Spatial Grounding (held-out)
TAO Training History
1.0
1.0
80000
0.818
70000
0.8
Koeunsse o ulerl
0.8
中
60000
0.643
0.6
0.6
50000
40000
0.4
0.4
30000
20000
0.2
0.2
Mimic
10000
Nonzero weigh
s
0.0
0.0
0
Box overlap
Pointing score
2
4
6
8
10
12
14
16
TAO iteration*
 
Gambar 4.15 Metriks deteksi NeSy.
 
Model NeSy mencatatkan nilai *Recall* yang sangat tinggi, yaitu 0,985, serta *F1-Score* sebesar 0,897. Nilai *Recall* yang mendekati 1,0 ini mengindikasikan bahwa model hampir tidak melewatkan cacat yang ada pada citra uji (*low false negative*). Namun, terdapat kesenjangan yang cukup signifikan pada nilai *Precision* yang bernilai 0,823 dan *mAP@0.5 yang bernilai* 0,87. Penurunan pada metrik *Precision* dan *mAP* ini tidak disebabkan oleh kegagalan model dalam mendeteksi cacat, melainkan akibat tingginya tingkat *False Positive* yang dihasilkan oleh model. Fenomena penurunan presisi dan *mAP* ini terkonfirmasi secara visual melalui Matriks Konfusi pada Gambar 4.16.
 
![Text document](asset:sha256:87aa0213bf3a65a40a083ad29d65c5f8d6bc8d08b1588440f96d3c42a6821095)
 
*Single SODT Student Summary
Teacher-Student Agreement
Global SODT Sparsity
1.0
96.71%
96.77%
0.8
Koeunoe sis
0.6
Split weights
3,326 /97,216 active
96.6% sparse
0.4
0.2
Nonzero
Zero
0.0
Train
Held-out test
0
20000
40000
60000
80000
Weights across all internal nodes
Pruned Tree Structure (12 active nodes)
Held-out Per-Class Agreement
412
_background_
97.0%
399
background
open
94.7%
short
91.1%
619
317
mouse_bite
95.8%
380
78
461
0
spur
97.0%
background_
④
pinhole
24
476
48
short
?
pinhole-
96.0%
spurious_copper
95.9%
pinhole spur
open
p_backg spurious_copr spur
short
mouse_bite short
0.0
0.2
0.4
0.6
0.8
1.0
Agreement
Spatial Grounding (held-out)
TAO Training History
1.0
1.0
80000
0.818
70000
0.8
Koeunsse o ulerl
0.8
中
60000
0.643
0.6
0.6
50000
40000
0.4
0.4
30000
20000
0.2
0.2
Mimic
10000
Nonzero weigh
s
0.0
0.0
0
Box overlap
Pointing score
2
4
6
8
10
12
14
16
TAO iteration*
 
Gambar 4.16 *Confusion Matrix* pada Neuro-Symbolic
 
Pada baris *\_background\_* di Gambar 4.16, terdapat total 667 *False Positive* di mana area sirkuit normal (*background*) diklasifikasikan secara keliru sebagai salah satu dari enam kelas cacat. Sebaliknya, pada baris kelas cacat, hanya terdapat 47 *False Negative* (cacat yang terlewat dan dikira background). Dominasi *False Positive* dari background inilah yang secara matematis memberikan penalti besar pada perhitungan *Precision* dan *mAP*, sehingga menyebabkan nilai keduanya lebih rendah dibandingkan \*Recall. \*
 
Meskipun menghasilkan banyak Fal*se Positive* dari *background*, *Confusion Matrix* NeSy pada Gambar 4.16 menunjukkan jumlah True Positive pada beberapa kelas lebih tinggi dibanding model *teacher* (Faster R-CNN). Misalnya pada kelas open yang di mana pada NeSy sebanyak 650 sementara Faster RCNN sebanyak 647. Hal ini menunjukkan bahwa SODT mampu mengklasifikasikan ulang RoI yang sebelumnya salah dikategorikan oleh MLP Faster R-CNN. Namun, hal ini membuat SODT cenderung *over-sensitif* terhadap pola *background* yang kompleks dan mirip dengan cacat yang asli. Oleh karena itu, SODT berhasil memperbaiki batas keputusan untuk kasus-kasus ambigu, tetapi belum cukup selektif dalam membedakan background normal dari cacat mikro
 
Gambar 4.17 menyajikan rincian kinerja deteksi pada tingkat per-kelas. Nilai Recall untuk seluruh kelas cacat berada di atas 0,96, namun nilai Precision bervariasi. Kelas short (0,740) dan spurious\_copper (0,760) memiliki presisi terendah, yang mengindikasikan bahwa kedua kelas ini paling sering memicu False Positive dari area background.
 
![Text document](asset:sha256:87aa0213bf3a65a40a083ad29d65c5f8d6bc8d08b1588440f96d3c42a6821095)
 
*Single SODT Student Summary
Teacher-Student Agreement
Global SODT Sparsity
1.0
96.71%
96.77%
0.8
Koeunoe sis
0.6
Split weights
3,326 /97,216 active
96.6% sparse
0.4
0.2
Nonzero
Zero
0.0
Train
Held-out test
0
20000
40000
60000
80000
Weights across all internal nodes
Pruned Tree Structure (12 active nodes)
Held-out Per-Class Agreement
412
_background_
97.0%
399
background
open
94.7%
short
91.1%
619
317
mouse_bite
95.8%
380
78
461
0
spur
97.0%
background_
④
pinhole
24
476
48
short
?
pinhole-
96.0%
spurious_copper
95.9%
pinhole spur
open
p_backg spurious_copr spur
short
mouse_bite short
0.0
0.2
0.4
0.6
0.8
1.0
Agreement
Spatial Grounding (held-out)
TAO Training History
1.0
1.0
80000
0.818
70000
0.8
Koeunsse o ulerl
0.8
中
60000
0.643
0.6
0.6
50000
40000
0.4
0.4
30000
20000
0.2
0.2
Mimic
10000
Nonzero weigh
s
0.0
0.0
0
Box overlap
Pointing score
2
4
6
8
10
12
14
16
TAO iteration*
 
Gambar 4.17 Rincian \*Precision \*dan \*Recall \*per kelas cacat.
 
Untuk memvalidasi temuan kuantitatif tersebut secara visual, Gambar 4.18 menyajikan perbandingan hasil inferensi antara Faster R-CNN, NeSy, dan *Ground Truth* (GT). Berdasarkan gambar tersebut, terlihat bahwa NeSy secara umum berhasil mendeteksi mayoritas cacat yang sejalan dengan GT, membuktikan bahwa substitusi MLP dengan SODT tidak mengganggu kemampuan deteksi spasial. Namun, pada beberapa sampel seperti pada Gambar 4.19, NeSy menghasilkan *bounding box* tambahan pada area sirkuit normal, yang secara visual mengonfirmasi bahwa penurunan *Precision* murni disebabkan oleh sensitivitas berlebih terhadap *background*, bukan karena kegagalan melokalisasi cacat yang sebenarnya.
 
![Text document](asset:sha256:bc659570c32fe03f814b89c61b876d1113d85cc8befdbae31759321848f22256)
 
*NeSy Detection Metrics
1.0
0.985
0.897
0.870
0.847
0.808
0.795
0.823
0.8
0.668
0.6
0.4
0.2
0.0
mAP@0.5:0.95
mAP@0.5
AP75
AP@50:5:85
precision
recall
mar_100
fl_score*
 
Gambar 4.18 Perbandingan keberhasilan deteksi pada Neuro-Symbolic dengan Faster RCNN dan *Ground* *Truth*.
 
![Text document](asset:sha256:7b2f1a4a7fee6bc0504328ddf61f611cccc46e92b69e01f6771d9067e51e4a0d)
 
*NeSy Detection Confusion Matrix
-600
background
0
87
162
102
70
98
148
-500
open
9
650
0
0
0
0
0
short
16
0
462
0
0
0
0
-400
anull
mouse_bite
6
0
0
580
0
0
0
-300
spur-
12
0
0
0
471
0
0
-200
pinhole
3
0
0
0
0
461
0
100
spurious_copper
1
0
0
0
0
0
469
open
short
mouse_bite
spur
pinhole
spurious_copper
0
Predicted*
 
Gambar 4.19 Perbandingan kegagalan deteksi pada Neuro-Symbolic dengan Faster RCNN dan *Ground* *Truth*.
 
1. **Analisis Kegagalan Deteksi NeSy**
Berdasarkan evaluasi pada subbab sebelumnya, penurunan *Precision* dan *mAP* secara fundamental disebabkan oleh tingginya *False Positive* yang berasal dari area *background*. Kesalahan ini tidak terjadi secara acak, melainkan dipicu oleh pola-pola spesifik pada area sirkuit normal. Analisis difokuskan pada empat kelas dengan tingkat kesalahan signifikan, yaitu *short*, *spur*, *pinhole*, dan *spurious\_copper*. Untuk mengungkap akar masalah ini, komparasi visual antara prediksi NeSy, *Ground Truth*, dan pola unik pada sirkuit yang disajikan pada gambar-gambar berikut.
 
- 1. Kelas Short
Tingginya tingkat *False Positive* pada kelas *short* berbanding lurus dengan rendahnya jumlah RoI untuk kelas ini, yang hanya mencapai 1,87% pada data pelatihan dan merupakan yang paling rendah dibandingkan kelas lainnya sebagaimana dirangkum pada Tabel 4.3. Ketidakseimbangan data ini menyebabkan SODT kesulitan mempelajari variasi pola *short* secara komprehensif. Untuk memvisualisasikan dampak keterbatasan ini, Gambar 4.20 menyajikan komparasi visual deteksi pada kelas *short*.
 
*Gambar 4.20 Komparasi visual False Positive pada kelas short.*
 
Berdasarkan Gambar 4.20, SODT cenderung gagal mendeteksi *short* yang memiliki benjolan atau terlalu pendek. Sebaliknya, SODT lebih mampu mendeteksi *short* yang relatif panjang dan tidak memiliki timbulan.
 
- 1. Kelas Spur
Meskipun jumlah *Region of Interest* untuk kelas *spur* tergolong seimbang dengan persentase 2,15% pada data pelatihan seperti pada Tabel 4.3, SODT menunjukkan keterbatasan dalam menangani variasi orientasi geometris cacat ini. Pola kesalahan deteksi pada kelas *spur* diilustrasikan secara visual pada Gambar 4.21.
 
## Hasil dan Evaluasi *Explanation* pada Neuro-Symbolic
 
Evaluasi Explanation pada Neuro-Symbolic menggunakan metrik untuk mengukur *faithfulness* dan kualitas heatmap. Kemudian akan dibandingkan dengan GradCAM yang di mana telah disajikan pada Gambar 4.18.
 
![Text document](asset:sha256:f61ac9da8d86d4fcddeaeef30f4ab7012b23eb9678215b3ffa8a89d004d89c69)
 
*NeSy Per-Class Precision and Recall
1.0
0.986
0.967
0.990
0.975
0.994
0998
Precision
0.882
0.850
0.871
0.825
Recall
0.8
0.740
0.760
0.6
0.4
0.2
0.0
open
short
spur
pinhole
spurious_copper*
 
Gambar 4.18 Perbandingan *Faithfulness* dan kualitas *Heatmap* pada NeuroSymbolic dan GradCAM
 
1. **Faithfulness**
Neuro-Symbolic (SODT) mencapai Sufficiency Preservation 1,000 dan Necessity Flip Rate 0,971, menunjukkan penjelasan yang benar-benar merepresentasikan keputusan internal model. Sementara itu, Grad-CAM hanya mencapai Sufficiency 0,956 dan Necessity 0,047, mengindikasikan penjelasan yang tidak memenuhi prinsip *necessity* (hanya 4,7% kasus di mana penghapusan area penting mengubah prediksi).
 
1. **Spatial Grounding**
Grad-CAM memiliki IoU Heatmap (0,994) sedikit lebih tinggi dari SODT (0,920) karena *bleeding effect* (area heatmap melebar ke luar bounding box). Sementara itu, untuk Pointing Game, Kedua metode hampir identik (0,994 vs 0,983), membuktikan bahwa keduanya mampu mengidentifikasi wilayah cacat. Namun, SODT lebih *faithful* karena heatmapnya **terlokalisasi presisi** pada fitur yang secara eksplisit digunakan dalam keputusan.
 
1. **Waktu komputasi**
Grad-CAM membutuhkan 12,6× lebih lama dibanding Neuro-Symbolic (5045,4 ms vs 426,2 ms) karena mekanisme komputasi yang tidak efisien seperti ditunjukkan pada Gambar 4.19.
 
![Text document](asset:sha256:6c19c3da8cd49818e597a29d06b2344017ec40c4fded335a2e2bdc73bd48ae4b)
 
*Faster R-CNN Detections
NeSy (FRCNN+SODT) Detections
Ground Truth (6 defects)
#1 spurious_copper 1.0o
#5 spurious_copper 0.94
spurious_copper
#4 mouse_bite 1.00
#3 mouse_bite 0.94
mouse_bite
#2 mouse_bite 1.00
#6 short
#4 mouse_bite 0.94
#1 short
mouse_bite
short
R
R
#3 spur 1.00
#6 spur 0.94
V
V
#5 open 1.00
#2 open 0.95
open
1
1*
 
Gambar 4.19 Perbandingan *inference time* pada ketiga model.
 
Hal ini bisa terjadi karena pada Faster R-CNN, Grad-CAM harus:
 
1. Menjalankan satu kali ***backward pass* penuh** melalui seluruh arsitektur dua-tahap (RPN + RoI Head) untuk **setiap kotak deteksi (RoI)** yang dihasilkan.
2. Menghitung gradien dari skor kelas yang diprediksi hingga ke lapisan konvolusional terakhir.
Proses komputasi berulang untuk setiap RoI ini sangat tidak efisien dan membebani komputasi, sehingga tidak cocok untuk skenario *real-time* seperti inspeksi PCB.
 
# DAFTAR PUSTAKA
 
Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B., 2018. Sanity checks for saliency maps. In: *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*. Curran Associates, pp. 9505–9515.
 
Ali, A.M.M., Sahlan, S., Khamis, N., Ziyi, X., Nor Rashid, F.’A., 2024. Transparency in detecting defects of a printed circuit board: Harnessing XAI for improved quality control in electronic manufacturing industries. In: *2024 IEEE 10th International Conference on Smart Instrumentation, Measurement and Applications (ICSIMA)*. IEEE, pp. 287–292. https://doi.org/10.1109/ICSIMA62563.2024.10675544
 
Ancona, M., Ceolini, E., Öztireli, C., Gross, M., 2019. Gradient-based attribution methods. In: Samek, W., Montavon, G., Vedaldi, A., Hansen, L.K., Müller, K.-R. (Eds.), *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning*, Lecture Notes in Computer Science, vol. 11700. Springer, Cham, pp. 169–191. https://doi.org/10.1007/978-3-030-28954-6_9
 
Arun, N., Gaw, N., Singh, P., Chang, K., Aggarwal, M., Chen, B., Hoebel, K., Gupta, S., Patel, J., Gidwani, M., Adebayo, J., Li, M.D., Kalpathy-Cramer, J., 2021. Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging. *Radiology: Artificial Intelligence* 3(6), e200267. https://doi.org/10.1148/ryai.2021200267
 
Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R., Samek, W., 2015. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. *PLOS ONE* 10(7), e0130140. https://doi.org/10.1371/journal.pone.0130140
 
Barredo Arrieta, A., Díaz-Rodríguez, N., Del Ser, J., Bennetot, A., Tabik, S., Barbado, A., García, S., Gil-López, S., Molina, D., Benjamins, R., Chatila, R., Herrera, F., 2020. Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion* 58, 82–115. https://doi.org/10.1016/j.inffus.2019.12.012
 
Bodla, N., Singh, B., Chellappa, R., Davis, L.S., 2017. Soft-NMS — Improving object detection with one line of code. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. IEEE, pp. 5562–5570. https://doi.org/10.1109/ICCV.2017.593
 
Buciluǎ, C., Caruana, R., Niculescu-Mizil, A., 2006. Model compression. In: *Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD ’06)*. ACM, pp. 535–541. https://doi.org/10.1145/1150402.1150464
 
Carreira-Perpiñán, M.Á., Tavallali, P., 2018. Alternating optimization of decision trees, with application to learning sparse oblique trees. In: *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*. Curran Associates, pp. 1211–1221.
 
Chen, X., Wu, Y., He, X., Ming, W., 2023. A comprehensive review of deep learning-based PCB defect detection. *IEEE Access* 11, 139017–139038. https://doi.org/10.1109/ACCESS.2023.3339561
 
Coombs, C.F., Holden, H.T., 2016. *Printed Circuits Handbook*, 7th ed. McGraw-Hill Education, New York.
 
d'Avila Garcez, A., Lamb, L.C., 2023. Neurosymbolic AI: the 3rd wave. *Artificial Intelligence Review* 56(11), 12387–12406. https://doi.org/10.1007/s10462-023-10448-w
 
Elkan, C., 2001. The foundations of cost-sensitive learning. In: *Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 973–978.
 
Everingham, M., Van Gool, L., Williams, C.K.I., Winn, J., Zisserman, A., 2010. The Pascal Visual Object Classes (VOC) challenge. *International Journal of Computer Vision* 88(2), 303–338. https://doi.org/10.1007/s11263-009-0275-4
 
Fung, K.C., Xue, K.-W., Lai, C.-M., Lin, K.-H., Lam, K.-M., 2024. Improving PCB defect detection using selective feature attention and pixel shuffle pyramid. *Results in Engineering* 21, 101992. https://doi.org/10.1016/j.rineng.2024.101992
 
Girshick, R., 2015. Fast R-CNN. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. IEEE, pp. 1440–1448. https://doi.org/10.1109/ICCV.2015.169
 
Goodfellow, I., Bengio, Y., Courville, A., 2016. *Deep Learning*. MIT Press, Cambridge, MA.
 
Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., Pedreschi, D., 2019. A survey of methods for explaining black box models. *ACM Computing Surveys* 51(5), 1–42. https://doi.org/10.1145/3236009
 
Hada, S.S., Carreira-Perpiñán, M.Á., Zharmagambetov, A., 2024. Sparse oblique decision trees: a tool to understand and manipulate neural net features. *Data Mining and Knowledge Discovery* 38(5), 2863–2902. https://doi.org/10.1007/s10618-022-00892-7
 
Han, Z., Hong, M., Wang, D., 2017. Deep learning and applications. In: *Signal Processing and Networking for Big Data Applications*. Cambridge University Press, Cambridge, pp. 126–168. https://doi.org/10.1017/9781316408032.007
 
He, H., Garcia, E.A., 2009. Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering* 21(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239
 
He, K., Gkioxari, G., Dollár, P., Girshick, R., 2020. Mask R-CNN. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 42(2), 386–397. https://doi.org/10.1109/TPAMI.2018.2844175
 
He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. IEEE, pp. 770–778. https://doi.org/10.1109/CVPR.2016.90
 
IPC, 2015. *IPC-6012D: Qualification and Performance Specification for Rigid Printed Boards*. Association Connecting Electronics Industries.
 
IPC, 2020. *IPC-A-600K: Acceptability of Printed Boards*. Association Connecting Electronics Industries.
 
Kairgeldin, R., Carreira-Perpiñán, M.Á., 2025. Neurosymbolic models based on hybrids of convolutional neural networks and decision trees. In: *Proceedings of the 19th International Conference on Neurosymbolic Learning and Reasoning (NeSy 2025)*, Proceedings of Machine Learning Research 284, pp. 796–813.
 
Khandpur, R.S., 2005. *Printed Circuit Boards: Design, Fabrication, and Assembly*. McGraw-Hill Education, New York.
 
LeCun, Y., Bengio, Y., Hinton, G., 2015. Deep learning. *Nature* 521(7553), 436–444. https://doi.org/10.1038/nature14539
 
Li, X., Wang, W., Hu, X., Yang, J., 2019. Selective kernel networks. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. IEEE, pp. 510–519. https://doi.org/10.1109/CVPR.2019.00060
 
Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., Belongie, S., 2017. Feature pyramid networks for object detection. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. IEEE, pp. 936–944. https://doi.org/10.1109/CVPR.2017.106
 
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L., 2014. Microsoft COCO: Common objects in context. In: *Computer Vision – ECCV 2014*, Lecture Notes in Computer Science. Springer, Cham, pp. 740–755. https://doi.org/10.1007/978-3-319-10602-1_48
 
Luo, W., Li, Y., Urtasun, R., Zemel, R., 2016. Understanding the effective receptive field in deep convolutional neural networks. In: *Advances in Neural Information Processing Systems 29 (NeurIPS 2016)*. Curran Associates.
 
Manigrasso, F., Miro, F.D., Morra, L., Lamberti, F., 2021. Faster-LTN: A neuro-symbolic, end-to-end object detection architecture. In: *Artificial Neural Networks and Machine Learning – ICANN 2021*, Lecture Notes in Computer Science. Springer, Cham, pp. 40–52. https://doi.org/10.1007/978-3-030-86340-1_4
 
Minaee, S., Boykov, Y., Porikli, F., Plaza, A., Kehtarnavaz, N., Terzopoulos, D., 2022. Image segmentation using deep learning: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 44(7), 3523–3542. https://doi.org/10.1109/TPAMI.2021.3059968
 
Nauta, M., Trienes, J., Pathak, S., Nguyen, E., Peters, M., Schmitt, Y., Schlötterer, J., van Keulen, M., Seifert, C., 2023. From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable AI. *ACM Computing Surveys* 55(13s), 1–42. https://doi.org/10.1145/3583558
 
Petsiuk, V., Das, A., Saenko, K., 2018. RISE: Randomized input sampling for explanation of black-box models. In: *Proceedings of the British Machine Vision Conference (BMVC)*. BMVA Press.
 
Platt, J.C., 1999. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: Smola, A.J., Bartlett, P., Schölkopf, B., Schuurmans, D. (Eds.), *Advances in Large Margin Classifiers*. MIT Press, Cambridge, MA, pp. 61–74.
 
Prince, S.J.D., 2023. *Understanding Deep Learning*. MIT Press, Cambridge, MA.
 
Provost, F., Domingos, P., 2003. Tree induction for probability-based ranking. *Machine Learning* 52(3), 199–215. https://doi.org/10.1023/A:1024099825458
 
Ren, S., He, K., Girshick, R., Sun, J., 2017. Faster R-CNN: Towards real-time object detection with region proposal networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 39(6), 1137–1149. https://doi.org/10.1109/TPAMI.2016.2577031
 
Rudin, C., 2019. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence* 1(5), 206–215. https://doi.org/10.1038/s42256-019-0048-x
 
Saadallah, A., Büscher, J., Abdulaaty, O., Panusch, T., Deuse, J., Morik, K., 2022. Explainable predictive quality inspection using deep learning in electronics manufacturing. *Procedia CIRP* 107, 594–599. https://doi.org/10.1016/j.procir.2022.05.031
 
Samek, W., Montavon, G., Vedaldi, A., Hansen, L.K., Müller, K.-R. (Eds.), 2019. *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning*, Lecture Notes in Computer Science, vol. 11700. Springer, Cham. https://doi.org/10.1007/978-3-030-28954-6
 
Schapire, R.E., Singer, Y., 1999. Improved boosting algorithms using confidence-rated predictions. *Machine Learning* 37(3), 297–336. https://doi.org/10.1023/A:1007614523901
 
Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D., 2017. Grad-CAM: Visual explanations from deep networks via gradient-based localization. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*. IEEE, pp. 618–626. https://doi.org/10.1109/ICCV.2017.74
 
Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A.P., Bishop, R., Rueckert, D., Wang, Z., 2016. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. IEEE, pp. 1874–1883. https://doi.org/10.1109/CVPR.2016.207
 
Shrikumar, A., Greenside, P., Kundaje, A., 2017. Learning important features through propagating activation differences. In: *Proceedings of the 34th International Conference on Machine Learning (ICML)*, Proceedings of Machine Learning Research 70, pp. 3145–3153.
 
Sokolova, M., Lapalme, G., 2009. A systematic analysis of performance measures for classification tasks. *Information Processing & Management* 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002
 
Sundararajan, M., Taly, A., Yan, Q., 2017. Axiomatic attribution for deep networks. In: *Proceedings of the 34th International Conference on Machine Learning (ICML)*, Proceedings of Machine Learning Research 70, pp. 3319–3328.
 
Szeliski, R., 2022. *Computer Vision: Algorithms and Applications*, 2nd ed. Springer, Cham. https://doi.org/10.1007/978-3-030-34372-9
 
Tang, S., He, F., Huang, X., Yang, J., 2019. Online PCB defect detector on a new PCB defect dataset. arXiv preprint arXiv:1902.06197. https://doi.org/10.48550/arXiv.1902.06197
 
Tziolas, T., Papageorgiou, K., Theodosiou, T., Ioannidis, D., Dimitriou, N., Tinker, G., Papageorgiou, E., 2025. Explainable AI methods for identification of glue volume deficiencies in printed circuit boards. *Applied Sciences* 15(16), 9061. https://doi.org/10.3390/app15169061
 
Tzionis, G., Mouratidis, P., Kougka, G., Gialampoukidis, I., Vrochidis, S., Kompatsiaris, I., Vlachopoulou, M., 2026. A review of explainable AI methods and their application in manufacturing systems. *Discover Applied Sciences* 8, 52. https://doi.org/10.1007/s42452-025-07908-z
 
Wang, Y., Huang, J., Dipu, M.S.K., Zhao, H., Gao, S., Zhang, H., Lv, P., 2024. YOLO-RLC: An advanced target-detection algorithm for surface defects of printed circuit boards based on YOLOv5. *Computers, Materials & Continua* 80(3), 4973–4995. https://doi.org/10.32604/cmc.2024.055839
 
Zhang, J., Bargal, S.A., Lin, Z., Brandt, J., Shen, X., Sclaroff, S., 2018. Top-down neural attention by excitation backprop. *International Journal of Computer Vision* 126(10), 1084–1102. https://doi.org/10.1007/s11263-017-1059-x
 
Zhao, X., Wang, L., Zhang, Y., Han, X., Deveci, M., Parmar, M., 2024. A review of convolutional neural networks in computer vision. *Artificial Intelligence Review* 57(4), 99. https://doi.org/10.1007/s10462-024-10721-6
