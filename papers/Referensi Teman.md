# PENERAPAN BIDIRECTIONAL FEATURE PYRAMID NETWORK PADA NECK YOLOv12 UNTUK DETEKSI SENJATA API PADA SISTEM

# PENGAWASAN INDOOR

## SKRIPSI

## Diajukan untuk memenuhi sebagian persyaratan mendapatkan gelar Strata Satu Informatika

## Disusun oleh :

## Abraham Willem Hersubagyo

## L0122002

## PROGRAM STUDI INFORMATIKA

## FAKULTAS TEKNOLOGI INFORMASI DAN SAINS DATA

## UNIVERSITAS SEBELAS MARET

## SURAKARTA

## 2026

# HALAMAN PERSETUJUAN

# SKRIPSI

# PENERAPAN BIDIRECTIONAL FEATURE PYRAMID NETWORK PADA NECK YOLOv12 UNTUK DETEKSI SENJATA API PADA SISTEM

# PENGAWASAN INDOOR

# Disusun oleh :

# Abraham Willem Hersubagyo L0122002

# Skripsi ini telah disetujui dan dipertahankan di hadapan dewan penguji pada tanggal,

# Pembimbing I Pembimbing II

# Esti Suryani, S.Si, M.Kom. Herdito Ibnu Dewangkoro, M.Kom. NIP.197611292008122001 NIP. 199609232024061002

ii

## HALAMAN PENGESAHAN

## SKRIPSI

## PENERAPAN BIDIRECTIONAL FEATURE PYRAMID NETWORK PADA

## NECK YOLOv12 UNTUK DETEKSI SENJATA API PADA SISTEM

## PENGAWASAN INDOOR

## Disusun oleh :

## Abraham Willem Hersubagyo

## L0122002

## telah di pertahankan di hadapan Dewan Penguji pada tanggal:

## Susunan Dewan Penguji

|1. Nama Pembimbing 1 (Ketua)|(|)|
|---|---|---|
|2. Nama Pembimbing 2 (Sekretaris)|(|)|
|3. Nama Penguji Utama (Anggota)|(|)|
|4. Nama Penguji Pendamping (Anggota)|(|)|

## Disahkan Oleh

## Program Studi Informatika

## <u>Nama, Gelar</u> NIP.

iii

## HALAMAN MOTTO

*I want to be someone who makes a difference for others.*

iv

## HALAMAN PERSEMBAHAN

Skripsi ini saya persembahkan untuk kedua orang tua saya tercinta, yang tiada henti memberikan doa, kasih sayang, dan dukungan di setiap langkah perjalanan ini.

v

## KATA PENGANTAR

Puji dan syukur penulis panjatkan kepada Tuhan Yang Maha Esa atas segala berkat dan kasih karunia-Nya, sehingga penulis dapat menyelesaikan penulisan skripsi ini dengan baik. Skripsi ini tidak akan selesai tanpa adanya bantuan dan dukungan dari banyak pihak, karena itu penulis menyampaikan terima kasih yang sebesar-besarnya kepada:

1. Bapak Dr. Wiranto, M.Kom., M.Cs., selaku Dekan Fakultas Teknologi Informasi dan Sains Data Universitas Sebelas Maret.
2. Bapak Ristu Saptono, S.Si., M.T., Ph.D., selaku Ketua Program Studi Informatika Universitas Sebelas Maret atas dukungan dan kesempatan yang telah diberikan untuk menyelesaikan skripsi ini.
3. Ibu Esti Suryani, S.Si., M.Kom., selaku Pembimbing I yang telah dengan sabar memberikan bimbingan, arahan, dan petunjuk dalam menyelesaikan skripsi ini.
4. Bapak Herdito Ibnu Dewangkoro, M.Kom., selaku Pembimbing II yang telah dengan sabar memberikan bimbingan, arahan, dan petunjuk dalam menyelesaikan skripsi ini.
5. Bapak Hasan Dwi Cahyono., S.Kom., M.Kom., selaku Pembimbing Akademis yang telah memberikan arahan selama masa perkuliahan.
6. Seluruh Bapak dan Ibu dosen Program Studi Informatika Universitas Sebelas Maret atas ilmu yang telah diberikan selama perkuliahan.
7. Keluarga penulis yang senantiasa memberikan doa, semangat, dan dukungan moral selama proses penyelesaian skripsi ini.
8. Semua pihak yang tidak dapat penulis sebutkan satu persatu yang telah membantu dalam penyelesaian skripsi ini.
Penulis menyadari bahwa skripsi ini masih jauh dari sempurna. Oleh karena itu, penulis mengharapkan kritik dan saran yang membangun dari semua pihak. Kiranya Tuhan Yang Maha Esa membalas kebaikan semua pihak yang telah membantu. Semoga skripsi ini dapat memberikan manfaat bagi pembaca dan perkembangan ilmu pengetahuan.

vi

## PENERAPAN BIDIRECTIONAL FEATURE PYRAMID NETWORK PADA

## NECK YOLOv12 UNTUK DETEKSI SENJATA API PADA SISTEM

## PENGAWASAN INDOOR

## ABRAHAM WILLEM HERSUBAGYO

Program Studi Informatika Fakultas Teknologi Informasi dan Sains Data Universitas Sebelas Maret

## ABSTRAK

Keamanan di lingkungan publik indoor rentan terhadap ancaman senjata api, namun pengawasan manual melalui CCTV memiliki keterbatasan dalam kecepatan respons dan deteksi proaktif. Model deteksi objek berbasis deep learning seperti YOLOv12 menawarkan solusi real-time, tetapi masih kesulitan mendeteksi senjata yang berukuran kecil dan terhalang sebagian (occluded) dalam kondisi lingkungan yang padat. Penelitian ini mengintegrasikan modul Bidirectional Feature Pyramid Network (BiFPN) ke dalam komponen Neck YOLOv12. BiFPN memperkenalkan aliran informasi dua arah antar level fitur dan mekanisme pembobotan yang dapat dipelajari (learnable weighted fusion), sehingga model lebih efektif menggabungkan fitur multi-skala untuk mendeteksi objek kecil dan teroklusi. Model dilatih pada FDIE Dataset (Firearm Detection at Indoor Environments) dengan pembagian training (70%), validation (20%), dan testing (10%). Performa YOLOv12-BiFPN dibandingkan dengan YOLOv12 standar menggunakan metrik Precision, Recall, mean Average Precision (mAP), dan Frames Per Second (FPS).

***Kata kunci***: deteksi senjata api, YOLOv12, BiFPN, Bidirectional Feature Pyramid Network, FDIE dataset, sistem pengawasan indoor

vii

## APPLYING BIDIRECTIONAL FEATURE PYRAMID NETWORK TO THE

## YOLOV12 NECK FOR FIREARM DETECTION IN INDOOR

## SURVEILLANCE SYSTEMS

## ABRAHAM WILLEM HERSUBAGYO

Department of Informatics Faculty of Information Technology and Data Science Universitas Sebelas Maret

## ABSTRACT

*Indoor public spaces are vulnerable to firearm threats, yet manual CCTV-based* *surveillance is limited in response speed and proactive detection. Deep learning-based object detection models such as YOLOv12 offer real-time solutions, but* *struggle to detect small and partially occluded firearms in cluttered indoor* *environments. This study integrates a Bidirectional Feature Pyramid Network* *(BiFPN) module into the Neck component of YOLOv12. BiFPN introduces* *bidirectional cross-scale information flow and a learnable weighted fusion* *mechanism, enabling more effective multi-scale feature aggregation for detecting* *small and occluded objects. The model is trained on the FDIE Dataset (Firearm* *Detection at Indoor Environments), split into training (70%), validation (20%),* *and testing (10%) sets. The performance of YOLOv12-BiFPN is compared against* *standard YOLOv12 using Precision, Recall, mean Average Precision (mAP), and* *Frames Per Second (FPS).*

***Keywords***: *firearm detection, YOLOv12, BiFPN, Bidirectional Feature Pyramid* *Network, FDIE dataset, indoor surveillance*

viii

## DAFTAR ISI

HALAMAN PERSETUJUAN.................................................................................ii
HALAMAN PENGESAHAN.................................................................................iii
HALAMAN MOTTO............................................................................................. iv
HALAMAN PERSEMBAHAN...............................................................................v
KATA PENGANTAR.............................................................................................. vi
ABSTRAK............................................................................................................. vii
ABSTRACT..........................................................................................................viii
DAFTAR ISI........................................................................................................... ix
DAFTAR GAMBAR...............................................................................................xi
DAFTAR TABEL...................................................................................................xii
BAB I..................................................................................................................... 13
PENDAHULUAN..................................................................................................13

1.1 Latar Belakang.............................................................................................13
1.2 Rumusan Masalah........................................................................................16
1.3 Batasan Masalah..........................................................................................16
1.4 Tujuan Penelitian......................................................................................... 17
1.5 Manfaat Penelitian....................................................................................... 17
BAB II.................................................................................................................... 18
TINJAUAN PUSTAKA.........................................................................................18

2.1 Dasar Teori...................................................................................................18
2.1.1 Kecerdasan Buatan (Artificial Intelligence)........................................18
2.1.2 Deep Learning......................................................................................18
2.1.3 Convolutional Neural Network............................................................ 18
2.1.4 Object Detection..................................................................................20
2.1.5 You Only Look Once (YOLO).............................................................20
2.1.6 Arsitektur YOLOv12............................................................................21
2.1.7 Bidirectional Feature Pyramid Network (BiFPN)...............................23
2.1.8 Evaluation Metrics...............................................................................24
2.1.9 Fungsi Loss pada YOLOv12................................................................25
2.2 Penelitian Terkait......................................................................................... 28
BAB III...................................................................................................................32
METODOLOGI PENELITIAN.............................................................................32

3.1 Diagram Alir Proses Penelitian....................................................................32
3.2 Pengumpulan dan Pemrosesan Dataset.......................................................32
3.3 Perancangan Model dan Integrasi BiFPN...................................................35
3.3.1 Analisis Arsitektur YOLOv12 Standar.................................................35
3.3.2 Implementasi Modul BiFPN Kustom..................................................35
3.3.3 Integrasi Modul BiFPN ke dalam Arsitektur Utama............................36
3.3.4 Penyesuaian Komponen Head.............................................................37
3.4 Pelatihan dan Pengujian YOLOv12 Standar dan YOLOv12-BiFPN..........37
3.5 Analisis dan Evaluasi Hasil.........................................................................37
ix

BAB IV...................................................................................................................39
HASIL DAN PEMBAHASAN.............................................................................. 39

4.1 Pengumpulan Dataset.................................................................................. 39
4.2 Hasil Pelatihan Model.................................................................................40
4.3 Confusion Matrix.........................................................................................41
4.4 Perbandingan Performa Deteksi.................................................................. 43
4.5 Hasil Deteksi Visual....................................................................................45
4.6 Pengaruh BiFPN terhadap Performa Deteksi..............................................48
4.7 Trade-Off Precision-Recall dan Perilaku Deteksi........................................49
4.8 Analisis Trade-off Akurasi-Kecepatan.........................................................50
4.9 Perbandingan dengan Baseline dan Penelitian Sebelumnya.......................51
BAB V.................................................................................................................... 53
KESIMPULAN...................................................................................................... 53

5.1 Kesimpulan.................................................................................................. 53
5.2 Saran............................................................................................................ 53
DAFTAR PUSTAKA............................................................................................. 55

x

## DAFTAR GAMBAR

Gambar 2.1. Arsitektur Convolutional Neural Network (Diwan, Anirudh and
Tembhurne, 2023)..................................................................................................19
Gambar 2.2. Prinsip Kerja YOLO (Sapkota et al., 2025).......................................20
Gambar 2.3. Arsitektur YOLOv12 (Chandrashekhar *et al.*, 2025).........................22
Gambar 3.1. Diagram Alir Proses Penelitian.........................................................32
Gambar 3.2. Dataset (dipotong per frame).............................................................33
Gambar 3.3. Bounding Box Notations Berformat YOLO......................................33
Gambar 3.4. Ground Truth Dataset FDIE.............................................................. 34
Gambar 3.5. Arsitektur YOLOv12-BiFPN.............................................................36
Gambar 4.1. Train Loss YOLOv12 dan YOLOv12-BiFPN...................................40
Gambar 4.2. Val Loss YOLOv12 dan YOLOv12-BiFPN......................................40
Gambar 4.3. Confusion Matrix YOLOv12m-BiFPN.............................................41
Gambar 4.4. Confusion Matrix YOLOv12 dan YOLOv12-BiFPN.......................42
Gambar 4.5. Perbandingan Performa YOLOv8 (baseline), YOLOv12 dan
YOLOv12-BiFPN.................................................................................................. 45

xi

## DAFTAR TABEL

Tabel 2.1. Penelitian Terkait................................................................................... 28
Tabel 2: Distribusi Gambar dan Anotasi per Subset...............................................39
Tabel 4.3. Perbandingan Performa YOLOv8 (baseline), YOLOv12 standard dan
YOLOv12-BiFPN.................................................................................................. 44
Tabel 4.4. Perbandingan Deteksi Visual YOLOv12 dan YOLOv12-BiFPN..........46

xii

## BAB I

## PENDAHULUAN

## 1.1 Latar Belakang

Keamanan di ruang publik dalam ruangan (*indoor environments*) seperti pusat perbelanjaan, sekolah, bandara, dan perkantoran merupakan prioritas utama dalam menjaga ketertiban dan keselamatan masyarakat (Sujatha and Janani,

2024). Ruang-ruang ini memiliki kerentanan tinggi terhadap tindak kriminal, terutama yang melibatkan senjata api (K U and M, 2023). Insiden kekerasan di lokasi-lokasi ini dapat terjadi dengan cepat dan menimbulkan dampak fatal (Bhatti *et al.*, 2021). Oleh karena itu, diperlukan sistem pengawasan proaktif yang mampu mendeteksi potensi ancaman sebelum insiden terjadi. Keberhasilan pencegahan tindak kriminal di lingkungan indoor sangat membutuhkan kemampuan deteksi otomatis terhadap barang berbahaya, khususnya senjata (Shalini *et al.*, 2025). Deteksi cepat terhadap individu yang membawa senjata di area terlarang memungkinkan petugas keamanan untuk melakukan intervensi dan mitigasi risiko secara efektif (Ahmed and Echi, 2021). Tantangan utamanya adalah mengidentifikasi objek-objek ini secara akurat dan real-time dalam lingkungan yang dinamis dan sering kali padat (Tabassum *et al.*,
2024). Dalam praktik konvensional, tenaga keamanan masih mengandalkan observasi langsung melalui monitor *Closed-Circuit Television* (CCTV) atau pemeriksaan di pintu masuk (Kim *et al.*, 2024). Namun, keterbatasan manusia dalam mengawasi banyak monitor secara simultan dan mempertahankan fokus dalam waktu lama membuat metode ini kurang efektif (Al-Jawahry *et al.*, 2023). Selain itu, tidak semua tempat umum memiliki *metal detector* sehingga tidak mampu mendeteksi senjata logam (Ahmed and Echi, 2021).

Studi terbaru juga mendokumentasikan bahwa kombinasi pemantauan video dan algoritma pengenalan aksi abnormal dapat secara signifikan mempercepat waktu respons petugas keamanan, meskipun keandalannya sangat bergantung pada kualitas detektor objek yang mendasarinya (Khanam and R,

2025). Sebuah survei komprehensif terhadap sistem deteksi senjata berbasis AI dari tahun 2016 hingga 2025 (Murugan *et al.*, 2025) menyimpulkan bahwa model-model YOLO (You Only Look Once) mendominasi literatur deteksi senjata real-time karena keseimbangan kecepatan-akurasinya, namun tantangan masih tersisa untuk skenario indoor dengan kamera tunggal dan iluminasi yang tidak konsisten. Di sisi lain, perkembangan teknologi kecerdasan buatan (Artificial Intelligence/AI) dan visi komputer (Computer Vision) (Wu *et al.*, 2025) telah membuka peluang baru dalam bidang keamanan publik (D *et al.*, 2023). Beberapa penelitian sebelumnya menunjukkan bahwa metode deteksi berbasis *deep learning*, salah satunya *You Only Look Once* (YOLO), mampu memberikan solusi real-time untuk mengidentifikasi senjata di ruang publik (Diwan, Anirudh and Tembhurne, 2023). Sistem berbasis YOLO terbukti mampu mengenali senjata api dengan akurasi tinggi pada berbagai kondisi pencahayaan dan sudut pandang, sehingga sangat relevan untuk diterapkan dalam konteks keamanan publik. Penelitian dengan YOLOv4 menghasilkan *Average Precision* sebesar 96,91% dan *Mean Average Precision* sebesar 91,73% dengan IoU Threshold 0,35 (Bhatti *et al.*, 2021). Sementara itu, sistem berbasis YOLOv8 menghasilkan *Average Precision* sebesar 92,5% dan *Mean Average Precision* sebesar 88,2% dengan IoU Threshold 53,6% (P and V, 2025). Varian terbaru, YOLOv12, menghadirkan terobosan dalam deteksi objek real-time dengan meningkatkan kecepatan inferensi sekaligus akurasi deteksi, menawarkan *speed-accuracy trade-off* yang lebih superior dibandingkan YOLOv11, meskipun nilai mAP-nya sedikit lebih rendah (Chandrashekhar *et al.*, 2025).

Dalam konteks spesifik pengawasan CCTV indoor, terdapat penelitian yang mengusulkan dataset FDIE (Firearm Detection at Indoor Environments), sebuah dataset indoor dunia nyata dengan 2213 frame yang dianotasi dalam format YOLO, beserta eksperimen pertama deteksi senjata api pada dataset ini menggunakan varian YOLOv8, di mana YOLOv8s mencapai hasil terbaik dengan mAP@50 = 0,455 (Da Silva and Pereira, 2024).

Namun, deteksi senjata di lingkungan indoor menghadirkan serangkaian tantangan yang unik. Senjata seringkali merupakan objek berukuran kecil, kerap terhalang sebagian (partially occluded) oleh tubuh atau objek lain, dan sering muncul dalam latar visual yang kompleks dan padat (cluttered), yang semuanya dapat menurunkan akurasi deteksi (Al-Jawahry *et al.*, 2023). Tantangan ini diperparah oleh kondisi pencahayaan indoor yang rendah pada lingkungan pengawasan (Pravesh and Sahana, 2025). Dalam penelitian (Pravesh and Sahana,

2025), YOLOv11 digunakan untuk deteksi senjata di bawah kondisi pencahayaan rendah, menunjukkan bahwa kontras gambar yang rendah dapat menurunkan recall model deteksi lebih dari 65 poin persentase (dari 90,00% pada pencahayaan normal menjadi 24,39% pada kondisi gelap). Selain itu, keterbatasan komputasi pada perangkat CCTV dan edge device semakin menambah urgensi akan arsitektur yang efisien sekaligus akurat (Berardini *et al.*, 2023). Untuk mengatasi masalah deteksi objek kecil dan terhalang ini, beberapa peneliti telah mengeksplorasi arsitektur BiFPN sebagai solusi (Bai and Song, 2025). Sebagai contoh, salah satu penelitian menunjukkan bahwa implementasi modul Bidirectional Feature Pyramid Network (BiFPN) dapat meningkatkan kemampuan model dalam menggabungkan fitur dari berbagai skala, sehingga lebih efektif dalam mengidentifikasi objek kecil dan terhalang sebagian (Zhang and Du, 2023). Terdapat penelitian lain yang mengimplementasikan modul BiFPN pada Neck YOLOv11 sehingga dapat mendeteksi objek-objek yang kecil dengan lebih akurat (J. Zhu *et al.*, 2025). Penelitian BiFPN-YOLO berbasis YOLOv5 (Doherty et al.,
2025) juga menunjukkan peningkatan deteksi yang konsisten pada beberapa

benchmark publik. Bukti serupa juga disajikan dalam domain lain: SBEW-YOLOv8 (Yin *et al.*, 2025) yang memodifikasi neck dengan varian BiFPN untuk skenario autonomous driving melaporkan peningkatan mAP yang konsisten pada objek kecil, dan YOLO-WildASM (Y. Zhu *et al.*, 2025) yang menggabungkan BiFPN dengan multi-head attention mencapai mAP@50 sebesar 94,1% pada deteksi satwa liar yang dilindungi.

Berdasarkan latar belakang tersebut, penelitian ini berfokus pada penerapan YOLOv12 yang diintegrasikan dengan modul BiFPN pada komponen *neck* untuk meningkatkan kemampuan deteksi senjata api kecil dan teroklusi (Zhang and Du, 2023; Chandrashekhar *et al.*, 2025). Tujuan utamanya adalah merancang, mengimplementasikan, dan mengevaluasi sistem deteksi senjata api di lingkungan indoor, serta membandingkan performa YOLOv12-BiFPN dengan model standar YOLOv12.

## 1.2 Rumusan Masalah

Bagaimana merancang dan mengimplementasikan sistem deteksi berbasis model YOLOv12 yang dimodifikasi dengan Bidirectional Feature Pyramid Network (BiFPN) untuk mengidentifikasi senjata api yang terlihat kecil dan teroklusi (occluded) dalam video tindak kriminal, serta bagaimana mengevaluasi perbandingan performa YOLOv12 standar dengan model YOLOv12-BiFPN menggunakan metrik evaluasi standar deteksi objek seperti Precision, Recall, mean Average Precision (mAP), dan efisiensi komputasi melalui kecepatan deteksi (Frames Per Second/FPS)?

## 1.3 Batasan Masalah

**1.** Penelitian ini menggunakan dataset sekunder yang sudah ada, yaitu FDIE Dataset (Firearm Detection at Indoor Environments) Dataset dari platform Kaggle. Dengan demikian, lingkup data terbatas pada gambar dan skenario yang terdapat dalam dataset tersebut, dan penelitian ini tidak

melakukan pengumpulan data primer (seperti perekaman video atau pengambilan gambar baru).

**2.** Penelitian ini berfokus pada pengembangan dan pengujian model dalam lingkungan komputasi yang terkontrol. Implementasi sistem secara langsung (live deployment) di lapangan pada infrastruktur keamanan yang sesungguhnya berada di luar cakupan penelitian ini.
## 1.4 Tujuan Penelitian

Mengimplementasikan modifikasi Bidirectional Feature Pyramid Network (BiFPN) pada arsitektur YOLOv12 untuk meningkatkan kemampuan deteksi senjata api yang terlihat kecil dan teroklusi dalam skenario video tindak kriminal pada lingkungan publik indoor dan menganalisis performa model menggunakan metrik evaluasi standar deteksi objek seperti Precision, Recall, dan mean Average Precision (mAP), serta mengukur efisiensi komputasi model melalui kecepatan deteksi (Frames Per Second/FPS) untuk deteksi video secara real-time.

## 1.5 Manfaat Penelitian

Memberikan kontribusi keilmuan di bidang Computer Vision dengan menyajikan bukti empiris mengenai keunggulan integrasi BiFPN pada arsitektur YOLOv12 untuk mendeteksi objek kecil dan teroklusi, khususnya senjata. Evaluasi yang hanya dilakukan dalam lingkungan komputasi terkontrol ini dapat menjadi pijakan bagi penelitian selanjutnya untuk menguji implementasi model secara langsung (live deployment) pada infrastruktur keamanan yang sesungguhnya.

## BAB II

## TINJAUAN PUSTAKA

### 2.1 Dasar Teori 2.1.1 Kecerdasan Buatan (Artificial Intelligence)

Kecerdasan Buatan (AI) adalah cabang ilmu komputer yang berfokus pada pembuatan mesin cerdas yang mampu melakukan tugas-tugas yang biasanya memerlukan kecerdasan manusia (Wu *et al.*, 2025). AI mencakup berbagai sub-bidang, salah satunya adalah Machine Learning (Pembelajaran Mesin), di mana sistem komputer belajar dari data untuk membuat prediksi atau keputusan tanpa diprogram secara eksplisit untuk setiap tugas (Diwan, Anirudh and Tembhurne,

2023).
#### 2.1.2 Deep Learning

Deep Learning merupakan sub-bidang dari Machine Learning yang menggunakan Jaringan Saraf Tiruan (Artificial Neural Networks) dengan banyak lapisan (deep architectures) untuk mempelajari representasi data (Diwan, Anirudh and Tembhurne, 2023). Keunggulan utama deep learning adalah kemampuannya untuk secara otomatis dan hierarkis mengekstraksi fitur dari data mentah. Misalnya, dalam pengenalan gambar, lapisan pertama belajar mendeteksi tepi, lapisan berikutnya mendeteksi bentuk, dan lapisan yang lebih dalam lagi mendeteksi objek utuh.

#### 2.1.3 Convolutional Neural Network

Convolutional Neural Network (CNN atau ConvNet) adalah jenis arsitektur deep learning yang dirancang khusus untuk memproses data grid, seperti gambar. CNN sangat efektif dalam tugas-tugas visi komputer karena kemampuannya mengenali pola spasial dalam gambar (Diwan, Anirudh and Tembhurne, 2023).

Gambar 2.1. Arsitektur Convolutional Neural Network (Diwan, Anirudh and

Tembhurne, 2023)*.*

Pada Gambar 2.1, Arsitektur CNN terdiri dari tiga jenis lapisan utama:

**1.** Lapisan Konvolusi (Convolutional Layer): Lapisan ini adalah inti dari CNN. Lapisan ini menerapkan serangkaian filter (atau kernel) yang dapat dipelajari ke gambar input. Setiap filter dirancang untuk mendeteksi fitur tertentu, seperti tepi, warna, atau tekstur (Diwan, Anirudh and Tembhurne,
2023).
**2.** Lapisan Pooling (Pooling Layer): Lapisan ini berfungsi untuk mengurangi dimensi spasial (downsampling) dari representasi fitur, sehingga mengurangi jumlah parameter dan komputasi dalam jaringan. Jenis yang paling umum adalah Max Pooling, yang mengambil nilai maksimum dari setiap jendela filter (Diwan, Anirudh and Tembhurne, 2023).
**3.** Lapisan Terhubung Penuh (Fully Connected Layer): Setelah fitur diekstraksi dan diperkecil oleh lapisan konvolusi dan pooling, lapisan ini berfungsi untuk melakukan tugas klasifikasi atau regresi berdasarkan fitur tingkat tinggi yang telah dipelajari (Diwan, Anirudh and Tembhurne,
2023).

#### 2.1.4 Object Detection

Deteksi objek adalah tugas dalam visi komputer yang tidak hanya mengklasifikasikan objek dalam sebuah gambar, tetapi juga menentukan lokasi objek tersebut dengan menggambar sebuah kotak pembatas (bounding box) di sekelilingnya. Ini merupakan tugas yang lebih kompleks daripada klasifikasi gambar karena memerlukan kemampuan lokalisasi dan klasifikasi secara bersamaan (Diwan, Anirudh and Tembhurne, 2023).

#### 2.1.5 You Only Look Once (YOLO)

You Only Look Once (YOLO) adalah keluarga algoritma deteksi objek yang merevolusi bidang Object Detection karena kecepatan dan efisiensinya. Berbeda dengan detektor dua tahap (seperti R-CNN), YOLO membingkai deteksi objek sebagai satu masalah regresi tunggal (Sapkota *et al.*, 2025).

Gambar 2.2. Prinsip Kerja YOLO (Sapkota et al., 2025)

Pada Gambar 2.2 dapat dilihat bahwa prinsip kerja YOLO adalah sebagai berikut:

||1. 2. 3.|2025). pusatnya jatuh di dalam sel tersebut (Sapkota objek (Sapkota et al., 2025).|Gambar input dibagi menjadi sebuah grid berukuran S×S (Sapkota Setiap sel dalam grid bertanggung jawab untuk mendeteksi objek yang et al. Setiap sel memprediksi B bounding box dan skor kepercayaan (confidence score) untuk kotak-kotak tersebut, serta probabilitas kelas C untuk setiap|, 2025).||21 et al.,|
|---|---|---|---|---|---|---|
|||Bounding box direpresentasikan oleh 5 nilai: (||x, y, a, b, confidence||) di|
|mana|x, y|dan tinggi kotak relatif terhadap seluruh gambar, dan|adalah koordinat pusat kotak relatif terhadap sel grid, confidence|a, b|mencerminkan|adalah lebar|
|2.1.6||prediksinya (Sapkota et al., 2025). Arsitektur YOLOv12|seberapa yakin model bahwa kotak tersebut berisi objek dan seberapa akurat YOLOv12 merupakan salah satu varian terbaru dari YOLO yang dirancang khusus untuk meningkatkan kemampuan deteksi objek real-time. YOLOv12 menghadirkan terobosan signifikan dengan mengintegrasikan mekanisme attention pada backbone dan neck, sehingga mencapai keseimbangan kecepatan- akurasi yang lebih baik ibandingkan pendahulunya (Chandrashekhar||et al.|, 2025).|

Gambar 2.3. Arsitektur YOLOv12 *(*Chandrashekhar *et al., 2025)*

Pada Gambar 2.3 ada 3 layer utama pada YOLOv12 yaitu:

**1.** Backbone: Jaringan CNN yang dalam (misalnya, CSPDarknet) yang berfungsi sebagai ekstraktor fitur. Jaringan ini mengambil gambar input dan menghasilkan representasi fitur pada berbagai skala (Chandrashekhar *et al.*, 2025).
**2.** Neck: Bagian ini berfungsi untuk menggabungkan dan mencampur fitur dari berbagai skala yang dihasilkan oleh backbone. Tujuannya adalah untuk menciptakan peta fitur (feature maps) yang kaya akan informasi semantik (untuk objek besar) dan informasi spasial (untuk objek kecil). Di sinilah FPN, PANet, dan inovasi seperti BiFPN berperan (Chandrashekhar *et al.*, 2025).
**3.** Head (Prediction Layer): Bagian ini mengambil fitur dari neck dan melakukan prediksi akhir berupa bounding box, skor kepercayaan, dan probabilitas kelas (Chandrashekhar *et al.*, 2025).

|2.1.7|1. 2.|PANet (Zhang and Du, 2023). lebih efektif dan kaya (J. Zhu selama proses penggabungan.|Bidirectional Feature Pyramid Network (BiFPN) dipelajari (learnable weights) (Gao|mekanisme penggabungan fitur multi-skala yang efektif. BiFPN adalah arsitektur neck yang merupakan penyempurnaan dari arsitektur sebelumnya seperti FPN dan Keunggulan utama BiFPN terletak pada dua inovasi kuncinya: et al.|Untuk mengatasi tantangan deteksi objek kecil dan teroklusi, diperlukan Bidirectional Cross-Scale Connections: Tidak seperti FPN (yang alirannya hanya dari atas ke bawah) atau PANet (atas ke bawah lalu bawah ke atas sekali), BiFPN memungkinkan aliran informasi mengalir bolak-balik antar level fitur secara berulang. Ini menciptakan koneksi tambahan yang memungkinkan model menggabungkan fitur-fitur dari berbagai skala (baik fitur detail dari level rendah maupun fitur konteks dari level tinggi) secara, 2025). Weighted Feature Fusion: Tidak seperti metode fusi konvensional yang hanya menjumlahkan fitur, BiFPN memperkenalkan bobot yang dapat et al. model untuk secara dinamis memprioritaskan fitur yang lebih informatif|, 2024). Hal ini memungkinkan|23|
|---|---|---|---|---|---|---|---|
||||O =|∑ a ∑ b|w ⋅ I a a w + ϵ b Logika fusi ini diimplementasikan menggunakan rumus pada Persamaan||(2.1)|
|dan|2.1 dengan w serta a|O w b|merepresentasikan fitur keluaran (output),||adalah bobot yang dapat dipelajari, dan ε konstanta kecil untuk|I a|adalah peta fitur input,|

2024). menjaga stabilitas numerik (menghindari pembagian dengan nol) (Gao et al.,

||2.1.8 1.||Evaluation Metrics Metrik evaluasi yang digunakan antara lain: keseluruhan (Sapkota|et al.|, 2025),|mAP (mean Average Precision): untuk mengukur akurasi deteksi secara|||24|
|---|---|---|---|---|---|---|---|---|---|
||2.|AP|adalah kelas objek tertentu (Sapkota|mAP Average Precision AP|N 1 = ∑ N i =1 et al. 1 (= p r ∫ 0|AP i Rumus menghitung mAP dapat dilihat pada Persamaan 2.2 yang di mana untuk satu kelas AP (Average Precision): mengukur seberapa baik model mendeteksi satu, 2025). ) dr|i, dan|N|(2.2) adalah total kelas.|
||3. 4.|recall. dengan (Sapkota|TP adalah False Negative. et al.|Persamaan 2.3 menunjukkan rumus menghitung AP. precision sebagai fungsi dari recall, dan jumlah deteksi dan jumlah ground truth (Sapkota Precision Recall sebagai True Positive,, 2025).|= TP = TP|perubahan recall (infinitesimal), menandakan integral terhadap sumbu Precision dan Recall: untuk mengetahui rasio deteksi yang benar terhadap TP + FP TP + FN Persamaan 2.4 dan 2.5 merupakan rumus menghitung Precision dan Recall FP FPS (Frame Per Second): untuk mengukur kecepatan proses inferensi|dr|p (r adalah elemen kecil dari et al., 2025). sebagai False Positive, dan|(2.3) ) adalah fungsi (2.4) (2.5) FN|

<u>F</u> (2.6) *FPS*= *T*

Untuk menghitung FPS, dapat menggunakan Persamaan 2.6 dengan *F* sebagai jumlah frame yang diproses dan T sebagai total waktu inferensi (dalam detik).

**5.** Confusion Matrix: untuk melihat distribusi klasifikasi senjata yang benar dan salah.
#### 2.1.9 Fungsi Loss pada YOLOv12

Pada proses pelatihan, YOLOv12 menggunakan fungsi loss yang terdiri dari tiga komponen utama untuk mengoptimalkan parameter model:

**1.** Box Loss (CIoU Loss): Mengukur kesalahan lokalisasi bounding box. YOLOv12 menggunakan Complete Intersection over Union (CIoU) sebagai metrik regresi bounding box, yang tidak hanya mempertimbangkan area tumpang tindih (IoU) tetapi juga jarak titik pusat dan rasio aspek antara bounding box prediksi dan ground truth (Zheng *et* *al.*, 2020).
2 <u>ρ (k, k¿)</u> *Lbox*=1−*IoU*+ +*α*⋅*ν* (2.7) 2 *l* Persamaan 2.7 merupakan rumus Box Loss pada YOLOv12 dengan:

**a)** *IoU*: Intersection over Union 2
**b)** *ρ* (*k, k*¿): Jarak euclidean antara pusat predicted box dan ground truth 2
**c)** *l* : Diagonal kuadrat dari smallest enclosing box yang melingkupi kedua box
**d)** *ν*: Jumlah total anchor point positif (yang mengandung objek)
**e)** *α*: Koefisien trade-off yang menyeimbangan penalti aspect-ratio

|2. (||||Classification Loss (Binary Cross-Entropy): Komponen ini mengukur kesalahan prediksi kelas objek. Karena tugas deteksi senjata api merupakan klasifikasi biner (senjata vs. bukan senjata), YOLOv12 menggunakan Binary Cross-Entropy (BCE) Loss yang menghitung selisih antara probabilitas kelas yang diprediksi dengan label ground truth [https://github.com/ultralytics/ultralytics|](https://github.com/ultralytics/ultralytics|), Ultralytics, 2024).|||26|
|---|---|---|---|---|---|---|---|---|
|dengan: a)|u i|||−1 ^ L = u ⋅log (p)+(1− u ∑ [cls i i i N i pos Persamaan 2.8 merupakan rumus Classification Loss pada YOLOv12 : Ground truth label anchor point ke-i (0 = background, 1 = object)|^)⋅log (1− p)] i|||(2.8)|
|b)|^ p i|||: Probabilitas prediksi setelah sigmoid:|^ p = sigma (i|x)= i|1 − x i||
||||||||1+ e||
|c)|x i|||: Raw logit output dari classification head|||||
|d) 3. (Li YOLOv12 dengan: a)|N y|pos et al.|L (dfl|: Jumlah total anchor point positif (yang mengandung objek) Distribution Focal Loss (DFL): Komponen ini secara spesifik mengukur kualitas distribusi prediksi koordinat bounding box. Tidak seperti loss regresi tradisional yang hanya memprediksi nilai tunggal, DFL memodelkan distribusi probabilitas di sekitar tepi bounding box, memungkinkan model untuk menghasilkan prediksi lokalisasi yang lebih fleksibel dan akurat, terutama pada kasus tepi bounding box yang ambigu, 2020). S, S)=− (y − y) log (S)+(y [i i +1 i +1 i Persamaan 2.9 merupakan rumus dari Distribution Focal Loss pada : koordinat target sesungguhnya (misal: jarak dari anchor ke tepi kiri|− y) log (S i i|)] +1||(2.9)|
|||box)|||||||
|b)|yi =[||y||||]: Nilai bin diskret di kiri (floor atau pembulatan ke bawah dari||
||y)||||||||
|c)|y i +1 dari y)|=|y i||||+1: Nilai bin diskret di kanan (ceiling atau pembulatan ke atas||
|d)|S = i|P|(y|i|||): Probabilitas prediksi untuk bin kiri||
|e)|S i +1|=|P|(y|i|+1|): Probabilitas prediksi untuk bin kanan||
|f)|(y i|+1|− y||||): Bobot interpolasi untuk bin kiri||
|g)|(y −|y|i||||): Bobot interpolasi untuk bin kanan Total loss selama pelatihan merupakan penjumlahan terbobot dari ketiga komponen tersebut, di mana penurunan nilai loss secara keseluruhan mengindikasikan bahwa model semakin baik dalam melakukan lokalisasi, klasifikasi, dan estimasi distribusi bounding box secara simultan. L = λ ⋅ L + λ ⋅ L + λ ⋅ L total box box cls cls dfl dfl Persamaan 2.10 merupakan rumus Total Loss YOLOv12 dengan:|(2.10)|
|a)|L box||||||: CIoU Loss (kesalahan lokalisasi bounding box)||
|b)|L cls||||||: Binary Cross-Entropy Loss (kesalahan klasifikasi)||
|c)|L dfl||||||: Distribution Focal Loss (kesalahan distribusi koordinat)||
|d)|λ box (|,|λ cls|,|λ|dfl|: Bobot hyperparameter masing-masing komponen loss [https://github.com/ultralytics/ultralytics|](https://github.com/ultralytics/ultralytics|), Ultralytics, 2024)|

### 2.2 Penelitian Terkait

Tabel 2.1 menyajikan ringkasan penelitian-penelitian terkait yang menjadi landasan bagi studi ini.

## Tabel 2.1. Penelitian Terkait

|NO|Artikel|Masalah|Tujuan|Metode|Data dan Hasil|Keterkaitan|
|---|---|---|---|---|---|---|
|1.|Proactive Headcount Kesulitan untuk and Suspicious Activity mengantisipasi dan Detection using mengendalikan insiden desak- YOLOv8 (D et al., 2023)|desakan ( crowd-smashing ) di ruang publik yang padat secara manual|Menganalisis rekaman CCTV secara real- time dengan YOLOv8 guna memantau kepadatan kerumunan sekaligus mendeteksi potensi ancaman atau kelainan secara proaktif.|YOLOv8|Hasil eksperimen menunjukkan bahwa YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, dan YOLOv8x berturut- turut mendapatkan mAP sebesar 37.3, 44.9, 50.2, 52.9, 53.9|Penggunaan metode YOLO (meskipun berbeda versi namun masih relevan) untuk deteksi objek yang sama (senjata api)|
|2.|Weapon Detection in Tidak efektifnya pengawasan Real-Time CCTV keamanan manual dan Videos Using Deep kesulitan teknis model deteksi deteksi senjata Learning (Bhatti et al., 2021)|standar dalam mengidentifikasi senjata berukuran kecil dan terhalang (oklusi) di ruang publik dalam|Merancang dan mengevaluasi sistem otomatis yang lebih akurat, khususnya untuk target yang sulit dideteksi tersebut.|VGG16, Inception- V3, Inception- ResnetV2, SSDMobileNetV1, Faster-RCNN Inception- ResnetV2|Nilai akurasi dan F1- score untuk masing- masing model adalah sebagai berikut: VGG (78.20%; 81.69%), Inceptionv3 (85.20%; 84.36%),|Penggunaan metode YOLOv3 dan YOLOv4 (meskipun berbeda versi namun masih relevan) untuk|

|NO|Artikel|Masalah|Tujuan|Metode|Data dan Hasil|Keterkaitan|
|---|---|---|---|---|---|---|
|||ruangan|Untuk mencapainya|(FRIRv2), YOLOv3, and YOLOv4|InceptionResNetv2 (92.20%; 85.74%), SSDMobileNet (79%; 59%), FasterRCNN (96%; 87%), Yolov3 (94%; 86%), dan Yolov4 (99%; 91%).|deteksi objek yang sama (senjata api)|
|3.|YOLO advances to its Kurangnya tinjauan genesis: a decadal and komprehensif yang mencakup komprehensif satu comprehensive review seluruh evolusi YOLO dari of the You Only Look awal hingga versi terbaru Once (YOLO) series (Sapkota et al., 2025)||Menyajikan tinjauan dekade perkembangan YOLOv1 hingga YOLO dari YOLOv1 hingga versi terbaru|Review literatur sistematis. YOLOv12|YOLOv12-N: 40,6% mAP (unggul +2,1% vs YOLOv10-N), YOLOv12-S: 48,0% mAP (unggul +3,0% vs YOLOv8-S), YOLOv12-X: 55,2% mAP.|Menyediakan landasan evolusi YOLO yang mendasari pemilihan YOLOv12, memvalidasi YOLO untuk aplikasi surveillance dan security|
|4.|Object detection using Banyak varian YOLO telah YOLO: challenges, dikembangkan namun belum architectural|ada tinjauan yang merangkum arsitektur single-stage|Meninjau secara komprehensif|Literature review terhadap YOLO dan variannya|YOLO mencapai akurasi 63,4% mAP dengan inferensi|Mereview YOLO sebagai single- stage detector|
||successors, datasets and tantangan dan aplikasi deteksi applications (Diwan, Anirudh and Tembhurne, 2023)|objek menggunakan YOLO|detector khususnya YOLO, perkembangan arsitektur, performa, serta aplikasi di berbagai domain.||300× lebih cepat dibanding Fast- RCNN (70% mAP). Mendokumentasikan peningkatan mAP tiap versi: batch normalization (+2%), versi YOLO yang fine-grained features (+1%), anchor boxes (+5%).|yang menjadi dasar arsitektur YOLOv12, menyediakan perbandingan performa antar relevan sebagai acuan|
|5.|An efficient YOLOv12- Deteksi objek berskala sangat based framework for kecil masih menjadi tantangan detecting extremely utama pada model deteksi small-scale objects objek. (Chandrashekhar et al., 2025)||Mengusulkan framework berbasis YOLOv12 yang menggabungkan modul A2C2F dan C3K2 dengan multi- scale feature fusion untuk mendeteksi objek berskala sangat kecil.|YOLOv12 + A2C2F + C3K2|Pada dataset VisDrone: Precision 69,1%, Recall 48,5%, F1-score 56,99%, mAP@50 58,8%, 40 FPS (A100 GPU) mengungguli model- model terbaru pada deteksi objek kecil|Meneliti YOLOv12 sebagai base model untuk deteksi objek kecil; relevan langsung dengan fokus penelitian pada deteksi senjata api berukuran kecil.|
|6.|An Effective Object Sistem pengawasan video||Meningkatkan akurasi|Mengganti Neck|Mendapat mAP|Mengubah Neck|
||Tracking Using Yolov3 sering kesulitan mendeteksi with Bidirectional kendaraan secara akurat dan Feature Pyramid andal, terutama objek yang Network on Video (Al-Jawahry et al., kejauhan, yang dapat 2023)|berukuran kecil atau berada di mengganggu performa pelacakan (tracking) dan analisis lalu lintas.|deteksi kendaraan dengan berbagai ukuran (kecil, sedang, dan besar) dalam format gambar dan video untuk sistem pengawasan.|bawaan YOLOv3 dengan BiFPN|sebesar 90.50% dan waktu deteksi 6 milidetik serta mendapat 5 FPS.|YOLO dengan modul BiFPN|
|7.|Pedestrian detection Mendeteksi pejalan kaki di with Bi-Directional kerumunan padat sulit Feature Pyramid and dilakukan karena pose tidak Channel-Spatial Attention Modules (occlusion) (Zhang and Du, 2023)|terduga dan objek terhalang|Meningkatkan akurasi deteksi pejalan kaki, khususnya untuk target yang terhalang dan berjarak jauh.|YOLOv5 dengan BiFPN sebagai Neck dan modul CBAM|YOLOv5 yang dimodifikasi berhasil mendeteksi obyek terhalang dan kecil dengan lebih baik.|Menggunakan modul BiFPN pada Neck YOLO.|

## BAB III

## METODOLOGI PENELITIAN

##### 3.1 Diagram Alir Proses Penelitian

Penelitian ini dilaksanakan secara sistematis dan terstruktur untuk memastikan bahwa setiap tahapan, mulai dari perumusan masalah hingga penarikan kesimpulan, dapat berjalan dengan baik dan tervalidasi. Metodologi yang digunakan adalah pendekatan eksperimental di mana dua model arsitektur dibandingkan untuk mengevaluasi performa dalam tugas deteksi senjata. Secara garis besar, alur proses penelitian ini digambarkan dalam diagram alir pada Gambar 3.1.

Gambar 3.1. Diagram Alir Proses Penelitian

Proses penelitian dimulai dengan pengumpulan dataset, yang kemudian dilanjutkan dengan perancangan model YOLOv12 dengan BiFPN, lalu diikuti dengan pelatihan model YOLOv12 Standar dan YOLOv12-BiFPN, lalu model tersebut diuji dengan *Test Set* yang sudah ada, kemudian dari hasil pengujian model yang didapatkan, akan dilakukan analisis dan evaluasi hasil.

##### 3.2 Pengumpulan dan Pemrosesan Dataset

Tahapan awal dalam penelitian ini adalah pengumpulan dataset yang akan digunakan untuk melatih dan menguji model deteksi senjata api. Dataset yang digunakan terdiri dari video-video tindakan kriminal yang melibatkan senjata api

yang terekam dari CCTV. FDIE Dataset dari Kaggle ([https://www.kaggle.com/datasets/arnaldovitor/fdie-dataset/data](https://www.kaggle.com/datasets/arnaldovitor/fdie-dataset/data)) digunakan sebagai sumber dataset karena memiliki banyak video yang sudah diberikan bounding box dengan format YOLO (Da Silva and Pereira, 2024). Dataset FDIE dibangun dari 2213 frame dari dataset AIE (Assaults at Indoor Environments) yang merupakan kumpulan 700 video pengawasan CCTV indoor dari bank, minimarket, dan toko ritel, dan menganotasi semua senjata api yang terlihat dengan bounding box dalam format YOLO, dengan total 2312 anotasi (Da Silva and Pereira, 2024). Ciri utama dari dataset ini adalah variasi ukuran bounding box yang sangat lebar: minimum 82 piksel, maksimum 59.673 piksel, rata-rata 5005 piksel, dan standar deviasi 7387 piksel (Da Silva and Pereira, 2024).

Gambar 3.2. Dataset (dipotong per frame)

Gambar 3.3. Bounding Box Notations Berformat YOLO

Gambar 3.4. Ground Truth Dataset FDIE

Contoh video yang sudah dipotong setiap frame dapat dilihat pada Gambar

3.2 dan contoh Bounding box notations berformat YOLO dapat dilihat pada Gambar 3.3 yang terdiri dari 5 angka yang dipisahkan oleh spasi dengan struktur:
## <object-class> <x_center> <y_center> <width> <height>

<object-class> merupakan angka yang menunjukkan kelas atau kategori dari objek. Karena kelas hanya ada 1 yaitu senjata api, maka semua notasi memiliki <object-class> = 0. <x_center> dan <y_center> merupakan titik koordinat X dan Y dari titik tengah bounding box yang dinormalisasi terhadap lebar total gambar, sehingga nilainya berada di antara 0 hingga 1. <width> dan <height> merupakan tinggi dan lebar dari bounding box yang juga dinormalisasi terhadap tinggi dan lebar total gambar. Ground truth dataset FDIE (gambar yang sudah digabungkan dengan anotasi) ditunjukkan pada Gambar 3.4

Untuk mendukung keberagaman dan akurasi model dalam lingkungan nyata, dataset yang dipilih mencakup objek yang terlihat kecil dan teroklusi. Distribusi piksel objek dalam dataset FDIE memenuhi definisi objek kecil menurut konvensi MS-COCO (Aldubaikhi and Patel, 2025) dengan luas kurang

dari 32×32 piksel pada lebih dari 30% sampelnya, menjadikan dataset ini relevan sebagai pengujian yang ketat untuk modul fusi fitur multi-skala. Dataset ini kemudian dibagi menjadi tiga bagian: training set (70%), validation set (20%), dan testing set (10%) secara acak.

##### 3.3 Perancangan Model dan Integrasi BiFPN

Proses pembuatan arsitektur model YOLOv12-BiFPN melibatkan modifikasi pada source code YOLOv12, khususnya pada komponen Neck. Tujuannya adalah mengganti mekanisme fusi fitur standar yang berbasis PANet dengan modul BiFPN. Proses ini dapat dipecah menjadi empat langkah utama:

###### 3.3.1 Analisis Arsitektur YOLOv12 Standar

Langkah pertama adalah melakukan dekomposisi dan analisis terhadap arsitektur asli YOLOv12. Fokus utama dari tahap ini adalah untuk mengidentifikasi dan memahami struktur komponen Neck, yang berfungsi sebagai jembatan antara Backbone (ekstraktor fitur) dan Head (prediktor deteksi). Kami memetakan alur data secara cermat: mencatat dari lapisan Backbone mana saja peta fitur (feature maps) diambil sebagai input untuk Neck, serta spesifikasi output yang dihasilkan Neck untuk diteruskan ke Head. Analisis ini sangat krusial untuk memastikan bahwa modul BiFPN yang akan diintegrasikan secara presisi tanpa mengganggu alur kerja arsitektur secara keseluruhan.

###### 3.3.2 Implementasi Modul BiFPN Kustom

Setelah memahami titik integrasi, kami merancang dan mengimplementasikan modul BiFPN sebagai sebuah komponen kustom. Implementasi ini mengubah komponen Neck dari YOLOv12 yang ditunjukkan pada Gambar 2.3 sehingga menggunakan modul BiFPN. Modul BiFPN ini ditunjukkan dengan lingkaran pada Gambar 3.5 yang berwarna oranye. BiFPN akan membuat proses fusi fitur menjadi lebih dinamis.

Gambar 3.5. Arsitektur YOLOv12-BiFPN

Tidak seperti neck standar yang alirannya cenderung linear, BiFPN memperkenalkan koneksi dua arah (bidirectional cross-scale connections). Ini berarti informasi dari fitur level tinggi (yang kaya akan konteks) dapat mengalir ke bawah untuk memperkaya fitur level rendah (yang kaya akan detail), dan pada saat yang sama, informasi dari fitur level rendah akan mengalir ke atas untuk memberikan detail spasial yang lebih presisi ke fitur level tinggi (Zhang and Du,

2023). Lebih dari itu, proses aliran informasi dua arah ini diulang beberapa kali dalam blok-blok BiFPN. Hal ini memungkinkan setiap level fitur untuk secara iteratif berkomunikasi dan memadukan informasi dari berbagai arah dan skala.
###### 3.3.3 Integrasi Modul BiFPN ke dalam Arsitektur Utama

Tahap ini adalah inti dari modifikasi arsitektur. Dengan menggunakan berkas konfigurasi model, kami melakukan "transplantasi" komponen. Seluruh definisi lapisan yang membentuk Neck standar (misalnya, PANet) dihapus dari arsitektur. Sebagai gantinya, kami menyisipkan panggilan ke modul BiFPN kustom yang telah dirancang. Koneksi input untuk modul BiFPN ini kemudian disambungkan ke output yang relevan dari lapisan-lapisan Backbone yang telah diidentifikasi pada tahap pertama. Koneksi lateral skip pada level P4 dipilih karena P4 merupakan skala menengah yang menerima informasi gradien paling banyak dari lapisan dalam (Doherty *et al.*, 2025).

Pseudocode 3.1. Algoritma Proses Inferensi pada Arsitektur YOLOv12-BiFPN **Input** **I** : Citra input **M** : Model YOLOv12-BiFPN terlatih **Output** **D** : Kumpulan bounding box hasil deteksi **Notasi** **Pi** : Feature map backbone level-i (i=3,4,5) **Fi** : Feature map hasil fusi BiFPN level-i (i=3,4,5) **Langkah-langkah**

**1).** Lakukan preprocessing: resize I ke dimensi input model, normalisasi pixel ke [0, 1]
**2).** Ekstraksi fitur backbone: Backbone (I) → P3, P4, P5
**3).** Fusi fitur dengan BiFPN Neck (2-pass bidirectional): BiFPN_Neck(P3, P4, P5) → F3, F4, F5
**4).** Regresi bounding box dan klasifikasi object: Detect_Head(F3, F4, F5) → Prediksi
**5).** Non-Maximum Suppresion (hilangkan box tumpang tindih): NMS(Prediksi) → D
**6).** Kembalikan hasil D Secara konseptual, alur modifikasi berkas konfigurasi dan proses inferensi dapat diilustrasikan dengan pseudocode seperti pada Pseudocode 3.1.
###### 3.3.4 Penyesuaian Komponen Head

Langkah terakhir adalah memastikan kompatibilitas antara output dari Neck BiFPN yang baru dengan input yang diharapkan oleh komponen Head. Kami memverifikasi bahwa jumlah dan dimensi (jumlah channels) dari peta fitur yang dihasilkan oleh BiFPN sudah sesuai dengan konfigurasi lapisan deteksi pada Head. Jika terdapat ketidaksesuaian, lapisan konvolusi proyeksi 1×1 disisipkan sebelum operasi penambahan BiFPN untuk menyelaraskan dimensi. Setelah sinkronisasi ini selesai, arsitektur model YOLOv12-BiFPN secara utuh telah selesai dirancang dan siap untuk tahap kompilasi dan pelatihan.

##### 3.4 Pelatihan dan Pengujian YOLOv12 Standar dan YOLOv12-BiFPN

Model dilatih pada dataset yang telah disiapkan dan dievaluasi menggunakan validation set. Kedua model dilatih selama 100 epoch dengan

learning rate tetap lr₀ sebanyak 0,0001. Batch size ditetapkan pada 6 secara konsisten di semua varian sesuai dengan batasan VRAM GPU (16 GB), dengan ukuran gambar 640×640 piksel sebagai resolusi input standar. Kedua model dilatih sepenuhnya dari awal tanpa menggunakan pre-trained weights. Pengujian dilakukan pada data uji dengan metrik evaluasi seperti Precision, Recall, dan mAP (mean Average Precision) untuk mengukur akurasi deteksi, serta FPS (Frames Per Second) untuk mengetahui efektivitas sistem dalam bekerja secara real-time.

##### 3.5 Analisis dan Evaluasi Hasil

Setelah proses pelatihan dan pengujian selesai, hasil dari kedua model dievaluasi dan dibandingkan. Model dengan kombinasi BiFPN dievaluasi untuk melihat sejauh mana peningkatan akurasi dan kecepatan dapat dicapai dibandingkan YOLOv12 standar. Hasil evaluasi disajikan dalam bentuk grafik dan tabel, termasuk perbandingan metrik kinerja dan visualisasi hasil deteksi secara real-time. Selain itu, dilakukan juga analisis terhadap kasus-kasus di mana model gagal mengenali senjata dengan benar, untuk mengevaluasi kelemahan model yang masih dapat ditingkatkan pada penelitian selanjutnya.

Untuk mengukur kontribusi dari modul BiFPN, penelitian ini akan menerapkan metodologi studi ablasi (ablation study). Pendekatan ini dilakukan dengan membandingkan dua konfigurasi secara langsung: pertama adalah model YOLOv12 standar yang komponen Neck-nya tidak dimodifikasi. Model ini merepresentasikan kondisi di mana komponen baru (BiFPN) belum diimplementasikan. Kedua adalah model YOLOv12-BiFPN, yang merupakan arsitektur yang telah ditambahkan dengan modul fusi fitur yang baru. Dengan membandingkan metrik kinerja antara model yang dikurangi (baseline) dan yang ditambahi (proposisi), selisih performa yang terukur dapat secara langsung diatribusikan pada efektivitas modul BiFPN dalam meningkatkan kemampuan deteksi model.

Untuk mengukur performa model secara kuantitatif dan komparatif, digunakan metrik evaluasi Precision, Recall, mAP@50, mAP@50-95, dan FPS yang telah dijelaskan sebelumnya pada Subbab 2.1.8. Seluruh metrik tersebut dihitung pada test set sebesar 10% dari total dataset FDIE, sebagaimana dijelaskan pada Subbab 3.2. Test set ini tidak digunakan selama proses pelatihan maupun validasi, sehingga metrik yang diperoleh merepresentasikan kemampuan generalisasi model terhadap data yang belum pernah dilihat sebelumnya.

## BAB IV

## HASIL DAN PEMBAHASAN

###### 4.1 Pengumpulan Dataset

Dataset FDIE yang berisi 2213 frame dan 2312 anotasi instance senjata api dipartisi menjadi subset training, validation, dan testing mengikuti rasio pembagian 70%/20%/10% sebagaimana disajikan pada Bab 3.2. Untuk mencegah kebocoran data, frame yang diekstraksi dari video sumber yang sama ditetapkan secara eksklusif ke satu subset, memastikan tidak ada tumpang tindih spasial atau temporal antara data training, validation, dan testing. Distribusi gambar dan anotasi yang dihasilkan di ketiga subset dirangkum dalam Tabel 4.1, dan distribusi ukuran bounding box per subset disajikan dalam Tabel 4.2.

Tabel 4.1. Distribusi Gambar dan Anotasi per Subset

|Subset|||Gambar|||Anotasi|
|---|---|---|---|---|---|---|
|Training||1549|||1614||
|Validation||424|||432||
|Test||240|||266||
|Total||2213|Tabel 4.2. Distribusi Ukuran Bounding Box per Subset||2312||
|Subset|Min (px²|)|Max (px²)||Rata-rata (px²)|Std Dev (px²)|
|Training|82||59.673|4.993||7.369|
|Validation|142||48.209|5.022||7.389|
|Testing|154||50.083|5.064||7.519|
|Keseluruhan|82||59.673|5.006||7.388|

Subset testing berisi 266 anotasi ground truth senjata api yang tersebar di 240 frame, dengan ukuran bounding box berkisar dari senjata kecil dan jauh yang menempati kurang dari 32×32 piksel hingga senjata api jarak dekat yang besar mengisi sebagian besar frame. Rentang ukuran ini konsisten di ketiga subset, mengonfirmasi bahwa test set mempertahankan karakteristik kesulitan deteksi yang sama dengan data training.

###### 4.2 Hasil Pelatihan Model

Gambar 4.1 dan Gambar 4.2 menunjukkan progresi train loss dan val loss YOLOv12 standar dan YOLOv12-BiFPN yang memiliki tiga komponen loss seperti yang telah dijelaskan pada Bab 2.1.9. Classification loss pada Gambar 4.2b dan 4.2e mengalami lonjakan ekstrem di epoch awal yang diikuti dengan penurunan signifikan dan akhirnya naik kembali.

Gambar 4.1. Train Loss YOLOv12 dan YOLOv12-BiFPN

Gambar 4.2. Val Loss YOLOv12 dan YOLOv12-BiFPN

Lonjakan loss awal sebagian besar berasal dari komponen cls_loss dan dfl_loss selama beberapa epoch pertama, konsisten dengan fase eksplorasi learning rate di mana model belum mencapai kesepakatan antara prediksi kelas dan distribusi koordinat bounding box terhadap targetnya. Setelah epoch ke-15, ketiga komponen loss menurun secara monoton untuk semua varian. Varian BiFPN menunjukkan pola konvergensi yang lebih halus pada val/cls_loss dibandingkan varian standar, menunjukkan bahwa weighted feature fusion membantu mengurangi varians prediksi kelas pada level mini-batch selama pelatihan.

###### 4.3 Confusion Matrix

Confusion Matrix digunakan untuk menganalisis distribusi deteksi benar dan salah pada setiap model. Gambar 4.3 menyajikan confusion matrix untuk YOLOv12m-BiFPN. Varian YOLOv12m-BiFPN dipilih karena merupakan model terbaik secara keseluruhan berdasarkan keseimbangan Precision-Recall serta peningkatan TP dan penurunan FN secara simultan.

Gambar 4.3. Confusion Matrix YOLOv12m-BiFPN

Pada Gambar 4.3, confusion matrix menunjukkan bahwa dari 266 ground truth senjata api pada FDIE test set, model berhasil mendeteksi 66 sebagai True Positive (TP) dan gagal mendeteksi 200 sebagai False Negative (FN). Sementara itu, model menghasilkan 45 False Positive (FP) yaitu area latar belakang yang salah diklasifikasikan sebagai senjata. Sebagian besar prediksi latar belakang lainnya berhasil diklasifikasikan dengan benar sebagai True Negative (TN). Pola ini menegaskan karakteristik YOLOv12m-BiFPN yaitu Precision yang cukup baik yaitu sebesar 0,595 karena FP relatif rendah, namun Recall yang masih terbatas sebesar 0,248 akibat dominasi FN, menunjukkan bahwa mayoritas senjata di dataset FDIE masih sulit terdeteksi.

Gambar 4.4. Confusion Matrix YOLOv12 dan YOLOv12-BiFPN

Untuk perbandingan, Gambar 4.4 merupakan diagram batang TP/FP/FN untuk seluruh varian YOLOv12 yang dievaluasi pada FDIE test set dengan confidence threshold sebesar 0,25, IoU sebesar 0,7, dan 266 Ground Truth box. Detail nilai per model ditunjukkan pada Tabel 4.3.

Tabel 4.3. Confusion Matrix YOLOv12 dan YOLOv12-BiFPN

|Model|TP|FP|FN|
|---|---|---|---|
|YOLOv12n|58|65|208|
|YOLOv12s|53|29|213|
|YOLOv12m|56|46|210|
|YOLOv12l|59|28|207|
|YOLOv12x|6|11|219|
|YOLOv12n-BiFPN|51|24|215|
|YOLOv12s-BiFPN|51|27|215|
|YOLOv12m-BiFPN|66|45|200|
|YOLOv12l-BiFPN|46|7|220|
|YOLOv12x-BiFPN|37|33|188|

Seluruh model menunjukkan FN yang tinggi yaitu 188–220 dari 266 Ground Truth box, mengindikasikan banyaknya senjata yang tidak terdeteksi. BiFPN secara konsisten mengurangi FP di hampir semua skala, dengan dampak terbesar pada varian large di mana FP menurun dari 28 menjadi 7, penurunan sebesar 75%, dan nano dari FP 65 menjadi 24, penurunan sebesar 63%. Satu-satunya varian yang secara simultan meningkatkan TP dan menurunkan FN adalah YOLOv12m-BiFPN, dengan TP dari 56 menjadi 66 dan FN dari 210 menjadi 200. Varian YOLOv12x standar hampir tidak menghasilkan deteksi dengan TP sebesar 6, namun BiFPN secara signifikan memulihkan kemampuan deteksi YOLOv12x dengan menaikkan TP dari 6 menjadi 37.

###### 4.4 Perbandingan Performa Deteksi

Performa deteksi seluruh model yang dievaluasi pada subset uji FDIE dirangkum dalam Gambar 4.5 dan Tabel 4.4 di bawah. Tiga varian YOLOv8n, YOLOv8s, YOLOv8m disertakan sebagai baseline referensi (Da Silva and Pereira, 2024). Perlu dicatat bahwa dataset FDIE tidak dipublikasikan dalam keadaan pre-split, sehingga frame mana yang dimasukkan ke subset train, val, dan test pada penelitian (Da Silva and Pereira, 2024) tidak diketahui. Split dalam penelitian ini dilakukan secara independen, sehingga komposisi frame per subset bisa sangat mungkin berbeda dari penelitian asli. Deteksi dilakukan pada test set

dengan parameter confidence threshold sebesar 0,25, Intersection over Union sebesar 0,7, ukuran gambar sebesar 640 pixel.

Tabel 4.4. Perbandingan Performa YOLOv8 (baseline), YOLOv12 standard dan YOLOv12-BiFPN

|Model|Precision|Recall|mAP@50|mAP@50-90|FPS|
|---|---|---|---|---|---|
|YOLOv8n (Silva dan Pereira)|0.699|0.318|0.383|-|-|
|YOLOv8s (Silva dan Pereira)|0.730|0.395|0.455|-|-|
|YOLOv8m (Silva dan Pereira)|0.690|0.320|0.378|-|-|
|YOLOv12n|0.471|0.217|0.374|0.198|255.6|
|YOLOv12s|0.642|0.200|0.435|0.231|263.9|
|YOLOv12m|0.546|0.208|0.402|0.238|190.8|
|YOLOv12l|0.676|0.223|0.436|0.291|191.5|
|YOLOv12x|0.352|0.026|0.192|0.096|112.4|
|YOLOv12n- BiFPN|0.682|0.191|0.458|0.252|112.9|
|YOLOv12s- BiFPN|0.651|0.191|0.433|0.233|81.8|
|YOLOv12m- BiFPN|0.595|0.248|0.448|0.278|81.9|
|YOLOv12l- BiFPN|0.867|0.173|0.529|0.305|81.0|
|YOLOv12x- BiFPN|0.500|0.015|0.322|0.184|57.5|

Pada Tabel 4.2 dan Gambar 4.5 terbukti bahwa pada YOLOv12 standar, model yang lebih besar tidak selalu lebih baik. YOLOv12l mencapai mAP@50 tertinggi sebesar 0,436, sementara YOLOv12x turun drastis akibat overfitting. BiFPN meningkatkan mAP@50 pada empat dari lima varian, dengan hasil terbaik YOLOv12l-BiFPN dengan mAP@50 sebesar 0,530, melampaui baseline terbaik YOLOv8s dari (Da Silva and Pereira, 2024) sebesar 0,455. Seluruh varian YOLOv12 menunjukkan Recall yang lebih rendah dibandingkan YOLOv8.

Gambar 4.5. Perbandingan Performa YOLOv8 (baseline), YOLOv12 dan

YOLOv12-BiFPN

Pada varian BiFPN, YOLOv12l-BiFPN menunjukkan performa paling seimbang: Recall tertinggi di antara seluruh varian BiFPN sebesar 0.173, Precision sebesar 0.867, mAP@50 sebesar 0.529, mAP@50-95 sebesar 0.305, serta FPS sebesar 81.0. Pada Tabel 4.4, YOLOv12m-BiFPN merupakan satu-satunya varian yang secara simultan meningkatkan TP sebanyak 10 dan menurunkan FN sebanyak 10 terhadap YOLOv12m standar, menjadikannya model dengan keseimbangan Precision-Recall terbaik di antara seluruh varian yang diuji.

###### 4.5 Hasil Deteksi Visual

Analisis dilakukan pada YOLOv12l dan YOLOv12l-BiFPN. Varian YOLOv12l-BiFPN dipilih karena mencapai mAP@50 tertinggi. Tiga kasus mewakili skenario berbeda: False Positive, deteksi objek kecil berhasil, dan kegagalan akibat kamuflase warna, hasil ditunjukkan pada Tabel 4.5.

Tabel 4.5. Perbandingan Deteksi Visual YOLOv12 dan YOLOv12-BiFPN

|Case|YOLOv12|YOLOv12-BiFPN|Ground Truth|Confidence||
|---|---|---|---|---|---|
|YOLOv12|YOLOv12-BiFPN|||||
|1||||0.26|-|
|2||||-|0.64|
|3||||-|-|
|4||||-|0.31|

Tabel 4.5 Kasus 1 mengilustrasikan kasus di mana model YOLOv12l standar menghasilkan deteksi False Positive, yaitu area latar belakang yang salah dideteksi sebagai senjata api. Model BiFPN pada frame yang sama tidak mendeteksi apa pun, menunjukkan bahwa weighted feature fusion BiFPN membantu model membedakan fitur senjata asli dari pola latar belakang yang mirip secara visual.

Tabel 4.5 Kasus 2 mendemonstrasikan kasus di mana YOLOv12l-BiFPN berhasil mendeteksi senjata api yang berukuran sangat kecil dalam frame di bawah kondisi pencahayaan sangat terang yang semakin mengurangi kontras gambar di sekitar objek. Koneksi bidirectional cross-scale BiFPN memungkinkan detail spasial dari level fitur rendah untuk secara iteratif memperkuat fitur semantik di level tinggi, memungkinkan model mempertahankan isyarat lokalisasi untuk objek kecil.

Tabel 4.5 Kasus 3 menyajikan kasus kegagalan bersama di mana baik YOLOv12l standar maupun YOLOv12l-BiFPN gagal mendeteksi senjata api. Pada frame ini, senjata dipegang di depan tubuh seseorang yang mengenakan pakaian gelap, menyebabkan senjata secara visual menyatu dengan latar belakang karena kesamaan warna dan tekstur. Kedua model tidak menghasilkan bounding box pada frame ini, menunjukkan bahwa kegagalan yang disebabkan oleh kamuflase warna tidak dapat diselesaikan hanya melalui modifikasi arsitektur neck.

Tabel 4.5 Kasus 4 menunjukkan kasus di mana YOLOv12l-BiFPN berhasil mendeteksi senjata api yang teroklusi sebagian, sementara YOLOv12l standar gagal mendeteksinya. Hal ini mendemonstrasikan bahwa fusi fitur bidirectional multi-skala BiFPN memungkinkan model untuk mengagregasi informasi kontekstual dari wilayah sekitar, sehingga dapat menginferensi keberadaan senjata meskipun sebagian darinya tersembunyi di balik objek lain.

###### 4.6 Pengaruh BiFPN terhadap Performa Deteksi

Integrasi BiFPN meningkatkan empat dari lima varian ukuran YOLOv12 pada mAP@50. Peningkatan terbesar terjadi pada skala large: YOLOv12l-BiFPN mencapai mAP@50 sebesar 0,530 dibandingkan 0,436 untuk YOLOv12l standar, peningkatan sebesar 0,094. Seluruh varian BiFPN juga meningkatkan mAP@50- 95, menunjukkan bahwa BiFPN menghasilkan lokalisasi bounding box yang lebih ketat pada threshold IoU yang lebih tinggi. Pengecualian adalah YOLOv12s- BiFPN di mana mAP@50 menurun sebesar 0,002, disebabkan oleh parameter tambahan BiFPN yang memperkenalkan tantangan optimasi pada skala ini tanpa peningkatan kapasitas representasional yang sebanding. Varian YOLOv12x menunjukkan performa jauh di bawah skala lainnya dengan YOLOv12x standar mAP@50 sebesar 0,192, namun BiFPN tetap memberikan peningkatan absolut terbesar sebesar 0,130, menunjukkan bahwa kapasitas fusi fitur BiFPN lebih efektif dimanfaatkan oleh model yang lebih besar.

Pola peningkatan ini menarik karena bertentangan dengan asumsi umum bahwa modul fusi tambahan akan memberikan manfaat yang seragam di semua ukuran model. Hasil di atas menunjukkan interaksi non-linear antara kapasitas backbone dan kemampuan agregasi neck: pada ukuran nano dan small, kapasitas representasional backbone relatif terbatas, sehingga manfaat fusi multi-skala tambahan dibatasi oleh rendahnya kekayaan fitur input. Sebaliknya, pada ukuran large dan extra-large, backbone menghasilkan peta fitur yang lebih kaya, memungkinkan BiFPN untuk lebih efektif memilih dan membobot kontribusi setiap level. Temuan ini konsisten dengan observasi dari BiFPN-YOLO (Doherty et al., 2025) yang melaporkan bahwa manfaat BiFPN paling terlihat pada konfigurasi backbone dengan parameter yang lebih besar.

Lebih lanjut, peningkatan yang lebih konsisten pada mAP@50-95 dibandingkan mAP@50 menunjukkan bahwa BiFPN tidak hanya meningkatkan keberadaan deteksi tetapi juga akurasi lokalisasi bounding box, aspek yang krusial dalam konteks pengawasan di mana penentuan posisi senjata yang akurat

diperlukan untuk analisis ancaman dan pelacakan multi-objek tingkat lanjut (Sapkota et al., 2025). Peningkatan rata-rata mAP@50-95 sebesar 0,03 poin di seluruh varian BiFPN setara dengan pergeseran Intersection over Union yang signifikan pada threshold tinggi 0,7–0,9, dan ini secara khusus berdampak pada deteksi senjata kecil yang hanya berukuran beberapa puluh piksel.

###### 4.7 Trade-Off Precision-Recall dan Perilaku Deteksi

Rincian TP/FP/FN pada Tabel 4.3 menunjukkan pola yang bervariasi di seluruh ukuran model. Pada varian n, s, m, dan l YOLOv12 standar dan YOLOv12-BiFPN, BiFPN secara konsisten mengurangi FP: nano dari 65 menjadi 24, penurunan 63%, small dari 29 menjadi 27, penurunan 7%, medium dari 46 menjadi 45, penurunan 2%, dan large dari 28 menjadi 7, penurunan 75%. Tabel

4.3 Kasus 1 secara langsung mengilustrasikan hal ini: area latar belakang yang salah dideteksi sebagai senjata oleh YOLOv12l standar berhasil ditekan oleh varian BiFPN. Namun, peningkatan Precision ini datang dengan biaya penurunan Recall pada sebagian besar varian, hanya YOLOv12m-BiFPN yang mencapai peningkatan TP simultan sebesar 10 dan penurunan FN sebesar 10. Varian extra-large menunjukkan pola yang berbeda: YOLOv12x standar hanya menghasilkan TP yang sangat rendah sebesar 6 dengan FP kecil sebesar 11, mengindikasikan model yang overfitting terhadap training set. BiFPN pada skala ini secara dramatis meningkatkan TP dari 6 menjadi 37, peningkatan 517%, dan menurunkan FN dari 219 menjadi 188, penurunan 14%, meskipun dengan peningkatan FP dari 11 menjadi 33. Pola ini mengkonfirmasi bahwa BiFPN membantu model extra-large keluar dari kondisi overfitting dengan meningkatkan kemampuan deteksinya, namun dengan trade-off precision yang lebih tinggi. Tabel 4.3 Kasus 2 menunjukkan keunggulan BiFPN dalam mendeteksi senjata kecil di bawah kondisi pencahayaan terang, sementara Tabel
4.3 Kasus 3 mengungkapkan kegagalan bersama: kedua model gagal mendeteksi senjata yang dipegang di depan pakaian gelap, menunjukkan bahwa kegagalan

yang disebabkan oleh kamuflase warna tidak dapat diatasi melalui perubahan arsitektur neck saja.

###### 4.8 Analisis Trade-off Akurasi-Kecepatan

Integrasi BiFPN meningkatkan akurasi deteksi (mAP@50) pada empat dari lima varian, namun dengan konsekuensi penurunan kecepatan inferensi (FPS) yang signifikan di seluruh varian. Berdasarkan data pada Tabel 4.4, seluruh varian BiFPN mengalami penurunan FPS yang substansial: YOLOv12n-BiFPN turun dari 255,6 menjadi 112,9 FPS atau sebanyak 55,8%, YOLOv12s-BiFPN dari 263,9 menjadi 81,8 FPS atau sebanyak 69,0%, YOLOv12m-BiFPN dari 190,8 menjadi 81,9 FPS yaitu sebanyak 57,1%, YOLOv12l-BiFPN dari 191,5 menjadi 81,0 FPS, turun sebanyak 57,7%, dan YOLOv12x-BiFPN dari 112,4 menjadi 57,5 FPS yaitu sebanyak 48,8%. Penurunan ini disebabkan oleh tambahan komputasi dari koneksi bidirectional cross-scale dan mekanisme weighted feature fusion yang dijalankan pada setiap forward pass.

Meskipun BiFPN memperlambat inferensi, seluruh model masih mampu memproses video secara real-time tanpa mengalami frame drop yang mengganggu operasional pengawasan.

Untuk mengevaluasi apakah trade-off ini sepadan, digunakan metrik efisiensi akurasi terhadap kecepatan, yaitu rasio peningkatan mAP@50 terhadap persentase penurunan FPS:

## Tabel 4.6. Rasio Efisiensi Trade-off Akurasi-Kecepatan

|Varian|Δ mAP@50|Penurunan FPS (%)|Δ mAP per % FPS Loss|
|---|---|---|---|
|n|+0,084|-55,8%|0,0015|
|s|-0,002|-69,0%|-0,00003|
|m|+0,046|-57,1%|0,0008|
|l|+0,096|-57,7%|0,0016|
|x|+0,130|-48,8%|0,0027|

Berdasarkan Tabel 4.6, varian large dan extra-large menunjukkan efisiensi trade-off tertinggi, dengan YOLOv12x-BiFPN mencapai rasio 0,0027 dan

YOLOv12l-BiFPN sebesar 0,0016. Varian small justru menunjukkan trade-off negatif di mana akurasi sedikit menurun dengan penurunan kecepatan terbesar. Pola ini konsisten dengan temuan sebelumnya bahwa manfaat BiFPN paling terlihat pada model berkapasitas besar yang backbone-nya mampu menghasilkan peta fitur yang cukup kaya untuk dimanfaatkan oleh mekanisme fusi multi-skala.

###### 4.9 Perbandingan dengan Baseline dan Penelitian Sebelumnya

Fokus utama analisis penelitian ini adalah perbandingan internal antara YOLOv12 standar dan YOLOv12-BiFPN yang dilatih dan dievaluasi pada split data yang sama, sehingga perbedaan performa yang teramati dapat sepenuhnya diatribusikan pada modifikasi arsitektur neck. Hasil eksperimen menunjukkan bahwa YOLOv12l-BiFPN mencapai mAP@50 tertinggi sebesar 0,529, melampaui YOLOv12l standar dengan mAP@50 sebesar 0,436 dengan selisih 0,093. Varian YOLOv12 standar pada skala large dengan mAP@50 sebesar 0,436 dan skala small dengan mAP@50 sebesar 0,435 menunjukkan performa yang sudah mendekati baseline YOLOv8s dari (Da Silva and Pereira, 2024) dengan mAP@50 sebesar 0,455, mengindikasikan bahwa arsitektur YOLOv12 tanpa modifikasi neck sudah kompetitif pada dataset FDIE. Sebagai konteks, varian YOLOv8 dari (Da Silva and Pereira, 2024) disertakan dalam Tabel 4.4 sebagai referensi historis, bukan sebagai perbandingan head-to-head, karena split dataset yang digunakan tidak dipublikasikan sehingga komposisi test set bisa sangat mungkin berbeda dari penelitian ini. Dibandingkan dengan studi deteksi senjata pada dataset lain, nilai mAP absolut tetap lebih rendah daripada yang dilaporkan oleh studi (Bhatti *et al.*,

2021) dengan mAP@50 sebesar 91,73% untuk YOLOv4 dan studi YOLOv8 berbasis FMR-CNN (P and V, 2025), mengingat dataset FDIE memiliki rentang ukuran senjata yang sangat lebar. Pola peningkatan BiFPN konsisten dengan (Zhang and Du, 2023) yang melaporkan peningkatan serupa dalam konteks deteksi pengawasan. Perlu dicatat bahwa dataset FDIE tidak dipublikasikan dalam keadaan pre-split, sehingga split dilakukan secara independen dalam penelitian ini. Ini berarti

angka mAP tidak dapat secara langsung dan terkontrol dibandingkan dengan studi asli (Da Silva and Pereira, 2024), karena komposisi test set kemungkinan berbeda. Lebih lanjut, dibandingkan dengan studi deteksi senjata pada dataset outdoor atau gambar tersusun seperti (Bhatti *et al.*, 2021) dan (P and V, 2025), dataset FDIE secara intrinsik lebih sulit karena mencakup frame CCTV aktual dengan kompresi video, motion blur, dan variasi sudut kamera. Hal ini menjelaskan mengapa nilai mAP absolut pada FDIE lebih rendah dari benchmark lain meskipun arsitektur yang digunakan lebih baru.

## BAB V

## KESIMPULAN

###### 5.1 Kesimpulan

Penelitian ini berhasil mengimplementasikan arsitektur YOLOv12-BiFPN pada dataset FDIE dan mengevaluasinya secara komparatif terhadap YOLOv12 standar pada lima varian ukuran model n, s, m, l, x. Hasil eksperimen menunjukkan bahwa integrasi BiFPN meningkatkan mAP@50 pada empat dari lima varian, dengan performa terbaik pada YOLOv12l-BiFPN dengan mAP@50 sebesar 0,529 dan mAP@50-95 sebesar 0,305, peningkatan sebesar 0,093 dibandingkan YOLOv12l standar. Secara konsisten, BiFPN juga mengurangi False Positive sehingga meningkatkan Precision, meskipun disertai dengan penurunan Recall pada sebagian besar varian seiring dengan prediksi model yang menjadi lebih konservatif. Analisis kualitatif menunjukkan bahwa perbaikan neck efektif dalam menekan false alarm dan membantu deteksi objek kecil pada beberapa kasus, namun belum mampu mengatasi kegagalan pada kondisi kamuflase warna (senjata menyatu dengan pakaian gelap). Dengan demikian, tujuan penelitian untuk menilai dampak fusi fitur dua arah multi-skala BiFPN terhadap performa YOLOv12 telah tercapai, terbukti memberikan peningkatan akurasi deteksi dan pengurangan false alarm. Penelitian lebih lanjut masih diperlukan terutama untuk meningkatkan Recall dan memperkuat validitas evaluasi mengingat dataset FDIE tidak menyediakan split yang telah ditentukan sebelumnya.

###### 5.2 Saran

Nilai Recall yang secara konsisten rendah di seluruh model menunjukkan bahwa masih terdapat ruang peningkatan yang signifikan, terutama melalui penyesuaian confidence threshold, parameter Non-Maximum Suppression (NMS), dan augmentasi data yang ditargetkan. Selain itu, pengujian deploy langsung pada infrastruktur pengawasan yang sesungguhnya berada di luar cakupan penelitian

ini. Berdasarkan keterbatasan tersebut, beberapa saran untuk penelitian selanjutnya adalah sebagai berikut:

**1.** Melakukan augmentasi data yang ditargetkan untuk skenario sulit seperti sintesis kondisi low-light, simulasi motion blur, dan augmentasi warna untuk kasus kamuflase guna meningkatkan Recall model. Penelitian tentang deteksi senjata pada kondisi low-light dengan image enhancement (Pravesh and Sahana, 2025) menunjukkan pendekatan ini efektif.
**2.** Mengeksplorasi integrasi modul attention tambahan pada backbone seperti CBAM, ECA, atau coordinate attention (Bai and Song, 2025) untuk meningkatkan kemampuan diskriminasi fitur senjata dari latar belakang yang mirip secara visual.
**3.** Mengintegrasikan modalitas tambahan seperti thermal imagery (Veranyurt and Sakar, 2023) untuk membantu deteksi senjata yang tersembunyi atau terkamuflase warna, mengingat senjata logam memiliki tanda termal yang berbeda dari kain dan tubuh manusia.
**4.** Melakukan pengujian deploy langsung pada infrastruktur pengawasan yang sesungguhnya untuk memvalidasi performa model dalam kondisi operasional nyata.

## DAFTAR PUSTAKA

Ahmed, A.A. and Echi, M. (2021) “Hawk-Eye: An AI-Powered Threat Detector for Intelligent Surveillance Cameras,” *IEEE Access*, 9, pp. 63283–63293. Available at: [https://doi.org/10.1109/ACCESS.2021.3074319](https://doi.org/10.1109/ACCESS.2021.3074319). Aldubaikhi, A. and Patel, S. (2025) “Advancements in Small-Object Detection (2023–2025): Approaches, Datasets, Benchmarks, Applications, and Practical Guidance,” *Applied Sciences*, 15(22), p. 11882. Available at: [https://doi.org/10.3390/app152211882](https://doi.org/10.3390/app152211882). Al-Jawahry, H.M. *et al.* (2023) “An Effective Object Tracking Using Yolov3 with Bidirectional Feature Pyramid Network on Video Surveillance,” *2023 3rd* *International Conference on Mobile Networks and Wireless Communications* *(ICMNWC)*. *2023 3rd International Conference on Mobile Networks and* *Wireless Communications (ICMNWC)*, Tumkur, India: IEEE, pp. 1–6. Available at: [https://doi.org/10.1109/ICMNWC60182.2023.10435865](https://doi.org/10.1109/ICMNWC60182.2023.10435865). Bai, L. and Song, Z.J. (2025) “Omni-dimensional dynamic convolution with coordinate attention detection scheme,” *Science Progress*, 108(2), p. 00368504251336695. Available at: [https://doi.org/10.1177/00368504251336695](https://doi.org/10.1177/00368504251336695). Berardini, D. *et al.* (2023) “A deep-learning framework running on edge devices for handgun and knife detection from indoor video-surveillance cameras,” *Multimedia Tools and Applications*, 83(7), pp. 19109–19127. Available at: [https://doi.org/10.1007/s11042-023-16231-x](https://doi.org/10.1007/s11042-023-16231-x). Bhatti, M.T. *et al.* (2021) “Weapon Detection in Real-Time CCTV Videos Using Deep Learning,” *IEEE Access*, 9, pp. 34366–34382. Available at: [https://doi.org/10.1109/ACCESS.2021.3059170](https://doi.org/10.1109/ACCESS.2021.3059170). Chandrashekhar, A. *et al.* (2025) “An efficient YOLOv12-based framework for detecting extremely small-scale objects,” *Scientific Reports*, 16(1), p. 2062. Available at: [https://doi.org/10.1038/s41598-025-31803-7](https://doi.org/10.1038/s41598-025-31803-7). D, S. *et al.* (2023) “Proactive Headcount and Suspicious Activity Detection using YOLOv8,” *Procedia Computer Science*, 230, pp. 61–69. Available at: [https://doi.org/10.1016/j.procs.2023.12.061](https://doi.org/10.1016/j.procs.2023.12.061). Da Silva, A.V.B. and Pereira, L.F.A. (2024) “Evaluating Methods for Violence Classification and Firearm Detection in Indoor CCTV Environment,” *Journal* *of the Brazilian Computer Society*, 30(1), pp. 411–420. Available at: [https://doi.org/10.5753/jbcs.2024.3282](https://doi.org/10.5753/jbcs.2024.3282).

Diwan, T., Anirudh, G. and Tembhurne, J.V. (2023) “Object detection using YOLO: challenges, architectural successors, datasets and applications,” *Multimedia Tools and Applications*, 82(6), pp. 9243–9275. Available at: [https://doi.org/10.1007/s11042-022-13644-y](https://doi.org/10.1007/s11042-022-13644-y). Doherty, J. *et al.* (2025) “BiFPN-YOLO: One-stage object detection integrating Bi-Directional Feature Pyramid Networks,” *Pattern Recognition*, 160, p. 111209. Available at: [https://doi.org/10.1016/j.patcog.2024.111209](https://doi.org/10.1016/j.patcog.2024.111209). Gao, J. *et al.* (2024) “Augmented weighted bidirectional feature pyramid network for marine object detection,” *Expert Systems with Applications*, 237, p. 121688. Available at: [https://doi.org/10.1016/j.eswa.2023.121688](https://doi.org/10.1016/j.eswa.2023.121688). K U, F.S. and M, S. (2023) “Subduing Crime and Threat in Real-Time by Detecting Weapons Using Yolov8,” *2023 International Conference on Circuit* *Power and Computing Technologies (ICCPCT)*. *2023 International Conference* *on Circuit Power and Computing Technologies (ICCPCT)*, Kollam, India: IEEE, pp. 864–868. Available at: [https://doi.org/10.1109/ICCPCT58313.2023.10245146](https://doi.org/10.1109/ICCPCT58313.2023.10245146). Khanam, M.H. and R, R. (2025) “Hybrid Deep Learning Models for Anomaly Detection in CCTV Video Surveillance,” *2025 4th International Conference on* *Sentiment Analysis and Deep Learning (ICSADL)*. *2025 4th International* *Conference on Sentiment Analysis and Deep Learning (ICSADL)*, Bhimdatta, Nepal: IEEE, pp. 1345–1351. Available at: [https://doi.org/10.1109/ICSADL65848.2025.10933441](https://doi.org/10.1109/ICSADL65848.2025.10933441). Kim, H. *et al.* (2024) “Elevating urban surveillance: A deep CCTV monitoring system for detection of anomalous events via human action recognition,” *Sustainable Cities and Society*, 114, p. 105793. Available at: [https://doi.org/10.1016/j.scs.2024.105793](https://doi.org/10.1016/j.scs.2024.105793). Li, X. *et al.* (2020) “Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection.” arXiv. Available at: [https://doi.org/10.48550/arXiv.2006.04388](https://doi.org/10.48550/arXiv.2006.04388). Murugan, T. *et al.* (2025) “AI-Based Weapon Detection for Security Surveillance: Recent Research Advances (2016–2025),” *Electronics*, 14(23), p. 4609. Available at: [https://doi.org/10.3390/electronics14234609](https://doi.org/10.3390/electronics14234609). P, S. and V, M. (2025) “Weapon detection with FMR-CNN and YOLOv8 for enhanced crime prevention and security,” *Scientific Reports*, 15(1), p. 26766. Available at: [https://doi.org/10.1038/s41598-025-07782-0](https://doi.org/10.1038/s41598-025-07782-0).

Pravesh, R. and Sahana, B.C. (2025) “Robust Firearm Detection in Low-Light Surveillance Conditions Using YOLOv11 with Image Enhancement,” *International Journal of Safety and Security Engineering*, 15(4), pp. 797–809. Available at: [https://doi.org/10.18280/ijsse.150416](https://doi.org/10.18280/ijsse.150416). Sapkota, R. *et al.* (2025) “YOLO advances to its genesis: a decadal and comprehensive review of the You Only Look Once (YOLO) series,” *Artificial* *Intelligence Review*, 58(9), p. 274. Available at: [https://doi.org/10.1007/s10462-](https://doi.org/10.1007/s10462-) 025-11253-3. Shalini, A. *et al.* (2025) “Enhanced Surveillance Through YOLOv3-Based on Deep Learning for Real-Time Weapon Detection,” *2025 1st International* *Conference on AIML-Applications for Engineering &amp; Technology* *(ICAET)*. *2025 1st International Conference on AIML-Applications for* *Engineering &amp; Technology (ICAET)*, Pune, India: IEEE, pp. 1–6. Available at: [https://doi.org/10.1109/ICAET63349.2025.10932160](https://doi.org/10.1109/ICAET63349.2025.10932160). Sujatha, E. and Janani, D. (2024) “Real Time Activity Monitoring Using Deep Learning,” *2024 5th International Conference on Innovative Trends in* *Information Technology (ICITIIT)*. *2024 5th International Conference on* *Innovative Trends in Information Technology (ICITIIT)*, Kottayam, India: IEEE, pp. 1–6. Available at: [https://doi.org/10.1109/ICITIIT61487.2024.10580124](https://doi.org/10.1109/ICITIIT61487.2024.10580124). Tabassum, H. *et al.* (2024) “Performance Analysis on Enhancing Security Through Object Detection using YOLO G,” *2024 2nd International Conference* *on Advances in Computation, Communication and Information Technology* *(ICAICCIT)*. *2024 2nd International Conference on Advances in Computation,* *Communication and Information Technology (ICAICCIT)*, Faridabad, India: IEEE, pp. 852–857. Available at: [https://doi.org/10.1109/ICAICCIT64383.2024.10912375](https://doi.org/10.1109/ICAICCIT64383.2024.10912375). Veranyurt, O. and Sakar, C.O. (2023) “Concealed pistol detection from thermal images with deep neural networks,” *Multimedia Tools and Applications*, 82(28), pp. 44259–44275. Available at: [https://doi.org/10.1007/s11042-023-](https://doi.org/10.1007/s11042-023-) 15358-1. Wu, P. *et al.* (2025) “A Review on Research and Application of AI-Based Image Analysis in the Field of Computer Vision,” *IEEE Access*, 13, pp. 76684–76702. Available at: [https://doi.org/10.1109/ACCESS.2025.3565300](https://doi.org/10.1109/ACCESS.2025.3565300). Yin, C. *et al.* (2025) “SBEW-YOLOV8: a small object detection algorithm for autonomous driving based on multi-scale feature fusion,” *The Journal of* *Supercomputing*, 81(10), p. 1125. Available at: [https://doi.org/10.1007/s11227-](https://doi.org/10.1007/s11227-) 025-07577-0.

Zhang, J. and Du, Y. (2023) “Pedestrian detection with Bi-Directional Feature Pyramid and Channel-Spatial Attention Modules,” *2023 International* *Conference on Advances in Electrical Engineering and Computer Applications* *(AEECA)*. *2023 International Conference on Advances in Electrical* *Engineering and Computer Applications (AEECA)*, Dalian, China: IEEE, pp. 695–699. Available at: [https://doi.org/10.1109/AEECA59734.2023.00128](https://doi.org/10.1109/AEECA59734.2023.00128). Zheng, Z. *et al.* (2020) “Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression,” *Proceedings of the AAAI Conference on Artificial* *Intelligence*, 34(07), pp. 12993–13000. Available at: [https://doi.org/10.1609/aaai.v34i07.6999](https://doi.org/10.1609/aaai.v34i07.6999). Zhu, J. *et al.* (2025) “Underwater Side-Scan Sonar Target Detection: An Enhanced YOLOv11 Framework Integrating Attention Mechanisms and a Bi-Directional Feature Pyramid Network,” *Journal of Marine Science and Engineering*, 13(5),

p. 926. Available at: [https://doi.org/10.3390/jmse13050926](https://doi.org/10.3390/jmse13050926).
Zhu, Y. *et al.* (2025) “YOLO-WildASM: An Object Detection Algorithm for Protected Wildlife,” *Animals*, 15(18), p. 2699. Available at: [https://doi.org/10.3390/ani15182699](https://doi.org/10.3390/ani15182699).