# Catatan untuk BAB IV–V

Poin-poin ini dipindahkan dari BAB III saat pemadatan. Isinya benar, tetapi lebih tepat dibahas saat angka hasil ditampilkan (BAB IV) atau sebagai keterbatasan (BAB V).

## Dari 3.4 — Deviasi arsitektur
- Kanal *neck* 64 (bukan 256 seperti Fung et al., 2024) dipilih agar setiap *node* SODT memiliki lebih sedikit bobot yang diestimasi dari data yang sama. Konsekuensinya, nilai AP absolut **tidak dapat dibandingkan langsung** dengan tabel Fung et al. (2024), meskipun topologi, *loss*, *Soft*-NMS, dan skema multi-skala sama.
- *Bottleneck SF Attention* = 32, karena Fung et al. (2024) tidak menyebutkan ukurannya.

## Dari 3.5 — Epoch
- 15 *epoch*, bukan 12 seperti Fung et al. (2024). Diungkapkan sebagai penyimpangan dan tidak dilatih ulang.

## Dari 3.8 — Hiperparameter SODT
- Seluruh hiperparameter (kedalaman, λ, α, bobot kelas) ditetapkan di awal mengikuti praktik Hada et al. (2024) dan Kairgeldin dan Carreira-Perpiñán (2025), **tidak dituning terhadap data uji**.
- α kecil (0,15) dipilih agar bobot tidak tersebar ke seluruh grid, sehingga *heatmap* tidak menyebar. λ besar (10) menjaga agar setiap *node* tetap *sparse*.
- Bobot kelas diberikan pada kelas dengan *recall* terlemah pada *teacher* (*short*, *spur*, *open*, *pinhole*).

## Dari 3.10.2 — *Faithfulness* tingkat jalur
- Himpunan RoI SODT (proposal RPN, maksimal 8 per citra) dan Grad-CAM (deteksinya sendiri dengan IoU ≥ 0,5 terhadap GT) **tidak identik**. Proporsi penghapusan sama, jadi perbandingannya bersifat **arah** (mana yang lebih tinggi), bukan selisih yang presisi.
- *Sufficiency* dilaporkan tetapi tidak dijadikan dasar kesimpulan apabila kontrol acaknya setara. Menghapus sebagian besar sel membuat masukan keluar dari distribusi, apa pun sel yang dipilih.
- *Deletion*/*Insertion* AUC Grad-CAM diukur atas probabilitas kelas, sedangkan AUC SODT hanya tersedia per *node* atas skor keyakinan *routing*. Keduanya besaran berbeda, jadi jangan dijejerkan dalam satu tabel.
- Arti kontrol: *shuffled weights* tinggi tetapi di bawah *exact* berarti struktur bobot membawa sinyal; *activation only* rendah berarti hasilnya bukan sekadar mengikuti piksel terang.

## Dari 3.10.3 — Lokalisasi
- Proposal longgar (IoU 0,05–0,35) dipakai karena pada proposal rapat *ground truth* hampir menutupi seluruh grid, sehingga peta acak pun tampak tepat (metrik jenuh). Subset dengan cakupan GT < 50% grid dilaporkan terpisah.
- Lokalisasi adalah **pemeriksaan batas** ("penjelasan tidak acak dan berada di sekitar cacat"), bukan klaim utama. *Heatmap* SODT menjelaskan daerah yang **ditimbang keputusan**, bukan lokasi cacat.

## Dari 3.10.4 — *Faithfulness* per *node*
- *Ceiling*: *flip rate* maksimum yang dapat dicapai peta SODT bila seluruh kontribusi *node* dihapus. Tercapai hanya bila tanda bias berlawanan dengan arah keputusan. Bandingkan *exact* dengan *ceiling*, bukan dengan acak.
- *Support fraction*: proporsi posisi di dalam proposal yang memiliki kontribusi. *Ceiling* hanya ketat bila nilainya ≤ 0,5 (anggaran penghapusan).
- Kemiripan kosinus antar-*node* berurutan: nilai rendah menunjukkan setiap langkah menimbang daerah berbeda.
- Hasil per *node* **tidak dirata-ratakan** untuk dibandingkan dengan hasil tingkat jalur atau Grad-CAM, karena targetnya berbeda (tanda *node* vs label akhir).

## Dari 3.9.3 — Keterbatasan *heatmap*
- Perhitungan eksak berhenti pada peta fitur *neck*, karena *backbone* di bawahnya non-linear. Satu posisi P2′ setara petak 4×4 piksel, tetapi *receptive field*-nya jauh lebih besar, sehingga ketelitiannya tingkat daerah, bukan piksel.
- Penjumlahan antarkanal dan nilai mutlak pada Persamaan 3.7 adalah pilihan tampilan; sifat eksak berlaku sebelum langkah tersebut (Persamaan 3.6).
- Koordinat *bounding box* tetap berasal dari kepala regresi (MLP) *Faster* R-CNN, jadi klaim *faithful* hanya berlaku untuk keputusan kelas.
