# Agent talimatları: bu sunucuda decider eğitimi

Bu repo, decider karar modelinin bir LoRA turunu bir GPU sunucusunda çalıştırır. Bu dosya, turu başlatan, izleyen ve sonucunu
raporlayan agent içindir. Kullanıcı logları başka bir yere taşıyamıyor; senin işin, turu güvenle çalıştırmak, her aşamada aşağıdaki
kontrolleri yapmak ve sonunda `summarize.py` çıktısını kullanıcıya vermek. Kullanıcı o çıktıyı modeli geliştiren tarafa iletecek.

## Bağlam

- **decider**: metin üretmeyen bir karar modeli. Bir `state` (metin ya da JSON) ve sorular alır; her soruya tek ileri geçişte
  kalibre bir olasılık verir. Soru türleri: `noul` (evet/hayır), `choice` (seçenekler), `score` (sıralı seviyeler).
- **Model zinciri**: `Mapika/decider-4b` → `decider-4b-tr` (Türkçe) → `-rag` → `-judge` → `decider-4b-tr-judge-ttl` (başlangıç
  modeli) → bu tur.
- **Bu modelin iki üretim görevi** (canlı testte LLM'e çok yakın çalıştı, hız ~1,7 kat):
  1. **Relevance judge** (`relevant`, noul): getirilen belge sorunun cevabına doğrudan katkı sağlıyor mu? Kısmi cevap yeterli;
     bilgi sorudaki varlığa, kuruma, özelliğe ve döneme ait olmalı.
  2. **Cache TTL** (`ttl`, choice): realtime (5 dk), hourly (6 sa), daily (3 gün), stable (30 gün).
- **Bu turun amacı (`configs/decider_judge4.yaml`)**:
  - TTL tanım düzeltmesi: fiyat, pazar payı, gelir, piyasa verisi ya da dönemsel istatistik veren haberler artık **daily**
    (önceki model bunlara stable diyordu; canlı testte üretimdeki LLM ile 23 ayrışmanın hepsi buydu).
  - RAGBench-tr: finans raporu, teknik kılavuz, tıbbi makale ve ürün desteği belgeleri (önceki veri Vikipedi ağırlıklıydı).
  - Yargıcın önceki davranışı bozulmamalı (varlık, kurum, dönem negatifleri; kısmi kanıtı kabul).
- **Girdi biçimi sabit**: state alanları `question`, `document`, `today` sırasıyla; `today` en sonda. Belge `[tarih] başlık`, boş
  satır, gövde. Bunlar eğitim verisinde zaten böyle; değiştirme.

## Kurallar

- **GPU**: yalnızca kullanıcının verdiği kart (`CUDA_VISIBLE_DEVICES`, şu an `5`). O kartta kullanıcının vLLM container'ları
  çalışıyor. Docker'a, vLLM'e, başka süreçlere ve servislere **dokunma**; hiçbir süreci öldürme, hiçbir servisi yeniden başlatma.
- **Yükleme yok**: bu sunucudan Hugging Face'e upload yapılamıyor. Her zaman `PUSH=0`. Hugging Face'ten indirme çalışıyor.
- **Sistem değişikliği yok**: sürücü, CUDA, sistem Python'u, sistem paketleri. Script her şeyi repo klasörüne kurar.
- **Çalışırken dosya değiştirme**: `run.sh` ya da `bench.sh` çalışırken `git pull` yapma, script'leri düzenleme (bash script'i
  çalışırken okur; değişiklik çalışan süreci bozar).
- **Veriye dokunma**: eğitim ve test verisi Hub'dan iner, olduğu gibi kullanılır. Bench verisi (`bench/`) **yalnızca
  değerlendirme** içindir, hiçbir zaman eğitime girmez.
- **Hiperparametreleri değiştirme**, aşağıdaki sorun giderme adımları dışında.
- Disk: en az 40 GB boş yer (`df -h .`).

## Başlatma

```bash
cd decider-training
git pull                                          # yalnızca hiçbir şey çalışmıyorken
nvidia-smi -i 5                                   # boş bellek >= 40 GB olmalı
export HF_TOKEN=<kullanıcının token'ı> CUDA_VISIBLE_DEVICES=5 MIN_FREE_MIB=40000
tmux new -d -s train 'PUSH=0 ./run.sh configs/decider_judge4.yaml 2>&1 | tee run.log'
```

`run.sh` sırasıyla: uv ve ortamlar → causal-conv1d derlemesi → başlangıç ölçümü → LoRA eğitimi, export, kalibrasyon, son ölçüm
→ `results/` altına paketleme → **bench** (başlangıç ve yeni model, system-one-bench-tr). Yarıda kalırsa aynı komut kaldığı
yerden devam eder. Eğitimi temiz durdurmak: `touch judge4_run/STOP` (checkpoint alır ve durur).

## Aşama aşama neye bakılır

| Aşama | Log | Beklenen | Sorun belirtisi |
|---|---|---|---|
| Ortam | `logs/run_gpu.log` | `torch 2.14.1+cu126 CUDA 12.6 GPU True` (iki satır) | `GPU False`, `0 active drivers` |
| causal-conv1d | `logs/causal_conv1d.log` | `causal_conv1d 1.7.0 ...` | `not built ...` kabul edilebilir (eğitim yavaşlar), hata değil |
| Başlangıç ölçümü | `logs/decider_judge4_eval_start.log` | `on 23,056 rows`; ilerleme satırlarında `accuracy so far` ~0,85-0,90 | < 0,75: model ya da ortam yanlış |
| Eğitim verisi | `logs/decider_judge4_lora.log` | kaynak başına satırlar (ör. `ragbench_tr: 13,337`, `ttl_tr: ~3,700`), `items train: ~37,000 rows` | bir kaynak 0 satır |
| Eğitim | aynı log, 20 adımda bir `train epoch=1 step=...` | `ce` ilk ~60 adımda 0,2'nin altına iner ve 0,03-0,30 arasında kalır; `acc` 0,90-0,99; replay adımlarında `kl` 0,2-0,4 (`kl=0.0000` o aralıkta replay olmadığı anlamına gelir, normal) | `ce` sürekli > 0,5 ya da `nan`; `out of memory` |
| Checkpoint | aynı log | 30 dakikada bir `checkpoint: step N` | |
| Export | aynı log | `validation ... val_slot_accuracy` ~0,93-0,98; `temperatures ... {'choice': ..., 'noul': ...}`; `exported` | |
| Son ölçüm | aynı log | `eval: accuracy ...`, sonra `EXIT 0` | `EXIT` 0 değil |
| Paket | `results/` | `decider_judge4_reports.tar.gz`, `decider_judge4_model.tar` (~8 GB) | |
| Bench | `logs/bench_run.log` | model başına `=== ...`, görev başına `... test: N items in Ns` | `Traceback` |

Hız: `tok_per_s` ve `eta_h` değerlerini not et (kart vLLM ile paylaşıldığı için önceden kestirilemiyor; 5090'da ~4.000 tok/s idi).
Eğitim ~37K satır, ~1-1,5 saat; bench iki model için ~1-2 saat.

## Sorun giderme

- **`GPU False` / `RuntimeError: 0 active drivers`**: sürücü 570 (CUDA 12.8); PyPI'daki torch CUDA 13 ister. `run.sh` uv'nin
  `UV_TORCH_BACKEND=auto` ayarıyla doğru torch'u seçer. Hâlâ False ise: `export UV_TORCH_BACKEND=cu126`, sonra
  `rm -rf .uv-cache/environments-v2/decider-* logs/causal_conv1d.log` ve aynı komutla yeniden başlat.
- **causal-conv1d derlenemedi**: çoğunlukla `nvcc` yok ya da sürümü uyuşmuyor. Eğitim onsuz çalışır. Yeniden denemek için
  `rm logs/causal_conv1d.log` (bir sonraki çalıştırmada derler).
- **`out of memory`**: önce `nvidia-smi -i 5` ile vLLM'in belleği artmış mı bak. Eğitimi durdur, `configs/decider_judge4.yaml`
  içinde `micro_tokens: 16384` → `8192` yap, aynı komutla yeniden başlat (checkpoint'ten devam eder). Bunu kullanıcıya bildir.
- **vLLM'in cevap süreleri çok uzadı (kullanıcı söylerse)**: `touch judge4_run/STOP`; trafik azalınca aynı komutla devam.
- **Disk doldu**: yalnızca bu repo içinde sil: `.hf-home/hub` (indirilen modeller; yeniden iner). Başka bir şeyi silme.
- **Bench HateDay'de ya da bir görevde hata**: `./bench.sh` aynı argümanlarla yeniden çalıştırılabilir; biten görevler atlanır.
  Bench'i tek başına çalıştırmak için:
  `./bench.sh hayriyigit/decider-4b-tr-judge-ttl@8b69f8a8dead233b95cc82e960a507a53a68af2c judge4_run/model`

## Sonuç kontrolleri

Tur bitince (`EXIT 0` ve bench bitti):

```bash
python3 summarize.py configs/decider_judge4.yaml
```

Çıktıda şunlara bak ve kullanıcıya ayrıca belirt:

- **TTL**: `TTL daily` yükselmeli (bu turun hedefi); `TTL stable` 0,95'in altına düşmemeli; `realtime` ve `hourly` test sayfası
  az (17-18), birkaç puanlık oynama gürültüdür.
- **RAGBench** (`ragbench_tr/...` satırları): yeni > başlangıç beklenir. Hangi alanların (`finqa`, `techqa`, ...) en zayıf kaldığını yaz.
- **Gerileme**: `<-- dropped` işaretli (≥ 3 puan düşen) her satırı listele. Beklenen eşikler: NoMIRACL relevant ≥ 0,71,
  NoMIRACL not relevant ≥ 0,80, SQuAD-tr unanswerable ≥ 0,78, `synth v2 other entity / institution` ≥ 0,95.
- **Bench**: yeni modelin her alanı başlangıç modeliyle karşılaştır; 2 puandan fazla düşen alanı belirt. Tabloda bench'in kendi
  yayımladığı modeller referans olarak durur.

## Kullanıcıya rapor

Kullanıcıya şunları ver (logların tamamını değil):

1. `summarize.py` çıktısının tamamı, olduğu gibi.
2. Toplam süre, ortalama `tok_per_s`, causal-conv1d derlendi mi.
3. Karşılaştığın her hata ve ne yaptığın.
4. Yukarıdaki sonuç kontrollerinden çıkan uyarılar.

`results/decider_judge4_reports.tar.gz` ve `results/s1bench_predictions.tar.gz` küçük dosyalardır; kullanıcı isterse bunları da
taşıyabilir. `results/decider_judge4_model.tar` (~8 GB) eğitilen modeldir; Hub'a başka bir makineden yüklenecek.
