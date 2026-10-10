# decider training

Bu repo [decider](https://huggingface.co/Mapika/decider-4b) modelleri için tek bir LoRA turunu, boş bir GPU sunucusunda baştan sona
çalıştırır. Eğitim verisi Hugging Face'teki private bir dataset repo'sundan iner; eğitilen model, değerlendirme raporu ve
loglar hedef model repo'suna yüklenir. Sonuçlar oradan okunur, sunucuya SSH gerekmez.

## Çalıştırma

```bash
git clone https://github.com/hayriyigit/decider-training.git && cd decider-training
export HF_TOKEN=hf_...          # data ve başlangıç modeli için read, hedef model repo'su için write yetkisi
./run.sh configs/decider_judge3.yaml
```

**Sunucudan Hugging Face'e yükleme yapılamıyorsa** `PUSH=0` ile çalıştırın. Bu modda hiçbir şey yüklenmez; HF'ten yalnızca veri
ve başlangıç modeli indirilir. Sonuçlar `results/` altında paketlenir:

- `results/<config>_reports.tar.gz` (birkaç MB): başlangıç ve son değerlendirme raporları, `train_log.jsonl`, config ve bütün loglar.
  Sonuçları görmek için yalnızca bu dosya yeterli.
- `results/<config>_model.tar` (~8 GB): merge edilmiş, kalibre edilmiş model (`decider_config.json` ve `decider/` kodu dahil).

```bash
PUSH=0 ./run.sh configs/decider_judge3.yaml
```

**Başka servislerin çalıştığı bir sunucuda:**

- Script her şeyi bu klasörün içine kurar ve indirir: uv, Python ortamları (torch dahil), causal-conv1d derlemesi, Hugging Face
  cache'i (`.uv-*`, `.hf-home`). Sistem Python'una, CUDA'ya, sürücüye ya da başka ortamlara dokunmaz; klasörü silmek her şeyi kaldırır.
- `CUDA_VISIBLE_DEVICES` ile **boş bir GPU** seçilmesi zorunludur. Script, seçilen kartta en az 70 GB boş bellek yoksa başlamaz
  (`MIN_FREE_MIB` ile değiştirilebilir). Bir inference engine'in (vLLM vb.) belleğini ayırdığı karta eğitim koyarsanız ikisi de
  bellek hatası alır.
- causal-conv1d derlemesi yalnızca CPU kullanır; varsayılan 8 paralel iş ve düşük öncelikle (`nice`) çalışır (`MAX_JOBS` ile değiştirilebilir).

Uzun sürdüğü için `tmux` ya da `nohup` içinde çalıştırın:

```bash
export CUDA_VISIBLE_DEVICES=3     # boş bir GPU: nvidia-smi
tmux new -s train 'PUSH=0 ./run.sh configs/decider_judge3.yaml 2>&1 | tee run.log'
```

Gereken tek şey NVIDIA GPU, CUDA toolkit (`nvcc`, causal-conv1d derlemesi için) ve internet. `uv` yoksa script kurar; Python
ortamlarını script başlıklarındaki (PEP 723) bağımlılıklardan kendisi oluşturur.

## run.sh ne yapar

1. `uv`'yi kurar (yoksa).
2. Config'teki `data.train` ve `data.test` dosyalarını `hayriyigit/decider-judge-ttl-mix` dataset repo'sundan indirir
   (`DATA_REPO` ile değiştirilebilir).
3. `decider_eval` ve `decider_lora` ortamlarını kurar; `causal-conv1d`'yi bu GPU'nun compute capability değeriyle derler
   (H200: 9.0, RTX 5090: 12.0). Derleme ~15 dakika sürer ve eğitimi ~1,7 kat hızlandırır; başarısız olursa eğitim yine çalışır.
4. Başlangıç modelini test split'inde ölçer (`<config>_eval_start.json`).
5. `scripts/decider_lora.py`: items → train → export (LoRA merge + temperature kalibrasyonu) → evaluate → push.
6. Config dosyasını, başlangıç ölçümünü, `train_log.jsonl`'u ve bütün logları hedef model repo'sunun `training/` klasörüne yükler
   (`PUSH=0`: hepsini `results/` altında paketler, push adımını atlar).

Yarıda kalırsa aynı komutu tekrar çalıştırın: biten adımlar atlanır, eğitim son checkpoint'ten devam eder (checkpoint her 30
dakikada bir). Eğitimi temiz durdurmak için `touch <workdir>/STOP`.

## Dosyalar

- `run.sh`: yukarıdaki adımlar
- `scripts/decider_lora.py`: LoRA eğitimi, export, değerlendirme, push
- `scripts/decider_eval.py`: bir decider modelini laya formatındaki bir test parquet'inde ölçer
- `configs/decider_judge3.yaml`: beşinci tur. `hayriyigit/decider-4b-tr-judge-ttl` üzerine, TTL tanım değişikliği (fiyat ve
  piyasa haberleri daily); hedef `hayriyigit/decider-4b-tr-judge-ttl2`

## Süre ve bellek

- RTX 5090 (32 GB): `micro_tokens: 16384`, ~4.000 token/s.
- H200 (141 GB): kart boşsa `micro_tokens: 32768`; başka servislerle paylaşılan kartta 16384 (~25 GB) ve `MIN_FREE_MIB=40000`.
- `decider_judge3`: ~24K satır, ~11M token; başlangıç ölçümü ~15 dk, eğitim tahminen 1 saatin altında.
- Disk: başlangıç modeli ~8 GB, merge edilmiş çıktı ~8 GB, Python ortamları ~7 GB. En az 40 GB boş yer bırakın.

## Benchmark: system-one-bench-tr

[system-one-bench-tr](https://github.com/UlkuTuncerKucuktas/system-one-bench-tr), 66 Türkçe karar görevi (choice / noul / score)
içeren bir benchmark. `bench.sh` decider modellerinin cevaplarını bench'in sonuç formatında üretir; puanlama GPU istemez ve
başka bir makinede `s1bench score` ile yapılır.

```bash
export HF_TOKEN=hf_... CUDA_VISIBLE_DEVICES=5 MIN_FREE_MIB=20000
tmux new -s bench './bench.sh hayriyigit/decider-4b-tr-judge-ttl@8b69f8a8dead233b95cc82e960a507a53a68af2c judge3_run/model 2>&1 | tee bench.log'
```

- Bench'in görev dosyaları `hayriyigit/s1bench-tr-data`'dan (private) iner; sunucunun bench'in kaynaklarına (GitHub, Zenodo,
  Kaggle) erişmesi gerekmez.
- Model bir Hub repo'su (`repo@revision`) ya da yerel bir klasör olabilir (ör. `run.sh`'nin çıktısı `judge3_run/model`).
- `MAIN=1`: yalnızca bench'in 26 ana görevi (daha hızlı).
- Çıktı: `results/s1bench_predictions.tar.gz`. Bunu getirin, puanlama ve diğer modellerle karşılaştırma oradan yapılır.
- Eğitimle aynı anda çalıştırmayın; `run.sh` ya da `bench.sh` çalışırken `git pull` yapmayın.

## Sorun giderme

- **`RuntimeError: 0 active drivers` ya da torch GPU görmüyor:** PyPI'daki torch CUDA 13 ile derlenir ve NVIDIA sürücüsü 580+ ister. `run.sh`, uv'nin
  `UV_TORCH_BACKEND=auto` ayarıyla sürücüye uyan torch'u (ör. `cu126`) seçer ve eğitimden önce torch'un GPU'yu gördüğünü kontrol eder.
  Elle seçmek için: `export UV_TORCH_BACKEND=cu126`. Ayrıntı `logs/run_gpu.log`'da.
- **`No module named causal_conv1d`:** derleme başarısız oldu (çoğunlukla `nvcc` yok ya da sürümü torch'unkiyle uyuşmuyor). Eğitim onsuz
  çalışır, yalnızca daha yavaş. Yeniden denemek için `rm logs/causal_conv1d.log`.
