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

Uzun sürdüğü için `tmux` ya da `nohup` içinde çalıştırın:

```bash
tmux new -s train './run.sh configs/decider_judge3.yaml 2>&1 | tee run.log'
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
6. Config dosyasını, başlangıç ölçümünü, `train_log.jsonl`'u ve bütün logları hedef model repo'sunun `training/` klasörüne yükler.

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
- H200 (141 GB): `micro_tokens: 32768`.
- `decider_judge3`: ~24K satır, ~11M token; başlangıç ölçümü ~15 dk, eğitim tahminen 1 saatin altında.
- Disk: başlangıç modeli ~8 GB, merge edilmiş çıktı ~8 GB, Python ortamları ~7 GB. En az 40 GB boş yer bırakın.
