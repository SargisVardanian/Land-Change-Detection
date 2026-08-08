# QCPR TemporalSigLIP: полный текущий разбор

Дата отчёта: 2026-08-08  
Ветка: `codex/qcpr-siglip2-temporal-training`  
Код, на котором выполнены последние пилоты и source audit: `5e49864b33f170a9f020a70b4dfa6dd1af9f507d`
Текущая опубликованная голова ветки после config/launcher/static-analysis/report hardening: `6223d172307fce4a09b739ebc5442d822e437534`
Исторические пилоты намеренно не переименованы: их `evaluated_code_sha` остаётся SHA, на котором они реально выполнялись.
Локальный, удалённый и cluster SHA совпадают; worktree чистый.
PR: #7, draft  
Финальное Dataset-Agent handoff отсутствует; поэтому это отчёт о проверенном bounded pilot, а не о готовом production training.

## 1. Короткий вердикт

На текущей ветке реализован прямой temporal retrieval трек на SigLIP2:

```text
T1, T2, ...
  -> общий SigLIP2 vision tower для каждого frame
  -> 256 native patch tokens × 768
  -> два temporal Transformer blocks + PAIR token
  -> один нормализованный 768-D pair vector

text
  -> тот же SigLIP2 text tower
  -> один нормализованный text vector

scaled cosine(pair, text)
  -> full-gallery retrieval
```

Stage A и Stage B дали положительное bounded улучшение на одной и той же common gallery. Лучший проверенный checkpoint сейчас — Stage B, step 1024. Однако:

- это один pilot seed, а не финальное научное сравнение;
- source identity всё ещё почти идеально восстанавливается из visual embeddings;
- query-conditioned карта в активном TemporalSigLIP является диагностической, но не участвует в retrieval score;
- semantic/localized/long-series final handoff отсутствует;
- полноценное multi-seed обучение и финальный test ещё не разрешены.

Иными словами: retrieval-механизм уже технически работает и улучшился, но claim «модель научилась устойчивой query-conditioned soft segmentation на разнообразном датасете» пока не доказан.

## 2. Какие модели фактически внедрены

### Активная модель для дообучения

| Компонент | Что используется |
|---|---|
| Vision | `google/siglip2-base-patch16-256` |
| Runtime class | `SiglipModel` |
| Local checkpoint | `/mnt/weka/svardanyan/rs_change_project/models/siglip2-base-patch16-256` |
| Weights SHA256 | `6125cacc01fa93bd98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863` |
| Hidden size | 768 |
| Spatial tokens | 256 = 16×16 |
| Text tokens | 64 |
| Tokenizer | `GemmaTokenizer`, vocab 256000, BOS=2, EOS=1, PAD=0 |
| Input resolution | 256×256 в текущем контракте |

Один SigLIP2 vision tower повторно обрабатывает все временные кадры с общими весами. Один SigLIP2 text tower кодирует запрос. Это не два независимых image encoders и не отдельный captioning model.

### Модель для сравнения

`GeoRSCLIP` (`Zilun/GeoRSCLIP`, revision `4920188e6eba4e711ef9848cfd7cb77e874ee33f`, checkpoint `ckpt/RS5M_ViT-B-32.pt`) использовался как внешний frozen/temporal-head baseline. Он не является активной основной моделью и не заменяет SigLIP2.

### Что не является активным в этом треке

Прежняя линия `UniverSat + Jina v5` и Model-v3 evidence scaffold сохранены в истории проекта, но текущий TemporalSigLIP напрямую их не использует. В этой ветке нет активного Jina adapter, UniverSat adapter, второго VLM, обязательного reranker или отдельной segmentation head.

Это важное различие: нельзя описывать текущий результат как совместно обученный UniverSat/Jina результат.

## 3. Как изменена архитектура

### Temporal adapter

Для входа `[B,T,256,768]` добавляются:

- learned PAIR token;
- learned frame/temporal embeddings;
- zero-initialized projection для delta-time features;
- два одинаково структурированных pre-LN temporal Transformer blocks;
- MHA и FFN с ratio 4.

PAIR token и усреднённые frame tokens формируют начальное состояние. После двух блоков берётся первый token и L2-нормализуется. На выходе получается query-independent 768-D pair embedding, пригодный для ANN/full-gallery индексации.

Текущий код допускает `2 ≤ T ≤ 8`, но проверенный реальный контракт — `T=2`. При `T>2` математический интерфейс есть, однако отдельный real long-series benchmark ещё не запускался.

### Retrieval score

```text
score(q, pair) = exp(clamped(logit_scale))
                 × cosine(text_embedding, pair_embedding)
```

Это один глобальный scalar score. Новая spatial карта не подмешивается в него скрытым обходным путём.

### Soft map

В активном треке `TemporalSoftChangeMap` вычисляет диагностическую map приблизительно так:

```text
delta_tokens = LayerNorm(T2_tokens - T1_tokens)
map_logits   = dot(delta_tokens, text_embedding)
soft_map     = sigmoid(map_logits / temperature)
```

Она query-conditioned в смысле зависимости от текста, но retrieval loss через неё не проходит. Поэтому сейчас это **query-conditioned temporal similarity map**, а не обученная segmentation probability map.

## 4. Что обучается

### Stage A — замороженные башни

Обучаемые параметры: **14,184,193**.

Обучаются только:

- PAIR token;
- temporal embeddings;
- delta-time projection;
- два temporal Transformer blocks;
- trainable `logit_scale`.

Vision и text towers SigLIP2 полностью frozen.

### Stage B — частичная адаптация

Обучаемые параметры: **50,216,449** из **375,234,050** параметров двух SigLIP2 towers.

Добавляются:

- последние два SigLIP2 vision Transformer blocks;
- последние два SigLIP2 text Transformer blocks;
- разрешённые final norms/projections;
- все Stage-A temporal parameters;
- `logit_scale`.

Ранние backbone blocks остаются frozen. Stage B использует свежий optimizer/scheduler; это не продолжение старого optimizer state.

### Learning-rate contract

| Группа | LR |
|---|---:|
| temporal adapter | 1e-4 |
| последние vision/text blocks | 5e-6 |
| final norms/projections | 2e-5 |
| logit scale | 1e-5 |

Общие настройки: AdamW, weight decay 0.05, без decay для bias/norm/embedding/PAIR/logit scale, warmup 5%, cosine decay, gradient clip 1.0, BF16.

## 5. Как изменён процесс обучения

Старый риск «градиентная аккумуляция увеличивает только время, но не negative set» был устранён через GradCache-style recomputation:

- physical microbatch: 32 пары;
- logical physical batch: 128 пар;
- 2 captions на пару;
- 256 текстовых запросов;
- одна общая score matrix `256×128`;
- 4 feature-recompute microbatches по 32;
- hard-negative mining отключён;
- early stopping не используется в fixed-step сравнении.

Stage A: 512 optimizer steps.  
Stage B: ещё 1536 steps, итоговый global step 2048.  
Параметры экспозиции считаются по уникальным физическим парам, а не по формальному числу эпох.

Primary objective — один симметричный multi-positive listwise scalar:

```text
L = 0.5 * text_to_pair_listwise_loss
  + 0.5 * pair_to_text_listwise_loss
```

Grade-3 exact relevance является положительной. Grade-1/2 ambiguity в exact режиме не превращается в positive. Не добавлялись отдельные direction/object/count/mask/segmentation losses.

## 6. Реальные технические проверки

### Real SigLIP2 smoke

Job 209877 завершился `0:0`:

- реальные T1/T2 images;
- visual tensor `[8,2,256,768]`;
- text tensor `[16,64,768]`;
- 8 optimizer steps;
- score matrix `16×8`;
- BF16;
- checkpoint roundtrip PASS;
- peak allocated примерно 1.46 GiB, reserved 1.70 GiB;
- masks не открывались;
- multi-positive runtime не был реально проверен, потому что exact manifest не дал подходящего batch.

### Stage A

- Job 209853;
- 512/512 steps;
- wall 810.0 s;
- peak allocated 8.840 GiB, reserved 10.006 GiB;
- checkpoint SHA256 `7d634f18e8bac645a302cadbe7279429f29dfe7effa01dc406b8d48c737697db`.

### Stage B

- Job 209871;
- global step 512 → 2048;
- 1536 additional steps;
- wall 4344.3 s, примерно 72.4 минуты;
- peak allocated 9.186 GiB, reserved 10.668 GiB;
- final checkpoint SHA256 `fb8da9e39e2238a7800e11408d10306577bf65929f1e11e0c44a824ac0d65742`.

## 7. Текущие retrieval-метрики

Общая common gallery: **1,928 physical pairs**, **6,310 exact queries**. Ranking files используют одну и ту же пару/query order.

| Модель/checkpoint | MRR | R@1 | R@5 | R@10 | R@50 | R@100 | mean rank | median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TemporalSigLIP Stage A, 512 | 0.04617 | 0.01569 | 0.05309 | 0.09398 | 0.30063 | 0.45674 | 242.70 | 117 |
| Stage B, 1024 | **0.05771** | **0.02076** | 0.06941 | 0.11696 | **0.34834** | **0.52060** | **213.04** | **93** |
| Stage B, 2048 | 0.05669 | 0.01870 | **0.07322** | **0.12203** | 0.33677 | 0.49810 | 239.61 | 101 |

Stage B step 1024 относительно Stage A:

- MRR: `+0.01153`, или примерно `+25.0%`;
- R@1: примерно `+32.3%`;
- R@10: примерно `+24.5%`;
- median rank: `117 → 93`;
- mean rank: `242.70 → 213.04`.

Step 2048 даёт лучший R@10, но немного хуже MRR, чем step 1024. Это аргумент выбирать checkpoint по complete development ranking, а не всегда брать последний шаг.

### По источникам на Stage B, step 2048

| Источник | Queries | MRR | R@10 | median rank |
|---|---:|---:|---:|---:|
| LEVIR-MCI | 3335 | 0.06333 | 0.13583 | 95 |
| SECOND-CC | 2975 | 0.04924 | 0.10655 | 106 |

LEVIR заметно легче для текущего representation. Это не обязательно баг, но показывает domain gap.

### Сравнительный GeoRSCLIP baseline

На общей gallery frozen GeoRSCLIP получил MRR около `0.00368`, R@10 около `0.00301`. Его bounded temporal-head pilot на 256 steps достиг MRR `0.00816`, R@10 `0.01363`. Это только сравнительная линия, не основание для claim SOTA.

Старый QCPR R1 с другой архитектурой имел MRR около `0.0229`; его нельзя использовать как строгую causal comparison без полного одинакового backbone/data/training contract.

## 8. Source-shortcut audit

Evaluation-only job 209878 завершился `0:0` на gpu01. Проверка строила реальные 1,928 pair embeddings и 5-fold stratified ridge probe LEVIR против SECOND.

| Checkpoint | Source-probe accuracy | within-source cosine | cross-source cosine |
|---|---:|---:|---:|
| Stage A | 0.99948 | 0.25949 | -0.20547 |
| Stage B, 1024 | 0.99845 | 0.18089 | -0.07618 |
| Stage B, 2048 | 0.99844 | 0.11221 | -0.05652 |

Вывод двойной:

1. Stage B не усилил source separability: delta probe accuracy `−0.00104` относительно Stage A.
2. Но source identity уже почти идеально линейно читается из embeddings. Поэтому риск shortcut остаётся высоким; «не стало хуже» не означает «доменный shortcut исчез».

Этот риск нужно закрывать source-balanced training, cross-source held-out tests, leave-one-source-out diagnostics и простым source probe на каждом финальном checkpoint.

## 9. Collapse и embedding geometry

- pair effective matrix rank: 768;
- text rank: 655 на Stage A и 726–731 на Stage B;
- pair norm mean: 1.0;
- text norm около 1.0;
- явного embedding collapse не наблюдается.

Техническая оговорка: текущий отчёт называет это `effective_rank`, но реализация использует `torch.linalg.matrix_rank`, то есть это full numerical rank с текущим tolerance, а не participation-ratio effective rank. Для финальной статьи стоит дополнить spectrum-based effective rank.

## 10. Что с разноразмерными картинками

### Что уже работает

- синхронная обработка T1/T2 одним preprocessing контрактом;
- patch16 tokenization;
- T=2 проверен реально;
- интерфейс допускает `2 ≤ T ≤ 8`;
- timestamps могут передаваться в temporal adapter;
- sequence length не зашит только на два отдельных encoder-а.

### Что пока не решено полностью

Текущий активный SigLIP2 checkpoint — fixed-resolution `256×256`, поэтому native grid всегда `16×16` и 256 tokens. Это не полноценная resolution-agnostic модель.

Для первого честного расширения нужен scale-aware data/view contract:

1. хранить raw image и native metadata;
2. применять одинаковый resize/pad/crop к T1 и T2;
3. фиксировать transform hash;
4. добавлять GSD и physical footprint в manifest;
5. для крупных сцен использовать одинаковые temporal tiles, а не растягивать весь регион в 256;
6. валидировать small-object recall отдельно на 256/384/footprint views.

Позже можно добавить multi-view pooling или patch-grid variant, но нельзя незаметно смешивать grid 16×16 и другую token geometry в одном checkpoint. Для T>2 также нужно перейти от полного self-attention по `T×N` к локальному cross-time attention, иначе стоимость растёт квадратично.

## 11. Как сделать настоящую query-conditioned soft segmentation

Сейчас карта — useful diagnostic baseline, но не обученная segmentation head. Чтобы сделать её функционально необходимой, следующий отдельно разрешённый mechanism pilot должен быть таким:

```text
Stage 1: text vector × precomputed pair vector -> ANN Top-K
Stage 2: text tokens × temporal dense tokens -> evidence logits
         sparse/entmax weights -> evidence vector
         global + evidence score -> one unified ranking loss
         same weights -> query-conditioned map
```

Ключевые правила:

- map должна влиять на тот же scalar relevance score;
- не использовать global pair vector как обходной путь для локализации;
- не давать маски в primary training;
- masks использовать только evaluation-sidecar и отдельную supervised upper-bound ablation;
- trainable localized queries должны иметь проверенную provenance;
- для T>2 map должна хранить frame/interval identity;
- deletion of top-evidence tokens должен снижать score сильнее, чем deletion bottom-evidence;
- query swap должен менять map;
- uniform-map baseline не должен быть лучше learned-map baseline.

Эта causal evidence branch не включена в текущий прямой TemporalSigLIP pilot. Включать её до появления verified localized query view рискованно: модель может выучить равномерную карту или source/event shortcut.

## 12. Как встроить модель в общий цикл обучения

### Шаг 1 — финальная синхронизация Dataset Agent

Нужен файл `contracts/qcpr_shared/handoff/dataset_final_to_model.json` с:

- exact train/dev/test manifests;
- semantic capability;
- stable-scene capability;
- localized capability;
- long-series capability;
- source/split hashes;
- verification states;
- masks только в evaluation sidecars.

Сейчас этого handoff нет. Поэтому P2 и final three-seed training заблокированы.

### Шаг 2 — deterministic preflight

Перед GPU проверить code/data/manifest/query/gallery/preprocessing SHA, tokenizer IDs, pair/query sequence hashes, source proportions, no mask access и fixed exposure budget.

### Шаг 3 — выбрать Stage A или B на финальном dev

Пилотная предварительная селекция — Stage B step 1024. Финальная селекция должна использовать не старые pilot checkpoints, а одинаковую pretrained initialization и три clean seeds на final exact release.

### Шаг 4 — exact retrieval first

Сначала обучать exact retrieval на human-verified exact captions. Semantic positives подключать только когда Dataset Agent явно отдаст `SEMANTIC_TRAIN_READY`. Generic no-change держать диагностическим/semantic, а не exact target.

### Шаг 5 — затем semantic/localized mechanism

Отдельно проверять semantic multi-positive retrieval, затем causal evidence reranking. Не смешивать эти изменения в одной uncontrolled run.

### Шаг 6 — final evaluation

Считать exact full-gallery MRR/R@K, semantic mAP/nDCG, per-source metrics, source-probe, GSD/footprint buckets, pair-cluster bootstrap и localization diagnostics. Test gallery открывать один раз после выбора checkpoint.

## 13. Главные проблемы сейчас

| Приоритет | Проблема | Влияние |
|---|---|---|
| Critical | Нет final Dataset Agent handoff | Нельзя законно и воспроизводимо запускать final/P2 training |
| High | Source identity probe ≈ 0.999 | Модель может пользоваться стилем датасета, а не изменением |
| High | Map не входит в score | Нет доказательства query-conditioned soft segmentation |
| High | Только один bounded seed | Нет confidence interval и надёжной generalization claim |
| High | RSCC text/semantic/localized views не training-enabled | Новые физические domains пока не дают проверенного textual supervision |
| Medium | Fixed 256×256 / 16×16 grid | Малые объекты и разные physical footprints могут теряться |
| Medium | T>2 только contract-level | Long-series retrieval не доказан на реальных sequence views |
| Medium | Multi-positive runtime не упражнён | Semantic loss path не подтверждён реальным batch |
| Medium | Stage B step 2048 не монотонен по MRR | Нужно выбирать checkpoint по dev MRR, не по last step |
| Low | Upstream tokenizer config warning | Сейчас записан и проверен, но требует сохранения в reproducibility metadata |

## 14. Что отдать GPT на независимый разбор

Лучшие независимые задачи для GPT/ревьюера:

1. проверить, что exact/semantic relevance не смешаны и pair-to-text positives построены правильно;
2. проверить source/event leakage и сравнить source probe с cross-source retrieval;
3. проверить, что sampler не даёт одному событию/источнику доминировать;
4. проверить fixed-resolution transform и физический footprint для T1/T2;
5. проверить tokenizer/model preprocessing и special-token IDs;
6. проверить score-path graph: map действительно входит в scalar score в будущей evidence branch;
7. проверить deletion/insertion/query-swap diagnostics;
8. проверить, что checkpoint selection не использует test и не выбирает последний step автоматически;
9. посчитать pair-clustered bootstrap и seed variance;
10. провести ablation: frozen towers vs top-block adaptation vs temporal-only;
11. проверить synthetic/generated caption provenance и не допустить их в real-only test;
12. проверить реальные long-series timestamp/order contracts.

## 15. План получения хорошего результата

Нельзя получить устойчивое улучшение простым добавлением большего числа строк. Нужна последовательность:

1. сначала закрыть Dataset Agent handoff и exact data integrity;
2. держать source-balanced batches и ограничить RSCC/event contribution;
3. использовать verified short factual captions, а не длинные непроверенные descriptions;
4. не включать hard negatives, пока не измерен false-negative rate;
5. фиксировать unique pair presentations, а не только epochs;
6. обучить exact baseline и сделать same-gallery comparison;
7. только потом включить verified semantic positives;
8. только после localized data добавить causal evidence reranking;
9. использовать top-block LoRA/partial unfreezing с LR на порядок меньше temporal adapter;
10. выбирать checkpoint по MRR с guardrails по R@10, per-source и source probe;
11. подтверждать кандидат тремя чистыми seeds;
12. тестировать на held-out sources, GSD и physical footprint, а не только на смешанном среднем.

Критерий успеха: улучшение exact real-only retrieval на общей gallery, рост semantic mAP/nDCG после появления verified semantic data, отсутствие регрессии LEVIR/SECOND и перенос улучшения в causal evidence localization. Пока подтверждён только первый bounded retrieval-сигнал; остальные части ещё не доказаны.

## 16. Текущий readiness

```text
B1_CONTROL                     = VERIFIED
MODEL_V3_SYNTHETIC_SMOKE       = PASS
MODEL_V3_REAL_INTEGRATION      = PASS_TECHNICAL_SMOKE
MODEL_V3_MECHANISM              = READY_FOR_AUTHORIZATION_ONLY
MODEL_V3_MAIN                   = NOT_AUTHORIZED
P2_REAL                         = BLOCKED_DATASET_CONTRACT
P2_SEMANTIC                     = BLOCKED_DATASET_CONTRACT
P2_LONG_SERIES                  = BLOCKED_DATASET_CONTRACT
```

## 17. Последняя hardening-проверка после этого отчёта

В model-only ветку добавлены строгие защитные проверки:

- `scripts/validate_temporal_siglip_final_handoff.py` возвращает `WAIT_DATASET_FINAL_HANDOFF` с non-zero exit code, если final handoff отсутствует;
- при наличии handoff проверяются exact train/development/test manifests, SHA256, `N_TRAIN_UNIQUE_PHYSICAL_PAIRS`, capability flags и отсутствие mask/dense-label полей в primary manifests;
- `run_temporal_siglip.py --final-handoff` запрещает запуск с другим release или manifest;
- финальный driver пишет `caption_exposure.json` и `source_exposure.json`;
- `--max-pair-presentations` прерывает run при превышении exposure ceiling;
- текущий resolution decision зафиксирован как provisional `SigLIP2 B/16 256`, а 384 comparison не запускался.

Focused cluster contract: `4 passed, 2 warnings`. Полный cluster suite: `670 passed, 3 skipped, 16 warnings` за `418.72 секунд` (`6:58`). Compileall, shell syntax, git diff check and Model-v3 Pyright scope (`0 errors, 0 warnings`) также прошли. Final handoff по-прежнему отсутствует, поэтому эти gates являются подготовкой и не являются разрешением на final GPU training.

Итог: текущий TemporalSigLIP — хороший технический retrieval baseline с положительным bounded Stage-B сигналом. Это ещё не финальная модель для разнообразного Dataset-v2 и не query-conditioned soft segmentation. Следующий правильный шаг — закрыть final Dataset Agent handoff, провести три clean seeds на exact core, затем отдельно разрешить causal evidence mechanism pilot.
