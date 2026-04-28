# Land Surface Mapping

Проект переведён с `binary change detection` на `semantic-first` схему:

`T1 surface segmentation -> T2 surface segmentation -> class transitions`

Что сейчас является базой проекта:
- `mfaytin/mask2former-satellite` как уже работающий RGB semantic baseline
- `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL` как первый следующий multispectral target
- `ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL` как более тяжёлый benchmark target
- мультиспектральные данные OSCD/Sentinel-2 (`13` каналов)

Что удалено:
- `MaskCD`
- `ChangeFormer`
- `Satlas`
- `Open-CD` интеграции
- старые binary-first inference scripts

## Текущий практический режим

Сейчас локальный Streamlit UI работает в `VLM-first visual demo` режиме:
- показывает выбранные `Before` / `After` crop;
- строит понятную 4x4 contact sheet `A1..D4`;
- отправляет crop и contact sheet в выбранную VLM, по умолчанию `gemma4:e4b` через Ollama;
- выводит plain-English описание сцены, summary `before/after`, основные видимые изменения и отдельное наблюдение по каждой ячейке `A1..D4`.

`mask2former-satellite` остаётся локальным semantic artifact в `artifacts/models/semantic/mask2former-satellite`, но в UI его OpenEarthMap-классы теперь показываются только в debug-режиме. Для текущих RGB crop эти классы могут быть визуально misleading, поэтому они не используются как user-facing explanation.

`Prithvi-EO-2.0` не является прямой заменой текущего RGB `Mask2Former` backend. Для него нужен отдельный TerraTorch/multispectral путь: корректный порядок каналов, task head, checkpoint/config и проверка на EO данных. В коде это зафиксировано отдельным planned backend `prithvi_terratorch`, чтобы случайно не выдавать RGB inference за Prithvi-based результат.

## Ответ руководителю

Фраза про архитектуру:

`Mask2Former -> current semantic baseline; Prithvi-EO-2.0 -> next-stage semantic backbone; CDMamba -> binary validation model; VLMs -> explanation layer`

это наша проектная рекомендация, подготовленная с помощью GPT, но основанная на текущем коде, proposal-документе и проверенных источниках по remote sensing. Смысл такой:

- основной источник доказательства должен быть пиксельный и проверяемый: `T1 semantic segmentation -> T2 semantic segmentation -> transition matrix`;
- VLM не должен "рисовать" маску или решать, где произошла смена класса;
- VLM можно использовать только после расчёта масок и transition table, чтобы объяснить результат человеку;
- binary change model вроде `CDMamba` полезен для проверки, где вообще есть изменение, но он не отвечает на вопрос `что во что изменилось`.

## Установка

```bash
cd /Users/sargisvardanyan/Land-Change-Detection
chmod +x scripts/create_env.sh
./scripts/create_env.sh
source .venv/bin/activate
```

## Загрузка данных

```bash
source .venv/bin/activate
python scripts/download_oscd.py
```

## Загрузка semantic-моделей

```bash
source .venv/bin/activate
python scripts/download_semantic_models.py
```

Скрипт скачает:
- `mfaytin/mask2former-satellite` в `artifacts/models/semantic/mask2former-satellite`
- `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL` в `artifacts/models/semantic/Prithvi-EO-2.0-300M-TL`
- `ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL` в `artifacts/models/semantic/Prithvi-EO-2.0-600M-TL`

## Локальное приложение

```bash
source .venv/bin/activate
streamlit run app.py
```

В UI сейчас доступны:
- `Before`
- `After`
- `Mask2Former surface segmentation` from `artifacts/models/semantic/mask2former-satellite`
- `A1..D4 before/after contact sheet`
- `Gemma visual interpretation`
- `Cell observations`
- debug traces/raw model output, если включён `Show debug trace/details`

## Remote-sensing VLM models

Для более смыслового описания спутниковых изменений добавлен локальный кандидат:

- `AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct` -> `artifacts/models/vlm/remote-sensing-Qwen2-VL-2B-Instruct`
- `akshaydudhane/EarthDial_4B_RGB` -> `artifacts/models/vlm/EarthDial_4B_RGB`

Он меньше и практичнее для MacBook, чем 7B-модели, и дообучен на remote-sensing visual instruction данных. Более тяжёлый следующий кандидат:

- `AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct` -> `artifacts/models/vlm/remote-sensing-Qwen2.5-VL-3B-Instruct`

`Remote-sensing Qwen` модели прописаны в `scripts/download_vlm_models.py`. `EarthDial_4B_RGB` скачивается отдельным скриптом:

```bash
python scripts/download_earthdial.py
```

`GeoChat-7B` остаётся исследовательским кандидатом: он заметно тяжелее для MacBook и пока не интегрирован в текущий runtime.

## DINOv3 SAT feature diagnostics

DINOv3 больше не показывается как normal `Semantic model`: без отдельного decoder/head он не выдаёт OpenEarthMap классы вроде `road`, `building`, `water`. В приложении semantic path остаётся Mask2Former, а DINOv3 оставлен только как debug/research feature extractor.

Оставлен только самый сильный практичный SAT-вариант:
- `facebook/dinov3-vitl16-pretrain-sat493m` -> official SAT checkpoint, gated на Hugging Face.
- `timm/vit_large_patch16_dinov3.sat493m` -> открытый runnable mirror для Mac, около `1.21 GB` на HF и около `2.3 GB` локально.

Непрактичный локальный вариант:
- `facebook/dinov3-vit7b16-pretrain-lvd1689m` / `...sat493m` -> около `26.86 GB`

Скрипт загрузки:

```bash
python scripts/download_dinov3_models.py
```

## Архитектурная цель

Следующий шаг проекта:

1. перейти с RGB semantic map на полноценный multispectral semantic layer
2. встроить `Prithvi-EO-2.0-300M-TL` через TerraTorch как первый практический target
3. масштабировать на `Prithvi-EO-2.0-600M-TL` после рабочего 300M pipeline
4. добавить нормальный transition engine по классам поверхности
5. держать VLM только как explain-layer

## Источники

- `mfaytin/mask2former-satellite`: [Hugging Face](https://huggingface.co/mfaytin/mask2former-satellite)
- `Prithvi-EO-2.0-300M-TL`: [Hugging Face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL)
- `Prithvi-EO-2.0-600M-TL`: [Hugging Face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL)
- `Prithvi-EO-2.0` paper: [arXiv:2412.02732](https://arxiv.org/abs/2412.02732)
- `CDMamba`: [arXiv:2406.04207](https://arxiv.org/abs/2406.04207)
- `OpenMapCD`: [Zenodo](https://zenodo.org/records/14028095)
- `ESA WorldCover v200`: [Google Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200)
- `Dynamic World V1`: [Google Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1)
