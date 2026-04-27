from __future__ import annotations

import argparse
import json
from pathlib import Path

from land_change_detection.data import OSCDSceneRepository
from land_change_detection.explain import describe_change
from land_change_detection.inference import run_difference_baseline


SYSTEM_PROMPT = (
    "Ты помощник аналитика по мониторингу изменений земной поверхности. "
    "Пиши короткое, аккуратное и проверяемое объяснение по-русски. "
    "Не придумывай семантические классы, если они не подтверждены данными."
)


def build_target(summary_json: str, city: str) -> str:
    summary = json.loads(summary_json)
    return (
        f"Сцена: {city}. "
        f"Доля изменений по рабочему порогу: {summary['changed_percent']}%. "
        f"Уровень: {summary['severity']}. "
        "Цветовые классы отражают только интенсивность вероятности изменений, а не тип объекта. "
        "Для трактовки как строительство, дорога, вырубка или иное событие нужен ручной просмотр "
        "или семантическая модель с отдельной разметкой."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/oscd"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen_lora/data"))
    args = parser.parse_args()

    repo = OSCDSceneRepository(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for city in repo.split_cities("all"):
        scene = repo.load_scene(city)
        prob = run_difference_baseline(scene.pre, scene.post, band_mode="rgb")
        summary = describe_change(prob, threshold=0.5)
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Вот структурированная сводка детекции изменений. "
                            "Сделай краткий отчёт для аналитика:\n\n"
                            f"{summary}"
                        ),
                    },
                    {"role": "assistant", "content": build_target(summary, city)},
                ]
            }
        )

    n = len(rows)
    train_end = max(1, int(n * 0.7))
    valid_end = max(train_end + 1, int(n * 0.85))
    splits = {
        "train": rows[:train_end],
        "valid": rows[train_end:valid_end],
        "test": rows[valid_end:],
    }
    if not splits["valid"]:
        splits["valid"] = rows[:1]
    if not splits["test"]:
        splits["test"] = rows[-1:]

    for split_name, split_rows in splits.items():
        out = args.output_dir / f"{split_name}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(out)


if __name__ == "__main__":
    main()
