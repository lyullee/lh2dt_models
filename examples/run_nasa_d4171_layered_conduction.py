"""Record the unfitted six-cell molecular-conduction NASA benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from lh2dt import run_nasa_d4171_layered_conduction_benchmark


def main() -> None:
    output = Path("docs/benchmarks")
    output.mkdir(parents=True, exist_ok=True)
    report = run_nasa_d4171_layered_conduction_benchmark()
    (output / "nasa_tn_d4171_layered_conduction_results.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    labels = {"lower": "하부", "uniform": "균일", "upper": "상부"}
    lines = [
        "# NASA TN D-4171 다중셀 분자전도 기준해석",
        "",
        f"출처: [{report.source_report}]({report.source_url}), {report.source_title}",
        "",
        "공통압력 아래 액상 3셀과 기상 3셀을 사용했다. Table I의 젖은 벽·건조 벽 순열량을 각 상의 "
        "초기 벽 접촉면적에 따라 셀에 배분하고, CoolProp 분자 열전도도와 포화 계면 상변화만 적용했다. "
        "관측 압력상승률은 비교기간을 정하는 데만 사용했으며 파라미터 탐색은 수행하지 않았다.",
        "",
        "| 가열 방식 | 시험 수 | 평균 예측/관측 | MAPE | NRMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("lower", "uniform", "upper"):
        item = report.by_heating_mode[mode]
        lines.append(
            f"| {labels[mode]} | {item.count} | {item.mean_predicted_over_observed:.3f} | "
            f"{item.mean_absolute_percentage_error_percent:.1f}% | {item.normalized_rmse:.3f} |"
        )
    item = report.overall
    lines += [
        f"| 전체 | {item.count} | {item.mean_predicted_over_observed:.3f} | "
        f"{item.mean_absolute_percentage_error_percent:.1f}% | {item.normalized_rmse:.3f} |",
        "",
        "## 판정",
        "",
        "분자전도만 사용한 다중셀 모델의 전체 평균 예측/관측비는 1.324다. HEM 하한 0.480보다 "
        "관측에 가까워졌고 상간전달 0 한계 1.362보다 낮아졌지만, 하부 1.241·균일 1.349·상부 "
        "1.367로 모두 과대예측한다. 또한 Table II가 요구하는 큰 기상→액체 계면 열전달을 분자전도만으로 "
        "설명할 수 없다. 따라서 다중셀 보존 구조는 유지하고 자연대류·계면 혼합 폐합을 문헌 상관식으로 "
        "추가하는 것이 다음 단계다. 이 결과는 V2 정량 검증 통과가 아니다.",
        "",
        "## 고정 가정",
        "",
    ]
    lines.extend(f"- {assumption}" for assumption in report.assumptions)
    (output / "nasa_tn_d4171_layered_conduction_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
