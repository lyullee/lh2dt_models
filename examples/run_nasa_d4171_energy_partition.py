"""Record the unfitted NASA TN D-4171 Table II structure audit."""

from __future__ import annotations

import json
from pathlib import Path

from lh2dt import run_nasa_d4171_energy_partition_benchmark


def main() -> None:
    output = Path("docs/benchmarks")
    output.mkdir(parents=True, exist_ok=True)
    report = run_nasa_d4171_energy_partition_benchmark()
    (output / "nasa_tn_d4171_energy_partition_results.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# NASA TN D-4171 Table II 에너지 분배 구조 검증",
        "",
        f"출처: [{report.source_report}]({report.source_url}), {report.source_title}",
        "",
        "공식 원문의 17개 에너지 분배 시험을 사용했다. 기존 로컬 전사본의 보고서 번호와 0.1%p "
        "전사 차이를 원문 표에 맞춰 정정했다. 표의 관측값으로 열전달계수나 다른 파라미터를 찾지 않았다.",
        "",
        "상간전달 0 한계는 젖은 벽 열량이 액체에, 건조 벽 열량이 기체에 그대로 남고 증발 에너지가 "
        "0이라고 예측한다. 관측 에너지 분배와 비교해 현재 모델 구조에서 빠진 경로를 판정했다.",
        "",
        "| 가열 방식 | 시험 | 액체 열층 관측 | 증발 관측 | 건조측에서 액체·증발로 이동한 열 | 액체 MAE | 기체 MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"lower": "하부", "uniform": "균일", "upper": "상부"}
    for mode in ("lower", "uniform", "upper"):
        summary = report.by_heating_mode[mode]
        lines.append(
            f"| {labels[mode]} | {summary.count} | "
            f"{summary.mean_observed_liquid_layer_energy_percent:.2f}% | "
            f"{summary.mean_observed_evaporation_energy_percent:.2f}% | "
            f"{summary.mean_inferred_cross_phase_transfer_percent:.2f}% | "
            f"{summary.zero_transfer_liquid_mae_percent_point:.2f}%p | "
            f"{summary.zero_transfer_vapor_mae_percent_point:.2f}%p |"
        )
    overall = report.overall
    lines += [
        f"| 전체 | {overall.count} | {overall.mean_observed_liquid_layer_energy_percent:.2f}% | "
        f"{overall.mean_observed_evaporation_energy_percent:.2f}% | "
        f"{overall.mean_inferred_cross_phase_transfer_percent:.2f}% | "
        f"{overall.zero_transfer_liquid_mae_percent_point:.2f}%p | "
        f"{overall.zero_transfer_vapor_mae_percent_point:.2f}%p |",
        "",
        "## 판정",
        "",
        "하부가열에서는 건조측에서 액체·증발로 이동한 순에너지가 평균 0.40%로 작다. 상부가열에서는 "
        "평균 81.56%이며 액체 열층에 평균 60.96%, 증발에 20.24%가 배분된다. 균일가열은 두 거동의 "
        "중간이다. 따라서 액상과 기상의 단순 분리만으로는 부족하고, 기상에서 계면 액체 열층으로 전달되는 "
        "열경로와 액체 벌크-열층 분리가 필요하다.",
        "",
        "현재 2영역 모델의 상간전달 0 한계는 Table II 전체에서 액체 28.28%p, 기체 35.63%p, 증발 "
        "10.17%p의 평균절대오차를 보이므로 정량 모델로 기각한다. 다음 모델은 압력상승률뿐 아니라 이 세 "
        "에너지 분배와 액체 열층 비율을 동시에 출력해 검증해야 한다.",
        "",
        "## 고정 가정",
        "",
    ]
    lines.extend(f"- {item}" for item in report.assumptions)
    (output / "nasa_tn_d4171_energy_partition_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
