"""Run and record the unfitted NASA TN D-4171 HEM benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lh2dt import run_nasa_d4171_benchmark


PA_S_TO_PSI_MIN = 60.0 / 6_894.757293168


def markdown(report) -> str:
    lines = [
        "# NASA TN D-4171 균질평형 탱크 독립 검증",
        "",
        f"출처: [{report.source_report}]({report.source_url}), {report.source_title}",
        "",
        "센터 데이터나 NASA 관측값으로 파라미터를 맞추지 않았다. 9인치 구형 탱크, 1 atm 시작, "
        "명목 100 psia 종료와 Table I 평균 순열유속을 그대로 사용해 고정 질량·고정 체적 HEM 전방계산을 수행했다.",
        "",
        "## 판정",
        "",
        "21개 시험 모두에서 HEM이 압력상승률을 낮게 예측했다. 하부가열에서는 비교적 가까웠지만 "
        "상부가열에서는 관측의 약 23%만 예측했다. 총열량만 보존하는 균질평형 모델은 열 위치에 따른 "
        "기상 과열·액상 성층을 표현할 수 없으므로 센터 저장·벤트 압력동역학의 주 모델로 사용할 수 없다. "
        "보존식 기준모델과 하한 비교용으로 유지하고, 비평형 성층 모델을 다음 계층으로 구현해야 한다.",
        "",
        "| 가열 방식 | 시험 수 | 평균 예측/관측 | MAPE | NRMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in ("lower", "uniform", "upper"):
        summary = report.by_heating_mode[mode]
        lines.append(
            f"| {mode} | {summary.count} | {summary.mean_predicted_over_observed:.3f} | "
            f"{summary.mean_absolute_percentage_error_percent:.1f}% | {summary.normalized_rmse:.3f} |"
        )
    lines += [
        f"| 전체 | {report.overall.count} | {report.overall.mean_predicted_over_observed:.3f} | "
        f"{report.overall.mean_absolute_percentage_error_percent:.1f}% | {report.overall.normalized_rmse:.3f} |",
        "",
        "## 시험별 결과",
        "",
        "| 시험 | 가열 | 초기 충전율 | 순열유속 (W/m2) | 관측 (psi/min) | HEM (psi/min) | 예측/관측 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for point in report.points:
        lines.append(
            f"| {point.test_number} | {point.heating_mode} | {100*point.initial_fill_fraction:.1f}% | "
            f"{point.net_average_heat_flux_W_m2:.2f} | "
            f"{point.observed_pressure_rise_Pa_s*PA_S_TO_PSI_MIN:.3f} | "
            f"{point.predicted_pressure_rise_Pa_s*PA_S_TO_PSI_MIN:.3f} | "
            f"{point.predicted_over_observed:.3f} |"
        )
    lines += [
        "",
        "## 출처 식별자 정정",
        "",
        "기존 `02_분석_재현/data/nasa_tnd3742_table1.csv`의 21개 LH2 행은 NASA TN D-4171 Table I 자료다. "
        "NASA TN D-3742는 방사형 유입 터빈 보고서이므로 새 벤치마크에서는 D-4171로 정정했다. "
        "기존 파일은 다른 재현 작업의 입력일 수 있어 수정하지 않았고, 정정된 21개 행을 패키지 데이터로 분리했다.",
        "",
        "## 고정 가정",
        "",
    ]
    lines.extend(f"- {item}" for item in report.assumptions)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/benchmarks"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_nasa_d4171_benchmark()
    (args.output_dir / "nasa_tn_d4171_hem_results.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "nasa_tn_d4171_hem_validation.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(markdown(report).split("## 시험별 결과", 1)[0])


if __name__ == "__main__":
    main()
