"""Record the unfitted two-region limiting calculation for NASA TN D-4171."""

from __future__ import annotations

import json
from pathlib import Path

from lh2dt import run_nasa_d4171_benchmark, run_nasa_d4171_two_region_limit_benchmark


def main() -> None:
    output = Path("docs/benchmarks")
    output.mkdir(parents=True, exist_ok=True)
    hem = run_nasa_d4171_benchmark()
    limit = run_nasa_d4171_two_region_limit_benchmark()
    payload = {
        "homogeneous_equilibrium_lower_response": hem.to_dict(),
        "zero_interphase_transfer_upper_response": limit.to_dict(),
    }
    (output / "nasa_tn_d4171_two_region_bounds.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# NASA TN D-4171 2영역 탱크 무피팅 포위 계산",
        "",
        f"출처: [{limit.source_report}]({limit.source_url}), {limit.source_title}",
        "",
        "균질평형(HEM)과 액상-기상 상간전달 0 한계를 함께 계산했다. 전자는 모든 열이 즉시 평형화되는 "
        "응답 하한이고, 후자는 Table I의 젖은 벽 순열량을 액상에, 건조 벽 순열량을 기상에 주되 상간 "
        "열·질량 전달을 막은 응답 상한이다. 두 계산 모두 관측 압력상승률로 파라미터를 맞추지 않았다.",
        "",
        "| 가열 방식 | 시험 수 | HEM 예측/관측 | 상간전달 0 예측/관측 |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("lower", "uniform", "upper"):
        lines.append(
            f"| {mode} | {hem.by_heating_mode[mode].count} | "
            f"{hem.by_heating_mode[mode].mean_predicted_over_observed:.3f} | "
            f"{limit.by_heating_mode[mode].mean_predicted_over_observed:.3f} |"
        )
    lines += [
        f"| 전체 | {hem.overall.count} | {hem.overall.mean_predicted_over_observed:.3f} | "
        f"{limit.overall.mean_predicted_over_observed:.3f} |",
        "",
        "## 판정",
        "",
        "HEM은 전체 평균 0.480으로 낮게, 상간전달 0 한계는 1.362로 높게 예측했다. 공개된 젖은 벽·건조 벽 "
        "경계를 분리하자 하부 < 균일 < 상부의 열위치 민감도가 모델에 나타났다. 따라서 액상·기상 분리는 "
        "필요한 구조이며, 다음 검증 대상은 두 한계 사이를 결정하는 상간 열·질량 전달과 액상 열성층이다. "
        "상간전달 0 결과를 완성 모델의 정확도로 해석하지 않는다.",
        "",
        "## 고정 가정",
        "",
    ]
    lines.extend(f"- {item}" for item in limit.assumptions)
    (output / "nasa_tn_d4171_two_region_bounds.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
