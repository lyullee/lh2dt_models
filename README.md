# LH2DT: 제일원리 액화수소 공정 모델 라이브러리

LH2DT는 액화수소 저장·이송·기화·벤트 설비를 코드에서 조립할 수 있는
물리기반 모델 라이브러리입니다. 저장탱크, 진공단열, 밸브, 배관, 기화기,
재액화기, 펌프, 압축기와 이를 연결하는 정상·동적 네트워크 모델을 같은
열역학 상태·질량/에너지 유량 계약으로 제공합니다.

이 저장소는 **모델 코드, 공개 문헌 기준 예제, 단위 테스트**만 포함합니다.
현장 태그, 원계측 시계열, 사진·영상, P&ID, 시각화 저작도구와 기관별
구성파일은 포함하지 않습니다. 따라서 다른 설비나 시험 데이터에 모델을
그대로 연결할 수 있고, 현장 식별정보를 공개 저장소에 섞지 않습니다.

## 빠른 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -q
```

Python 3.10 이상과 CoolProp, NumPy, SciPy가 필요합니다. Windows와 Linux에서
동일한 SI 단위 계약을 사용합니다.

## 1분 예제: 균질 탱크의 열침입

```python
from lh2dt import HydrogenProperties, HomogeneousTank, MassEnergyFlow, TankGeometry

props = HydrogenProperties()
tank = HomogeneousTank(TankGeometry(
    volume_m3=10.0,
    wall_heat_capacity_J_K=2.0e6,
    ambient_UA_W_K=8.0,
    fluid_wall_UA_W_K=15.0,
), props)
state0 = tank.initialize_saturated(
    pressure_Pa=130_000.0, liquid_volume_fraction=0.70,
)

def boundary(_time_s, thermo):
    # 양수는 탱크로 유입, 음수는 탱크에서 유출입니다.
    return [MassEnergyFlow(0.0, thermo.specific_enthalpy_J_kg, "closed")]

trace = tank.simulate(
    state0, duration_s=600.0, time_step_s=10.0,
    ambient_temperature_K=293.15, flow_callback=boundary,
)
print(trace[-1].thermo.pressure_Pa, trace[-1].total_energy_residual_J)
```

`examples/quickstart_tank.py`는 같은 계산을 실행하고 질량·에너지 장부를
출력합니다. 탱크의 압력과 온도는 모델 출력이며, 시험 데이터는 초기상태와
경계조건을 정하는 데 사용하고 이후 블라인드 비교 입력으로 취급하는 것을
권장합니다.

## 제공 모델

| 영역 | 구현 |
|---|---|
| 물성 | Para-Hydrogen 기반 p-T, p-h, p-u, 포화액/포화증기, 밀도·엔탈피·내부에너지 |
| 저장탱크 | 균질 평형, 층상 액체/기체/벽, 반경-축 방향 다중셀, 강체 공통압력 DAE |
| 열전달 | 진공 복사, MLI/고체층, 지지대·잔류가스 열경로, 벽-유체 대류, 자연순환, 축방향 전도 |
| 상변화 | 계면 열수지, Schrage 질량플럭스, 기체막 열전달, 분산 비등 및 보존형 투영 |
| 유동 | 등엔탈피/HEM 밸브·배관, Darcy–Weisbach 단상, 유체·벽체를 적분하는 동적 HEM 배관 |
| 장치 | 명령형·체크·릴리프·온도보호 밸브, 펌프/가상 H-Q 펌프, 압축기, 기화기, 재액화기, 벤트 스택 |
| 연결 | 정상 압력망, 동적 노드망, 층상/반경-축 탱크망, 병렬 열유체 링크 |
| 계측 | 1차 지연 전송기와 단위 변환, 4–20 mA HART 보조변수 관측 계약 |
| 검증 | NASA TN D-4171 공개 기준, 질량·에너지·운동량 잔차, 시간간격 수렴, 시계열 nMAPE |

## 코드에서 연결하는 방법

1. `HydrogenProperties`로 모든 상태를 생성합니다.
2. 장치 형상·정격·열경계를 명시적으로 생성합니다.
3. 각 장치의 `initialize_*`로 초기 상태를 만들고 `step`/`simulate`로 진행합니다.
4. 장치 사이에는 `MassEnergyFlow` 또는 네트워크의 `NetworkLink`를 사용합니다.
5. 결과의 `thermo`, `mass/energy` 장부, `*_residual`을 확인한 뒤 외부 데이터와 비교합니다.

모든 압력은 **Pa 절대압**, 온도는 **K**, 질량유량은 **kg/s**, 비엔탈피는
**J/kg**, 열량은 **W**, 체적은 **m³**입니다. 게이지압은 경계에서
`gauge_to_absolute_pressure_Pa`로 변환하고 모델 내부에는 절대압만 전달합니다.

주요 공개 진입점은 `lh2dt.tank`, `lh2dt.layered_tank`,
`lh2dt.radial_axial_tank`, `lh2dt.valve`, `lh2dt.pipe`,
`lh2dt.dynamic_network`, `lh2dt.vaporizer`, `lh2dt.reliquefier`입니다.
모든 클래스는 직접 모듈에서 import할 수 있으며, 자주 쓰는 타입은 패키지
최상위에서도 export합니다.

## 문서와 예제

- [`docs/모델_사용_및_인터페이스.md`](docs/모델_사용_및_인터페이스.md): 코드 연결 계약과 예제
- [`docs/물리모델_구조와_방정식.md`](docs/물리모델_구조와_방정식.md): 상태변수·보존식·열전달·상변화
- [`docs/검증과_재현성.md`](docs/검증과_재현성.md): 공개 기준시험, 잔차, 테스트 절차
- `examples/`: 탱크·네트워크·기화·재액화·NASA 공개 기준 예제
- `tests/`: 센터 데이터에 의존하지 않는 모델 단위 테스트

## 모델 사용 원칙

기본 모델 파라미터는 형상·물성·공개 상관식에서 계산하며 현장 시계열에
자동 피팅하지 않습니다. 특정 설비에 적용할 때는 설계자료로 형상·열경계·
밸브/배관 정격을 바인딩하고, 센서 시계열은 초기조건과 독립 검증용으로
분리합니다. 계측 범위·단위·시간축이 확정되지 않은 데이터는 모델 입력으로
조용히 보정하지 않고 명시적으로 오류를 냅니다.

## 범위와 제한

본 저장소는 안전등급 계산이나 규제 인증을 대신하지 않습니다. 극저온 2상
유동의 모든 형상과 제어기를 하나의 기본값으로 결정하지 않으며, 적용 장치의
압력범위·열경계·밸브 특성·상변화 상관식의 유효범위를 검토한 뒤 사용해야
합니다. 정량 성능 평가는 동일한 시간축과 경계조건으로 별도 시험 데이터에
대해 수행합니다.

