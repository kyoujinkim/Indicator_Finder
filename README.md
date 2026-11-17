# Indicator_Finder

**Multi-Frequency Macro-Equity Analytics (MFMEA)**: 서로 다른 빈도의 경제지표와 주가·EPS 사이의 관계를 여러 분석 경로로 비교하고, 설명력·시차·충격·임계값 특성을 순위화하는 Python 리서치 도구입니다.

## 분석 방법

| 경로 | 주요 분석 |
|---|---|
| 지속적 관계 | Granger 검정, 전이 엔트로피, ElasticNet |
| 충격의 지속 | VAR 충격반응함수(IRF)와 지속성 |
| 발표·충격 | 이벤트 연구와 XGBoost/SHAP |
| 임계값 | 분위수 임계값, 상향·하향 돌파, 조건부 선도수익률 |
| 추가 분석 | 이름이 있는 이벤트의 영향, MI·Boruta 방식 변수선택, 국면별 중요도 |
| 시간 비교 | 이동 구간을 이용한 중요도 변화 비교 |

통계적 선행성·예측 관계를 탐색하는 도구이며 구조적 인과관계를 확정하는 시스템은 아닙니다.

## 구조

```text
data/data.xlsx + data/equity.xlsx + data/impacts.txt
  → src/data_loader.py
  → src/preprocessor.py
  → 관계·IRF·충격·이벤트·임계값 분석
  → 변수선택·src/relation_scorer.py
  → output/mfmea/ 보고서
```

`main.py`는 관계 분석 진입점이며 `forecast.py`는 별도 예측 연구 스크립트입니다. 예측 관련 결과는 `output/forecast/` 설정을 참조합니다.

## 현재 상태

분석 경로와 CLI는 구현돼 있으며, 사용자가 준비한 경제·시장 워크북을 입력으로 사용합니다. 의존성 파일과 공개용 예제 데이터는 추가 준비가 필요합니다. 아래 명령은 실제 CLI 형식을 따른 예시이며 전체 실행 결과를 검증했다는 의미는 아닙니다.

## 설치

현재 저장소 루트에는 `requirements.txt`가 확인되지 않습니다. 다음은 소스 import를 바탕으로 한 **초기 의존성 후보**이며, 설치·호환성 검증 후 버전을 확정해야 합니다.

```bash
python -m venv .venv
# 가상환경 활성화 후
python -m pip install pandas numpy scipy scikit-learn statsmodels xgboost shap openpyxl python-dateutil
```

선택 기능에서 쓰는 의존성과 전이 엔트로피·변수선택의 대체 구현 여부는 해당 모듈을 확인하세요. 입력 데이터를 제거한 공개본에는 스키마에 맞는 합성 워크북을 별도로 제공하는 것이 좋습니다.

## 입력 형식

`src/config.py`는 기본 입력을 `data/`에서 찾으며 `DATA_START_ROW=15`를 사용합니다. 다음은 주석의 과거 설명이 아닌 **현재 로더 구현** 기준입니다.

### 경제지표: `data/data.xlsx`

- 시트별로 지표 묶음을 배치합니다.
- **Excel 1행:** 첫 열은 날짜 열 이름, 이후 열은 지표명입니다.
- **2~15행:** 메타데이터 구간이며 로더가 건너뜁니다.
- **16행 이후:** 날짜와 숫자 데이터입니다.
- 빈도 판별 함수는 시트명의 `일`, `주`, `월`, `분기`, `연`을 사용합니다. 예: `매크로_월별`. 영문 `Macro_Monthly`만으로는 현재 함수가 빈도를 인식하지 못합니다.

### 대상 자산: `data/equity.xlsx`

- 분석 대상에 따라 `price`, `eps` 시트를 사용합니다.
- **1행:** 날짜 열 이름과 대상 코드(`G0` 등)를 둡니다.
- **16행 이후:** 해당 날짜의 값입니다.
- `--all`을 사용하려면 `list` 시트에 `Code`, `Length` 열도 필요합니다.

### 이벤트: `data/impacts.txt`

이름이 있는 이벤트 분석 입력입니다. 이벤트 파일의 파싱 규칙은 `src/event_impact.py`를 기준으로 맞춰야 합니다.

## 실행

입력 워크북의 시트·코드·데이터 범위를 준비한 뒤 저장소 루트에서 실행합니다.

```bash
python main.py --code G0 --target price --top 20 --align step_fill
python main.py --code G0 --target eps --max-lag 6
python main.py --code G0 --compare --window 52 --shift 4
```

| 옵션 | 설명 |
|---|---|
| `--code` | 대상 코드, 기본 `G0` |
| `--target` | `price` 또는 `eps` |
| `--align` | `step_fill` 또는 `midas` |
| `--date` | 분석 기준일. 생략 시 실행일 |
| `--compare` | 과거 구간과 중요도 비교 |
| `--window`, `--shift` | 비교 구간 길이·이동 폭(주) |
| `--all`, `--s_len` | 목록의 코드 길이 조건으로 여러 대상 실행 |
| `--no-save` | 보고서 파일 저장 생략 |

코드가 요구하는 최소 관측 수와 시계열 길이를 충족해야 합니다. `midas` 옵션의 존재만으로 모든 MIDAS 추정법·빈티지 정렬을 지원한다고 해석해서는 안 됩니다. 실제 정렬 구현을 확인하세요.

## 점수와 결과

현재 설정의 경로 가중치는 다음과 같습니다.

```text
final_score
  = 0.35 × continuous_score
  + 0.15 × irf_score
  + 0.35 × shock_score
  + 0.15 × threshold_score
  + 0.15 × Boruta 채택 여부
  + 0.05 × 정규화된 MI 점수
```

경로 내부의 세부 가중치는 `METRIC_WEIGHTS`를 따릅니다. 순위 비교용 합성점수이며 확률이 아닙니다. 보너스 때문에 최종 값이 1을 넘을 수도 있습니다. 과거 문서의 가중치와 다를 수 있으므로 재현 시 설정을 함께 보관해야 합니다.

결과는 기본적으로 `output/mfmea/` 아래 CSV·Excel 등의 보고서로 저장합니다. 유의확률·상관·SHAP은 표본과 모형에 의존합니다. 다중검정, 임계값 사후선택, 공시 시차, 데이터 수정, 국면 변화에 따른 과최적화를 고려하고 시간순 검증을 수행해야 합니다.

## 데이터와 라이선스

경제·시장 워크북은 사용자가 접근·분석할 권한이 있는 자료로 준비하세요. 공개 예제에는 합성 또는 배포 허용 데이터를 사용합니다. 코드 재사용 조건은 저장소의 `LICENSE`를 확인하세요.

## 기존 문서

기존 연구 설명은 [이전 README](README.previous.md)에 보존했습니다. 입력 규격과 현재 설정은 이 문서와 실제 코드를 기준으로 확인하세요.
