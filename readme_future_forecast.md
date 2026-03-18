# MFMEA Step 2 — Future Forecast 설계 및 구현 가이드

본 문서는 Step 1(지표 분석)의 산출물을 기반으로 `price` 또는 `eps`의 미래 수치를 예측하는 **2단계 예측 아키텍처**를 정의하고, 실제 구현 내용을 기술합니다.

---

## 실행 방법

```bash
python forecast.py --target price --top 10
python forecast.py --target eps   --top 8  --add_exp    # 지표별 정성 평가 입력
python forecast.py --target price --no-save             # 저장 생략
```

| 인자 | 기본값 | 설명 |
| :--- | :---: | :--- |
| `--target` | `price` | 예측 대상 (`price` / `eps`) |
| `--top` | `10` | 기본 모델에 사용할 상위 지표 수 |
| `--align` | `step_fill` | 주파수 정렬 방식 (`step_fill` / `midas`) |
| `--add_exp` | off | 지표별 정성 평가 프롬프트 활성화 |
| `--no-save` | off | Excel 저장 생략 |

---

## 📊 입력 데이터 구조 (Input Artifacts)

에이전트는 `output/` 디렉터리의 최신 파일을 자동으로 탐지하여 참조합니다.

| Artifact | 파일 패턴 | 활용 목적 |
| :--- | :--- | :--- |
| **`report_{target}`** | `mfmea_report_{target}_*.csv` | 장기 설명력 기준 지표 선택 (`final_score` 정렬) |
| **`comparison_{target}`** | `mfmea_comparison_{target}_*.csv` | `status == 'fallen'` 지표 제거 |
| **`event_impact_{target}`** | `mfmea_event_impact_{target}_*.xlsx` | `Equity_Summary` 시트의 이벤트별 `ar_+Xw` 조회 |
| **`impacts.txt`** | `data/impacts.txt` | 이벤트명 및 발생일 파싱 |

---

## 예측 파이프라인 (2-Step Architecture)

### Step 1: 장기 추세 모델링 (Base Trend Model)

**목적:** `{target}`의 기초 방향성 및 크기 추정

#### 1-1. 지표 선택 (`select_base_features`)
- `report_{target}`에서 `final_score` 기준 상위 `top_n × 2`개 후보 추출
- `comparison_{target}`에서 `status == 'fallen'` 지표 제거
- 동일 indicator 내 중복 제거 (최고 점수만 유지)
- 최종 `top_n`개 선택

#### 1-2. 모델 학습 (`fit_base_model`)
- 선택된 지표에 대해 **ElasticNetCV** (l1_ratio: 0.1/0.5/0.7/0.9/1.0, cv=5) 학습
- 종속변수: stationary 타겟 수익률 (`target` 컬럼, pct_change 변환값)
- 각 지표의 **EWMA (span=8주)** 를 현재 상태값으로 사용
- 지표별 trailing std 저장 (정성 평가 스케일링에 활용)

#### 1-3. 정성 평가 반영 (`--add_exp` 플래그)
- `--add_exp` 미사용 시: 모든 지표 arg = 0 (neutral), 프롬프트 생략
- `--add_exp` 사용 시: 각 지표에 대해 CLI에서 평가 입력 요청

  ```
  [-2 very bad | -1 bad | 0 neutral | +1 good | +2 very good]
  (Enter = skip → neutral)
  ```

- 반영 방식: 지표의 EWMA 값을 `arg × std(feat) × 0.5` 만큼 이동시킨 후 예측
  - 모델의 계수(coef) 부호가 방향을 결정: coef > 0이면 positive arg → 상승, coef < 0이면 반전
  - CLI에서 `[+]` / `[-]` 부호로 계수 방향을 표시

#### 1-4. 예측 생성
- 수정된 EWMA 값으로 모델 예측 → **주간 신호 (weekly_signal)** 산출
- 각 예측 윈도우에 대해 선형 누적: `base_return[window] = weekly_signal × (days / 7)`

---

### Step 2: 단기 변동성 보정 (Event Volatility Adjustment)

**목적:** 최근 이벤트의 잔존 영향력을 반영한 수익률 보정

#### 2-1. 이벤트 로드
- `data/impacts.txt` 파싱: `event_name : YYYY-MM-DD` 형식
- 동일 이벤트명이 복수 날짜로 존재할 수 있음 (예: `트럼프_관세전쟁`)

#### 2-2. 시간 이동 윈도우 매핑 (`_active_windows`)
- 각 예측 슬롯 `a` (1w/2w/4w/13w/26w), 해당 슬롯의 주 수 `n`에 대해:

  ```
  w1 = abs(-(days_since + n×7) // 7)
  ```

- `w1`을 기준으로 참조할 `ar_+Xw` 컬럼 결정:

  | w1 범위 | 참조 컬럼 |
  | :---: | :--- |
  | < 1 | `ar_+1w` |
  | 1 ≤ w1 < 2 | `ar_+2w` |
  | 2 ≤ w1 < 4 | `ar_+4w` |
  | 4 ≤ w1 < 13 | `ar_+13w` (→ `ar_+8w` 근사) |
  | 13 ≤ w1 < 26 | `ar_+26w` (→ `ar_+8w` 근사) |
  | ≥ 26 | 비활성 (제외) |

- **설계 의도:** 오늘 기준 1주 후를 예측할 때, 이벤트가 2주 전에 발생했다면 이벤트 관점에서의 유효 지평은 3주(경과 2주 + 예측 1주)이므로 `ar_+4w` 열을 참조함

#### 2-3. 충격 조정값 계산
- `Equity_Summary` 시트에서 해당 이벤트의 `ar_+Xw` (Z-score 단위) 조회
- **지수 감쇠 (half_life = 14일)** 적용: `decay = exp(-ln2 / 14 × days_since)`
- `target_series_std` (최근 `BASELINE_WINDOW=12`주 타겟 수익률의 표준편차)로 스케일링:

  ```
  event_adjustment[window] = Σ ar_+Xw × decay × target_series_std
  ```

- Z-score 단위의 `ar_` 값을 실제 수익률 단위로 변환하는 역할

---

## 최종 예측 산출

```
total_return[window] = base_return[window] + event_adjustment[window]
projected_value      = current_value × (1 + total_return)   # price
                     = current_value × (1 + total_return)   # eps (동일 방식)
```

**출력 윈도우:** 1w / 2w / 4w / 13w / 26w

---

## 출력 파일

`output/mfmea_forecast_{target}_{timestamp}.xlsx`

| 시트 | 내용 |
| :--- | :--- |
| `Forecast` | 윈도우별 base_return / event_adjustment / total_return / direction / magnitude / projected_value |
| `Events_Used` | 활성 이벤트 목록, 경과일, decay 가중치, 윈도우별 기여값 |
| `Model_Coefs` | ElasticNet 계수 전체 (절대값 내림차순) |

---

## 설계 원칙 요약

| 구성 요소 | 역할 |
| :--- | :--- |
| ElasticNet 계수 | 각 지표의 타겟에 대한 통계적 영향력 크기·방향 |
| EWMA (span=8w) | 지표의 현재 모멘텀 상태 |
| `--add_exp` 정성 평가 | 모델이 포착하지 못한 최신 정보를 EWMA 이동으로 반영 |
| ar_ × decay × std | 이벤트 충격의 잔존 강도를 수익률 단위로 환산 |
| 시간 이동 윈도우 | "오늘 기준" 남은 지평에 맞는 ar_ 열 선택으로 현실적 조정 |