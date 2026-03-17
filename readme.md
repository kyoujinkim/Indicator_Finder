I want to make a algorithm which make comparison between stock/sector's price with various economic indices. Economic indice is at the data/data.xlsx. Each sheet name contatins category of data and frequency. At the sheet first row is name of the index and data is begin at the 16th row. Final purpose of comparison is which index is relative to the stock's price and eps most, and how much.

Since data's characteristics is different, way of impact can be very different. Some will continously impact the equity's price or eps, and some will shock at the noticed moment only. Therefore you should use various technique to find out relations.



이 프로젝트는 한국 시장의 특수성(수출 의존도, 매크로 민감도)을 반영하여, 다양한 경제 지표와 주가/EPS 간의 상관관계 및 인과관계를 정교하게 분석하기 위한 프레임워크입니다.



사용자가 제시한 데이터 구조와 분석 목적을 바탕으로, \*\*퀀트 리서치 프로젝트\*\* 수준의 전문성을 갖춘 `README.md`로 재구성해 드립니다.



---



\# Multi-Frequency Macro-Equity Analytics (MFMEA)



본 프로젝트는 주가(Price) 및 기업 실적(EPS)에 유의미한 영향을 미치는 핵심 경제 지표(Core Economic Indicators)를 자동으로 식별하고, 지표별 영향력의 형태(지속적 영향 vs 일시적 충격)를 분석하는 알고리즘 설계를 목표로 합니다.



\## 1. Project Overview



단순한 상관관계를 넘어, 서로 다른 빈도(Daily, Monthly, Quarterly)를 가진 데이터 간의 \*\*시차(Lag) 효과\*\*와 \*\*충격 반응(Shock Response)\*\*을 정량화합니다.



\* \*\*Target:\*\* 개별 종목 주가, 섹터 지수, 분기별 EPS.

\* \*\*Features:\*\* `data/data.xlsx` 내의 범주별/주기별 경제 지표.

\* \*\*Core Objective:\*\* 지표별 설명력(Feature Importance) 산출 및 향후 방향성 예측을 위한 변수 선정 최적화.



---



\## 2. Data Specification



데이터는 `data/data.xlsx` 파일 내에 체계적으로 관리되며, 엔진은 다음 구조를 자동으로 파싱합니다.



\* \*\*Sheet Structure:\*\* 각 시트 명칭은 `\[Category]\_\[Frequency]` 형식을 따릅니다. (예: `Macro\_Monthly`, `Trade\_Daily`)

\* \*\*Header Logic:\*\* \* \*\*Row 1:\*\* 지표 명칭 (Index Name).

\* \*\*Rows 2-15:\*\* 메타데이터 및 설명 (Skip 대상).

\* \*\*Row 16 ~:\*\* 타임스탬프 및 수치 데이터 시작.





\* \*\*Handling Multi-frequency:\*\* 저빈도(분기 EPS) 데이터와 고빈도(일일 주가) 데이터 간의 정렬을 위해 \*\*MIDAS(Mixed-Data Sampling)\*\* 및 \*\*Step-fill\*\* 보간법을 선택적으로 적용합니다.



---



\## 3. Methodology: Triple-Path Analysis



지표의 성격에 따라 세 가지 분석 경로를 병렬로 수행합니다.



\### A. Continuous Impact Analysis (지속적 영향)



주가와 동행하거나 선행하며 장기적인 추세를 결정하는 지표를 분석합니다.



\* \*\*Techniques:\*\*

\* \*\*Granger Causality:\*\* 지표가 주가에 앞서 움직이는지 통계적 인과성 검정.

\* \*\*Transfer Entropy:\*\* 정보 전이의 방향성과 강도 측정 (비선형 관계 포착).

\* \*\*ElasticNet Regression:\*\* 다중공선성을 제거하며 핵심 변수 선택.



\### B. Shock \& Event Analysis (일시적 충격)



발표 시점에 시장에 강한 충격을 주고 사라지거나, 변동성을 확대시키는 지표를 분석합니다.



\* \*\*Techniques:\*\*

\* \*\*Impulse Response Function (IRF):\*\* 특정 지표의 충격이 몇 거래일 동안 주가에 잔류하는지 분석.

\* \*\*Event Study:\*\* 지표 발표 전후 $T \\pm n$ 기간의 초과수익률(Abnormal Return) 계산.

\* \*\*SHAP Value (Time-series):\*\* 머신러닝 모델에서 특정 시점의 지표가 예측값 변화에 기여한 정도를 수치화.



\### C. Threshold / Indicator-Function Analysis (임계값 트리거)



일부 지표는 연속적인 움직임이 아닌 **특정 임계값(threshold)을 돌파하는 순간**에만 주가에 영향을 미치는 지표 함수(indicator function)처럼 작동합니다. 예: PMI 50선 돌파, 장단기 금리 역전, 심리지표 극단 구간 진입.



\* \*\*Optimal Threshold Search:\*\* 분위수 그리드 탐색(quantile sweep)을 통해 지표를 두 레짐(above/below)으로 분할할 때 타겟 수익률 차이가 최대화되는 임계값 $T^*$를 자동으로 산출합니다. 레짐 분리의 통계적 유의성은 Welch t-test로 검증합니다.

\* \*\*Crossing Detection:\*\* $T^*$에 대한 상향 돌파(up-crossing)와 하향 이탈(down-crossing) 시점을 탐지합니다.

\* \*\*Conditional Forward Returns:\*\* 돌파 이벤트 이후 $h = \{1, 3, 5, 10\}$ 기간의 타겟 누적 수익률을 측정합니다.

\* \*\*Signal Quality Metrics:\*\*

\  \* \*\*Hit Rate:\*\* 돌파 방향과 타겟 수익률 부호가 일치하는 비율.

\  \* \*\*Information Ratio:\*\* 방향-조정 수익률(direction-signed return)의 평균/표준편차.

\  \* \*\*Directional Asymmetry:\*\* 상향 돌파와 하향 이탈의 적중률 차이 — 비대칭성이 클수록 방향성 정보가 높음.

\  \* \*\*Trigger Score:\*\* 위 지표를 종합한 $[0, 1]$ 복합 점수.





\## 4. Feature Selection Strategy

방대한 지표 풀에서 노이즈를 제거하고 유효한 신호만 추출하기 위해 다단계 필터링을 수행합니다.

\### A. Statistical \& Causal Filtering

Mutual Information (MI): 지표와 타겟 간의 비선형적 의존성을 측정합니다.

Transfer Entropy: 지표 $X$가 타겟 $Y$의 미래를 예측하는 데 미치는 정보 전이량을 계산하여 인과 방향을 확정합니다.



\### B. Iterative Selection

Boruta Algorithm: 무작위로 섞인 가짜 지표(Shadow Features)보다 성능이 우수한 실제 지표만을 엄격하게 선별합니다.

Regime-Aware Selection: 시장 국면(상승/하락/횡보)에 따라 지표의 중요도를 동적으로 재산출합니다.





---



\## 5. System Architecture



1\. \*\*Preprocessing Engine\*\* (`src/preprocessor.py`): 엑셀 시트별 빈도 처리, 주간 리샘플링, 지표 유형별 정상화 변환 (Look-ahead bias 방지).

2\. \*\*Continuous Analysis\*\* (`src/continuous_analysis.py`): Granger Causality, Transfer Entropy, ElasticNet.

3\. \*\*Shock Analysis\*\* (`src/shock_analysis.py`): VAR-IRF, Event Study, SHAP (XGBoost).

4\. \*\*Threshold Analysis\*\* (`src/threshold_analysis.py`): Optimal threshold search, crossing detection, conditional forward returns, signal quality metrics (hit rate, info ratio, asymmetry, trigger score).

5\. \*\*Feature Discovery\*\* (`src/feature_selector.py`): Boruta, Mutual Information, Regime-Aware importance.

6\. \*\*Relation Scorer\*\* (`src/relation_scorer.py`): 세 경로(Continuous / Shock / Threshold) 점수를 가중합산하고 Boruta 보너스 및 MI 점수를 반영하여 최종 랭킹 산출.

7\. \*\*Reporting\*\* (`src/reporter.py`): 콘솔 요약 + CSV + Excel (다중 시트) 저장.



\### Relation Scoring Formula

```
Final Score = 0.40 × Continuous_Score
            + 0.30 × Shock_Score
            + 0.30 × Threshold_Score
            + 0.15  [if Boruta-confirmed]
            + 0.05 × MI_Score_norm

Threshold_Score = 0.40 × |regime_diff|_norm × (1 - pval)
                + 0.40 × trigger_score
                + 0.20 × asymmetry_norm
```

--

