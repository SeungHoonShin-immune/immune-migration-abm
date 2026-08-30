# 병원체의 공간 분포와 화학주성 신호가 면역세포 이동 전략의 효율에 미치는 영향

**정상적 환경과 염증성 환경의 Agent-Based Model 비교**

제68회 서울과학전람회 출품작 · 생물 부문

---

## 연구 개요

면역세포는 감염이 없는 평상시에는 조직을 방향성 없이 순찰하다가, 감염이 발생하면 케모카인 농도구배를 따라 병소로 이동한다. 두 방식이 모두 유지되는 이유를 확인하기 위해, 이동 규칙만을 독립변인으로 조작하고 나머지 조건을 모두 통제한 Agent-Based Model을 구현하여 두 전략의 효율을 정량적으로 비교하였다.

성질이 대조적인 두 병원체(인플루엔자 A, 폐렴사슬알균)에 동일한 모형을 적용한 결과, **두 이동 전략의 우열이 병원체에 따라 역전**되는 것을 확인하였다.

### 두 실험 조건

| 구분 | 정상적 환경 (Homeostatic) | 염증성 환경 (Inflammatory) |
|---|---|---|
| 생리적 상태 | 감염·손상이 없는 항상성 상태 | 케모카인 농도장이 형성된 감염 상태 |
| 이동 규칙 | 방향을 균일난수로 결정 | 자기 수용체 축의 농도가 최대인 방향으로 이동 |
| 신호 감지 | 감지하지 않음 (0%) | 전적으로 감지 |

두 조건은 이동 규칙만 다르며, 격자 구조·시간 축척·면역세포 초기 개체수·병원체 파라미터·개체수 축소비율(κ = 1/1000)은 완전히 동일하게 통제하였다.

---

## 주요 결과

각 조건 5회 독립 시행. Welch t-검정과 Mann-Whitney U 검정 중 보수적인 p값을 채택하였다.

### 인플루엔자 A (H1N1)

| 지표 | 정상적 환경 | 염증성 환경 | 우세 |
|---|---:|---:|:---:|
| 최대 병원체 수 | **412,545** | 481,097 | 정상적 |
| 식세포 포식 제거량 | **507,493** | 230,120 | 정상적 |
| 누적 조직손상 | **1.690** | 1.729 | 정상적 |
| 병원체 완전 제거 | 11.13일 | 11.06일 | 차이 없음 |

→ 13개 지표 중 정상적 환경 8개, 염증성 환경 3개, 차이 없음 2개

### 폐렴사슬알균 (*Streptococcus pneumoniae*)

| 지표 | 정상적 환경 | 염증성 환경 | 우세 |
|---|---:|---:|:---:|
| 최대 병원체 수 | 94,571 | **26,185** | 염증성 |
| 병원체 총량 AUC | 276,958 | **54,062** | 염증성 |
| 감염 종식일 | 4.46일 | **4.22일** | 염증성 |
| 식세포–세균 접촉 횟수 | 83,284 | **2,243,899** | 염증성 |
| 누적 조직손상 | **0.020** | 0.030 | 정상적 |

→ 10개 지표 중 염증성 환경 8개, 정상적 환경 2개

### 우열이 역전된 이유

| 결정 조건 | 인플루엔자 A | 폐렴사슬알균 |
|---|---|---|
| 공간 확산 | σ = 30칸/tick (조직 전역 균일 확산) | σ = 0.6칸/tick (비운동성, 미세군락 집중) |
| 자체 신호 | 없음 (자유 바이러스는 침묵) | fMLF 방출 (표적 = 신호원) |
| 주 제거 경로 | 세포자멸 (탐색과 무관) | 식균작용 (탐색에 의존) |

표적이 흩어져 있고 신호를 내지 않으면 무작위 순찰이 유리하고, 표적이 뭉쳐 있고 신호를 방출하면 주화성 이동이 유리하다.

---

## 파일 구성

| 파일 | 내용 | 줄 수 |
|---|---|---:|
| `immune_influenza_random.py` | 인플루엔자 A 모형 — 정상적 환경 (무작위 순찰) | 2,389 |
| `immune_influenza_inflammatory.py` | 인플루엔자 A 모형 — 염증성 환경 (주화성 유도 이동) | 2,365 |
| `immune_pneumococcus.py` | 폐렴사슬알균 모형 — 두 조건 통합 | 1,342 |
| `run_replicates_influenza.py` | 인플루엔자 A 반복 시행 스크립트 | 44 |
| `run_replicates_pneumococcus.py` | 폐렴사슬알균 반복 시행 스크립트 | 50 |

두 조건의 차이가 구현된 부분은 각 파일의 `MovementEngine` 클래스이다.

---

## 실행 방법

### 환경

```
Python 3.12
numpy, scipy, matplotlib
```

```bash
pip install numpy scipy matplotlib
```

### 단일 실행

```bash
# 인플루엔자 A — 정상적 환경
python immune_influenza_random.py

# 인플루엔자 A — 염증성 환경
python immune_influenza_inflammatory.py

# 폐렴사슬알균 (두 조건을 인자로 선택)
python immune_pneumococcus.py
```

### 반복 시행

```bash
python run_replicates_influenza.py        # 각 조건 5회
python run_replicates_pneumococcus.py     # 각 조건 5회, 감염 종식까지
```

결과는 JSON 파일로 저장되며, 일별 시계열과 요약 지표를 모두 포함한다.

### 재현성

본 연구에 사용한 난수 시드는 다음과 같다.

```
700000, 700037, 700074, 700111, 700148
```

두 조건에 동일한 시드를 사용하여, 초기 배치가 같은 상태에서 이동 규칙만 달라지도록 짝지어 설계하였다.

---

## 모형 구조

### 축척

| 항목 | 설정 |
|---|---|
| 격자 | 1000 × 1000 (100만 칸) = 상기도 상피 15mm × 15mm |
| 격자 한 칸 | 15 μm (상피세포 1개) |
| 시간 | 1 tick = 6분, 1일 = 240 tick |
| 개체수 축소비율 | κ = 1/1000 |

### 화학주성 축 (수용체별 분리 구현)

| 축 | 신호물질 | 수용체 | 반응 세포 | 계층 |
|---|---|---|---|---|
| 1 | fMLF | FPR1 | 중성구, 대식세포 | 말단표적 |
| 2 | C5a | C5aR1 | 중성구, 단핵구 | 말단표적 |
| 3 | CXCL8 (IL-8) | CXCR2 | 중성구 | 중간매개 |
| 4 | CCL2 (MCP-1) | CCR2 | 단핵구/대식세포 | 중간매개 |
| 5 | CXCL9/10 | CXCR3 | NK, 조력T, 살해T | — |
| 6 | CXCL13 | CXCR5 | B세포 | — |

말단표적 신호(fMLF, C5a)가 역치를 넘으면 중간매개 신호(CXCL8, CCL2)를 무시하는 화학주성 계층 구조를 반영하였다.

---

## 파라미터 출처

모든 파라미터는 선행 연구에서 도출하였으며, 각 값의 출처를 코드 내 주석에 태그로 표시하였다. 논문에서 직접 인용한 실측값과, 정성적 기전만 문헌에 근거하고 정량값은 모형에서 설정한 가정값을 구분하여 표기하였다.

### 인플루엔자 A

| 파라미터 | 값 | 출처 |
|---|---|---|
| 잠복기 / 생산 기간 / 감염세포 수명 | 6h / 5h / 11h | Baccam et al. (2006) |
| 유리 바이러스 반감기 | 3시간 | Baccam et al. (2006) |
| 바이러스 크기 | 80–120 nm | Harris et al. (2006) |
| 배출 기간 | 평균 4.8일 | Carrat et al. (2008) |
| 초기 접종량 | 10⁵ TCID₅₀ | Memoli et al. (2015) |

### 폐렴사슬알균

| 파라미터 | 값 | 출처 |
|---|---|---|
| 폐 내 배가시간 | 56분 | Jose et al. (2015) |
| 폐포대식세포 제거 반감기 | 42분 | Jose et al. (2015) |
| 초기 접종량 | 1×10⁶ CFU | Lin et al. (2020) |
| 최대 균량 | 1×10⁸ CFU | Hamilton et al. (2019) |
| 호중구 포식 용량 | 약 50 CFU/세포 | Rubio et al. (2023) |
| 옵소닌 요구량 | 혈청 40% 이상 | Gordon et al. (1980) |
| 협막의 포식 저항 | 다중 기전 저해 | Hyams et al. (2010) |

### 화학주성 및 세포 이동

| 파라미터 | 출처 |
|---|---|
| 화학주성 계층 (말단표적 > 중간매개) | Heit et al. (2002) |
| 호중구의 무작위 → 방향성 이동 전환 | Lämmermann (2016) |
| 호중구 군집 형성과 조직 손상 | Poplimont et al. (2020) |
| 조직 내 백혈구 이동 속도 | Friedl & Weigelin (2008) |
| 맹목 탐색과 구배 감지 탐색의 효율 | Metzner et al. (2019) |

---

## 참고문헌

1. Baccam, P.; Beauchemin, C.; Macken, C. A.; Hayden, F. G.; Perelson, A. S. Kinetics of Influenza A Virus Infection in Humans. *J. Virol.* **2006**, *80* (15), 7590–7599.
2. Carrat, F.; Vergu, E.; Ferguson, N. M.; et al. Time Lines of Infection and Disease in Human Influenza. *Am. J. Epidemiol.* **2008**, *167* (7), 775–785.
3. Harris, A.; Cardone, G.; Winkler, D. C.; et al. Influenza Virus Pleiomorphy Characterized by Cryoelectron Tomography. *PNAS* **2006**, *103* (50), 19123–19127.
4. Memoli, M. J.; Czajkowski, L.; Reed, S.; et al. Validation of the Wild-Type Influenza A Human Challenge Model. *Clin. Infect. Dis.* **2015**, *60* (5), 693–702.
5. Jose, R. J.; Williams, A. E.; Chambers, R. C.; Brown, J. S.; et al. Importance of Bacterial Replication and Alveolar Macrophage-Independent Clearance Mechanisms during Early Lung Infection with *Streptococcus pneumoniae*. *Infect. Immun.* **2015**, *83* (4), 1181–1189.
6. Lin, J.; Zhu, L.; Lau, G. W. *Streptococcus pneumoniae* Elaborates Persistent and Prolonged Competent State during Pneumonia-Derived Sepsis. *Infect. Immun.* **2020**, *88* (4), e00919-19.
7. Hamilton, J. A.; Nguyen, V. T.; Wilson, C.; et al. Clinically Relevant Model of Pneumococcal Pneumonia, ARDS, and Nonpulmonary Organ Dysfunction in Mice. *Am. J. Physiol. Lung Cell. Mol. Physiol.* **2019**, *317* (5), L659–L675.
8. Rubio, A. J.; Bakker, M. G.; Bhatnagar, S.; et al. Model-Based Assessment of Neutrophil-Mediated Phagocytosis and Digestion of Bacteria. *CPT Pharmacometrics Syst. Pharmacol.* **2023**, *12* (12), 1934–1946.
9. Hyams, C.; Camberlein, E.; Cohen, J. M.; Bax, K.; Brown, J. S. The *Streptococcus pneumoniae* Capsule Inhibits Complement Activity and Neutrophil Phagocytosis by Multiple Mechanisms. *Infect. Immun.* **2010**, *78* (2), 704–715.
10. Gordon, D. L.; Rice, J.; Finlay-Jones, J. J.; et al. Phagocytosis by Human Alveolar Macrophages and Neutrophils. *J. Infect. Dis.* **1980**, *141* (6), 718–724.
11. Heit, B.; Tavener, S.; Raharjo, E.; Kubes, P. An Intracellular Signaling Hierarchy Determines Direction of Migration in Opposing Chemotactic Gradients. *J. Cell Biol.* **2002**, *159* (1), 91–102.
12. Lämmermann, T. In the Eye of the Neutrophil Swarm. *J. Leukoc. Biol.* **2016**, *100* (1), 55–63.
13. Poplimont, H.; Georgantzoglou, A.; Boulch, M.; et al. Neutrophil Swarming in Damaged Tissue Is Orchestrated by Connexins and Cooperative Calcium Alarm Signals. *Curr. Biol.* **2020**, *30* (14), 2761–2776.
14. Metzner, C.; Mark, C.; Steinwachs, J.; et al. On the Efficiency of Chemotactic Pursuit. *Sci. Rep.* **2019**, *9*, 14119.
15. Friedl, P.; Weigelin, B. Interstitial Leukocyte Migration and Immune Function. *Nat. Immunol.* **2008**, *9* (9), 960–969.
16. Folcik, V. A.; An, G. C.; Orosz, C. G. The Basic Immune Simulator. *Theor. Biol. Med. Model.* **2007**, *4*, 39.
17. Parr, A.; Anderson, N. R.; Hammer, D. A. A Simulation of the Random and Directed Motion of Dendritic Cells in Chemokine Fields. *PLoS Comput. Biol.* **2019**, *15* (10), e1007295.
18. Lu, Y. J.; Gross, J.; Bogaert, D.; Finn, A.; et al. Interleukin-17A Mediates Acquired Immunity to Pneumococcal Colonization. *PLoS Pathog.* **2008**, *4* (9), e1000159.



---

## 라이선스

교육 및 연구 목적으로 자유롭게 사용할 수 있다.
