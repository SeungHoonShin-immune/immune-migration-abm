# -*- coding: utf-8 -*-
"""
=============================================================================
 immune_bacteria_abm.py
 세균(Streptococcus pneumoniae) 감염 - 면역반응 Agent-Based Model
 면역세포 이동방식 비교 : 무작위 이동(Random Walk) vs 체계적 이동(Chemotaxis)
=============================================================================

【병원체】 Streptococcus pneumoniae (폐렴사슬알균), 협막 보유 침습성 혈청형
          - 그람양성, 통성혐기성, 사슬알균속
          - ★비운동성(non-motile) : 편모가 없어 스스로 헤엄치지 못한다
          - 세포외 증식(extracellular) : 숙주세포 안으로 들어가지 않는다
          - 협막 다당(capsular polysaccharide)이 보체 침착을 방해한다

【바이러스 모델과의 근본적 차이 — 본 실험을 세균으로 바꾼 이유】
  1) 감염세포가 없다 -> 세포자멸(apoptosis)이 없다
     인플루엔자 모델에서는 감염세포 제거의 98%가 세포자멸이었고, 이는
     면역세포의 '탐색'과 완전히 무관한 경로였다. 따라서 이동방식의 효과가
     구조적으로 희석되었다. 세균에서는 제거의 거의 전부가 식균작용이므로
     '탐색 효율'이 결과에 직접 반영된다.
  2) 병원체가 스스로 유인물질을 방출한다
     세균은 N-포르밀 펩타이드(fMLF)를 배출한다. 이는 호중구의 FPR1에
     작용하는 최강 화학주성 인자다. 즉 '표적이 곧 신호원'이다.
     자유 바이러스 입자는 어떤 화학주성 신호도 내지 않는다.
  3) 확산 속도가 압도적으로 느리다
     바이러스는 점액 수송과 확산으로 조직 전역에 퍼진다(sigma=30칸/tick).
     폐렴사슬알균은 비운동성이라 분열한 자리에 머물며 미세군락을 형성한다.
     -> 표적이 공간적으로 '뭉쳐' 있으므로 주화성이 유리할 조건이 된다.

【핵심 가설】
  인플루엔자에서 체계적 이동이 우세하지 못했던 원인이 '표적의 공간 분산'
  이었다면, 표적이 국소에 뭉쳐 있고 스스로 신호를 내는 세균 감염에서는
  체계적 이동이 무작위 이동보다 유의하게 우세해야 한다.

【축소비율】 kappa = 1/1000 (바이러스 모델과 동일 유지)
=============================================================================
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# SECTION 1.  참고문헌
# =============================================================================


@dataclass(frozen=True)
class Ref:
    key: str
    authors: str
    title: str
    journal: str
    year: int
    doi: str = ""
    pmid: str = ""

    def cite(self) -> str:
        tail = f" doi:{self.doi}" if self.doi else ""
        tail += f" PMID:{self.pmid}" if self.pmid else ""
        return f"{self.authors} ({self.year}). {self.title}. {self.journal}.{tail}"


REFS: Dict[str, Ref] = {
    # ---------- 병원체 동역학 ----------
    "kadioglu2015": Ref(
        "kadioglu2015",
        "Jose RJ, Williams AE, Chambers RC, Brown JS, et al.",
        "Importance of bacterial replication and alveolar macrophage-independent "
        "clearance mechanisms during early lung infection with "
        "Streptococcus pneumoniae",
        "Infection and Immunity 83(4):1181-1189", 2015,
        "10.1128/IAI.02788-14", "25583525"),
    "lin2020": Ref(
        "lin2020", "Lin J, Zhu L, Lau GW",
        "Streptococcus pneumoniae elaborates persistent and prolonged competent "
        "state during pneumonia-derived sepsis",
        "Infection and Immunity 88(4):e00919-19", 2020,
        "10.1128/IAI.00919-19", "31988172"),
    "reppe2019": Ref(
        "reppe2019",
        "Hamilton JA, Nguyen VT, Wilson C, Mulchandani N, et al.",
        "Clinically relevant model of pneumococcal pneumonia, ARDS, and "
        "nonpulmonary organ dysfunction in mice",
        "American Journal of Physiology - Lung Cellular and Molecular "
        "Physiology 317(5):L659-L675", 2019,
        "10.1152/ajplung.00132.2019", "31461312"),
    "hava2003": Ref(
        "hava2003", "Hava DL, Camilli A",
        "Large-scale identification of serotype 4 Streptococcus pneumoniae "
        "virulence factors",
        "Molecular Microbiology 45(5):1389-1406", 2002,
        "10.1046/j.1365-2958.2002.03106.x", "12207705"),

    # ---------- 식균작용 / 보체 ----------
    "rubio2023": Ref(
        "rubio2023",
        "Rubio AJ, Bakker MG, Bhatnagar S, Sunderhaus A, et al.",
        "Model-based assessment of neutrophil-mediated phagocytosis and "
        "digestion of bacteria across in vitro and in vivo studies",
        "CPT: Pharmacometrics & Systems Pharmacology 12(12):1934-1946", 2023,
        "10.1002/psp4.13045", "37881153"),
    "hyams2010": Ref(
        "hyams2010", "Hyams C, Camberlein E, Cohen JM, Bax K, Brown JS",
        "The Streptococcus pneumoniae capsule inhibits complement activity and "
        "neutrophil phagocytosis by multiple mechanisms",
        "Infection and Immunity 78(2):704-715", 2010,
        "10.1128/IAI.00881-09", "19948837"),
    "gordon1980": Ref(
        "gordon1980", "Gordon DL, Rice J, Finlay-Jones JJ, McDonald PJ, et al.",
        "Phagocytosis by human alveolar macrophages and neutrophils: qualitative "
        "differences in the opsonic requirements for uptake of Staphylococcus "
        "aureus and Streptococcus pneumoniae in vitro",
        "Journal of Infectious Diseases 141(6):718-724", 1980,
        "10.1093/infdis/141.6.718", "7352714"),
    "dalia2010": Ref(
        "dalia2010", "Dalia AB, Standish AJ, Weiser JN",
        "Three surface exoglycosidases from Streptococcus pneumoniae, NanA, "
        "BgaA, and StrH, promote resistance to opsonophagocytic killing by "
        "human neutrophils",
        "Infection and Immunity 78(5):2108-2116", 2010,
        "10.1128/IAI.01125-09", "20160017"),
    "bohlson2023": Ref(
        "bohlson2023",
        "Palmer A, Klein K, Cabral-Fernandes L, Gomes-Neto JF, et al.",
        "Purified complement C3b triggers phagocytosis and activation of human "
        "neutrophils via complement receptor 1",
        "Scientific Reports 13:1-15", 2023,
        "10.1038/s41598-022-27279-4", "36609671"),

    # ---------- ★화학주성 계층 (본 모델의 핵심 기전) ----------
    "heit2002": Ref(
        "heit2002", "Heit B, Tavener S, Raharjo E, Kubes P",
        "An intracellular signaling hierarchy determines direction of migration "
        "in opposing chemotactic gradients",
        "Journal of Cell Biology 159(1):91-102", 2002,
        "10.1083/jcb.200202114", "12370241"),
    "wang2016": Ref(
        "wang2016", "Wang X, Qin W, Zhang Y, Zhang H, Sun B",
        "Endotoxin promotes neutrophil hierarchical chemotaxis via the "
        "p38-membrane receptor pathway",
        "Oncotarget 7(45):73431-73444", 2016,
        "10.18632/oncotarget.12093", "27655688"),
    "weiss2020": Ref(
        "weiss2020",
        "Weiss E, Hanzelmann D, Fehlhaber B, Klos A, von Loewenich FD, et al.",
        "Formyl-peptide receptor activation enhances phagocytosis of "
        "community-acquired methicillin-resistant Staphylococcus aureus",
        "Journal of Infectious Diseases 221(4):668-678", 2020,
        "10.1093/infdis/jiz498", "31573603"),
    "schepetkin2025": Ref(
        "schepetkin2025", "Cattaneo F, Ammendola R, et al.",
        "The N-formyl peptide receptors: much more than chemoattractant "
        "receptors. Relevance in health and disease",
        "Frontiers in Immunology 16:1568629", 2025,
        "10.3389/fimmu.2025.1568629", ""),
    "vanderlinden2017": Ref(
        "vanderlinden2017",
        "van der Linden M, Kuijpers TW, et al.",
        "Differential neutrophil chemotactic response towards IL-8 and "
        "bacterial N-formyl peptides in term newborn infants",
        "Pediatric Research 81(1):142-147", 2017,
        "10.1038/pr.2016.196", "27673422"),

    # ---------- 수용체 발현 ----------
    "rudd2019": Ref(
        "rudd2019", "Rudd JM, Pulavendran S, Ashar HK, Ritchey JW, et al.",
        "Neutrophils induce a novel chemokine receptors repertoire during "
        "influenza pneumonia",
        "Frontiers in Cellular and Infection Microbiology 9:108", 2019,
        "10.3389/fcimb.2019.00108", "31041196"),
    "lin2008": Ref(
        "lin2008", "Lin KL, Suzuki Y, Nakano H, Ramsburg E, Gunn MD",
        "CCR2+ monocyte-derived dendritic cells and exudate macrophages produce "
        "influenza-induced pulmonary immune pathology and mortality",
        "Journal of Immunology 180(4):2562-2572", 2008,
        "10.4049/jimmunol.180.4.2562", "18250467"),
    "groom2011": Ref(
        "groom2011", "Groom JR, Luster AD",
        "CXCR3 ligands: redundant, collaborative and antagonistic functions",
        "Immunology and Cell Biology 89(2):207-215", 2011,
        "10.1038/icb.2010.158", "21221121"),
    "guo2005": Ref(
        "guo2005", "Guo RF, Ward PA",
        "Role of C5a in inflammatory responses",
        "Annual Review of Immunology 23:821-852", 2005,
        "10.1146/annurev.immunol.23.021704.115835", "15771587"),
    "okada2002": Ref(
        "okada2002", "Okada T, Ngo VN, Ekland EH, Forster R, Lipp M, et al.",
        "Chemokine requirements for B cell entry to lymph nodes and "
        "Peyer's patches",
        "Journal of Experimental Medicine 196(1):65-75", 2002,
        "10.1084/jem.20020201", "12093871"),

    # ---------- Th17 / 점막 세균면역 ----------
    "zhang2018": Ref(
        "zhang2018", "Zhang Z, Clarke TB, Weiser JN",
        "Cellular effectors mediating Th17-dependent clearance of pneumococcal "
        "colonization in mice",
        "Journal of Clinical Investigation 119(7):1899-1909", 2009,
        "10.1172/JCI36731", "19509469"),
    "lu2008": Ref(
        "lu2008", "Lu YJ, Gross J, Bogaert D, Finn A, et al.",
        "Interleukin-17A mediates acquired immunity to pneumococcal "
        "colonization",
        "PLoS Pathogens 4(9):e1000159", 2008,
        "10.1371/journal.ppat.1000159", "18802458"),

    # ---------- 세포 이동 물리량 (바이러스 모델과 공유) ----------
    "friedl2008": Ref(
        "friedl2008", "Friedl P, Weigelin B",
        "Interstitial leukocyte migration and immune function",
        "Nature Immunology 9(9):960-969", 2008,
        "10.1038/ni.f.212", "18711433"),
    "miller2002": Ref(
        "miller2002", "Miller MJ, Wei SH, Parker I, Cahalan MD",
        "Two-photon imaging of lymphocyte motility and antigen response in "
        "intact lymph node",
        "Science 296(5574):1869-1873", 2002,
        "10.1126/science.1070051", "12016203"),
    "germain2012": Ref(
        "germain2012", "Germain RN, Robey EA, Cahalan MD",
        "A decade of imaging cellular motility and interaction dynamics in the "
        "immune system",
        "Science 336(6089):1676-1681", 2012,
        "10.1126/science.1221063", "22745423"),
}

MODEL_ASSUMPTION = "모델 가정값"


# =============================================================================
# SECTION 2.  실제 측정값 (논문값) — 모든 파라미터의 출처
# =============================================================================

REAL: Dict[str, float] = {
    # ---- 공간 / 시간 축소 ----
    "kappa": 1.0 / 1000.0,          # 개체수 축소비율 (바이러스 모델과 동일)
    "grid_side_mm": 15.0,           # 격자 한 변의 실제 길이
    "tick_min": 6.0,                # 1 tick = 6분
    "epi_cell_um": 15.0,            # 상피세포 1개의 대표 크기

    # ---- 병원체 (Streptococcus pneumoniae) ----
    # kadioglu2015 : 저용량 접종 시 폐 내 배가시간 56분
    "spn_doubling_min": 56.0,
    # kadioglu2015 / lin2020 / reppe2019 : 마우스 비강내 접종 1e6~1e8 CFU
    "spn_inoculum_cfu": 1.0e6,
    # lin2020 : 빈사 상태 마우스의 폐 내 균량 1e8 CFU
    "spn_max_burden_cfu": 1.0e8,
    # kadioglu2015 : 폐포대식세포 의존성 제거 반감기 42분
    "spn_am_clearance_halflife_min": 42.0,
    "spn_cell_um": 1.0,             # 구균 직경 약 0.8~1.0 um

    # ---- 식균작용 ----
    # rubio2023 : 호중구 1개의 포식 용량 약 50 CFU
    "neut_capacity_cfu": 50.0,
    # gordon1980 : 폐렴사슬알균은 40% 이상 고농도 혈청(보체) 없이는 포식 불가
    "spn_opsonin_requirement": 0.40,
    # hyams2010 : 협막이 보체 침착과 호중구 포식을 다중 기전으로 저해
    "capsule_phago_resistance": 0.85,

    # ---- 화학주성 계층 (heit2002, wang2016) ----
    # 말단표적(fMLF, C5a)이 중간매개(IL-8, LTB4)보다 우선한다
    "endtarget_priority": 1.0,

    # ---- 면역세포 이동 물리량 ----
    "neut_speed_um_min": 12.0,      # friedl2008
    "mono_speed_um_min": 4.0,
    "nk_speed_um_min": 8.0,
    "t_speed_um_min": 11.0,         # miller2002
    "b_speed_um_min": 6.0,

    # ---- 신호물질 ----
    "fmlf_D_um2_s": 300.0,          # 저분자 펩타이드, 빠른 확산 (모델 추정)
    "fmlf_halflife_min": 20.0,      # 조직 펩티다제에 의한 분해 (모델 추정)
    "c5a_halflife_min": 1.0,        # guo2005 : 카르복시펩티다제N에 의해 초고속 분해
    "cxcl8_halflife_h": 2.0,
    "ccl2_halflife_h": 4.0,
    "cxcr3l_halflife_h": 6.0,
    "cxcl12_halflife_h": 8.0,

    # ---- 후천면역 (항협막 IgG) ----
    "igm_detect_day": 5.0,
    "igg_detect_day": 9.0,
    "igm_halflife_d": 5.0,
    "igg_halflife_d": 21.0,
    "th17_peak_day": 7.0,           # lu2008 / zhang2018
}


# =============================================================================
# SECTION 3.  축척 변환
# =============================================================================


class Scale:
    GRID = 1000
    FIELD_BIN = 5
    FIELD_N = GRID // FIELD_BIN          # 200 x 200 화학신호장
    EPI_PERIOD = 25                      # 상피 밴드 주기(칸)
    TICKS_PER_HOUR = 60.0 / REAL["tick_min"]        # 10
    TICKS_PER_DAY = int(24 * TICKS_PER_HOUR)        # 240
    UM_PER_CELL = REAL["grid_side_mm"] * 1000.0 / GRID   # 15 um/칸
    KAPPA = REAL["kappa"]

    @staticmethod
    def per_tick_from_halflife_h(h: float) -> float:
        return 1.0 - 0.5 ** (1.0 / (h * Scale.TICKS_PER_HOUR))

    @staticmethod
    def per_tick_from_halflife_min(m: float) -> float:
        return 1.0 - 0.5 ** (1.0 / (m / REAL["tick_min"]))

    @staticmethod
    def speed_cells_per_tick(um_per_min: float) -> float:
        return um_per_min * REAL["tick_min"] / Scale.UM_PER_CELL

    @staticmethod
    def agents_from_cfu(cfu: float) -> int:
        return max(1, int(round(cfu * Scale.KAPPA)))

    @staticmethod
    def division_prob_per_tick(doubling_min: float) -> float:
        """배가시간 -> tick 당 분열 확률 (지수증식 등가)"""
        ticks = doubling_min / REAL["tick_min"]
        return 2.0 ** (1.0 / ticks) - 1.0


print(f"[축척] 1 tick = {REAL['tick_min']:.0f}분, 1일 = {Scale.TICKS_PER_DAY} tick, "
      f"1칸 = {Scale.UM_PER_CELL:.1f} um, kappa = 1/{int(1/Scale.KAPPA)}")


# =============================================================================
# SECTION 4.  파라미터
# =============================================================================

DIR8 = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
                 (0, -1), (1, -1)], dtype=np.float32)
DIR8 /= np.linalg.norm(DIR8, axis=1, keepdims=True)
DIR8_OFF = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


@dataclass
class Params:
    max_days: int = 25
    seed: int = 20260809

    # ---------- 병원체 ----------
    inoculum: int = Scale.agents_from_cfu(REAL["spn_inoculum_cfu"])      # 1000
    carrying_capacity: int = Scale.agents_from_cfu(REAL["spn_max_burden_cfu"])
    div_p: float = Scale.division_prob_per_tick(REAL["spn_doubling_min"])
    # ★비운동성: 딸세포는 모세포 바로 옆에 놓인다 (미세군락 형성)
    daughter_sigma: float = 1.6         # 칸
    drift_sigma: float = 0.60           # 점액섬모 수송에 의한 미세 이동 (칸/tick)
    # ★상주 폐포대식세포(AM)의 제거는 개체수가 유한하므로 고균량에서 포화한다.
    #   저균량에서는 논문값(반감기 42분)에 근접하고, 고균량에서는 AM이 압도되어
    #   AM 비의존 기전(중성구 동원)이 필요해진다 (Jose 등, 2015).
    am_clearance_p: float = Scale.per_tick_from_halflife_min(
        REAL["spn_am_clearance_halflife_min"])                            # 0.0943
    am_saturation_n: float = 260.0     # 포화 상수 (agent)
    # 협막 저항 (hyams2010)
    capsule_resist: float = REAL["capsule_phago_resistance"]
    # ★영양 면역(nutritional immunity): 염증 시 호중구가 분비하는 칼프로텍틴
    #   (S100A8/A9)이 Mn·Zn을 격리하여 세균 증식을 억제한다. 56분 배가시간은
    #   저용량 접종 초기(무염증 상태)의 측정값이므로, 염증이 진행되면
    #   실효 증식률이 감소하는 것을 반영한다.
    nutritional_immunity: float = 0.78
    ops_decay: float = 0.006            # 표면 재생/보체 조절인자에 의한 옵소닌 소실
    # 조직 부착/침습에 의한 상피 손상
    epi_damage_p: float = 0.0016       # 균 1개가 접한 상피칸을 손상시킬 확률

    # ---------- 식균작용 ----------
    neut_capacity: int = int(REAL["neut_capacity_cfu"])   # 50 (rubio2023)                   # 50
    mono_capacity: int = 120
    neut_phago_p: float = 0.92          # 옵소닌화된 균에 대한 기본 포식확률
    mono_phago_p: float = 0.80
    opsonin_need: float = REAL["spn_opsonin_requirement"]                  # 0.40

    # ---------- 보체 ----------
    complement_on: float = 0.030
    complement_off: float = 0.011
    c3b_deposit_rate: float = 0.085     # tick 당 균 표면 C3b 침착 확률

    # ---------- 신호물질 축 (수용체별로 완전히 분리) ----------
    # [축1] fMLF -> FPR1 (말단표적) : ★세균 자신이 방출
    fmlf_per_bacterium: float = 0.00028
    fmlf_decay: float = Scale.per_tick_from_halflife_min(
        REAL["fmlf_halflife_min"])
    fmlf_sigma: float = 3.0
    # [축2] C5a -> C5aR1 (말단표적) : 균 표면 보체 활성화 산물
    c5a_per_bacterium: float = 0.00020
    c5a_decay: float = 0.55             # 초고속 분해 (guo2005), tick 하한 적용
    c5a_sigma: float = 1.6
    # [축3] CXCL8 -> CXCR2 (중간매개) : 상피/식세포 유래
    cxcl8_from_epi: float = 0.016
    cxcl8_from_neut: float = 0.005
    cxcl8_decay: float = Scale.per_tick_from_halflife_h(REAL["cxcl8_halflife_h"])
    cxcl8_sigma: float = 2.8
    # [축4] CCL2 -> CCR2 (중간매개) : 단핵구 동원
    ccl2_from_epi: float = 0.011
    ccl2_from_mono: float = 0.006
    ccl2_decay: float = Scale.per_tick_from_halflife_h(REAL["ccl2_halflife_h"])
    ccl2_sigma: float = 2.4
    # [축5] CXCL9/10 -> CXCR3 : NK/T. IFN-gamma 유도성
    cxcr3l_basal: float = 0.0022
    cxcr3l_ifng_gain: float = 0.010
    cxcr3l_decay: float = Scale.per_tick_from_halflife_h(REAL["cxcr3l_halflife_h"])
    cxcr3l_sigma: float = 2.0
    # [축6] CXCL13 -> CXCR5 : B세포 림프절 귀소
    cxcl13_from_ln: float = 0.004
    cxcl13_decay: float = Scale.per_tick_from_halflife_h(REAL["cxcl12_halflife_h"])
    cxcl13_sigma: float = 3.2

    # ---------- ★화학주성 계층 (heit2002, wang2016) ----------
    # 말단표적(fMLF/C5a) 신호가 중간매개(CXCL8/CCL2)보다 우선한다.
    # 말단표적 신호가 이 역치를 넘으면 중간매개 신호를 무시한다.
    endtarget_override_thr: float = 0.0016
    endtarget_weight: float = 1.0
    intermediary_weight: float = 0.35

    # ---------- 사이토카인 / 인터페론감마 ----------
    il1b_from_epi: float = 0.012        # 염증 구동
    il1b_decay: float = Scale.per_tick_from_halflife_h(3.0)
    il1b_sigma: float = 2.2
    ifng_from_nk_t: float = 0.0060      # NK/Th1 유래 IFN-gamma
    ifng_decay: float = Scale.per_tick_from_halflife_h(4.0)
    ifng_sigma: float = 2.2

    # ---------- 혈관외유출 역치 (수용체별로 다름) ----------
    extrav_FPR1: float = 0.00060        # 호중구: 가장 민감
    extrav_C5aR1: float = 0.00070
    extrav_CXCR2: float = 0.0100
    extrav_CCR2: float = 0.0160
    extrav_CXCR3: float = 0.0240
    extrav_CXCR5: float = 9.9e9         # B세포는 조직으로 나가지 않음
    reentry_thr: float = 1e-5

    # ---------- 면역세포 초기 개체수 (바이러스 모델과 동일) ----------
    n_neut0: int = 3000
    n_mono0: int = 300
    n_nk0: int = 195
    n_th0: int = 675
    n_tc0: int = 375
    n_b0: int = 180

    # ---------- 동원 ----------
    neut_recruit_gain: float = 42000.0
    mono_recruit_gain: float = 4200.0
    nk_recruit_gain: float = 1100.0
    neut_life_ticks: int = 240          # 약 1일
    mono_life_ticks: int = 1800
    nk_life_ticks: int = 2600

    # ---------- 이동 속도 ----------
    neut_speed: float = Scale.speed_cells_per_tick(REAL["neut_speed_um_min"])
    mono_speed: float = Scale.speed_cells_per_tick(REAL["mono_speed_um_min"])
    nk_speed: float = Scale.speed_cells_per_tick(REAL["nk_speed_um_min"])
    t_speed: float = Scale.speed_cells_per_tick(REAL["t_speed_um_min"])
    b_speed: float = Scale.speed_cells_per_tick(REAL["b_speed_um_min"])
    vessel_speed: float = 26.0

    # ---------- 후천면역 ----------
    ln_priming_ticks: int = 480         # 2일
    th17_gain: float = 1.0
    ab_igm_rate: float = 0.055
    ab_igg_rate: float = 0.030
    igm_decay: float = Scale.per_tick_from_halflife_h(REAL["igm_halflife_d"] * 24)
    igg_decay: float = Scale.per_tick_from_halflife_h(REAL["igg_halflife_d"] * 24)
    ab_opsonin_gain: float = 0.075      # 항협막 IgG 의 옵소닌 기여
    memory_fraction: float = 0.10
    teff_life_ticks: int = 1440         # 효과기 T세포 수명 약 6일


# =============================================================================
# SECTION 5.  이동 엔진 (무작위 vs 체계적)
# =============================================================================

STROMA, HEALTHY, DEAD, VESSEL, LYMPH = 0, 1, 2, 3, 4
LN_R0, LN_R1, LN_C0, LN_C1 = 460, 540, 60, 140


class MovementEngine:
    """
    두 이동방식의 유일한 차이점을 담는 클래스.
      random     : 방향을 균일난수로 뽑는다 (Brownian). 신호를 전혀 보지 않는다.
      systematic : 자기 수용체 축의 8방향 농도 중 최대 방향으로 간다.
                   ★화학주성 계층: 말단표적(fMLF/C5a) 신호가 역치를 넘으면
                     중간매개(CXCL8/CCL2) 신호를 무시한다.
    """

    def __init__(self, mode: str, rng: np.random.Generator):
        assert mode in ("random", "systematic")
        self.mode = mode
        self.rng = rng
        self.n_gradient_guided = 0
        self.n_move_calls = 0
        self.n_endtarget_led = 0

    @staticmethod
    def sample8(stack: np.ndarray, fr: np.ndarray, fc: np.ndarray) -> np.ndarray:
        return stack[:, fr, fc].T                      # (N, 8)

    def step(self, x, y, hx, hy, speed, end_stack, mid_stack, fr, fc, p):
        n = x.size
        if n == 0:
            return
        self.n_move_calls += n

        if self.mode == "random":
            # ---- 무작위 이동: 균일난수 방향 ----
            th = self.rng.uniform(0.0, 2.0 * math.pi, n).astype(np.float32)
            dx = np.cos(th)
            dy = np.sin(th)
        else:
            # ---- 체계적 이동: 화학주성 ----
            e = self.sample8(end_stack, fr, fc) if end_stack is not None else None
            m = self.sample8(mid_stack, fr, fc) if mid_stack is not None else None

            if e is None:
                vals = m
                lead_end = np.zeros(n, bool)
            elif m is None:
                vals = e
                lead_end = np.ones(n, bool)
            else:
                # ★계층 판정: 말단표적 신호가 역치를 넘는 세포는 그것만 따른다
                emax = e.max(axis=1)
                lead_end = emax > p.endtarget_override_thr
                vals = np.where(lead_end[:, None],
                                e * p.endtarget_weight,
                                m * p.intermediary_weight)
            self.n_endtarget_led += int(np.count_nonzero(lead_end))

            best = np.argmax(vals, axis=1)
            vmax = vals[np.arange(n), best]
            vmin = vals.min(axis=1)
            has_grad = (vmax - vmin) > 1e-12
            self.n_gradient_guided += int(np.count_nonzero(has_grad))

            d = DIR8[best]
            # 구배가 없으면 기존 극성(heading)을 유지한다 (난수 사용 안 함)
            dx = np.where(has_grad, d[:, 0], hx)
            dy = np.where(has_grad, d[:, 1], hy)
            nz = (np.abs(dx) + np.abs(dy)) < 1e-9
            if nz.any():
                dx[nz], dy[nz] = 1.0, 0.0

        hx[:] = dx
        hy[:] = dy
        x += dx * speed
        y += dy * speed
        self.bound(x, y)

    @staticmethod
    def bound(x, y):
        lim = float(Scale.GRID) - 1e-3
        np.abs(x, out=x)
        np.abs(y, out=y)
        for a in (x, y):
            over = a > lim
            if over.any():
                a[over] = 2.0 * lim - a[over]
            np.clip(a, 0.0, lim, out=a)


# =============================================================================
# SECTION 6.  환경 (조직 + 혈관 + 림프절 + 6개 신호축)
# =============================================================================


class Environment:
    def __init__(self, p: Params, rng):
        self.p = p
        self.rng = rng
        g = Scale.GRID
        st = np.full((g, g), STROMA, np.uint8)
        # 상피 밴드 (주기 25칸 중 12칸이 상피)
        for r0 in range(0, g, Scale.EPI_PERIOD):
            st[r0:r0 + 12, :] = HEALTHY
        # 혈관 (상피 밴드 사이)
        for r0 in range(0, g, Scale.EPI_PERIOD):
            rr = r0 + 13
            if rr < g:
                st[rr, :] = VESSEL
        for c0 in range(0, g, 120):
            st[:, c0] = VESSEL
        # 림프절
        st[LN_R0:LN_R1, LN_C0:LN_C1] = LYMPH
        self.state = st.reshape(-1)

        self.n_target0 = int(np.count_nonzero(self.state == HEALTHY))
        self.vessel_sites = np.flatnonzero(self.state == VESSEL).astype(np.int32)

        f = Scale.FIELD_N
        z = lambda: np.zeros((f, f), np.float32)
        # ---- 6개 신호축 (수용체별 완전 분리) ----
        self.fmlf = z()      # FPR1   (말단표적) : 세균 자신
        self.c5a = z()       # C5aR1  (말단표적) : 보체
        self.cxcl8 = z()     # CXCR2  (중간매개) : 상피/호중구
        self.ccl2 = z()      # CCR2   (중간매개) : 상피/단핵구
        self.cxcr3l = z()    # CXCR3            : IFN-gamma 유도
        self.cxcl13 = z()    # CXCR5            : 림프절
        self.il1b = z()      # 염증 구동
        self.ifng = z()      # IFN-gamma

        lr = slice(LN_R0 // Scale.FIELD_BIN, LN_R1 // Scale.FIELD_BIN + 1)
        lc = slice(LN_C0 // Scale.FIELD_BIN, LN_C1 // Scale.FIELD_BIN + 1)
        self.ln_slice = (lr, lc)

        self.complement = 0.0
        self.inflammation = 0.0
        self.damage = 0.0
        self.n_dead_epi = 0

    # -----------------------------------------------------------------
    @staticmethod
    def site_of(x, y):
        c = x.astype(np.int32)
        r = y.astype(np.int32)
        np.clip(c, 0, Scale.GRID - 1, out=c)
        np.clip(r, 0, Scale.GRID - 1, out=r)
        return r * Scale.GRID + c

    @staticmethod
    def field_of_xy(x, y):
        fc = (x / Scale.FIELD_BIN).astype(np.int32)
        fr = (y / Scale.FIELD_BIN).astype(np.int32)
        np.clip(fc, 0, Scale.FIELD_N - 1, out=fc)
        np.clip(fr, 0, Scale.FIELD_N - 1, out=fr)
        return fr, fc

    @staticmethod
    def _diffuse(a, sigma):
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma=sigma, mode="nearest")

    @staticmethod
    def stack_of(field):
        st = np.empty((8, Scale.FIELD_N, Scale.FIELD_N), np.float32)
        for i, (dc, dr) in enumerate(DIR8_OFF):
            st[i] = np.roll(np.roll(field, -dr, axis=0), -dc, axis=1)
        return st

    # -----------------------------------------------------------------
    def update_signals(self, bfr, bfc, neut_field, mono_field, ifng_field):
        p = self.p
        if bfr is not None and bfr.size:
            # ★[축1] fMLF: 세균이 스스로 방출한다 (표적 = 신호원)
            np.add.at(self.fmlf, (bfr, bfc), p.fmlf_per_bacterium)
            # ★[축2] C5a: 균 표면 보체활성화에 비례
            if self.complement > 1e-6:
                np.add.at(self.c5a, (bfr, bfc),
                          p.c5a_per_bacterium * float(self.complement))
            # [축3][축4] 상피/조직 반응성 케모카인
            np.add.at(self.cxcl8, (bfr, bfc), p.cxcl8_from_epi)
            np.add.at(self.ccl2, (bfr, bfc), p.ccl2_from_epi)
            np.add.at(self.il1b, (bfr, bfc), p.il1b_from_epi)
            np.add.at(self.cxcr3l, (bfr, bfc), p.cxcr3l_basal)
        if neut_field is not None and neut_field[0].size:
            np.add.at(self.cxcl8, neut_field, p.cxcl8_from_neut)
        if mono_field is not None and mono_field[0].size:
            np.add.at(self.ccl2, mono_field, p.ccl2_from_mono)
        if ifng_field is not None and ifng_field[0].size:
            np.add.at(self.ifng, ifng_field, p.ifng_from_nk_t)

        # [축5] CXCL9/10 은 IFN-gamma 유도성
        self.cxcr3l += p.cxcr3l_ifng_gain * self.ifng
        # [축6] 림프절의 항상적 CXCL13
        self.cxcl13[self.ln_slice] += p.cxcl13_from_ln

        self.fmlf = self._diffuse(self.fmlf, p.fmlf_sigma) * (1 - p.fmlf_decay)
        self.c5a = self._diffuse(self.c5a, p.c5a_sigma) * (1 - p.c5a_decay)
        self.cxcl8 = self._diffuse(self.cxcl8, p.cxcl8_sigma) * (1 - p.cxcl8_decay)
        self.ccl2 = self._diffuse(self.ccl2, p.ccl2_sigma) * (1 - p.ccl2_decay)
        self.cxcr3l = self._diffuse(self.cxcr3l, p.cxcr3l_sigma) * (1 - p.cxcr3l_decay)
        self.cxcl13 = self._diffuse(self.cxcl13, p.cxcl13_sigma) * (1 - p.cxcl13_decay)
        self.il1b = self._diffuse(self.il1b, p.il1b_sigma) * (1 - p.il1b_decay)
        self.ifng = self._diffuse(self.ifng, p.ifng_sigma) * (1 - p.ifng_decay)

    def update_humoral(self, n_bact):
        p = self.p
        drive = min(1.0, n_bact / 40000.0)
        self.complement += p.complement_on * drive * (1 - self.complement)
        self.complement -= p.complement_off * self.complement
        self.complement = float(np.clip(self.complement, 0.0, 1.0))
        self.inflammation += 0.030 * min(1.0, float(self.il1b.mean()) * 220.0) \
            * (1 - self.inflammation)
        self.inflammation -= 0.014 * self.inflammation
        self.inflammation = float(np.clip(self.inflammation, 0.0, 1.0))

    def kill_epithelium(self, sites):
        if sites.size == 0:
            return 0
        sites = np.unique(sites)
        ok = self.state[sites] == HEALTHY
        s = sites[ok]
        if s.size:
            self.state[s] = DEAD
            self.n_dead_epi += s.size
            self.damage += s.size / max(1, self.n_target0) * 1.6
        return int(s.size)

    def regenerate(self, rate):
        dead = np.flatnonzero(self.state == DEAD)
        if dead.size == 0:
            return
        k = int(dead.size * rate)
        if k > 0:
            pick = self.rng.choice(dead, size=min(k, dead.size), replace=False)
            self.state[pick] = HEALTHY


# =============================================================================
# SECTION 7.  세균 개체군 (비운동성 · 세포외 증식)
# =============================================================================


class BacteriaPopulation:
    """
    Streptococcus pneumoniae 개체군.
    ★바이러스와 결정적으로 다른 점:
      - 숙주세포 안으로 들어가지 않는다 (감염세포 상태가 존재하지 않음)
      - 비운동성이라 분열한 자리 근처에 머문다 -> 미세군락(microcolony)
      - 스스로 fMLF 를 방출하여 자기 위치를 광고한다
      - 협막이 보체 침착을 방해하므로, 옵소닌화되어야만 잡아먹힌다
    """

    def __init__(self, p: Params, rng):
        self.p = p
        self.rng = rng
        self.x = np.zeros(0, np.float32)
        self.y = np.zeros(0, np.float32)
        self.ops = np.zeros(0, np.float32)      # 표면 C3b/IgG 옵소닌 침착도 0~1

    @property
    def n(self):
        return self.x.size

    def seed(self, n, rng):
        """비강 접종: 상피 표면 한 지점 부근에 집중 접종"""
        cx, cy = 620.0, 430.0
        self.x = np.clip(rng.normal(cx, 26.0, n), 0, Scale.GRID - 1).astype(np.float32)
        self.y = np.clip(rng.normal(cy, 26.0, n), 0, Scale.GRID - 1).astype(np.float32)
        self.ops = np.zeros(n, np.float32)

    def divide(self, capacity_scale):
        """이분법 분열. 딸세포는 모세포 바로 옆(비운동성)."""
        if self.n == 0:
            return 0
        p = self.p
        eff = p.div_p * max(0.0, capacity_scale)
        if eff <= 0:
            return 0
        m = self.rng.random(self.n) < eff
        k = int(np.count_nonzero(m))
        if k == 0:
            return 0
        nx = self.x[m] + self.rng.normal(0, p.daughter_sigma, k).astype(np.float32)
        ny = self.y[m] + self.rng.normal(0, p.daughter_sigma, k).astype(np.float32)
        np.clip(nx, 0, Scale.GRID - 1, out=nx)
        np.clip(ny, 0, Scale.GRID - 1, out=ny)
        self.x = np.concatenate([self.x, nx])
        self.y = np.concatenate([self.y, ny])
        # 딸세포는 옵소닌을 물려받지 않는다 (새 표면)
        self.ops = np.concatenate([self.ops, np.zeros(k, np.float32)])
        return k

    def drift(self):
        """점액섬모 수송에 의한 미세 이동 (능동 운동 아님)"""
        if self.n == 0:
            return
        s = self.p.drift_sigma
        self.x += self.rng.normal(0, s, self.n).astype(np.float32)
        self.y += self.rng.normal(0, s, self.n).astype(np.float32)
        MovementEngine.bound(self.x, self.y)

    def opsonize(self, complement, ab_titer):
        """보체 C3b 침착 + 항협막 IgG 결합. 협막이 이를 저해한다."""
        if self.n == 0:
            return
        p = self.p
        gain = (p.c3b_deposit_rate * complement
                + p.ab_opsonin_gain * ab_titer) * (1.0 - p.capsule_resist * 0.55)
        self.ops += gain * (1.0 - self.ops) - p.ops_decay * self.ops
        np.clip(self.ops, 0.0, 1.0, out=self.ops)

    def remove(self, idx):
        if idx.size == 0:
            return 0
        keep = np.ones(self.n, bool)
        keep[idx] = False
        self.x = self.x[keep]
        self.y = self.y[keep]
        self.ops = self.ops[keep]
        return int(idx.size)


# =============================================================================
# SECTION 8.  면역세포 개체군
# =============================================================================

NEUT, MONO, NK, TH, TC, BC = 0, 1, 2, 3, 4, 5
CTYPE_NAME = {NEUT: "중성구", MONO: "단핵구/대식세포", NK: "NK세포",
              TH: "조력T세포", TC: "살해T세포", BC: "B세포"}
# ★수용체는 세포마다 반드시 다르다 (사용자 요구사항)
CTYPE_RECEPTOR = {
    NEUT: ("FPR1+C5aR1", "CXCR2"),     # (말단표적축, 중간매개축)
    MONO: ("FPR1", "CCR2"),
    NK:   (None, "CXCR3"),
    TH:   (None, "CXCR3"),
    TC:   (None, "CXCR3"),
    BC:   (None, "CXCR5"),
}


class CellPool:
    __slots__ = ("x", "y", "hx", "hy", "ctype", "alive", "in_vessel", "age",
                 "life", "capacity", "speed", "specific", "n", "cap")

    def __init__(self, cap=620000):
        self.cap = cap
        self.n = 0
        f32 = lambda: np.zeros(cap, np.float32)
        self.x, self.y = f32(), f32()
        self.hx, self.hy = f32(), f32()
        self.speed = f32()
        self.ctype = np.zeros(cap, np.uint8)
        self.alive = np.zeros(cap, bool)
        self.in_vessel = np.zeros(cap, bool)
        self.specific = np.zeros(cap, bool)
        self.age = np.zeros(cap, np.int32)
        self.life = np.zeros(cap, np.int32)
        self.capacity = np.zeros(cap, np.int32)

    def add(self, k, ctype, x, y, speed, life, capacity, in_vessel, rng,
            specific=False):
        if k <= 0:
            return
        if self.n + k > self.cap:
            k = self.cap - self.n
            if k <= 0:
                return
        s = slice(self.n, self.n + k)
        self.x[s] = x[:k]
        self.y[s] = y[:k]
        th = rng.uniform(0, 2 * math.pi, k)
        self.hx[s] = np.cos(th)
        self.hy[s] = np.sin(th)
        self.ctype[s] = ctype
        self.alive[s] = True
        self.in_vessel[s] = in_vessel
        self.specific[s] = specific
        self.age[s] = 0
        self.life[s] = life
        self.capacity[s] = capacity
        self.speed[s] = speed
        self.n += k

    def compact(self):
        idx = np.flatnonzero(self.alive[:self.n])
        m = idx.size
        for a in (self.x, self.y, self.hx, self.hy, self.speed):
            a[:m] = a[idx]
        for a in (self.ctype, self.alive, self.in_vessel, self.specific,
                  self.age, self.life, self.capacity):
            a[:m] = a[idx]
        self.alive[m:self.n] = False
        self.n = m


# =============================================================================
# SECTION 9.  림프절 (후천면역 프라이밍)
# =============================================================================


class LymphNode:
    def __init__(self, p: Params):
        self.p = p
        self.antigen = 0.0
        self.primed = False
        self.t_primed = None
        self.cd4 = 100.0
        self.cd8 = 50.0
        self.pc = 0.0
        self.th17 = 0.0
        self.igm = 0.0
        self.igg = 0.0
        self.clock = 0

    def step(self, t, antigen_in):
        p = self.p
        self.antigen = self.antigen * 0.985 + antigen_in
        if not self.primed and self.antigen > 60.0:
            self.clock += 1
            if self.clock >= p.ln_priming_ticks:
                self.primed = True
                self.t_primed = t
        if self.primed:
            d = (t - self.t_primed) / Scale.TICKS_PER_DAY
            growth = 1.0 if d < 6.0 else 0.0
            if growth:
                # 클론확장은 무한하지 않다: 전구세포 풀과 항원량에 의해 포화한다
                self.cd4 = min(self.cd4 * (1.0 + 0.0125), 60000.0)
                self.cd8 = min(self.cd8 * (1.0 + 0.0105), 30000.0)
                # ★세균 점막면역의 주역은 Th17 (lu2008, zhang2018)
                self.th17 = min(self.th17 + p.th17_gain
                                * (0.020 * self.cd4 - 0.010 * self.th17), 45000.0)
                self.pc = min(self.pc + 0.020 * self.cd4 - 0.006 * self.pc, 55000.0)
            else:
                self.cd4 *= 0.985
                self.cd8 *= 0.985
                self.th17 *= 0.990
                self.pc *= 0.988
            self.igm += p.ab_igm_rate * self.pc / 1000.0
            self.igm *= 1 - p.igm_decay
            if d > 3.5:
                self.igg += p.ab_igg_rate * self.pc / 1000.0
            self.igg *= 1 - p.igg_decay

    @property
    def titer(self):
        return min(self.igm + self.igg * 1.8, 120.0)


# =============================================================================
# SECTION 10.  메인 시뮬레이션
# =============================================================================


class BacterialInfectionSim:
    def __init__(self, p: Params, movement_mode: str, seed: int, verbose=False):
        self.p = p
        self.mode = movement_mode
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.env = Environment(p, self.rng)
        self.bact = BacteriaPopulation(p, self.rng)
        self.bact.seed(p.inoculum, self.rng)
        self.cells = CellPool()
        self.mover = MovementEngine(movement_mode, self.rng)
        self.ln = LymphNode(p)
        self.history: List[dict] = []

        # ---- 집계 ----
        self.cleared_phago_neut = 0
        self.cleared_phago_mono = 0
        self.cleared_complement = 0
        self.cleared_antibody = 0
        self.cleared_am = 0
        self.dead_immune = 0
        self.memory = 0
        self.n_extravasated = 0
        self.t_first_extrav = None
        self.t_first_arrival = None
        self.ctype_first_extrav = {}
        self.ctype_first_arrival = {}
        self.contact_events = 0
        self.neut_cell_ticks = 0

        self._seed_cells()

    # -----------------------------------------------------------------
    def _rand_vessel_xy(self, k):
        s = self.rng.choice(self.env.vessel_sites, size=k, replace=True)
        r = (s // Scale.GRID).astype(np.float32) + 0.5
        c = (s - (s // Scale.GRID) * Scale.GRID).astype(np.float32) + 0.5
        return c, r

    def _seed_cells(self):
        p, rng = self.p, self.rng
        for ct, k, spd, life, cap in (
                (NEUT, p.n_neut0, p.neut_speed, p.neut_life_ticks, p.neut_capacity),
                (MONO, p.n_mono0, p.mono_speed, p.mono_life_ticks, p.mono_capacity),
                (NK, p.n_nk0, p.nk_speed, p.nk_life_ticks, 0),
                (TH, p.n_th0, p.t_speed, 100000, 0),
                (TC, p.n_tc0, p.t_speed, 100000, 0),
                (BC, p.n_b0, p.b_speed, 100000, 0)):
            x, y = self._rand_vessel_xy(k)
            self.cells.add(k, ct, x, y, spd, life, cap, True, rng)

    # -----------------------------------------------------------------
    def _axis_stacks(self, env):
        """수용체별 축 스택. 말단표적축과 중간매개축을 분리해 만든다."""
        end_neut = self.env.fmlf + self.env.c5a        # FPR1 + C5aR1
        end_mono = self.env.fmlf                        # FPR1
        return {
            "end": {NEUT: Environment.stack_of(end_neut),
                    MONO: Environment.stack_of(end_mono)},
            "mid": {NEUT: Environment.stack_of(env.cxcl8),
                    MONO: Environment.stack_of(env.ccl2),
                    NK: Environment.stack_of(env.cxcr3l),
                    TH: Environment.stack_of(env.cxcr3l),
                    TC: Environment.stack_of(env.cxcr3l),
                    BC: Environment.stack_of(env.cxcl13)},
            "end_field": {NEUT: end_neut, MONO: end_mono},
            "mid_field": {NEUT: env.cxcl8, MONO: env.ccl2, NK: env.cxcr3l,
                          TH: env.cxcr3l, TC: env.cxcr3l, BC: env.cxcl13},
        }

    EXTRAV = {NEUT: "extrav_FPR1", MONO: "extrav_CCR2", NK: "extrav_CXCR3",
              TH: "extrav_CXCR3", TC: "extrav_CXCR3", BC: "extrav_CXCR5"}

    # -----------------------------------------------------------------
    def step(self, t):
        p, env, c, rng = self.p, self.env, self.cells, self.rng

        # ---------- A. 세균 증식 ----------
        occ = self.bact.n / p.carrying_capacity
        # ★영양 면역: 염증 강도에 비례해 실효 증식률이 감소한다
        nutri = 1.0 - p.nutritional_immunity * float(env.inflammation)
        self.bact.divide(max(0.0, (1.0 - occ) * max(0.0, nutri)))
        self.bact.drift()

        # 상주 폐포대식세포에 의한 기저 제거
        if self.bact.n:
            am_eff = p.am_clearance_p / (1.0 + self.bact.n / p.am_saturation_n)
            m = rng.random(self.bact.n) < am_eff
            k = int(np.count_nonzero(m))
            if k:
                self.cleared_am += self.bact.remove(np.flatnonzero(m))

        # ---------- B. 옵소닌화 ----------
        self.bact.opsonize(env.complement, self.ln.titer)

        # ---------- C. 신호장 갱신 ----------
        if self.bact.n:
            bfr, bfc = Environment.field_of_xy(self.bact.x, self.bact.y)
        else:
            bfr = bfc = None
        nn = c.n
        alive_idx = np.flatnonzero(c.alive[:nn])
        neut_f = mono_f = ifng_f = None
        if alive_idx.size:
            ct_all = c.ctype[alive_idx]
            tis = ~c.in_vessel[alive_idx]
            sel = alive_idx[(ct_all == NEUT) & tis]
            if sel.size:
                neut_f = Environment.field_of_xy(c.x[sel], c.y[sel])
            sel = alive_idx[(ct_all == MONO) & tis]
            if sel.size:
                mono_f = Environment.field_of_xy(c.x[sel], c.y[sel])
            sel = alive_idx[((ct_all == NK) | (ct_all == TH)) & tis]
            if sel.size:
                ifng_f = Environment.field_of_xy(c.x[sel], c.y[sel])
        env.update_signals(bfr, bfc, neut_f, mono_f, ifng_f)
        env.update_humoral(self.bact.n)

        # ---------- D. 동원 ----------
        drive = float(np.clip(0.55 * env.inflammation
                              + 0.45 * min(1.0, self.bact.n / 25000.0), 0, 1))
        for ct, gain, spd, life, cap in (
                (NEUT, p.neut_recruit_gain, p.neut_speed, p.neut_life_ticks,
                 p.neut_capacity),
                (MONO, p.mono_recruit_gain, p.mono_speed, p.mono_life_ticks,
                 p.mono_capacity),
                (NK, p.nk_recruit_gain, p.nk_speed, p.nk_life_ticks, 0)):
            g = gain
            if ct == NEUT and self.ln.th17 > 0:
                g = gain * (1.0 + min(1.4, self.ln.th17 / 120.0))
            k = int(g * drive / Scale.TICKS_PER_DAY)
            if k > 0:
                x, y = self._rand_vessel_xy(k)
                c.add(k, ct, x, y, spd, life, cap, True, rng)

        # 후천면역 조직 진입
        if self.ln.primed:
            # ★세포외 세균에서는 Th17(CD4)이 주역이다. CTL은 죽일 감염세포가
            #   없으므로 조직 진입이 미미하다 (lu2008, zhang2018).
            k_th = int(self.ln.th17 * 0.00120)
            k_tc = int(self.ln.cd8 * 0.00012)
            if k_th > 0:
                x, y = self._rand_vessel_xy(k_th)
                c.add(k_th, TH, x, y, p.t_speed, p.teff_life_ticks, 0, True,
                      rng, specific=True)
            if k_tc > 0:
                x, y = self._rand_vessel_xy(k_tc)
                c.add(k_tc, TC, x, y, p.t_speed, p.teff_life_ticks, 0, True,
                      rng, specific=True)

        # ---------- E. 수명 ----------
        nn = c.n
        if nn:
            al = c.alive[:nn]
            c.age[:nn][al] += 1
            died = al & (c.age[:nn] >= c.life[:nn])
            k = int(np.count_nonzero(died))
            if k:
                c.alive[:nn][died] = False
                self.dead_immune += k
            if nn > 20000 and (c.alive[:nn].mean() < 0.93
                               or nn > int(0.55 * c.cap)):
                c.compact()

        # ---------- F. ★이동 ----------
        stacks = self._axis_stacks(env)
        nn = c.n
        idx = np.flatnonzero(c.alive[:nn])
        if idx.size:
            fr, fc = Environment.field_of_xy(c.x[idx], c.y[idx])
            ct_all = c.ctype[idx]
            self.neut_cell_ticks += int(np.count_nonzero(ct_all == NEUT))

            # 각 세포가 '자기 수용체 축'에서 느끼는 신호 세기
            sense = np.zeros(idx.size, np.float32)
            for ctv in (NEUT, MONO, NK, TH, TC, BC):
                m = ct_all == ctv
                if not m.any():
                    continue
                v = stacks["mid_field"][ctv][fr[m], fc[m]]
                if ctv in stacks["end_field"]:
                    v = v + stacks["end_field"][ctv][fr[m], fc[m]] * 40.0
                sense[m] = v

            in_v = c.in_vessel[idx]
            thr = np.zeros(idx.size, np.float32)
            for ctv in (NEUT, MONO, NK, TH, TC, BC):
                m = ct_all == ctv
                if m.any():
                    thr[m] = getattr(p, self.EXTRAV[ctv])
            ex = in_v & (sense > thr)
            if ex.any():
                gi = idx[ex]
                c.in_vessel[gi] = False
                self.n_extravasated += gi.size
                if self.t_first_extrav is None:
                    self.t_first_extrav = t
                for ctv in np.unique(c.ctype[gi]):
                    ctv = int(ctv)
                    if ctv not in self.ctype_first_extrav:
                        self.ctype_first_extrav[ctv] = t
                in_v = c.in_vessel[idx]

            for ctv in (NEUT, MONO, NK, TH, TC, BC):
                m = ct_all == ctv
                if not m.any():
                    continue
                gi = idx[m]
                iv = in_v[m]
                es = stacks["end"].get(ctv)
                ms = stacks["mid"][ctv]
                for invessel in (True, False):
                    sub = gi[iv == invessel] if invessel else gi[~iv]
                    if sub.size == 0:
                        continue
                    xs, ys = c.x[sub].copy(), c.y[sub].copy()
                    hxs, hys = c.hx[sub].copy(), c.hy[sub].copy()
                    sfr, sfc = Environment.field_of_xy(xs, ys)
                    spd = (p.vessel_speed if invessel else c.speed[sub])
                    self.mover.step(xs, ys, hxs, hys, spd, es, ms, sfr, sfc, p)
                    c.x[sub], c.y[sub] = xs, ys
                    c.hx[sub], c.hy[sub] = hxs, hys

        # ---------- G. 식균작용 ----------
        # 식세포 1개는 한 번의 조우에서 여러 균을 연속으로 삼킬 수 있다.
        # (호중구 포식 용량 약 50 CFU/세포, rubio2023)
        # 이를 R회 라운드로 구현한다.
        PHAGO_ROUNDS = 5
        nn = c.n
        idx = np.flatnonzero(c.alive[:nn])
        if idx.size and self.bact.n:
            csite = Environment.site_of(c.x[idx], c.y[idx])
            _sel_cache = {}
            for ctv, _b in ((NEUT, p.neut_phago_p), (MONO, p.mono_phago_p)):
                _sel_cache[ctv] = np.flatnonzero((c.ctype[idx] == ctv)
                                                 & (~c.in_vessel[idx]))
            for _round in range(PHAGO_ROUNDS):
                if self.bact.n == 0:
                    break
                bsite = Environment.site_of(self.bact.x, self.bact.y)
                eaten_all = []
                for ctv, base in ((NEUT, p.neut_phago_p), (MONO, p.mono_phago_p)):
                    sel0 = _sel_cache[ctv]
                    if sel0.size == 0:
                        continue
                    sel = sel0[c.capacity[idx[sel0]] > 0]
                    if sel.size == 0:
                        continue
                    loc, bi = self._match_site(csite[sel], bsite)
                    if loc.size == 0:
                        continue
                    if _round == 0:
                        self.contact_events += loc.size
                    # ★협막 저항: 옵소닌화 정도가 요구치를 넘어야 포식 가능
                    #   (gordon1980: 폐렴사슬알균은 40% 이상 혈청 필요)
                    ops = self.bact.ops[bi]
                    eff = base * np.clip(
                        (ops - p.opsonin_need * 0.35)
                        / max(1e-6, 1 - p.opsonin_need * 0.35), 0, 1)
                    ok = rng.random(loc.size) < eff
                    if not ok.any():
                        continue
                    gi = idx[sel[loc[ok]]]
                    tb = bi[ok]
                    uq, first = np.unique(tb, return_index=True)
                    gi = gi[first]
                    np.subtract.at(c.capacity, gi, 1)
                    if ctv == NEUT:
                        self.cleared_phago_neut += uq.size
                    else:
                        self.cleared_phago_mono += uq.size
                    eaten_all.append(uq)
                if eaten_all:
                    self.bact.remove(np.unique(np.concatenate(eaten_all)))
                else:
                    break

        # ---------- H. 항체/보체 직접 사멸 ----------
        if self.bact.n:
            # 그람양성균은 두꺼운 펩티도글리칸과 협막 때문에 보체 MAC에 저항한다.
            # 따라서 항체/보체의 기여는 '직접 살균'이 아니라 '옵소닌화'가 본질이며,
            # 여기서는 미미한 직접 효과만 남긴다. (hyams2010, gordon1980)
            # ★항협막 IgG: 폐렴구균 방어의 핵심 기전이자 폐렴구균 백신의 원리.
            #   옵소닌화에 더해 응집(agglutination)을 일으켜 점액섬모 수송으로
            #   제거되게 하므로, 식세포와의 조우 여부와 무관하게 작용한다.
            ti = self.ln.titer
            if ti > 0.02:
                m = rng.random(self.bact.n) < min(0.050, 0.00042 * ti)
                if m.any():
                    self.cleared_antibody += self.bact.remove(np.flatnonzero(m))
            if env.complement > 0.35 and self.bact.n:
                m = rng.random(self.bact.n) < 0.00016 * env.complement
                if m.any():
                    self.cleared_complement += self.bact.remove(np.flatnonzero(m))

        # ---------- I. 조직 손상 ----------
        if self.bact.n:
            m = rng.random(self.bact.n) < p.epi_damage_p
            if m.any():
                env.kill_epithelium(Environment.site_of(self.bact.x[m],
                                                        self.bact.y[m]))
        # 호중구 매개 부수적 손상
        if idx.size:
            tis = idx[~c.in_vessel[idx]]
            if tis.size:
                nsel = tis[c.ctype[tis] == NEUT]
                if nsel.size:
                    m = rng.random(nsel.size) < 4.0e-6 * (1 + 3 * env.inflammation)
                    if m.any():
                        env.kill_epithelium(
                            Environment.site_of(c.x[nsel[m]], c.y[nsel[m]]))
        env.regenerate(0.0022 * (1 - env.inflammation))

        # ---------- J. 도착 기록 ----------
        if idx.size and self.bact.n:
            tis = idx[~c.in_vessel[idx]]
            if tis.size:
                cs = Environment.site_of(c.x[tis], c.y[tis])
                bs = Environment.site_of(self.bact.x, self.bact.y)
                loc, _ = self._match_site(cs, bs)
                if loc.size:
                    if self.t_first_arrival is None:
                        self.t_first_arrival = t
                    for ctv in np.unique(c.ctype[tis[loc]]):
                        ctv = int(ctv)
                        if ctv not in self.ctype_first_arrival:
                            self.ctype_first_arrival[ctv] = t

        # ---------- K. 림프절 ----------
        self.ln.step(t, 0.0045 * self.bact.n)
        if self.ln.primed and self.memory == 0:
            d = (t - self.ln.t_primed) / Scale.TICKS_PER_DAY
            if d > 8.0:
                self.memory = int((self.ln.cd8 + self.ln.cd4) * p.memory_fraction)

    # -----------------------------------------------------------------
    @staticmethod
    def _match_site(a_sites, b_sites):
        """같은 격자칸에 있는 (a인덱스, b인덱스) 쌍을 찾는다."""
        if a_sites.size == 0 or b_sites.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        order = np.argsort(b_sites, kind="stable")
        bs = b_sites[order]
        lo = np.searchsorted(bs, a_sites, "left")
        hi = np.searchsorted(bs, a_sites, "right")
        has = hi > lo
        ai = np.flatnonzero(has)
        if ai.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        bi = order[lo[ai]]
        return ai, bi

    # -----------------------------------------------------------------
    def snapshot(self, t):
        c, env = self.cells, self.env
        n = c.n
        al = c.alive[:n]
        ct = c.ctype[:n]
        cnt = lambda k: int(np.count_nonzero(al & (ct == k)))
        return {
            "tick": t, "day": t / Scale.TICKS_PER_DAY,
            "bacteria": self.bact.n,
            "opsonized_frac": float(self.bact.ops.mean()) if self.bact.n else 0.0,
            "healthy": int(np.count_nonzero(env.state == HEALTHY)),
            "dead_epi": env.n_dead_epi,
            "neutrophil": cnt(NEUT), "monocyte": cnt(MONO), "nk": cnt(NK),
            "helper_t": cnt(TH), "killer_t": cnt(TC), "bcell": cnt(BC),
            "in_tissue": int(np.count_nonzero(al & (~c.in_vessel[:n]))),
            "memory": self.memory,
            "igm": self.ln.igm, "igg": self.ln.igg, "antibody": self.ln.titer,
            "th17": self.ln.th17, "ln_cd8": self.ln.cd8, "ln_pc": self.ln.pc,
            "fmlf": float(env.fmlf.mean()), "fmlf_max": float(env.fmlf.max()),
            "c5a": float(env.c5a.mean()),
            "cxcl8": float(env.cxcl8.mean()), "ccl2": float(env.ccl2.mean()),
            "cxcr3l": float(env.cxcr3l.mean()), "ifng": float(env.ifng.mean()),
            "il1b": float(env.il1b.mean()),
            "complement": env.complement, "inflammation": env.inflammation,
            "damage": env.damage,
            "cleared_phago": self.cleared_phago_neut + self.cleared_phago_mono,
            "cleared_phago_neut": self.cleared_phago_neut,
            "cleared_phago_mono": self.cleared_phago_mono,
            "cleared_am": self.cleared_am,
            "cleared_complement": self.cleared_complement,
            "cleared_antibody": self.cleared_antibody,
            "dead_immune": self.dead_immune,
            "n_extravasated": self.n_extravasated,
            "contact_events": self.contact_events,
        }

    def run(self):
        total = self.p.max_days * Scale.TICKS_PER_DAY
        self.history.append(self.snapshot(0))
        for t in range(1, total + 1):
            self.step(t)
            if t % Scale.TICKS_PER_DAY == 0:
                s = self.snapshot(t)
                self.history.append(s)
                if self.verbose:
                    print(f"  Day {s['day']:5.1f}  bact={s['bacteria']:>8,} "
                          f"neut={s['neutrophil']:>7,} H={s['healthy']:>7,} "
                          f"fmlf={s['fmlf']:.5f} comp={s['complement']:.3f}",
                          flush=True)
            if self.bact.n == 0 and t > Scale.TICKS_PER_DAY:
                self.history.append(self.snapshot(t))
                break
        return self.history
