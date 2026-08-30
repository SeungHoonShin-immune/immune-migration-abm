#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 immune_systematic_abm.py
 면역세포 "체계적 이동(Systematic / Chemotaxis-only)" 감염-면역 반응 ABM
--------------------------------------------------------------------------------
 실험 조건 : 면역세포의 이동 방향은 100% 화학신호(케모카인 농도구배) +
             혈관구조 + 혈류로만 결정된다.
             무작위 방향 선택, 확률적 이동, 랜덤 탐색을 일체 사용하지 않는다.
             동시에 면역세포는 병원체/감염세포의 좌표를 직접 참조하지 못한다.
             (오직 자신이 있는 지점 주변 8방향의 신호 농도만 감지)

 대표 병원체 : Influenza A virus (H1N1)
 모델 도메인 : 상기도 상피 슬랩 + 혈관 네트워크 + 배액 림프절

 축소 원칙  : 실제값 -> 논문 출처 -> 축소비율 -> 시뮬레이션값
              개체 축소비 kappa = 1/1000 (상피세포/면역세포/바이러스 공통)
              논문에서 값을 찾지 못한 항목은 "모델 가정값"으로 명시

 ※ 본 파일은 무작위이동 모델과 코드/상태를 전혀 공유하지 않는 독립 실행 파일이다.
   생물학 파라미터는 동일 문헌에서 독립적으로 재산출하였다.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# SECTION 1.  LITERATURE REFERENCE DATABASE
# =============================================================================


@dataclass(frozen=True)
class Ref:
    tag: str
    authors: str
    title: str
    journal: str
    year: int
    doi: str = ""
    pmid: str = ""

    def short(self) -> str:
        return f"{self.authors.split(',')[0]} et al. {self.year}"

    def full(self) -> str:
        ident = []
        if self.doi:
            ident.append(f"DOI:{self.doi}")
        if self.pmid:
            ident.append(f"PMID:{self.pmid}")
        return (f"{self.authors}. \"{self.title}\" {self.journal} "
                f"({self.year}). {'  '.join(ident)}")


REFS: Dict[str, Ref] = {
    # ---- 바이러스 동역학 ----
    "baccam2006": Ref(
        "baccam2006",
        "Baccam P, Beauchemin C, Macken CA, Hayden FG, Perelson AS",
        "Kinetics of influenza A virus infection in humans",
        "Journal of Virology 80(15):7590-7599", 2006,
        "10.1128/JVI.01623-05", "16840338"),
    "carrat2008": Ref(
        "carrat2008",
        "Carrat F, Vergu E, Ferguson NM, Lemaitre M, Cauchemez S, Leach S, Valleron AJ",
        "Time lines of infection and disease in human influenza: a review of "
        "volunteer challenge studies",
        "American Journal of Epidemiology 167(7):775-785", 2008,
        "10.1093/aje/kwm375", "18230677"),
    "mohler2005": Ref(
        "mohler2005", "Mohler L, Flockerzi D, Sann H, Reichl U",
        "Mathematical model of influenza A virus production in large-scale "
        "microcarrier culture",
        "Biotechnology and Bioengineering 90(1):46-58", 2005,
        "10.1002/bit.20363", "15736163"),
    "harris2006": Ref(
        "harris2006",
        "Harris A, Cardone G, Winkler DC, Heymann JB, Brecher M, White JM, Steven AC",
        "Influenza virus pleiomorphy characterized by cryoelectron tomography",
        "PNAS 103(50):19123-19127", 2006, "10.1073/pnas.0607614103", "17146053"),
    "memoli2015": Ref(
        "memoli2015",
        "Memoli MJ, Czajkowski L, Reed S, Athota R, Bristol T, Proudfoot K, et al.",
        "Validation of the wild-type influenza A human challenge model H1N1pdMIST",
        "Clinical Infectious Diseases 60(5):693-702", 2015,
        "10.1093/cid/ciu924", "25416753"),
    "olmsted2001": Ref(
        "olmsted2001",
        "Olmsted SS, Padgett JL, Yudin AI, Whaley KJ, Moench TR, Cone RA",
        "Diffusion of macromolecules and virus-like particles in human cervical mucus",
        "Biophysical Journal 81(4):1930-1937", 2001,
        "10.1016/S0006-3495(01)75844-4", "11566767"),

    # ---- 혈액 세포 조성 ----
    "dacie": Ref("dacie", "Bain BJ, Bates I, Laffan MA",
                 "Dacie and Lewis Practical Haematology, 12th ed.",
                 "Elsevier (textbook, reference intervals)", 2017),
    "bisset2004": Ref(
        "bisset2004", "Bisset LR, Lung TL, Kaelin M, Ludwig E, Dubs RW",
        "Reference values for peripheral blood lymphocyte phenotypes applicable "
        "to the healthy adult population in Switzerland",
        "European Journal of Haematology 72(3):203-212", 2004,
        "10.1046/j.0902-4441.2003.00199.x", "14962239"),

    # ---- 세포 운동성 / 주화성 ----
    "lammermann2013": Ref(
        "lammermann2013",
        "Lammermann T, Afonso PV, Angermann BR, Wang JM, Kastenmuller W, "
        "Parent CA, Germain RN",
        "Neutrophil swarms require LTB4 and integrins at sites of cell death in vivo",
        "Nature 498(7454):371-375", 2013, "10.1038/nature12175", "23708969"),
    "miller2002": Ref(
        "miller2002", "Miller MJ, Wei SH, Parker I, Cahalan MD",
        "Two-photon imaging of lymphocyte motility and antigen response in "
        "intact lymph node", "Science 296(5574):1869-1873", 2002,
        "10.1126/science.1070051", "12016203"),
    "ariotti2012": Ref(
        "ariotti2012",
        "Ariotti S, Beltman JB, Chodaczek G, Hoekstra ME, van Beek AE, et al.",
        "Tissue-resident memory CD8+ T cells continuously patrol skin epithelia "
        "to quickly recognize local antigen", "PNAS 109(48):19739-19744", 2012,
        "10.1073/pnas.1208927109", "23150545"),
    "bhat2007": Ref(
        "bhat2007", "Bhat R, Watzl C",
        "Serial killing of tumor cells by human natural killer cells - "
        "enhancement by therapeutic antibodies", "PLoS ONE 2(3):e326", 2007,
        "10.1371/journal.pone.0000326", "17389917"),
    "sackmann2012": Ref(
        "sackmann2012",
        "Sackmann EK, Berthier E, Schwantes EA, Fichtinger PS, Evans MD, et al.",
        "Characterizing bacteria-neutrophil chemotaxis and migration velocity "
        "using a microfluidic device",
        "PNAS 109(15):5813-5818", 2012, "10.1073/pnas.1119578109", "22451913"),
    "boneschansker2014": Ref(
        "boneschansker2014",
        "Boneschansker L, Yan J, Wong E, Briscoe DM, Irimia D",
        "Microfluidic platform for the quantitative analysis of leukocyte "
        "migration signatures", "Nature Communications 5:4787", 2014,
        "10.1038/ncomms5787", "25183261"),
    "beltman2007": Ref(
        "beltman2007", "Beltman JB, Maree AFM, Lynch JN, Miller MJ, de Boer RJ",
        "Lymph node topology dictates T cell migration behavior",
        "Journal of Experimental Medicine 204(4):771-780", 2007,
        "10.1084/jem.20061278", "17389236"),

    # ---- 케모카인 / 사이토카인 ----
    "hayden1998": Ref(
        "hayden1998",
        "Hayden FG, Fritz R, Lobo MC, Alvord W, Strober W, Straus SE",
        "Local and systemic cytokine responses during experimental human "
        "influenza A virus infection",
        "Journal of Clinical Investigation 101(3):643-649", 1998,
        "10.1172/JCI1355", "9449698"),
    "proudfoot2003": Ref(
        "proudfoot2003",
        "Proudfoot AEI, Handel TM, Johnson Z, Lau EK, LiWang P, et al.",
        "Glycosaminoglycan binding and oligomerization are essential for the "
        "in vivo activity of certain chemokines",
        "PNAS 100(4):1885-1890", 2003, "10.1073/pnas.0334864100", "12571364"),
    "weber2013": Ref(
        "weber2013",
        "Weber M, Hauschild R, Schwarz J, Moussion C, de Vries I, et al.",
        "Interstitial dendritic cell guidance by haptotactic chemokine gradients",
        "Science 339(6117):328-332", 2013, "10.1126/science.1228456", "23329049"),
    "ivashkiv2014": Ref(
        "ivashkiv2014", "Ivashkiv LB, Donlin LT",
        "Regulation of type I interferon responses",
        "Nature Reviews Immunology 14(1):36-49", 2014, "10.1038/nri3581",
        "24362405"),

    # ---- 혈관 / 혈관외 유출 ----
    "ley2007": Ref(
        "ley2007", "Ley K, Laudanna C, Cybulsky MI, Nourshargh S",
        "Getting to the site of inflammation: the leukocyte adhesion cascade "
        "updated", "Nature Reviews Immunology 7(9):678-689", 2007,
        "10.1038/nri2156", "17717539"),
    "nourshargh2010": Ref(
        "nourshargh2010", "Nourshargh S, Hordijk PL, Sixt M",
        "Breaching multiple barriers: leukocyte motility through venular walls "
        "and the interstitium", "Nature Reviews Molecular Cell Biology 11(5):366-378",
        2010, "10.1038/nrm2889", "20414258"),

    # ---- 세포 수명 ----
    "pillay2010": Ref(
        "pillay2010",
        "Pillay J, den Braber I, Vrisekoop N, Kwast LM, de Boer RJ, Borghans JAM, "
        "Tesselaar K, Koenderman L",
        "In vivo labeling with 2H2O reveals a human neutrophil lifespan of 5.4 days",
        "Blood 116(4):625-627", 2010, "10.1182/blood-2010-01-259028", "20410504"),
    "patel2017": Ref(
        "patel2017",
        "Patel AA, Zhang Y, Fullerton JN, Boelen L, Rongvaux A, Maini AA, et al.",
        "The fate and lifespan of human monocyte subsets in steady state and "
        "systemic inflammation",
        "Journal of Experimental Medicine 214(7):1913-1923", 2017,
        "10.1084/jem.20170355", "28606987"),
    "zhang2007": Ref(
        "zhang2007",
        "Zhang Y, Wallace DL, de Lara CM, Ghattas H, Asquith B, Worth A, et al.",
        "In vivo kinetics of human natural killer cells: the effects of ageing "
        "and acute and chronic viral infection",
        "Immunology 121(2):258-265", 2007,
        "10.1111/j.1365-2567.2007.02573.x", "17346281"),

    # ---- 후천면역 ----
    "mempel2004": Ref(
        "mempel2004", "Mempel TR, Henrickson SE, von Andrian UH",
        "T-cell priming by dendritic cells in lymph nodes occurs in three "
        "distinct phases", "Nature 427(6970):154-159", 2004,
        "10.1038/nature02238", "14712275"),
    "lawrence2005": Ref(
        "lawrence2005", "Lawrence CW, Ream RM, Braciale TJ",
        "Frequency, specificity, and sites of expansion of CD8+ T cells during "
        "primary pulmonary influenza virus infection",
        "Journal of Immunology 174(9):5332-5340", 2005,
        "10.4049/jimmunol.174.9.5332", "15843530"),
    "wrammert2008": Ref(
        "wrammert2008",
        "Wrammert J, Smith K, Miller J, Langley WA, Kokko K, Larsen C, et al.",
        "Rapid cloning of high-affinity human monoclonal antibodies against "
        "influenza virus", "Nature 453(7195):667-671", 2008,
        "10.1038/nature06890", "18449194"),
    "kaech2002": Ref(
        "kaech2002", "Kaech SM, Wherry EJ, Ahmed R",
        "Effector and memory T-cell differentiation: implications for vaccine "
        "development", "Nature Reviews Immunology 2(4):251-262", 2002,
        "10.1038/nri778", "11970378"),
    "vanstipdonk2001": Ref(
        "vanstipdonk2001", "van Stipdonk MJB, Lemmens EE, Schoenberger SP",
        "Naive CTLs require a single brief period of antigenic stimulation for "
        "clonal expansion and differentiation",
        "Nature Immunology 2(5):423-429", 2001, "10.1038/87730", "11323696"),
    "alanio2010": Ref(
        "alanio2010", "Alanio C, Lemaitre F, Law HKW, Hasan M, Albert ML",
        "Enumeration of human antigen-specific naive CD8+ T cells reveals "
        "conserved precursor frequencies", "Blood 115(18):3718-3725", 2010,
        "10.1182/blood-2009-10-251124", "20220118"),
    "blattman2002": Ref(
        "blattman2002",
        "Blattman JN, Antia R, Sourdive DJD, Wang X, Kaech SM, et al.",
        "Estimating the precursor frequency of naive antigen-specific CD8 T cells",
        "Journal of Experimental Medicine 195(5):657-664", 2002,
        "10.1084/jem.20001021", "11877489"),
    "clements1986": Ref(
        "clements1986", "Clements ML, Betts RF, Tierney EL, Murphy BR",
        "Serum and nasal wash antibodies associated with resistance to "
        "experimental challenge with influenza A wild-type virus",
        "Journal of Clinical Microbiology 24(1):157-160", 1986,
        "10.1128/jcm.24.1.157-160.1986", "3722363"),
    "regoes2007": Ref(
        "regoes2007", "Regoes RR, Barber DL, Ahmed R, Antia R",
        "Estimation of the rate of killing by cytotoxic T lymphocytes in vivo",
        "PNAS 104(5):1599-1603", 2007, "10.1073/pnas.0508830104", "17242364"),
    "ganusov2008": Ref(
        "ganusov2008", "Ganusov VV, De Boer RJ",
        "Estimating in vivo death rates of targets due to CD8 T-cell-mediated "
        "killing", "Journal of Virology 82(23):11749-11757", 2008,
        "10.1128/JVI.01128-08", "18815293"),
    "janeway": Ref("janeway", "Murphy K, Weaver C",
                   "Janeway's Immunobiology, 9th ed.",
                   "Garland Science (textbook)", 2016),
    "histology": Ref("histology", "Mescher AL",
                     "Junqueira's Basic Histology (respiratory epithelium, "
                     "microvascular dimensions)", "McGraw-Hill (textbook)", 2018),
}

MODEL_ASSUMPTION = "모델 가정값"


# =============================================================================
# SECTION 2.  SCALE
# =============================================================================


class Scale:
    """
    [공간] 1,000 x 1,000 = 1,000,000 칸.  1칸 = 15 um (상피세포 직경 10~20um 중앙값)
           -> 도메인 15 mm x 15 mm 상기도 상피 슬랩
    [개체] kappa = 1/1000 (상피세포/면역세포/바이러스 공통)
    [시간] 1 tick = 6분 = 0.1시간, 1일 = 240 tick
    """
    GRID = 1000
    SITES = GRID * GRID
    SITE_UM = 15.0
    DOMAIN_MM = GRID * SITE_UM / 1000.0

    KAPPA = 1.0 / 1000.0
    INV_KAPPA = 1000.0

    DT_MIN = 6.0
    DT_HOUR = 0.1
    TICKS_PER_HOUR = 10
    TICKS_PER_DAY = 240

    # 조직 배치: 20행 주기 중 8행 상피(40%), 12행 간질(혈관/림프관 포함)
    EPI_PERIOD = 20
    EPI_ROWS = 8

    # 화학신호 농도장 해상도: 5x5칸 = 75 um 단위 -> 200 x 200
    FIELD_BIN = 5
    FIELD_N = GRID // FIELD_BIN
    FIELD_UM = SITE_UM * FIELD_BIN

    @staticmethod
    def per_tick_from_per_day(rate_per_day: float) -> float:
        return 1.0 - math.exp(-rate_per_day / Scale.TICKS_PER_DAY)

    @staticmethod
    def per_tick_from_halflife_h(halflife_hours: float) -> float:
        k = math.log(2.0) / halflife_hours
        return 1.0 - math.exp(-k * Scale.DT_HOUR)

    @staticmethod
    def speed_to_sites_per_tick(speed_um_per_min: float) -> float:
        """
        논문의 이동'속도'(um/min) -> tick 당 이동 거리(칸).
        체계적 이동에서는 방향이 신호로 결정되므로 속도는 그대로 경로길이가 된다.
        (무작위보행처럼 운동성계수로 환산하지 않는다 - 이것이 두 모델의 정의 차이)
        """
        return speed_um_per_min * Scale.DT_MIN / Scale.SITE_UM


# =============================================================================
# SECTION 3.  REAL VALUES (논문값) & PARAMETER TABLE
# =============================================================================

REAL = {
    # 혈액 조성
    "wbc_per_ml": 5.0e6, "neut_frac": 0.60, "mono_frac": 0.06, "lymph_frac": 0.30,
    "cd4_of_lymph": 0.45, "cd8_of_lymph": 0.25, "bcell_of_lymph": 0.12,
    "nk_of_lymph": 0.13,

    # 바이러스
    "virion_nm": 100.0, "eclipse_h": 6.0, "productive_h": 5.0,
    "infected_lifetime_h": 11.0, "virus_halflife_h": 3.0, "R0_within_host": 22.0,
    "burst_particles": 5000.0, "target_cells_urt": 4.0e8,
    "inoculum_tcid50": 1.0e5, "virion_D_um2_s": 1.0,

    # 이동속도 (um/min)
    "neut_speed": 15.0, "mono_speed": 2.5, "nk_speed": 8.0,
    "naiveT_speed": 11.0, "effT_speed": 5.0, "b_speed": 6.0,
    "neut_chemotaxis_speed": 7.0,   # IL-8 구배 하 3D 이동속도 2~7 um/min
    "rolling_velocity_um_s": 10.0,  # 세정맥 내 rolling 속도 5~50 um/s

    # 주화성 지수 (실제)
    "chemotactic_index_real": 0.35,  # 0.25~0.5

    # 수명 (일)
    "neut_lifespan_d": 1.5, "mono_lifespan_d": 3.0, "nk_lifespan_d": 14.0,
    "naive_lymph_lifespan_d": 180.0, "effector_T_halflife_d": 1.5,
    "eff_t_halflife_expand_d": 4.0,

    # 후천면역
    "dc_migration_h": 18.0, "priming_h": 24.0,
    "cd8_doubling_h": 7.0, "cd4_doubling_h": 9.0, "b_doubling_h": 10.0,
    "cd8_max_divisions": 11.0, "cd4_max_divisions": 9.0, "b_max_divisions": 10.0,
    "expansion_program_d": 6.0, "net_expansion_factor": 0.53,
    "contraction_rate_d": 0.75, "memory_fraction": 0.075,
    "plasmablast_halflife_d": 3.5, "gc_pc_halflife_d": 7.0,
    "igm_halflife_d": 5.0, "igg_halflife_d": 21.0,
    "cd8_precursor_systemic": 5.0e4, "cd4_precursor_systemic": 1.0e5,
    "b_precursor_systemic": 5.0e4,

    # 화학신호
    "chemokine_D_um2_s": 45.0,     # GAG 결합으로 자유확산보다 크게 저해됨
    "chemokine_halflife_h": 3.5,
    "cytokine_D_um2_s": 120.0,
    "cytokine_halflife_h": 1.5,
    "ifn_peak_day": 2.0, "antiviral_state_h": 5.0,
    "cytokine_peak_day": 2.0,
    "c3_mg_per_ml": 1.2,
    "symptom_peak_day": 3.0, "shedding_days": 4.8,

    # 혈관
    "capillary_spacing_um": 300.0,   # 조직 내 모세혈관 간격 20~300 um
    "diapedesis_min": 5.0,           # 혈관벽 통과 2~10분
}


def _sp(v: float) -> float:
    return Scale.speed_to_sites_per_tick(v)


@dataclass(frozen=True)
class ParamRow:
    item: str
    real: str
    unit: str
    source: str
    scaling: str
    sim: str
    note: str = ""


PARAM_TABLE: List[ParamRow] = [
    ParamRow("기준 부피", "1", "mL 상당", "본 모델 정의", "기준",
             f"{Scale.SITES:,}칸 (1000x1000)", "사양서 ①②"),
    ParamRow("격자 1칸", "10~20", "um (상피세포 직경)", "histology", "공간축소",
             f"{Scale.SITE_UM:.0f} um -> 도메인 {Scale.DOMAIN_MM:.0f}x{Scale.DOMAIN_MM:.0f} mm", ""),
    ParamRow("표적 상피세포", f"{REAL['target_cells_urt']:.1e}", "cells", "baccam2006",
             "kappa=1/1000", "약 397,000칸 (격자의 40%, 림프절 영역 제외)",
             "Baccam 2006 T0 = 4x10^8"),
    ParamRow("총 백혈구", f"{REAL['wbc_per_ml']:.1e}", "cells/mL", "dacie",
             "kappa=1/1000", "5,000 agent", "4.0~11.0 x10^9/L 중앙 부근"),
    ParamRow("중성구 비율", "40~70 (60 채택)", "%", "dacie", "동일", "3,000 agent", ""),
    ParamRow("단핵구 비율", "2~10 (6 채택)", "%", "dacie", "동일", "300 agent", ""),
    ParamRow("NK세포 비율", "림프구의 5~20 (13 채택)", "%", "bisset2004", "동일",
             "195 agent", ""),
    ParamRow("조력T세포 비율", "림프구의 45", "%", "bisset2004", "동일", "675 agent", ""),
    ParamRow("살해T세포 비율", "림프구의 25", "%", "bisset2004", "동일", "375 agent", ""),
    ParamRow("B세포 비율", "림프구의 12", "%", "bisset2004", "동일", "180 agent", ""),
    ParamRow("병원체 초기량", f"{REAL['inoculum_tcid50']:.1e}", "TCID50", "memoli2015",
             "kappa=1/1000", "100 agent", "인체감염모델 접종량 10^4~10^6"),
    ParamRow("바이러스 크기", "80~120", "nm", "harris2006", "공간축소",
             "점입자 (1칸 15um 대비 무시)", "세포>>바이러스 관계 유지"),
    ParamRow("바이러스 확산", f"{REAL['virion_D_um2_s']}", "um^2/s", "olmsted2001",
             "시간/공간축소", "확산+점액수송 (목표탐색 없음)", "사양서 ⑪"),
    ParamRow("바이러스 잠복기", f"{REAL['eclipse_h']}", "시간", "baccam2006", "시간축소",
             f"{int(REAL['eclipse_h']*Scale.TICKS_PER_HOUR)} tick", ""),
    ParamRow("감염세포 생산기간", f"{REAL['productive_h']}", "시간", "baccam2006",
             "시간축소", f"{int(REAL['productive_h']*Scale.TICKS_PER_HOUR)} tick",
             "감염세포 평균수명 11h = 6h + 5h"),
    ParamRow("바이러스 반감기", f"{REAL['virus_halflife_h']}", "시간", "baccam2006",
             "시간축소",
             f"{Scale.per_tick_from_halflife_h(REAL['virus_halflife_h']):.4f}/tick", ""),
    ParamRow("감염세포당 생산량", "10^3~10^4", "particles/cell", "mohler2005",
             "kappa+감염성비율", "25 agent/칸", "입자:감염단위 ~100:1"),
    ParamRow("체내 R0", f"{REAL['R0_within_host']}", "-", "baccam2006", "동일",
             "창발 결과로 검증", ""),
    # ---- 이동 ----
    ParamRow("중성구 이동속도(주화성)", "2~7 (7 채택)", "um/min", "sackmann2012",
             "시간/공간축소", f"{_sp(REAL['neut_chemotaxis_speed']):.1f} 칸/tick",
             "IL-8 구배 하 3D 콜라겐 내 속도. 무구배 대조군은 0~1 um/min"),
    ParamRow("중성구 최대속도", f"{REAL['neut_speed']}", "um/min", "lammermann2013",
             "시간/공간축소", f"{_sp(REAL['neut_speed']):.1f} 칸/tick", "생체 내 10~20"),
    ParamRow("단핵구/대식세포 속도", f"{REAL['mono_speed']}", "um/min",
             "lammermann2013", "시간/공간축소", f"{_sp(REAL['mono_speed']):.1f} 칸/tick", ""),
    ParamRow("NK세포 속도", f"{REAL['nk_speed']}", "um/min", "bhat2007",
             "시간/공간축소", f"{_sp(REAL['nk_speed']):.1f} 칸/tick", ""),
    ParamRow("나이브 T세포 속도", f"{REAL['naiveT_speed']}", "um/min", "miller2002",
             "시간/공간축소", f"{_sp(REAL['naiveT_speed']):.1f} 칸/tick", "림프절 2광자"),
    ParamRow("효과기 CD8 조직내 속도", f"{REAL['effT_speed']}", "um/min", "ariotti2012",
             "시간/공간축소", f"{_sp(REAL['effT_speed']):.1f} 칸/tick", ""),
    ParamRow("B세포 속도", f"{REAL['b_speed']}", "um/min", "miller2002",
             "시간/공간축소", f"{_sp(REAL['b_speed']):.1f} 칸/tick", ""),
    ParamRow("혈관 내 rolling 속도", "5~50 (10 채택)", "um/s", "ley2007",
             "시간/공간축소", f"{10.0*60*Scale.DT_MIN/Scale.SITE_UM:.0f} 칸/tick",
             "혈관외유출 가능한 marginated 세포 기준"),
    ParamRow("혈관벽 통과(diapedesis)", "2~10", "분", "nourshargh2010", "시간축소",
             "1 tick 내 완료", ""),
    ParamRow("모세혈관 간격", "20~300", "um", "histology", "공간축소",
             "혈관 격자 간격 300 um (20칸)", ""),
    # ---- 화학신호 ----
    ParamRow("케모카인 확산계수", "GAG 결합으로 크게 저해", "um^2/s", "proudfoot2003",
             "시간/공간축소",
             f"{REAL['chemokine_D_um2_s']:.0f} um^2/s -> 구배 감쇠거리 약 0.9 mm",
             "자유확산(~150)이 아닌 haptotactic 고정형 구배 (weber2013)"),
    ParamRow("케모카인 반감기", f"{REAL['chemokine_halflife_h']}", "시간",
             MODEL_ASSUMPTION, "시간축소",
             f"{Scale.per_tick_from_halflife_h(REAL['chemokine_halflife_h']):.4f}/tick",
             "조직 내 케모카인 turnover 정량치 부재"),
    ParamRow("사이토카인 반감기", f"{REAL['cytokine_halflife_h']}", "시간",
             "hayden1998", "시간축소",
             f"{Scale.per_tick_from_halflife_h(REAL['cytokine_halflife_h']):.4f}/tick",
             "IL-6/TNF 비강세척액 동태"),
    ParamRow("사이토카인 정점", f"{REAL['cytokine_peak_day']}", "일", "hayden1998",
             "시간축소", "창발 결과로 검증", ""),
    ParamRow("인터페론 정점", f"{REAL['ifn_peak_day']}", "일", "hayden1998",
             "시간축소", "창발 결과로 검증", ""),
    ParamRow("항바이러스 상태 확립", f"{REAL['antiviral_state_h']}", "시간",
             "ivashkiv2014", "시간축소",
             f"{int(REAL['antiviral_state_h']*Scale.TICKS_PER_HOUR)} tick", ""),
    ParamRow("보체 C3 혈장농도", f"{REAL['c3_mg_per_ml']}", "mg/mL", "janeway",
             "정규화", "활성도 0~1", ""),
    # ---- 수명 ----
    ParamRow("중성구 수명", f"{REAL['neut_lifespan_d']}", "일", "pillay2010",
             "시간축소", f"{int(REAL['neut_lifespan_d']*Scale.TICKS_PER_DAY)} tick",
             "혈중 t1/2 7~19h vs 5.4일 보고 상충, 조직 1~2일 채택"),
    ParamRow("단핵구 수명", f"{REAL['mono_lifespan_d']}", "일", "patel2017",
             "시간축소", f"{int(REAL['mono_lifespan_d']*Scale.TICKS_PER_DAY)} tick", ""),
    ParamRow("NK세포 수명", f"{REAL['nk_lifespan_d']}", "일", "zhang2007",
             "시간축소", f"{int(REAL['nk_lifespan_d']*Scale.TICKS_PER_DAY)} tick", ""),
    # ---- 후천면역 ----
    ParamRow("DC 항원수송 지연", f"{REAL['dc_migration_h']}", "시간", "lawrence2005",
             "시간축소", f"{int(REAL['dc_migration_h']*Scale.TICKS_PER_HOUR)} tick",
             "12~24h 범위 중앙"),
    ParamRow("T세포 프라이밍", f"{REAL['priming_h']}", "시간", "mempel2004",
             "시간축소", f"{int(REAL['priming_h']*Scale.TICKS_PER_HOUR)} tick",
             "3단계 priming 후 첫 분열까지"),
    ParamRow("CD8 분열주기", f"{REAL['cd8_doubling_h']}", "시간", "lawrence2005",
             "시간축소", f"{int(REAL['cd8_doubling_h']*Scale.TICKS_PER_HOUR)} tick", ""),
    ParamRow("CD4 분열주기", f"{REAL['cd4_doubling_h']}", "시간", "mempel2004",
             "시간축소", f"{int(REAL['cd4_doubling_h']*Scale.TICKS_PER_HOUR)} tick", ""),
    ParamRow("B세포 분열주기", f"{REAL['b_doubling_h']}", "시간", "wrammert2008",
             "시간축소", f"{int(REAL['b_doubling_h']*Scale.TICKS_PER_HOUR)} tick", ""),
    ParamRow("자율증식 프로그램", f"{REAL['expansion_program_d']}", "일",
             "vanstipdonk2001", "시간축소",
             f"{int(REAL['expansion_program_d']*Scale.TICKS_PER_DAY)} tick",
             "항원은 개시신호, 이후 자율 증식 후 수축"),
    ParamRow("CD8 나이브 전구세포", f"{REAL['cd8_precursor_systemic']:.1e}",
             "cells(전신)", "alanio2010", "kappa=1/1000", "50 agent", ""),
    ParamRow("CD4 나이브 전구세포", f"{REAL['cd4_precursor_systemic']:.1e}",
             "cells(전신)", "blattman2002", "kappa=1/1000", "100 agent", ""),
    ParamRow("B 나이브 전구세포", f"{REAL['b_precursor_systemic']:.1e}",
             "cells(전신)", "blattman2002", "kappa=1/1000", "50 agent", ""),
    ParamRow("IgM 반감기", f"{REAL['igm_halflife_d']}", "일", "janeway", "시간축소",
             f"{Scale.per_tick_from_halflife_h(REAL['igm_halflife_d']*24):.5f}/tick", ""),
    ParamRow("IgG 반감기", f"{REAL['igg_halflife_d']}", "일", "janeway", "시간축소",
             f"{Scale.per_tick_from_halflife_h(REAL['igg_halflife_d']*24):.5f}/tick", ""),
    ParamRow("기억세포 형성비율", "5~10", "% (정점 대비)", "kaech2002", "동일",
             f"{REAL['memory_fraction']*100:.1f}%", ""),
    # ---- 모델 가정값 ----
    ParamRow("주화성 지수(CI)", "0.25~0.5 (실제 호중구)", "-",
             "sackmann2012/boneschansker2014", "실험조건으로 대체", "1.0 (완전 지향)",
             "★ 실제 면역세포는 '편향된 무작위보행'이며 CI<1 이다. "
             "본 실험은 '체계적으로만' 이동하는 조건이므로 CI=1 의 이상적 상한을 사용한다"),
    ParamRow("혈관 면적비", "-", "-", MODEL_ASSUMPTION, "-", "격자의 약 7%",
             "간질층 내 수평 세정맥 + 수직 연결혈관 네트워크"),
    ParamRow("혈관외유출 역치", "-", "-", MODEL_ASSUMPTION, "-", "케모카인 0.02 (정규화)",
             "국소 케모카인이 역치 초과 시 결정론적으로 유출"),
    ParamRow("신호 감지 반경", "-", "-", MODEL_ASSUMPTION, "-", "75 um (필드 1칸)",
             "세포는 자기 주변 8방향만 감지. 전역 좌표 접근 불가 (사양서 ㉒)"),
    ParamRow("식세포 접촉 반경", "-", "-", MODEL_ASSUMPTION, "-", "동일 격자칸(15um)",
             "중성구 직경 ~12um"),
    ParamRow("중성구 식균 용량", "-", "-", MODEL_ASSUMPTION, "-", "10회", "기능소진"),
    ParamRow("NK 접촉당 사멸확률", "-", "-", MODEL_ASSUMPTION, "-", "0.25/tick",
             "NK-표적 접합 20~60분(bhat2007) 을 6분 tick 으로 환산"),
    ParamRow("CTL 접촉당 사멸확률", "-", "-", MODEL_ASSUMPTION, "-", "0.80/tick",
             "결과를 regoes2007/ganusov2008 범위와 대조 검증"),
    ParamRow("상피 재생속도", "-", "-", MODEL_ASSUMPTION, "-",
             "0.02/일 x (1-염증)", "급성 염증기에는 기저세포 증식 억제"),
]


# =============================================================================
# SECTION 4.  PARAMS
# =============================================================================


@dataclass
class Params:
    # 초기 개체수 (kappa = 1/1000)
    n_virus0: int = 100
    n_neutrophil: int = 3000
    n_monocyte: int = 300
    n_nk: int = 195
    n_helper_t: int = 675
    n_killer_t: int = 375
    n_bcell: int = 180

    precursor_cd8: float = 50.0
    precursor_cd4: float = 100.0
    precursor_b: float = 50.0

    # 바이러스
    eclipse_ticks: int = int(REAL["eclipse_h"] * Scale.TICKS_PER_HOUR)
    productive_ticks: int = int(REAL["productive_h"] * Scale.TICKS_PER_HOUR)
    virus_decay_p: float = Scale.per_tick_from_halflife_h(REAL["virus_halflife_h"])
    burst_agents_per_site: float = 25.0
    p_infect: float = 0.15
    virus_sigma: float = 30.0          # 확산 + 점액섬모 수송 (등방, 목표탐색 없음)
    max_virions: int = 800_000

    # ---- 화학신호 필드 ----
    # 케모카인: 이동 방향 결정 (사양서 ⑭⑯)
    chemo_from_infected: float = 0.020      # 생산 감염칸 1개가 tick 당 방출
    chemo_from_phagocyte: float = 0.004     # 활성 식세포의 2차 방출 (사양서 19)
    chemo_decay: float = Scale.per_tick_from_halflife_h(REAL["chemokine_halflife_h"])
    chemo_sigma_field: float = 2.4          # tick 당 확산 sigma (필드칸)
    # 사이토카인: 면역반응 조절 (사양서 ㉓) - 이동방향 결정에 사용하지 않음
    cyto_from_infected: float = 0.020
    cyto_decay: float = Scale.per_tick_from_halflife_h(REAL["cytokine_halflife_h"])
    cyto_sigma_field: float = 3.8
    cyto_activation_threshold: float = 0.004   # 식세포 활성화 역치
    # 인터페론: 항바이러스 상태
    ifn_production: float = 0.02
    ifn_decay_p: float = Scale.per_tick_from_halflife_h(2.0)
    ifn_sigma_field: float = 1.6
    antiviral_rate: float = 1.0 / (REAL["antiviral_state_h"] * Scale.TICKS_PER_HOUR)
    antiviral_decay: float = 1.0 / (24.0 * Scale.TICKS_PER_HOUR)
    ifn_block_release: float = 0.80

    # ---- 이동 (체계적) ----
    speed_neut: float = Scale.speed_to_sites_per_tick(REAL["neut_chemotaxis_speed"])
    speed_mono: float = Scale.speed_to_sites_per_tick(REAL["mono_speed"])
    speed_nk: float = Scale.speed_to_sites_per_tick(REAL["nk_speed"])
    speed_naive_t: float = Scale.speed_to_sites_per_tick(REAL["naiveT_speed"])
    speed_eff_t: float = Scale.speed_to_sites_per_tick(REAL["effT_speed"])
    speed_b: float = Scale.speed_to_sites_per_tick(REAL["b_speed"])
    vessel_speed: float = REAL["rolling_velocity_um_s"] * 60.0 * Scale.DT_MIN \
        / Scale.SITE_UM                       # 240 칸/tick
    extravasation_threshold: float = 0.02     # 국소 케모카인 역치 (정규화)
    reentry_threshold: float = 0.002          # 이하로 떨어지면 혈관 복귀(역주행 해소)

    # ---- 보체 / 항체 ----
    complement_on: float = 0.06
    complement_off: float = 0.02
    complement_lysis: float = 0.010
    complement_opsonin: float = 0.35
    ab_per_pc_igm: float = 5.0e-6
    ab_per_pc_igg: float = 2.5e-5
    igm_decay_p: float = Scale.per_tick_from_halflife_h(REAL["igm_halflife_d"] * 24)
    igg_decay_p: float = Scale.per_tick_from_halflife_h(REAL["igg_halflife_d"] * 24)
    ab_kd: float = 2.0
    ab_neutralize_max: float = 0.030
    ab_opsonin: float = 0.35
    ab_block_max: float = 0.95
    ab_block_kd: float = 3.0
    ab_detect_threshold: float = 0.5

    # ---- 세포 기능 ----
    neut_phago_p: float = 0.45
    neut_capacity: int = 10
    mono_phago_p: float = 0.55
    mono_capacity: int = 25
    nk_kill_p: float = 0.25
    ctl_kill_p: float = 0.80

    # ---- 수명 (tick) ----
    life_neut: int = int(REAL["neut_lifespan_d"] * Scale.TICKS_PER_DAY)
    life_mono: int = int(REAL["mono_lifespan_d"] * Scale.TICKS_PER_DAY)
    life_nk: int = int(REAL["nk_lifespan_d"] * Scale.TICKS_PER_DAY)
    life_naive: int = int(REAL["naive_lymph_lifespan_d"] * Scale.TICKS_PER_DAY)
    eff_t_death_expand: float = Scale.per_tick_from_halflife_h(
        REAL["eff_t_halflife_expand_d"] * 24)
    eff_t_death_contract: float = Scale.per_tick_from_halflife_h(
        REAL["effector_T_halflife_d"] * 24)

    # ---- 염증 / 동원 / 재생 ----
    inflam_gain_virus: float = 1.0 / 8000.0
    inflam_gain_infected: float = 1.0 / 40000.0
    inflam_decay: float = 0.010
    recruit_neut_max: float = 3.0
    recruit_mono_max: float = 9.0
    recruit_nk_max: float = 5.0
    recruit_rate: float = 0.020
    damage_rate: float = 0.0015
    epi_regen_p: float = Scale.per_tick_from_per_day(0.02)

    # ---- 림프절 / 후천면역 ----
    dc_delay_ticks: int = int(REAL["dc_migration_h"] * Scale.TICKS_PER_HOUR)
    priming_ticks: int = int(REAL["priming_h"] * Scale.TICKS_PER_HOUR)
    cd8_growth: float = math.log(2.0) / (REAL["cd8_doubling_h"] * Scale.TICKS_PER_HOUR)
    cd4_growth: float = math.log(2.0) / (REAL["cd4_doubling_h"] * Scale.TICKS_PER_HOUR)
    b_growth: float = math.log(2.0) / (REAL["b_doubling_h"] * Scale.TICKS_PER_HOUR)
    cd8_max_expansion: float = 2.0 ** REAL["cd8_max_divisions"]
    cd4_max_expansion: float = 2.0 ** REAL["cd4_max_divisions"]
    b_max_expansion: float = 2.0 ** REAL["b_max_divisions"]
    net_expansion: float = REAL["net_expansion_factor"]
    program_ticks: int = int(REAL["expansion_program_d"] * Scale.TICKS_PER_DAY)
    contraction_p: float = Scale.per_tick_from_per_day(REAL["contraction_rate_d"])
    b_contraction_p: float = Scale.per_tick_from_per_day(0.30)
    ef_delay_ticks: int = int(48.0 * Scale.TICKS_PER_HOUR)
    gc_delay_ticks: int = int(144.0 * Scale.TICKS_PER_HOUR)
    gc_end_ticks: int = int(21.0 * Scale.TICKS_PER_DAY)
    ef_diff_rate: float = 0.0016
    gc_diff_rate: float = 3.0e-5
    pc_short_decay: float = Scale.per_tick_from_per_day(
        math.log(2.0) / REAL["plasmablast_halflife_d"])
    pc_gc_decay: float = Scale.per_tick_from_per_day(
        math.log(2.0) / REAL["gc_pc_halflife_d"])
    memory_fraction: float = REAL["memory_fraction"]
    memory_b_fraction: float = 0.10
    emigration_p: float = 1.6e-4
    emigration_start_h: float = 96.0
    antigen_threshold: float = 20.0
    antigen_decay: float = Scale.per_tick_from_per_day(1.5)

    max_days: int = 30
    stop_after_clear_days: int = 4
    seed: int = 20260809


# =============================================================================
# SECTION 5.  SYSTEMATIC MOVEMENT ENGINE   (★ 본 실험의 핵심)
# =============================================================================

# 8방향 단위벡터 (사양서 ⑯: 주변 8방향 농도 측정 후 최대 방향으로 이동)
_R2 = 1.0 / math.sqrt(2.0)
DIR8 = np.array([
    (0.0,  1.0), ( _R2,  _R2), ( 1.0, 0.0), ( _R2, -_R2),
    (0.0, -1.0), (-_R2, -_R2), (-1.0, 0.0), (-_R2,  _R2)], dtype=np.float32)
DIR8_OFF = np.array([
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1)], dtype=np.int32)   # (dcol, drow) 필드단위


class SystematicMovementEngine:
    """
    100% 체계적 이동.  난수 발생기를 일절 사용하지 않는다.

    이동 방향 결정 규칙 (우선순위):
      1) 조직 내 세포 : 자기 위치 주변 8방향의 '케모카인' 농도를 측정하여
                        가장 높은 방향으로 이동한다. (사양서 ⑯)
                        - 전역 좌표/병원체 위치는 참조하지 않는다. (사양서 ㉒㊴)
      2) 구배가 없을 때 : 직전 이동방향(세포 극성)을 유지한다.
                        - 무작위 방향 재선택을 하지 않는다. (사양서 ㊳)
      3) 혈관 내 세포 : 혈류 방향으로 이동하되, 혈관을 따라 케모카인이
                        증가하는 방향이 있으면 그쪽으로 이동한다. (사양서 ㉛)
      4) 경로의 굴곡  : 8방향 양자화 + 시간에 따라 변하는 농도장 + 혈관 구조
                        때문에 자연스럽게 굽는다. 별도의 난수를 넣지 않는다. (사양서 ⑰)
    """

    MODE = "systematic"

    def __init__(self):
        self.n_calls = 0
        self.n_agents_moved = 0
        self.n_gradient_guided = 0     # 구배로 방향이 결정된 횟수
        self.n_persistence = 0         # 구배 없어 극성 유지한 횟수
        self.n_flow = 0                # 혈류로 이동한 횟수

    # ---------------------------------------------------------------
    @staticmethod
    def sample_8(field_stack: np.ndarray, fr: np.ndarray, fc: np.ndarray
                 ) -> np.ndarray:
        """
        미리 shift 해 둔 8장의 농도장에서 각 세포 위치의 8방향 값을 읽는다.
        field_stack shape = (8, FIELD_N, FIELD_N)
        반환 shape = (N, 8)
        """
        return field_stack[:, fr, fc].T

    # ---------------------------------------------------------------
    def move_tissue(self, x, y, heading, speed, chemo_stack, fr, fc):
        """조직 내 세포: 8방향 케모카인 argmax 로 방향 결정"""
        n = x.size
        if n == 0:
            return
        self.n_calls += 1
        self.n_agents_moved += n

        vals = self.sample_8(chemo_stack, fr, fc)          # (N,8)
        best = np.argmax(vals, axis=1)
        vmax = vals[np.arange(n), best]
        vmin = vals.min(axis=1)
        has_grad = (vmax - vmin) > 1e-9                     # 감지 가능한 구배 존재

        d = DIR8[best]                                      # (N,2)
        new_head = np.where(has_grad[:, None], d, heading)
        # 극성이 아직 정해지지 않은 세포(heading=0)는 신호가 생길 때까지 정지
        zero = (np.abs(new_head).sum(axis=1) < 1e-9)
        heading[:] = new_head

        self.n_gradient_guided += int(has_grad.sum())
        self.n_persistence += int((~has_grad & ~zero).sum())

        step = np.where(zero, 0.0, speed)
        x += (new_head[:, 0] * step).astype(x.dtype)
        y += (new_head[:, 1] * step).astype(y.dtype)
        self._bound(x, y)

    # ---------------------------------------------------------------
    def move_vessel(self, x, y, heading, flow_dx, flow_dy, speed,
                    chemo_stack, fr, fc):
        """
        혈관 내 세포: 기본은 혈류 방향.
        단, 혈관을 따라 케모카인이 뚜렷이 증가하는 방향이 있으면 그쪽 우선.
        """
        n = x.size
        if n == 0:
            return
        self.n_calls += 1
        self.n_agents_moved += n
        self.n_flow += n

        vals = self.sample_8(chemo_stack, fr, fc)
        best = np.argmax(vals, axis=1)
        vmax = vals[np.arange(n), best]
        here = vals[:, 0] * 0.0
        # 현재 위치 농도는 8방향 중앙값 대용으로 min 을 사용 (구배 유무 판정용)
        vmin = vals.min(axis=1)
        strong = (vmax - vmin) > 1e-6

        d = DIR8[best]
        dx = np.where(strong, d[:, 0], flow_dx)
        dy = np.where(strong, d[:, 1], flow_dy)
        heading[:, 0] = dx
        heading[:, 1] = dy
        x += (dx * speed).astype(x.dtype)
        y += (dy * speed).astype(y.dtype)
        self._bound(x, y)

    # ---------------------------------------------------------------
    @staticmethod
    def _bound(x, y):
        lim = float(Scale.GRID) - 1e-3
        np.abs(x, out=x)
        np.abs(y, out=y)
        over = x > lim
        if over.any():
            x[over] = 2.0 * lim - x[over]
        over = y > lim
        if over.any():
            y[over] = 2.0 * lim - y[over]
        np.clip(x, 0.0, lim, out=x)
        np.clip(y, 0.0, lim, out=y)


# =============================================================================
# SECTION 6.  ENVIRONMENT  (조직 + 혈관 네트워크 + 림프절 + 화학신호장)
# =============================================================================

STROMA, HEALTHY, ECLIPSE, PRODUCTIVE, DEAD, VESSEL, LYMPH = 0, 1, 2, 3, 4, 5, 6

# 림프절 영역 (사양서 ⑩㉕)
LN_R0, LN_R1 = 460, 540
LN_C0, LN_C1 = 40, 120


class Environment:

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        n = Scale.GRID

        # ---- 1. 조직 기본 배치 (상피 밴드 40%) ----
        rows = np.arange(n)
        epi_row = (rows % Scale.EPI_PERIOD) < Scale.EPI_ROWS
        base = np.where(epi_row, HEALTHY, STROMA).astype(np.uint8)
        state = np.repeat(base, n).reshape(n, n)

        # ---- 2. 혈관 네트워크 (사양서 ⑩⑳) ----
        #  수평 세정맥: 간질층 내 20행 주기로 1행 (간격 300 um)
        #  수직 연결혈관: 50열 주기로 1열 (간격 750 um)
        #  혈류 방향: 수평혈관은 행 인덱스에 따라 좌우 교대, 수직은 상행
        vessel_rows = np.flatnonzero((rows % Scale.EPI_PERIOD) == 13)
        state[vessel_rows, :] = VESSEL
        vessel_cols = np.arange(25, n, 50)
        for c in vessel_cols:
            col_mask = ~epi_row                  # 간질층에서만 수직혈관 통과
            state[col_mask, c] = VESSEL

        # 혈류 방향 필드 (혈관 칸에만 의미)
        self.flow_dx = np.zeros((n, n), dtype=np.float32)
        self.flow_dy = np.zeros((n, n), dtype=np.float32)
        for i, r in enumerate(vessel_rows):
            self.flow_dx[r, :] = 1.0 if (i % 2 == 0) else -1.0
        for c in vessel_cols:
            m = ~epi_row
            self.flow_dx[m, c] = 0.0
            self.flow_dy[m, c] = 1.0

        # ---- 3. 림프절 영역 ----
        state[LN_R0:LN_R1, LN_C0:LN_C1] = LYMPH
        self.flow_dx[LN_R0:LN_R1, LN_C0:LN_C1] = 0.0
        self.flow_dy[LN_R0:LN_R1, LN_C0:LN_C1] = 0.0

        self.state = state.reshape(-1)
        self.flow_dx = self.flow_dx.reshape(-1)
        self.flow_dy = self.flow_dy.reshape(-1)

        self.is_vessel = (self.state == VESSEL)
        self.vessel_sites = np.flatnonzero(self.is_vessel)
        self.n_target = int(np.count_nonzero(self.state == HEALTHY))
        self.n_vessel = int(self.is_vessel.sum())

        # 림프절 출구(혈관) 지점: 림프절에 인접한 혈관 칸
        ln_rows = np.arange(LN_R0 - 25, LN_R1 + 25)
        vr = self.vessel_sites // Scale.GRID
        vc = self.vessel_sites - vr * Scale.GRID
        near = ((vr >= ln_rows[0]) & (vr <= ln_rows[-1]) &
                (vc >= LN_C0 - 30) & (vc <= LN_C1 + 60))
        self.ln_exit_sites = self.vessel_sites[near]
        if self.ln_exit_sites.size == 0:
            self.ln_exit_sites = self.vessel_sites[:1000]

        # ---- 4. 감염세포 리스트 ----
        self.inf_site = np.empty(0, dtype=np.int32)
        self.inf_timer = np.empty(0, dtype=np.int16)
        self.inf_stage = np.empty(0, dtype=np.uint8)
        self.inf_alive = np.empty(0, dtype=bool)

        # ---- 5. 화학신호 농도장 (200 x 200, 1칸 = 75 um) ----
        f = Scale.FIELD_N
        self.chemokine = np.zeros((f, f), dtype=np.float32)
        self.cytokine = np.zeros((f, f), dtype=np.float32)
        self.ifn = np.zeros((f, f), dtype=np.float32)
        self.antiviral = np.zeros((f, f), dtype=np.float32)

        # ---- 6. 전신 스칼라 ----
        self.complement = 0.0
        self.inflammation = 0.0
        self.tissue_damage = 0.0
        self.n_dead_epi = 0
        self.n_regen = 0

    # ------------------------------------------------------------------
    @staticmethod
    def site_of(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        col = x.astype(np.int32)
        row = y.astype(np.int32)
        np.clip(col, 0, Scale.GRID - 1, out=col)
        np.clip(row, 0, Scale.GRID - 1, out=row)
        return row * Scale.GRID + col

    @staticmethod
    def field_of_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        fc = (x / Scale.FIELD_BIN).astype(np.int32)
        fr = (y / Scale.FIELD_BIN).astype(np.int32)
        np.clip(fc, 0, Scale.FIELD_N - 1, out=fc)
        np.clip(fr, 0, Scale.FIELD_N - 1, out=fr)
        return fr, fc

    @staticmethod
    def field_of_site(site: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        row = site // Scale.GRID
        col = site - row * Scale.GRID
        return row // Scale.FIELD_BIN, col // Scale.FIELD_BIN

    # ------------------------------------------------------------------
    def infect_sites(self, sites: np.ndarray) -> int:
        if sites.size == 0:
            return 0
        self.state[sites] = ECLIPSE
        k = sites.size
        self.inf_site = np.concatenate([self.inf_site, sites.astype(np.int32)])
        self.inf_timer = np.concatenate([self.inf_timer, np.zeros(k, np.int16)])
        self.inf_stage = np.concatenate([self.inf_stage, np.zeros(k, np.uint8)])
        self.inf_alive = np.concatenate([self.inf_alive, np.ones(k, bool)])
        return k

    def kill_sites(self, sites: np.ndarray) -> int:
        if sites.size == 0:
            return 0
        sites = np.unique(sites)
        m = (self.state[sites] == ECLIPSE) | (self.state[sites] == PRODUCTIVE)
        sites = sites[m]
        if sites.size == 0:
            return 0
        self.state[sites] = DEAD
        self.n_dead_epi += sites.size
        return sites.size

    def advance_infected(self) -> Tuple[np.ndarray, int]:
        if self.inf_site.size == 0:
            return np.empty(0, np.int32), 0
        st = self.state[self.inf_site]
        self.inf_alive &= (st == ECLIPSE) | (st == PRODUCTIVE)
        alive = self.inf_alive
        if not alive.any():
            self._compact()
            return np.empty(0, np.int32), 0

        self.inf_timer[alive] += 1
        to_prod = alive & (self.inf_stage == 0) & (self.inf_timer >= self.p.eclipse_ticks)
        if to_prod.any():
            self.inf_stage[to_prod] = 1
            self.inf_timer[to_prod] = 0
            self.state[self.inf_site[to_prod]] = PRODUCTIVE
        to_dead = alive & (self.inf_stage == 1) & (self.inf_timer >= self.p.productive_ticks)
        n_apop = int(to_dead.sum())
        if n_apop:
            self.state[self.inf_site[to_dead]] = DEAD
            self.inf_alive[to_dead] = False
            self.n_dead_epi += n_apop

        prod_sites = self.inf_site[self.inf_alive & (self.inf_stage == 1)]
        if self.inf_alive.size > 20000 and self.inf_alive.mean() < 0.5:
            self._compact()
        return prod_sites, n_apop

    def _compact(self):
        k = self.inf_alive
        self.inf_site = self.inf_site[k]
        self.inf_timer = self.inf_timer[k]
        self.inf_stage = self.inf_stage[k]
        self.inf_alive = self.inf_alive[k]

    # ------------------------------------------------------------------
    @staticmethod
    def _diffuse(a: np.ndarray, sigma: float) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma=sigma, mode="nearest")

    def update_signals(self, prod_sites: np.ndarray,
                       phago_field: Optional[Tuple[np.ndarray, np.ndarray]]):
        """
        케모카인 / 사이토카인 / 인터페론 농도장 갱신.
        - 케모카인: 감염세포 + 활성 식세포가 생성 (사양서 ⑭⑲)
        - 사이토카인: 감염세포가 생성, 면역세포 활성화 조절 (사양서 ㉓)
        - 인터페론: 감염세포가 생성, 항바이러스 상태 유도 (사양서 ㉒)
        """
        p = self.p
        if prod_sites.size:
            fr, fc = self.field_of_site(prod_sites)
            np.add.at(self.chemokine, (fr, fc), p.chemo_from_infected)
            np.add.at(self.cytokine, (fr, fc), p.cyto_from_infected)
            np.add.at(self.ifn, (fr, fc), p.ifn_production)
        if phago_field is not None and phago_field[0].size:
            np.add.at(self.chemokine, phago_field, p.chemo_from_phagocyte)

        self.chemokine = self._diffuse(self.chemokine, p.chemo_sigma_field) \
            * (1.0 - p.chemo_decay)
        self.cytokine = self._diffuse(self.cytokine, p.cyto_sigma_field) \
            * (1.0 - p.cyto_decay)
        self.ifn = self._diffuse(self.ifn, p.ifn_sigma_field) * (1.0 - p.ifn_decay_p)

        drive = np.clip(self.ifn * 6.0, 0.0, 1.0)
        self.antiviral += (p.antiviral_rate * drive * (1.0 - self.antiviral)
                           - p.antiviral_decay * self.antiviral)
        np.clip(self.antiviral, 0.0, 1.0, out=self.antiviral)

    def build_chemo_stack(self) -> np.ndarray:
        """8방향 shift 스택 (세포가 '주변 8방향'을 감지하기 위한 자료)"""
        st = np.empty((8, Scale.FIELD_N, Scale.FIELD_N), dtype=np.float32)
        for i, (dc, dr) in enumerate(DIR8_OFF):
            st[i] = np.roll(np.roll(self.chemokine, -dr, axis=0), -dc, axis=1)
        return st

    # ------------------------------------------------------------------
    def update_humoral(self, n_virus: int, n_infected: int):
        p = self.p
        drive = min(1.0, n_virus / 30000.0 + n_infected / 60000.0)
        self.complement += p.complement_on * drive * (1.0 - self.complement)
        self.complement -= p.complement_off * self.complement
        self.complement = float(np.clip(self.complement, 0.0, 1.0))

        target = min(1.0, n_virus * p.inflam_gain_virus
                     + n_infected * p.inflam_gain_infected)
        self.inflammation += 0.05 * (target - self.inflammation)
        self.inflammation -= p.inflam_decay * self.inflammation
        self.inflammation = float(np.clip(self.inflammation, 0.0, 1.0))
        self.tissue_damage += p.damage_rate * self.inflammation

    def regenerate(self):
        if self.n_dead_epi <= self.n_regen:
            return
        dead_idx = np.flatnonzero(self.state == DEAD)
        if dead_idx.size == 0:
            return
        rate = self.p.epi_regen_p * max(0.0, 1.0 - self.inflammation)
        if rate <= 0:
            return
        k = self.rng.binomial(dead_idx.size, rate)
        if k <= 0:
            return
        pick = self.rng.choice(dead_idx, size=min(k, dead_idx.size), replace=False)
        self.state[pick] = HEALTHY
        self.n_regen += pick.size

    def counts(self) -> Dict[str, int]:
        st = self.state
        return {"healthy": int(np.count_nonzero(st == HEALTHY)),
                "eclipse": int(np.count_nonzero(st == ECLIPSE)),
                "productive": int(np.count_nonzero(st == PRODUCTIVE)),
                "dead": int(np.count_nonzero(st == DEAD))}


# =============================================================================
# SECTION 7.  VIRUS POOL  (혈류/확산 이동, 목표탐색 없음 - 사양서 ⑪)
# =============================================================================


class VirusPool:

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.x = np.empty(0, np.float32)
        self.y = np.empty(0, np.float32)
        self.thin_weight = 1.0
        self.thinning_events = 0

    @property
    def n(self) -> int:
        return self.x.size

    def seed_uniform(self, k: int):
        self.x = self.rng.uniform(0, Scale.GRID - 1, k).astype(np.float32)
        self.y = self.rng.uniform(0, Scale.GRID - 1, k).astype(np.float32)

    def diffuse(self):
        """물리적 확산 + 점액섬모 수송. 등방성이며 목적지를 찾지 않는다."""
        n = self.n
        if n == 0:
            return
        s = self.p.virus_sigma
        self.x += self.rng.normal(0.0, s, n).astype(np.float32)
        self.y += self.rng.normal(0.0, s, n).astype(np.float32)
        SystematicMovementEngine._bound(self.x, self.y)

    def add(self, x, y):
        self.x = np.concatenate([self.x, x.astype(np.float32)])
        self.y = np.concatenate([self.y, y.astype(np.float32)])
        if self.x.size > self.p.max_virions:
            keep = self.rng.random(self.x.size) < 0.5
            self.x = self.x[keep]
            self.y = self.y[keep]
            self.thin_weight *= 2.0
            self.thinning_events += 1

    def remove(self, mask) -> int:
        k = int(np.count_nonzero(mask))
        if k:
            self.x = self.x[~mask]
            self.y = self.y[~mask]
        return k

    def remove_idx(self, idx) -> int:
        if idx.size == 0:
            return 0
        m = np.zeros(self.x.size, bool)
        m[idx] = True
        return self.remove(m)


# =============================================================================
# SECTION 8.  IMMUNE CELL POOL
# =============================================================================

NEUT, MONO, NK, TH, TC, BC = 0, 1, 2, 3, 4, 5
CELL_NAME = {NEUT: "중성구", MONO: "단핵구/대식세포", NK: "NK세포",
             TH: "조력T세포", TC: "살해T세포", BC: "B세포"}


class ImmuneCellPool:

    FIELDS = ("x", "y", "hx", "hy", "ctype", "age", "life", "speed",
              "capacity", "specific", "in_vessel", "activated", "alive")

    def __init__(self, cap: int = 16384):
        self.cap = cap
        self.n = 0
        self.x = np.zeros(cap, np.float32)
        self.y = np.zeros(cap, np.float32)
        self.hx = np.zeros(cap, np.float32)     # 극성(직전 이동방향)
        self.hy = np.zeros(cap, np.float32)
        self.ctype = np.zeros(cap, np.uint8)
        self.age = np.zeros(cap, np.int32)
        self.life = np.zeros(cap, np.int32)
        self.speed = np.zeros(cap, np.float32)
        self.capacity = np.zeros(cap, np.int16)
        self.specific = np.zeros(cap, bool)
        self.in_vessel = np.zeros(cap, bool)
        self.activated = np.zeros(cap, bool)
        self.alive = np.zeros(cap, bool)

    def _grow(self, need: int):
        while self.n + need > self.cap:
            self.cap *= 2
        for name in self.FIELDS:
            arr = getattr(self, name)
            new = np.zeros(self.cap, arr.dtype)
            new[:arr.size] = arr
            setattr(self, name, new)

    def add(self, k, ctype, speed, life, capacity=0, specific=False,
            x=None, y=None, in_vessel=True, age=None):
        if k <= 0:
            return
        if self.n + k > self.cap:
            self._grow(k)
        s = slice(self.n, self.n + k)
        self.x[s] = x
        self.y[s] = y
        self.hx[s] = 0.0
        self.hy[s] = 0.0
        self.ctype[s] = ctype
        self.life[s] = life
        self.age[s] = 0 if age is None else age
        self.speed[s] = speed
        self.capacity[s] = capacity
        self.specific[s] = specific
        self.in_vessel[s] = in_vessel
        self.activated[s] = False
        self.alive[s] = True
        self.n += k

    def mask(self, ctype) -> np.ndarray:
        return self.alive[:self.n] & (self.ctype[:self.n] == ctype)

    def count(self, ctype) -> int:
        return int(np.count_nonzero(self.mask(ctype)))

    def count_specific(self, ctype) -> int:
        return int(np.count_nonzero(self.mask(ctype) & self.specific[:self.n]))

    def compact(self):
        keep = np.flatnonzero(self.alive[:self.n])
        k = keep.size
        for name in self.FIELDS:
            arr = getattr(self, name)
            arr[:k] = arr[keep]
            arr[k:self.n] = 0
        self.alive[:k] = True
        self.n = k


# =============================================================================
# SECTION 9.  LYMPH NODE  (사양서 ㉕~㉘)
# =============================================================================


class LymphNode:

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.dc_queue = np.zeros(p.dc_delay_ticks + 1, np.float64)
        self.qi = 0
        self.antigen = 0.0
        self.primed = False
        self.prime_clock = 0
        self.activated = False
        self.contracting = False

        self.n_cd4 = p.precursor_cd4
        self.n_cd8 = p.precursor_cd8
        self.n_b = p.precursor_b
        self.n_pc_short = 0.0
        self.n_pc_gc = 0.0
        self.memory_t = 0.0
        self.memory_b = 0.0
        self.peak_cd4 = p.precursor_cd4
        self.peak_cd8 = p.precursor_cd8
        self.peak_b = p.precursor_b
        self.peak_pc = 0.0

        self.t_antigen_arrival: Optional[int] = None
        self.t_priming_done: Optional[int] = None
        self.t_cd4_act: Optional[int] = None
        self.t_cd8_act: Optional[int] = None
        self.t_b_act: Optional[int] = None
        self.t_first_emigration: Optional[int] = None

    def deposit_antigen(self, amount: float):
        if amount <= 0:
            return
        j = (self.qi + self.p.dc_delay_ticks) % self.dc_queue.size
        self.dc_queue[j] += amount

    def update(self, tick: int):
        p = self.p
        arrive = self.dc_queue[self.qi]
        self.dc_queue[self.qi] = 0.0
        self.qi = (self.qi + 1) % self.dc_queue.size
        if arrive > 0 and self.t_antigen_arrival is None:
            self.t_antigen_arrival = tick
        self.antigen += arrive
        self.antigen *= (1.0 - p.antigen_decay)

        if not self.primed and self.antigen >= p.antigen_threshold:
            self.primed = True
            self.prime_clock = 0
        if self.primed and not self.activated:
            self.prime_clock += 1
            if self.prime_clock >= p.priming_ticks:
                self.activated = True
                self.t_priming_done = tick
                self.t_cd4_act = tick
                self.t_cd8_act = tick
                self.t_b_act = tick
        if not self.activated:
            return

        elapsed = tick - (self.t_priming_done or tick)
        expansion_on = elapsed < p.program_ticks
        gr = p.net_expansion
        help_f = self.n_cd4 / (self.n_cd4 + 5.0 * p.precursor_cd4)

        if expansion_on:
            if self.n_cd4 < p.precursor_cd4 * p.cd4_max_expansion:
                self.n_cd4 *= math.exp(p.cd4_growth * gr)
            if self.n_cd8 < p.precursor_cd8 * p.cd8_max_expansion:
                self.n_cd8 *= math.exp(p.cd8_growth * gr * (0.55 + 0.9 * help_f))
            if self.n_b < p.precursor_b * p.b_max_expansion:
                self.n_b *= math.exp(p.b_growth * gr * (0.45 + 1.0 * help_f))
        else:
            self.contracting = True
            self.n_cd8 = max(self.n_cd8 * (1 - p.contraction_p),
                             self.peak_cd8 * p.memory_fraction)
            self.n_cd4 = max(self.n_cd4 * (1 - p.contraction_p),
                             self.peak_cd4 * p.memory_fraction)
            self.n_b = max(self.n_b * (1 - p.b_contraction_p),
                           self.peak_b * p.memory_b_fraction)
            self.memory_t = min(self.n_cd8 + self.n_cd4,
                                (self.peak_cd8 + self.peak_cd4) * p.memory_fraction)
            self.memory_b = min(self.n_b, self.peak_b * p.memory_b_fraction)

        if expansion_on and elapsed >= p.ef_delay_ticks:
            self.n_pc_short += p.ef_diff_rate * self.n_b
        self.n_pc_short *= (1.0 - p.pc_short_decay)
        if p.gc_delay_ticks <= elapsed < p.gc_end_ticks:
            self.n_pc_gc += p.gc_diff_rate * self.n_b * min(1.0, help_f * 2.0)
        self.n_pc_gc *= (1.0 - p.pc_gc_decay)

        self.peak_cd8 = max(self.peak_cd8, self.n_cd8)
        self.peak_cd4 = max(self.peak_cd4, self.n_cd4)
        self.peak_b = max(self.peak_b, self.n_b)
        self.peak_pc = max(self.peak_pc, self.n_pc_short + self.n_pc_gc)

    def emigrate(self, tick: int) -> Tuple[int, int]:
        p = self.p
        if not self.activated or tick * Scale.DT_HOUR < p.emigration_start_h:
            return 0, 0
        n_ctl = int(self.rng.poisson(p.emigration_p * self.n_cd8))
        n_th = int(self.rng.poisson(p.emigration_p * 0.30 * self.n_cd4))
        if (n_ctl or n_th) and self.t_first_emigration is None:
            self.t_first_emigration = tick
        return n_ctl, n_th


class Antibody:

    def __init__(self, p: Params):
        self.p = p
        self.igm = 0.0
        self.igg = 0.0
        self.t_igm: Optional[int] = None
        self.t_igg: Optional[int] = None
        self.peak_total = 0.0
        self.neutralized = 0

    @property
    def total(self) -> float:
        return self.igm + self.igg

    def update(self, ln: LymphNode, tick: int):
        p = self.p
        self.igm += p.ab_per_pc_igm * ln.n_pc_short
        self.igg += p.ab_per_pc_igg * ln.n_pc_gc
        self.igm *= (1.0 - p.igm_decay_p)
        self.igg *= (1.0 - p.igg_decay_p)
        if self.t_igm is None and self.igm >= p.ab_detect_threshold:
            self.t_igm = tick
        if self.t_igg is None and self.igg >= p.ab_detect_threshold:
            self.t_igg = tick
        self.peak_total = max(self.peak_total, self.total)

    def neutralization_rate(self) -> float:
        c = self.total
        return self.p.ab_neutralize_max * c / (c + self.p.ab_kd)

    def opsonin_bonus(self) -> float:
        c = self.total
        return self.p.ab_opsonin * c / (c + self.p.ab_kd)

    def infection_block(self) -> float:
        c = self.total
        return self.p.ab_block_max * c / (c + self.p.ab_block_kd)


# =============================================================================
# SECTION 10.  SIMULATION CONTROLLER
# =============================================================================


class SystematicImmuneSimulation:

    def __init__(self, p: Optional[Params] = None, seed: Optional[int] = None,
                 verbose: bool = True):
        self.p = p or Params()
        if seed is not None:
            self.p.seed = seed
        self.rng = np.random.default_rng(self.p.seed)
        self.movement_mode = "systematic"
        self.mover = SystematicMovementEngine()
        self.verbose = verbose

        self.env = Environment(self.p, self.rng)
        self.virus = VirusPool(self.p, self.rng)
        self.cells = ImmuneCellPool()
        self.ln = LymphNode(self.p, self.rng)
        self.ab = Antibody(self.p)

        self.tick = 0
        self.history: List[dict] = []
        self.daily: List[dict] = []

        self.cleared_virus = 0
        self.cleared_by_phago = 0
        self.cleared_by_decay = 0
        self.cleared_by_complement = 0
        self.cleared_by_antibody = 0
        self.killed_infected = 0
        self.killed_by_nk = 0
        self.killed_by_ctl = 0
        self.apoptosis_infected = 0
        self.dead_immune = 0
        self.total_infections = 0
        self.ctl_contact = 0
        self.ctl_agent_ticks = 0
        self.n_extravasated = 0

        self.t_first_arrival: Optional[int] = None   # 사양서 40-⑥
        self.t_first_extravasation: Optional[int] = None

        self._init_population()

    # ------------------------------------------------------------------
    def _vessel_positions(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """혈관 칸 위에 균등 배치 (초기 배치는 이동이 아니라 초기조건)"""
        idx = self.rng.choice(self.env.vessel_sites, size=k, replace=True)
        row = idx // Scale.GRID
        col = idx - row * Scale.GRID
        return (col + 0.5).astype(np.float32), (row + 0.5).astype(np.float32)

    def _ln_exit_positions(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.rng.choice(self.env.ln_exit_sites, size=k, replace=True)
        row = idx // Scale.GRID
        col = idx - row * Scale.GRID
        return (col + 0.5).astype(np.float32), (row + 0.5).astype(np.float32)

    def _init_population(self):
        p = self.p
        self.virus.seed_uniform(p.n_virus0)

        def put(k, ctype, speed, life, cap=0):
            x, y = self._vessel_positions(k)
            age = self.rng.integers(0, max(1, life), k)
            self.cells.add(k, ctype, speed, life, cap, x=x, y=y,
                           in_vessel=True, age=age)

        # 사양서 ㉚: 면역세포는 혈관에서 출발한다
        put(p.n_neutrophil, NEUT, p.speed_neut, p.life_neut, p.neut_capacity)
        put(p.n_monocyte, MONO, p.speed_mono, p.life_mono, p.mono_capacity)
        put(p.n_nk, NK, p.speed_nk, p.life_nk)
        put(p.n_helper_t, TH, p.speed_naive_t, p.life_naive)
        put(p.n_killer_t, TC, p.speed_naive_t, p.life_naive)
        put(p.n_bcell, BC, p.speed_b, p.life_naive)

    # ------------------------------------------------------------------
    def _match_same_site(self, p_site: np.ndarray, v_site: np.ndarray):
        if p_site.size == 0 or v_site.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        pu = np.unique(p_site)
        cand = np.flatnonzero(np.isin(v_site, pu, kind="table"))
        if cand.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        cs = v_site[cand]
        order = np.argsort(cs, kind="stable")
        cs_s, cand_s = cs[order], cand[order]
        uniq, start, counts = np.unique(cs_s, return_index=True, return_counts=True)
        pos = np.minimum(np.searchsorted(uniq, p_site), uniq.size - 1)
        has = uniq[pos] == p_site
        idx_p = np.flatnonzero(has)
        if idx_p.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        g = pos[idx_p]
        off = self.rng.integers(0, counts[g])
        v_idx = cand_s[start[g] + off]
        _, keep = np.unique(v_idx, return_index=True)
        return idx_p[keep], v_idx[keep]

    # ------------------------------------------------------------------
    def step(self):
        p, env, c = self.p, self.env, self.cells
        self.tick += 1
        t = self.tick

        # ---------- A. 감염세포 진행 ----------
        prod_sites, n_apop = env.advance_infected()
        self.apoptosis_infected += n_apop

        # ---------- B. 바이러스 생산 ----------
        if prod_sites.size:
            fr, fc = Environment.field_of_site(prod_sites)
            av = env.antiviral[fr, fc]
            rate = (p.burst_agents_per_site / p.productive_ticks) * \
                   (1.0 - p.ifn_block_release * av) / max(1.0, self.virus.thin_weight)
            k = self.rng.poisson(np.maximum(rate, 0.0))
            tot = int(k.sum())
            if tot:
                src = np.repeat(prod_sites, k)
                row = src // Scale.GRID
                col = src - row * Scale.GRID
                self.virus.add((col + self.rng.random(tot)).astype(np.float32),
                               (row + self.rng.random(tot)).astype(np.float32))

        # ---------- C. 바이러스 이동 (확산/수송) ----------
        self.virus.diffuse()

        # ---------- D. 감염 ----------
        if self.virus.n:
            v_site = Environment.site_of(self.virus.x, self.virus.y)
            sus = np.flatnonzero(env.state[v_site] == HEALTHY)
            if sus.size:
                s_sites = v_site[sus]
                fr, fc = Environment.field_of_site(s_sites)
                pe = p.p_infect * (1.0 - env.antiviral[fr, fc]) * \
                    (1.0 - self.ab.infection_block())
                if self.virus.thin_weight > 1.0:
                    pe = 1.0 - (1.0 - pe) ** self.virus.thin_weight
                ok = self.rng.random(sus.size) < pe
                if ok.any():
                    n_new = env.infect_sites(np.unique(s_sites[ok]))
                    self.total_infections += n_new
                    self.virus.remove_idx(sus[ok])

        # ---------- E. 면역세포 노화/사멸 ----------
        n = c.n
        if n:
            alive = c.alive[:n]
            c.age[:n][alive] += 1
            died = alive & (c.age[:n] >= c.life[:n])
            eff = alive & c.specific[:n] & ((c.ctype[:n] == TC) | (c.ctype[:n] == TH))
            if eff.any():
                pd_eff = (p.eff_t_death_contract if self.ln.contracting
                          else p.eff_t_death_expand)
                died |= eff & (self.rng.random(n) < pd_eff)
            k = int(np.count_nonzero(died))
            if k:
                c.alive[:n][died] = False
                self.dead_immune += k
            if n > 5000 and c.alive[:n].mean() < 0.85:
                c.compact()

        # ---------- F. 체계적 이동 (★ 핵심) ----------
        chemo_stack = env.build_chemo_stack()
        n = c.n
        idx = np.flatnonzero(c.alive[:n])
        if idx.size:
            fr, fc = Environment.field_of_xy(c.x[idx], c.y[idx])
            here = env.chemokine[fr, fc]

            in_v = c.in_vessel[idx]
            # -- 혈관외 유출 판정 (결정론적 역치, 사양서 ㉚) --
            ex = in_v & (here > p.extravasation_threshold)
            if ex.any():
                gi = idx[ex]
                c.in_vessel[gi] = False
                self.n_extravasated += gi.size
                if self.t_first_extravasation is None:
                    self.t_first_extravasation = t
            # -- 조직 내 세포가 신호를 완전히 잃으면 혈관으로 복귀 --
            back = (~in_v) & (here < p.reentry_threshold) & (c.ctype[idx] == NEUT)
            in_v = c.in_vessel[idx]

            vsel = np.flatnonzero(in_v)
            tsel = np.flatnonzero(~in_v)

            if vsel.size:
                gi = idx[vsel]
                xs, ys = c.x[gi].copy(), c.y[gi].copy()
                hd = np.stack([c.hx[gi], c.hy[gi]], axis=1)
                site = Environment.site_of(xs, ys)
                fdx = env.flow_dx[site].copy()
                fdy = env.flow_dy[site].copy()
                # 혈관 밖으로 벗어난 경우 가장 가까운 혈관 행으로 스냅
                nov = (np.abs(fdx) + np.abs(fdy)) < 1e-6
                if nov.any():
                    rr = ys[nov].astype(np.int32)
                    base = (rr // Scale.EPI_PERIOD) * Scale.EPI_PERIOD + 13
                    ys[nov] = np.clip(base, 0, Scale.GRID - 1) + 0.5
                    site2 = Environment.site_of(xs[nov], ys[nov])
                    fdx[nov] = env.flow_dx[site2]
                    fdy[nov] = env.flow_dy[site2]
                    fdx[nov] = np.where(np.abs(fdx[nov]) + np.abs(fdy[nov]) < 1e-6,
                                        1.0, fdx[nov])
                vfr, vfc = Environment.field_of_xy(xs, ys)
                self.mover.move_vessel(xs, ys, hd, fdx, fdy, p.vessel_speed,
                                       chemo_stack, vfr, vfc)
                c.x[gi], c.y[gi] = xs, ys
                c.hx[gi], c.hy[gi] = hd[:, 0], hd[:, 1]

            if tsel.size:
                gi = idx[tsel]
                xs, ys = c.x[gi].copy(), c.y[gi].copy()
                hd = np.stack([c.hx[gi], c.hy[gi]], axis=1)
                tfr, tfc = Environment.field_of_xy(xs, ys)
                self.mover.move_tissue(xs, ys, hd, c.speed[gi],
                                       chemo_stack, tfr, tfc)
                c.x[gi], c.y[gi] = xs, ys
                c.hx[gi], c.hy[gi] = hd[:, 0], hd[:, 1]

        # ---------- G. 활성화 상태 (사이토카인, 사양서 ㉓) ----------
        n = c.n
        idx = np.flatnonzero(c.alive[:n])
        if idx.size:
            fr, fc = Environment.field_of_xy(c.x[idx], c.y[idx])
            cyto = env.cytokine[fr, fc]
            c.activated[idx] = cyto > p.cyto_activation_threshold
            cell_site = Environment.site_of(c.x[idx], c.y[idx])
        else:
            cell_site = np.empty(0, np.int32)
        ct = c.ctype[idx] if idx.size else np.empty(0, np.uint8)

        # ---------- H. 식균작용 ----------
        antigen_in = 0.0
        phago_field = None
        if idx.size and self.virus.n:
            v_site = Environment.site_of(self.virus.x, self.virus.y)
            opson = min(0.95, self.ab.opsonin_bonus()
                        + p.complement_opsonin * env.complement)
            act_fr, act_fc = [], []
            for ctype, base_p in ((NEUT, p.neut_phago_p), (MONO, p.mono_phago_p)):
                sel = np.flatnonzero((ct == ctype) & (c.capacity[idx] > 0)
                                     & (~c.in_vessel[idx]))
                if sel.size == 0:
                    continue
                loc, vidx = self._match_same_site(cell_site[sel], v_site)
                if loc.size == 0:
                    continue
                succ = self.rng.random(loc.size) < min(0.98, base_p + opson)
                if not succ.any():
                    continue
                gi = idx[sel[loc[succ]]]
                c.capacity[gi] -= 1
                spent = gi[c.capacity[gi] <= 0]
                if spent.size:
                    c.alive[spent] = False
                    self.dead_immune += spent.size
                k = self.virus.remove_idx(vidx[succ])
                self.cleared_by_phago += k
                self.cleared_virus += k
                # 활성 식세포는 2차 케모카인을 방출한다 (사양서 ⑲)
                a, b = Environment.field_of_xy(c.x[gi], c.y[gi])
                act_fr.append(a); act_fc.append(b)
                if ctype == MONO:
                    antigen_in += k * 1.0
                v_site = Environment.site_of(self.virus.x, self.virus.y)
            if act_fr:
                phago_field = (np.concatenate(act_fr), np.concatenate(act_fc))

        # ---------- I. NK세포 ----------
        if idx.size:
            sel = np.flatnonzero((ct == NK) & (~c.in_vessel[idx]))
            if sel.size:
                s = cell_site[sel]
                stt = env.state[s]
                tgt = (stt == PRODUCTIVE) | (stt == ECLIPSE)
                if tgt.any():
                    if self.t_first_arrival is None:
                        self.t_first_arrival = t
                    boost = 1.0 + 0.5 * float(env.inflammation)
                    ok = self.rng.random(int(tgt.sum())) < min(0.95, p.nk_kill_p * boost)
                    k = env.kill_sites(s[tgt][ok])
                    self.killed_by_nk += k
                    self.killed_infected += k

        # ---------- J. 살해 T세포 ----------
        if idx.size:
            sel = np.flatnonzero((ct == TC) & c.specific[idx] & (~c.in_vessel[idx]))
            self.ctl_agent_ticks += int(np.count_nonzero(
                (ct == TC) & c.specific[idx]))
            if sel.size:
                s = cell_site[sel]
                stt = env.state[s]
                tgt = (stt == PRODUCTIVE) | (stt == ECLIPSE)
                nt = int(tgt.sum())
                self.ctl_contact += nt
                if nt:
                    ok = self.rng.random(nt) < p.ctl_kill_p
                    k = env.kill_sites(s[tgt][ok])
                    self.killed_by_ctl += k
                    self.killed_infected += k

        # 첫 도착 시각: 식세포가 감염칸에 도달한 경우도 포함
        if self.t_first_arrival is None and idx.size:
            tis = np.flatnonzero(~c.in_vessel[idx])
            if tis.size:
                stt = env.state[cell_site[tis]]
                if np.any((stt == PRODUCTIVE) | (stt == ECLIPSE)):
                    self.t_first_arrival = t

        # ---------- K. 항원 전달 ----------
        n_infected_now = int(env.inf_alive.sum()) if env.inf_alive.size else 0
        antigen_in += 0.005 * n_infected_now
        self.ln.deposit_antigen(antigen_in)

        # ---------- L. 바이러스 제거 (비특이/보체/항체) ----------
        if self.virus.n:
            pd_ = p.virus_decay_p
            pc = p.complement_lysis * env.complement
            pa = self.ab.neutralization_rate()
            ptot = 1.0 - (1 - pd_) * (1 - pc) * (1 - pa)
            rem = self.rng.random(self.virus.n) < ptot
            k = self.virus.remove(rem)
            if k:
                tot = pd_ + pc + pa
                self.cleared_by_decay += int(k * pd_ / tot)
                self.cleared_by_complement += int(k * pc / tot)
                self.cleared_by_antibody += int(k * pa / tot)
                self.cleared_virus += k

        # ---------- M. 림프절 / 항체 ----------
        self.ln.update(t)
        n_ctl, n_th = self.ln.emigrate(t)
        if n_ctl:
            x, y = self._ln_exit_positions(n_ctl)
            c.add(n_ctl, TC, p.speed_eff_t, p.life_naive, specific=True,
                  x=x, y=y, in_vessel=True)
        if n_th:
            x, y = self._ln_exit_positions(n_th)
            c.add(n_th, TH, p.speed_eff_t, p.life_naive, specific=True,
                  x=x, y=y, in_vessel=True)
        self.ab.update(self.ln, t)

        # ---------- N. 신호장 / 체액 / 동원 / 재생 ----------
        env.update_signals(prod_sites, phago_field)
        env.update_humoral(self.virus.n, n_infected_now)
        self._recruit()
        env.regenerate()

        self._record()

    # ------------------------------------------------------------------
    def _recruit(self):
        p = self.p
        infl = self.env.inflammation
        spec = {NEUT: (p.n_neutrophil, p.recruit_neut_max, p.speed_neut,
                       p.life_neut, p.neut_capacity),
                MONO: (p.n_monocyte, p.recruit_mono_max, p.speed_mono,
                       p.life_mono, p.mono_capacity),
                NK: (p.n_nk, p.recruit_nk_max, p.speed_nk, p.life_nk, 0)}
        for ctype, (base, mx, sp, lf, cap) in spec.items():
            gap = base * (1.0 + mx * infl) - self.cells.count(ctype)
            if gap <= 0:
                continue
            k = int(self.rng.poisson(gap * p.recruit_rate))
            if k:
                x, y = self._vessel_positions(k)
                self.cells.add(k, ctype, sp, lf, cap, x=x, y=y, in_vessel=True)

    # ------------------------------------------------------------------
    RECORD_EVERY = 5

    def _snapshot(self) -> dict:
        env, c, ln, ab = self.env, self.cells, self.ln, self.ab
        cn = env.counts()
        infected = cn["eclipse"] + cn["productive"]
        n = c.n
        alive = c.alive[:n]
        in_tissue = int(np.count_nonzero(alive & ~c.in_vessel[:n]))
        return {
            "tick": self.tick, "hours": self.tick * Scale.DT_HOUR,
            "day": self.tick / Scale.TICKS_PER_DAY,
            "virus": self.virus.n, "infected": infected,
            "eclipse": cn["eclipse"], "productive": cn["productive"],
            "healthy": cn["healthy"], "dead_epi": cn["dead"],
            "neutrophil": c.count(NEUT), "monocyte": c.count(MONO),
            "nk": c.count(NK), "helper_t": c.count(TH),
            "helper_t_spec": c.count_specific(TH),
            "killer_t": c.count(TC), "killer_t_spec": c.count_specific(TC),
            "bcell": c.count(BC),
            "memory": ln.memory_t + ln.memory_b,
            "antibody": ab.total, "igm": ab.igm, "igg": ab.igg,
            "chemokine": float(env.chemokine.mean()),
            "chemokine_max": float(env.chemokine.max()),
            "cytokine": float(env.cytokine.mean()),
            "interferon": float(env.ifn.mean()),
            "antiviral": float(env.antiviral.mean()),
            "complement": env.complement,
            "inflammation": env.inflammation,
            "damage": env.tissue_damage,
            "cleared_virus": self.cleared_virus,
            "killed_infected": self.killed_infected,
            "apoptosis": self.apoptosis_infected,
            "dead_immune": self.dead_immune,
            "in_tissue": in_tissue,
            "ln_cd8": ln.n_cd8, "ln_pc": ln.n_pc_short + ln.n_pc_gc,
        }

    def _record(self):
        if self.tick % self.RECORD_EVERY == 0 or self.tick % Scale.TICKS_PER_DAY == 0:
            self.history.append(self._snapshot())

    # ------------------------------------------------------------------
    def run(self):
        p = self.p
        t0 = time.time()
        self.daily.append(self._snapshot())
        clear_streak = 0
        max_ticks = p.max_days * Scale.TICKS_PER_DAY
        while self.tick < max_ticks:
            self.step()
            if self.tick % Scale.TICKS_PER_DAY == 0:
                self.daily.append(self._snapshot())
                if self.verbose:
                    d = self.tick // Scale.TICKS_PER_DAY
                    s = self.daily[-1]
                    print(f"  Day {d:2d}: V={s['virus']:>8,} I={s['infected']:>7,} "
                          f"H={s['healthy']:>7,} chemo={s['chemokine']:.4f} "
                          f"tissue-cells={s['in_tissue']:>7,}", flush=True)
            n_inf = int(self.env.inf_alive.sum()) if self.env.inf_alive.size else 0
            if self.virus.n == 0 and n_inf == 0:
                clear_streak += 1
            else:
                clear_streak = 0
            if clear_streak >= p.stop_after_clear_days * Scale.TICKS_PER_DAY:
                break
        if self.daily[-1]["tick"] != self.tick:
            self.daily.append(self._snapshot())
        self.wall_time = time.time() - t0


# =============================================================================
# SECTION 11.  DAILY TABLE  (사양서 ㉟ - 하루 경과마다 필수 출력)
# =============================================================================

DAILY_ROWS = [
    ("병원체 수",            lambda s: f"{s['virus']:,}"),
    ("감염세포 수",          lambda s: f"{s['infected']:,}"),
    ("정상세포 수",          lambda s: f"{s['healthy']:,}"),
    ("중성구 수",            lambda s: f"{s['neutrophil']:,}"),
    ("단핵구/대식세포 수",   lambda s: f"{s['monocyte']:,}"),
    ("NK세포 수",            lambda s: f"{s['nk']:,}"),
    ("조력 T세포 수",        lambda s: f"{s['helper_t']:,}"),
    ("살해 T세포 수",        lambda s: f"{s['killer_t']:,}"),
    ("B세포 수",             lambda s: f"{s['bcell']:,}"),
    ("기억세포 수",          lambda s: f"{s['memory']:,.0f}"),
    ("항체 농도(ug/mL)",     lambda s: f"{s['antibody']:.3f}"),
    ("케모카인 농도",        lambda s: f"{s['chemokine']:.4f}"),
    ("사이토카인 농도",      lambda s: f"{s['cytokine']:.4f}"),
    ("인터페론 농도",        lambda s: f"{s['interferon']:.4f}"),
    ("보체 활성도(%)",       lambda s: f"{s['complement']*100:.2f}"),
    ("염증 정도(%)",         lambda s: f"{s['inflammation']*100:.2f}"),
    ("제거된 병원체 수",     lambda s: f"{s['cleared_virus']:,}"),
    ("제거된 감염세포 수",   lambda s: f"{s['killed_infected']:,}"),
    ("면역세포 사망/소모량", lambda s: f"{s['dead_immune']:,}"),
]


def print_daily_table(daily: List[dict], title: str, width: int = 12):
    print()
    print("#" * 110)
    print(f"  {title}")
    print("#" * 110)
    labels = [f"Day {int(round(s['day']))}" if abs(s['day'] - round(s['day'])) < 0.01
              else f"D{s['day']:.2f}" for s in daily]
    labels[-1] = labels[-1] + "*" if abs(daily[-1]['day']
                                         - round(daily[-1]['day'])) > 0.01 else labels[-1]
    hdr = f"{'항목':<22}" + "".join(f"{l:>{width}}" for l in labels)
    print(hdr)
    print("-" * len(hdr))
    for name, fn in DAILY_ROWS:
        print(f"{name:<22}" + "".join(f"{fn(s):>{width}}" for s in daily))
    print("-" * len(hdr))
    print("  * 마지막 열 = 병원체 완전 소멸 후 최종 시점 (일 단위가 아닐 수 있음)")
    print()


# =============================================================================
# SECTION 12.  ANALYSIS & VALIDATION
# =============================================================================


def _ser(h, k):
    return np.array([x[k] for x in h], float)




def _first_reach(h, key, frac=0.95):
    """정점의 frac 배에 처음 도달한 시각. 포화 평탄구간에서 argmax 가
    임의의 늦은 시점을 고르는 문제를 피하기 위한 유입 시점 지표."""
    v = _ser(h, key)
    if v.size == 0 or v.max() <= 0:
        return float("nan")
    thr = v.max() * frac
    for j in range(v.size):
        if v[j] >= thr:
            return float(h[j]["day"])
    return float("nan")


def _peak(h, k):
    v = _ser(h, k)
    if v.size == 0:
        return 0.0, float("nan")
    i = int(np.argmax(v))
    return float(v[i]), float(h[i]["day"])


def _below(h, k, frac):
    v = _ser(h, k)
    if v.size == 0 or v.max() <= 0:
        return None
    i = int(np.argmax(v))
    thr = v[i] * frac
    for j in range(i, v.size):
        if v[j] <= thr:
            return float(h[j]["day"])
    return None


def _zero(h, k):
    v = _ser(h, k)
    if v.size == 0:
        return None
    i = int(np.argmax(v))
    for j in range(i, v.size):
        if v[j] <= 0:
            return float(h[j]["day"])
    return None


def analyze(sim: SystematicImmuneSimulation) -> dict:
    h = sim.history
    tpd = Scale.TICKS_PER_DAY
    R = {}
    R["virus_peak"], R["virus_peak_day"] = _peak(h, "virus")
    R["virus_final"] = h[-1]["virus"]
    R["virus_t50"] = _below(h, "virus", 0.5)
    R["virus_t90"] = _below(h, "virus", 0.1)
    R["virus_shed_end_1pct"] = _below(h, "virus", 0.01)
    R["virus_clear_day"] = _zero(h, "virus")
    R["inf_peak"], R["inf_peak_day"] = _peak(h, "infected")
    R["inf_t50"] = _below(h, "infected", 0.5)
    R["inf_clear_day"] = _zero(h, "infected")
    R["neut_peak"], _ = _peak(h, "neutrophil")
    R["neut_peak_day"] = _first_reach(h, "neutrophil")
    R["mono_peak"], _ = _peak(h, "monocyte")
    R["mono_peak_day"] = _first_reach(h, "monocyte")
    R["nk_peak"], _ = _peak(h, "nk")
    R["nk_peak_day"] = _first_reach(h, "nk")
    R["chemo_peak"], R["chemo_peak_day"] = _peak(h, "chemokine")
    R["cyto_peak"], R["cyto_peak_day"] = _peak(h, "cytokine")
    R["ifn_peak"], R["ifn_peak_day"] = _peak(h, "interferon")
    R["comp_peak"], R["comp_peak_day"] = _peak(h, "complement")
    R["inflam_peak"], R["inflam_peak_day"] = _peak(h, "inflammation")
    R["damage_final"] = h[-1]["damage"]
    R["ctl_peak"], R["ctl_peak_day"] = _peak(h, "killer_t_spec")
    R["th_peak"], R["th_peak_day"] = _peak(h, "helper_t_spec")
    R["pc_peak"], R["pc_peak_day"] = _peak(h, "ln_pc")
    R["ab_peak"], R["ab_peak_day"] = _peak(h, "antibody")
    ln = sim.ln
    R["t_antigen"] = ln.t_antigen_arrival / tpd if ln.t_antigen_arrival else None
    R["t_prime"] = ln.t_priming_done / tpd if ln.t_priming_done else None
    R["t_cd4"] = ln.t_cd4_act / tpd if ln.t_cd4_act else None
    R["t_cd8"] = ln.t_cd8_act / tpd if ln.t_cd8_act else None
    R["t_b"] = ln.t_b_act / tpd if ln.t_b_act else None
    R["t_emig"] = ln.t_first_emigration / tpd if ln.t_first_emigration else None
    R["t_igm"] = sim.ab.t_igm / tpd if sim.ab.t_igm else None
    R["t_igg"] = sim.ab.t_igg / tpd if sim.ab.t_igg else None
    R["t_first_extravasation"] = (sim.t_first_extravasation / tpd
                                  if sim.t_first_extravasation else None)
    R["t_first_arrival"] = (sim.t_first_arrival / tpd
                            if sim.t_first_arrival else None)
    R["memory_final"] = ln.memory_t + ln.memory_b
    R["memory_ratio"] = ln.memory_t / ln.peak_cd8 if ln.peak_cd8 > 0 else 0.0

    days = _ser(h, "day"); inf = _ser(h, "infected")
    m = (days >= 0.4) & (days <= 1.3) & (inf > 5)
    if m.sum() > 10:
        r = float(np.polyfit(days[m], np.log(inf[m]), 1)[0])
        Tg = (sim.p.eclipse_ticks + 0.5 * sim.p.productive_ticks) / Scale.TICKS_PER_HOUR
        R["growth_rate_per_day"] = r
        R["R0_est"] = float(math.exp(r * Tg / 24.0))
    else:
        R["growth_rate_per_day"] = float("nan"); R["R0_est"] = float("nan")

    R["ctl_kill_per_day"] = (sim.killed_by_ctl / sim.ctl_agent_ticks * tpd
                             if sim.ctl_agent_ticks else float("nan"))
    R["ctl_contact_per_day"] = (sim.ctl_contact / sim.ctl_agent_ticks * tpd
                                if sim.ctl_agent_ticks else float("nan"))
    R["max_target_depletion"] = 1.0 - float(_ser(h, "healthy").min()) / sim.env.n_target
    R["target_depletion"] = 1.0 - h[-1]["healthy"] / sim.env.n_target
    R["killed_by_nk"] = sim.killed_by_nk
    R["killed_by_ctl"] = sim.killed_by_ctl
    R["apoptosis"] = sim.apoptosis_infected
    R["cleared_by_phago"] = sim.cleared_by_phago
    R["cleared_by_decay"] = sim.cleared_by_decay
    R["cleared_by_complement"] = sim.cleared_by_complement
    R["cleared_by_antibody"] = sim.cleared_by_antibody
    R["total_infections"] = sim.total_infections
    R["n_extravasated"] = sim.n_extravasated
    R["grad_frac"] = (sim.mover.n_gradient_guided /
                      max(1, sim.mover.n_gradient_guided + sim.mover.n_persistence))
    return R


@dataclass
class Check:
    item: str
    lit: str
    ref: str
    sim: str
    verdict: str


def _judge(v, lo, hi, tol=0.35):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "판정불가"
    if lo <= v <= hi:
        return "일치"
    span = max(hi - lo, 1e-9)
    if lo - span * tol <= v <= hi + span * tol:
        return "근접"
    return "불일치"


def validate(R: dict) -> List[Check]:
    def f(v):
        return "미발생" if v is None or (isinstance(v, float) and math.isnan(v)) \
            else f"Day {v:.2f}"
    C = []
    C.append(Check("바이러스 정점 시각", "Day 2 (0.5~1일 급증 후 2일 정점)",
                   "carrat2008", f(R["virus_peak_day"]),
                   _judge(R["virus_peak_day"], 1.5, 2.5)))
    C.append(Check("감염세포 정점 시각", "Day 1~3 (표적세포제한 모델)", "baccam2006",
                   f(R["inf_peak_day"]), _judge(R["inf_peak_day"], 1.0, 3.0)))
    C.append(Check("바이러스 배출 종료(정점 1%)", "Day 4.3~5.3 (평균 4.8일)",
                   "carrat2008", f(R["virus_shed_end_1pct"]),
                   _judge(R["virus_shed_end_1pct"], 4.3, 6.5)))
    C.append(Check("바이러스 완전 제거", "배출 종료 후 수일, 최대 Day 10 내외",
                   "carrat2008", f(R["virus_clear_day"]),
                   _judge(R["virus_clear_day"], 5.0, 12.0)))
    C.append(Check("바이러스 50% 감소", "정점 후 1~2일 내", "carrat2008",
                   f(R["virus_t50"]),
                   _judge((R["virus_t50"] - R["virus_peak_day"])
                          if R["virus_t50"] else None, 0.2, 2.0)))
    C.append(Check("체내 R0", "약 22 (감염세포 1개당 신규 생산감염)", "baccam2006",
                   f"{R['R0_est']:.1f}", _judge(R["R0_est"], 8.0, 40.0)))
    C.append(Check("인터페론 정점", "Day 2 (비강세척액)", "hayden1998",
                   f(R["ifn_peak_day"]), _judge(R["ifn_peak_day"], 1.5, 3.0)))
    C.append(Check("사이토카인 정점", "Day 2 (IL-6/TNF 비강세척액)", "hayden1998",
                   f(R["cyto_peak_day"]), _judge(R["cyto_peak_day"], 1.5, 3.0)))
    C.append(Check("염증/증상 정점", "Day 2~3", "carrat2008",
                   f(R["inflam_peak_day"]), _judge(R["inflam_peak_day"], 1.5, 3.5)))
    C.append(Check("중성구 조직유입 정점", "Day 2~3", "hayden1998",
                   f(R["neut_peak_day"]), _judge(R["neut_peak_day"], 1.5, 3.5)))
    C.append(Check("NK세포 유입 정점", "Day 2~3", "zhang2007",
                   f(R["nk_peak_day"]), _judge(R["nk_peak_day"], 1.5, 3.5)))
    C.append(Check("T세포 프라이밍 완료", "Day 1.5~2.5", "mempel2004",
                   f(R["t_prime"]), _judge(R["t_prime"], 1.2, 3.0)))
    C.append(Check("항원특이 CD8 조직 정점", "Day 8~10", "lawrence2005",
                   f(R["ctl_peak_day"]), _judge(R["ctl_peak_day"], 7.0, 11.0)))
    C.append(Check("형질모세포 정점", "Day 7", "wrammert2008",
                   f(R["pc_peak_day"]), _judge(R["pc_peak_day"], 6.0, 9.0)))
    C.append(Check("IgM 검출", "Day 5~7", "clements1986", f(R["t_igm"]),
                   _judge(R["t_igm"], 4.5, 8.0)))
    C.append(Check("IgG 검출", "Day 10~14", "clements1986", f(R["t_igg"]),
                   _judge(R["t_igg"], 9.0, 15.0)))
    C.append(Check("기억세포 형성비율", "정점 대비 5~10%", "kaech2002",
                   f"{R['memory_ratio']*100:.1f}%",
                   _judge(R["memory_ratio"] * 100, 5.0, 10.0)))
    C.append(Check("CTL 세포당 살상률", "2~16 표적/CTL/일 (모델의존적)",
                   "regoes2007/ganusov2008",
                   f"{R['ctl_kill_per_day']:.2f} 칸/CTL/일",
                   _judge(R["ctl_kill_per_day"], 2.0, 16.0)))
    C.append(Check("표적 상피 소모율", "표적세포제한 모델에서 상당 비율 소모",
                   "baccam2006", f"{R['max_target_depletion']*100:.1f}%",
                   _judge(R["max_target_depletion"] * 100, 20.0, 95.0)))
    C.append(Check("첫 면역세포 감염부위 도착", "혈관외유출 개시 수시간 내 (급성 염증)",
                   "ley2007/nourshargh2010", f(R["t_first_arrival"]),
                   _judge(R["t_first_arrival"], 0.2, 2.0)))
    return C


# =============================================================================
# SECTION 13.  REPORT
# =============================================================================


def print_param_table():
    print()
    print("#" * 130)
    print("  [논문 근거 표]  실제값 -> 논문출처 -> 축소비율 -> 시뮬레이션값  (사양서 ㊱㊲)")
    print("#" * 130)
    hdr = (f"{'항목':<24}{'실제값':>22} {'단위':<20}{'출처':<24}"
           f"{'축소':<14}{'시뮬레이션값'}")
    print(hdr); print("-" * 150)
    for r in PARAM_TABLE:
        src = REFS[r.source].short() if r.source in REFS else r.source
        print(f"{r.item:<24}{r.real:>22} {r.unit:<20}{src:<24}{r.scaling:<14}{r.sim}")
        if r.note:
            print(f"{'':<24}   └ {r.note}")
    print("-" * 150)
    print("  * '모델 가정값' 항목은 논문에서 직접 대응 수치를 찾지 못한 값이며, "
          "실제값처럼 제시하지 않았다 (사양서 ㉕).")
    print()
    print("  [인용 문헌]")
    for k in sorted(REFS):
        print(f"   - {REFS[k].full()}")
    print("  * DOI/PMID 는 작성 시점 기준. 인용 전 PubMed 재확인 권장.")
    print()


def print_final_report(sim, R, checks):
    p = sim.p
    def d(v):
        return "미발생" if v is None or (isinstance(v, float) and math.isnan(v)) \
            else f"Day {v:.2f}"

    print()
    print("#" * 110)
    print("  최 종 결 과   (체계적 이동 조건, 1회 실행)")
    print("#" * 110)
    print("  [병원체]")
    print(f"   · 최대 병원체 수            : {R['virus_peak']:,.0f} agent "
          f"(= {R['virus_peak']*Scale.INV_KAPPA:.2e} 감염단위) @ {d(R['virus_peak_day'])}")
    print(f"   · 감소 시작 시각            : {d(R['virus_peak_day'])}")
    print(f"   · 50% / 90% 감소            : {d(R['virus_t50'])} / {d(R['virus_t90'])}")
    print(f"   · 배출 종료(정점 1%)        : {d(R['virus_shed_end_1pct'])}")
    print(f"   · 완전 제거                 : {d(R['virus_clear_day'])}")
    print(f"   · 최종 병원체 수            : {R['virus_final']:,.0f}")
    print(f"   · 초기 증식률 / 추정 R0     : {R['growth_rate_per_day']:.2f}/일 "
          f"/ {R['R0_est']:.1f}")
    print()
    print("  [감염세포]")
    print(f"   · 최대 감염세포 수          : {R['inf_peak']:,.0f} 칸 @ {d(R['inf_peak_day'])}")
    print(f"   · 50% 감소 / 제거           : {d(R['inf_t50'])} / {d(R['inf_clear_day'])}")
    print(f"   · 누적 감염 발생칸          : {R['total_infections']:,}")
    print(f"   · 표적세포 최대 소모율      : {R['max_target_depletion']*100:.1f}%")
    print()
    print("  [면역세포 도달]")
    print(f"   · 첫 혈관외유출             : {d(R['t_first_extravasation'])}")
    print(f"   · 첫 감염부위 도착          : {d(R['t_first_arrival'])}  ★사양서 40-⑥")
    print(f"   · 누적 혈관외유출 세포      : {R['n_extravasated']:,} agent")
    print(f"   · 구배로 방향 결정된 비율   : {R['grad_frac']*100:.1f}% "
          f"(나머지는 극성 유지, 무작위 0%)")
    print()
    print("  [선천면역]")
    print(f"   · 중성구 정점 / 소모        : {R['neut_peak']:,.0f} @ {d(R['neut_peak_day'])}"
          f"  / 식균제거 {R['cleared_by_phago']:,}")
    print(f"   · 단핵구/대식세포 정점      : {R['mono_peak']:,.0f} @ {d(R['mono_peak_day'])}")
    print(f"   · NK세포 정점 / 살상        : {R['nk_peak']:,.0f} @ {d(R['nk_peak_day'])}"
          f"  / 감염세포 {R['killed_by_nk']:,} 칸")
    print(f"   · 면역세포 총 사망/소모     : {sim.dead_immune:,} agent")
    print(f"   · 케모카인 정점             : {R['chemo_peak']:.4f} @ {d(R['chemo_peak_day'])}")
    print(f"   · 사이토카인 정점           : {R['cyto_peak']:.4f} @ {d(R['cyto_peak_day'])}")
    print(f"   · 인터페론 정점             : {R['ifn_peak']:.4f} @ {d(R['ifn_peak_day'])}")
    print(f"   · 보체 정점                 : {R['comp_peak']*100:.1f}% @ {d(R['comp_peak_day'])}")
    print(f"   · 염증 정점 / 조직손상      : {R['inflam_peak']*100:.1f}% "
          f"@ {d(R['inflam_peak_day'])} / 누적 {R['damage_final']:.3f}")
    print()
    print("  [후천면역]")
    print(f"   · 항원 림프절 도착          : {d(R['t_antigen'])}")
    print(f"   · 조력T/살해T/B 활성화      : {d(R['t_cd4'])} / {d(R['t_cd8'])} / {d(R['t_b'])}")
    print(f"   · 항원특이 CTL 조직 정점    : {R['ctl_peak']:,.0f} @ {d(R['ctl_peak_day'])}")
    print(f"   · 형질세포 정점             : {R['pc_peak']:,.0f} @ {d(R['pc_peak_day'])}")
    print(f"   · 항체 생성 시작(IgM/IgG)   : {d(R['t_igm'])} / {d(R['t_igg'])}")
    print(f"   · 최대 항체 농도            : {R['ab_peak']:.2f} ug/mL @ {d(R['ab_peak_day'])}")
    print(f"   · CTL 감염세포 제거         : {R['killed_by_ctl']:,} 칸")
    print(f"   · 기억세포 형성량           : {R['memory_final']:,.0f} "
          f"(정점의 {R['memory_ratio']*100:.1f}%)")
    print()
    tot = max(1, sim.cleared_virus)
    print("  [병원체 제거 기여도]")
    for name, v in (("비특이 제거", R['cleared_by_decay']),
                    ("식세포 포식", R['cleared_by_phago']),
                    ("보체", R['cleared_by_complement']),
                    ("항체 중화", R['cleared_by_antibody'])):
        print(f"   · {name:<22}: {v:>10,}  ({v/tot*100:5.1f}%)")
    toti = max(1, sim.killed_infected + sim.apoptosis_infected)
    print("  [감염세포 제거 기여도]")
    for name, v in (("NK세포 살상", R['killed_by_nk']),
                    ("살해T세포 살상", R['killed_by_ctl']),
                    ("바이러스 유도 세포자멸", R['apoptosis'])):
        print(f"   · {name:<22}: {v:>10,}  ({v/toti*100:5.1f}%)")
    print()

    print("#" * 110)
    print("  [이론(문헌) 대비 검증표]  - 입력값이 아니라 '창발 결과'만 대조")
    print("#" * 110)
    print(f"{'검증 항목':<26}{'문헌 기준값':<42}{'시뮬레이션':<22}{'판정'}")
    print("-" * 110)
    ok = near = bad = 0
    for ck in checks:
        print(f"{ck.item:<26}{ck.lit:<42}{ck.sim:<22}{ck.verdict}")
        print(f"{'':<26}└ 출처: "
              f"{REFS[ck.ref].short() if ck.ref in REFS else ck.ref}")
        ok += ck.verdict == "일치"
        near += ck.verdict == "근접"
        bad += ck.verdict == "불일치"
    tot_c = ok + near + bad
    print("-" * 110)
    print(f"  종합: 일치 {ok} / 근접 {near} / 불일치 {bad}  "
          f"(판정대상 {tot_c}개)")
    print(f"  ★ 이론 일치율 = {(ok+near)/max(1,tot_c)*100:.1f}% "
          f"(엄격기준 '일치'만 = {ok/max(1,tot_c)*100:.1f}%)")
    print()
    return {"ok": ok, "near": near, "bad": bad, "total": tot_c,
            "rate": (ok + near) / max(1, tot_c) * 100,
            "strict": ok / max(1, tot_c) * 100}


# =============================================================================
# SECTION 14.  COMPARISON WITH RANDOM-WALK RESULT
# =============================================================================


def print_comparison(R: dict, score: dict, rand_path: str):
    if not os.path.exists(rand_path):
        print("  (무작위 이동 결과 파일을 찾을 수 없어 비교를 생략한다)")
        return
    with open(rand_path, encoding="utf-8") as f:
        rnd = json.load(f)["summary"]

    def g(k, default=None):
        v = rnd.get(k, default)
        return v

    def fmt(v, kind="day"):
        if v is None:
            return "미발생"
        if kind == "day":
            return f"Day {v:.2f}"
        if kind == "pct":
            return f"{v*100:.1f}%"
        if kind == "num":
            return f"{v:,.0f}"
        return f"{v:.2f}"

    print("#" * 110)
    print("  [무작위 이동 vs 체계적 이동 비교]")
    print("  ※ 두 실험은 서로 독립적으로 실행되었으며, 상태를 공유하지 않는다.")
    print("     생물학 파라미터(세포수/비율/바이러스동역학/후천면역시간/항체/보체/IFN)는")
    print("     동일 문헌에서 동일하게 산출하였고, '이동 규칙'만 다르다.")
    print("#" * 110)
    rows = [
        ("최대 병원체 수",        fmt(g("virus_peak"), "num"),   fmt(R["virus_peak"], "num"), "낮을수록 우세"),
        ("병원체 정점 시각",      fmt(g("virus_peak_day")),      fmt(R["virus_peak_day"]), "-"),
        ("병원체 50% 감소",       fmt(g("virus_t50")),           fmt(R["virus_t50"]), "빠를수록 우세"),
        ("배출 종료(정점1%)",     fmt(g("virus_shed_end_1pct")), fmt(R["virus_shed_end_1pct"]), "빠를수록 우세"),
        ("병원체 완전 제거",      fmt(g("virus_clear_day")),     fmt(R["virus_clear_day"]), "빠를수록 우세"),
        ("최대 감염세포 수",      fmt(g("inf_peak"), "num"),     fmt(R["inf_peak"], "num"), "낮을수록 우세"),
        ("감염세포 제거 시각",    fmt(g("inf_clear_day")),       fmt(R["inf_clear_day"]), "빠를수록 우세"),
        ("누적 감염 발생칸",      fmt(g("total_infections"), "num"), fmt(R["total_infections"], "num"), "낮을수록 우세"),
        ("표적세포 최대 소모율",  fmt(g("max_target_depletion"), "pct"), fmt(R["max_target_depletion"], "pct"), "낮을수록 우세"),
        ("누적 조직손상",         fmt(g("damage_final"), "raw"), fmt(R["damage_final"], "raw"), "낮을수록 우세"),
        ("염증 정점",             fmt(g("inflam_peak"), "pct"),  fmt(R["inflam_peak"], "pct"), "낮을수록 우세"),
        ("면역세포 총 사망",      fmt(g("cleared_by_phago") and None, "num") if False else "-", "-", "-"),
        ("식세포 포식 제거량",    fmt(g("cleared_by_phago"), "num"), fmt(R["cleared_by_phago"], "num"), "높을수록 탐색기여"),
        ("NK 감염세포 살상",      fmt(g("killed_by_nk"), "num"), fmt(R["killed_by_nk"], "num"), "높을수록 탐색기여"),
        ("CTL 감염세포 살상",     fmt(g("killed_by_ctl"), "num"), fmt(R["killed_by_ctl"], "num"), "높을수록 탐색기여"),
        ("CTL 접촉률(회/CTL/일)", fmt(g("ctl_contact_per_day"), "raw"), fmt(R["ctl_contact_per_day"], "raw"), "높을수록 탐색효율"),
        ("첫 감염부위 도착",      "해당없음(전역배치)",          fmt(R["t_first_arrival"]), "-"),
    ]
    print(f"{'지표':<24}{'무작위 이동':>22}{'체계적 이동':>22}   {'해석'}")
    print("-" * 110)
    for name, a, b, note in rows:
        if a == "-" and b == "-":
            continue
        print(f"{name:<24}{a:>22}{b:>22}   {note}")
    print("-" * 110)
    print()
    return rnd


# =============================================================================
# SECTION 15.  VISUALIZATION
# =============================================================================


def save_figures(sim, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = sim.history
    d = _ser(h, "day")
    fig, ax = plt.subplots(2, 3, figsize=(19, 9))
    fig.suptitle("Influenza A ABM - SYSTEMATIC (chemotaxis-only) immune search",
                 fontweight="bold")

    a = ax[0, 0]
    a.plot(d, _ser(h, "virus"), "r-", lw=1.6, label="free virus")
    a.plot(d, _ser(h, "infected"), "m--", label="infected cells")
    a.plot(d, _ser(h, "healthy"), "g-", alpha=.6, label="healthy epithelium")
    a.set_yscale("log"); a.set_ylim(1, None); a.set_title("Infection dynamics")
    a.set_xlabel("day"); a.grid(alpha=.3); a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(d, _ser(h, "neutrophil"), label="neutrophil")
    a.plot(d, _ser(h, "monocyte"), label="monocyte/macrophage")
    a.plot(d, _ser(h, "nk"), label="NK")
    a.plot(d, _ser(h, "in_tissue"), "k--", alpha=.6, label="cells in tissue")
    a.set_title("Innate cells"); a.set_xlabel("day"); a.grid(alpha=.3); a.legend(fontsize=8)

    a = ax[0, 2]
    a.plot(d, _ser(h, "chemokine"), label="chemokine (mean)")
    a.plot(d, _ser(h, "cytokine"), label="cytokine (mean)")
    a.plot(d, _ser(h, "interferon"), label="interferon (mean)")
    a.set_yscale("log"); a.set_title("Chemical signal fields")
    a.set_xlabel("day"); a.grid(alpha=.3); a.legend(fontsize=8)

    a = ax[1, 0]
    a.plot(d, _ser(h, "killer_t_spec"), "g-", label="specific CTL (tissue)")
    a.plot(d, _ser(h, "helper_t_spec"), "b-", label="specific Th (tissue)")
    a.plot(d, _ser(h, "ln_cd8"), "g--", alpha=.5, label="CD8 in lymph node")
    a.set_yscale("symlog"); a.set_title("Adaptive cellular response")
    a.set_xlabel("day"); a.grid(alpha=.3); a.legend(fontsize=8)

    a = ax[1, 1]
    a.plot(d, _ser(h, "antibody"), "b-", label="antibody (ug/mL)")
    a.plot(d, _ser(h, "complement") * 100, "purple", label="complement (%)")
    a.plot(d, _ser(h, "inflammation") * 100, "r--", label="inflammation (%)")
    a.set_title("Humoral factors"); a.set_xlabel("day"); a.grid(alpha=.3)
    a.legend(fontsize=8)

    a = ax[1, 2]
    im = a.imshow(np.log10(sim.env.chemokine + 1e-6), origin="lower", cmap="inferno")
    a.set_title("Chemokine field at end (log10)")
    fig.colorbar(im, ax=a, fraction=0.046)

    plt.tight_layout()
    path = os.path.join(outdir, "immune_systematic_result.png")
    fig.savefig(path, dpi=125)
    return path


# =============================================================================
# SECTION 16.  MAIN
# =============================================================================


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=Params.max_days)
    ap.add_argument("--seed", type=int, default=Params.seed)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--random-json", default="")
    ap.add_argument("--no-params", action="store_true")
    args = ap.parse_args(argv)

    print("=" * 110)
    print("  면역 시뮬레이션 - 체계적 이동(SYSTEMATIC / CHEMOTAXIS-ONLY) 조건")
    print("  Influenza A / 상기도 상피 + 혈관 네트워크 + 배액 림프절 / kappa = 1/1000")
    print(f"  격자 {Scale.GRID}x{Scale.GRID} = {Scale.SITES:,}칸, "
          f"1 tick = {Scale.DT_MIN:.0f}분, 1일 = {Scale.TICKS_PER_DAY} tick")
    print(f"  이동 규칙: 케모카인 8방향 argmax + 혈류 + 혈관구조. 난수 방향 0%.")
    print(f"  seed = {args.seed}, 시행 횟수 = 1")
    print("=" * 110)

    if not args.no_params:
        print_param_table()

    p = Params(); p.max_days = args.days; p.seed = args.seed
    sim = SystematicImmuneSimulation(p, seed=args.seed, verbose=True)
    print(f"  [환경] 표적 상피 {sim.env.n_target:,}칸, 혈관 {sim.env.n_vessel:,}칸, "
          f"림프절 {(LN_R1-LN_R0)*(LN_C1-LN_C0):,}칸")
    sim.run()

    R = analyze(sim)
    checks = validate(R)
    print_daily_table(sim.daily,
                      "하루 경과별 결과표 (Day 0 = 감염 직후, 마지막 열 = 병원체 소멸 후 최종)")
    score = print_final_report(sim, R, checks)
    print_comparison(R, score, args.random_json)

    try:
        path = save_figures(sim, args.outdir)
        print(f"  [그래프 저장] {path}")
    except Exception as e:
        print(f"  (그래프 저장 실패: {e})")

    with open(os.path.join(args.outdir, "immune_systematic_history.json"),
              "w", encoding="utf-8") as f:
        json.dump({"daily": sim.daily,
                   "summary": {k: (None if (isinstance(v, float) and math.isnan(v))
                                   else v) for k, v in R.items()},
                   "score": score}, f, ensure_ascii=False, indent=1)
    print(f"  [원자료 저장] immune_systematic_history.json")
    print(f"  [실행시간] {sim.wall_time:.1f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
