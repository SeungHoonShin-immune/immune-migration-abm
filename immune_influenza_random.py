#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 immune_random_walk_abm.py
 면역세포 무작위 이동(Random Walk) 기반 감염-면역 반응 Agent-Based Model
--------------------------------------------------------------------------------
 연구 주제 : "면역세포는 무작위로 이동하는 것이 효과적인가,
              체계적으로 이동하는 것이 효과적인가?"
 본 버전   : 무작위 이동(RANDOM WALK) 조건 단독 실험 (1회 실행)
              -> 체계적 이동(systematic) 모델은 MovementEngine 만 교체하여
                 동일 생물학 조건에서 후속 비교 예정

 대표 병원체 : Influenza A virus (H1N1)
 모델 도메인 : 상기도(URT) 상피 조직 슬랩 + 배액 림프절(lymph node) 구획

 축소 원칙  : 실제값 -> 논문 출처 -> 축소비율 -> 시뮬레이션값
              모든 개체(상피세포/면역세포/바이러스)에 동일 축소비 kappa = 1/1000 적용
              논문에서 값을 찾지 못한 항목은 "모델 가정값"으로 명시

 주의       : 본 스크립트에 기재된 DOI/PMID 는 작성 시점 기준이며,
              보고서에 인용하기 전 PubMed 에서 최종 확인 권장.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# =============================================================================
# SECTION 1.  LITERATURE REFERENCE DATABASE  (논문 근거 데이터베이스)
# =============================================================================


@dataclass(frozen=True)
class Ref:
    """논문 1건의 서지 정보"""
    tag: str
    authors: str
    title: str
    journal: str
    year: int
    doi: str = ""
    pmid: str = ""

    def short(self) -> str:
        return f"{self.authors.split(',')[0]} et al., {self.journal} {self.year}"

    def full(self) -> str:
        ident = []
        if self.doi:
            ident.append(f"DOI:{self.doi}")
        if self.pmid:
            ident.append(f"PMID:{self.pmid}")
        return (f"{self.authors}. \"{self.title}\" "
                f"{self.journal} ({self.year}). {'  '.join(ident)}")


REFS: Dict[str, Ref] = {
    "baccam2006":
    Ref("baccam2006",
        "Baccam P, Beauchemin C, Macken CA, Hayden FG, Perelson AS",
        "Kinetics of influenza A virus infection in humans",
        "Journal of Virology 80(15):7590-7599", 2006,
        "10.1128/JVI.01623-05", "16840338"),
    "carrat2008":
    Ref("carrat2008",
        "Carrat F, Vergu E, Ferguson NM, Lemaitre M, Cauchemez S, Leach S, Valleron AJ",
        "Time lines of infection and disease in human influenza: "
        "a review of volunteer challenge studies",
        "American Journal of Epidemiology 167(7):775-785", 2008,
        "10.1093/aje/kwm375", "18230677"),
    "hayden1998":
    Ref("hayden1998",
        "Hayden FG, Fritz R, Lobo MC, Alvord W, Strober W, Straus SE",
        "Local and systemic cytokine responses during experimental human "
        "influenza A virus infection",
        "Journal of Clinical Investigation 101(3):643-649", 1998,
        "10.1172/JCI1355", "9449698"),
    "harris2006":
    Ref("harris2006",
        "Harris A, Cardone G, Winkler DC, Heymann JB, Brecher M, White JM, Steven AC",
        "Influenza virus pleiomorphy characterized by cryoelectron tomography",
        "PNAS 103(50):19123-19127", 2006,
        "10.1073/pnas.0607614103", "17146053"),
    "miller2002":
    Ref("miller2002", "Miller MJ, Wei SH, Parker I, Cahalan MD",
        "Two-photon imaging of lymphocyte motility and antigen response "
        "in intact lymph node", "Science 296(5574):1869-1873", 2002,
        "10.1126/science.1070051", "12016203"),
    "miller2003":
    Ref("miller2003", "Miller MJ, Wei SH, Cahalan MD, Parker I",
        "Autonomous T cell trafficking examined in vivo with intravital "
        "two-photon microscopy", "PNAS 100(5):2604-2609", 2003,
        "10.1073/pnas.2628040100", "12601158"),
    "lammermann2013":
    Ref("lammermann2013",
        "Lammermann T, Afonso PV, Angermann BR, Wang JM, Kastenmuller W, "
        "Parent CA, Germain RN",
        "Neutrophil swarms require LTB4 and integrins at sites of cell death in vivo",
        "Nature 498(7454):371-375", 2013, "10.1038/nature12175", "23708969"),
    "ariotti2012":
    Ref("ariotti2012",
        "Ariotti S, Beltman JB, Chodaczek G, Hoekstra ME, van Beek AE, "
        "Gomez-Eerland R, et al.",
        "Tissue-resident memory CD8+ T cells continuously patrol skin "
        "epithelia to quickly recognize local antigen", "PNAS 109(48):19739-19744",
        2012, "10.1073/pnas.1208927109", "23150545"),
    "mempel2004":
    Ref("mempel2004", "Mempel TR, Henrickson SE, von Andrian UH",
        "T-cell priming by dendritic cells in lymph nodes occurs in three "
        "distinct phases", "Nature 427(6970):154-159", 2004,
        "10.1038/nature02238", "14712275"),
    "lawrence2005":
    Ref("lawrence2005", "Lawrence CW, Ream RM, Braciale TJ",
        "Frequency, specificity, and sites of expansion of CD8+ T cells "
        "during primary pulmonary influenza virus infection",
        "Journal of Immunology 174(9):5332-5340", 2005,
        "10.4049/jimmunol.174.9.5332", "15843530"),
    "wrammert2008":
    Ref("wrammert2008",
        "Wrammert J, Smith K, Miller J, Langley WA, Kokko K, Larsen C, et al.",
        "Rapid cloning of high-affinity human monoclonal antibodies against "
        "influenza virus", "Nature 453(7195):667-671", 2008,
        "10.1038/nature06890", "18449194"),
    "kaech2002":
    Ref("kaech2002", "Kaech SM, Wherry EJ, Ahmed R",
        "Effector and memory T-cell differentiation: implications for vaccine "
        "development", "Nature Reviews Immunology 2(4):251-262", 2002,
        "10.1038/nri778", "11970378"),
    "regoes2007":
    Ref("regoes2007", "Regoes RR, Barber DL, Ahmed R, Antia R",
        "Estimation of the rate of killing by cytotoxic T lymphocytes in vivo",
        "PNAS 104(5):1599-1603", 2007, "10.1073/pnas.0508830104", "17242364"),
    "ganusov2008":
    Ref("ganusov2008", "Ganusov VV, De Boer RJ",
        "Estimating in vivo death rates of targets due to CD8 T-cell-mediated "
        "killing", "Journal of Virology 82(23):11749-11757", 2008,
        "10.1128/JVI.01128-08", "18815293"),
    "alanio2010":
    Ref("alanio2010", "Alanio C, Lemaitre F, Law HKW, Hasan M, Albert ML",
        "Enumeration of human antigen-specific naive CD8+ T cells reveals "
        "conserved precursor frequencies", "Blood 115(18):3718-3725", 2010,
        "10.1182/blood-2009-10-251124", "20220118"),
    "blattman2002":
    Ref("blattman2002",
        "Blattman JN, Antia R, Sourdive DJD, Wang X, Kaech SM, Murali-Krishna K, "
        "Altman JD, Ahmed R",
        "Estimating the precursor frequency of naive antigen-specific CD8 T cells",
        "Journal of Experimental Medicine 195(5):657-664", 2002,
        "10.1084/jem.20001021", "11877489"),
    "bisset2004":
    Ref("bisset2004", "Bisset LR, Lung TL, Kaelin M, Ludwig E, Dubs RW",
        "Reference values for peripheral blood lymphocyte phenotypes "
        "applicable to the healthy adult population in Switzerland",
        "European Journal of Haematology 72(3):203-212", 2004,
        "10.1046/j.0902-4441.2003.00199.x", "14962239"),
    "pillay2010":
    Ref("pillay2010",
        "Pillay J, den Braber I, Vrisekoop N, Kwast LM, de Boer RJ, Borghans JAM, "
        "Tesselaar K, Koenderman L",
        "In vivo labeling with 2H2O reveals a human neutrophil lifespan of 5.4 days",
        "Blood 116(4):625-627", 2010, "10.1182/blood-2010-01-259028", "20410504"),
    "patel2017":
    Ref("patel2017",
        "Patel AA, Zhang Y, Fullerton JN, Boelen L, Rongvaux A, Maini AA, et al.",
        "The fate and lifespan of human monocyte subsets in steady state and "
        "systemic inflammation", "Journal of Experimental Medicine 214(7):1913-1923",
        2017, "10.1084/jem.20170355", "28606987"),
    "zhang2007":
    Ref("zhang2007",
        "Zhang Y, Wallace DL, de Lara CM, Ghattas H, Asquith B, Worth A, et al.",
        "In vivo kinetics of human natural killer cells: the effects of ageing "
        "and acute and chronic viral infection", "Immunology 121(2):258-265", 2007,
        "10.1111/j.1365-2567.2007.02573.x", "17346281"),
    "mohler2005":
    Ref("mohler2005", "Mohler L, Flockerzi D, Sann H, Reichl U",
        "Mathematical model of influenza A virus production in large-scale "
        "microcarrier culture", "Biotechnology and Bioengineering 90(1):46-58",
        2005, "10.1002/bit.20363", "15736163"),
    "memoli2015":
    Ref("memoli2015",
        "Memoli MJ, Czajkowski L, Reed S, Athota R, Bristol T, Proudfoot K, et al.",
        "Validation of the wild-type influenza A human challenge model H1N1pdMIST",
        "Clinical Infectious Diseases 60(5):693-702", 2015,
        "10.1093/cid/ciu924", "25416753"),
    "clements1986":
    Ref("clements1986", "Clements ML, Betts RF, Tierney EL, Murphy BR",
        "Serum and nasal wash antibodies associated with resistance to "
        "experimental challenge with influenza A wild-type virus",
        "Journal of Clinical Microbiology 24(1):157-160", 1986,
        "10.1128/jcm.24.1.157-160.1986", "3722363"),
    "ivashkiv2014":
    Ref("ivashkiv2014", "Ivashkiv LB, Donlin LT",
        "Regulation of type I interferon responses",
        "Nature Reviews Immunology 14(1):36-49", 2014, "10.1038/nri3581",
        "24362405"),
    "olmsted2001":
    Ref("olmsted2001", "Olmsted SS, Padgett JL, Yudin AI, Whaley KJ, Moench TR, Cone RA",
        "Diffusion of macromolecules and virus-like particles in human cervical mucus",
        "Biophysical Journal 81(4):1930-1937", 2001,
        "10.1016/S0006-3495(01)75844-4", "11566767"),
    "bhat2007":
    Ref("bhat2007", "Bhat R, Watzl C",
        "Serial killing of tumor cells by human natural killer cells - "
        "enhancement by therapeutic antibodies", "PLoS ONE 2(3):e326", 2007,
        "10.1371/journal.pone.0000326", "17389917"),
    "janeway":
    Ref("janeway", "Murphy K, Weaver C", "Janeway's Immunobiology, 9th ed.",
        "Garland Science (textbook)", 2016),
    "dacie":
    Ref("dacie", "Bain BJ, Bates I, Laffan MA",
        "Dacie and Lewis Practical Haematology, 12th ed. (reference intervals)",
        "Elsevier (textbook)", 2017),
    "textbook_hist":
    Ref("textbook_hist", "Mescher AL",
        "Junqueira's Basic Histology (respiratory epithelium cell dimensions)",
        "McGraw-Hill (textbook)", 2018),
}

MODEL_ASSUMPTION = "모델 가정값"


# =============================================================================
# SECTION 2.  SCALING  (축소비율 정의)
# =============================================================================


class Scale:
    """
    공간/시간/개체수 축소 규약.

    [공간]
      1,000 x 1,000 = 1,000,000 격자칸.
      1칸 = 15 um (호흡기 상피세포 직경 10~20 um 의 중앙값, textbook_hist)
      -> 도메인 = 15 mm x 15 mm 상피 조직 슬랩

    [개체]
      kappa = 1/1000 을 상피세포, 면역세포, 바이러스에 공통 적용.
      즉 격자 1칸(상피) = 실제 상피세포 1,000개
         면역세포 agent 1개 = 실제 백혈구 1,000개
         바이러스 agent 1개 = 실제 감염성 바이러스 1,000 TCID50

    [시간]
      1 tick = 6분 = 0.1시간  -> 1일 = 240 tick
      모든 생물학적 속도상수는 (1/day 또는 1/hour) -> tick 단위로 동일 환산.
    """

    GRID = 1000
    SITES = GRID * GRID                     # 1,000,000 칸
    SITE_UM = 15.0                          # 칸 1개의 한 변 길이 (um)
    DOMAIN_MM = GRID * SITE_UM / 1000.0     # 15 mm

    KAPPA = 1.0 / 1000.0                    # 개체 축소비율
    INV_KAPPA = 1000.0

    DT_MIN = 6.0                            # 1 tick = 6 분
    DT_HOUR = DT_MIN / 60.0                 # 0.1 시간
    TICKS_PER_HOUR = 10
    TICKS_PER_DAY = 240

    # 조직 구성: 20행 주기로 8행은 상피(target), 12행은 간질/혈관/림프관
    #  -> 상피 비율 40%  => 400,000 칸 = 실제 4.0e8 상피세포 (Baccam 2006 T0)
    EPI_PERIOD = 20
    EPI_ROWS = 8
    EPI_FRACTION = EPI_ROWS / EPI_PERIOD    # 0.40
    N_TARGET_SITES = int(SITES * EPI_FRACTION)   # 400,000

    # 인터페론/보체 필드 해상도 (5x5 칸 = 75 um 단위)
    FIELD_BIN = 5
    FIELD_N = GRID // FIELD_BIN             # 200 x 200

    @staticmethod
    def per_tick_from_per_day(rate_per_day: float) -> float:
        """1/day 속도상수 -> tick 당 사건확률 (지수분포)"""
        return 1.0 - math.exp(-rate_per_day / Scale.TICKS_PER_DAY)

    @staticmethod
    def per_tick_from_halflife_h(halflife_hours: float) -> float:
        """반감기(시간) -> tick 당 소멸확률"""
        k = math.log(2.0) / halflife_hours          # 1/hour
        return 1.0 - math.exp(-k * Scale.DT_HOUR)

    @staticmethod
    def motility_to_sigma_sites(motility_um2_per_min: float) -> float:
        """
        운동성계수 M (um^2/min) -> 1 tick 당 축(axis)별 변위 표준편차 (격자칸 단위)

        2차원 무작위보행에서 <r^2> = 4*M*t 이므로 축별 분산 = 2*M*t.
        sigma_um = sqrt(2 * M * dt),  sigma_sites = sigma_um / SITE_UM
        """
        sigma_um = math.sqrt(2.0 * motility_um2_per_min * Scale.DT_MIN)
        return sigma_um / Scale.SITE_UM

    @staticmethod
    def speed_to_motility(speed_um_per_min: float,
                          persistence_min: float = 1.0) -> float:
        """
        순간속도 v(um/min)와 방향지속시간 tau(min) -> 운동성계수 M = v^2*tau/4 (2D)
        * 논문은 보통 '속도'를 보고하므로, 무작위보행으로 옮길 때 반드시
          이 변환을 거친다 (사양서 23항: 이동방향=무작위, 이동속도=논문값 축소).
        """
        return speed_um_per_min ** 2 * persistence_min / 4.0


# =============================================================================
# SECTION 3.  PARAMETER TABLE  (논문 근거 표 / 사양서 34항)
# =============================================================================


@dataclass(frozen=True)
class ParamRow:
    item: str            # 항목
    real: str            # 실제값
    unit: str            # 단위
    source: str          # 논문 출처(REFS key) 또는 MODEL_ASSUMPTION
    scaling: str         # 축소 방식
    sim: str             # 시뮬레이션값
    note: str = ""       # 선택 이유 / 비고


def _M(v: float) -> str:
    return f"{v:,.0f}"


# ---- 실제값 상수 (논문값) ---------------------------------------------------
REAL = {
    # 혈액 / 면역세포 (Dacie&Lewis, Bisset 2004)
    "wbc_per_ml": 5.0e6,          # 백혈구 4.0~11.0 x10^9/L 의 중앙 부근
    "neut_frac": 0.60,            # 호중구 40~70%
    "mono_frac": 0.06,            # 단핵구 2~10%
    "lymph_frac": 0.30,           # 림프구 20~45%
    "cd4_of_lymph": 0.45,
    "cd8_of_lymph": 0.25,
    "bcell_of_lymph": 0.12,
    "nk_of_lymph": 0.13,

    # 바이러스 (Baccam 2006, Harris 2006, Mohler 2005)
    "virion_nm": 100.0,           # 80~120 nm
    "eclipse_h": 6.0,             # Baccam 2006: ~6 h
    "productive_h": 5.0,          # Baccam 2006: ~5 h 생산
    "infected_lifetime_h": 11.0,  # Baccam 2006: 평균 11 h
    "virus_halflife_h": 3.0,      # Baccam 2006: 유리 바이러스 반감기 ~3 h
    "R0_within_host": 22.0,       # Baccam 2006: 1개 감염세포 -> ~22 신규감염
    "burst_particles": 5000.0,    # 10^3~10^4 particles/cell (Mohler 2005)
    "target_cells_urt": 4.0e8,    # Baccam 2006 T0
    "inoculum_tcid50": 1.0e5,     # 인체감염모델 접종량 10^4~10^6 (Memoli 2015)
    "virion_D_um2_s": 1.0,        # 점액 내 확산 (Olmsted 2001 계열, 하한 채택)

    # 세포 운동성 (논문 '속도' 값)
    "neut_speed": 15.0,           # 10~20 um/min (Lammermann 2013)
    "mono_speed": 2.5,            # 1~4 um/min
    "nk_speed": 8.0,              # 6~10 um/min
    "naiveT_speed": 11.0,         # Miller 2002/2003
    "effT_speed": 5.0,            # Ariotti 2012 (조직 내 effector/TRM)
    "b_speed": 6.0,               # Miller 2002 계열

    # 수명
    "neut_lifespan_d": 1.5,       # 혈중 t1/2 7~19h, 조직 1~2일 (Pillay 2010 논쟁적)
    "mono_lifespan_d": 3.0,       # Patel 2017 classical monocyte ~1.6d, 조직 연장
    "nk_lifespan_d": 14.0,        # Zhang 2007
    "naive_lymph_lifespan_d": 180.0,
    "effector_T_halflife_d": 1.5,  # 수축기 반감기

    # 후천면역 타임라인
    "dc_migration_h": 18.0,       # DC 이주 12~24h (Lawrence 2005 계열)
    "priming_h": 24.0,            # 3-phase priming ~24h (Mempel 2004)
    "cd8_doubling_h": 7.0,        # in vivo 6~8 h
    "cd4_doubling_h": 9.0,
    "b_doubling_h": 10.0,
    "cd8_max_divisions": 11.0,
    "expansion_program_d": 6.0,   # 항원 노출 후 자율 증식 프로그램 (van Stipdonk 2001 / Kaech 2002)
    "net_expansion_factor": 0.53,  # 분열 6~8h + 확장기 사멸 -> 순 배가시간 ~13h
    "cd4_max_divisions": 9.0,
    "b_max_divisions": 10.0,
    "contraction_rate_d": 0.75,    # 정점 후 90~95% 소실 (Kaech 2002)
    "plasmablast_halflife_d": 3.5, # 여포외 형질모세포 단명 (3~5일)
    "gc_pc_halflife_d": 7.0,       # GC 유래 형질세포 (골수 이주 전)
    "eff_t_halflife_expand_d": 4.0,  # 확장기 조직 효과기 반감기 (모델 가정값)
    "cd8_peak_day": 9.0,          # 8~10일 (Lawrence 2005)
    "plasmablast_peak_day": 7.0,  # Wrammert 2008
    "memory_fraction": 0.075,     # 정점의 5~10% (Kaech 2002)
    "igm_halflife_d": 5.0,        # Janeway
    "igg_halflife_d": 21.0,       # Janeway
    "igm_detect_day": 6.0,        # 5~7일 (Clements 1986)
    "igg_detect_day": 12.0,       # 10~14일 (Clements 1986)

    # 전구세포 빈도 (Alanio 2010, Blattman 2002)
    "cd8_precursor_systemic": 5.0e4,
    "cd4_precursor_systemic": 1.0e5,
    "b_precursor_systemic": 5.0e4,

    # 자연면역
    "ifn_peak_day": 2.0,          # Hayden 1998
    "antiviral_state_h": 5.0,     # 4~6h (Ivashkiv 2014)
    "c3_mg_per_ml": 1.2,          # 보체 C3 혈장농도 0.9~1.8 mg/mL
    "symptom_peak_day": 3.0,      # Carrat 2008 (전신증상 2일, 총증상 3일)
    "shedding_days": 4.8,         # Carrat 2008 (95%CI 4.31~5.29)
}


PARAM_TABLE: List[ParamRow] = [
    ParamRow("혈액/조직 기준 부피", "1", "mL 상당", "본 모델 정의", "기준",
             f"{_M(Scale.SITES)}칸 (15mm x 15mm 상피 슬랩)",
             "사양서 2항: 100만 칸 = 축소 기준 공간"),
    ParamRow("격자 1칸 크기", "10~20", "um (상피세포 직경)", "textbook_hist",
             "공간축소", f"{Scale.SITE_UM:.0f} um/칸", "중앙값 채택"),
    ParamRow("표적 상피세포 총수", f"{REAL['target_cells_urt']:.1e}", "cells",
             "baccam2006", "kappa=1/1000",
             f"{_M(Scale.N_TARGET_SITES)}칸 (40% 격자)",
             "Baccam 2006 T0 = 4x10^8 을 1/1000 축소"),
    ParamRow("백혈구 총수", f"{REAL['wbc_per_ml']:.1e}", "cells/mL", "dacie",
             "kappa=1/1000", "5,000 agent", "4.0~11.0 x10^9/L 중앙 부근"),
    ParamRow("중성구", f"{REAL['wbc_per_ml']*REAL['neut_frac']:.1e}", "cells/mL",
             "dacie", "kappa=1/1000", "3,000 agent", "백혈구의 40~70%"),
    ParamRow("단핵구/대식세포", f"{REAL['wbc_per_ml']*REAL['mono_frac']:.1e}",
             "cells/mL", "dacie", "kappa=1/1000", "300 agent", "백혈구의 2~10%"),
    ParamRow("NK세포",
             f"{REAL['wbc_per_ml']*REAL['lymph_frac']*REAL['nk_of_lymph']:.1e}",
             "cells/mL", "bisset2004", "kappa=1/1000", "195 agent",
             "림프구의 5~20%"),
    ParamRow("조력 T세포(CD4)",
             f"{REAL['wbc_per_ml']*REAL['lymph_frac']*REAL['cd4_of_lymph']:.1e}",
             "cells/mL", "bisset2004", "kappa=1/1000", "675 agent", ""),
    ParamRow("살해 T세포(CD8)",
             f"{REAL['wbc_per_ml']*REAL['lymph_frac']*REAL['cd8_of_lymph']:.1e}",
             "cells/mL", "bisset2004", "kappa=1/1000", "375 agent", ""),
    ParamRow("B세포",
             f"{REAL['wbc_per_ml']*REAL['lymph_frac']*REAL['bcell_of_lymph']:.1e}",
             "cells/mL", "bisset2004", "kappa=1/1000", "180 agent", ""),
    ParamRow("초기 병원체(접종량)", f"{REAL['inoculum_tcid50']:.1e}", "TCID50",
             "memoli2015", "kappa=1/1000", "100 agent",
             "인체감염모델 접종량 10^4~10^6 중 10^5 채택"),
    ParamRow("바이러스 크기", "80~120", "nm", "harris2006", "공간축소",
             "점(무부피) 입자, 1칸=15um 대비 무시 가능",
             "세포/공간이 바이러스보다 훨씬 크다는 관계 유지"),
    ParamRow("바이러스 확산계수", f"{REAL['virion_D_um2_s']}", "um^2/s",
             "olmsted2001", "시간/공간축소",
             f"sigma={Scale.motility_to_sigma_sites(REAL['virion_D_um2_s']*60):.2f}칸/tick",
             "점액 결합성으로 자유확산보다 느림, 하한값 채택"),
    ParamRow("바이러스 잠복기(eclipse)", f"{REAL['eclipse_h']}", "시간",
             "baccam2006", "시간축소",
             f"{int(REAL['eclipse_h']*Scale.TICKS_PER_HOUR)} tick", ""),
    ParamRow("감염세포 생산기간", f"{REAL['productive_h']}", "시간", "baccam2006",
             "시간축소", f"{int(REAL['productive_h']*Scale.TICKS_PER_HOUR)} tick",
             "평균 감염세포 수명 11h = 6h + 5h"),
    ParamRow("유리 바이러스 반감기", f"{REAL['virus_halflife_h']}", "시간",
             "baccam2006", "시간축소",
             f"{Scale.per_tick_from_halflife_h(REAL['virus_halflife_h']):.4f}/tick",
             "비특이 제거(점액섬모 등)"),
    ParamRow("체내 기초재생산수 R0", f"{REAL['R0_within_host']}", "-",
             "baccam2006", "동일", "burst/감염확률 보정으로 재현",
             "감염세포 1개당 신규 생산감염 ~22"),
    ParamRow("감염세포당 바이러스 생산", "10^3~10^4", "particles/cell",
             "mohler2005", "kappa + 감염성비율",
             "25 agent/site (site=1000세포)",
             "입자:감염단위 비 ~100:1 반영, R0=22 정합되도록 보정"),
    ParamRow("중성구 이동속도", f"{REAL['neut_speed']}", "um/min", "lammermann2013",
             "시간/공간축소",
             f"M={Scale.speed_to_motility(REAL['neut_speed']):.0f} um^2/min "
             f"-> sigma={Scale.motility_to_sigma_sites(Scale.speed_to_motility(REAL['neut_speed'])):.2f}칸/tick",
             "속도->운동성계수 변환 후 무작위보행 적용"),
    ParamRow("단핵구/대식세포 이동속도", f"{REAL['mono_speed']}", "um/min",
             "lammermann2013", "시간/공간축소",
             f"sigma={Scale.motility_to_sigma_sites(Scale.speed_to_motility(REAL['mono_speed'])):.2f}칸/tick",
             "1~4 um/min 범위 중앙"),
    ParamRow("NK세포 이동속도", f"{REAL['nk_speed']}", "um/min", "bhat2007",
             "시간/공간축소",
             f"sigma={Scale.motility_to_sigma_sites(Scale.speed_to_motility(REAL['nk_speed'])):.2f}칸/tick",
             ""),
    ParamRow("나이브 T세포 이동속도", f"{REAL['naiveT_speed']}", "um/min",
             "miller2002", "시간/공간축소",
             f"M~{Scale.speed_to_motility(REAL['naiveT_speed']):.0f} um^2/min",
             "림프절 내 2광자 관측 (M=50~70 보고와 정합)"),
    ParamRow("효과기 CD8 조직내 이동속도", f"{REAL['effT_speed']}", "um/min",
             "ariotti2012", "시간/공간축소",
             f"sigma={Scale.motility_to_sigma_sites(Scale.speed_to_motility(REAL['effT_speed'])):.2f}칸/tick",
             "조직 내에서는 림프절보다 느림"),
    ParamRow("B세포 이동속도", f"{REAL['b_speed']}", "um/min", "miller2002",
             "시간/공간축소",
             f"sigma={Scale.motility_to_sigma_sites(Scale.speed_to_motility(REAL['b_speed'])):.2f}칸/tick",
             ""),
    ParamRow("중성구 수명", f"{REAL['neut_lifespan_d']}", "일", "pillay2010",
             "시간축소", f"{int(REAL['neut_lifespan_d']*Scale.TICKS_PER_DAY)} tick",
             "혈중 t1/2 7~19h vs 5.4일 보고 상충, 조직 1~2일 채택"),
    ParamRow("단핵구 수명", f"{REAL['mono_lifespan_d']}", "일", "patel2017",
             "시간축소", f"{int(REAL['mono_lifespan_d']*Scale.TICKS_PER_DAY)} tick",
             "classical monocyte ~1.6일 + 조직 대식세포 전환 연장"),
    ParamRow("NK세포 수명", f"{REAL['nk_lifespan_d']}", "일", "zhang2007",
             "시간축소", f"{int(REAL['nk_lifespan_d']*Scale.TICKS_PER_DAY)} tick", ""),
    ParamRow("DC 항원수송 지연", f"{REAL['dc_migration_h']}", "시간", "lawrence2005",
             "시간축소", f"{int(REAL['dc_migration_h']*Scale.TICKS_PER_HOUR)} tick",
             "12~24h 범위 중앙"),
    ParamRow("T세포 프라이밍 시간", f"{REAL['priming_h']}", "시간", "mempel2004",
             "시간축소", f"{int(REAL['priming_h']*Scale.TICKS_PER_HOUR)} tick",
             "3단계 priming 후 첫 분열까지 ~24h"),
    ParamRow("CD8 분열주기", f"{REAL['cd8_doubling_h']}", "시간", "lawrence2005",
             "시간축소", f"{int(REAL['cd8_doubling_h']*Scale.TICKS_PER_HOUR)} tick",
             "in vivo 6~8h"),
    ParamRow("CD4 분열주기", f"{REAL['cd4_doubling_h']}", "시간", "mempel2004",
             "시간축소", f"{int(REAL['cd4_doubling_h']*Scale.TICKS_PER_HOUR)} tick", ""),
    ParamRow("B세포 분열주기", f"{REAL['b_doubling_h']}", "시간", "wrammert2008",
             "시간축소", f"{int(REAL['b_doubling_h']*Scale.TICKS_PER_HOUR)} tick", ""),
    ParamRow("CD8 나이브 전구세포수",
             f"{REAL['cd8_precursor_systemic']:.1e}", "cells(전신)", "alanio2010",
             "kappa=1/1000", "50 agent",
             "인체 나이브 CD8 특이빈도 ~1/10^6 (전신 풀 기준). "
             "1 mL 혈액만으로는 1개 미만이므로 전신 풀에서 산출"),
    ParamRow("CD4 나이브 전구세포수",
             f"{REAL['cd4_precursor_systemic']:.1e}", "cells(전신)", "blattman2002",
             "kappa=1/1000", "100 agent", ""),
    ParamRow("B 나이브 전구세포수",
             f"{REAL['b_precursor_systemic']:.1e}", "cells(전신)", "blattman2002",
             "kappa=1/1000", "50 agent", ""),
    ParamRow("항바이러스 상태 확립시간", f"{REAL['antiviral_state_h']}", "시간",
             "ivashkiv2014", "시간축소",
             f"{int(REAL['antiviral_state_h']*Scale.TICKS_PER_HOUR)} tick",
             "IFN 노출 후 4~6h"),
    ParamRow("보체 C3 혈장농도", f"{REAL['c3_mg_per_ml']}", "mg/mL", "janeway",
             "정규화", "활성도 0~1 스케일",
             "절대농도 대신 활성분율로 표현"),
    ParamRow("IgM 반감기", f"{REAL['igm_halflife_d']}", "일", "janeway", "시간축소",
             f"{Scale.per_tick_from_halflife_h(REAL['igm_halflife_d']*24):.5f}/tick", ""),
    ParamRow("IgG 반감기", f"{REAL['igg_halflife_d']}", "일", "janeway", "시간축소",
             f"{Scale.per_tick_from_halflife_h(REAL['igg_halflife_d']*24):.5f}/tick", ""),
    ParamRow("기억세포 형성비율", "5~10", "% (정점 대비)", "kaech2002", "동일",
             f"{REAL['memory_fraction']*100:.1f}%", "수축기 후 잔존"),
    # ---- 모델 가정값 ----
    ParamRow("상피/간질 배치", "-", "-", MODEL_ASSUMPTION, "-",
             "20행 주기 중 8행 상피(40%)",
             "실제 점막의 상피-고유판 층상구조를 2D로 단순화. "
             "논문에 대응 수치 없음"),
    ParamRow("식세포 접촉 반경", "-", "-", MODEL_ASSUMPTION, "-", "동일 격자칸(15um)",
             "중성구 직경 ~12um, 격자 1칸과 유사하여 '동일칸=접촉'으로 처리"),
    ParamRow("중성구 1개 식균 용량", "-", "-", MODEL_ASSUMPTION, "-", "10회",
             "식균 후 기능소진. 논문 정량치 부재하여 가정"),
    ParamRow("NK 접촉당 사멸확률", "-", "-", MODEL_ASSUMPTION, "-", "0.25/tick",
             "NK-표적 접합 20~60분(bhat2007) 을 6분 tick 확률로 환산한 근사"),
    ParamRow("CTL 접촉당 사멸확률", "-", "-", MODEL_ASSUMPTION, "-", "0.80/tick",
             "agent 1개=CTL 1,000개 이므로 접촉 시 해당 칸 제거 가능. "
             "결과의 세포당 살상률을 regoes2007/ganusov2008 범위와 대조 검증"),
    ParamRow("염증-동원 결합계수", "-", "-", MODEL_ASSUMPTION, "-",
             "중성구 최대 4배, 단핵구 10배, NK 6배",
             "감염 조직 내 유입 배수 관측범위를 반영한 가정"),
    ParamRow("인터페론 확산/감쇄", "-", "-", MODEL_ASSUMPTION, "-",
             "5점 스텐실 확산 + 반감기 2h",
             "조직 내 IFN 농도장 논문값 부재"),
    ParamRow("상피 재생속도", "-", "-", MODEL_ASSUMPTION, "-", "0.05/day",
             "인플루엔자 후 상피 복구는 수일~수주. 감염 재점화 방지 위해 저속 설정"),
]


# =============================================================================
# SECTION 4.  SIMULATION PARAMETERS  (실제값 -> tick/칸 단위로 환산)
# =============================================================================


@dataclass
class Params:
    # ---- 초기 개체수 (kappa=1/1000) ------------------------------------------
    n_virus0: int = 100
    n_neutrophil: int = 3000
    n_monocyte: int = 300
    n_nk: int = 195
    n_helper_t: int = 675
    n_killer_t: int = 375
    n_bcell: int = 180

    # ---- 항원특이 전구세포 (전신 풀 기준) -------------------------------------
    precursor_cd8: float = 50.0
    precursor_cd4: float = 100.0
    precursor_b: float = 50.0

    # ---- 바이러스 동역학 ------------------------------------------------------
    eclipse_ticks: int = int(REAL["eclipse_h"] * Scale.TICKS_PER_HOUR)        # 60
    productive_ticks: int = int(REAL["productive_h"] * Scale.TICKS_PER_HOUR)  # 50
    virus_decay_p: float = Scale.per_tick_from_halflife_h(REAL["virus_halflife_h"])
    burst_agents_per_site: float = 25.0
    p_infect: float = 0.15          # 감수성 칸 위 virion agent 의 tick 당 감염확률
    virus_sigma: float = Scale.motility_to_sigma_sites(REAL["virion_D_um2_s"] * 60.0)
    # 점액섬모 수송에 의한 광역 이동. 비강 점액섬모 청소율 4~10 mm/min 의
    # 1/100 수준(0.05 mm/min)만 '무방향 분산'으로 반영한 보수적 설정.
    # 방향성 없는 등방 분산이므로 표적 추적과 무관하다. (모델 가정값)
    virus_sigma_mucus: float = 30.0

    # ---- 인터페론 -------------------------------------------------------------
    ifn_production: float = 0.02        # 생산 감염칸 1개가 tick 당 방출 (정규화)
    ifn_decay_p: float = Scale.per_tick_from_halflife_h(2.0)
    ifn_diffuse: float = 0.18           # 5점 스텐실 계수
    antiviral_rate: float = 1.0 / (REAL["antiviral_state_h"] * Scale.TICKS_PER_HOUR)
    antiviral_decay: float = 1.0 / (24.0 * Scale.TICKS_PER_HOUR)
    ifn_block_release: float = 0.80     # 항바이러스 상태 최대 생산억제율

    # ---- 보체 -----------------------------------------------------------------
    complement_on: float = 0.06         # 활성화 속도상수 (tick^-1)
    complement_off: float = 0.02        # 불활성화
    complement_lysis: float = 0.010     # 활성도 1.0 일 때 추가 virion 제거율/tick
    complement_opsonin: float = 0.35    # 식균확률 가산 최대치

    # ---- 항체 -----------------------------------------------------------------
    ab_per_pc_igm: float = 5.0e-6       # 단기 형질세포 agent 당 IgM 생산 (ug/mL/tick)
    ab_per_pc_igg: float = 2.5e-5       # GC 유래 형질세포 agent 당 IgG 생산
    igm_decay_p: float = Scale.per_tick_from_halflife_h(REAL["igm_halflife_d"] * 24)
    igg_decay_p: float = Scale.per_tick_from_halflife_h(REAL["igg_halflife_d"] * 24)
    ab_kd: float = 2.0                  # 중화 반포화 농도 (ug/mL)
    ab_neutralize_max: float = 0.030    # 최대 중화율/tick
    ab_opsonin: float = 0.35            # 식균확률 가산 최대치
    ab_block_max: float = 0.95          # 중화항체의 최대 감염차단율
    ab_block_kd: float = 3.0            # 감염차단 반포화 농도 (ug/mL)
    ab_detect_threshold: float = 0.5    # 검출한계 (ug/mL) - 모델 가정값

    # ---- 선천 면역세포 기능 ----------------------------------------------------
    neut_phago_p: float = 0.45          # 접촉당 식균 성공확률 (기본)
    neut_capacity: int = 10
    mono_phago_p: float = 0.55
    mono_capacity: int = 25
    nk_kill_p: float = 0.25
    ctl_kill_p: float = 0.80

    # ---- 이동 (운동성계수 -> sigma 칸/tick) -------------------------------------
    sigma_neut: float = Scale.motility_to_sigma_sites(
        Scale.speed_to_motility(REAL["neut_speed"]))
    sigma_mono: float = Scale.motility_to_sigma_sites(
        Scale.speed_to_motility(REAL["mono_speed"]))
    sigma_nk: float = Scale.motility_to_sigma_sites(
        Scale.speed_to_motility(REAL["nk_speed"]))
    sigma_naive_t: float = Scale.motility_to_sigma_sites(
        Scale.speed_to_motility(REAL["naiveT_speed"]))
    sigma_eff_t: float = Scale.motility_to_sigma_sites(
        Scale.speed_to_motility(REAL["effT_speed"]))
    sigma_b: float = Scale.motility_to_sigma_sites(
        Scale.speed_to_motility(REAL["b_speed"]))

    # ---- 수명 (tick) -----------------------------------------------------------
    life_neut: int = int(REAL["neut_lifespan_d"] * Scale.TICKS_PER_DAY)
    life_mono: int = int(REAL["mono_lifespan_d"] * Scale.TICKS_PER_DAY)
    life_nk: int = int(REAL["nk_lifespan_d"] * Scale.TICKS_PER_DAY)
    life_naive: int = int(REAL["naive_lymph_lifespan_d"] * Scale.TICKS_PER_DAY)
    eff_t_death_expand: float = Scale.per_tick_from_halflife_h(
        REAL["eff_t_halflife_expand_d"] * 24)      # 확장기 (생존신호 존재)
    eff_t_death_contract: float = Scale.per_tick_from_halflife_h(
        REAL["effector_T_halflife_d"] * 24)        # 수축기

    # ---- 염증 / 동원 -----------------------------------------------------------
    inflam_gain_virus: float = 1.0 / 8000.0
    inflam_gain_infected: float = 1.0 / 40000.0
    inflam_decay: float = 0.010
    recruit_neut_max: float = 3.0       # 기저 대비 추가 배수
    recruit_mono_max: float = 9.0
    recruit_nk_max: float = 5.0
    recruit_rate: float = 0.020         # tick 당 목표치 접근속도
    damage_rate: float = 0.0015         # 염증 1.0 에서 tick 당 조직손상 누적
    epi_regen_p: float = Scale.per_tick_from_per_day(0.02)

    # ---- 림프절 / 후천면역 ------------------------------------------------------
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
    # 형질세포 분화율 (tick 당 활성 B세포 중 분화 분율) - 모델 가정값
    ef_diff_rate: float = 0.0016
    gc_diff_rate: float = 3.0e-5
    pc_short_decay: float = Scale.per_tick_from_per_day(
        math.log(2.0) / REAL["plasmablast_halflife_d"])
    pc_gc_decay: float = Scale.per_tick_from_per_day(
        math.log(2.0) / REAL["gc_pc_halflife_d"])
    gc_end_ticks: int = int(21.0 * Scale.TICKS_PER_DAY)
    memory_b_fraction: float = 0.10
    emigration_p: float = 1.6e-4         # 림프절 -> 조직 이주율/tick
    emigration_start_h: float = 96.0    # 조직 진입 개시 (~day 4, Lawrence 2005)
    antigen_threshold: float = 20.0     # 프라이밍 개시 항원량 (정규화)
    antigen_decay: float = Scale.per_tick_from_per_day(1.5)
    ef_delay_ticks: int = int(48.0 * Scale.TICKS_PER_HOUR)   # 여포외 형질모세포 ~활성화+2일
    gc_delay_ticks: int = int(144.0 * Scale.TICKS_PER_HOUR)  # GC 유래 IgG ~활성화+6일
    memory_fraction: float = REAL["memory_fraction"]

    # ---- 실행 제어 -------------------------------------------------------------
    max_days: int = 30
    stop_after_clear_days: int = 4
    max_virions: int = 800_000        # 계산부하 상한 (초과 시 가중치 병합)
    seed: int = 20260809


# =============================================================================
# SECTION 5.  MOVEMENT ENGINE  (이동 알고리즘 - 교체 가능 지점)
# =============================================================================


class MovementEngine:
    """
    사양서 23/24/36/37항 대응.

    mode = "random"     : 순수 무작위 보행. 방향은 매 tick 균일난수.
                          병원체 위치/농도/사이토카인 경사 일체 참조하지 않음.
    mode = "systematic" : (미구현) 후속 버전에서 이 클래스만 교체하여
                          동일 생물학 조건에서 비교.

    중요: '이동방향'은 무작위, '이동속도(변위 크기)'는 논문값 축소치.
          두 가지를 반드시 구분한다.
    """

    ALLOWED = ("random", "systematic")

    def __init__(self, mode: str, rng: np.random.Generator):
        if mode not in self.ALLOWED:
            raise ValueError(f"movement_mode must be one of {self.ALLOWED}")
        if mode == "systematic":
            raise NotImplementedError(
                "체계적 이동 모델은 본 버전에서 사용하지 않는다(사양서 24항). "
                "동일 생물학 파라미터로 후속 버전에서 구현할 것.")
        self.mode = mode
        self.rng = rng
        self.n_calls = 0
        self.n_agents_moved = 0

    def move(self, x: np.ndarray, y: np.ndarray,
             sigma_sites: np.ndarray | float) -> None:
        """
        제자리(in-place) 무작위 변위.
        등방성 가우시안 변위 = 방향 균일난수 + 크기 Rayleigh 분포와 동등.
        """
        n = x.size
        if n == 0:
            return
        self.n_calls += 1
        self.n_agents_moved += n

        # --- 방향: 완전 균일 난수 (외부 정보 미참조) ---
        theta = self.rng.uniform(0.0, 2.0 * math.pi, size=n)
        # --- 크기: 논문 운동성계수에서 유도된 스케일 ---
        r = self.rng.normal(0.0, 1.0, size=n) * np.asarray(sigma_sites) * math.sqrt(2.0)

        x += (r * np.cos(theta)).astype(x.dtype)
        y += (r * np.sin(theta)).astype(y.dtype)

        # 경계: 반사(reflecting) - 조직 슬랩 경계
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
# SECTION 6.  ENVIRONMENT  (조직 격자 / 인터페론 / 보체 / 염증)
# =============================================================================

STROMA, HEALTHY, ECLIPSE, PRODUCTIVE, DEAD = 0, 1, 2, 3, 4


class Environment:
    """
    조직 슬랩 격자 + 체액성 인자 필드.

    격자 상태(uint8, 1,000,000칸):
      0 STROMA     : 간질/혈관/림프관 (표적 아님, 면역세포 통행로)
      1 HEALTHY    : 정상 상피세포 (표적)
      2 ECLIPSE    : 감염 직후 잠복기 (바이러스 미생산)
      3 PRODUCTIVE : 감염세포 (바이러스 생산 중)
      4 DEAD       : 사멸 (세포자멸 또는 NK/CTL 살상)  -> 바이러스 생산 중단
    """

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        n = Scale.GRID

        # ---- 정적 조직 배치 (상피 밴드) ----
        rows = np.arange(n)
        epi_row = (rows % Scale.EPI_PERIOD) < Scale.EPI_ROWS
        base = np.where(epi_row, HEALTHY, STROMA).astype(np.uint8)
        self.state = np.repeat(base, n)                   # (1,000,000,)
        self.is_epi = (self.state == HEALTHY).copy()      # 정적 마스크
        self.n_target = int(self.is_epi.sum())

        # ---- 감염세포 컴팩트 리스트 ----
        self.inf_site = np.empty(0, dtype=np.int32)
        self.inf_timer = np.empty(0, dtype=np.int16)
        self.inf_stage = np.empty(0, dtype=np.uint8)      # 0=eclipse, 1=productive
        self.inf_alive = np.empty(0, dtype=bool)

        # ---- 분자 필드 (200 x 200, 1칸 = 75 um) ----
        f = Scale.FIELD_N
        self.ifn = np.zeros((f, f), dtype=np.float32)
        self.antiviral = np.zeros((f, f), dtype=np.float32)

        # ---- 전신 스칼라 ----
        self.complement = 0.0
        self.inflammation = 0.0
        self.tissue_damage = 0.0

        # ---- 누적 ----
        self.n_dead_epi = 0
        self.n_regen = 0

    # -- 좌표 변환 -----------------------------------------------------------
    @staticmethod
    def site_of(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        col = x.astype(np.int32)
        row = y.astype(np.int32)
        np.clip(col, 0, Scale.GRID - 1, out=col)
        np.clip(row, 0, Scale.GRID - 1, out=row)
        return row * Scale.GRID + col

    @staticmethod
    def field_idx(site: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        row = site // Scale.GRID
        col = site - row * Scale.GRID
        return row // Scale.FIELD_BIN, col // Scale.FIELD_BIN

    # -- 감염 등록 -----------------------------------------------------------
    def infect_sites(self, sites: np.ndarray) -> int:
        if sites.size == 0:
            return 0
        self.state[sites] = ECLIPSE
        k = sites.size
        self.inf_site = np.concatenate([self.inf_site, sites.astype(np.int32)])
        self.inf_timer = np.concatenate(
            [self.inf_timer, np.zeros(k, dtype=np.int16)])
        self.inf_stage = np.concatenate(
            [self.inf_stage, np.zeros(k, dtype=np.uint8)])
        self.inf_alive = np.concatenate([self.inf_alive, np.ones(k, dtype=bool)])
        return k

    def kill_sites(self, sites: np.ndarray) -> int:
        """NK/CTL 에 의한 감염세포 제거 (바이러스 생산 즉시 중단)"""
        if sites.size == 0:
            return 0
        sites = np.unique(sites)
        mask = (self.state[sites] == ECLIPSE) | (self.state[sites] == PRODUCTIVE)
        sites = sites[mask]
        if sites.size == 0:
            return 0
        self.state[sites] = DEAD
        self.n_dead_epi += sites.size
        return sites.size

    # -- 감염세포 시간진행 ----------------------------------------------------
    def advance_infected(self) -> Tuple[np.ndarray, int]:
        """
        반환: (현재 생산중인 감염칸 site 배열, 자연사멸한 감염세포 수)
        """
        if self.inf_site.size == 0:
            return np.empty(0, dtype=np.int32), 0

        # NK/CTL 로 이미 죽은 칸 정리
        st = self.state[self.inf_site]
        self.inf_alive &= (st == ECLIPSE) | (st == PRODUCTIVE)

        alive = self.inf_alive
        if not alive.any():
            self._compact()
            return np.empty(0, dtype=np.int32), 0

        self.inf_timer[alive] += 1

        # eclipse -> productive
        to_prod = alive & (self.inf_stage == 0) & (self.inf_timer >= self.p.eclipse_ticks)
        if to_prod.any():
            self.inf_stage[to_prod] = 1
            self.inf_timer[to_prod] = 0
            self.state[self.inf_site[to_prod]] = PRODUCTIVE

        # productive -> 자연 사멸(세포자멸)
        to_dead = alive & (self.inf_stage == 1) & (self.inf_timer >= self.p.productive_ticks)
        n_apoptosis = int(to_dead.sum())
        if n_apoptosis:
            self.state[self.inf_site[to_dead]] = DEAD
            self.inf_alive[to_dead] = False
            self.n_dead_epi += n_apoptosis

        prod_mask = self.inf_alive & (self.inf_stage == 1)
        prod_sites = self.inf_site[prod_mask]

        if self.inf_alive.size > 20000 and self.inf_alive.mean() < 0.5:
            self._compact()
        return prod_sites, n_apoptosis

    def _compact(self):
        keep = self.inf_alive
        self.inf_site = self.inf_site[keep]
        self.inf_timer = self.inf_timer[keep]
        self.inf_stage = self.inf_stage[keep]
        self.inf_alive = self.inf_alive[keep]

    # -- 인터페론 / 항바이러스 상태 -------------------------------------------
    def update_interferon(self, prod_sites: np.ndarray):
        p = self.p
        if prod_sites.size:
            fr, fc = self.field_idx(prod_sites)
            np.add.at(self.ifn, (fr, fc), p.ifn_production)

        # 5점 스텐실 확산 (등방, 방향정보 없음)
        a = self.ifn
        lap = (np.roll(a, 1, 0) + np.roll(a, -1, 0) +
               np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4.0 * a)
        a += p.ifn_diffuse * lap
        a *= (1.0 - p.ifn_decay_p)
        np.clip(a, 0.0, None, out=a)

        # 항바이러스 상태 (포화형)
        drive = np.clip(self.ifn * 6.0, 0.0, 1.0)
        self.antiviral += (p.antiviral_rate * drive * (1.0 - self.antiviral)
                           - p.antiviral_decay * self.antiviral)
        np.clip(self.antiviral, 0.0, 1.0, out=self.antiviral)

    # -- 보체 / 염증 / 손상 ---------------------------------------------------
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

    # -- 상피 재생 -------------------------------------------------------------
    def regenerate(self):
        """
        상피 복구는 염증이 해소되어야 진행된다. 급성 염증기의 IFN/TNF 환경은
        기저세포 증식을 억제하므로 재생속도에 (1 - 염증) 계수를 적용한다.
        (모델 가정값 - 정량 논문값 부재)
        """
        if self.n_dead_epi <= self.n_regen:
            return
        dead_idx = np.flatnonzero(self.state == DEAD)
        if dead_idx.size == 0:
            return
        rate = self.p.epi_regen_p * max(0.0, 1.0 - self.inflammation)
        if rate <= 0.0:
            return
        k = self.rng.binomial(dead_idx.size, rate)
        if k <= 0:
            return
        pick = self.rng.choice(dead_idx, size=min(k, dead_idx.size), replace=False)
        self.state[pick] = HEALTHY
        self.n_regen += pick.size

    # -- 집계 ------------------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        st = self.state
        return {
            "healthy": int(np.count_nonzero(st == HEALTHY)),
            "eclipse": int(np.count_nonzero(st == ECLIPSE)),
            "productive": int(np.count_nonzero(st == PRODUCTIVE)),
            "dead": int(np.count_nonzero(st == DEAD)),
        }


# =============================================================================
# SECTION 7.  VIRUS AGENT POOL  (병원체: 확산 + 감염, 추적 기능 없음)
# =============================================================================


class VirusPool:
    """
    Influenza A 자유 바이러스 agent 집합.
    - 이동: 등방성 확산 (혈류/조직액 이동 + 브라운 운동). 목표 탐색 없음.
    - 감염: 자기가 위치한 칸이 감수성 상피이면 확률적으로 감염.
    - 제거: 비특이 제거(반감기 3h) + 보체 + 항체 중화 + 식세포 포식.
    """

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.x = np.empty(0, dtype=np.float32)
        self.y = np.empty(0, dtype=np.float32)
        self.thinning_events = 0
        self.thin_weight = 1.0     # 상한 초과 시 1 agent 가 대표하는 배수

    @property
    def n(self) -> int:
        return self.x.size

    @property
    def effective_n(self) -> float:
        return self.x.size * self.thin_weight

    def seed_uniform(self, k: int):
        """흡입 에어로졸/비말은 상기도 표면 전체에 분산 침착하므로
        단일 병소가 아니라 다병소로 접종한다 (Memoli 2015 인체감염모델 기준)."""
        self.x = self.rng.uniform(0, Scale.GRID - 1, k).astype(np.float32)
        self.y = self.rng.uniform(0, Scale.GRID - 1, k).astype(np.float32)

    def add(self, x: np.ndarray, y: np.ndarray):
        self.x = np.concatenate([self.x, x.astype(np.float32)])
        self.y = np.concatenate([self.y, y.astype(np.float32)])
        if self.x.size > self.p.max_virions:
            self._thin()

    def _thin(self):
        keep = self.rng.random(self.x.size) < 0.5
        self.x = self.x[keep]
        self.y = self.y[keep]
        self.thin_weight *= 2.0
        self.thinning_events += 1

    def remove(self, mask_remove: np.ndarray) -> int:
        k = int(np.count_nonzero(mask_remove))
        if k:
            keep = ~mask_remove
            self.x = self.x[keep]
            self.y = self.y[keep]
        return k

    def remove_idx(self, idx: np.ndarray) -> int:
        if idx.size == 0:
            return 0
        m = np.zeros(self.x.size, dtype=bool)
        m[idx] = True
        return self.remove(m)


# =============================================================================
# SECTION 8.  IMMUNE CELL POOL  (면역세포 agent - 구조체 배열)
# =============================================================================

NEUT, MONO, NK, TH, TC, BC, MEM = 0, 1, 2, 3, 4, 5, 6
CELL_NAME = {NEUT: "중성구", MONO: "단핵구/대식세포", NK: "NK세포",
             TH: "조력T세포", TC: "살해T세포", BC: "B세포", MEM: "기억세포"}


class ImmuneCellPool:
    """면역세포 agent 를 numpy 구조체배열(SoA)로 관리."""

    def __init__(self, rng: np.random.Generator, cap: int = 8192):
        self.rng = rng
        self.cap = cap
        self.n = 0
        self.x = np.zeros(cap, np.float32)
        self.y = np.zeros(cap, np.float32)
        self.ctype = np.zeros(cap, np.uint8)
        self.age = np.zeros(cap, np.int32)
        self.life = np.zeros(cap, np.int32)
        self.sigma = np.zeros(cap, np.float32)
        self.capacity = np.zeros(cap, np.int16)
        self.specific = np.zeros(cap, bool)     # 항원특이 효과기 여부
        self.alive = np.zeros(cap, bool)

    def _grow(self, need: int):
        while self.n + need > self.cap:
            self.cap *= 2
        for name in ("x", "y", "ctype", "age", "life", "sigma",
                     "capacity", "specific", "alive"):
            arr = getattr(self, name)
            new = np.zeros(self.cap, arr.dtype)
            new[:arr.size] = arr
            setattr(self, name, new)

    def add(self, k: int, ctype: int, sigma: float, life: int,
            capacity: int = 0, specific: bool = False,
            x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
            age_spread: bool = True):
        if k <= 0:
            return
        if self.n + k > self.cap:
            self._grow(k)
        s = slice(self.n, self.n + k)
        self.x[s] = (self.rng.uniform(0, Scale.GRID, k) if x is None else x)
        self.y[s] = (self.rng.uniform(0, Scale.GRID, k) if y is None else y)
        self.ctype[s] = ctype
        self.life[s] = life
        # 정상상태 연령분포: 균등하게 흩어 놓아야 초기에 몰살하지 않음
        self.age[s] = (self.rng.integers(0, max(1, life), k) if age_spread
                       else np.zeros(k, np.int32))
        self.sigma[s] = sigma
        self.capacity[s] = capacity
        self.specific[s] = specific
        self.alive[s] = True
        self.n += k

    # -- 조회 --------------------------------------------------------------
    def mask(self, ctype: int) -> np.ndarray:
        return self.alive[:self.n] & (self.ctype[:self.n] == ctype)

    def count(self, ctype: int) -> int:
        return int(np.count_nonzero(self.mask(ctype)))

    def count_specific(self, ctype: int) -> int:
        return int(np.count_nonzero(self.mask(ctype) & self.specific[:self.n]))

    def compact(self):
        keep = np.flatnonzero(self.alive[:self.n])
        k = keep.size
        for name in ("x", "y", "ctype", "age", "life", "sigma",
                     "capacity", "specific", "alive"):
            arr = getattr(self, name)
            arr[:k] = arr[keep]
            arr[k:self.n] = 0
        self.alive[:k] = True
        self.n = k


# =============================================================================
# SECTION 9.  LYMPH NODE  (림프절: 항원제시 -> 클론증식 -> 효과기/항체/기억)
# =============================================================================


class LymphNode:
    """
    배액 림프절을 well-mixed 구획으로 단순화 (사양서 25/26항).
    개체 단위가 아닌 '개체수(count)' 로 클론증식을 계산하고,
    조직으로 이주하는 효과기만 agent 로 생성한다.
    (효과기 정점은 실제 10^7~10^8 세포 규모여서 전량 agent 화가 불가능)
    """

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.dc_queue = np.zeros(p.dc_delay_ticks + 1, dtype=np.float64)
        self.qi = 0

        self.antigen = 0.0
        self.antigen_total = 0.0
        self.primed = False
        self.prime_clock = 0
        self.activated = False

        self.n_cd4 = p.precursor_cd4      # 림프절 내 항원특이 CD4
        self.n_cd8 = p.precursor_cd8
        self.n_b = p.precursor_b
        self.n_pc_short = 0.0             # 여포외 형질모세포 (IgM)
        self.n_pc_gc = 0.0                # 배중심 유래 형질세포 (IgG)
        self.memory_t = 0.0
        self.memory_b = 0.0

        self.peak_cd8 = p.precursor_cd8
        self.peak_cd4 = p.precursor_cd4
        self.peak_b = p.precursor_b
        self.peak_pc = 0.0
        self.t_expansion_end: Optional[int] = None
        self.contracting = False

        # 이벤트 시각 (tick)
        self.t_antigen_arrival: Optional[int] = None
        self.t_priming_done: Optional[int] = None
        self.t_cd4_activated: Optional[int] = None
        self.t_cd8_activated: Optional[int] = None
        self.t_b_activated: Optional[int] = None
        self.t_first_emigration: Optional[int] = None

    def deposit_antigen(self, amount: float):
        """식세포가 항원을 획득 -> DC 이주지연 후 림프절 도착"""
        if amount <= 0:
            return
        j = (self.qi + self.p.dc_delay_ticks) % self.dc_queue.size
        self.dc_queue[j] += amount

    def update(self, tick: int) -> None:
        p = self.p
        # 1) 지연 도착 항원
        arrive = self.dc_queue[self.qi]
        self.dc_queue[self.qi] = 0.0
        self.qi = (self.qi + 1) % self.dc_queue.size
        if arrive > 0 and self.t_antigen_arrival is None:
            self.t_antigen_arrival = tick
        self.antigen += arrive
        self.antigen_total += arrive
        self.antigen *= (1.0 - p.antigen_decay)

        # 2) 프라이밍 개시
        if not self.primed and self.antigen >= p.antigen_threshold:
            self.primed = True
            self.prime_clock = 0
        if self.primed and not self.activated:
            self.prime_clock += 1
            if self.prime_clock >= p.priming_ticks:
                self.activated = True
                self.t_priming_done = tick
                self.t_cd4_activated = tick
                self.t_cd8_activated = tick
                self.t_b_activated = tick

        if not self.activated:
            return

        elapsed = tick - (self.t_priming_done or tick)

        # ---------------------------------------------------------------
        #  프로그램된 클론증식 (van Stipdonk 2001; Kaech & Ahmed 2002)
        #  항원은 '개시 신호'이며, 일단 개시되면 증식은 자율적으로 진행되고
        #  정해진 기간 경과 후 반드시 수축기로 전환된다.
        #  -> 항원 잔존 여부와 무관하게 정점이 형성되므로 무한증식이 불가능.
        # ---------------------------------------------------------------
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
            # ---- 수축기: 정점의 memory_fraction 까지 감소 후 기억세포로 잔존 ----
            if not self.contracting:
                self.contracting = True
                self.t_expansion_end = tick
            self.n_cd8 = max(self.n_cd8 * (1.0 - p.contraction_p),
                             self.peak_cd8 * p.memory_fraction)
            self.n_cd4 = max(self.n_cd4 * (1.0 - p.contraction_p),
                             self.peak_cd4 * p.memory_fraction)
            self.n_b = max(self.n_b * (1.0 - p.b_contraction_p),
                           self.peak_b * p.memory_b_fraction)
            self.memory_t = min(self.n_cd8 + self.n_cd4,
                                (self.peak_cd8 + self.peak_cd4) * p.memory_fraction)
            self.memory_b = min(self.n_b, self.peak_b * p.memory_b_fraction)

        # ---------------------------------------------------------------
        #  형질세포 분화 (사양서 19/20항)
        #   - 여포외 반응(IgM): 활성화 +2일 부터, 증식기 동안만 신규 분화
        #   - 배중심 반응(IgG): 활성화 +4일 부터 3주까지, CD4 도움 의존
        #  두 경로 모두 '신규 분화 창(window)'이 닫히면 소멸만 남으므로
        #  항체 농도가 무한히 상승하지 않는다.
        # ---------------------------------------------------------------
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
        """조직으로 이주하는 (CTL, Th) agent 수 산출"""
        p = self.p
        if not self.activated:
            return 0, 0
        if tick * Scale.DT_HOUR < p.emigration_start_h:
            return 0, 0
        # 조직 귀소는 림프절 풀 크기에 비례하므로 수축기에는 자동으로 감소한다.
        n_ctl = self.rng.poisson(p.emigration_p * self.n_cd8)
        n_th = self.rng.poisson(p.emigration_p * 0.30 * self.n_cd4)
        if (n_ctl or n_th) and self.t_first_emigration is None:
            self.t_first_emigration = tick
        return int(n_ctl), int(n_th)


# =============================================================================
# SECTION 10.  ANTIBODY & COMPLEMENT  (체액성 면역)
# =============================================================================


class Antibody:
    """
    IgM(초기) / IgG(후기) 를 ug/mL 로 추적.
    효과: 중화(감염력 감소) / 옵소닌화(식균 보조) / 보체 활성 보조.
    접촉 즉시 사멸이 아니라 농도의존적 확률로 작용한다 (사양서 20항).
    """

    def __init__(self, p: Params):
        self.p = p
        self.igm = 0.0
        self.igg = 0.0
        self.t_igm_detect: Optional[int] = None
        self.t_igg_detect: Optional[int] = None
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
        if self.t_igm_detect is None and self.igm >= p.ab_detect_threshold:
            self.t_igm_detect = tick
        if self.t_igg_detect is None and self.igg >= p.ab_detect_threshold:
            self.t_igg_detect = tick
        self.peak_total = max(self.peak_total, self.total)

    def neutralization_rate(self) -> float:
        """tick 당 바이러스 중화 확률 (포화형)"""
        c = self.total
        return self.p.ab_neutralize_max * c / (c + self.p.ab_kd)

    def opsonin_bonus(self) -> float:
        c = self.total
        return self.p.ab_opsonin * c / (c + self.p.ab_kd)

    def infection_block(self) -> float:
        """
        중화항체가 바이러스의 '세포 침입 능력' 자체를 차단하는 비율.
        HA 결합부위를 가려 부착/융합을 저해하는 기전이며, 유리 바이러스를
        제거하는 것과는 별개의 효과이다 (사양서 20항).
        접촉 즉시 사멸이 아니라 농도의존적 확률로 작용한다.
        """
        c = self.total
        return self.p.ab_block_max * c / (c + self.p.ab_block_kd)


# =============================================================================
# SECTION 11.  SIMULATION CONTROLLER
# =============================================================================


class ImmuneSimulation:

    def __init__(self, p: Optional[Params] = None, movement_mode: str = "random",
                 seed: Optional[int] = None, verbose: bool = True):
        self.p = p or Params()
        if seed is not None:
            self.p.seed = seed
        self.rng = np.random.default_rng(self.p.seed)
        self.movement_mode = movement_mode
        self.mover = MovementEngine(movement_mode, self.rng)
        self.verbose = verbose

        self.env = Environment(self.p, self.rng)
        self.virus = VirusPool(self.p, self.rng)
        self.cells = ImmuneCellPool(self.rng)
        self.ln = LymphNode(self.p, self.rng)
        self.ab = Antibody(self.p)

        self.tick = 0
        self.history: List[dict] = []
        self.daily: List[dict] = []

        # 누적 통계
        self.cleared_virus = 0            # 총 제거 병원체
        self.cleared_by_phago = 0
        self.cleared_by_decay = 0
        self.cleared_by_complement = 0
        self.cleared_by_antibody = 0
        self.killed_infected = 0          # NK+CTL 살상
        self.killed_by_nk = 0
        self.killed_by_ctl = 0
        self.apoptosis_infected = 0       # 세포자멸(면역세포 무관)
        self.dead_immune = 0
        self.total_infections = 0         # 누적 감염 발생칸
        self.ctl_contact = 0
        self.ctl_kill_events = 0
        self.ctl_agent_ticks = 0
        self.consumed_first_gen = 0

        self._recruit_target = {NEUT: float(self.p.n_neutrophil),
                                MONO: float(self.p.n_monocyte),
                                NK: float(self.p.n_nk)}
        self._init_population()

    # ------------------------------------------------------------------
    def _init_population(self):
        p = self.p
        # 병원체: 도메인 중앙부에 국소 접종 (비강 접종 부위 모사)
        self.virus.seed_uniform(p.n_virus0)

        c = self.cells
        c.add(p.n_neutrophil, NEUT, p.sigma_neut, p.life_neut, p.neut_capacity)
        c.add(p.n_monocyte, MONO, p.sigma_mono, p.life_mono, p.mono_capacity)
        c.add(p.n_nk, NK, p.sigma_nk, p.life_nk)
        c.add(p.n_helper_t, TH, p.sigma_naive_t, p.life_naive)
        c.add(p.n_killer_t, TC, p.sigma_naive_t, p.life_naive)
        c.add(p.n_bcell, BC, p.sigma_b, p.life_naive)

    # ------------------------------------------------------------------
    #  헬퍼: 같은 격자칸에 있는 (식세포, 바이러스) 매칭
    # ------------------------------------------------------------------
    def _match_same_site(self, p_site: np.ndarray, v_site: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """반환 (매칭된 식세포 지역인덱스, 매칭된 바이러스 전역인덱스)"""
        if p_site.size == 0 or v_site.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        pu = np.unique(p_site)
        inmask = np.isin(v_site, pu, kind="table")
        cand = np.flatnonzero(inmask)
        if cand.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        cs = v_site[cand]
        order = np.argsort(cs, kind="stable")
        cs_s = cs[order]
        cand_s = cand[order]
        uniq, start, counts = np.unique(cs_s, return_index=True, return_counts=True)

        pos = np.searchsorted(uniq, p_site)
        pos_c = np.minimum(pos, uniq.size - 1)
        has = uniq[pos_c] == p_site
        idx_p = np.flatnonzero(has)
        if idx_p.size == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        g = pos_c[idx_p]
        off = self.rng.integers(0, counts[g])
        v_idx = cand_s[start[g] + off]
        # 동일 바이러스 중복 포식 방지
        _, keep = np.unique(v_idx, return_index=True)
        return idx_p[keep], v_idx[keep]

    # ------------------------------------------------------------------
    def step(self):
        p = self.p
        env = self.env
        self.tick += 1
        t = self.tick

        # ---------------- A. 감염세포 시간진행 -------------------------
        prod_sites, n_apop = env.advance_infected()
        self.apoptosis_infected += n_apop

        # ---------------- B. 바이러스 생산 (인터페론 억제 반영) ---------
        if prod_sites.size:
            fr, fc = Environment.field_idx(prod_sites)
            av = env.antiviral[fr, fc]
            rate = (p.burst_agents_per_site / p.productive_ticks) * \
                   (1.0 - p.ifn_block_release * av) / max(1.0, self.virus.thin_weight)
            k = self.rng.poisson(np.maximum(rate, 0.0))
            tot = int(k.sum())
            if tot:
                src = np.repeat(prod_sites, k)
                row = src // Scale.GRID
                col = src - row * Scale.GRID
                nx = (col + self.rng.random(tot)).astype(np.float32)
                ny = (row + self.rng.random(tot)).astype(np.float32)
                self.virus.add(nx, ny)

        # ---------------- C. 바이러스 이동 (무작위 확산) ----------------
        if self.virus.n:
            self.mover.move(self.virus.x, self.virus.y,
                            math.hypot(p.virus_sigma, p.virus_sigma_mucus))

        # ---------------- D. 감염 시도 ---------------------------------
        n_new_inf = 0
        if self.virus.n:
            v_site = Environment.site_of(self.virus.x, self.virus.y)
            st = env.state[v_site]
            sus = np.flatnonzero(st == HEALTHY)
            if sus.size:
                s_sites = v_site[sus]
                row = s_sites // Scale.GRID
                col = s_sites - row * Scale.GRID
                av = env.antiviral[row // Scale.FIELD_BIN, col // Scale.FIELD_BIN]
                # 감염확률 = 기본감염력 x (1 - 항바이러스상태) x (1 - 항체차단)
                pe = p.p_infect * (1.0 - av) * (1.0 - self.ab.infection_block())
                if self.virus.thin_weight > 1.0:
                    pe = 1.0 - (1.0 - pe) ** self.virus.thin_weight
                ok = self.rng.random(sus.size) < pe
                if ok.any():
                    hit_sites = np.unique(s_sites[ok])
                    n_new_inf = env.infect_sites(hit_sites)
                    self.total_infections += n_new_inf
                    self.virus.remove_idx(sus[ok])
        else:
            v_site = np.empty(0, np.int32)

        # ---------------- E. 면역세포 노화/사멸 -------------------------
        c = self.cells
        n = c.n
        if n:
            alive = c.alive[:n]
            c.age[:n][alive] += 1
            died = alive & (c.age[:n] >= c.life[:n])
            # 효과기 T세포 사멸: 확장기에도 상시 적용, 수축기에 가속
            # (확장기 조직 효과기도 지속적 교체가 일어나므로 무한 누적 불가)
            eff = alive & c.specific[:n] & ((c.ctype[:n] == TC) | (c.ctype[:n] == TH))
            if eff.any():
                pd_eff = (p.eff_t_death_contract if self.ln.contracting
                          else p.eff_t_death_expand)
                r = self.rng.random(n) < pd_eff
                died |= eff & r
            k = int(np.count_nonzero(died))
            if k:
                c.alive[:n][died] = False
                self.dead_immune += k
            if n > 5000 and c.alive[:n].mean() < 0.85:
                c.compact()
                n = c.n

        # ---------------- F. 면역세포 이동 (핵심: 무작위 보행) ----------
        n = c.n
        alive_idx = np.flatnonzero(c.alive[:n])
        if alive_idx.size:
            xa = c.x[alive_idx]
            ya = c.y[alive_idx]
            self.mover.move(xa, ya, c.sigma[alive_idx])
            c.x[alive_idx] = xa
            c.y[alive_idx] = ya
            cell_site = Environment.site_of(xa, ya)
        else:
            cell_site = np.empty(0, np.int32)

        ct = c.ctype[alive_idx] if alive_idx.size else np.empty(0, np.uint8)

        antigen_in = 0.0

        # ---------------- G. 식균작용 (중성구 / 단핵구) -----------------
        if alive_idx.size and self.virus.n:
            v_site = Environment.site_of(self.virus.x, self.virus.y)
            opson = min(0.95, self.ab.opsonin_bonus()
                        + p.complement_opsonin * env.complement)

            for ctype, base_p in ((NEUT, p.neut_phago_p), (MONO, p.mono_phago_p)):
                sel = np.flatnonzero((ct == ctype) & (c.capacity[alive_idx] > 0))
                if sel.size == 0:
                    continue
                loc, vidx = self._match_same_site(cell_site[sel], v_site)
                if loc.size == 0:
                    continue
                succ = self.rng.random(loc.size) < min(0.98, base_p + opson)
                if not succ.any():
                    continue
                gi = alive_idx[sel[loc[succ]]]
                c.capacity[gi] -= 1
                spent = gi[c.capacity[gi] <= 0]
                if spent.size:
                    c.alive[spent] = False
                    self.dead_immune += spent.size
                k = self.virus.remove_idx(vidx[succ])
                self.cleared_by_phago += k
                self.cleared_virus += k
                if ctype == MONO:
                    antigen_in += k * 1.0
                v_site = Environment.site_of(self.virus.x, self.virus.y)

        # ---------------- H. NK세포: 감염세포 용해 ----------------------
        if alive_idx.size:
            sel = np.flatnonzero(ct == NK)
            if sel.size:
                s = cell_site[sel]
                stt = env.state[s]
                tgt = (stt == PRODUCTIVE) | (stt == ECLIPSE)
                if tgt.any():
                    boost = 1.0 + 0.5 * float(env.inflammation)
                    ok = self.rng.random(int(tgt.sum())) < min(0.95, p.nk_kill_p * boost)
                    ks = s[tgt][ok]
                    k = env.kill_sites(ks)
                    self.killed_by_nk += k
                    self.killed_infected += k

        # ---------------- I. 살해 T세포(항원특이 효과기) ----------------
        if alive_idx.size:
            spec = c.specific[alive_idx]
            sel = np.flatnonzero((ct == TC) & spec)
            self.ctl_agent_ticks += sel.size
            if sel.size:
                s = cell_site[sel]
                stt = env.state[s]
                tgt = (stt == PRODUCTIVE) | (stt == ECLIPSE)
                nt = int(tgt.sum())
                self.ctl_contact += nt
                if nt:
                    ok = self.rng.random(nt) < p.ctl_kill_p
                    ks = s[tgt][ok]
                    k = env.kill_sites(ks)
                    self.killed_by_ctl += k
                    self.killed_infected += k
                    self.ctl_kill_events += k

        # ---------------- J. 상주 DC/대식세포 항원 표본추출 --------------
        n_infected_now = int(env.inf_alive.sum()) if env.inf_alive.size else 0
        antigen_in += 0.005 * n_infected_now
        self.ln.deposit_antigen(antigen_in)

        # ---------------- K. 바이러스 제거 (비특이/보체/항체) -----------
        if self.virus.n:
            pd = p.virus_decay_p
            pc = p.complement_lysis * env.complement
            pa = self.ab.neutralization_rate()
            ptot = 1.0 - (1.0 - pd) * (1.0 - pc) * (1.0 - pa)
            rem = self.rng.random(self.virus.n) < ptot
            k = self.virus.remove(rem)
            if k:
                tot = pd + pc + pa
                self.cleared_by_decay += int(k * pd / tot)
                self.cleared_by_complement += int(k * pc / tot)
                self.cleared_by_antibody += int(k * pa / tot)
                self.ab.neutralized += int(k * pa / tot)
                self.cleared_virus += k

        # ---------------- L. 림프절 / 항체 -------------------------------
        self.ln.update(t)
        n_ctl, n_th = self.ln.emigrate(t)
        if n_ctl:
            c.add(n_ctl, TC, p.sigma_eff_t, p.life_naive, specific=True,
                  age_spread=False)
        if n_th:
            c.add(n_th, TH, p.sigma_eff_t, p.life_naive, specific=True,
                  age_spread=False)
        self.ab.update(self.ln, t)

        # ---------------- M. 체액인자 / 염증 / 동원 / 재생 ---------------
        env.update_interferon(prod_sites)
        env.update_humoral(self.virus.n, n_infected_now)
        self._recruit()
        env.regenerate()

        self._record()

    # ------------------------------------------------------------------
    def _recruit(self):
        """염증에 비례한 골수/혈류로부터의 선천면역세포 동원 (사양서 15항)"""
        p = self.p
        infl = self.env.inflammation
        spec = {NEUT: (p.n_neutrophil, p.recruit_neut_max, p.sigma_neut,
                       p.life_neut, p.neut_capacity),
                MONO: (p.n_monocyte, p.recruit_mono_max, p.sigma_mono,
                       p.life_mono, p.mono_capacity),
                NK: (p.n_nk, p.recruit_nk_max, p.sigma_nk, p.life_nk, 0)}
        for ctype, (base, mx, sg, lf, cap) in spec.items():
            target = base * (1.0 + mx * infl)
            cur = self.cells.count(ctype)
            gap = target - cur
            if gap <= 0:
                continue
            k = self.rng.poisson(gap * p.recruit_rate)
            if k:
                self.cells.add(int(k), ctype, sg, lf, cap, age_spread=False)

    # ------------------------------------------------------------------
    def _snapshot(self) -> dict:
        env, c, ln, ab = self.env, self.cells, self.ln, self.ab
        cn = env.counts()
        infected = cn["eclipse"] + cn["productive"]
        return {
            "tick": self.tick,
            "hours": self.tick * Scale.DT_HOUR,
            "day": self.tick / Scale.TICKS_PER_DAY,
            "virus": self.virus.n,
            "virus_eff": self.virus.effective_n,
            "infected": infected,
            "eclipse": cn["eclipse"],
            "productive": cn["productive"],
            "healthy": cn["healthy"],
            "dead_epi": cn["dead"],
            "neutrophil": c.count(NEUT),
            "monocyte": c.count(MONO),
            "nk": c.count(NK),
            "helper_t": c.count(TH),
            "helper_t_spec": c.count_specific(TH),
            "killer_t": c.count(TC),
            "killer_t_spec": c.count_specific(TC),
            "bcell": c.count(BC),
            "memory": ln.memory_t + ln.memory_b,
            "antibody": ab.total,
            "igm": ab.igm,
            "igg": ab.igg,
            "complement": env.complement,
            "interferon": float(env.ifn.mean()),
            "interferon_max": float(env.ifn.max()),
            "antiviral": float(env.antiviral.mean()),
            "inflammation": env.inflammation,
            "damage": env.tissue_damage,
            "cleared_virus": self.cleared_virus,
            "killed_infected": self.killed_infected,
            "apoptosis": self.apoptosis_infected,
            "dead_immune": self.dead_immune,
            "ln_cd8": ln.n_cd8,
            "ln_cd4": ln.n_cd4,
            "ln_pc": ln.n_pc_short + ln.n_pc_gc,
        }

    RECORD_EVERY = 5    # 30분 간격 기록 (전 격자 스캔 비용 절감)

    def _record(self):
        if (self.tick % self.RECORD_EVERY == 0
                or self.tick % Scale.TICKS_PER_DAY == 0):
            self.history.append(self._snapshot())

    # ==================================================================
    #  SECTION 12.  DAILY SNAPSHOT  (사양서 29항 - 하루마다 필수 출력)
    # ==================================================================
    REQUIRED_FIELDS = [
        ("현재 시뮬레이션 시간", "time"),
        ("병원체 수", "virus"),
        ("감염세포 수", "infected"),
        ("정상세포 수", "healthy"),
        ("중성구 수", "neutrophil"),
        ("단핵구/대식세포 수", "monocyte"),
        ("NK세포 수", "nk"),
        ("조력 T세포 수", "helper_t"),
        ("살해 T세포 수", "killer_t"),
        ("B세포 수", "bcell"),
        ("기억세포 수", "memory"),
        ("항체 농도", "antibody"),
        ("보체 활성도", "complement"),
        ("인터페론 농도", "interferon"),
        ("염증 정도", "inflammation"),
        ("제거된 병원체 수", "cleared_virus"),
        ("제거된 감염세포 수", "killed_infected"),
        ("사망한 면역세포 수", "dead_immune"),
    ]

    @staticmethod
    def format_block(s: dict, title: str) -> str:
        """18개 필수 항목 블록. 하루 경과 시 및 최종 출력에 항상 사용."""
        L = []
        L.append("=" * 78)
        L.append(f"  {title}")
        L.append("-" * 78)
        d = s["day"]
        L.append(f"  현재 시뮬레이션 시간   : Day {d:5.2f}   "
                 f"({s['hours']:7.1f} h, tick {s['tick']})")
        L.append(f"  병원체 수              : {s['virus']:>12,}  agent"
                 f"   (= {s['virus_eff']*Scale.INV_KAPPA:.3e} 감염단위 상당)")
        L.append(f"  감염세포 수            : {s['infected']:>12,}  칸"
                 f"   (잠복 {s['eclipse']:,} / 생산 {s['productive']:,})")
        L.append(f"  정상세포 수            : {s['healthy']:>12,}  칸"
                 f"   (사멸상피 {s['dead_epi']:,})")
        L.append(f"  중성구 수              : {s['neutrophil']:>12,}  agent")
        L.append(f"  단핵구/대식세포 수     : {s['monocyte']:>12,}  agent")
        L.append(f"  NK세포 수              : {s['nk']:>12,}  agent")
        L.append(f"  조력 T세포 수          : {s['helper_t']:>12,}  agent"
                 f"   (항원특이 효과기 {s['helper_t_spec']:,})")
        L.append(f"  살해 T세포 수          : {s['killer_t']:>12,}  agent"
                 f"   (항원특이 효과기 {s['killer_t_spec']:,})")
        L.append(f"  B세포 수               : {s['bcell']:>12,}  agent"
                 f"   (림프절 형질세포 {s['ln_pc']:,.0f})")
        L.append(f"  기억세포 수            : {s['memory']:>12,.0f}  agent 상당")
        L.append(f"  항체 농도              : {s['antibody']:>12.3f}  ug/mL"
                 f"   (IgM {s['igm']:.2f} / IgG {s['igg']:.2f})")
        L.append(f"  보체 활성도            : {s['complement']*100:>12.2f}  %")
        L.append(f"  인터페론 농도          : {s['interferon']*500:>12.2f}  pg/mL 상당"
                 f"   (국소최대 {s['interferon_max']*500:.1f}, "
                 f"항바이러스상태 평균 {s['antiviral']*100:.1f}%)")
        L.append(f"  염증 정도              : {s['inflammation']*100:>12.2f}  %"
                 f"   (누적 조직손상 {s['damage']:.3f})")
        L.append(f"  제거된 병원체 수       : {s['cleared_virus']:>12,}  누적")
        L.append(f"  제거된 감염세포 수     : {s['killed_infected']:>12,}  누적 "
                 f"(NK/CTL 살상, 별도 세포자멸 {s['apoptosis']:,})")
        L.append(f"  사망한 면역세포 수     : {s['dead_immune']:>12,}  누적")
        L.append("=" * 78)
        return "\n".join(L)

    # ==================================================================
    #  RUN LOOP
    # ==================================================================
    def run(self) -> None:
        p = self.p
        t0 = time.time()

        snap0 = self._snapshot()
        self.daily.append(snap0)
        if self.verbose:
            print(self.format_block(snap0, "Day 0  |  초기 상태 (감염 직후)"))
            print()

        clear_streak = 0
        max_ticks = p.max_days * Scale.TICKS_PER_DAY
        while self.tick < max_ticks:
            self.step()

            if self.tick % Scale.TICKS_PER_DAY == 0:
                s = self._snapshot()
                self.daily.append(s)
                if self.verbose:
                    day = self.tick // Scale.TICKS_PER_DAY
                    print(self.format_block(s, f"Day {day}  |  경과 {day}일"))
                    print()

            # 완전 제거 판정
            if self.virus.n == 0 and (self.env.inf_alive.sum() if
                                      self.env.inf_alive.size else 0) == 0:
                clear_streak += 1
            else:
                clear_streak = 0
            if clear_streak >= p.stop_after_clear_days * Scale.TICKS_PER_DAY:
                break

        # 마지막 시점이 일 단위가 아니면 추가 기록
        if self.daily[-1]["tick"] != self.tick:
            self.daily.append(self._snapshot())

        self.wall_time = time.time() - t0


# =============================================================================
# SECTION 13.  STATISTICS & LITERATURE VALIDATION
# =============================================================================


def _series(hist: List[dict], key: str) -> np.ndarray:
    return np.array([h[key] for h in hist], dtype=float)


def _days(hist: List[dict]) -> np.ndarray:
    return np.array([h["day"] for h in hist], dtype=float)


def _peak(hist, key) -> Tuple[float, float]:
    v = _series(hist, key)
    if v.size == 0:
        return 0.0, float("nan")
    i = int(np.argmax(v))
    return float(v[i]), float(hist[i]["day"])


def _first_below_after_peak(hist, key, frac) -> Optional[float]:
    v = _series(hist, key)
    if v.size == 0 or v.max() <= 0:
        return None
    i = int(np.argmax(v))
    thr = v[i] * frac
    for j in range(i, v.size):
        if v[j] <= thr:
            return float(hist[j]["day"])
    return None


def _first_zero_after_peak(hist, key) -> Optional[float]:
    v = _series(hist, key)
    if v.size == 0:
        return None
    i = int(np.argmax(v))
    for j in range(i, v.size):
        if v[j] <= 0:
            return float(hist[j]["day"])
    return None


def analyze(sim: "ImmuneSimulation") -> dict:
    h = sim.history
    R = {}

    R["virus_peak"], R["virus_peak_day"] = _peak(h, "virus")
    R["virus_final"] = h[-1]["virus"] if h else 0
    R["virus_t50"] = _first_below_after_peak(h, "virus", 0.5)
    R["virus_t90"] = _first_below_after_peak(h, "virus", 0.1)
    R["virus_clear_day"] = _first_zero_after_peak(h, "virus")
    # 문헌의 '바이러스 배출기간'은 배양/PCR 검출한계 이하가 되는 시점이므로
    # agent 가 정확히 0 이 되는 시점이 아니라 정점 대비 상대역치로 비교한다.
    R["virus_shed_end_1pct"] = _first_below_after_peak(h, "virus", 0.01)
    R["virus_shed_end_01pct"] = _first_below_after_peak(h, "virus", 0.001)

    R["inf_peak"], R["inf_peak_day"] = _peak(h, "infected")
    R["inf_t50"] = _first_below_after_peak(h, "infected", 0.5)
    R["inf_clear_day"] = _first_zero_after_peak(h, "infected")

    R["neut_peak"], R["neut_peak_day"] = _peak(h, "neutrophil")
    R["mono_peak"], R["mono_peak_day"] = _peak(h, "monocyte")
    R["nk_peak"], R["nk_peak_day"] = _peak(h, "nk")
    R["ifn_peak"], R["ifn_peak_day"] = _peak(h, "interferon")
    R["comp_peak"], R["comp_peak_day"] = _peak(h, "complement")
    R["inflam_peak"], R["inflam_peak_day"] = _peak(h, "inflammation")
    R["damage_final"] = h[-1]["damage"] if h else 0.0

    R["ctl_peak"], R["ctl_peak_day"] = _peak(h, "killer_t_spec")
    R["th_peak"], R["th_peak_day"] = _peak(h, "helper_t_spec")
    R["pc_peak"], R["pc_peak_day"] = _peak(h, "ln_pc")
    R["ab_peak"], R["ab_peak_day"] = _peak(h, "antibody")

    tpd = Scale.TICKS_PER_DAY
    ln = sim.ln
    R["t_antigen_arrival"] = (ln.t_antigen_arrival / tpd
                              if ln.t_antigen_arrival else None)
    R["t_priming_done"] = (ln.t_priming_done / tpd if ln.t_priming_done else None)
    R["t_cd4_act"] = (ln.t_cd4_activated / tpd if ln.t_cd4_activated else None)
    R["t_cd8_act"] = (ln.t_cd8_activated / tpd if ln.t_cd8_activated else None)
    R["t_b_act"] = (ln.t_b_activated / tpd if ln.t_b_activated else None)
    R["t_emigration"] = (ln.t_first_emigration / tpd
                         if ln.t_first_emigration else None)
    R["t_igm"] = (sim.ab.t_igm_detect / tpd if sim.ab.t_igm_detect else None)
    R["t_igg"] = (sim.ab.t_igg_detect / tpd if sim.ab.t_igg_detect else None)

    R["memory_final"] = ln.memory_t + ln.memory_b
    R["ln_cd8_peak"] = ln.peak_cd8
    R["memory_ratio"] = (ln.memory_t / ln.peak_cd8 if ln.peak_cd8 > 0 else 0.0)

    # 초기 지수증식률 -> 체내 R0 추정
    days = _days(h)
    inf = _series(h, "infected")
    m = (days >= 0.4) & (days <= 1.3) & (inf > 5)
    if m.sum() > 10:
        r = float(np.polyfit(days[m], np.log(inf[m]), 1)[0])   # per day
        Tg = (sim.p.eclipse_ticks + 0.5 * sim.p.productive_ticks) / Scale.TICKS_PER_HOUR
        R["growth_rate_per_day"] = r
        R["R0_est"] = float(math.exp(r * Tg / 24.0))
    else:
        R["growth_rate_per_day"] = float("nan")
        R["R0_est"] = float("nan")

    # CTL 세포당 살상률 (cells / CTL / day)  * kappa 균일축소이므로 직접 비교 가능
    if sim.ctl_agent_ticks > 0:
        R["ctl_kill_per_day"] = sim.killed_by_ctl / sim.ctl_agent_ticks * tpd
    else:
        R["ctl_kill_per_day"] = float("nan")
    if sim.ctl_agent_ticks > 0:
        R["ctl_contact_per_day"] = sim.ctl_contact / sim.ctl_agent_ticks * tpd
    else:
        R["ctl_contact_per_day"] = float("nan")

    R["target_depletion"] = 1.0 - (h[-1]["healthy"] / sim.env.n_target) if h else 0.0
    R["max_target_depletion"] = 1.0 - float(_series(h, "healthy").min()) / sim.env.n_target

    R["killed_by_nk"] = sim.killed_by_nk
    R["killed_by_ctl"] = sim.killed_by_ctl
    R["apoptosis"] = sim.apoptosis_infected
    R["cleared_by_phago"] = sim.cleared_by_phago
    R["cleared_by_decay"] = sim.cleared_by_decay
    R["cleared_by_complement"] = sim.cleared_by_complement
    R["cleared_by_antibody"] = sim.cleared_by_antibody
    R["total_infections"] = sim.total_infections
    return R


@dataclass
class Check:
    item: str
    lit: str
    ref: str
    sim: str
    verdict: str


def _judge(val: Optional[float], lo: float, hi: float,
           tol: float = 0.35) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "판정불가"
    if lo <= val <= hi:
        return "일치"
    span = max(hi - lo, 1e-9)
    if lo - span * tol <= val <= hi + span * tol:
        return "근접"
    return "불일치"


def validate(sim: "ImmuneSimulation", R: dict) -> List[Check]:
    def f(v, unit="일"):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "미발생"
        return f"Day {v:.2f}" if unit == "일" else f"{v:.3g}"

    C: List[Check] = []
    C.append(Check("바이러스 정점 시각", "Day 2 (0.5~1일 급증 후 2일 정점)",
                   "carrat2008", f(R["virus_peak_day"]),
                   _judge(R["virus_peak_day"], 1.5, 2.5)))
    C.append(Check("감염세포 정점 시각", "Day 1~3 (표적세포제한 모델)",
                   "baccam2006", f(R["inf_peak_day"]),
                   _judge(R["inf_peak_day"], 1.0, 3.0)))
    C.append(Check("바이러스 배출 종료(정점의 1%)", "Day 4.3~5.3 (평균 4.8일)",
                   "carrat2008", f(R["virus_shed_end_1pct"]),
                   _judge(R["virus_shed_end_1pct"], 4.3, 6.5)))
    C.append(Check("바이러스 완전 제거 시각", "배출 종료 후 수일, 최대 Day 10 내외",
                   "carrat2008", f(R["virus_clear_day"]),
                   _judge(R["virus_clear_day"], 5.0, 12.0)))
    C.append(Check("바이러스 50% 감소 시각", "정점 후 1~2일 내",
                   "carrat2008", f(R["virus_t50"]),
                   _judge((R["virus_t50"] - R["virus_peak_day"])
                          if R["virus_t50"] else None, 0.2, 2.0)))
    C.append(Check("체내 기초재생산수 R0", "약 22 (감염세포 1개당 신규 생산감염)",
                   "baccam2006", f"{R['R0_est']:.1f} (초기 증식률 추정)",
                   _judge(R["R0_est"], 8.0, 40.0)))
    C.append(Check("인터페론 정점 시각", "Day 2 (비강세척액 IFN 정점)",
                   "hayden1998", f(R["ifn_peak_day"]),
                   _judge(R["ifn_peak_day"], 1.5, 3.0)))
    C.append(Check("염증/증상 정점 시각", "Day 2~3 (전신증상 2일, 총증상 3일)",
                   "carrat2008", f(R["inflam_peak_day"]),
                   _judge(R["inflam_peak_day"], 1.5, 3.5)))
    C.append(Check("중성구 조직유입 정점", "Day 2~3",
                   "hayden1998", f(R["neut_peak_day"]),
                   _judge(R["neut_peak_day"], 1.5, 3.5)))
    C.append(Check("NK세포 유입 정점", "Day 2~3",
                   "zhang2007", f(R["nk_peak_day"]),
                   _judge(R["nk_peak_day"], 1.5, 3.5)))
    C.append(Check("T세포 프라이밍 완료", "Day 1.5~2.5 (DC 이주 12~24h + 프라이밍 24h)",
                   "mempel2004", f(R["t_priming_done"]),
                   _judge(R["t_priming_done"], 1.2, 3.0)))
    C.append(Check("항원특이 CD8 조직 정점", "Day 8~10 (일차 인플루엔자 감염)",
                   "lawrence2005", f(R["ctl_peak_day"]),
                   _judge(R["ctl_peak_day"], 7.0, 11.0)))
    C.append(Check("형질모세포 정점 시각", "Day 7",
                   "wrammert2008", f(R["pc_peak_day"]),
                   _judge(R["pc_peak_day"], 6.0, 9.0)))
    C.append(Check("IgM 검출 시각", "Day 5~7",
                   "clements1986", f(R["t_igm"]),
                   _judge(R["t_igm"], 4.5, 8.0)))
    C.append(Check("IgG 검출 시각", "Day 10~14",
                   "clements1986", f(R["t_igg"]),
                   _judge(R["t_igg"], 9.0, 15.0)))
    C.append(Check("기억세포 형성비율", "정점 대비 5~10%",
                   "kaech2002", f"{R['memory_ratio']*100:.1f}%",
                   _judge(R["memory_ratio"] * 100, 5.0, 10.0)))
    C.append(Check("CTL 세포당 살상률", "2~16 표적세포/CTL/일 (모델의존적)",
                   "regoes2007/ganusov2008",
                   f"{R['ctl_kill_per_day']:.2f} 칸/CTL/일",
                   _judge(R["ctl_kill_per_day"], 2.0, 16.0)))
    C.append(Check("표적 상피세포 소모율", "표적세포제한 모델에서 상당 비율 소모",
                   "baccam2006", f"{R['max_target_depletion']*100:.1f}%",
                   _judge(R["max_target_depletion"] * 100, 20.0, 95.0)))
    return C


# =============================================================================
# SECTION 14.  REPORT PRINTERS
# =============================================================================


def print_param_table():
    print()
    print("#" * 100)
    print("  [논문 근거 표]  실제값 -> 논문출처 -> 축소비율 -> 시뮬레이션값   (사양서 34항)")
    print("#" * 100)
    hdr = f"{'항목':<24}{'실제값':>14} {'단위':<18}{'출처':<16}{'축소비율':<14}{'시뮬레이션값'}"
    print(hdr)
    print("-" * 130)
    for r in PARAM_TABLE:
        src = REFS[r.source].short() if r.source in REFS else r.source
        print(f"{r.item:<24}{r.real:>14} {r.unit:<18}{src:<16}{r.scaling:<14}{r.sim}")
        if r.note:
            print(f"{'':<24}   └ {r.note}")
    print("-" * 130)
    print("  * '모델 가정값' 표기 항목은 논문에서 직접 대응 수치를 찾지 못한 값이며,")
    print("    실제값처럼 제시하지 않고 가정임을 명시하였다 (사양서 35항).")
    print()
    print("  [인용 문헌 전체]")
    for k in sorted(REFS):
        print(f"   - {REFS[k].full()}")
    print("  * DOI/PMID 는 작성 시점 기준. 보고서 인용 전 PubMed 재확인 권장.")
    print()


def print_final_report(sim: "ImmuneSimulation", R: dict, checks: List[Check]):
    p = sim.p
    last = sim.history[-1] if sim.history else sim._snapshot()

    print()
    print("#" * 100)
    print("  최 종 결 과   (무작위 이동 조건, 1회 실행)")
    print("#" * 100)

    # ---- (1) 필수 18항목 블록 재출력 ----
    print(ImmuneSimulation.format_block(
        last, f"FINAL  |  종료 시점 Day {last['day']:.2f}  "
              f"(병원체 소멸 후 관찰기간 포함)"))
    print()

    # ---- (2) 상세 결과 (사양서 33항) ----
    def d(v):
        return "미발생" if v is None or (isinstance(v, float) and math.isnan(v)) \
            else f"Day {v:.2f}"

    print("-" * 78)
    print("  [병원체]")
    print(f"   · 최대 병원체 수            : {R['virus_peak']:,.0f} agent "
          f"(= {R['virus_peak']*Scale.INV_KAPPA:.2e} 감염단위 상당)  @ {d(R['virus_peak_day'])}")
    print(f"   · 최종 병원체 수            : {R['virus_final']:,.0f} agent")
    print(f"   · 병원체 50% 감소시간       : {d(R['virus_t50'])}")
    print(f"   · 병원체 90% 감소시간       : {d(R['virus_t90'])}")
    print(f"   · 병원체 배출종료(정점 1%)  : {d(R['virus_shed_end_1pct'])}"
          f"   [문헌 '배출기간'과 동일 기준]")
    print(f"   · 병원체 배출종료(정점 0.1%): {d(R['virus_shed_end_01pct'])}")
    print(f"   · 병원체 완전 제거시간      : {d(R['virus_clear_day'])}"
          f"   [agent 0 도달, 문헌보다 엄격한 기준]")
    print(f"   · 폭발적 증가 최대값        : {R['virus_peak']:,.0f} agent "
          f"(초기 {p.n_virus0} agent 대비 {R['virus_peak']/max(1,p.n_virus0):,.0f}배)")
    print(f"   · 초기 지수증식률           : {R['growth_rate_per_day']:.2f} /일  "
          f"-> 추정 R0 = {R['R0_est']:.1f}")
    print()
    print("  [감염세포]")
    print(f"   · 최대 감염세포 수          : {R['inf_peak']:,.0f} 칸  @ {d(R['inf_peak_day'])}")
    print(f"   · 감염세포 50% 감소시간     : {d(R['inf_t50'])}")
    print(f"   · 감염세포 제거시간         : {d(R['inf_clear_day'])}")
    print(f"   · 누적 감염 발생칸          : {R['total_infections']:,}")
    print(f"   · 표적세포 최대 소모율      : {R['max_target_depletion']*100:.1f}% "
          f"(종료시점 {R['target_depletion']*100:.1f}%)")
    print()
    print("  [선천면역]")
    print(f"   · 중성구 정점 / 소모량      : {R['neut_peak']:,.0f} agent @ {d(R['neut_peak_day'])}"
          f"  /  식균제거 {R['cleared_by_phago']:,} agent")
    print(f"   · 단핵구/대식세포 정점      : {R['mono_peak']:,.0f} agent @ {d(R['mono_peak_day'])}")
    print(f"   · NK세포 정점 / 살상량      : {R['nk_peak']:,.0f} agent @ {d(R['nk_peak_day'])}"
          f"  /  감염세포 {R['killed_by_nk']:,} 칸 제거")
    print(f"   · 사망 면역세포 총계        : {sim.dead_immune:,} agent")
    print(f"   · 보체 활성도 정점          : {R['comp_peak']*100:.1f}% @ {d(R['comp_peak_day'])}"
          f"  (보체 제거 {R['cleared_by_complement']:,} agent)")
    print(f"   · 인터페론 정점             : {R['ifn_peak']*500:.1f} pg/mL 상당 @ {d(R['ifn_peak_day'])}")
    print(f"   · 염증 정점 / 조직손상      : {R['inflam_peak']*100:.1f}% @ {d(R['inflam_peak_day'])}"
          f"  /  누적손상 {R['damage_final']:.3f}")
    print()
    print("  [후천면역]")
    print(f"   · 항원 림프절 도착          : {d(R['t_antigen_arrival'])}")
    print(f"   · 조력 T세포 활성화 시간    : {d(R['t_cd4_act'])}")
    print(f"   · 살해 T세포 활성화 시간    : {d(R['t_cd8_act'])}")
    print(f"   · B세포 활성화 시간         : {d(R['t_b_act'])}")
    print(f"   · 조직 진입 개시            : {d(R['t_emigration'])}")
    print(f"   · 항원특이 CTL 조직 정점    : {R['ctl_peak']:,.0f} agent @ {d(R['ctl_peak_day'])}")
    print(f"   · 항원특이 Th 조직 정점     : {R['th_peak']:,.0f} agent @ {d(R['th_peak_day'])}")
    print(f"   · 형질세포 정점             : {R['pc_peak']:,.0f} agent @ {d(R['pc_peak_day'])}")
    print(f"   · 항체 생성 시작(IgM/IgG)   : {d(R['t_igm'])} / {d(R['t_igg'])}")
    print(f"   · 최대 항체 농도            : {R['ab_peak']:.2f} ug/mL @ {d(R['ab_peak_day'])}")
    print(f"   · 항체에 의한 병원체 감소   : {R['cleared_by_antibody']:,} agent")
    print(f"   · CTL 감염세포 제거         : {R['killed_by_ctl']:,} 칸")
    print(f"   · 기억세포 형성량           : {R['memory_final']:,.0f} agent 상당 "
          f"(정점의 {R['memory_ratio']*100:.1f}%)")
    print("-" * 78)
    print()

    # ---- (3) 병원체 제거 기여도 ----
    tot = max(1, sim.cleared_virus)
    print("  [병원체 제거 기여도]")
    print(f"   · 비특이 제거(점액섬모 등)  : {R['cleared_by_decay']:>10,}  "
          f"({R['cleared_by_decay']/tot*100:5.1f}%)")
    print(f"   · 식세포 포식               : {R['cleared_by_phago']:>10,}  "
          f"({R['cleared_by_phago']/tot*100:5.1f}%)")
    print(f"   · 보체                      : {R['cleared_by_complement']:>10,}  "
          f"({R['cleared_by_complement']/tot*100:5.1f}%)")
    print(f"   · 항체 중화                 : {R['cleared_by_antibody']:>10,}  "
          f"({R['cleared_by_antibody']/tot*100:5.1f}%)")
    print()
    tot_i = max(1, sim.killed_infected + sim.apoptosis_infected)
    print("  [감염세포 제거 기여도]")
    print(f"   · NK세포 살상               : {R['killed_by_nk']:>10,}  "
          f"({R['killed_by_nk']/tot_i*100:5.1f}%)")
    print(f"   · 살해T세포(CTL) 살상       : {R['killed_by_ctl']:>10,}  "
          f"({R['killed_by_ctl']/tot_i*100:5.1f}%)")
    print(f"   · 바이러스 유도 세포자멸    : {R['apoptosis']:>10,}  "
          f"({R['apoptosis']/tot_i*100:5.1f}%)")
    print()

    # ---- (4) 이론(문헌) 대비 검증표 ----
    print("#" * 100)
    print("  [이론(문헌) 대비 검증표]  - 입력값이 아니라 '창발 결과'만 대조")
    print("#" * 100)
    print(f"{'검증 항목':<26}{'문헌 기준값':<42}{'시뮬레이션 결과':<24}{'판정'}")
    print("-" * 110)
    n_ok = n_near = n_bad = 0
    for ck in checks:
        print(f"{ck.item:<26}{ck.lit:<42}{ck.sim:<24}{ck.verdict}")
        print(f"{'':<26}└ 출처: "
              f"{REFS[ck.ref].short() if ck.ref in REFS else ck.ref}")
        if ck.verdict == "일치":
            n_ok += 1
        elif ck.verdict == "근접":
            n_near += 1
        elif ck.verdict == "불일치":
            n_bad += 1
    print("-" * 110)
    n_tot = n_ok + n_near + n_bad
    print(f"  종합: 일치 {n_ok} / 근접 {n_near} / 불일치 {n_bad}  "
          f"(판정대상 {n_tot}개, 일치+근접 = {(n_ok+n_near)/max(1,n_tot)*100:.0f}%)")
    print()

    # ---- (5) 무작위 이동 조건에 대한 결론 (선입견 없이 결과 기반) ----
    print("#" * 100)
    print("  [무작위 이동 조건 실험 결론 - 결과 기반]")
    print("#" * 100)
    cleared = R["virus_clear_day"] is not None
    print(f"   · 이동 방식                 : {sim.movement_mode.upper()} "
          f"(방향 = 균일난수, 위치/농도 경사 일체 미참조)")
    print(f"   · 이동 호출 횟수            : {sim.mover.n_calls:,}회, "
          f"누적 이동 agent {sim.mover.n_agents_moved:,}")
    if cleared:
        print(f"   · 감염 결과                 : 병원체 완전 제거 성공 "
              f"({d(R['virus_clear_day'])})")
    else:
        print(f"   · 감염 결과                 : 관찰기간({p.max_days}일) 내 "
              f"완전 제거 실패, 잔존 {R['virus_final']:,.0f} agent")
    print(f"   · CTL 접촉률                : {R['ctl_contact_per_day']:.2f} 회/CTL/일 "
          f"(무작위 조우 기반)")
    print(f"   · 조직 손상                 : 표적세포 최대 {R['max_target_depletion']*100:.1f}% 소모")
    print()
    print("   ※ 본 실행은 1회 시행 결과이다. 무작위 보행은 확률과정이므로")
    print("      결론 확정에는 반복시행(사양서 32항)이 필요하며,")
    print("      체계적 이동과의 우열은 동일 생물학 조건에서 이동함수만 교체한")
    print("      비교실험 이후에 판단한다.")
    print()
    if sim.virus.thinning_events:
        print(f"   ! 계산부하 상한 도달로 바이러스 agent 가중치 병합 "
              f"{sim.virus.thinning_events}회 발생 (1 agent = {sim.virus.thin_weight:.0f} 배 대표)")
        print()


# =============================================================================
# SECTION 15.  VISUALIZATION  (0.5초 갱신, 사양서 29항)
# =============================================================================


def run_interactive(sim: "ImmuneSimulation", ticks_per_frame: int = 12):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    try:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
    except Exception:
        pass

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_i = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[:, 2])
    ax_txt.axis("off")
    fig.suptitle("Influenza A ABM  |  RANDOM WALK immune search "
                 "(no gradient / no target tracking)", fontweight="bold")

    def upd(_):
        for _ in range(ticks_per_frame):
            if sim.tick >= sim.p.max_days * Scale.TICKS_PER_DAY:
                break
            sim.step()
        s = sim._snapshot()

        ax_map.clear()
        ax_map.set_xlim(0, Scale.GRID); ax_map.set_ylim(0, Scale.GRID)
        ax_map.set_title(f"Tissue  Day {s['day']:.2f}")
        inf = sim.env.inf_site[sim.env.inf_alive] if sim.env.inf_alive.size else np.empty(0, int)
        if inf.size:
            sub = inf[::max(1, inf.size // 4000)]
            ax_map.scatter(sub % Scale.GRID, sub // Scale.GRID, s=2,
                           c="magenta", alpha=.5, label=f"infected({s['infected']:,})")
        if sim.virus.n:
            k = max(1, sim.virus.n // 4000)
            ax_map.scatter(sim.virus.x[::k], sim.virus.y[::k], s=1.5, c="red",
                           alpha=.35, label=f"virus({s['virus']:,})")
        c = sim.cells
        for ctype, col, lab in ((NEUT, "royalblue", "neutrophil"),
                                (NK, "orange", "NK"),
                                (TC, "lime", "killerT")):
            m = np.flatnonzero(c.mask(ctype))
            if m.size:
                m = m[::max(1, m.size // 800)]
                ax_map.scatter(c.x[m], c.y[m], s=4, c=col, alpha=.5, label=lab)
        ax_map.legend(loc="upper right", fontsize=7)

        d = _days(sim.history)
        ax_v.clear()
        ax_v.plot(d, _series(sim.history, "virus"), "r-", label="virus")
        ax_v.plot(d, _series(sim.history, "infected"), "m--", label="infected cells")
        ax_v.set_yscale("symlog"); ax_v.legend(fontsize=8); ax_v.grid(alpha=.3)
        ax_v.set_xlabel("day")

        ax_i.clear()
        ax_i.plot(d, _series(sim.history, "neutrophil"), label="neutrophil")
        ax_i.plot(d, _series(sim.history, "nk"), label="NK")
        ax_i.plot(d, _series(sim.history, "killer_t_spec"), label="specific CTL")
        ax_i.plot(d, _series(sim.history, "antibody") * 100, label="Ab x100")
        ax_i.plot(d, _series(sim.history, "interferon") * 5e4, label="IFN(scaled)")
        ax_i.legend(fontsize=7); ax_i.grid(alpha=.3); ax_i.set_xlabel("day")

        ax_txt.clear(); ax_txt.axis("off")
        ax_txt.text(0, 1, ImmuneSimulation.format_block(s, "LIVE"),
                    fontsize=6.5, family="monospace", va="top")

    ani = animation.FuncAnimation(fig, upd, frames=100000, interval=500,
                                  repeat=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    return ani


def save_figures(sim: "ImmuneSimulation", outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = sim.history
    d = _days(h)
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Influenza A ABM - RANDOM WALK immune search (single run)",
                 fontweight="bold")

    a = ax[0, 0]
    a.plot(d, _series(h, "virus"), "r-", lw=1.6, label="free virus (agents)")
    a.plot(d, _series(h, "infected"), "m--", lw=1.4, label="infected cells")
    a.plot(d, _series(h, "healthy"), "g-", lw=1.0, alpha=.6, label="healthy epithelium")
    a.set_yscale("log"); a.set_ylim(1, None)
    a.set_xlabel("day"); a.set_title("Infection dynamics"); a.grid(alpha=.3)
    a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(d, _series(h, "neutrophil"), label="neutrophil")
    a.plot(d, _series(h, "monocyte"), label="monocyte/macrophage")
    a.plot(d, _series(h, "nk"), label="NK")
    a.set_xlabel("day"); a.set_title("Innate immune cells"); a.grid(alpha=.3)
    a.legend(fontsize=8)

    a = ax[1, 0]
    a.plot(d, _series(h, "killer_t_spec"), "g-", label="specific CTL (tissue)")
    a.plot(d, _series(h, "helper_t_spec"), "b-", label="specific Th (tissue)")
    a.plot(d, _series(h, "ln_cd8"), "g--", alpha=.5, label="CD8 in lymph node")
    a.set_yscale("symlog")
    a.set_xlabel("day"); a.set_title("Adaptive cellular response"); a.grid(alpha=.3)
    a.legend(fontsize=8)

    a = ax[1, 1]
    a.plot(d, _series(h, "antibody"), "b-", label="antibody (ug/mL)")
    a.plot(d, _series(h, "interferon") * 500, "orange", label="IFN (pg/mL eq.)")
    a.plot(d, _series(h, "complement") * 100, "purple", label="complement (%)")
    a.plot(d, _series(h, "inflammation") * 100, "r--", label="inflammation (%)")
    a.set_xlabel("day"); a.set_title("Humoral factors"); a.grid(alpha=.3)
    a.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(outdir, "immune_random_walk_result.png")
    fig.savefig(path, dpi=130)
    return path


# =============================================================================
# SECTION 16.  MAIN
# =============================================================================


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="면역 무작위이동 ABM (Influenza A)")
    ap.add_argument("--mode", default="random", choices=["random", "systematic"])
    ap.add_argument("--seed", type=int, default=Params.seed)
    ap.add_argument("--days", type=int, default=Params.max_days)
    ap.add_argument("--runs", type=int, default=1,
                    help="반복 시행 횟수 (이번 실험은 1회)")
    ap.add_argument("--live", action="store_true", help="0.5초 갱신 실시간 화면")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--no-params", action="store_true")
    args = ap.parse_args(argv)

    print("=" * 100)
    print("  면역 시뮬레이션 - 무작위 이동(RANDOM WALK) 조건")
    print("  Influenza A / 상기도 상피 슬랩 + 배액 림프절 / kappa = 1/1000")
    print(f"  격자 {Scale.GRID}x{Scale.GRID} = {Scale.SITES:,}칸 "
          f"(상피 {Scale.N_TARGET_SITES:,}칸), 1 tick = {Scale.DT_MIN:.0f}분, "
          f"1일 = {Scale.TICKS_PER_DAY} tick")
    print(f"  seed = {args.seed}, 시행 횟수 = {args.runs}")
    print("=" * 100)

    if not args.no_params:
        print_param_table()

    p = Params()
    p.max_days = args.days
    p.seed = args.seed

    sim = ImmuneSimulation(p, movement_mode=args.mode, seed=args.seed)

    if args.live:
        run_interactive(sim)
        return 0

    sim.run()
    R = analyze(sim)
    checks = validate(sim, R)
    print_final_report(sim, R, checks)

    try:
        path = save_figures(sim, args.outdir)
        print(f"  [그래프 저장] {path}")
    except Exception as e:
        print(f"  (그래프 저장 실패: {e})")

    with open(os.path.join(args.outdir, "immune_random_walk_history.json"),
              "w", encoding="utf-8") as f:
        json.dump({"daily": sim.daily, "summary": {k: (None if (isinstance(v, float)
                   and math.isnan(v)) else v) for k, v in R.items()}},
                  f, ensure_ascii=False, indent=1)
    print(f"  [원자료 저장] immune_random_walk_history.json")
    print(f"  [실행시간] {sim.wall_time:.1f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
