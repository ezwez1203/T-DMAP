import pandas as pd
import numpy as np
import os
from config import Config
from collections import Counter

def check_qualitative_reasoning():
    print("🔍 TDA 질적 특성 검증 (Qualitative Check)...")
    
    # 1. 파일 로드
    try:
        df_loop = pd.read_csv(os.path.join(Config.TEXT_OUTPUT_DIR, Config.FILE_LOOP_PATIENTS))
        df_super = pd.read_csv(os.path.join(Config.TEXT_OUTPUT_DIR, Config.FILE_SUPER_RESPONDERS))
        
        # 원본 데이터 로드
        df_spec = pd.read_csv(os.path.join(Config.DATA_DIR, "ST200_renamed.csv"))
        df_treat = pd.read_csv(os.path.join(Config.DATA_DIR, "ST530_renamed.csv"))
    except FileNotFoundError:
        print("❌ 필수 파일이 없습니다. 06_run_analysis.py를 먼저 실행하세요.")
        return

    # 2. 데이터 병합 (환자 ID 기준)
    print("⏳ 데이터 병합 및 매핑 중...")
    
    # 2-1. 환자별 주상병(MSICK_CD) 가져오기 (가장 빈도 높은 상병)
    # 한 환자가 여러 상병을 가질 수 있으므로, 가장 자주 등장한 상병 하나를 대표로 선정
    spec_main = df_spec.groupby("SPEC_ID_SNO")['MSICK_CD'].agg(lambda x: x.mode()[0] if not x.mode().empty else None).reset_index()
    
    # 2-2. 환자별 주사용 약물(GNL_NM_CD) 가져오기 (가장 많이 쓴 약)
    treat_main = df_treat.groupby("SPEC_ID_SNO")['GNL_NM_CD'].agg(lambda x: x.mode()[0] if not x.mode().empty else "No_Drug").reset_index()
    
    # 2-3. 통합
    df_final = spec_main.merge(treat_main, on="SPEC_ID_SNO", how="left").fillna("No_Drug")
    
    # 3. 그룹 라벨링
    loop_ids = set(df_loop['SPEC_ID_SNO'])
    super_ids = set(df_super['SPEC_ID_SNO'])
    
    def get_group(pid):
        if pid in loop_ids: return "1. Loop (악순환)"
        if pid in super_ids: return "2. Super (모범)"
        return "3. Normal (일반)"
    
    df_final['Group'] = df_final['SPEC_ID_SNO'].apply(get_group)
    
    # 4. 그룹별 최빈 질병/약물 분석 함수
    def get_top_k(series, k=3):
        counts = series.value_counts()
        top_k = counts.head(k).index.tolist()
        # 보기 좋게 문자열로 변환 "A, B, C"
        return ", ".join([str(x) for x in top_k])

    # 5. 결과 출력
    print("\n📊 [그룹별 핵심 질병 및 약물 패턴]")
    print("="*100)
    
    # 그룹별로 묶어서 Top 3 뽑기
    summary = df_final.groupby("Group").agg({
        "SPEC_ID_SNO": "count",
        "MSICK_CD": lambda x: get_top_k(x, 3),   # 가장 흔한 질병 3개
        "GNL_NM_CD": lambda x: get_top_k(x, 3)   # 가장 흔한 약물 3개
    })
    
    summary.columns = ['인원수', '주요 질병 (Top 3)', '주요 약물 (Top 3)']
    
    # 보기 좋게 출력
    pd.set_option('display.max_colwidth', None) # 컬럼 내용 안 잘리게
    print(summary)
    print("="*100)
    
    # 6. 해석
    print("\n💡 [분석 포인트]")
    print(" 1. Loop 그룹의 '주요 약물'이 항생제/스테로이드 계열인지 확인하세요.")
    print(" 2. Super 그룹이 Loop와 같은 질병('J20' 등)인데 '주요 약물'이 다른지 보세요.")
    print("    -> 만약 질병은 같은데 약이 다르다면, 그 약이 탈출의 열쇠(Key)입니다!")

if __name__ == "__main__":
    check_qualitative_reasoning()