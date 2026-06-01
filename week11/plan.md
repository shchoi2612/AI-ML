# Week 11 제출 작업 plan.md

> **제출 과제**: "AI for science" YouTube 강의 시청 + week11(양자역학 시뮬레이션) 문제 풀이를 정리하여 제출.
> **제출 위치**: `/home/dullear/aicoursework/week11/`
> **저장소**: https://github.com/shchoi2612/AI-ML (branch: main)
> **제출자**: 물리학과 / 최상현 / 202312162

## 작업 항목

### 1. AI for Science 강의 요약 (Chris Bishop, Microsoft)
- 강의 핵심을 `README.md`에 정리:
  - 과학의 수학적 기술과 시뮬레이션 계산 비용의 한계
  - AI 에뮬레이터(Emulator) 도입 → 수천 배 가속
  - No Free Lunch & 물리적 귀납 편향(슈뢰딩거 방정식 등 inductive bias) 통합
  - 적용 사례: 기상예측(파운데이션 모델), 재료/분자(MatterGen, Skala/DFT), 생물/단백질 동역학
  - 결론: AI 에뮬레이터가 과학적 발견을 가속, 실험-시뮬레이션 간극을 메움
- week11 양자역학 풀이(슈뢰딩거 방정식 수치해)와의 연결고리 명시
  → 강의가 말한 "물리 법칙을 모델에 넣는다"는 아이디어의 출발점이 곧 이번 과제의 슈뢰딩거 방정식 수치해.

### 2. week11 양자역학 문제 풀이 (이미 구현 완료)
- `05_h2plus.py` — H₂⁺ 전자 파동함수 (LCAO 변분 + 3D 유한차분)
- `06_helium.py` — 중성 He 다체 문제 (변분법 + 변분 몬테카를로 VMC)
- 풀이 접근법 상세: `solution_approach.md`
- 원본 문제: `week11_problem.md`
- 결과 그림: `outputs/` (3장)

### 3. 제출 문서 작성
- `README.md` 하나에 모두 정리: 학생 정보 + git 주소 + 강의 요약 + week11 풀이 결과.
- 이 `README.md`가 제출용 메인 md 파일.

### 4. Git 제출
- `git add week11/` → `git commit` → `git push origin main`
- 원격: https://github.com/shchoi2612/AI-ML.git

## 산출물 체크리스트
- [x] `plan.md` (이 파일)
- [x] `05_h2plus.py`, `06_helium.py`
- [x] `outputs/05_h2plus_energy_curve.png`, `05_h2plus_orbitals.png`, `06_he_vmc.png`
- [x] `week11_problem.md`, `solution_approach.md`
- [x] `README.md` (강의 요약 + git 주소 + 학번/이름)
- [x] git push
