"""
01. Three-Body Problem Simulation (Fixed Version)
3체 문제 시뮬레이션 (수정 버전)

주요 수정사항:
1. 질량중심 좌표계로 변환 (운동량 보존)
2. 태양-지구-달 시스템 스케일 수정
3. 혼돈 시스템 초기 조건 개선
4. 더 정확한 에너지 계산
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정 함수
def set_korean_font():
    """시스템에 설치된 한글 폰트를 자동으로 찾아 설정"""
    font_list = [f.name for f in fm.fontManager.ttflist]
    if 'Malgun Gothic' in font_list:
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif 'Gulim' in font_list:
        plt.rcParams['font.family'] = 'Gulim'
    elif 'Batang' in font_list:
        plt.rcParams['font.family'] = 'Batang'
    elif 'AppleGothic' in font_list:
        plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Three-Body Problem Simulation (Fixed)")
print("="*70)

# 물리 상수 (기본값, 시나리오별로 변경됨)
# G = 1.0 

# ============================================================================
# RK4 적분
# ============================================================================

def rk4_step(f, y, t, dt):
    """Runge-Kutta 4차 방법"""
    k1 = f(y, t)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(y + dt*k3, t + dt)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def make_three_body_derivatives(masses, G=1.0):
    """3체 문제의 운동 방정식을 생성하는 팩토리 함수"""
    m1, m2, m3 = masses
    
    def three_body_derivatives(y, t):
        """3체 문제의 운동 방정식"""
        # 위치와 속도 추출
        r1 = np.array([y[0], y[1]])
        v1 = np.array([y[2], y[3]])
        r2 = np.array([y[4], y[5]])
        v2 = np.array([y[6], y[7]])
        r3 = np.array([y[8], y[9]])
        v3 = np.array([y[10], y[11]])
        
        # 거리 벡터
        r12 = r2 - r1
        r13 = r3 - r1
        r23 = r3 - r2
        
        # 거리 (충돌 방지를 위한 최소 거리)
        d12 = max(np.linalg.norm(r12), 1e-6)
        d13 = max(np.linalg.norm(r13), 1e-6)
        d23 = max(np.linalg.norm(r23), 1e-6)
        
        # 가속도
        a1 = G * m2 / d12**3 * r12 + G * m3 / d13**3 * r13
        a2 = -G * m1 / d12**3 * r12 + G * m3 / d23**3 * r23
        a3 = -G * m1 / d13**3 * r13 - G * m2 / d23**3 * r23
        
        # dy/dt
        return np.array([
            v1[0], v1[1], a1[0], a1[1],
            v2[0], v2[1], a2[0], a2[1],
            v3[0], v3[1], a3[0], a3[1]
        ])
    
    return three_body_derivatives

def transform_to_com_frame(y0, masses):
    """질량중심 좌표계로 변환"""
    m1, m2, m3 = masses
    M_total = m1 + m2 + m3
    
    # 현재 위치와 속도
    r1 = y0[0:2]
    v1 = y0[2:4]
    r2 = y0[4:6]
    v2 = y0[6:8]
    r3 = y0[8:10]
    v3 = y0[10:12]
    
    # 질량중심
    r_com = (m1*r1 + m2*r2 + m3*r3) / M_total
    v_com = (m1*v1 + m2*v2 + m3*v3) / M_total
    
    # 질량중심 좌표계로 변환
    y0_com = np.array([
        r1[0] - r_com[0], r1[1] - r_com[1], v1[0] - v_com[0], v1[1] - v_com[1],
        r2[0] - r_com[0], r2[1] - r_com[1], v2[0] - v_com[0], v2[1] - v_com[1],
        r3[0] - r_com[0], r3[1] - r_com[1], v3[0] - v_com[0], v3[1] - v_com[1]
    ])
    
    return y0_com

def simulate_three_body(y0, t_max, dt, masses, G=1.0):
    """3체 시뮬레이션"""
    derivs_func = make_three_body_derivatives(masses, G)
    
    n_steps = int(t_max / dt)
    t_array = np.zeros(n_steps)
    trajectory = np.zeros((n_steps, 12))
    
    t = 0
    y = y0.copy()
    
    for i in range(n_steps):
        t_array[i] = t
        trajectory[i] = y
        y = rk4_step(derivs_func, y, t, dt)
        t += dt
    
    return t_array, trajectory

def calculate_energy(traj, masses, G=1.0):
    """총 에너지 계산"""
    m1, m2, m3 = masses
    
    # 위치
    r1 = traj[:, 0:2]
    r2 = traj[:, 4:6]
    r3 = traj[:, 8:10]
    
    # 속도
    v1 = traj[:, 2:4]
    v2 = traj[:, 6:8]
    v3 = traj[:, 10:12]
    
    # 운동 에너지
    T = 0.5 * m1 * np.sum(v1**2, axis=1) + \
        0.5 * m2 * np.sum(v2**2, axis=1) + \
        0.5 * m3 * np.sum(v3**2, axis=1)
    
    # 위치 에너지
    d12 = np.linalg.norm(r1 - r2, axis=1)
    d13 = np.linalg.norm(r1 - r3, axis=1)
    d23 = np.linalg.norm(r2 - r3, axis=1)
    
    V = -G * (m1*m2/d12 + m1*m3/d13 + m2*m3/d23)
    
    return T + V

def calculate_momentum(traj, masses):
    """총 운동량 계산"""
    m1, m2, m3 = masses
    
    v1 = traj[:, 2:4]
    v2 = traj[:, 6:8]
    v3 = traj[:, 10:12]
    
    p_total = m1 * v1 + m2 * v2 + m3 * v3
    return p_total

# ============================================================================
# Scenario 1: Figure-8 궤도
# ============================================================================

print("\n" + "="*70)
print("Scenario 1: Figure-8 Orbit")
print("="*70)

masses_fig8 = [1.0, 1.0, 1.0]

# Figure-8 초기 조건 (정밀한 값)
x1_0 = -0.97000436
y1_0 = 0.24308753
vx1_0 = 0.4662036850
vy1_0 = 0.4323657300

x2_0 = -x1_0
y2_0 = -y1_0
vx2_0 = vx1_0
vy2_0 = vy1_0

x3_0 = 0.0
y3_0 = 0.0
vx3_0 = -2 * vx1_0
vy3_0 = -2 * vy1_0

y0_fig8 = np.array([
    x1_0, y1_0, vx1_0, vy1_0,
    x2_0, y2_0, vx2_0, vy2_0,
    x3_0, y3_0, vx3_0, vy3_0
])

# 질량중심 좌표계로 변환
y0_fig8 = transform_to_com_frame(y0_fig8, masses_fig8)

print("시뮬레이션 실행 중...")
t_max_fig8 = 6.32591398  # 정확한 주기
dt_fig8 = 0.0001  # 정밀도 향상을 위해 dt 축소

t_fig8, traj_fig8 = simulate_three_body(y0_fig8, t_max_fig8, dt_fig8, masses_fig8, G=1.0)

E_fig8 = calculate_energy(traj_fig8, masses_fig8, G=1.0)
p_fig8 = calculate_momentum(traj_fig8, masses_fig8)

print(f"[OK] 완료: {len(t_fig8)} 시간 단계")
print(f"에너지 보존: ΔE/E0 = {(E_fig8.max() - E_fig8.min())/abs(E_fig8[0])*100:.6f}%")
print(f"운동량 보존: |p| < {np.max(np.linalg.norm(p_fig8, axis=1)):.2e}")

# ============================================================================
# Scenario 2: 태양-지구-달 시스템 (수정)
# ============================================================================

print("\n" + "="*70)
print("Scenario 2: Sun-Earth-Moon System (Fixed)")
print("="*70)

# 실제 질량비 사용 (태양 질량 = 1)
M_sun = 1.0
M_earth = 3.0e-6  # 태양 대비 지구 질량
M_moon = 3.7e-8   # 태양 대비 달 질량

masses_sem = [M_sun, M_earth, M_moon]

# 초기 조건 (단위: AU, year)
# 지구-달 거리: 0.00257 AU
# 지구 공전 속도: 2π AU/year
# 달 공전 속도 (지구 주변): 2π * 13.4 AU/year

# 태양 (원점에서 약간 이동 - 질량중심을 위해)
r_sun = np.array([0.0, 0.0])
v_sun = np.array([0.0, 0.0])

# 지구 (1 AU)
r_earth = np.array([1.0, 0.0])
v_earth_orbit = 2 * np.pi  # 공전 속도
v_earth = np.array([0.0, v_earth_orbit])

# 달 (지구로부터 0.00257 AU)
moon_distance = 0.00257
r_moon = np.array([1.0 + moon_distance, 0.0])
omega_moon = 2 * np.pi * 13.4  # 달의 각속도 (rad/year)
v_moon_rel = moon_distance * omega_moon  # 달의 공전 속도 (v = r * omega)
v_moon = np.array([0.0, v_earth_orbit + v_moon_rel])

y0_sem = np.array([
    r_sun[0], r_sun[1], v_sun[0], v_sun[1],
    r_earth[0], r_earth[1], v_earth[0], v_earth[1],
    r_moon[0], r_moon[1], v_moon[0], v_moon[1]
])

# 질량중심 좌표계로 변환
y0_sem = transform_to_com_frame(y0_sem, masses_sem)

print("시뮬레이션 실행 중...")
t_max_sem = 1.0  # 1년
dt_sem = 0.0005

# 천문 단위계에서 G = 4π²
G_sem = 4 * np.pi**2

t_sem, traj_sem = simulate_three_body(y0_sem, t_max_sem, dt_sem, masses_sem, G=G_sem)

E_sem = calculate_energy(traj_sem, masses_sem, G=G_sem)
p_sem = calculate_momentum(traj_sem, masses_sem)

print(f"[OK] 완료: {len(t_sem)} 시간 단계")
print(f"에너지 보존: ΔE/E0 = {(E_sem.max() - E_sem.min())/abs(E_sem[0])*100:.6f}%")
print(f"운동량 보존: |p| < {np.max(np.linalg.norm(p_sem, axis=1)):.2e}")

# ============================================================================
# Scenario 3: 안정적인 3체 시스템
# ============================================================================

print("\n" + "="*70)
print("Scenario 3: Stable Three-Body (Lagrange L4)")
print("="*70)

masses_stable = [1.0, 1.0, 0.001]  # 두 개의 큰 질량 + 작은 질량

# 라그랑주 L4 점 근처 (정삼각형 배치)
# 두 질량이 거리 1만큼 떨어져 회전할 때의 속도 v = sqrt(GM / (4r_cm)) -> v = sqrt(1*1/2) = 0.707...
# 시계 방향 회전 가정
v_orbital = np.sqrt(1.0 / 2.0)  # 약 0.70710678
r_L4 = np.sqrt(3) / 2.0
v_L4 = r_L4 * np.sqrt(2.0)  # v = r * omega, omega = sqrt(2)

y0_stable = np.array([
    -0.5, 0.0, 0.0, v_orbital,      # Body 1 (왼쪽, 위로 이동)
    0.5, 0.0, 0.0, -v_orbital,      # Body 2 (오른쪽, 아래로 이동)
    0.0, r_L4, v_L4, 0.0            # Body 3 (위쪽, 오른쪽으로 이동)
])

# 질량중심 좌표계로 변환
y0_stable = transform_to_com_frame(y0_stable, masses_stable)

print("시뮬레이션 실행 중...")
t_max_stable = 30.0
dt_stable = 0.005

t_stable, traj_stable = simulate_three_body(y0_stable, t_max_stable, dt_stable, masses_stable, G=1.0)

E_stable = calculate_energy(traj_stable, masses_stable, G=1.0)
p_stable = calculate_momentum(traj_stable, masses_stable)

print(f"[OK] 완료: {len(t_stable)} 시간 단계")
print(f"에너지 보존: ΔE/E0 = {(E_stable.max() - E_stable.min())/abs(E_stable[0])*100:.6f}%")
print(f"운동량 보존: |p| < {np.max(np.linalg.norm(p_stable, axis=1)):.2e}")

# ============================================================================
# 시각화
# ============================================================================

print("\n그래프 생성 중...")

# 그림 1: Figure-8 궤도
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))

axes1[0].plot(traj_fig8[:, 0], traj_fig8[:, 1], 'r-', linewidth=1.5, alpha=0.7, label='천체 1')
axes1[0].plot(traj_fig8[:, 4], traj_fig8[:, 5], 'g-', linewidth=1.5, alpha=0.7, label='천체 2')
axes1[0].plot(traj_fig8[:, 8], traj_fig8[:, 9], 'b-', linewidth=1.5, alpha=0.7, label='천체 3')

axes1[0].plot(traj_fig8[0, 0], traj_fig8[0, 1], 'ro', markersize=10, markeredgecolor='black')
axes1[0].plot(traj_fig8[0, 4], traj_fig8[0, 5], 'go', markersize=10, markeredgecolor='black')
axes1[0].plot(traj_fig8[0, 8], traj_fig8[0, 9], 'bo', markersize=10, markeredgecolor='black')

axes1[0].set_xlabel('x', fontsize=12, fontweight='bold')
axes1[0].set_ylabel('y', fontsize=12, fontweight='bold')
axes1[0].set_title('Figure-8 궤도 (1 주기)', fontsize=13, fontweight='bold')
axes1[0].legend(fontsize=10)
axes1[0].grid(True, alpha=0.3)
axes1[0].axis('equal')

axes1[1].plot(t_fig8, (E_fig8 - E_fig8[0])/abs(E_fig8[0])*100, 'purple', linewidth=2)
axes1[1].axhline(0, color='k', linestyle='--', alpha=0.5)
axes1[1].set_xlabel('시간', fontsize=12, fontweight='bold')
axes1[1].set_ylabel('에너지 오차 (%)', fontsize=12, fontweight='bold')
axes1[1].set_title('에너지 보존', fontsize=13, fontweight='bold')
axes1[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_figure8_orbit_fixed.png', dpi=150, bbox_inches='tight')
print(f"[OK] {output_dir}/01_figure8_orbit_fixed.png")
plt.close()

# 그림 2: 태양-지구-달
fig2 = plt.figure(figsize=(16, 10))
gs2 = GridSpec(2, 2, figure=fig2, hspace=0.3, wspace=0.3)

# 전체 궤도
ax21 = fig2.add_subplot(gs2[0, :])
ax21.plot(traj_sem[:, 0], traj_sem[:, 1], 'yo', markersize=12, 
         markeredgecolor='orange', markeredgewidth=2, label='태양')
ax21.plot(traj_sem[:, 4], traj_sem[:, 5], 'b-', linewidth=1.5, alpha=0.7, label='지구')
ax21.plot(traj_sem[:, 8], traj_sem[:, 9], 'gray', linewidth=0.8, alpha=0.5, label='달')
ax21.set_xlabel('x (AU)', fontsize=12, fontweight='bold')
ax21.set_ylabel('y (AU)', fontsize=12, fontweight='bold')
ax21.set_title('태양-지구-달 시스템 (1년)', fontsize=13, fontweight='bold')
ax21.legend(fontsize=10)
ax21.grid(True, alpha=0.3)
ax21.axis('equal')

# 지구-달 확대
# 지구-달 확대 (상대 좌표로 변경)
ax22 = fig2.add_subplot(gs2[1, 0])
r_earth_t = traj_sem[:, 4:6]
r_moon_t = traj_sem[:, 8:10]
rel_moon = r_moon_t - r_earth_t  # 지구에 대한 달의 상대 위치

ax22.plot(rel_moon[:, 0], rel_moon[:, 1], 'gray', linewidth=1.5, alpha=0.7, label='달 궤도')
ax22.plot(0, 0, 'b.', markersize=15, label='지구')  # 지구를 중심에 고정

ax22.set_xlabel('x (AU) - 지구 기준', fontsize=11, fontweight='bold')
ax22.set_ylabel('y (AU) - 지구 기준', fontsize=11, fontweight='bold')
ax22.set_title('지구 중심 달 궤도', fontsize=12, fontweight='bold')
ax22.legend(fontsize=9)
ax22.grid(True, alpha=0.3)
ax22.axis('equal')

# 에너지
ax23 = fig2.add_subplot(gs2[1, 1])
ax23.plot(t_sem, (E_sem - E_sem[0])/abs(E_sem[0])*100, 'purple', linewidth=2)
ax23.axhline(0, color='k', linestyle='--', alpha=0.5)
ax23.set_xlabel('시간 (년)', fontsize=11, fontweight='bold')
ax23.set_ylabel('에너지 오차 (%)', fontsize=11, fontweight='bold')
ax23.set_title('에너지 보존', fontsize=12, fontweight='bold')
ax23.grid(True, alpha=0.3)

plt.suptitle('태양-지구-달 3체 시스템', fontsize=16, fontweight='bold')
plt.savefig(f'{output_dir}/01_sun_earth_moon_fixed.png', dpi=150, bbox_inches='tight')
print(f"[OK] {output_dir}/01_sun_earth_moon_fixed.png")
plt.close()

# 그림 3: 안정적 3체
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

axes3[0].plot(traj_stable[:, 0], traj_stable[:, 1], 'r-', linewidth=1, alpha=0.7, label='천체 1')
axes3[0].plot(traj_stable[:, 4], traj_stable[:, 5], 'g-', linewidth=1, alpha=0.7, label='천체 2')
axes3[0].plot(traj_stable[:, 8], traj_stable[:, 9], 'b-', linewidth=1, alpha=0.7, label='천체 3 (소형)')

axes3[0].plot(traj_stable[0, 0], traj_stable[0, 1], 'ro', markersize=10, markeredgecolor='black')
axes3[0].plot(traj_stable[0, 4], traj_stable[0, 5], 'go', markersize=10, markeredgecolor='black')
axes3[0].plot(traj_stable[0, 8], traj_stable[0, 9], 'bo', markersize=8, markeredgecolor='black')

axes3[0].set_xlabel('x', fontsize=12, fontweight='bold')
axes3[0].set_ylabel('y', fontsize=12, fontweight='bold')
axes3[0].set_title('안정적 3체 (라그랑주 L4형)', fontsize=13, fontweight='bold')
axes3[0].legend(fontsize=10)
axes3[0].grid(True, alpha=0.3)
axes3[0].axis('equal')

axes3[1].plot(t_stable, (E_stable - E_stable[0])/abs(E_stable[0])*100, 'purple', linewidth=2)
axes3[1].axhline(0, color='k', linestyle='--', alpha=0.5)
axes3[1].set_xlabel('시간', fontsize=12, fontweight='bold')
axes3[1].set_ylabel('에너지 오차 (%)', fontsize=12, fontweight='bold')
axes3[1].set_title('에너지 보존', fontsize=13, fontweight='bold')
axes3[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_stable_three_body.png', dpi=150, bbox_inches='tight')
print(f"[OK] {output_dir}/01_stable_three_body.png")
plt.close()

# 그림 4: 보존량 분석
fig4, axes4 = plt.subplots(2, 3, figsize=(16, 10))

scenarios = [
    ('Figure-8', t_fig8, traj_fig8, E_fig8, masses_fig8),
    ('태양-지구-달', t_sem, traj_sem, E_sem, masses_sem),
    ('안정 3체', t_stable, traj_stable, E_stable, masses_stable)
]

for i, (name, t, traj, E, masses) in enumerate(scenarios):
    # 에너지
    ax_e = axes4[0, i]
    ax_e.plot(t, (E - E[0])/abs(E[0])*100, 'b-', linewidth=2)
    ax_e.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax_e.set_xlabel('시간', fontsize=10, fontweight='bold')
    ax_e.set_ylabel('에너지 오차 (%)', fontsize=10, fontweight='bold')
    ax_e.set_title(f'{name}: 에너지', fontsize=11, fontweight='bold')
    ax_e.grid(True, alpha=0.3)
    
    # 운동량
    ax_p = axes4[1, i]
    p = calculate_momentum(traj, masses)
    p_mag = np.linalg.norm(p, axis=1)
    ax_p.plot(t, p_mag, 'r-', linewidth=2)
    ax_p.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax_p.set_xlabel('시간', fontsize=10, fontweight='bold')
    ax_p.set_ylabel('총 운동량', fontsize=10, fontweight='bold')
    ax_p.set_title(f'{name}: 운동량', fontsize=11, fontweight='bold')
    ax_p.grid(True, alpha=0.3)
    ax_p.set_ylim([-0.1, max(p_mag.max(), 0.1)*1.5])

plt.suptitle('보존 법칙 분석', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_conservation_analysis_fixed.png', dpi=150, bbox_inches='tight')
print(f"[OK] {output_dir}/01_conservation_analysis_fixed.png")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n주요 개선사항:")
print("  1. 질량중심 좌표계 사용 → 운동량이 0으로 보존")
print("  2. 태양-지구-달 시스템 스케일 수정 → 태양이 거의 정지")
print("  3. 혼돈 시스템 대신 안정적 라그랑주 L4형 사용")
print("  4. 에너지 보존 정확도 향상")
print("\n생성된 파일:")
print(f"  - 01_figure8_orbit_fixed.png")
print(f"  - 01_sun_earth_moon_fixed.png")
print(f"  - 01_stable_three_body.png")
print(f"  - 01_conservation_analysis_fixed.png")