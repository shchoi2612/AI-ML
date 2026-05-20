"""
03. Chaotic Double Pendulum Simulation
혼돈 이중 진자 시뮬레이션

이중 진자는 고전 역학에서 가장 유명한 혼돈 시스템입니다.
초기 조건에 매우 민감하게 반응하여 "나비 효과"를 보여줍니다.

특징:
- 결정론적 시스템 (방정식이 명확)
- 하지만 예측 불가능 (초기 조건에 극도로 민감)
- 긴 시간 후에는 완전히 다른 궤적
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# 출력 디렉토리 확인
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("="*70)
print("Chaotic Double Pendulum Simulation")
print("="*70)

# 물리 파라미터
g = 9.81  # 중력 가속도 (m/s²)
L1 = 1.0  # 첫 번째 진자 길이 (m)
L2 = 1.0  # 두 번째 진자 길이 (m)
m1 = 1.0  # 첫 번째 질량 (kg)
m2 = 1.0  # 두 번째 질량 (kg)

print(f"\n진자 파라미터:")
print(f"  L1 = {L1} m, L2 = {L2} m")
print(f"  m1 = {m1} kg, m2 = {m2} kg")
print(f"  g = {g} m/s²")

# ============================================================================
# 이중 진자 운동 방정식
# ============================================================================

def double_pendulum_derivs(y, t):
    """
    이중 진자의 운동 방정식
    
    y = [θ1, ω1, θ2, ω2]
    
    복잡한 결합 미분 방정식
    (라그랑지안 역학에서 유도)
    """
    theta1, omega1, theta2, omega2 = y
    
    # 각도 차이
    delta = theta2 - theta1
    
    # 분모 항 (특이점 방지)
    den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    den2 = (L2 / L1) * den1
    
    # dθ/dt = ω
    dtheta1 = omega1
    dtheta2 = omega2
    
    # dω/dt (복잡한 결합 방정식)
    domega1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
               m2 * g * np.sin(theta2) * np.cos(delta) +
               m2 * L2 * omega2**2 * np.sin(delta) -
               (m1 + m2) * g * np.sin(theta1)) / den1
    
    domega2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
               (m1 + m2) * (g * np.sin(theta1) * np.cos(delta) -
                           L1 * omega1**2 * np.sin(delta) -
                           g * np.sin(theta2))) / den2
    
    return np.array([dtheta1, domega1, dtheta2, domega2])

def rk4_step(f, y, t, dt):
    """Runge-Kutta 4차 방법"""
    k1 = f(y, t)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(y + dt*k3, t + dt)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_double_pendulum(theta1_0, omega1_0, theta2_0, omega2_0, t_max, dt):
    """
    이중 진자 시뮬레이션
    
    Parameters:
    -----------
    theta1_0, theta2_0 : float
        초기 각도 (radians)
    omega1_0, omega2_0 : float
        초기 각속도 (rad/s)
    t_max : float
        시뮬레이션 시간 (s)
    dt : float
        시간 간격 (s)
    
    Returns:
    --------
    t, theta1, omega1, theta2, omega2, x1, y1, x2, y2
    """
    y0 = np.array([theta1_0, omega1_0, theta2_0, omega2_0])
    
    n_steps = int(t_max / dt)
    t_array = np.zeros(n_steps)
    states = np.zeros((n_steps, 4))
    
    t = 0
    y = y0.copy()
    
    for i in range(n_steps):
        t_array[i] = t
        states[i] = y
        y = rk4_step(double_pendulum_derivs, y, t, dt)
        t += dt
    
    theta1 = states[:, 0]
    omega1 = states[:, 1]
    theta2 = states[:, 2]
    omega2 = states[:, 3]
    
    # 카테시안 좌표 계산
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    
    return t_array, theta1, omega1, theta2, omega2, x1, y1, x2, y2

# ============================================================================
# 혼돈 시연: 매우 작은 초기 조건 차이
# ============================================================================

print("\n" + "="*70)
print("혼돈 시연: 초기 조건 민감도")
print("="*70)

# 초기 조건 (약간의 차이)
theta1_0 = np.pi / 2  # 90도
theta2_0 = np.pi / 2  # 90도

# 두 번째 진자: 0.001도 (약 0.00001745 radian) 차이
epsilon = 0.001 * np.pi / 180

print(f"\n초기 조건:")
print(f"  진자 A: θ1 = {np.degrees(theta1_0):.3f}°, θ2 = {np.degrees(theta2_0):.3f}°")
print(f"  진자 B: θ1 = {np.degrees(theta1_0):.3f}°, θ2 = {np.degrees(theta2_0 + epsilon):.3f}°")
print(f"  차이: Δθ2 = {epsilon*180/np.pi:.6f}° = {epsilon:.8f} rad")

t_max = 30.0  # 30초
dt = 0.01  # 10ms

print(f"\n시뮬레이션 실행 중... (t_max = {t_max}s)")

# 진자 A
t_A, theta1_A, omega1_A, theta2_A, omega2_A, x1_A, y1_A, x2_A, y2_A = \
    simulate_double_pendulum(theta1_0, 0, theta2_0, 0, t_max, dt)

# 진자 B (약간 다른 초기 조건)
t_B, theta1_B, omega1_B, theta2_B, omega2_B, x1_B, y1_B, x2_B, y2_B = \
    simulate_double_pendulum(theta1_0, 0, theta2_0 + epsilon, 0, t_max, dt)

print(f"[OK] 완료: {len(t_A)} 시간 단계")

# 차이 계산
delta_theta1 = np.abs(theta1_A - theta1_B)
delta_theta2 = np.abs(theta2_A - theta2_B)
delta_pos = np.sqrt((x2_A - x2_B)**2 + (y2_A - y2_B)**2)

# 리아푸노프 지수 추정 (대략적)
# 두 궤적이 지수적으로 발산하는 비율
idx_diverge = np.where(delta_pos > 0.1)[0]
if len(idx_diverge) > 0:
    t_diverge = t_A[idx_diverge[0]]
    lyapunov_approx = np.log(delta_pos[idx_diverge[0]] / epsilon) / t_diverge
    print(f"\n리아푸노프 지수 (근사): lambda ~= {lyapunov_approx:.3f} s^-1")
    print(f"발산 시간: t ~= {t_diverge:.2f} s")
    print(f"예측 가능 시간: ~{1/lyapunov_approx:.2f} s")

# ============================================================================
# 에너지 계산
# ============================================================================

def calculate_energy(theta1, omega1, theta2, omega2):
    """총 에너지 계산"""
    # 운동 에너지
    v1_sq = (L1 * omega1)**2
    v2_sq = (L1 * omega1)**2 + (L2 * omega2)**2 + \
            2 * L1 * L2 * omega1 * omega2 * np.cos(theta1 - theta2)
    
    T = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
    
    # 위치 에너지
    y1 = -L1 * np.cos(theta1)
    y2 = y1 - L2 * np.cos(theta2)
    
    V = m1 * g * y1 + m2 * g * y2
    
    return T + V

E_A = calculate_energy(theta1_A, omega1_A, theta2_A, omega2_A)
E_B = calculate_energy(theta1_B, omega1_B, theta2_B, omega2_B)

print(f"\n에너지 보존:")
print(f"  진자 A: ΔE/E0 = {(E_A.max() - E_A.min())/abs(E_A[0])*100:.4f}%")
print(f"  진자 B: ΔE/E0 = {(E_B.max() - E_B.min())/abs(E_B[0])*100:.4f}%")

# ============================================================================
# 시각화
# ============================================================================

# 그림 1: 이중 진자 궤적 및 위상 공간
fig1 = plt.figure(figsize=(16, 12))
gs1 = GridSpec(3, 3, figure=fig1, hspace=0.35, wspace=0.35)

# 1-1: 끝점 궤적 (진자 A)
ax11 = fig1.add_subplot(gs1[0, 0])
ax11.plot(x2_A, y2_A, 'b-', linewidth=0.5, alpha=0.7)
ax11.plot(x2_A[0], y2_A[0], 'go', markersize=10, label='Start')
ax11.plot(x2_A[-1], y2_A[-1], 'ro', markersize=10, label='End')
ax11.plot(0, 0, 'ko', markersize=8, label='Pivot')
ax11.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax11.set_ylabel('y (m)', fontsize=11, fontweight='bold')
ax11.set_title('Pendulum A: Tip Trajectory', fontsize=12, fontweight='bold')
ax11.legend(fontsize=9)
ax11.grid(True, alpha=0.3)
ax11.axis('equal')

# 1-2: 끝점 궤적 (진자 B)
ax12 = fig1.add_subplot(gs1[0, 1])
ax12.plot(x2_B, y2_B, 'r-', linewidth=0.5, alpha=0.7)
ax12.plot(x2_B[0], y2_B[0], 'go', markersize=10, label='Start')
ax12.plot(x2_B[-1], y2_B[-1], 'ro', markersize=10, label='End')
ax12.plot(0, 0, 'ko', markersize=8, label='Pivot')
ax12.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax12.set_ylabel('y (m)', fontsize=11, fontweight='bold')
ax12.set_title('Pendulum B: Tip Trajectory', fontsize=12, fontweight='bold')
ax12.legend(fontsize=9)
ax12.grid(True, alpha=0.3)
ax12.axis('equal')

# 1-3: 궤적 중첩
ax13 = fig1.add_subplot(gs1[0, 2])
ax13.plot(x2_A, y2_A, 'b-', linewidth=0.5, alpha=0.5, label='Pendulum A')
ax13.plot(x2_B, y2_B, 'r-', linewidth=0.5, alpha=0.5, label='Pendulum B')
ax13.plot(0, 0, 'ko', markersize=8, label='Pivot')
ax13.set_xlabel('x (m)', fontsize=11, fontweight='bold')
ax13.set_ylabel('y (m)', fontsize=11, fontweight='bold')
ax13.set_title('Overlay: Divergence', fontsize=12, fontweight='bold')
ax13.legend(fontsize=9)
ax13.grid(True, alpha=0.3)
ax13.axis('equal')

# 2-1: 각도 vs 시간 (θ2)
ax21 = fig1.add_subplot(gs1[1, :])
ax21.plot(t_A, np.degrees(theta2_A), 'b-', linewidth=1.5, alpha=0.7, label='Pendulum A')
ax21.plot(t_B, np.degrees(theta2_B), 'r-', linewidth=1.5, alpha=0.7, label='Pendulum B')
ax21.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax21.set_ylabel('theta_2 (degrees)', fontsize=12, fontweight='bold')
ax21.set_title('Second Pendulum Angle: Chaos Emerges', fontsize=13, fontweight='bold')
ax21.legend(fontsize=10)
ax21.grid(True, alpha=0.3)

# 2-2: 위상 공간 (θ1 vs ω1)
ax22 = fig1.add_subplot(gs1[2, 0])
ax22.plot(np.degrees(theta1_A), omega1_A, 'b-', linewidth=0.5, alpha=0.7, label='A')
ax22.plot(np.degrees(theta1_B), omega1_B, 'r-', linewidth=0.5, alpha=0.7, label='B')
ax22.set_xlabel('theta_1 (degrees)', fontsize=11, fontweight='bold')
ax22.set_ylabel('omega_1 (rad/s)', fontsize=11, fontweight='bold')
ax22.set_title('Phase Space: First Pendulum', fontsize=12, fontweight='bold')
ax22.legend(fontsize=9)
ax22.grid(True, alpha=0.3)

# 2-3: 위상 공간 (θ2 vs ω2)
ax23 = fig1.add_subplot(gs1[2, 1])
ax23.plot(np.degrees(theta2_A), omega2_A, 'b-', linewidth=0.5, alpha=0.7, label='A')
ax23.plot(np.degrees(theta2_B), omega2_B, 'r-', linewidth=0.5, alpha=0.7, label='B')
ax23.set_xlabel('theta_2 (degrees)', fontsize=11, fontweight='bold')
ax23.set_ylabel('omega_2 (rad/s)', fontsize=11, fontweight='bold')
ax23.set_title('Phase Space: Second Pendulum', fontsize=12, fontweight='bold')
ax23.legend(fontsize=9)
ax23.grid(True, alpha=0.3)

# 2-4: 에너지
ax24 = fig1.add_subplot(gs1[2, 2])
ax24.plot(t_A, E_A, 'b-', linewidth=1.5, alpha=0.7, label='Pendulum A')
ax24.plot(t_B, E_B, 'r-', linewidth=1.5, alpha=0.7, label='Pendulum B')
ax24.axhline(E_A[0], color='k', linestyle='--', linewidth=1, alpha=0.5, label='E_0')
ax24.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax24.set_ylabel('Total Energy (J)', fontsize=11, fontweight='bold')
ax24.set_title('Energy Conservation', fontsize=12, fontweight='bold')
ax24.legend(fontsize=9)
ax24.grid(True, alpha=0.3)

plt.suptitle('Double Pendulum: Chaotic Dynamics', fontsize=16, fontweight='bold')
plt.savefig(f'{output_dir}/03_double_pendulum.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_dir}/03_double_pendulum.png")
plt.close()

# 그림 2: 혼돈 분석
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2-1: 궤적 차이
axes[0, 0].semilogy(t_A, delta_pos, 'purple', linewidth=2)
axes[0, 0].axhline(epsilon, color='g', linestyle='--', linewidth=2, label='Initial Delta')
axes[0, 0].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Position Difference (m)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Exponential Divergence', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3, which='both')

# 2-2: 각도 차이
axes[0, 1].plot(t_A, np.degrees(delta_theta2), 'orange', linewidth=2)
axes[0, 1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Delta_theta_2 (degrees)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Angle Difference Growth', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 2-3: 포앵카레 단면 (Poincaré section)
# theta1 = 0을 지나갈 때의 (theta2, omega2) 기록
idx_poincare_A = np.where(np.diff(np.sign(theta1_A)) != 0)[0]
idx_poincare_B = np.where(np.diff(np.sign(theta1_B)) != 0)[0]

axes[1, 0].plot(np.degrees(theta2_A[idx_poincare_A]), omega2_A[idx_poincare_A], 
               'b.', markersize=3, alpha=0.7, label='Pendulum A')
axes[1, 0].plot(np.degrees(theta2_B[idx_poincare_B]), omega2_B[idx_poincare_B], 
               'r.', markersize=3, alpha=0.7, label='Pendulum B')
axes[1, 0].set_xlabel('theta_2 (degrees)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('omega_2 (rad/s)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Poincare Section (theta_1=0)', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 2-4: 통계 요약
axes[1, 1].axis('off')

if len(idx_diverge) > 0:
    summary = f"""
CHAOS ANALYSIS SUMMARY
{'='*45}

Initial Conditions:
  Pendulum A: theta_2 = {np.degrees(theta2_0):.6f} deg
  Pendulum B: theta_2 = {np.degrees(theta2_0 + epsilon):.6f} deg
  Difference: Delta_theta_2 = {epsilon*180/np.pi:.8f} deg

Chaos Metrics:
  Lyapunov exponent: lambda ~= {lyapunov_approx:.3f} s^-1
  Divergence time: t ~= {t_diverge:.2f} s
  Predictability: ~{1/lyapunov_approx:.2f} s

Final Difference (t={t_max}s):
  Position: {delta_pos[-1]:.3f} m
  Angle theta_2: {np.degrees(delta_theta2[-1]):.1f} deg
  
Growth Factor:
  Position: {delta_pos[-1]/epsilon:.1e}x
  
Interpretation:
  초기 차이가 {epsilon*180/np.pi:.6f} deg에서
  {t_max}초 후 완전히 다른 궤적!
  
  이것이 "나비 효과"입니다.
"""
else:
    summary = f"""
CHAOS ANALYSIS SUMMARY
{'='*45}

시뮬레이션 시간 내에 
큰 발산이 관측되지 않았습니다.

더 긴 시간 또는 다른 초기 조건을
시도해보세요.
"""

axes[1, 1].text(0.1, 0.5, summary, fontsize=10, family='monospace',
               verticalalignment='center', transform=axes[1, 1].transAxes)

plt.suptitle('Chaos Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/03_chaos_analysis.png', dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_dir}/03_chaos_analysis.png")
plt.close()

# 그림 3: 위상 공간 상세
fig3 = plt.figure(figsize=(16, 6))
gs3 = GridSpec(1, 3, figure=fig3, hspace=0.3, wspace=0.3)

# 3개의 다른 초기 조건으로 시뮬레이션
initial_conditions = [
    (np.pi/2, np.pi/2, 'blue'),
    (np.pi/2 + 0.01, np.pi/2, 'red'),
    (np.pi/2, np.pi/2 + 0.01, 'green')
]

for ax_idx in range(3):
    ax = fig3.add_subplot(gs3[0, ax_idx])
    
    for i, (th1, th2, color) in enumerate(initial_conditions):
        t, theta1, omega1, theta2, omega2, x1, y1, x2, y2 = \
            simulate_double_pendulum(th1, 0, th2, 0, 20.0, 0.01)
        
        if ax_idx == 0:
            ax.plot(np.degrees(theta1), omega1, color=color, linewidth=0.5, alpha=0.7, 
                   label=f'IC{i+1}')
            ax.set_xlabel('theta_1 (degrees)', fontsize=11, fontweight='bold')
            ax.set_ylabel('omega_1 (rad/s)', fontsize=11, fontweight='bold')
            ax.set_title('Phase Space: theta_1-omega_1', fontsize=12, fontweight='bold')
        elif ax_idx == 1:
            ax.plot(np.degrees(theta2), omega2, color=color, linewidth=0.5, alpha=0.7, 
                   label=f'IC{i+1}')
            ax.set_xlabel('theta_2 (degrees)', fontsize=11, fontweight='bold')
            ax.set_ylabel('omega_2 (rad/s)', fontsize=11, fontweight='bold')
            ax.set_title('Phase Space: theta_2-omega_2', fontsize=12, fontweight='bold')
        else:
            ax.plot(x2, y2, color=color, linewidth=0.5, alpha=0.7, label=f'IC{i+1}')
            ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
            ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
            ax.set_title('Tip Trajectory', fontsize=12, fontweight='bold')
            ax.axis('equal')
    
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Phase Space: Different Initial Conditions', fontsize=16, fontweight='bold')
plt.savefig(f'{output_dir}/03_phase_space.png', dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_dir}/03_phase_space.png")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n생성된 파일:")
print(f"  1. {output_dir}/03_double_pendulum.png - 이중 진자 궤적")
print(f"  2. {output_dir}/03_chaos_analysis.png - 혼돈 분석")
print(f"  3. {output_dir}/03_phase_space.png - 위상 공간")
print("\n주요 결과:")
if len(idx_diverge) > 0:
    print(f"  - 리아푸노프 지수: lambda ~= {lyapunov_approx:.3f} s^-1")
    print(f"  - 예측 가능 시간: ~{1/lyapunov_approx:.2f} s")
print(f"  - 초기 차이 {epsilon*180/np.pi:.6f} deg -> 완전히 다른 궤적!")

