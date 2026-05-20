"""
01. Numerical Integration Methods: Euler vs RK4
오일러 방법과 룽게-쿠타 4차 방법 비교

수치 적분의 두 가지 대표적인 방법을 비교합니다:
- Euler Method: 1차 정확도, 간단하지만 오차 누적
- RK4 Method: 4차 정확도, 정확하고 안정적

테스트 시스템:
1. 단순 조화 진동자 (Simple Harmonic Oscillator)
2. 감쇠 진자 (Damped Pendulum)
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
print("Numerical Integration Methods: Euler vs RK4")
print("="*70)

# ============================================================================
# 수치 적분 방법 구현
# ============================================================================

def euler_step(f, y, t, dt):
    """
    오일러 방법 (1차 정확도)
    
    y(t+dt) = y(t) + dt * f(y, t)
    
    Parameters:
    -----------
    f : function
        미분 방정식 dy/dt = f(y, t)
    y : array
        현재 상태
    t : float
        현재 시간
    dt : float
        시간 간격
    
    Returns:
    --------
    y_next : array
        다음 시간의 상태
    """
    return y + dt * f(y, t)

def rk4_step(f, y, t, dt):
    """
    Runge-Kutta 4차 방법 (4차 정확도)
    
    더 정확한 수치 적분 방법
    4번의 기울기를 계산하여 가중 평균
    
    Parameters:
    -----------
    f : function
        미분 방정식 dy/dt = f(y, t)
    y : array
        현재 상태
    t : float
        현재 시간
    dt : float
        시간 간격
    
    Returns:
    --------
    y_next : array
        다음 시간의 상태
    """
    k1 = f(y, t)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(y + dt*k3, t + dt)
    
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# ============================================================================
# 테스트 시스템 1: 단순 조화 진동자
# ============================================================================

def harmonic_oscillator(y, t):
    """
    단순 조화 진동자
    
    운동 방정식: d²x/dt² = -ω²x
    상태 변수: y = [x, v] where v = dx/dt
    
    dy/dt = [v, -ω²x]
    """
    omega = 2*np.pi  # 각진동수 (주기 T = 1초)
    x, v = y
    return np.array([v, -omega**2 * x])

print("\n" + "="*70)
print("Test 1: Simple Harmonic Oscillator")
print("="*70)
print("운동 방정식: d²x/dt² = -ω²x")
print(f"각진동수 ω = {2*np.pi:.4f} rad/s (주기 T = 1.0 s)")
print()

# 초기 조건
y0 = np.array([1.0, 0.0])  # x(0) = 1, v(0) = 0
t_max = 10.0  # 10주기
dt = 0.1

# 시간 배열
t_euler = [0]
y_euler = [y0]

t_rk4 = [0]
y_rk4 = [y0]

# 오일러 방법으로 적분
t = 0
y = y0.copy()
while t < t_max:
    y = euler_step(harmonic_oscillator, y, t, dt)
    t += dt
    t_euler.append(t)
    y_euler.append(y.copy())

# RK4 방법으로 적분
t = 0
y = y0.copy()
while t < t_max:
    y = rk4_step(harmonic_oscillator, y, t, dt)
    t += dt
    t_rk4.append(t)
    y_rk4.append(y.copy())

# 배열로 변환
t_euler = np.array(t_euler)
y_euler = np.array(y_euler)
t_rk4 = np.array(t_rk4)
y_rk4 = np.array(y_rk4)

# 해석해 (정확한 해)
omega = 2*np.pi
t_exact = np.linspace(0, t_max, 1000)
x_exact = np.cos(omega * t_exact)

# 에너지 계산 (보존되어야 함)
E_euler = 0.5 * (y_euler[:, 1]**2 + (omega * y_euler[:, 0])**2)
E_rk4 = 0.5 * (y_rk4[:, 1]**2 + (omega * y_rk4[:, 0])**2)
E0 = 0.5 * omega**2  # 초기 에너지

print(f"오일러 방법:")
print(f"  최종 에너지 오차: {abs(E_euler[-1] - E0)/E0*100:.2f}%")
print(f"RK4 방법:")
print(f"  최종 에너지 오차: {abs(E_rk4[-1] - E0)/E0*100:.2f}%")

# ============================================================================
# 테스트 시스템 2: 감쇠 진자
# ============================================================================

def damped_pendulum(y, t):
    """
    감쇠 진자
    
    운동 방정식: d²θ/dt² = -ω²sin(θ) - γdθ/dt
    상태 변수: y = [θ, ω] where ω = dθ/dt
    
    dy/dt = [ω, -ω₀²sin(θ) - γω]
    """
    omega0 = 2*np.pi  # 고유 각진동수
    gamma = 0.1  # 감쇠 계수
    theta, omega = y
    return np.array([omega, -omega0**2 * np.sin(theta) - gamma * omega])

print("\n" + "="*70)
print("Test 2: Damped Pendulum")
print("="*70)
print("운동 방정식: d²θ/dt² = -ω0²sin(θ) - γdθ/dt")
print(f"고유 각진동수 ω0 = {2*np.pi:.4f} rad/s")
print(f"감쇠 계수 γ = 0.1")
print()

# 초기 조건 (큰 각도)
y0_pend = np.array([np.pi/3, 0.0])  # θ(0) = 60°, ω(0) = 0
t_max_pend = 20.0
dt_pend = 0.05

# 오일러 방법
t_pend_euler = [0]
y_pend_euler = [y0_pend]

t = 0
y = y0_pend.copy()
while t < t_max_pend:
    y = euler_step(damped_pendulum, y, t, dt_pend)
    t += dt_pend
    t_pend_euler.append(t)
    y_pend_euler.append(y.copy())

# RK4 방법
t_pend_rk4 = [0]
y_pend_rk4 = [y0_pend]

t = 0
y = y0_pend.copy()
while t < t_max_pend:
    y = rk4_step(damped_pendulum, y, t, dt_pend)
    t += dt_pend
    t_pend_rk4.append(t)
    y_pend_rk4.append(y.copy())

# 배열로 변환
t_pend_euler = np.array(t_pend_euler)
y_pend_euler = np.array(y_pend_euler)
t_pend_rk4 = np.array(t_pend_rk4)
y_pend_rk4 = np.array(y_pend_rk4)

print(f"오일러 방법:")
print(f"  최종 각도: {np.degrees(y_pend_euler[-1, 0]):.2f}°")
print(f"RK4 방법:")
print(f"  최종 각도: {np.degrees(y_pend_rk4[-1, 0]):.2f}°")

# ============================================================================
# 시각화
# ============================================================================

# 그림 1: Euler vs RK4 비교
fig1 = plt.figure(figsize=(16, 10))
gs1 = GridSpec(3, 2, figure=fig1, hspace=0.3, wspace=0.3)

# 1-1: 조화 진동자 - 위치
ax11 = fig1.add_subplot(gs1[0, 0])
ax11.plot(t_exact, x_exact, 'k-', linewidth=2, label='Exact Solution', alpha=0.5)
ax11.plot(t_euler, y_euler[:, 0], 'r--', linewidth=2, label='Euler Method')
ax11.plot(t_rk4, y_rk4[:, 0], 'b-', linewidth=2, label='RK4 Method', alpha=0.7)
ax11.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax11.set_ylabel('Position x(t)', fontsize=12, fontweight='bold')
ax11.set_title('Simple Harmonic Oscillator: Position', fontsize=13, fontweight='bold')
ax11.legend(fontsize=10)
ax11.grid(True, alpha=0.3)

# 1-2: 조화 진동자 - 에너지
ax12 = fig1.add_subplot(gs1[0, 1])
ax12.plot(t_euler, (E_euler - E0)/E0 * 100, 'r--', linewidth=2, label='Euler')
ax12.plot(t_rk4, (E_rk4 - E0)/E0 * 100, 'b-', linewidth=2, label='RK4')
ax12.axhline(0, color='k', linestyle=':', alpha=0.5)
ax12.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax12.set_ylabel('Energy Error (%)', fontsize=12, fontweight='bold')
ax12.set_title('Energy Conservation Error', fontsize=13, fontweight='bold')
ax12.legend(fontsize=10)
ax12.grid(True, alpha=0.3)

# 2-1: 감쇠 진자 - 각도
ax21 = fig1.add_subplot(gs1[1, 0])
ax21.plot(t_pend_euler, np.degrees(y_pend_euler[:, 0]), 'r--', linewidth=2, label='Euler')
ax21.plot(t_pend_rk4, np.degrees(y_pend_rk4[:, 0]), 'b-', linewidth=2, label='RK4', alpha=0.7)
ax21.axhline(0, color='k', linestyle=':', alpha=0.5)
ax21.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax21.set_ylabel('Angle θ(t) (degrees)', fontsize=12, fontweight='bold')
ax21.set_title('Damped Pendulum: Angle', fontsize=13, fontweight='bold')
ax21.legend(fontsize=10)
ax21.grid(True, alpha=0.3)

# 2-2: 감쇠 진자 - 위상 공간
ax22 = fig1.add_subplot(gs1[1, 1])
ax22.plot(np.degrees(y_pend_euler[:, 0]), y_pend_euler[:, 1], 'r--', linewidth=2, label='Euler', alpha=0.7)
ax22.plot(np.degrees(y_pend_rk4[:, 0]), y_pend_rk4[:, 1], 'b-', linewidth=2, label='RK4', alpha=0.7)
ax22.plot(np.degrees(y0_pend[0]), y0_pend[1], 'go', markersize=10, label='Start')
ax22.set_xlabel('Angle θ (degrees)', fontsize=12, fontweight='bold')
ax22.set_ylabel('Angular Velocity ω (rad/s)', fontsize=12, fontweight='bold')
ax22.set_title('Phase Space', fontsize=13, fontweight='bold')
ax22.legend(fontsize=10)
ax22.grid(True, alpha=0.3)

# 3-1, 3-2: 오차 분석 (긴 시간)
# 더 긴 시간 동안 시뮬레이션하여 오차 누적 확인
t_max_long = 50.0
dt_test = 0.05

t_long_euler = [0]
y_long_euler = [y0]
t = 0
y = y0.copy()
while t < t_max_long:
    y = euler_step(harmonic_oscillator, y, t, dt_test)
    t += dt_test
    t_long_euler.append(t)
    y_long_euler.append(y.copy())

t_long_rk4 = [0]
y_long_rk4 = [y0]
t = 0
y = y0.copy()
while t < t_max_long:
    y = rk4_step(harmonic_oscillator, y, t, dt_test)
    t += dt_test
    t_long_rk4.append(t)
    y_long_rk4.append(y.copy())

t_long_euler = np.array(t_long_euler)
y_long_euler = np.array(y_long_euler)
t_long_rk4 = np.array(t_long_rk4)
y_long_rk4 = np.array(y_long_rk4)

# 해석해와 비교
x_exact_long = np.cos(omega * t_long_euler)
error_euler = np.abs(y_long_euler[:, 0] - x_exact_long)
error_rk4 = np.abs(y_long_rk4[:, 0] - np.cos(omega * t_long_rk4))

ax31 = fig1.add_subplot(gs1[2, 0])
ax31.semilogy(t_long_euler, error_euler, 'r-', linewidth=2, label='Euler')
ax31.semilogy(t_long_rk4, error_rk4, 'b-', linewidth=2, label='RK4')
ax31.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax31.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
ax31.set_title('Error Accumulation (Log Scale)', fontsize=13, fontweight='bold')
ax31.legend(fontsize=10)
ax31.grid(True, alpha=0.3, which='both')

# 3-2: 시간 간격에 따른 오차
dt_values = np.array([0.2, 0.1, 0.05, 0.02, 0.01])
errors_euler = []
errors_rk4 = []

for dt_test in dt_values:
    # 오일러
    t = 0
    y = y0.copy()
    while t < 10.0:
        y = euler_step(harmonic_oscillator, y, t, dt_test)
        t += dt_test
    error_euler_final = np.abs(y[0] - np.cos(omega * t))
    errors_euler.append(error_euler_final)
    
    # RK4
    t = 0
    y = y0.copy()
    while t < 10.0:
        y = rk4_step(harmonic_oscillator, y, t, dt_test)
        t += dt_test
    error_rk4_final = np.abs(y[0] - np.cos(omega * t))
    errors_rk4.append(error_rk4_final)

ax32 = fig1.add_subplot(gs1[2, 1])
ax32.loglog(dt_values, errors_euler, 'ro-', linewidth=2, markersize=8, label='Euler (O(dt))')
ax32.loglog(dt_values, errors_rk4, 'bs-', linewidth=2, markersize=8, label='RK4 (O(dt⁴))')
# 기울기 참조선
ax32.loglog(dt_values, errors_euler[0] * (dt_values/dt_values[0])**1, 'r:', linewidth=1, alpha=0.5, label='slope = 1')
ax32.loglog(dt_values, errors_rk4[0] * (dt_values/dt_values[0])**4, 'b:', linewidth=1, alpha=0.5, label='slope = 4')
ax32.set_xlabel('Time Step dt (s)', fontsize=12, fontweight='bold')
ax32.set_ylabel('Error at t=10s', fontsize=12, fontweight='bold')
ax32.set_title('Convergence Rate', fontsize=13, fontweight='bold')
ax32.legend(fontsize=9)
ax32.grid(True, alpha=0.3, which='both')

plt.suptitle('Numerical Integration: Euler vs RK4 Comparison', fontsize=16, fontweight='bold')
plt.savefig(f'{output_dir}/01_euler_vs_rk4.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_dir}/01_euler_vs_rk4.png")
plt.close()

# 그림 2: 오차 분석 상세
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# 오차 누적 비교
axes[0, 0].plot(t_long_euler, error_euler, 'r-', linewidth=2, label='Euler')
axes[0, 0].plot(t_long_rk4, error_rk4, 'b-', linewidth=2, label='RK4')
axes[0, 0].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Error vs Time (Linear Scale)', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 에너지 보존 (장시간)
E_long_euler = 0.5 * (y_long_euler[:, 1]**2 + (omega * y_long_euler[:, 0])**2)
E_long_rk4 = 0.5 * (y_long_rk4[:, 1]**2 + (omega * y_long_rk4[:, 0])**2)

axes[0, 1].plot(t_long_euler, E_long_euler, 'r-', linewidth=2, label='Euler')
axes[0, 1].plot(t_long_rk4, E_long_rk4, 'b-', linewidth=2, label='RK4')
axes[0, 1].axhline(E0, color='k', linestyle='--', linewidth=2, label='True Energy')
axes[0, 1].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Total Energy', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Energy Conservation (Long Time)', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 위상 공간 (조화 진동자)
axes[1, 0].plot(y_long_euler[:, 0], y_long_euler[:, 1], 'r-', linewidth=1, alpha=0.5, label='Euler')
axes[1, 0].plot(y_long_rk4[:, 0], y_long_rk4[:, 1], 'b-', linewidth=1, alpha=0.7, label='RK4')
# 정확한 타원 (에너지 보존)
theta_circle = np.linspace(0, 2*np.pi, 100)
r = np.sqrt(2*E0) / omega
axes[1, 0].plot(r*np.cos(theta_circle), omega*r*np.sin(theta_circle), 'k--', linewidth=2, label='Exact (E=const)')
axes[1, 0].set_xlabel('Position x', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Velocity v', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Phase Space Portrait', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

# 통계 요약
axes[1, 1].axis('off')
summary_text = f"""
NUMERICAL INTEGRATION COMPARISON
{'='*45}

Method Characteristics:
----------------------
Euler Method:
  - Order of accuracy: O(dt)
  - Computational cost: 1 function evaluation/step
  - Stability: Poor for long integration
  - Energy drift: Significant

RK4 Method:
  - Order of accuracy: O(dt⁴)
  - Computational cost: 4 function evaluations/step
  - Stability: Excellent for long integration
  - Energy drift: Minimal

Test Results (t=50s, dt=0.05):
------------------------------
Simple Harmonic Oscillator:
  Euler error: {error_euler[-1]:.6f}
  RK4 error:   {error_rk4[-1]:.8f}
  Improvement: {error_euler[-1]/error_rk4[-1]:.0f}x better

Energy Conservation:
  Euler drift: {abs(E_long_euler[-1] - E0)/E0*100:.2f}%
  RK4 drift:   {abs(E_long_rk4[-1] - E0)/E0*100:.4f}%

Conclusion:
-----------
RK4 is the industry standard for ODE integration.
4x more expensive but 100-1000x more accurate!
"""

axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
               verticalalignment='center', transform=axes[1, 1].transAxes)

plt.suptitle('Detailed Error Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_error_analysis.png', dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_dir}/01_error_analysis.png")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n생성된 파일:")
print(f"  1. {output_dir}/01_euler_vs_rk4.png - Euler vs RK4 비교")
print(f"  2. {output_dir}/01_error_analysis.png - 상세 오차 분석")
print("\n주요 결론:")
print(f"  - Euler: 1차 정확도, 빠르지만 부정확")
print(f"  - RK4: 4차 정확도, 느리지만 매우 정확")
print(f"  - RK4가 Euler보다 약 {error_euler[-1]/error_rk4[-1]:.0f}배 더 정확!")

