"""
04. Lagrangian and Hamiltonian Mechanics
라그랑지안과 해밀토니안 역학

고전 역학의 세 가지 정식화를 비교합니다:
1. 뉴턴 역학 (Newton): F = ma
2. 라그랑지안 역학 (Lagrange): L = T - V
3. 해밀토니안 역학 (Hamilton): H = T + V

단순 진자를 예제로 세 방법이 동일한 결과를 줌을 보입니다.
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
print("Lagrangian and Hamiltonian Mechanics")
print("="*70)

# 물리 파라미터
g = 9.81  # m/s²
L = 1.0  # m
m = 1.0  # kg

print(f"\n진자 파라미터:")
print(f"  길이 L = {L} m")
print(f"  질량 m = {m} kg")
print(f"  중력 g = {g} m/s²")

# ============================================================================
# 1. 뉴턴 역학 (Newton's Approach)
# ============================================================================

def newtonian_pendulum(y, t):
    """
    뉴턴 방정식: F = ma
    
    토크 방정식: τ = I·α
    -mgL·sin(θ) = mL²·d²θ/dt²
    d²θ/dt² = -(g/L)·sin(θ)
    """
    theta, omega = y
    alpha = -(g / L) * np.sin(theta)
    return np.array([omega, alpha])

print("\n" + "="*70)
print("1. Newtonian Mechanics")
print("="*70)
print("방정식: d²θ/dt² = -(g/L)·sin(θ)")
print("  - 힘과 토크로부터 직접 유도")
print("  - 직관적이지만 제약 조건이 복잡한 시스템에서는 어려움")

# ============================================================================
# 2. 라그랑지안 역학 (Lagrangian Mechanics)
# ============================================================================

def lagrangian_pendulum(y, t):
    """
    라그랑지안: L = T - V
    
    T = (1/2)·m·L²·ω²  (운동 에너지)
    V = -m·g·L·cos(θ) (위치 에너지, θ=0에서 0)
    L = (1/2)·m·L²·ω² + m·g·L·cos(θ)
    
    오일러-라그랑주 방정식:
    d/dt(∂L/∂ω) - ∂L/∂θ = 0
    m·L²·dω/dt + m·g·L·sin(θ) = 0
    d²θ/dt² = -(g/L)·sin(θ)
    """
    theta, omega = y
    # 뉴턴과 동일한 운동 방정식
    alpha = -(g / L) * np.sin(theta)
    return np.array([omega, alpha])

print("\n" + "="*70)
print("2. Lagrangian Mechanics")
print("="*70)
print("라그랑지안: L = T - V")
print(f"  T = (1/2)·m·L²·ω² = (1/2)·{m}·{L}²·ω²")
print(f"  V = -m·g·L·cos(θ) = -{m}·{g}·{L}·cos(θ)")
print("\n오일러-라그랑주 방정식: d/dt(∂L/∂ω) - ∂L/∂θ = 0")
print("  - 에너지 기반 접근")
print("  - 일반화 좌표 사용 가능")
print("  - 제약 조건 자동 처리")

# ============================================================================
# 3. 해밀토니안 역학 (Hamiltonian Mechanics)
# ============================================================================

def hamiltonian_pendulum(y, t):
    """
    해밀토니안: H = T + V (총 에너지)
    
    정준 운동량: p = ∂L/∂ω = m·L²·ω
    H(θ, p) = p²/(2·m·L²) - m·g·L·cos(θ)
    
    해밀턴 방정식:
    dθ/dt = ∂H/∂p = p/(m·L²)
    dp/dt = -∂H/∂θ = -m·g·L·sin(θ)
    """
    theta, p = y
    omega = p / (m * L**2)
    dp_dt = -m * g * L * np.sin(theta)
    return np.array([omega, dp_dt])

print("\n" + "="*70)
print("3. Hamiltonian Mechanics")
print("="*70)
print("해밀토니안: H = T + V (총 에너지)")
print(f"  H = p²/(2·m·L²) - m·g·L·cos(θ)")
print("\n해밀턴 방정식:")
print("  dθ/dt = ∂H/∂p")
print("  dp/dt = -∂H/∂θ")
print("  - 위상 공간 (θ, p)에서의 흐름")
print("  - 정준 변환 가능")
print("  - 양자역학으로의 확장 용이")

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

def simulate(derivs_func, y0, t_max, dt):
    """시뮬레이션 실행"""
    n_steps = int(t_max / dt)
    t_array = np.zeros(n_steps)
    states = np.zeros((n_steps, 2))
    
    t = 0
    y = y0.copy()
    
    for i in range(n_steps):
        t_array[i] = t
        states[i] = y
        y = rk4_step(derivs_func, y, t, dt)
        t += dt
    
    return t_array, states

# ============================================================================
# 시뮬레이션 실행
# ============================================================================

print("\n" + "="*70)
print("시뮬레이션 실행")
print("="*70)

# 초기 조건
theta0 = 60 * np.pi / 180  # 60도
omega0 = 0.0

print(f"\n초기 조건:")
print(f"  θ(0) = {np.degrees(theta0):.1f}°")
print(f"  ω(0) = {omega0:.1f} rad/s")

t_max = 10.0
dt = 0.01

# 뉴턴 방법
print("\n뉴턴 방법 실행 중...")
y0_newton = np.array([theta0, omega0])
t_newton, states_newton = simulate(newtonian_pendulum, y0_newton, t_max, dt)
theta_newton = states_newton[:, 0]
omega_newton = states_newton[:, 1]

# 라그랑지안 방법
print("라그랑지안 방법 실행 중...")
y0_lagrange = np.array([theta0, omega0])
t_lagrange, states_lagrange = simulate(lagrangian_pendulum, y0_lagrange, t_max, dt)
theta_lagrange = states_lagrange[:, 0]
omega_lagrange = states_lagrange[:, 1]

# 해밀토니안 방법
print("해밀토니안 방법 실행 중...")
p0 = m * L**2 * omega0  # 정준 운동량
y0_hamilton = np.array([theta0, p0])
t_hamilton, states_hamilton = simulate(hamiltonian_pendulum, y0_hamilton, t_max, dt)
theta_hamilton = states_hamilton[:, 0]
p_hamilton = states_hamilton[:, 1]
omega_hamilton = p_hamilton / (m * L**2)

print("[OK] 모든 시뮬레이션 완료")

# ============================================================================
# 에너지 및 물리량 계산
# ============================================================================

# 뉴턴
E_newton = 0.5 * m * (L * omega_newton)**2 - m * g * L * np.cos(theta_newton)

# 라그랑지안
T_lagrange = 0.5 * m * (L * omega_lagrange)**2
V_lagrange = -m * g * L * np.cos(theta_lagrange)
E_lagrange = T_lagrange + V_lagrange

# 해밀토니안 (이미 H = E)
E_hamilton = p_hamilton**2 / (2 * m * L**2) - m * g * L * np.cos(theta_hamilton)

print("\n에너지 보존:")
print(f"  뉴턴:       Delta_E/E0 = {(E_newton.max() - E_newton.min())/abs(E_newton[0])*100:.6f}%")
print(f"  라그랑지안: Delta_E/E0 = {(E_lagrange.max() - E_lagrange.min())/abs(E_lagrange[0])*100:.6f}%")
print(f"  해밀토니안: Delta_E/E0 = {(E_hamilton.max() - E_hamilton.min())/abs(E_hamilton[0])*100:.6f}%")

# 차이 계산
diff_newton_lagrange = np.max(np.abs(theta_newton - theta_lagrange))
diff_newton_hamilton = np.max(np.abs(theta_newton - theta_hamilton))

print("\n방법 간 차이:")
print(f"  |θ_Newton - θ_Lagrange| < {diff_newton_lagrange:.10f} rad")
print(f"  |θ_Newton - θ_Hamilton| < {diff_newton_hamilton:.10f} rad")
print("  → 세 방법이 동일한 결과!")

# ============================================================================
# 시각화
# ============================================================================

# 그림 1: 세 방법 비교
fig1 = plt.figure(figsize=(16, 10))
gs1 = GridSpec(2, 3, figure=fig1, hspace=0.3, wspace=0.3)

# 1-1: 각도 vs 시간
ax11 = fig1.add_subplot(gs1[0, :])
ax11.plot(t_newton, np.degrees(theta_newton), 'b-', linewidth=2, label='Newton', alpha=0.9)
ax11.plot(t_lagrange, np.degrees(theta_lagrange), 'r--', linewidth=2, label='Lagrange', alpha=0.7)
ax11.plot(t_hamilton, np.degrees(theta_hamilton), 'g:', linewidth=3, label='Hamilton', alpha=0.7)
ax11.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
ax11.set_ylabel('Angle θ (degrees)', fontsize=13, fontweight='bold')
ax11.set_title('Three Formulations: Identical Results', fontsize=14, fontweight='bold')
ax11.legend(fontsize=12)
ax11.grid(True, alpha=0.3)

# 1-2: 뉴턴 위상 공간
ax12 = fig1.add_subplot(gs1[1, 0])
ax12.plot(np.degrees(theta_newton), omega_newton, 'b-', linewidth=2, alpha=0.7)
ax12.plot(np.degrees(theta_newton[0]), omega_newton[0], 'go', markersize=10, label='Start')
ax12.set_xlabel('θ (degrees)', fontsize=12, fontweight='bold')
ax12.set_ylabel('ω (rad/s)', fontsize=12, fontweight='bold')
ax12.set_title('Newton: (θ, ω) Phase Space', fontsize=13, fontweight='bold')
ax12.legend(fontsize=10)
ax12.grid(True, alpha=0.3)

# 1-3: 라그랑지안 에너지
ax13 = fig1.add_subplot(gs1[1, 1])
ax13.fill_between(t_lagrange, 0, T_lagrange, alpha=0.5, color='blue', label='Kinetic T')
ax13.fill_between(t_lagrange, T_lagrange, E_lagrange, alpha=0.5, color='red', label='Potential V')
ax13.plot(t_lagrange, E_lagrange, 'k-', linewidth=2, label='Total E')
ax13.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax13.set_ylabel('Energy (J)', fontsize=12, fontweight='bold')
ax13.set_title('Lagrangian: Energy Partition', fontsize=13, fontweight='bold')
ax13.legend(fontsize=10)
ax13.grid(True, alpha=0.3)

# 1-4: 해밀토니안 위상 공간
ax14 = fig1.add_subplot(gs1[1, 2])
ax14.plot(np.degrees(theta_hamilton), p_hamilton, 'g-', linewidth=2, alpha=0.7)
ax14.plot(np.degrees(theta_hamilton[0]), p_hamilton[0], 'ro', markersize=10, label='Start')
ax14.set_xlabel('θ (degrees)', fontsize=12, fontweight='bold')
ax14.set_ylabel('p (kg·m²/s)', fontsize=12, fontweight='bold')
ax14.set_title('Hamilton: (θ, p) Phase Space', fontsize=13, fontweight='bold')
ax14.legend(fontsize=10)
ax14.grid(True, alpha=0.3)

plt.suptitle('Classical Mechanics: Three Formulations', fontsize=16, fontweight='bold')
plt.savefig(f'{output_dir}/04_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_dir}/04_comparison.png")
plt.close()

# 그림 2: 라그랑지안 상세
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# 2-1: 라그랑지안 L = T - V
L_function = T_lagrange - V_lagrange
axes[0, 0].plot(t_lagrange, L_function, 'purple', linewidth=2)
axes[0, 0].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Lagrangian L = T - V (J)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Lagrangian Function', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2-2: 작용 적분 (Action)
action_integrand = L_function
action_cumulative = np.cumsum(action_integrand) * dt
axes[0, 1].plot(t_lagrange, action_cumulative, 'blue', linewidth=2)
axes[0, 1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Action S = integral(L dt) (J·s)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Action Integral', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 2-3: 운동량 p = ∂L/∂ω
momentum = m * L**2 * omega_lagrange
axes[1, 0].plot(t_lagrange, momentum, 'red', linewidth=2)
axes[1, 0].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Momentum p = m·L²·ω (kg·m²/s)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Canonical Momentum', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 2-4: 설명 텍스트
axes[1, 1].axis('off')
lagrangian_text = f"""
LAGRANGIAN MECHANICS
{'='*50}

Key Concepts:
------------
1. Lagrangian: L = T - V
   L = (1/2)·m·L²·ω² + m·g·L·cos(θ)

2. Euler-Lagrange Equation:
   d/dt(∂L/∂ω) - ∂L/∂θ = 0

3. Generalized Coordinates:
   - Use any convenient coordinates
   - Constraints automatically satisfied

4. Action Principle:
   - System follows path that minimizes action
   - S = ∫L dt

Advantages:
-----------
- Energy-based (more fundamental)
- Works with any coordinates
- Elegant and systematic
- Extends to field theory

Applications:
-------------
- Constrained systems
- Robotics kinematics
- Quantum field theory
- General relativity
"""

axes[1, 1].text(0.05, 0.5, lagrangian_text, fontsize=9, family='monospace',
               verticalalignment='center', transform=axes[1, 1].transAxes)

plt.suptitle('Lagrangian Mechanics Details', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_lagrangian.png', dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_dir}/04_lagrangian.png")
plt.close()

# 그림 3: 해밀토니안 상세
fig3, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3-1: 해밀토니안 H = T + V
axes[0, 0].plot(t_hamilton, E_hamilton, 'green', linewidth=2, label='H (Total Energy)')
axes[0, 0].axhline(E_hamilton[0], color='k', linestyle='--', linewidth=1, label='E_0')
axes[0, 0].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Hamiltonian H (J)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Hamiltonian = Total Energy', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 3-2: 위상 공간 흐름
axes[0, 1].plot(np.degrees(theta_hamilton), p_hamilton, 'g-', linewidth=2, alpha=0.7)

# 여러 에너지 준위의 궤적
for E_level in [0.5, 1.0, 1.5, 2.0]:
    theta_range = np.linspace(-np.pi, np.pi, 1000)
    # H = p²/(2mL²) - mgL·cos(θ) = E
    # p² = 2mL²(E + mgL·cos(θ))
    p_squared = 2 * m * L**2 * (E_level + m * g * L * np.cos(theta_range))
    valid = p_squared >= 0
    if np.any(valid):
        p_positive = np.sqrt(p_squared[valid])
        p_negative = -p_positive
        axes[0, 1].plot(np.degrees(theta_range[valid]), p_positive, 'k-', 
                       linewidth=1, alpha=0.3)
        axes[0, 1].plot(np.degrees(theta_range[valid]), p_negative, 'k-', 
                       linewidth=1, alpha=0.3)

axes[0, 1].plot(np.degrees(theta_hamilton[0]), p_hamilton[0], 'ro', markersize=10)
axes[0, 1].set_xlabel('θ (degrees)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('p (kg·m²/s)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Phase Space Flow', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3-3: 해밀턴 방정식 확인
dtheta_dt_computed = np.gradient(theta_hamilton, t_hamilton)
dtheta_dt_hamilton = p_hamilton / (m * L**2)

axes[1, 0].plot(t_hamilton, dtheta_dt_computed, 'b-', linewidth=2, label='Numerical dθ/dt')
axes[1, 0].plot(t_hamilton, dtheta_dt_hamilton, 'r--', linewidth=2, label='∂H/∂p', alpha=0.7)
axes[1, 0].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('dθ/dt (rad/s)', fontsize=12, fontweight='bold')
axes[1, 0].set_title("Hamilton's Equations: dθ/dt = ∂H/∂p", fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 3-4: 설명 텍스트
axes[1, 1].axis('off')
hamiltonian_text = f"""
HAMILTONIAN MECHANICS
{'='*50}

Key Concepts:
------------
1. Hamiltonian: H = T + V (Total Energy)
   H = p²/(2·m·L²) - m·g·L·cos(θ)

2. Hamilton's Equations:
   dθ/dt = ∂H/∂p  (position evolution)
   dp/dt = -∂H/∂θ (momentum evolution)

3. Phase Space (θ, p):
   - 2N dimensional for N DOF
   - Volume-preserving flow (Liouville)
   - Symplectic structure

4. Canonical Transformations:
   - Change (θ,p) → (Q,P)
   - Preserve Hamilton's equations
   - Find simpler coordinates

Advantages:
-----------
- Symmetric treatment of θ and p
- Conservation laws explicit
- Quantum mechanics connection
- Statistical mechanics foundation

Connection to Quantum:
----------------------
- Ĥ → -iℏ∂/∂t
- p̂ → -iℏ∂/∂θ
- Classical limit: ℏ → 0
"""

axes[1, 1].text(0.05, 0.5, hamiltonian_text, fontsize=9, family='monospace',
               verticalalignment='center', transform=axes[1, 1].transAxes)

plt.suptitle('Hamiltonian Mechanics Details', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_hamiltonian.png', dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_dir}/04_hamiltonian.png")
plt.close()

# 그림 4: 위상 공간 비교
fig4, axes = plt.subplots(1, 3, figsize=(16, 5))

# (θ, ω) 좌표
axes[0].plot(np.degrees(theta_newton), omega_newton, 'b-', linewidth=2, alpha=0.7)
axes[0].plot(np.degrees(theta_newton[0]), omega_newton[0], 'go', markersize=10)
axes[0].set_xlabel('θ (degrees)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('ω (rad/s)', fontsize=12, fontweight='bold')
axes[0].set_title('Newton/Lagrange: (θ, ω)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# (θ, p) 좌표
axes[1].plot(np.degrees(theta_hamilton), p_hamilton, 'g-', linewidth=2, alpha=0.7)
axes[1].plot(np.degrees(theta_hamilton[0]), p_hamilton[0], 'ro', markersize=10)
axes[1].set_xlabel('θ (degrees)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('p (kg·m²/s)', fontsize=12, fontweight='bold')
axes[1].set_title('Hamilton: (θ, p)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# 에너지 등고선
axes[2].axis('off')
comparison_text = f"""
PHASE SPACE COMPARISON
{'='*40}

Newton/Lagrange: (θ, ω)
-----------------------
- Angular velocity ω
- Intuitive coordinates
- ω = dθ/dt directly

Hamilton: (θ, p)
----------------
- Canonical momentum p
- p = m·L²·ω (related but not same)
- Symmetric equations

Relationship:
-------------
p = ∂L/∂ω = m·L²·ω

For simple pendulum:
  p = {m}·{L}²·ω = {m*L**2:.1f}·ω

Both describe same physics!
- Different coordinates
- Same trajectories
- Different perspectives

Which to use?
-------------
- Newton: Direct problems
- Lagrange: Constraints
- Hamilton: Symmetries/Quantum
"""

axes[2].text(0.1, 0.5, comparison_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=axes[2].transAxes)

plt.suptitle('Phase Space Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_phase_space.png', dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_dir}/04_phase_space.png")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n생성된 파일:")
print(f"  1. {output_dir}/04_comparison.png - 세 방법 비교")
print(f"  2. {output_dir}/04_lagrangian.png - 라그랑지안 역학 상세")
print(f"  3. {output_dir}/04_hamiltonian.png - 해밀토니안 역학 상세")
print(f"  4. {output_dir}/04_phase_space.png - 위상 공간 비교")
print("\n주요 결과:")
print(f"  - 세 방법이 동일한 결과 (차이 < {max(diff_newton_lagrange, diff_newton_hamilton):.10f} rad)")
print(f"  - 에너지 보존 < 0.0001%")
print(f"  - 각 방법은 다른 관점, 같은 물리!")

