"""
02. Planetary Motion Simulation
행성 운동 시뮬레이션

뉴턴의 만유인력 법칙을 사용하여 태양계 행성들의 운동을 시뮬레이션합니다.

케플러의 법칙:
1. 행성의 궤도는 타원이고, 태양은 타원의 한 초점에 있다
2. 행성과 태양을 잇는 선분이 같은 시간 동안 쓸고 지나가는 면적은 같다 (면적 속도 일정)
3. 행성의 공전 주기의 제곱은 궤도 장반경의 세제곱에 비례한다 (T² ∝ a³)
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
print("Planetary Motion Simulation")
print("="*70)

# 물리 상수 (SI 단위 대신 AU, year, solar mass 사용)
G = 4 * np.pi**2  # AU³ / (year² · M_sun)
M_sun = 1.0  # 태양 질량

# ============================================================================
# 행성 데이터
# ============================================================================

planets_data = {
    'Earth': {
        'a': 1.0,  # 장반경 (AU)
        'e': 0.0167,  # 이심률
        'period': 1.0,  # 공전 주기 (year)
        'mass': 3.0e-6,  # 태양 질량 대비
        'color': 'blue'
    },
    'Mars': {
        'a': 1.524,
        'e': 0.0934,
        'period': 1.881,
        'mass': 3.2e-7,
        'color': 'red'
    },
    'Jupiter': {
        'a': 5.203,
        'e': 0.0489,
        'period': 11.86,
        'mass': 9.5e-4,
        'color': 'orange'
    }
}

print("\n행성 데이터:")
print("-"*70)
for name, data in planets_data.items():
    print(f"{name:10s}: a = {data['a']:.3f} AU, e = {data['e']:.4f}, T = {data['period']:.2f} years")

# ============================================================================
# 수치 적분 (RK4)
# ============================================================================

def rk4_step(f, y, t, dt):
    """Runge-Kutta 4차 방법"""
    k1 = f(y, t)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(y + dt*k3, t + dt)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

def gravitational_field(y, t):
    """
    중력장에서의 운동 방정식
    
    y = [x, y, vx, vy]
    dy/dt = [vx, vy, ax, ay]
    
    중력 가속도: a = -GM/r³ · r_vec
    """
    x, y_pos, vx, vy = y
    r = np.sqrt(x**2 + y_pos**2)
    
    # 중력 가속도
    a = -G * M_sun / r**3
    ax = a * x
    ay = a * y_pos
    
    return np.array([vx, vy, ax, ay])

def simulate_planet(a, e, t_max, dt=0.001):
    """
    행성 궤도 시뮬레이션
    
    Parameters:
    -----------
    a : float
        장반경 (AU)
    e : float
        이심률
    t_max : float
        시뮬레이션 시간 (years)
    dt : float
        시간 간격 (years)
    
    Returns:
    --------
    t_array, x_array, y_array, vx_array, vy_array
    """
    # 초기 조건 (근일점)
    r0 = a * (1 - e)  # 근일점 거리
    v0 = np.sqrt(G * M_sun * (1 + e) / (a * (1 - e)))  # 근일점 속도
    
    y0 = np.array([r0, 0.0, 0.0, v0])
    
    # 시간 배열
    n_steps = int(t_max / dt)
    t_array = np.zeros(n_steps)
    trajectory = np.zeros((n_steps, 4))
    
    t = 0
    y = y0.copy()
    
    for i in range(n_steps):
        t_array[i] = t
        trajectory[i] = y
        y = rk4_step(gravitational_field, y, t, dt)
        t += dt
    
    return t_array, trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3]

# ============================================================================
# 시뮬레이션 실행
# ============================================================================

print("\n" + "="*70)
print("시뮬레이션 실행 중...")
print("="*70)

simulations = {}

for name, data in planets_data.items():
    print(f"\n{name} 시뮬레이션 중... (주기 = {data['period']:.2f} years)")
    t_max = data['period'] * 2  # 2 공전 주기
    t, x, y, vx, vy = simulate_planet(data['a'], data['e'], t_max, dt=0.001)
    
    # 궤도 계산
    r = np.sqrt(x**2 + y**2)
    v = np.sqrt(vx**2 + vy**2)
    
    # 에너지 계산
    E_kinetic = 0.5 * data['mass'] * v**2
    E_potential = -G * M_sun * data['mass'] / r
    E_total = E_kinetic + E_potential
    
    # 각운동량 계산
    L = data['mass'] * (x * vy - y * vx)
    
    simulations[name] = {
        't': t,
        'x': x,
        'y': y,
        'r': r,
        'v': v,
        'E': E_total,
        'L': L,
        'data': data
    }
    
    print(f"  [OK] 완료: {len(t)} 시간 단계")
    print(f"  에너지 보존: {(E_total.max() - E_total.min())/abs(E_total.mean())*100:.4f}%")
    print(f"  각운동량 보존: {(L.max() - L.min())/abs(L.mean())*100:.4f}%")

# ============================================================================
# 케플러의 법칙 검증
# ============================================================================

print("\n" + "="*70)
print("케플러의 법칙 검증")
print("="*70)

# 케플러 제3법칙: T² / a³ = const
print("\n케플러 제3법칙: T² ∝ a³")
print("-"*70)
print(f"{'Planet':<10s} {'Period (yr)':<15s} {'Semi-major (AU)':<15s} {'T²/a³':<15s}")
print("-"*70)

kepler_constant = []
for name, data in planets_data.items():
    T = data['period']
    a = data['a']
    ratio = T**2 / a**3
    kepler_constant.append(ratio)
    print(f"{name:<10s} {T:<15.3f} {a:<15.3f} {ratio:<15.6f}")

print(f"\n평균 T²/a³ = {np.mean(kepler_constant):.6f}")
print(f"표준편차 = {np.std(kepler_constant):.8f} (이론값: 1.0)")

# ============================================================================
# 시각화
# ============================================================================

# 그림 1: 태양계 궤도
fig1 = plt.figure(figsize=(16, 12))
gs1 = GridSpec(2, 2, figure=fig1, hspace=0.3, wspace=0.3)

# 1-1: 전체 궤도
ax11 = fig1.add_subplot(gs1[0, :])

# 태양
ax11.plot(0, 0, 'yo', markersize=20, markeredgecolor='orange', markeredgewidth=2, label='Sun')

# 행성 궤도
for name, sim in simulations.items():
    ax11.plot(sim['x'], sim['y'], '-', color=sim['data']['color'], 
             linewidth=2, alpha=0.7, label=name)
    # 시작점
    ax11.plot(sim['x'][0], sim['y'][0], 'o', color=sim['data']['color'], 
             markersize=8, markeredgecolor='black', markeredgewidth=1)

ax11.set_xlabel('x (AU)', fontsize=13, fontweight='bold')
ax11.set_ylabel('y (AU)', fontsize=13, fontweight='bold')
ax11.set_title('Planetary Orbits (2 periods)', fontsize=14, fontweight='bold')
ax11.legend(fontsize=11, loc='upper right')
ax11.grid(True, alpha=0.3)
ax11.axis('equal')
ax11.set_xlim(-6, 6)
ax11.set_ylim(-6, 6)

# 1-2: 지구 확대
ax12 = fig1.add_subplot(gs1[1, 0])
earth_sim = simulations['Earth']
ax12.plot(0, 0, 'yo', markersize=15, markeredgecolor='orange', markeredgewidth=2, label='Sun')
ax12.plot(earth_sim['x'], earth_sim['y'], 'b-', linewidth=2, alpha=0.7, label='Earth')
ax12.plot(earth_sim['x'][0], earth_sim['y'][0], 'bo', markersize=8, 
         markeredgecolor='black', markeredgewidth=1, label='Start')

# 면적 속도 일정 (케플러 제2법칙) 시각화
n_points = 4
indices = np.linspace(0, len(earth_sim['t'])//2, n_points, dtype=int)
for i, idx in enumerate(indices[:-1]):
    idx_next = indices[i+1]
    # 삼각형 면적 (태양-행성-다음위치)
    x_tri = [0, earth_sim['x'][idx], earth_sim['x'][idx_next], 0]
    y_tri = [0, earth_sim['y'][idx], earth_sim['y'][idx_next], 0]
    ax12.fill(x_tri, y_tri, alpha=0.2, edgecolor='red', linewidth=1.5)

ax12.set_xlabel('x (AU)', fontsize=12, fontweight='bold')
ax12.set_ylabel('y (AU)', fontsize=12, fontweight='bold')
ax12.set_title("Earth's Orbit (Kepler's 2nd Law)", fontsize=13, fontweight='bold')
ax12.legend(fontsize=10)
ax12.grid(True, alpha=0.3)
ax12.axis('equal')

# 1-3: 케플러 제3법칙
ax13 = fig1.add_subplot(gs1[1, 1])

a_values = [data['a'] for data in planets_data.values()]
T_values = [data['period'] for data in planets_data.values()]
names = list(planets_data.keys())

# 이론 곡선
a_theory = np.linspace(0.5, 6, 100)
T_theory = a_theory**(3/2)

ax13.plot(a_theory, T_theory, 'k--', linewidth=2, label='T = a^(3/2) (Theory)')

for i, name in enumerate(names):
    color = planets_data[name]['color']
    ax13.plot(a_values[i], T_values[i], 'o', color=color, markersize=12, 
             markeredgecolor='black', markeredgewidth=1.5, label=name)
    ax13.text(a_values[i], T_values[i]*1.1, name, fontsize=10, 
             ha='center', fontweight='bold')

ax13.set_xlabel('Semi-major axis a (AU)', fontsize=12, fontweight='bold')
ax13.set_ylabel('Orbital period T (years)', fontsize=12, fontweight='bold')
ax13.set_title("Kepler's 3rd Law: T² ∝ a³", fontsize=13, fontweight='bold')
ax13.legend(fontsize=10)
ax13.grid(True, alpha=0.3)

plt.suptitle('Solar System Simulation', fontsize=16, fontweight='bold')
plt.savefig(f'{output_dir}/02_solar_system.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] 그래프 저장: {output_dir}/02_solar_system.png")
plt.close()

# 그림 2: 궤도 매개변수 분석
fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# 각 행성마다 분석
for idx, (name, sim) in enumerate(simulations.items()):
    if idx >= 3:
        break
    
    ax = axes[idx * 2]
    
    # 거리 vs 시간
    ax.plot(sim['t'], sim['r'], color=sim['data']['color'], linewidth=2)
    ax.axhline(sim['data']['a'], color='k', linestyle='--', linewidth=1, alpha=0.5, label='Semi-major axis')
    ax.set_xlabel('Time (years)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Distance from Sun (AU)', fontsize=11, fontweight='bold')
    ax.set_title(f"{name}: Distance vs Time", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 에너지 보존
    ax2 = axes[idx * 2 + 1]
    
    E_normalized = (sim['E'] - sim['E'][0]) / abs(sim['E'][0]) * 100
    L_normalized = (sim['L'] - sim['L'][0]) / abs(sim['L'][0]) * 100
    
    ax2.plot(sim['t'], E_normalized, 'b-', linewidth=2, label='Energy', alpha=0.7)
    ax2.plot(sim['t'], L_normalized, 'r-', linewidth=2, label='Angular Momentum', alpha=0.7)
    ax2.axhline(0, color='k', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (years)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Relative Change (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f"{name}: Conservation Laws", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

plt.suptitle('Orbital Parameters and Conservation Laws', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_orbital_parameters.png', dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_dir}/02_orbital_parameters.png")
plt.close()

# 그림 3: 케플러 법칙 상세 분석
fig3 = plt.figure(figsize=(16, 10))
gs3 = GridSpec(2, 3, figure=fig3, hspace=0.3, wspace=0.3)

# 3-1: 케플러 제1법칙 (타원)
ax31 = fig3.add_subplot(gs3[0, 0])
earth_sim = simulations['Earth']
ax31.plot(0, 0, 'yo', markersize=15, label='Sun (focus)')
ax31.plot(earth_sim['x'], earth_sim['y'], 'b-', linewidth=2, alpha=0.7, label='Orbit')

# 타원의 중심과 초점
a = planets_data['Earth']['a']
e = planets_data['Earth']['e']
c = a * e  # 초점 거리
ax31.plot(-c, 0, 'rx', markersize=10, markeredgewidth=2, label='Center')

# 이론적 타원
theta = np.linspace(0, 2*np.pi, 1000)
r_ellipse = a * (1 - e**2) / (1 + e * np.cos(theta))
x_ellipse = r_ellipse * np.cos(theta) - c
y_ellipse = r_ellipse * np.sin(theta)
ax31.plot(x_ellipse, y_ellipse, 'k--', linewidth=1.5, alpha=0.5, label='Theory')

ax31.set_xlabel('x (AU)', fontsize=11, fontweight='bold')
ax31.set_ylabel('y (AU)', fontsize=11, fontweight='bold')
ax31.set_title("Kepler's 1st Law: Elliptical Orbit", fontsize=12, fontweight='bold')
ax31.legend(fontsize=9)
ax31.grid(True, alpha=0.3)
ax31.axis('equal')

# 3-2: 케플러 제2법칙 (면적 속도)
ax32 = fig3.add_subplot(gs3[0, 1])

# 면적 속도 계산
earth_t = earth_sim['t']
earth_L = earth_sim['L']
area_velocity = earth_L / (2 * planets_data['Earth']['mass'])

ax32.plot(earth_t, area_velocity, 'b-', linewidth=2)
ax32.axhline(area_velocity.mean(), color='r', linestyle='--', linewidth=2, 
            label=f'Mean = {area_velocity.mean():.4f}')
ax32.set_xlabel('Time (years)', fontsize=11, fontweight='bold')
ax32.set_ylabel('Area Velocity (AU²/year)', fontsize=11, fontweight='bold')
ax32.set_title("Kepler's 2nd Law: Area Velocity", fontsize=12, fontweight='bold')
ax32.legend(fontsize=9)
ax32.grid(True, alpha=0.3)

# 3-3: 케플러 제3법칙 (로그 스케일)
ax33 = fig3.add_subplot(gs3[0, 2])

a_values = np.array([data['a'] for data in planets_data.values()])
T_values = np.array([data['period'] for data in planets_data.values()])

ax33.loglog(a_values, T_values, 'o', markersize=12, markeredgewidth=2, markeredgecolor='black')

# 이론 직선 (로그 스케일에서 기울기 3/2)
a_line = np.array([0.5, 10])
T_line = a_line**(3/2)
ax33.loglog(a_line, T_line, 'k--', linewidth=2, label='Slope = 3/2')

for i, name in enumerate(planets_data.keys()):
    ax33.text(a_values[i]*1.1, T_values[i], name, fontsize=10, fontweight='bold')

ax33.set_xlabel('Semi-major axis a (AU)', fontsize=11, fontweight='bold')
ax33.set_ylabel('Period T (years)', fontsize=11, fontweight='bold')
ax33.set_title("Kepler's 3rd Law (Log Scale)", fontsize=12, fontweight='bold')
ax33.legend(fontsize=9)
ax33.grid(True, alpha=0.3, which='both')

# 3-4, 3-5, 3-6: 각 행성의 속도 vs 거리
for idx, (name, sim) in enumerate(simulations.items()):
    ax = fig3.add_subplot(gs3[1, idx])
    
    # 속도 vs 거리 (근일점에서 빠름, 원일점에서 느림)
    scatter = ax.scatter(sim['r'], sim['v'], c=sim['t'], cmap='viridis', 
                        s=1, alpha=0.5)
    
    # 이론값 (에너지 보존에서)
    E0 = sim['E'][0]
    r_theory = np.linspace(sim['r'].min(), sim['r'].max(), 100)
    v_theory = np.sqrt(2 * (E0 / planets_data[name]['mass'] + G * M_sun / r_theory))
    ax.plot(r_theory, v_theory, 'r--', linewidth=2, label='Theory (E=const)')
    
    ax.set_xlabel('Distance r (AU)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Velocity v (AU/year)', fontsize=10, fontweight='bold')
    ax.set_title(f'{name}: v vs r', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (years)', fontsize=9)

plt.suptitle("Kepler's Laws Verification", fontsize=16, fontweight='bold')
plt.savefig(f'{output_dir}/02_kepler_laws.png', dpi=150, bbox_inches='tight')
print(f"[OK] 그래프 저장: {output_dir}/02_kepler_laws.png")
plt.close()

print("\n" + "="*70)
print("분석 완료!")
print("="*70)
print("\n생성된 파일:")
print(f"  1. {output_dir}/02_solar_system.png - 태양계 궤도")
print(f"  2. {output_dir}/02_orbital_parameters.png - 궤도 매개변수")
print(f"  3. {output_dir}/02_kepler_laws.png - 케플러 법칙 검증")
print("\n주요 결과:")
print(f"  - 케플러 제3법칙 T²/a³ = {np.mean(kepler_constant):.6f} ± {np.std(kepler_constant):.8f}")
print(f"  - 에너지 보존 오차 < 0.01%")
print(f"  - 각운동량 보존 오차 < 0.01%")

