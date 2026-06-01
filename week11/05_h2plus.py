"""
05. H2+ Molecular Ion - Electron Wave Function
H2+ 수소 분자 이온: 전자 1개의 파동함수(분자 오비탈)를 수치적으로 구한다.

Born-Oppenheimer 근사: 양성자 2개를 거리 R 만큼 떨어뜨려 고정하고,
전자의 시간 독립 슈뢰딩거 방정식을 푼다.

    Ĥ = -½∇²  - 1/r_A - 1/r_B
    E_total(R) = ε(R) + 1/R   (양성자-양성자 반발 포함)

두 가지 방법:
  A. LCAO 변분법  -> E(R) 곡선, 평형 결합길이, 결합/반결합
  B. 3D 유한차분(희소행렬) -> 실제 분자 오비탈 시각화

원자 단위(atomic units): ℏ = m_e = e = 1.  길이 a0, 에너지 Hartree.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import os

plt.rcParams['axes.unicode_minus'] = False

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

HARTREE_TO_EV = 27.211386
A0_TO_ANGSTROM = 0.529177

print("=" * 70)
print("H2+ Molecular Ion - Electron Wave Function")
print("=" * 70)


# ======================================================================
# A. LCAO 변분법 (Linear Combination of Atomic Orbitals)
# ======================================================================
# 분자 오비탈 ψ = c_A φ_A + c_B φ_B,  φ = (1/√π) e^(-r)  (수소 1s)
# 2x2 일반화 고유값 문제 H c = ε S c 의 해석해:
#   ε_bonding = (H_AA + H_AB)/(1 + S)
#   ε_anti    = (H_AA - H_AB)/(1 - S)
# 표준 2중심 적분(원자 단위):
#   S    = e^(-R)(1 + R + R²/3)
#   H_AA = -1/2 - 1/R + (1 + 1/R) e^(-2R)
#   H_AB = -1/2 S - (1 + R) e^(-R)

def lcao_energies(R):
    S = np.exp(-R) * (1 + R + R**2 / 3)
    H_AA = -0.5 - 1.0 / R + (1 + 1.0 / R) * np.exp(-2 * R)
    H_AB = -0.5 * S - (1 + R) * np.exp(-R)
    eps_b = (H_AA + H_AB) / (1 + S)      # bonding 전자 에너지
    eps_a = (H_AA - H_AB) / (1 - S)      # antibonding 전자 에너지
    return eps_b, eps_a, S


print("\n" + "=" * 70)
print("A. LCAO 변분법: E(R) 곡선과 평형 결합길이")
print("=" * 70)

R_vals = np.linspace(0.4, 8.0, 400)
eps_b, eps_a, S = lcao_energies(R_vals)
E_total_b = eps_b + 1.0 / R_vals          # 결합: 핵반발 포함 총에너지
E_total_a = eps_a + 1.0 / R_vals          # 반결합

# 평형 결합길이 = 결합 총에너지의 최소점
i_min = np.argmin(E_total_b)
R_e = R_vals[i_min]
E_min = E_total_b[i_min]
E_dissoc = -0.5            # H(-0.5 Hartree) + p(0)
D_e = (E_dissoc - E_min)   # 해리에너지 (양수)

print(f"  평형 결합길이 R_e = {R_e:.3f} a0 = {R_e * A0_TO_ANGSTROM:.3f} Å")
print(f"  평형 총에너지   E  = {E_min:.4f} Hartree = {E_min * HARTREE_TO_EV:.2f} eV")
print(f"  해리에너지     D_e = {D_e:.4f} Hartree = {D_e * HARTREE_TO_EV:.2f} eV")
print(f"  (정확값 참고: R_e=2.00 a0, D_e=2.79 eV;  단순 LCAO(ζ=1)는 약간 과대평가)")

# --- 그림 1: 에너지 곡선 ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(R_vals, E_total_b, 'b-', lw=2, label='Bonding  E_total(R)')
ax.plot(R_vals, E_total_a, 'r--', lw=2, label='Antibonding  E_total(R)')
ax.axhline(-0.5, color='gray', ls=':', label='H + p+ dissociation limit (-0.5 Ha)')
ax.plot(R_e, E_min, 'ko', ms=9)
ax.annotate(f'  R_e = {R_e:.2f} a0\n  D_e = {D_e*HARTREE_TO_EV:.2f} eV',
            xy=(R_e, E_min), xytext=(R_e + 1.2, E_min + 0.05),
            arrowprops=dict(arrowstyle='->'))
ax.set_xlabel('Internuclear distance R (a0)')
ax.set_ylabel('Total energy (Hartree)')
ax.set_title('H2+ Potential Energy Curve (LCAO)')
ax.set_ylim(-0.65, 0.1)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{output_dir}/05_h2plus_energy_curve.png', dpi=120)
plt.close(fig)
print(f"  ✓ 저장: {output_dir}/05_h2plus_energy_curve.png")


# ======================================================================
# B. 3D 유한차분(희소행렬)로 실제 분자 오비탈 구하기
# ======================================================================
print("\n" + "=" * 70)
print("B. 3D 유한차분: 실제 분자 오비탈 (R = 2.0 a0)")
print("=" * 70)

R = 2.0                       # 평형 근처 결합길이
N = 70                        # 축당 격자점
L = 8.0                       # 박스 반폭 (-L..L)
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

# 양성자를 z축 위 가장 가까운 격자점에 배치 (대칭)
zc = R / 2
iA = np.argmin(np.abs(x - zc))
iB = np.argmin(np.abs(x + zc))
zA, zB = x[iA], x[iB]

X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
eps_soft = 0.5 * dx           # Coulomb 특이점 완화 (soft-core)
rA = np.sqrt(X**2 + Y**2 + (Z - zA)**2 + eps_soft**2)
rB = np.sqrt(X**2 + Y**2 + (Z - zB)**2 + eps_soft**2)
V = (-1.0 / rA - 1.0 / rB).ravel()

# 1D 2차미분 (운동에너지) -> kron으로 3D 라플라시안
main = -2.0 * np.ones(N)
off = np.ones(N - 1)
D2 = sp.diags([off, main, off], [-1, 0, 1]) / dx**2
I = sp.identity(N)
Lap = (sp.kron(sp.kron(D2, I), I)
       + sp.kron(sp.kron(I, D2), I)
       + sp.kron(sp.kron(I, I), D2))
H = (-0.5 * Lap + sp.diags(V)).tocsr()

print(f"  격자 {N}^3 = {N**3} 점, 해밀토니안 {H.shape[0]}x{H.shape[0]} (희소)")
print("  eigsh로 최저 2개 상태(결합/반결합) 계산 중...")
E, psi = eigsh(H, k=2, which='SA')
idx = np.argsort(E)
E, psi = E[idx], psi[:, idx]
E_grid_total = E + 1.0 / R    # 핵반발 포함

print(f"  바닥상태(결합)   ε = {E[0]:.4f} Ha,  E_total = {E_grid_total[0]:.4f} Ha = {E_grid_total[0]*HARTREE_TO_EV:.2f} eV")
print(f"  첫들뜬(반결합)   ε = {E[1]:.4f} Ha,  E_total = {E_grid_total[1]:.4f} Ha")
print("  (soft-core·유한격자 근사라 LCAO보다 더 깊게/거칠게 나올 수 있음; 목적은 오비탈 형태)")

psi0 = psi[:, 0].reshape(N, N, N)
psi1 = psi[:, 1].reshape(N, N, N)
iy0 = N // 2                                  # y=0 단면 (xz 평면)
sl0 = psi0[:, iy0, :]
sl1 = psi1[:, iy0, :]

# --- 그림 2: 분자 오비탈 단면 ---
fig = plt.figure(figsize=(13, 6))
gs = GridSpec(1, 2, figure=fig, wspace=0.25)
extent = [-L, L, -L, L]   # [z(가로), x(세로)] -> imshow는 행=x, 열=z

for ax_i, (sl, title) in enumerate([
        (sl0, f'Bonding orbital  |psi|^2\n(electron density between nuclei)  eps={E[0]:.3f} Ha'),
        (sl1, f'Antibonding orbital  |psi|^2\n(node at center)  eps={E[1]:.3f} Ha')]):
    ax = fig.add_subplot(gs[0, ax_i])
    dens = np.abs(sl)**2          # 인덱싱 [ix, iz]: 행=x(세로), 열=z(가로, 결합축)
    im = ax.imshow(dens, extent=extent, origin='lower',
                   cmap='inferno', aspect='equal')
    ax.plot([zA, zB], [0, 0], 'c+', ms=14, mew=2, label='protons')
    ax.set_xlabel('z (a0)')
    ax.set_ylabel('x (a0)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle('H2+ Molecular Orbitals (3D finite-difference, xz-plane slice)')
fig.tight_layout()
fig.savefig(f'{output_dir}/05_h2plus_orbitals.png', dpi=120)
plt.close(fig)
print(f"  ✓ 저장: {output_dir}/05_h2plus_orbitals.png")

print("\n" + "=" * 70)
print("H2+ 완료: 결합 오비탈은 두 핵 사이에 전자밀도 집중 -> 결합 형성,")
print("          반결합 오비탈은 중앙에 노드(node) 존재.")
print("=" * 70)
