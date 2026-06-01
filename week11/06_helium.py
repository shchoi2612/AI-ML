"""
06. Neutral Helium Atom (2 electrons) - Many-Body Quantum Problem
중성 He(전자 2개)의 양자역학 문제를 수치적으로 푼다.

He+ (전자 1개)는 수소형이라 해석해가 있지만, 중성 He는 전자-전자 반발항
1/r12 때문에 파동함수가 6차원이고 변수분리가 안 되는 '다체 문제'다.

    Ĥ = -½∇₁² - ½∇₂²  - 2/r₁ - 2/r₂  + 1/r₁₂

두 가지 방법:
  1. 단순 변분법 (해석)        : 유효전하 Z_eff 의 1s 곱  -> 가림(screening) 효과
  2. 변분 몬테카를로 VMC        : Jastrow 인자 + Metropolis 샘플링 -> 전자상관 포함

원자 단위(ℏ=m=e=1). 에너지 Hartree.  정확값 E0 = -2.9037 Ha = -79.0 eV.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import os

plt.rcParams['axes.unicode_minus'] = False

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

HARTREE_TO_EV = 27.211386
E_EXACT = -2.90372       # He 바닥상태 정확값 (Hartree)
E_HE_PLUS = -2.0         # He+ 바닥상태 (Z=2 수소형)

rng = np.random.default_rng(42)

print("=" * 70)
print("Neutral Helium (2 electrons) - Many-Body Quantum Problem")
print("=" * 70)


# ======================================================================
# 1. 단순 변분법 (해석): 유효전하 Z_eff
# ======================================================================
# 시험함수 ψ = e^(-Z r1) e^(-Z r2) 에 대한 에너지 기댓값 해석식:
#     E(Z) = Z² - (27/8) Z
# 최소화하면 가림(screening) 때문에 Z_eff = 27/16 < 2 가 나온다.
print("\n" + "=" * 70)
print("1. 단순 변분법 (해석): 유효전하 Z_eff")
print("=" * 70)


def E_variational(Z):
    return Z**2 - (27.0 / 8.0) * Z


res = minimize_scalar(E_variational, bounds=(1.0, 2.5), method='bounded')
Z_eff = res.x
E_var = res.fun
print(f"  가림 무시 (Z=2)   : E = {E_variational(2.0):+.4f} Ha = {E_variational(2.0)*HARTREE_TO_EV:.2f} eV")
print(f"  최적 유효전하 Z_eff = {Z_eff:.4f}  (= 27/16 = {27/16:.4f})")
print(f"  변분 에너지        : E = {E_var:+.4f} Ha = {E_var*HARTREE_TO_EV:.2f} eV")
print(f"  -> 전자가 서로를 가려 유효 핵전하가 2보다 작아진다.")


# ======================================================================
# 2. 변분 몬테카를로 (VMC) - Metropolis + Jastrow 상관인자
# ======================================================================
# 시험함수: ψ = e^(-Z r1) e^(-Z r2) exp( r12 / (2(1 + b r12)) )
#   - 마지막 Jastrow 항이 두 전자의 상관(correlation)을 표현.
# 6차원 적분 ⟨E⟩ = ∫|ψ|² E_L / ∫|ψ|² 를 Metropolis 샘플링으로 계산.
#   - 국소에너지 E_L = (Ĥψ)/ψ.  운동항은 log-ψ의 수치 라플라시안으로 계산(안전).
print("\n" + "=" * 70)
print("2. 변분 몬테카를로 (VMC): Metropolis + Jastrow 상관인자")
print("=" * 70)

Z_J = 2.0          # Jastrow VMC에서는 지수부 전하를 2로 고정하고 b만 변분


def log_psi(r1, r2, b):
    """log ψ.  r1,r2: (M,3).  반환 (M,)"""
    a1 = np.linalg.norm(r1, axis=1)
    a2 = np.linalg.norm(r2, axis=1)
    r12 = np.linalg.norm(r1 - r2, axis=1)
    return -Z_J * a1 - Z_J * a2 + r12 / (2.0 * (1.0 + b * r12))


def local_energy(r1, r2, b, h=1e-4):
    """E_L = -½(∇²ψ)/ψ + V.  운동항은 log-ψ 기반 수치 라플라시안."""
    lp0 = log_psi(r1, r2, b)
    # 운동항: 6개 좌표에 대한 (ψ(+h)-2ψ+ψ(-h))/h² / ψ = Σ[exp(Δ+)-2+exp(Δ-)]/h²
    lap_over_psi = np.zeros(len(r1))
    for elec, r in ((1, r1), (2, r2)):
        for d in range(3):
            dr = np.zeros_like(r)
            dr[:, d] = h
            if elec == 1:
                lp_p = log_psi(r1 + dr, r2, b)
                lp_m = log_psi(r1 - dr, r2, b)
            else:
                lp_p = log_psi(r1, r2 + dr, b)
                lp_m = log_psi(r1, r2 - dr, b)
            lap_over_psi += (np.exp(lp_p - lp0) - 2.0 + np.exp(lp_m - lp0)) / h**2
    kinetic = -0.5 * lap_over_psi

    a1 = np.linalg.norm(r1, axis=1)
    a2 = np.linalg.norm(r2, axis=1)
    r12 = np.linalg.norm(r1 - r2, axis=1)
    potential = -2.0 / a1 - 2.0 / a2 + 1.0 / r12
    return kinetic + potential


def run_vmc(b, n_walkers=2000, n_steps=4000, n_burn=1000, step=0.4, measure_every=5):
    """Metropolis VMC: |ψ|² 분포를 샘플하고 E_L 평균. 반환 (E, err, accept, r12_samples)."""
    r1 = rng.normal(scale=1.0, size=(n_walkers, 3))
    r2 = rng.normal(scale=1.0, size=(n_walkers, 3))
    lp = log_psi(r1, r2, b)

    energies, r12_acc = [], []
    n_acc = 0
    for s in range(n_steps):
        # 두 전자를 동시에 흔들어 제안
        r1n = r1 + step * rng.normal(size=r1.shape)
        r2n = r2 + step * rng.normal(size=r2.shape)
        lpn = log_psi(r1n, r2n, b)
        acc = rng.random(n_walkers) < np.exp(2.0 * (lpn - lp))   # min(1,|ψ'/ψ|²)
        r1[acc], r2[acc], lp[acc] = r1n[acc], r2n[acc], lpn[acc]
        n_acc += acc.sum()

        if s >= n_burn and (s - n_burn) % measure_every == 0:
            EL = local_energy(r1, r2, b)
            energies.append(EL.mean())
            r12_acc.append(np.linalg.norm(r1 - r2, axis=1))

    energies = np.array(energies)
    E_mean = energies.mean()
    E_err = energies.std() / np.sqrt(len(energies))
    accept = n_acc / (n_steps * n_walkers)
    return E_mean, E_err, accept, np.concatenate(r12_acc)


b_scan = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
E_scan, err_scan = [], []
r12_best, b_best, E_best, err_best = None, None, 0.0, 0.0
print("   b      E (Ha)     ±err      accept")
print("  " + "-" * 40)
for b in b_scan:
    E, err, acc, r12s = run_vmc(b)
    E_scan.append(E)
    err_scan.append(err)
    print(f"  {b:.2f}   {E:+.4f}   {err:.4f}    {acc*100:.0f}%")
    if E < E_best:
        E_best, err_best, b_best, r12_best = E, err, b, r12s

E_scan = np.array(E_scan)
err_scan = np.array(err_scan)
print("  " + "-" * 40)
print(f"  최적 b = {b_best:.2f},  E_VMC = {E_best:+.4f} ± {err_best:.4f} Ha "
      f"= {E_best*HARTREE_TO_EV:.2f} eV")


# ======================================================================
# 결과 비교 + 이온화 에너지
# ======================================================================
print("\n" + "=" * 70)
print("결과 비교 (He 바닥상태 에너지)")
print("=" * 70)
print(f"  {'방법':<26}{'E (Hartree)':>14}{'E (eV)':>12}")
print("  " + "-" * 52)
rows = [
    ("Z=2 (가림 무시)", E_variational(2.0)),
    ("변분법 (Z_eff=27/16)", E_var),
    (f"VMC (Jastrow, b={b_best:.2f})", E_best),
    ("정확값 (실험/이론)", E_EXACT),
]
for name, e in rows:
    # 한글 정렬 폭 보정
    pad = 26 - sum(2 if ord(c) > 0x1100 else 1 for c in name)
    print(f"  {name}{' '*max(pad,1)}{e:>13.4f}{e*HARTREE_TO_EV:>12.2f}")

IE = (E_HE_PLUS - E_best)   # He -> He+ + e :  E(He+) - E(He)
IE_exact = (E_HE_PLUS - E_EXACT)
print("  " + "-" * 52)
print(f"  첫 이온화에너지 (VMC) = {IE:.4f} Ha = {IE*HARTREE_TO_EV:.2f} eV")
print(f"  첫 이온화에너지 (정확)= {IE_exact:.4f} Ha = {IE_exact*HARTREE_TO_EV:.2f} eV  (실험 24.59 eV)")


# ======================================================================
# 그림
# ======================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) 변분 곡선 E(Z)
Zs = np.linspace(1.2, 2.3, 200)
axes[0].plot(Zs, E_variational(Zs), 'b-', lw=2)
axes[0].plot(Z_eff, E_var, 'ro', ms=9, label=f'min: Z_eff={Z_eff:.3f}')
axes[0].axhline(E_EXACT, color='g', ls='--', label=f'exact {E_EXACT:.3f} Ha')
axes[0].set_xlabel('effective charge Z_eff')
axes[0].set_ylabel('E (Hartree)')
axes[0].set_title('(a) Simple variational  E(Z_eff)')
axes[0].legend()
axes[0].grid(alpha=0.3)

# (b) VMC 에너지 vs Jastrow b
axes[1].errorbar(b_scan, E_scan, yerr=err_scan, fmt='o-', color='purple',
                 capsize=4, label='VMC (Jastrow)')
axes[1].axhline(E_var, color='b', ls=':', label=f'simple var {E_var:.3f}')
axes[1].axhline(E_EXACT, color='g', ls='--', label=f'exact {E_EXACT:.3f}')
axes[1].plot(b_best, E_best, 'r*', ms=16, label=f'best b={b_best:.2f}')
axes[1].set_xlabel('Jastrow parameter b')
axes[1].set_ylabel('E (Hartree)')
axes[1].set_title('(b) VMC energy vs correlation parameter')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# (c) 전자간 거리 r12 분포 (상관 효과)
axes[2].hist(r12_best, bins=80, density=True, color='teal', alpha=0.7)
axes[2].axvline(r12_best.mean(), color='r', ls='--',
                label=f'<r12> = {r12_best.mean():.2f} a0')
axes[2].set_xlabel('inter-electron distance r12 (a0)')
axes[2].set_ylabel('probability density')
axes[2].set_title('(c) e-e distance distribution (correlation)')
axes[2].set_xlim(0, 8)
axes[2].legend()
axes[2].grid(alpha=0.3)

fig.suptitle('Neutral Helium: Variational + Variational Monte Carlo')
fig.tight_layout()
fig.savefig(f'{output_dir}/06_he_vmc.png', dpi=120)
plt.close(fig)
print(f"\n  ✓ 저장: {output_dir}/06_he_vmc.png")

print("\n" + "=" * 70)
print("He 완료: 단순 변분(-2.85) -> Jastrow VMC(상관 포함) 로 정확값(-2.90)에 접근.")
print("         1/r12 비분리항 때문에 격자법 대신 Monte Carlo가 필요했다.")
print("=" * 70)
