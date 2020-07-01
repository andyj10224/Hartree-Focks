import psi4
import numpy as np
from numpy import linalg
from numpy import matrix

def simple_scf(mol, max_iter, threshold):

    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("basis"))
    mints = psi4.core.MintsHelper(wfn.basisset())

    E_nuc = mol.nuclear_repulsion_energy()

    S = mints.ao_overlap()
    V = mints.ao_potential()
    T = mints.ao_kinetic()
    I = mints.ao_eri()

    H = np.asarray(T) + np.asarray(V)

    evecs = psi4.core.Matrix(S.shape[0], S.shape[0])
    evals = psi4.core.Vector(S.shape[0])

    S.diagonalize(evecs, evals, psi4.core.DiagonalizeOrder.Ascending)

    U = np.asarray(evecs)
    Ut = psi4.core.Matrix.from_array(np.matrix.getH(U))
    L = psi4.core.Matrix.from_array(np.diag(evals))

    La = psi4.core.Matrix(L.shape[0], L.shape[0])
    L.copy(La)
    La.power(-0.5, 0.0)

    #S_tmh = np.dot(np.dot(np.asarray(U), np.asarray(La)), np.asarray(Ut))

    S_tmh = mints.ao_overlap()
    S_tmh.power(-0.5, 0.0)

    S_tmh = np.asarray(S_tmh)

    F_0_p = np.dot(np.dot(np.matrix.getH(S_tmh), H), S_tmh)

    F_0_p = psi4.core.Matrix.from_array(F_0_p)

    evecs1 = psi4.core.Matrix(F_0_p.shape[0], F_0_p.shape[0])
    evals1 = psi4.core.Vector(F_0_p.shape[0])

    F_0_p.diagonalize(evecs1, evals1, psi4.core.DiagonalizeOrder.Ascending)

    C_0_p = evecs

    C_0 = np.dot(S_tmh, np.asarray(C_0_p))

    ndocc = wfn.nalpha()

    C_0_occ = C_0[:, :ndocc]

    D = np.einsum('ui,vi->uv', C_0_occ, C_0_occ, optimize=True)
    I = np.asarray(I)

    E_old = 0.0

    for i in range(max_iter + 1):
        J = np.einsum('rs,pqrs->pq', D, I, optimize=True)
        K = np.einsum('rs,prqs->pq', D, I, optimize=True)

        F = H + 2*J - K

        E_new = np.einsum('uv,uv->', D, (H + F), optimize=True) + E_nuc

        if abs(E_new - E_old) < threshold:
            return E_new

        E_old = E_new

        F_p = np.dot(np.dot(np.matrix.getH(S_tmh), F), S_tmh)

        F_p = psi4.core.Matrix.from_array(F_p)

        evecsn = psi4.core.Matrix(F_p.shape[0], F_p.shape[0])
        evalsn = psi4.core.Vector(F_p.shape[0])

        F_p.diagonalize(evecsn, evalsn)

        C_p = np.asarray(evecsn)

        C = np.dot(S_tmh, C_p)

        D = np.einsum('ui,vi->uv', C[:,:ndocc], C[:,:ndocc], optimize=True)

        if i == max_iter:
            raise Exception(F"Convergence failed in {max_iter} iterations")

mol = psi4.geometry("""
C 0 0 0
O 0 0 1.16
O 0 0 -1.16
symmetry c1
""")

psi4.set_options({'basis': 'cc-pvdz',
                  'scf_type': 'pk',
                  'e_convergence': 1e-8})

psi4_EN = psi4.energy('SCF')

psi4.compare_values(psi4_EN, simple_scf(mol, 40, 1.0e-6), 6, 'SCF Energy')
