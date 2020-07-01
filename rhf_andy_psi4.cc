/*
 * @BEGIN LICENSE
 *
 * rhf_andy by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2019 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/vector.h"
#include "psi4/libmints/mintshelper.h"

namespace psi{ namespace rhf_andy {

extern "C" PSI_API
int read_options(std::string name, Options &options)
{
    if (name == "RHF_ANDY"|| options.read_globals()) {
        /*- The amount of information printed
            to the output file -*/
        options.add_int("PRINT", 1);
        /*- Whether to compute two-electron integrals -*/
        options.add_bool("DO_TEI", true);
    }

    return true;
}

extern "C" PSI_API
SharedWavefunction rhf_andy(SharedWavefunction ref_wfn, Options& options) {
    // Grab options from the options object
    int print = options.get_int("PRINT");
    int doTei = options.get_bool("DO_TEI");
    double convergence = options.get_double("E_CONVERGENCE");

    int max_iter = 100;

    // Have the Wavefunction from python-side

    // Molecule is a member of the wavefunction option
    // Lets print out some molecule information here
    ref_wfn->molecule()->print();

    int num_alpha = ref_wfn->nalpha();

    // Compute the nuclear repulsion energy under a neutral field
    double nucrep = ref_wfn->molecule()->nuclear_repulsion_energy({0, 0, 0});
    psi::outfile->Printf("\n    Nuclear repulsion energy: %16.8f\n\n", nucrep);

    // MintsHelper are convenient objects that take a basisset, options, and print level
    // After this object is formed we can request a variety of matrix types
    MintsHelper mints(MintsHelper(ref_wfn->basisset(), options, 0));

    // mints is a reference, so we use the "." operator to access this object
    SharedMatrix sMat = mints.ao_overlap();
    SharedMatrix tMat = mints.ao_kinetic();
    SharedMatrix vMat = mints.ao_potential();

    // The SharedMatrix class is a std::shared_ptr to a Matrix
    // Since we have a point to the object we can access it through the "->" operator
    sMat->print();
    tMat->print();
    vMat->print();

    // Now that we have these matrices we can manipulate them in various ways
    // Form h = T + V by first cloning T and then adding V
    SharedMatrix hMat = tMat->clone();
    hMat->add(vMat);

    // Before we print lets give hMat a name so that we know what it is
    hMat->set_name("Core Hamiltonian Matrix");
    hMat->print();

    // We can build the two-electron integrals in a similar manner
        // As a note building the ERI's in this way is typically for debugging purposes and not normally recommended

        // First lets make sure this Matrix is not too large, lets stop at 100 basis function
    size_t nbf = ref_wfn->basisset()->nbf();

    if (nbf > 100){
        throw PSIEXCEPTION("There are too many basis function to construct the two-electron integrals!");
    }

    SharedMatrix eri = mints.ao_eri();

    //psi::outfile->Printf("To HECK with georgia!!!\n\n");

    int rows = hMat->nrow();
    int cols = hMat->ncol();

    //eri->print();

    int eri_rows = eri->nrow();
    int eri_cols = eri->ncol();

    //psi::outfile->Printf("%d %d\n\n", rows, cols);

    //psi::outfile->Printf("%d %d\n\n", eri_rows, eri_cols);

    Matrix uMat = Matrix(rows, cols);
    Vector lambda = Vector(rows);

    uMat.set(0.0);
    lambda.zero();

    sMat->diagonalize(uMat, lambda);

    Matrix b_lambda = Matrix(rows, cols);
    b_lambda.set(0.0);

    for (int i = 0; i < rows; i++) {
        b_lambda.set(i, i, 1.0/sqrt(lambda.get(i)));
    }

    Matrix S_tmh = Matrix(rows, cols);
    S_tmh.set(0.0);

    S_tmh.gemm(false, false, 1.0, uMat, b_lambda, 0.0);

    S_tmh.gemm(false, false, 1.0, S_tmh, uMat.transpose(), 0.0);

    int iter = 0;

    Matrix D_i = Matrix(rows, cols);
    D_i.set(0.0);

    Matrix J_c = Matrix(rows, cols);
    Matrix K_e = Matrix(rows, cols);
    Matrix F_i = Matrix(rows, cols);
    Matrix F_i_p = Matrix(rows, cols);
    Matrix C_i_p = Matrix(rows, cols);
    Vector eps = Vector(rows);
    Matrix C_i = Matrix(rows, cols);

    double energy = 0.0;

    while (iter < max_iter) {

        //psi::outfile->Printf("Entering Loop!!!\n");

        J_c.set(0.0);
        K_e.set(0.0);

        if (iter > 0) {
            for (int p = 0; p < rows; p++) {
                for (int q = 0; q < rows; q++) {
                    for (int r = 0; r < rows; r++) {
                        for (int s = 0; s < rows; s++) {
                            double val1 = J_c.get(p,q) + 2.0 * D_i.get(r,s) * eri->get(rows*p + q, rows*r + s);
                            J_c.set(p, q, val1);
                            double val2 = K_e.get(p,q) - D_i.get(r,s) * eri->get(rows*p + r, rows*q + s);
                            K_e.set(p, q, val2);
                        }
                    }
                }
            }
        }


        F_i.set(0.0);
        F_i.add(*(hMat->clone()));
        F_i.add(J_c);
        F_i.add(K_e);

        double old_energy = energy;

        energy = 0.0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                energy += D_i.get(i, j) * (hMat->get(i, j) + F_i.get(i, j));
            }
        }

        energy += nucrep;

        psi::outfile->Printf("Energy at iteration %d: %16.8f\n", iter + 1, energy);

        if (fabs(energy - old_energy) < convergence) {
            psi::outfile->Printf("Energy converged. Andy gets ice cream today!!!\n\n");
            break;
        }

        //psi::outfile->Printf("John 16:33\n");

        F_i_p.set(0.0);

        F_i_p.gemm(false, false, 1.0, S_tmh.transpose(), F_i, 0.0);
        F_i_p.gemm(false, false, 1.0, F_i_p, S_tmh, 0.0);

        //psi::outfile->Printf("GEORGIA TECH IS OUT FOR THE VICTORY!!!\n");

        C_i_p.set(0.0);
        eps.zero();

        F_i_p.diagonalize(C_i_p, eps);

        //psi::outfile->Printf("WHITE AND GOLD!!!\n");

        C_i.set(0.0);

        C_i.gemm(false, false, 1.0, S_tmh, C_i_p, 0.0);

        D_i.set(0.0);

        //psi::outfile->Printf("RAMBLIN WRECK!!!\n");

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < rows; j++) {
                double sum = 0.0;
                for (int k = 0; k < num_alpha; k++) {
                    sum += C_i.get(i,k) * C_i.get(j,k);
                }
                D_i.set(i, j, sum);
            }
        }

        //psi::outfile->Printf("PSI4 RULES!!!\n");

        iter += 1;

        //psi::outfile->Printf("DR. SHERRILL IS THE BEST!!!\n");

    }

    return ref_wfn;
}

}} // End Namespaces
