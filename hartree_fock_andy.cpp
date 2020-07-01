#include <iostream>
#include <fstream>
#include <cmath>
#include "eigen-3.3.7/Eigen/Dense"
#include "eigen-3.3.7/Eigen/Eigenvalues"

using namespace std;
using namespace Eigen;

Matrix<complex<double>, 7, 7> getMatrix(string filename) {

    ifstream m_file;
    m_file.open(filename);

    Matrix<complex<double>, 7, 7> m;

    string line;

    if (m_file.is_open()) {
        do {
            int i;
            int j;
            double val;
            m_file >> i >> j >> val;

            m(i-1,j-1) = (complex<double>) val;
            m(j-1,i-1) = (complex<double>) val;

        } while (getline(m_file, line));
    }

    m_file.close();

    cout << m << endl;

    return m;
}

int main() {

    int max_iter = 40;
    double convergence = 1.0e-6;

    ifstream nuc_repulsion_file;
    nuc_repulsion_file.open("input/h2o/STO-3G/enuc.dat");

    double nuc_repulsion_energy;
    nuc_repulsion_file >> nuc_repulsion_energy;
    nuc_repulsion_file.close();

    cout << nuc_repulsion_energy << endl;

    Matrix<complex<double>, 7, 7> s = getMatrix("input/h2o/STO-3G/s.dat"); //Overlap Integrals
    Matrix<complex<double>, 7, 7> t = getMatrix("input/h2o/STO-3G/t.dat"); //Kinetic energy
    Matrix<complex<double>, 7, 7> v = getMatrix("input/h2o/STO-3G/v.dat"); //Nuclear attraction

    Matrix<complex<double>, 7, 7> h = t + v;

    complex<double> eri[7][7][7][7];

    ifstream eri_file;
    eri_file.open("input/h2o/STO-3G/eri.dat"); //Electron repulsion integrals

    string line;

    if (eri_file.is_open()) {
        do {
            int i;
            int j;
            int k;
            int l;

            double val;
            eri_file >> i >> j >> k >> l >> val;

            i -= 1;
            j -= 1;
            k -= 1;
            l -= 1;

            eri[i][j][k][l] = (complex<double>) val;
            eri[j][i][k][l] = (complex<double>) val;
            eri[i][j][l][k] = (complex<double>) val;
            eri[j][i][l][k] = (complex<double>) val;
            eri[k][l][i][j] = (complex<double>) val;
            eri[l][k][i][j] = (complex<double>) val;
            eri[k][l][j][i] = (complex<double>) val;
            eri[l][k][j][i] = (complex<double>) val;

        } while (getline(eri_file, line));
    }

    eri_file.close();

    SelfAdjointEigenSolver<MatrixXcd> eig;
    eig.compute(s);

    cout << eig.eigenvalues() << endl;
    cout << eig.eigenvectors() << endl;

    Matrix<complex<double>, 7, 7> Ls = eig.eigenvectors();
    Matrix<complex<double>, 7, 7> lambda = eig.eigenvalues().asDiagonal();

    for (int i = 0; i < lambda.cols(); i++) {
        lambda(i, i) = 1.0/(sqrt(lambda(i, i)));
    }

    MatrixXcd s_tmh = Ls * lambda * Ls.transpose();

    cout << "S^(-0.5)\n" << s_tmh << endl;

    //TIME FOR THE REAL FUN!!!!
    MatrixXcd F0_p = s_tmh.transpose() * h * s_tmh;

    cout << "Fock matrix\n" << F0_p << endl;

    SelfAdjointEigenSolver<MatrixXcd> eig2;

    eig2.compute(F0_p);

    cout << "C0' Eigenvalues\n" << eig2.eigenvalues() << endl;

    Matrix<complex<double>, 7, 7> C0_p = eig2.eigenvectors();

    cout << "C0_p\n" << C0_p << endl;

    MatrixXcd C_i = s_tmh * C0_p;

    cout << "C0\n" << C_i << endl;

    MatrixXcd D_i(7, 7);

    for (int u = 0; u < 7; u++) {
        for (int v = 0; v < 7; v++) {
            D_i(u, v) = 0.0;
            for (int m = 0; m < 5; m++) {
                D_i(u, v) += C_i(u, m) * C_i(v, m);
            }
        }
    }

    cout << "D_i\n" << D_i << endl;

    complex<double> E_n(0.0, 0.0);

    for (int u = 0; u < 7; u++) {
        for (int v = 0; v < 7; v++) {
            E_n += D_i(u,v) * (h(u,v) + F0_p(u, v));
        }
    }

    E_n += (complex<float>) nuc_repulsion_energy;

    cout << E_n << endl;

    int iter = 0;

    while (iter < max_iter) {

        MatrixXcd F_n;

        MatrixXcd J_c = MatrixXcd::Zero(7, 7);
        MatrixXcd K_e = MatrixXcd::Zero(7, 7);

        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                for (int k = 0; k < 7; k++) {
                    for (int l = 0; l < 7; l++) {
                        J_c(i, j) += 2.0 * D_i(k, l) * eri[i][j][k][l];
                        K_e(i, j) -= D_i(k, l) * eri[i][k][j][l];
                    }
                }
            }
        }

        F_n = h + J_c + K_e;

        cout << "F\n" << F_n << endl;

        MatrixXcd F_p = s_tmh.transpose() * F_n * s_tmh;

        eig2.compute(F_p);

        MatrixXcd C_p = eig2.eigenvectors();
        MatrixXcd C_n = s_tmh * C_p;

        MatrixXcd D_prev = D_i;

        D_i = MatrixXcd::Zero(7, 7);

        for (int u = 0; u < 7; u++) {
            for (int v = 0; v < 7; v++) {
                for (int m = 0; m < 5; m++) {
                    D_i(u, v) += C_n(u, m) * C_n(v, m);
                }
            }
        }

        complex<double> E_prev = E_n;

        complex<double> E_temp(0.0, 0.0);

        E_n = E_temp;

        for (int u = 0; u < 7; u++) {
            for (int v = 0; v < 7; v++) {
                E_n += D_i(u,v) * (h(u,v) + F_n(u, v));
            }
        }

        E_n += (complex<double>) nuc_repulsion_energy;

        cout << "Energy at Iteration " << iter << ": " << E_n << endl;

        if (norm(E_n - E_prev) <= convergence) {
            cout << "Energy Converged!!! Buy Andy a ice cream!!!" << endl;
            break;
        }

        iter++;

    }

    return 0;

}
