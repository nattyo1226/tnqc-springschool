#[allow(unused_imports)]
use blas_src as _;

use anyhow::Result;
use ndarray::{Array2, array};
use num_complex::Complex64 as C64;

fn dagger(a: &Array2<C64>) -> Array2<C64> {
    a.t().mapv(|x| x.conj())
}

fn approx_eq(a: &Array2<C64>, b: &Array2<C64>, tol: f64) -> bool {
    a.dim() == b.dim() && a.iter().zip(b.iter()).all(|(x, y)| (*x - *y).norm() < tol)
}

fn kron(a: &Array2<C64>, b: &Array2<C64>) -> Array2<C64> {
    let (ar, ac) = a.dim();
    let (br, bc) = b.dim();
    let mut out = Array2::<C64>::zeros((ar * br, ac * bc));

    for i in 0..ar {
        for j in 0..ac {
            for k in 0..br {
                for l in 0..bc {
                    out[(i * br + k, j * bc + l)] = a[(i, j)] * b[(k, l)];
                }
            }
        }
    }

    out
}

fn main() -> Result<()> {
    let tol = 1e-10;

    // 1-qubit Pauli / Clifford gates
    let id: Array2<C64> = array![
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    .mapv(|x: f64| C64::from(x));

    let x: Array2<C64> = array![
        [0.0, 1.0],
        [1.0, 0.0],
    ]
    .mapv(|x: f64| C64::from(x));

    let y: Array2<C64> = array![
        [C64::new(0.0, 0.0), C64::new(0.0, -1.0)],
        [C64::new(0.0, 1.0), C64::new(0.0, 0.0)],
    ];

    let z: Array2<C64> = array![
        [1.0, 0.0],
        [0.0, -1.0],
    ]
    .mapv(|x: f64| C64::from(x));

    let h: Array2<C64> = ((1.0 / 2.0_f64.sqrt()) * array![
        [1.0, 1.0],
        [1.0, -1.0],
    ])
    .mapv(|x: f64| C64::from(x));

    let s: Array2<C64> = array![
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(0.0, 1.0)],
    ];

    // 2-qubit gates (basis order: |00>, |01>, |10>, |11>)
    let cnot: Array2<C64> = array![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    .mapv(|x: f64| C64::from(x));

    let cz: Array2<C64> = array![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
    ]
    .mapv(|x: f64| C64::from(x));

    let paulis = [id, x, y, z];
    let pauli_labels = ["I", "X", "Y", "Z"];
    let signs = [C64::from(1.0), C64::from(-1.0)];
    let sign_labels = ["+", "-"];

    // for H gate
    println!("H conjugation:");
    let h_dag = dagger(&h);
    let mut h_table = Vec::new();
    for i in 0..paulis.len() {
        let p = &paulis[i];
        let m = h.dot(p).dot(&h_dag);

        'search_h: for s_idx in 0..signs.len() {
            let sign = signs[s_idx];
            for j in 0..paulis.len() {
                let q = &paulis[j];
                let q_signed = q.mapv(|v| sign * v);
                if approx_eq(&m, &q_signed, tol) {
                    h_table.push((i, s_idx, j));
                    break 'search_h;
                }
            }
        }
    }
    for (i, s_idx, j) in h_table {
        println!("{} -> {} {}", pauli_labels[i], sign_labels[s_idx], pauli_labels[j]);
    }
    println!();

    // for S gate
    println!("S conjugation:");
    let s_dag = dagger(&s);
    let mut s_table = Vec::new();
    for i in 0..paulis.len() {
        let p = &paulis[i];
        let m = s.dot(p).dot(&s_dag);

        'search_s: for s_idx in 0..signs.len() {
            let sign = signs[s_idx];
            for j in 0..paulis.len() {
                let q = &paulis[j];
                let q_signed = q.mapv(|v| sign * v);
                if approx_eq(&m, &q_signed, tol) {
                    s_table.push((i, s_idx, j));
                    break 'search_s;
                }
            }
        }
    }
    for (i, s_idx, j) in s_table {
        println!("{} -> {} {}", pauli_labels[i], sign_labels[s_idx], pauli_labels[j]);
    }
    println!();

    // for CNOT gate
    println!("CNOT conjugation:");
    let cnot_dag = dagger(&cnot);
    let mut cnot_table = Vec::new();
    for i0 in 0..paulis.len() {
        for i1 in 0..paulis.len() {
            let p = kron(&paulis[i0], &paulis[i1]);
            let m = cnot.dot(&p).dot(&cnot_dag);

            'search_cnot: for s_idx in 0..signs.len() {
                let sign = signs[s_idx];
                for j0 in 0..paulis.len() {
                    for j1 in 0..paulis.len() {
                        let q = kron(&paulis[j0], &paulis[j1]);
                        let q_signed = q.mapv(|v| sign * v);
                        if approx_eq(&m, &q_signed, tol) {
                            cnot_table.push((i0, i1, s_idx, j0, j1));
                            break 'search_cnot;
                        }
                    }
                }
            }
        }
    }
    for (i0, i1, s_idx, j0, j1) in cnot_table {
        println!(
            "{}{} -> {} {}{}",
            pauli_labels[i0],
            pauli_labels[i1],
            sign_labels[s_idx],
            pauli_labels[j0],
            pauli_labels[j1]
        );
    }
    println!();

    // for CZ gate
    println!("CZ conjugation:");
    let cz_dag = dagger(&cz);
    let mut cz_table = Vec::new();
    for i0 in 0..paulis.len() {
        for i1 in 0..paulis.len() {
            let p = kron(&paulis[i0], &paulis[i1]);
            let m = cz.dot(&p).dot(&cz_dag);

            'search_cz: for s_idx in 0..signs.len() {
                let sign = signs[s_idx];
                for j0 in 0..paulis.len() {
                    for j1 in 0..paulis.len() {
                        let q = kron(&paulis[j0], &paulis[j1]);
                        let q_signed = q.mapv(|v| sign * v);
                        if approx_eq(&m, &q_signed, tol) {
                            cz_table.push((i0, i1, s_idx, j0, j1));
                            break 'search_cz;
                        }
                    }
                }
            }
        }
    }
    for (i0, i1, s_idx, j0, j1) in cz_table {
        println!(
            "{}{} -> {} {}{}",
            pauli_labels[i0],
            pauli_labels[i1],
            sign_labels[s_idx],
            pauli_labels[j0],
            pauli_labels[j1]
        );
    }
    println!();

    Ok(())
}
