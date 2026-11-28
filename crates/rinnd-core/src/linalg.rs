//! Small dense linear algebra helpers (no external dependency).

/// Eigen-decomposition of a symmetric `n x n` matrix (row-major `matrix`).
///
/// Returns `(values, vectors)` with the eigenvalues in decreasing order and
/// `vectors[j * n..(j + 1) * n]` the unit eigenvector of `values[j]`. Uses
/// Householder reduction to tridiagonal form followed by the implicit QL
/// algorithm (the classic EISPACK `tred2` / `tql2` pair), in `O(n^3)`.
pub fn symmetric_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(matrix.len(), n * n);
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut v = matrix.to_vec();
    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    tridiagonalize(&mut v, &mut d, &mut e, n);
    diagonalize(&mut v, &mut d, &mut e, n);

    // Columns of `v` are the eigenvectors; return them as rows, sorted.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| d[b].total_cmp(&d[a]));
    let values = order.iter().map(|&j| d[j]).collect();
    let mut vectors = vec![0.0f64; n * n];
    for (row, &j) in order.iter().enumerate() {
        for k in 0..n {
            vectors[row * n + k] = v[k * n + j];
        }
    }
    (values, vectors)
}

/// Householder reduction of the symmetric matrix `v` to tridiagonal form:
/// on return `d` is the diagonal, `e[1..]` the subdiagonal and `v` holds the
/// accumulated orthogonal transformation.
fn tridiagonalize(v: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    let at = |i: usize, j: usize| i * n + j;
    for j in 0..n {
        d[j] = v[at(n - 1, j)];
    }
    for i in (1..n).rev() {
        let mut scale = 0.0;
        let mut h = 0.0;
        for k in 0..i {
            scale += d[k].abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
                v[at(j, i)] = 0.0;
            }
        } else {
            for k in 0..i {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for value in e.iter_mut().take(i) {
                *value = 0.0;
            }
            for j in 0..i {
                f = d[j];
                v[at(j, i)] = f;
                g = e[j] + v[at(j, j)] * f;
                for k in j + 1..i {
                    g += v[at(k, j)] * d[k];
                    e[k] += v[at(k, j)] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..i {
                    v[at(k, j)] -= f * e[k] + g * d[k];
                }
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
            }
        }
        d[i] = h;
    }
    for i in 0..n - 1 {
        v[at(n - 1, i)] = v[at(i, i)];
        v[at(i, i)] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[at(k, i + 1)] / h;
            }
            for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += v[at(k, i + 1)] * v[at(k, j)];
                }
                for k in 0..=i {
                    v[at(k, j)] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[at(k, i + 1)] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = v[at(n - 1, j)];
        v[at(n - 1, j)] = 0.0;
    }
    v[at(n - 1, n - 1)] = 1.0;
    e[0] = 0.0;
}

/// Implicit QL iteration on the tridiagonal matrix (`d`, `e`), accumulating
/// the rotations into `v`; on return `d` holds the eigenvalues.
fn diagonalize(v: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    let at = |i: usize, j: usize| i * n + j;
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0.0;
    let mut largest = 0.0f64;
    let eps = f64::EPSILON;
    for l in 0..n {
        largest = largest.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n - 1 && e[m].abs() > eps * largest {
            m += 1;
        }
        if m > l {
            loop {
                let g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.hypot(1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for value in d.iter_mut().skip(l + 2) {
                    *value -= h;
                }
                f += h;

                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for k in 0..n {
                        h = v[at(k, i + 1)];
                        v[at(k, i + 1)] = s * v[at(k, i)] + c * h;
                        v[at(k, i)] = c * v[at(k, i)] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * largest {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::FastRng;

    #[test]
    fn eigen_decomposition_of_random_symmetric_matrices() {
        let mut rng = FastRng::new(9);
        for n in [1, 2, 3, 10, 64] {
            let mut a = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..=i {
                    let value = f64::from(rng.next_float()) - 0.5;
                    a[i * n + j] = value;
                    a[j * n + i] = value;
                }
            }
            let (values, vectors) = symmetric_eigen(&a, n);
            for j in 0..n {
                let v = &vectors[j * n..(j + 1) * n];
                for i in 0..n {
                    let av: f64 = (0..n).map(|k| a[i * n + k] * v[k]).sum();
                    assert!((av - values[j] * v[i]).abs() < 1e-9, "n={n} j={j}");
                }
                for other in 0..n {
                    let w = &vectors[other * n..(other + 1) * n];
                    let dot: f64 = v.iter().zip(w).map(|(x, y)| x * y).sum();
                    let expected = if other == j { 1.0 } else { 0.0 };
                    assert!((dot - expected).abs() < 1e-9);
                }
                if j > 0 {
                    assert!(values[j - 1] >= values[j]);
                }
            }
        }
    }

    #[test]
    fn eigen_decomposition_of_diagonal_and_degenerate_matrices() {
        let (values, _) = symmetric_eigen(&[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], 3);
        assert_eq!(values, vec![3.0, 2.0, 1.0]);
        let (values, vectors) = symmetric_eigen(&[0.0; 16], 4);
        assert!(values.iter().all(|&value| value == 0.0));
        assert!(vectors.iter().all(|value| value.is_finite()));
    }
}
