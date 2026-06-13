"""
Discrete-time Linear Quadratic Regulator (LQR).

Solves the infinite-horizon discrete-time algebraic Riccati equation (DARE)
via iterative value-function recursion.
"""

import numpy as np


def solve_dare(A, B, Q, R, max_iters=2000, tol=1e-10):
    """
    Solve DARE via Riccati iteration.

        P = AᵀPA − AᵀPB (R + BᵀPB)⁻¹ BᵀPA + Q

    Args:
        A: state transition matrix  [n, n]
        B: input matrix              [n, m]
        Q: state cost (≥0)           [n, n]
        R: input cost (>0)           [m, m]
        max_iters: max iterations
        tol: convergence tolerance

    Returns:
        K: optimal feedback gain  [m, n]
        P: DARE solution          [n, n]
    """
    n, m = B.shape
    P = Q.copy()

    for _ in range(max_iters):
        S = R + B.T @ P @ B                    # [m, m]
        K_temp = np.linalg.solve(S, B.T @ P @ A)  # [m, n]
        P_next = A.T @ P @ A - A.T @ P @ B @ K_temp + Q
        diff = np.max(np.abs(P_next - P))
        P = P_next
        if diff < tol:
            break

    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def dlqr(A, B, Q, R):
    """
    Discrete-time infinite-horizon LQR.

    Returns:
        K: optimal feedback gain, u* = −K x
        P: DARE solution (cost-to-go matrix)
    """
    return solve_dare(A, B, Q, R)


def finite_horizon_lqr(A, B, Q, R, Qf, N):
    """
    Finite-horizon discrete-time LQR (time-varying).

    Args:
        A, B, Q, R: dynamics & cost matrices
        Qf: terminal state cost  [n, n]
        N:  horizon length

    Returns:
        K_seq: list of gains  [K_0, K_1, ..., K_{N-1}], each [m, n]
        P_seq: list of cost-to-go  [P_0, P_1, ..., P_N], each [n, n]
    """
    n, m = B.shape
    P_seq = [Qf]
    K_seq = []

    P = Qf
    for _ in range(N):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P = A.T @ P @ A - A.T @ P @ B @ K + Q
        K_seq.insert(0, K)
        P_seq.insert(0, P)

    return K_seq, P_seq
