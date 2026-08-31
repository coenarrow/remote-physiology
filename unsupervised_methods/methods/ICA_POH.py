"""ICA
Non-contact, automated cardiac pulse measurements using video imaging and blind source separation.
Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010).
Optics express, 18(10), 10762-10774. DOI: 10.1364/OE.18.010762

Implementation note: the original leaned on ``np.matrix`` (``np.mat``, removed
in NumPy 2.0) for its ``.H`` conjugate transposes and for the 2-D indexing of
row slices. This is the same JADE algorithm on plain ndarrays, with reshapes in
einops; ``tests/test_unsupervised_methods.py`` pins it to the frozen
pre-NumPy-2 original.
"""
import math

import numpy as np
from einops import rearrange
from scipy import linalg
from scipy import signal
from unsupervised_methods import utils


def process_video(frames):
    """Spatial mean of each frame: ``(T, H, W, 3)`` -> ``(T, 3)``."""
    return utils.rgb_trace(frames)


def ICA_POH(frames, FS):
    # Cut off frequency.
    LPF = 0.7
    HPF = 2.5
    RGB = process_video(frames)

    NyquistF = 1 / 2 * FS
    BGRNorm = np.zeros(RGB.shape)
    Lambda = 100
    for c in range(3):
        BGRDetrend = utils.detrend(RGB[:, c], Lambda)
        BGRNorm[:, c] = (BGRDetrend - np.mean(BGRDetrend)) / np.std(BGRDetrend)
    # Sources are the observation channels: (T, 3) -> (3, T).
    _, S = ica(rearrange(BGRNorm, "t c -> c t"), 3)

    # select BVP Source
    MaxPx = np.zeros(3)
    for c in range(3):
        FF = np.fft.fft(S[c, :])[1:]
        N = FF.shape[0]
        Px = np.abs(FF[:math.floor(N / 2)]) ** 2
        MaxPx[c] = np.max(Px / np.sum(Px, axis=0))
    MaxComp = int(np.argmax(MaxPx))
    BVP_I = S[MaxComp, :]
    B, A = signal.butter(3, [LPF / NyquistF, HPF / NyquistF], 'bandpass')
    return signal.filtfilt(B, A, np.real(BVP_I).astype(np.double))


def ica(X, Nsources, Wprev=0):
    nRows, nCols = X.shape
    if nRows > nCols:
        print(
            "Warning - The number of rows is cannot be greater than the number of columns.")
        print("Please transpose input.")

    if Nsources > min(nRows, nCols):
        Nsources = min(nRows, nCols)
        print(
            'Warning - The number of soures cannot exceed number of observation channels.')
        print('The number of sources will be reduced to the number of observation channels ', Nsources)

    Winv, Zhat = jade(X, Nsources, Wprev)
    W = np.linalg.pinv(Winv)
    return W, Zhat


def _hermitian(a):
    """Conjugate transpose — what ``np.matrix.H`` used to provide."""
    return a.conj().T


def jade(X, m, Wprev):
    """Joint Approximate Diagonalization of Eigen-matrices (Cardoso & Souloumiac)."""
    n, T = X.shape
    nem = m
    seuil = 1 / math.sqrt(T) / 100
    if m < n:
        D, U = np.linalg.eig(X @ _hermitian(X) / T)
        Diag = D
        k = np.argsort(Diag)
        pu = Diag[k]
        ibl = np.sqrt(pu[n - m:n] - np.mean(pu[0:n - m]))
        bl = np.true_divide(np.ones(m), ibl)
        W = np.diag(bl) @ U[0:n, k[n - m:n]].T
        IW = U[0:n, k[n - m:n]] @ np.diag(ibl)
    else:
        IW = linalg.sqrtm(X @ _hermitian(X) / T)
        W = np.linalg.inv(IW)

    Y = W @ X
    R = Y @ _hermitian(Y) / T
    C = Y @ Y.T / T

    # Fourth-order cumulants, flattened into one (m^2, m^2) matrix.
    Q = np.zeros(m * m * m * m)
    index = 0
    for lx in range(m):
        Y1 = Y[lx, :]
        for kx in range(m):
            Yk1 = np.multiply(Y1, np.conj(Y[kx, :]))
            for jx in range(m):
                Yjk1 = np.multiply(Yk1, np.conj(Y[jx, :]))
                for ix in range(m):
                    Q[index] = (Yjk1 / math.sqrt(T)) @ (Y[ix, :].T / math.sqrt(T)) \
                        - R[ix, jx] * R[lx, kx] - R[ix, kx] * R[lx, jx] \
                        - C[ix, lx] * np.conj(C[jx, kx])
                    index += 1

    # Compute and Reshape the significant Eigen
    D, U = np.linalg.eig(rearrange(Q, "(row col) -> row col", row=m * m))
    Diag = abs(D)
    K = np.argsort(Diag)
    la = Diag[K]
    M = np.zeros((m, nem * m), dtype=complex)
    h = m * m - 1
    for u in range(0, nem * m, m):
        Z = rearrange(U[:, K[h]], "(row col) -> row col", row=m)
        M[:, u:u + m] = la[h] * Z
        h = h - 1

    # Approximate the Diagonalization of the Eigen Matrices:
    B = np.array([[1, 0, 0], [0, 1, 1], [0, 0 - 1j, 0 + 1j]])
    Bt = _hermitian(B)

    encore = 1
    if Wprev == 0:
        V = np.eye(m).astype(complex)
    else:
        V = np.linalg.inv(Wprev)
    # Main Loop:
    while encore:
        encore = 0
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, nem * m, m)
                Iq = np.arange(q, nem * m, m)
                g = np.array([M[p, Ip] - M[q, Iq], M[p, Iq], M[q, Ip]])
                temp = B @ (g @ _hermitian(g)) @ Bt
                D, vcp = np.linalg.eig(np.real(temp))
                K = np.argsort(D)
                angles = vcp[:, K[2]]
                if angles[0] < 0:
                    angles = -angles
                c = np.sqrt(0.5 + angles[0] / 2)
                s = 0.5 * (angles[1] - 1j * angles[2]) / c

                if abs(s) > seuil:
                    encore = 1
                    pair = [p, q]
                    G = np.array([[c, -np.conj(s)], [s, c]])  # Givens Rotation
                    V[:, pair] = V[:, pair] @ G
                    M[pair, :] = _hermitian(G) @ M[pair, :]
                    temp1 = c * M[:, Ip] + s * M[:, Iq]
                    temp2 = -np.conj(s) * M[:, Ip] + c * M[:, Iq]
                    M[:, Ip] = temp1
                    M[:, Iq] = temp2

    # Whiten the Matrix
    # Estimation of the Mixing Matrix and Signal Separation
    A = IW @ V
    S = _hermitian(V) @ Y
    return A, S
