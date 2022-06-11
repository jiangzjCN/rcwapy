import numpy as np


def baseK(epsr, source):
    kx0 = np.emath.sqrt(epsr) * np.sin(np.deg2rad(source[0])) * np.cos(np.deg2rad(source[1]))
    ky0 = np.emath.sqrt(epsr) * np.sin(np.deg2rad(source[0])) * np.sin(np.deg2rad(source[1]))
    return kx0, ky0


def inplaneK(kx0, ky0, period, hmcx, hmcy, lambda0):
    # generate column vectors of in-plane wavevectors for RCWA
    kx = kx0 - hmcx * lambda0 / period[0]
    ky = ky0 - hmcy * lambda0 / period[1]
    kx, ky = np.meshgrid(kx, ky)
    # order F is like MATLAB
    return kx.flatten(order='F'), ky.flatten(order='F')


def outplaneK(Kx, Ky, eps_gnd):
    # generate column vectors of out-plane wavevectors for standard RCWA
    I0 = np.ones_like(Kx)
    # I0 is important because sometimes conj(eps) may make the '0j' in eps
    # change to '-0j' and causes some numerical errors
    Kz0 = np.conj(np.emath.sqrt(I0 - Kx ** 2 - Ky ** 2))
    Kzr = -np.conj(np.emath.sqrt(np.conj(eps_gnd[0]) * I0 - Kx ** 2 - Ky ** 2))
    Kzt = np.conj(np.emath.sqrt(np.conj(eps_gnd[1]) * I0 - Kx ** 2 - Ky ** 2))
    return Kz0, Kzr, Kzt


def convfft(model, hmcx, hmcy):
    N1 = hmcx.size
    N2 = hmcy.size
    model = np.fft.fft2(model) / model.size
    model = np.fft.fftshift(model)
    epsf = np.zeros((N1 * N2, N1 * N2), dtype='cdouble')
    # find the (0, 0) order in model after fft
    row0 = int(model.shape[0] / 2)
    col0 = int(model.shape[1] / 2)
    row = 0
    col = 0
    for i in np.arange(N1):
        for j in np.arange(N2):
            for m in np.arange(N1):
                for n in np.arange(N2):
                    epsf[row, col] = model[row0 + hmcy[j] - hmcy[n], col0 + hmcx[i] - hmcx[m]]
                    col += 1
            col = 0
            row += 1
    return epsf


def freelayer(Kx, Ky, Kz0):
    # generate W, V matrix for 0-thickness gap layer
    # pay attention to the ndim of Kx, Ky, if they are 2D, np.diagflat should be used
    Q = np.block([[np.diag(Kx * Ky), np.diag(1 - Kx * Kx)], [np.diag(Ky * Ky - 1), -np.diag(Ky * Kx)]])
    lam = 1j * np.concatenate((Kz0, Kz0))  # eigen value
    lam_inv = np.diag(1 / lam)
    W0 = np.eye(lam_inv.shape[0])
    V0 = Q @ lam_inv
    return W0, V0


# np.linalg.solve(a, b) = a \ b
# np.linalg.solve(b.T, a.T).T = a / b
def reflayer(Kx, Ky, Kzr, epsr, V0):
    # generate scattering matrix for reflection layer
    Q = np.block([[np.diag(Kx * Ky), np.diag(epsr - Kx * Kx)], [np.diag(Ky * Ky - epsr), -np.diag(Ky * Kx)]])
    lam = -1j * np.concatenate((Kzr, Kzr))  # eigen value
    lam_inv = np.diag(1 / lam)
    W = np.eye(lam_inv.shape[0])
    V = Q @ lam_inv
    temp = np.linalg.solve(V0, V)
    A = W + temp
    B = W - temp
    S11 = -np.linalg.solve(A, B)
    S12 = 2 * np.linalg.inv(A)
    S22 = np.linalg.solve(A.T, B.T).T
    S21 = (A - S22 @ B) / 2
    lam = np.diag(lam)
    # return Sr, W, V, lam
    return (S11, S12, S21, S22), W, V, lam


def trmlayer(Kx, Ky, Kzt, epst, V0):
    # generate scattering matrix for transmission layer
    Q = np.block([[np.diag(Kx * Ky), np.diag(epst - Kx * Kx)],
                  [np.diag(Ky * Ky - epst), -np.diag(Ky * Kx)]])
    lam = 1j * np.concatenate((Kzt, Kzt))  # eigen value
    lam_inv = np.diag(1 / lam)
    W = np.eye(lam_inv.shape[0])
    V = Q @ lam_inv
    temp = np.linalg.solve(V0, V)
    A = W + temp
    B = W - temp
    S11 = np.linalg.solve(A.T, B.T).T
    S12 = (A - S11 @ B) / 2
    S21 = 2 * np.linalg.inv(A)
    S22 = -np.linalg.solve(A, B)
    lam = np.diag(lam)
    # return St, W, V, lam
    return (S11, S12, S21, S22), W, V, lam


def homolayer(Kx, Ky, k0, eps, thickness, V0):
    # generate scattering matrix for homogeneous layers
    Q = np.block([[np.diag(Kx * Ky), np.diag(eps - Kx * Kx)], [np.diag(Ky * Ky - eps), -np.diag(Ky * Kx)]])
    Kz = np.conj(np.emath.sqrt(np.conj(eps) * np.ones_like(Kx) - Kx ** 2 - Ky ** 2))
    lam = -1j * np.concatenate((Kz, Kz))  # eigen value
    lam_inv = np.diag(1 / lam)
    W = np.eye(lam_inv.shape[0])
    V = Q @ lam_inv
    X = np.diag(np.exp(-lam * k0 * thickness))
    temp1 = np.linalg.solve(V, V0)
    A = W + temp1
    B = W - temp1
    temp1 = X @ np.linalg.solve(A.T, B.T).T @ X
    temp2 = A - temp1 @ B
    S11 = np.linalg.solve(temp2, temp1 @ A - B)
    S12 = np.linalg.solve(temp2, X) @ (A - np.linalg.solve(A.T, B.T).T @ B)
    lam = np.diag(lam)
    # return Sl, W, V, lam
    return (S11, S12, S12, S11), W, V, lam


def inhomolayer(Kx, Ky, k0, epsf, thickness, V0):
    # generate scattering matrix for inhomogeneous layer
    Kx = np.diag(Kx)
    Ky = np.diag(Ky)
    miuf = np.eye(Kx.shape[0])
    temp1 = np.linalg.solve(epsf.T, Kx.T).T
    temp2 = np.linalg.solve(epsf.T, Ky.T).T
    P = np.block([[temp1 @ Ky, miuf - temp1 @ Kx], [temp2 @ Ky - miuf, -temp2 @ Kx]])
    Q = np.block([[Kx @ Ky, epsf - Kx @ Kx], [Ky @ Ky - epsf, -Ky @ Kx]])
    lam, W = np.linalg.eig(P @ Q)
    lam = np.diag(np.emath.sqrt(lam))
    V = np.linalg.solve(lam.T, np.transpose(Q @ W)).T
    X = np.diag(np.exp(np.diag(-lam * k0 * thickness)))
    temp1 = np.linalg.inv(W)
    temp2 = np.linalg.solve(V, V0)
    A = temp1 + temp2
    B = temp1 - temp2
    temp1 = X @ np.linalg.solve(A.T, B.T).T @ X
    temp2 = A - temp1 @ B
    S11 = np.linalg.solve(temp2, temp1 @ A - B)
    S12 = np.linalg.solve(temp2, X) @ (A - np.linalg.solve(A.T, B.T).T @ B)
    # return Sl, W, V, lam
    return (S11, S12, S12, S11), W, V, lam


def redhefferx(S1, S2):
    # return redheffer cross product of S1 and S2
    I0 = np.eye(S1[0].shape[0])
    # S11-0, S12-1, S21-2, S22-3
    temp1 = np.linalg.solve(np.transpose(I0 - S2[0] @ S1[3]), S1[1].T).T
    temp2 = np.linalg.solve(np.transpose(I0 - S1[3] @ S2[0]), S2[2].T).T
    S11 = S1[0] + temp1 @ S2[0] @ S1[2]
    S12 = temp1 @ S2[1]
    S21 = temp2 @ S1[2]
    S22 = S2[3] + temp2 @ S1[3] @ S2[1]
    return S11, S12, S21, S22
