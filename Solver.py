from Matrix import *
from Model import Model


def planewave(source, epsr, Kx, Ky, Kzr, Kzt, S):
    # return the output when excited by a plane wave
    # pre-process, vector to matrix
    Kx, Ky = np.diag(Kx), np.diag(Ky)
    Kzr, Kzt = np.diag(Kzr), np.diag(Kzt)

    # generate source
    kx0, ky0 = baseK(epsr, source)
    kz0 = np.emath.sqrt(epsr - kx0 ** 2 - ky0 ** 2)
    kinc = [kx0, ky0, kz0]
    kinc = kinc / np.linalg.norm(kinc)
    if source[0] == 0:
        # normal incidence, TE(s-polarization) is set to along with y-axis
        s = [0, 1, 0]
    else:
        # use cross product to determin the normal direction
        s = np.cross(kinc, np.array([0, 0, 1]))
        s = s / np.linalg.norm(s)
    # p polarization is perpendicular to the direction of s polarization and incident wave
    p = np.cross(s, kinc)
    p = p / np.linalg.norm(p)
    amplitude = np.linalg.norm([source[2], source[3]])
    # normalized polarization vector
    polarization = (source[2] * s + source[3] * p) / amplitude
    cinc = np.zeros(Kx.shape[0])
    # set coefficients of (0, 0) order to 1
    cinc[int((cinc.shape[0]) / 2)] = 1
    cinc = np.hstack((polarization[0] * cinc, polarization[1] * cinc))

    # results of reflection side
    c = S[0] @ cinc
    cx = c[0: int(c.shape[0] / 2)]
    cy = c[int(c.shape[0] / 2): c.shape[0] + 1]
    cz = -np.linalg.solve(-Kzr, Kx @ cx + Ky @ cy)
    cr = (cx, cy, cz)
    ita_r = np.real(-Kzr) / np.real(kz0) @ \
        (np.abs(cx) ** 2 + np.abs(cy) ** 2 + np.abs(cz) ** 2)
    R = np.sum(ita_r)
    # results of transmission side
    c = S[2] @ cinc
    cx = c[0: int(c.shape[0] / 2)]
    cy = c[int(c.shape[0] / 2): c.shape[0] + 1]
    cz = -np.linalg.solve(Kzt, Kx @ cx + Ky @ cy)
    ita_t = np.real(Kzt) / np.real(kz0) @ \
        (np.abs(cx) ** 2 + np.abs(cy) ** 2 + np.abs(cz) ** 2)
    ct = (cx, cy, cz)
    T = np.sum(ita_t)

    return R, T, ita_r, ita_t, cr, ct, cinc


def farfield(c, Kx, Ky, Kz):
    # define some parameters
    power = np.zeros_like(Kx)
    axis_ratio = np.zeros_like(Kx)
    handness = np.zeros_like(Kx)
    clcp, crcp = np.zeros_like(Kx)
    cx, cy, cz = c[0], c[1], c[2]
    # remove evesent waves
    index = np.nonzero(np.imag(Kz) == 0)
    for i in index:
        kx, ky, kz = Kx[i], Ky[i], Kz[i]
        ex, ey, ez = c[0][i], c[1][i], c[2][i]
        theta = np.arctan(np.emath.sqrt(kx ** 2 + ky ** 2) / np.abs(kz))
        phi = np.arctan2(ky, kx)

        # project to s polarization
        es = -ex * np.sin(phi) + ey * np.cos(phi)
        temp = ex * np.cos(phi) + ey * np.sin(phi)
        ep = temp * np.cos(theta) + ez * np.sin(theta)
        er = temp * np.cos(theta) - ez * np.sin(theta)
        assert abs(
            er) < 1e-5, 'Radian electric field is not perfectly eliminated'

        # power
        power[i] = np.abs(es) ** 2 + np.abs(ep) ** 2

        # axis ratio
        major = np.emath.sqrt(np.abs(es) ** 2 + np.abs(ep)
                              ** 2 + abs(es ** 2 + ep ** 2))
        minor = np.emath.sqrt(np.abs(es) ** 2 + np.abs(ep)
                              ** 2 - abs(es ** 2 + ep ** 2))
        if minor == 0:
            minor = 1e-8
        axis_ratio[i] = 20 * np.log10(major / abs(minor))

        # handness
        phasep = np.arctan2(np.imag(ep), np.real(ep))
        phases = np.arctan2(np.imag(es), np.real(es))
        phase = phasep - phases
        while phase < -np.pi:
            phase = phase + 2 * np.pi
        while phase > np.pi:
            phase = phase - 2 * np.pi
        if phase >= 0:
            handness[i] = 1  # right-handed
        else:
            handness[i] = -1  # left-handed

        # LCP componet and RCP component
        clcp[i] = (ep - 1j * es) / np.sqrt(2)
        crcp[i] = (ep + 1j * es) / np.sqrt(2)

    return power, axis_ratio, handness


def solver1(model: Model, source, lam, N: list):
    # solver 1 is a RCWA solver for plane wave source
    # define some parameters
    hmcx = np.arange((1 - N[0]) / 2, (N[0] + 1) / 2, dtype='int')
    hmcy = np.arange((1 - N[1]) / 2, (N[1] + 1) / 2, dtype='int')
    Ns = N[0] * N[1]
    k0 = 2 * np.pi / lam
    # materialize the model (need optimization)
    model.gen_model(lam)
    # generate the convolution fft matrix for each layer
    epsf = np.zeros((Ns, Ns, model.num_layer), dtype='cdouble')
    for i in range(model.num_layer):
        if not model.is_homo[i]:
            epsf[:, :, i] = convfft(model.model[:, :, i], hmcx, hmcy)
    # generate wavevectors
    kx0, ky0 = baseK(model.eps_gnd[0], source)
    Kx, Ky = inplaneK(kx0, ky0, model.period, hmcx, hmcy, lam)
    Kz0, Kzr, Kzt = outplaneK(Kx, Ky, model.eps_gnd)
    # free space layer
    _, V0 = freelayer(Kx, Ky, Kz0)
    # reflection layer
    Sr, _, _, _ = reflayer(Kx, Ky, Kzr, model.eps_gnd[0], V0)
    S = Sr  # global scattering matrix
    # structure layers
    Sl = list()
    for i in range(model.num_layer):
        if model.is_homo[i]:
            # homogeneous layer
            temp, _, _, _ = homolayer(
                Kx, Ky, k0, model.model[0, 0, i], model.thickness[i], V0)
            Sl.append(temp)
            S = redhefferx(S, temp)
        else:
            # inhomogeneous layer
            temp, _, _, _ = inhomolayer(
                Kx, Ky, k0, epsf[:, :, i], model.thickness[i], V0)
            Sl.append(temp)
            S = redhefferx(S, temp)
    # transmission layer
    St, _, _, _ = trmlayer(Kx, Ky, Kzt, model.eps_gnd[1], V0)
    S = redhefferx(S, St)
    # use plane wave
    R, T, ita_r, ita_t, cr, ct, _ = planewave(
        source, model.eps_gnd[0], Kx, Ky, Kzr, Kzt, S)
    # farfield projection
    return R, T, ita_r, ita_t, cr, ct
    