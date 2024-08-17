from __future__ import print_function,division
import numpy as np
import pandas as pd

def compute_2d_ctf(freqs, dfu, dfv, dfang, volt, cs, w, bfactor=None):
    # converting the voltage and spherical aberration into appropriate units.
    volt = volt * 1000
    cs = cs * 10 ** 7
    # electron wavelength(lam) is calculated based on the relativistic electron equation.
    lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt ** 2)
    # Calculating angle and squared spatial frequencies from the freqs array.
    x = freqs[:, 0]
    y = freqs[:, 1]
    ang = np.arctan2(y, x)
    s2 = x ** 2 + y ** 2
    # defocus as a function of the angle
    df = 0.5 * (dfu + dfv + (dfu - dfv) * np.cos(2 * (ang - dfang)))
    # gamma is the phase shift introduced by defocus and spherical aberration
    gamma = 2 * np.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam ** 3 * s2 ** 2)
    # CTF is then computed using sine and cosine functions with amplitude contrast.
    ctf = np.sqrt(1 - w ** 2) * np.sin(gamma) - w * np.cos(gamma)
    # B-factor is applied for damping high frequencies.
    if bfactor is not None:
        ctf *= np.exp(-bfactor / 4 * s2)
    return ctf.astype(freqs.dtype)

def parse_ctf(f):
    ctf_params = pd.read_csv(f, sep='\s+', header=None)
    ctf_params.columns = ['defocus', 'cs', 'voltage', 'apix', 'bfactor', 'ampcont', 'dfdiff', 'dfang']
    return ctf_params

# n, m: Dimensions of the output CTF array.
# scale: Scaling factor for pixel size.
def ctf_filter(ctf_params, n, m, scale=1):
    # calculate CTF
    # 2D arrays theta and gamma represents the frequencies in Fourier space.
    theta = np.fft.fftfreq(n)
    gamma = np.fft.fftfreq(m)
    theta,gamma= np.meshgrid(theta,gamma,indexing='ij')
    freqs = np.stack([theta.ravel(), gamma.ravel()], 1) #Combine gamma and theta
    ctf = np.zeros((len(ctf_params), n, m), dtype=np.float32) #For storing CTF values
    # Calculating CTF for each parameter
    for i in range(len(ctf_params)):
        apix = ctf_params.apix[i] * scale
        c = compute_2d_ctf(freqs / apix,
                           ctf_params.defocus[i] * 10000,
                           ctf_params.defocus[i] * 10000,
                           2 * np.pi * ctf_params.dfang[i] / 360,
                           ctf_params.voltage[i],
                           ctf_params.cs[i],
                           ctf_params.ampcont[i] / 100,
                           ctf_params.bfactor[i],
                           )
        c = c.reshape(n, m)
        # Applying the inverse FFT to the computed CTF to obtain the spatial domain representation. The result is shifted to center the CTF.
        ctf[i] = -np.fft.fftshift(np.fft.ifft2(c)).real
    return ctf