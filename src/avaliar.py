"""
Avaliação comparativa entre 1D-CAE e Filtro de Wiener.
"""
import numpy as np
from scipy.signal import wiener

def calcular_metricas(sinal_limpo, sinal_rec):
    mse = np.mean((sinal_limpo - sinal_rec)**2)
    pot_sinal = np.sum(sinal_limpo**2)
    pot_erro = np.sum((sinal_limpo - sinal_rec)**2)
    snr_out = 10.0 * np.log10(pot_sinal / pot_erro) if pot_erro > 0 else 0.0
    num = np.sum(sinal_limpo * sinal_rec)
    den = np.sqrt(np.sum(sinal_limpo**2) * np.sum(sinal_rec**2))
    rho = num / den if den > 0 else 0.0
    return mse, snr_out, rho
