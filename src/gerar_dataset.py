"""
Módulo para síntese de sinais de radar UWB (3 a 10 GHz) 
baseado no modelo de centros de espalhamento e contaminação AWGN.
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split

c = 3e8
f_min, f_max = 3e9, 10e9
N = 1024
f = np.linspace(f_min, f_max, N)
TOTAL_AMOSTRAS = 10000
NIVEIS_SNR = [-5, 0, 5, 10, 15]
AMOSTRAS_POR_SNR = TOTAL_AMOSTRAS // len(NIVEIS_SNR)

def sintetizar_sinal_limpo():
    num_centros = np.random.randint(2, 6)
    A_k = np.random.uniform(0.5, 2.0, size=num_centros)
    R_k = np.random.uniform(1.0, 20.0, size=num_centros)
    tau_k = 2.0 * R_k / c
    S_clean = np.zeros(N, dtype=complex)
    for a, tau in zip(A_k, tau_k):
        S_clean += a * np.exp(-1j * 2.0 * np.pi * f * tau)
    return np.real(np.fft.ifft(S_clean))

def contaminar_awgn(h_clean, snr_db):
    P_s = np.mean(h_clean**2)
    sigma = np.sqrt(P_s / (10.0 ** (snr_db / 10.0)))
    return h_clean + np.random.normal(0.0, sigma, size=len(h_clean))

def main():
    np.random.seed(42)
    X_noisy_list, Y_clean_list, rotulos_snr = [], [], []
    for snr in NIVEIS_SNR:
        for _ in range(AMOSTRAS_POR_SNR):
            limpo = sintetizar_sinal_limpo()
            ruidoso = contaminar_awgn(limpo, snr_db=snr)
            X_noisy_list.append(ruidoso)
            Y_clean_list.append(limpo)
            rotulos_snr.append(snr)
            
    X_noisy = np.expand_dims(np.array(X_noisy_list, dtype=np.float32), axis=-1)
    Y_clean = np.expand_dims(np.array(Y_clean_list, dtype=np.float32), axis=-1)
    rotulos = np.array(rotulos_snr, dtype=np.int32)
    
    X_train, X_temp, Y_train, Y_temp, snr_train, snr_temp = train_test_split(
        X_noisy, Y_clean, rotulos, test_size=0.30, random_state=42, stratify=rotulos
    )
    X_val, X_test, Y_val, Y_test, snr_val, snr_test = train_test_split(
        X_temp, Y_temp, snr_temp, test_size=0.50, random_state=42, stratify=snr_temp
    )
    os.makedirs("dados_radar", exist_ok=True)
    np.save("dados_radar/X_train.npy", X_train)
    np.save("dados_radar/Y_train.npy", Y_train)
    np.save("dados_radar/X_val.npy", X_val)
    np.save("dados_radar/Y_val.npy", Y_val)
    np.save("dados_radar/X_test.npy", X_test)
    np.save("dados_radar/Y_test.npy", Y_test)
    np.save("dados_radar/snr_test.npy", snr_test)
    print("Dataset sintético gerado com sucesso.")

if __name__ == "__main__":
    main()
