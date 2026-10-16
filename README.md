# Remoção de Ruído em Sinais de Radar UWB (3 a 10 GHz) via 1D-CAE

Projeto de Monografia / TCC desenvolvido na Universidade Federal do ABC (UFABC).
Este repositório contém a implementação do Autoencoder Convolucional Unidimensional (1D-CAE) com função de perda ponderada em centros de espalhamento para atenuação de ruído AWGN em radares de banda ultralarga.

## Estrutura do Repositório
- `src/gerar_dataset.py`: Síntese física de sinais UWB por centros de espalhamento e AWGN (-5 a 15 dB).
- `src/modelo_radar.py`: Arquitetura do 1D-CAE e função de perda customizada.
- `src/avaliar.py`: Métrica de avaliação (MSE, SNR_out, Rho de Pearson e latência).
- `modelos/`: Pesos do modelo treinado (.keras).
- `figuras/`: Gráficos gerados para o artigo/TCC.
- `requirements.txt`: Dependências do ambiente Python.
