# Suporte Computacional para Diagnóstico de Exames Biomédicos (ECG)

**Universidade Federal Fluminense (UFF) – Escola de Engenharia**  
**Disciplina:** Sinais e Sistemas – Prof. André Pinho (2º Semestre de 2026)

### Equipe (ordem alfabética)
1. Cauã Sales Cardoso de Carvalho (221038054)
2. Livia da Conceição Paes (221038088)
3. Mateus Figueiredo Abreu (122038027)
4. Pedro Lucas Nogueira Barbosa (221038084)
5. Raphael Mello Lasmar Gonçalves (123038028)

---

## Como Abrir o Aplicativo Web (60 FPS · FastAPI + HTML5 Canvas)

No PowerShell, dentro de `d:\sinais`:

```powershell
.\.venv\Scripts\python.exe server.py
```

Depois abra no navegador: **http://localhost:8050** (ou a porta exibida no terminal, caso 8050 esteja ocupada).

**O que você pode fazer no site:**
- **Enviar qualquer ECG seu** (botão **Enviar meu ECG** ou arrastando e soltando o arquivo em qualquer ponto da tela):
  - `.csv` ou `.txt` (detecta separador e taxa de amostragem $f_s$ automaticamente, aplica Notch 60 Hz + Butterworth 0,5–40 Hz fase zero, detecta picos R, calcula BPM/R-R/$\bar{N}$/SDNN, diagnóstico, FFT e Parseval);
  - Arquivos PhysioNet (`.dat` + `.hea` + opcional `.atr`);
  - Exames clínicos em `.pdf` ou imagem (`.png`/`.jpg`) → detecta automaticamente, exibe a pré-visualização e mostra aviso técnico claro sobre exames já pré-filtrados pelo aparelho hospitalar.
- **Explorar 6 casos reais do MIT-BIH em 1 clique** (Normal `100`, Ruído Severo `105`, Deriva Extrema `108`, Arritmia/Bigeminismo `119`, Bradicardia `231` e Taquicardia `209`).
- **Interagir a 60 FPS no Monitor ECG:** arrastar o traçado com o mouse, dar zoom com o scroll, apertar **Reproduzir (25 mm/s)** em tempo real, alternar **Antes × Depois** vs **Sobreposição Direta**, ligar/desligar filtros LIT em tempo real e exportar PNG (300 DPI) ou CSV processado.

*(A versão alternativa em Streamlit continua disponível via `.\.venv\Scripts\streamlit.exe run app.py`.)*

---

## Publicação no GitHub Pages (100% Estático, Sem Servidor)

A pasta `docs/` já contém o site completo com o motor de Processamento Digital de Sinais reimplementado em JavaScript puro (`dsp_engine.js` — Notch 60 Hz, Butterworth 0,5–40 Hz fase zero, Pan-Tompkins/Hamilton-Tompkins, FFT Bluestein/Radix-2 e Parseval) + trechos clínicos reais de 60 s dos 6 registros do MIT-BIH + upload local de `.csv`, `.txt`, `.dat`+`.hea` e `.pdf` direto no navegador:

1. Para atualizar a pasta `docs/` após qualquer edição em `web/`:
   ```powershell
   .\.venv\Scripts\python.exe build_gh_pages.py
   ```
2. Suba o repositório para o GitHub e vá em **Settings → Pages → Build and deployment**:
   - **Source:** `Deploy from a branch`
   - **Branch:** `main` · **Folder:** `/docs`
3. Salve e aguarde ~1 minuto — o site ficará online em `https://<seu-usuario>.github.io/<seu-repo>/` funcionando 100% no navegador sem precisar de servidor Python!

---

## Outros Comandos Úteis

- **Rodar validação completa no MIT-BIH e regenerar `outputs/`:**
  ```powershell
  .\.venv\Scripts\python.exe run_validation.py
  ```
- **Recompilar o Relatório 4 em PDF localmente (Tectonic):**
  ```powershell
  cd relatorio4; tectonic main.tex
  ```
- **Executar todos os testes automatizados (pipeline + API FastAPI + privacidade):**
  ```powershell
  .\.venv\Scripts\pytest.exe -v
  ```
