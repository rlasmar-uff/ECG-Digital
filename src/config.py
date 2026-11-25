"""Configurações centrais do projeto de extensão de Sinais e Sistemas (UFF 2026/2)."""

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "mitdb"
OUTPUTS_DIR = BASE_DIR / "outputs"
RELATORIO4_FIG_DIR = BASE_DIR / "relatorio4" / "figuras"

PROJECT_TITLE = (
    "Suporte Computacional para Diagnóstico de Exames Biomédicos: "
    "Automatização da Interpretação de Eletrocardiograma (ECG)"
)
INSTITUTION = "Universidade Federal Fluminense (UFF) – Escola de Engenharia"
COURSE_INFO = "Disciplina: Sinais e Sistemas – Prof. André Pinho (2º Semestre de 2026)"

# Equipe SEMPRE em ordem alfabética rigorosa
TEAM_MEMBERS: list[tuple[str, str]] = [
    ("Cauã Sales Cardoso de Carvalho", "221038054"),
    ("Livia da Conceição Paes", "221038088"),
    ("Mateus Figueiredo Abreu", "122038027"),
    ("Pedro Lucas Nogueira Barbosa", "221038084"),
    ("Raphael Mello Lasmar Gonçalves", "123038028"),
]

TEAM_INLINE = " | ".join(f"{name} ({mat})" for name, mat in TEAM_MEMBERS)

# Parâmetros padrão de filtragem (Sistemas LIT de Fase Zero)
DEFAULT_FS = 360.0
NOTCH_FREQ_HZ = 60.0
NOTCH_Q = 30.0
BANDPASS_LOW_HZ = 0.5
BANDPASS_HIGH_HZ = 40.0
BANDPASS_ORDER = 4

# Período refratário fisiológico mínimo (Hamilton & Tompkins, 1986)
REFRACTORY_PERIOD_MS = 200.0
VALIDATION_TOLERANCE_MS = 150.0

# Limiares clínicos definidos na etapa de levantamento de requisitos (Relatórios 2 e 3)
BPM_BRADYCARDIA_MAX = 60.0
BPM_TACHYCARDIA_MIN = 120.0

# Símbolos oficiais de batimentos na base MIT-BIH Arrhythmia Database (PhysioNet / AAMI)
MITBIH_BEAT_SYMBOLS: frozenset[str] = frozenset(
    {
        "N",  # Normal beat
        "L",  # Left bundle branch block beat
        "R",  # Right bundle branch block beat
        "B",  # Bundle branch block beat (unspecified)
        "A",  # Atrial premature beat
        "a",  # Aberrated atrial premature beat
        "J",  # Nodal (junctional) premature beat
        "S",  # Supraventricular premature or ectopic beat
        "V",  # Premature ventricular contraction (PVC)
        "r",  # R-on-T premature ventricular contraction
        "F",  # Fusion of ventricular and normal beat
        "e",  # Atrial escape beat
        "j",  # Nodal (junctional) escape beat
        "n",  # Supraventricular escape beat
        "E",  # Ventricular escape beat
        "/",  # Paced beat
        "f",  # Fusion of paced and normal beat
        "Q",  # Unclassifiable beat
        "?",  # Beat not classified during learning
    }
)


@dataclass(frozen=True)
class MitBihRecordInfo:
    """Metadados descritivos dos registros reais do MIT-BIH avaliados no projeto."""

    record_id: str
    category: str
    description: str
    default_window_start_s: float = 0.0
    default_window_duration_s: float = 10.0


# Registros reais do MIT-BIH selecionados para cobrir ritmo normal, ruído severo e anomalias cardíacas
MITBIH_RECORDS: list[MitBihRecordInfo] = [
    MitBihRecordInfo(
        record_id="100",
        category="Ritmo Sinusal Normal",
        description="Referência limpa com ritmo sinusal estável e batimentos predominantemente normais.",
        default_window_start_s=10.0,
        default_window_duration_s=10.0,
    ),
    MitBihRecordInfo(
        record_id="105",
        category="Ruído Severo (Linha de Base + EMG)",
        description="Registro com intensa oscilação de linha de base e ruído muscular de alta frequência.",
        default_window_start_s=240.0,
        default_window_duration_s=10.0,
    ),
    MitBihRecordInfo(
        record_id="108",
        category="Oscilação Extrema + QRS Atípico / Bradicardia Limítrofe",
        description="Forte deriva respiratória, complexos QRS invertidos/alargados e arritmia sinusal.",
        default_window_start_s=120.0,
        default_window_duration_s=10.0,
    ),
    MitBihRecordInfo(
        record_id="119",
        category="Arritmia Ventricular (Bigeminismo / PVCs)",
        description="Frequentes extrassístoles ventriculares (PVCs) com alta variabilidade dos intervalos R-R (SDNN).",
        default_window_start_s=60.0,
        default_window_duration_s=10.0,
    ),
    MitBihRecordInfo(
        record_id="231",
        category="Bradicardia e Bloqueio AV de 2º Grau",
        description="Frequência cardíaca reduzida (< 60 BPM) associada a bloqueio de condução atrioventricular.",
        default_window_start_s=30.0,
        default_window_duration_s=10.0,
    ),
    MitBihRecordInfo(
        record_id="209",
        category="Taquicardia Supraventricular (Episódios > 120 BPM)",
        description="Registro com episódios clínicos reais de taquicardia supraventricular paroxística.",
        default_window_start_s=860.0,
        default_window_duration_s=10.0,
    ),
]

LITERATURE_BENCHMARKS: list[dict[str, str]] = [
    {
        "Estudo": "Pan & Tompkins (1985)",
        "Método Principal": "Passa-faixa 5–15 Hz, derivada 5 pts, quadratura, MWI 150 ms",
        "Sensibilidade (%)": "99,75",
        "Precisão / +P (%)": "99,54",
        "Observação": "Taxa global de 99,3% na base MIT-BIH completa",
    },
    {
        "Estudo": "Hamilton & Tompkins (1986)",
        "Método Principal": "Limiares adaptativos, período refratário 200 ms e search-back",
        "Sensibilidade (%)": "99,69",
        "Precisão / +P (%)": "99,77",
        "Observação": "Redução expressiva de FP causados por ondas T hiperagudas",
    },
    {
        "Estudo": "Luz et al. (2016)",
        "Método Principal": "Pipeline 4 etapas: Butterworth 0,5–40 Hz fase zero + Notch 60 Hz",
        "Sensibilidade (%)": "> 99,0*",
        "Precisão / +P (%)": "> 99,0*",
        "Observação": "Survey de referência; métricas de BPM, R-R e SDNN (*faixa típica dos detectores revisados)",
    },
]
