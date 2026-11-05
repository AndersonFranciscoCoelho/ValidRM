"""Configuração de caminhos, nomes de arquivos e regras parametrizáveis."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DADOS = ROOT / "dados"
RESULTADOS = ROOT / "resultados"
ARQUIVOS = {
 "posicoes_completas": "Base_Posicoes_Completa.xlsx",
 "posicoes_filtradas": "Base_Posicoes.xlsx",
 "ids_elegiveis": "IDs_Exposicao_Sens.xlsx",
 "exposicoes": "Tab_Exposicoes.xlsx",
 "fatores": "Fatores_com_Exp.xlsx",
 "fluxos": "Fluxos_Por_Cenario.xlsx",
}
CHAVE = ["IDPos", "CodCompraVenda"]
# Os critérios de mapeamento são hipóteses a serem confirmadas no SAS e com a área.
CLASSES_VALIDAS = {"GIRR", "FX", "CSR", "COM", "EQ"}
