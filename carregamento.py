"""Leitura de planilhas com validação de esquema e preservação de IDs textuais."""
import pandas as pd
from pathlib import Path
from .configuracao import DADOS, ARQUIVOS, RESULTADOS

def caminho(nome):
    if nome not in ARQUIVOS: raise KeyError(f"Base desconhecida: {nome}")
    return DADOS / ARQUIVOS[nome]

def ler(nome, obrigatorias=(), aba=0):
    arquivo = caminho(nome)
    if not arquivo.exists():
        raise FileNotFoundError(f"Base ausente: {arquivo}. Atualize ARQUIVOS em configuracao.py.")
    if arquivo.suffix.lower() in ('.xlsx', '.xlsm'):
        df = pd.read_excel(arquivo, sheet_name=aba, dtype=str, engine='openpyxl')
    elif arquivo.suffix.lower() == '.csv':
        df = pd.read_csv(arquivo, dtype=str, sep=None, engine='python')
    else: raise ValueError(f"Formato não suportado: {arquivo.suffix}")
    df.columns = df.columns.astype(str).str.strip()
    faltantes = sorted(set(obrigatorias) - set(df.columns))
    if faltantes: raise ValueError(f"{nome}: colunas ausentes: {faltantes}. Disponíveis: {list(df.columns)}")
    return df

def salvar(teste, tabelas, resumo):
    destino = RESULTADOS / teste
    destino.mkdir(parents=True, exist_ok=True)
    for nome, df in tabelas.items():
        if not isinstance(df, pd.DataFrame): continue
        df.to_csv(destino / f"{nome}.csv", index=False, encoding='utf-8-sig')
    pd.DataFrame([resumo]).to_csv(destino / 'resumo.csv', index=False, encoding='utf-8-sig')
    return destino
