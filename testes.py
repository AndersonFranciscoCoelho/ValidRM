"""Testes independentes do pipeline FRTB: cada função pode ser executada isoladamente."""
import pandas as pd
from .carregamento import ler, salvar
from .configuracao import CHAVE

def _norm(df, cols):
    d=df.copy()
    for c in cols:
        d[c] = d[c].astype('string').str.strip().str.upper()
    return d

def _resultado(t, status, tabelas, **metricas):
    resumo={"teste":t, "status":status, **metricas}
    pasta=salvar(t,tabelas,resumo)
    print(f"{t}: {status} | {metricas} | {pasta}")
    return resumo

def t01_integridade():
    """Qualidade da população do conglomerado; não comprova completude externa."""
    a=ler('posicoes_completas', CHAVE)
    faltas=a[CHAVE].isna().any(axis=1) | a[CHAVE].apply(lambda x:x.astype('string').str.strip().eq('').fillna(True)).any(axis=1)
    dup=a.duplicated(CHAVE,keep=False)
    # Chave de filtro não é necessariamente identificador único de linha.
    status='ALERTA' if faltas.any() else 'CONCLUIDO_COM_RESSALVAS'
    return _resultado('T01_integridade',status,{'chaves_ausentes':a.loc[faltas], 'chaves_repetidas':a.loc[dup]}, linhas=len(a), chaves_ausentes=int(faltas.sum()), linhas_chave_repetida=int(dup.sum()))

def t02_replicacao_filtro():
    """Replica o hash SAS por (IDPos, CodCompraVenda), preservando multiplicidade de linhas."""
    a=_norm(ler('posicoes_completas',CHAVE),CHAVE)
    ids=_norm(ler('ids_elegiveis',CHAVE),CHAVE)
    b=_norm(ler('posicoes_filtradas',CHAVE),CHAVE)
    keys=ids[CHAVE].dropna().drop_duplicates()
    # SAS hash.check() usa existência de chave; não replica linhas quando há IDs repetidos.
    esperado=a.merge(keys, on=CHAVE, how='inner', validate='many_to_one')
    # Comparação de multiconjuntos: distingue linhas repetidas e evita falso positivo por set.
    comuns=sorted(set(esperado.columns)&set(b.columns))
    if not comuns: raise ValueError('Nenhuma coluna em comum entre esperado e observado')
    def contagem(d):
        x=d[comuns].fillna('<NULO>').astype(str)
        return x.value_counts(dropna=False).rename('n').reset_index()
    e=contagem(esperado).rename(columns={'n':'n_esperado'})
    o=contagem(b).rename(columns={'n':'n_observado'})
    comp=e.merge(o,on=comuns,how='outer').fillna({'n_esperado':0,'n_observado':0})
    comp['diferenca']=comp.n_esperado-comp.n_observado
    diverg=comp.loc[comp.diferenca!=0]
    status='APROVADO' if diverg.empty else 'DIVERGENCIA'
    return _resultado('T02_replicacao_filtro',status,{'divergencias':diverg,'contagem_comparada':comp},origem=len(a), ids_distintos=len(keys), esperado=len(esperado), observado=len(b), linhas_com_divergencia=int(diverg.diferenca.abs().sum()), colunas_comparadas=';'.join(comuns))

def t03_elegibilidade():
    """Analisa inclusões/exclusões por chave; referência é IDs_Exposicao_Sens, não regra regulatória independente."""
    a=_norm(ler('posicoes_completas',CHAVE),CHAVE)
    ids=_norm(ler('ids_elegiveis',CHAVE),CHAVE)
    b=_norm(ler('posicoes_filtradas',CHAVE),CHAVE)
    keys=ids[CHAVE].dropna().drop_duplicates()
    orig=a[CHAVE].drop_duplicates().merge(keys,on=CHAVE,how='left',indicator=True)
    fora=b.merge(keys,on=CHAVE,how='left',indicator=True)
    fora=fora.loc[fora['_merge']=='left_only'].drop(columns='_merge')
    sem_origem=b.merge(a[CHAVE].drop_duplicates(),on=CHAVE,how='left',indicator=True)
    sem_origem=sem_origem.loc[sem_origem['_merge']=='left_only'].drop(columns='_merge')
    status='APROVADO' if fora.empty and sem_origem.empty else 'DIVERGENCIA'
    return _resultado('T03_elegibilidade',status,{'posicoes_fora_dos_ids':fora,'posicoes_sem_origem':sem_origem,'chaves_origem_por_elegibilidade':orig},fora_ids=len(fora),sem_origem=len(sem_origem),chaves_origem_nao_elegiveis=int((orig['_merge']=='left_only').sum()))

def t04_duplicidades():
    """Avalia duplicidade de chave e duplicidade integral, sem presumir que operações C/V são duplicadas."""
    tabelas={}; metricas={}
    for nome in ('posicoes_completas','ids_elegiveis','posicoes_filtradas'):
        d=ler(nome,CHAVE)
        tabelas[f'{nome}_chave_repetida']=d.loc[d.duplicated(CHAVE,keep=False)]
        tabelas[f'{nome}_linha_repetida']=d.loc[d.duplicated(keep=False)]
        metricas[f'{nome}_chave_repetida']=int(d.duplicated(CHAVE,keep=False).sum())
        metricas[f'{nome}_linha_repetida']=int(d.duplicated(keep=False).sum())
    return _resultado('T04_duplicidades','CONCLUIDO_COM_RESSALVAS',tabelas,**metricas)

def t05_classificacao():
    """Analisa classes presentes e inconsistências de classificação; sem inferir enquadramento normativo."""
    ids=ler('ids_elegiveis',CHAVE+['CodClasse'])
    ids=_norm(ids,CHAVE+['CodClasse'])
    ambiguos=ids.groupby(CHAVE,dropna=False)['CodClasse'].nunique(dropna=False).reset_index(name='classes_distintas')
    ambiguos=ambiguos.loc[ambiguos.classes_distintas>1]
    dist=ids.groupby('CodClasse',dropna=False).size().reset_index(name='quantidade')
    vazios=ids.loc[ids.CodClasse.isna()|ids.CodClasse.eq('')]
    status='ALERTA' if len(ambiguos) or len(vazios) else 'CONCLUIDO_COM_RESSALVAS'
    return _resultado('T05_classificacao',status,{'distribuicao_classes':dist,'chaves_classes_ambiguas':ambiguos,'classe_ausente':vazios},chaves_ambiguas=len(ambiguos),classes_ausentes=len(vazios))

def t06_fluxos():
    """Integridade dos fluxos e vínculos com posições. Regras financeiras exigem dicionário de dados."""
    f=ler('fluxos', ['IDPos','CodCompraVenda'])
    p=_norm(ler('posicoes_filtradas',CHAVE),CHAVE)
    f=_norm(f,CHAVE)
    sem=f.merge(p[CHAVE].drop_duplicates(),on=CHAVE,how='left',indicator=True)
    sem=sem.loc[sem['_merge']=='left_only'].drop(columns='_merge')
    colunas_data=[c for c in ['flx_DtIni','flx_DtFim'] if c in f.columns]
    invalidas=pd.DataFrame()
    if len(colunas_data)==2:
        ini=pd.to_datetime(f[colunas_data[0]],errors='coerce',dayfirst=True)
        fim=pd.to_datetime(f[colunas_data[1]],errors='coerce',dayfirst=True)
        invalidas=f.loc[ini.isna()|fim.isna()|(fim<ini)]
    status='ALERTA' if len(sem) or len(invalidas) else 'CONCLUIDO_COM_RESSALVAS'
    return _resultado('T06_fluxos',status,{'fluxos_sem_posicao':sem,'datas_invalidas':invalidas},fluxos=len(f),fluxos_sem_posicao=len(sem),datas_invalidas=len(invalidas),colunas_data_verificadas=';'.join(colunas_data))
