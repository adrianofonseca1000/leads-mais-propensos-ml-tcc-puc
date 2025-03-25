"""
Módulo para coleta de dados de Leads.

Este módulo contém funções para conexão ao banco de dados e extração de dados
dos Leads, incluindo informações de contato, recargas e serviços utilizados.
"""
import os
import pandas as pd
import numpy as np
import pyodbc
from dotenv import load_dotenv

# Carregando variáveis de ambiente
load_dotenv()

def get_database_connection():
    """
    Estabelece conexão com o banco de dados MySQL.
    
    Returns:
        connection: Objeto de conexão com o banco de dados.
    """
    odbc_mysql = '{MySQL ODBC 8.0 ANSI driver}'
    
    # Recuperando credenciais do ambiente
    server = os.getenv('DB_SERVER')
    port = os.getenv('DB_PORT')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    
    # Conexão com o MySQL Server
    con_mysql = pyodbc.connect(
        DRIVER=odbc_mysql, 
        SERVER=server, 
        PORT=port, 
        USER=user, 
        PASSWORD=password, 
        charset='utf8'
    )
    
    return con_mysql

def get_contact_data(connection, id_contact=None):
    """
    Obtém dados de contato dos Leads.
    
    Args:
        connection: Conexão com o banco de dados.
        id_contact: ID opcional para filtrar um contato específico.
    
    Returns:
        DataFrame: Dados dos contatos.
    """
    where_clause = ""
    if id_contact:
        where_clause = f"AND l.id_contact = {id_contact}"
    
    query = f"""
        SELECT DISTINCT
            l.id_contact,
            l.plan_type
        FROM
            lead.leads AS l
        WHERE l.is_chat = 1
            AND l.id_campaign = 25
            {where_clause}
            AND l.created_at BETWEEN '2021-08-19 00:00:00' AND '2021-10-31 23:59:59';
    """
    
    return pd.read_sql_query(query, connection)

def get_recharge_data(connection, id_contact=None):
    """
    Obtém dados de recarga dos Leads.
    
    Args:
        connection: Conexão com o banco de dados.
        id_contact: ID opcional para filtrar um contato específico.
    
    Returns:
        DataFrame: Dados de recarga.
    """
    where_clause = "=" if id_contact else "IN"
    id_filter = id_contact if id_contact else """(
                    SELECT id_contact
                    FROM lead.leads
                    WHERE
                        is_chat = 1
                        AND data LIKE '%Chat%'
                        AND id_campaign = 25
                        AND created_at BETWEEN '2021-08-19 00:00:00' AND '2021-10-31 23:59:59')"""
    
    query = f"""
        SELECT DISTINCT
            rec.id_contact,
            SUM(rec.value) AS sum_recharge,
            COUNT(1) AS recharge_frequency
        FROM (
            SELECT DISTINCT
                id_contact, `type`, value, date_time
            FROM
                carrier.costs
            WHERE
                id_contact {where_clause} {id_filter}) AS rec
        GROUP BY rec.id_contact;
    """
    
    data = pd.read_sql_query(query, connection)
    if id_contact:
        data.drop(['id_contact'], axis=1, inplace=True)
    return data

def get_recharge_types(connection, id_contact=None):
    """
    Obtém dados detalhados de diferentes tipos de recarga utilizados pelos Leads.
    
    Args:
        connection: Conexão com o banco de dados.
        id_contact: ID opcional para filtrar um contato específico.
        
    Returns:
        DataFrame: Dados de tipos de recarga.
    """
    where_clause = "=" if id_contact else "IN"
    id_filter = id_contact if id_contact else """(
                    SELECT id_contact
                    FROM lead.leads
                    WHERE
                        is_chat = 1
                        AND data LIKE '%Chat%'
                        AND id_campaign = 25
                        AND created_at BETWEEN '2021-08-19 00:00:00' AND '2021-10-31 23:59:59')"""
    
    # Consulta completa de tipos de recarga
    query = f"""
        SELECT 
            *
        FROM (
            SELECT DISTINCT
                rec.id_contact, rec_online_10,
                rec_online_35_b5, rec_online_15,
                sos_rec_5, rec_online_20_b2,
                chip_pre_rec_10, crip_pre_rec_20,
                rec_online_13, rec_online_50_b8,
                rec_online_30_b4, rec_online_40_b6,
                pct_rec_1190, pct_rec_690,
                rec_online_100_b18, pct_rec_sos_5,
                sos_rec_3, rec_online_8
            FROM (
                SELECT DISTINCT
                rec.id_contact,
                SUM(rec.rec_online_10) AS rec_online_10,
                SUM(rec.rec_online_35_b5) AS rec_online_35_b5,
                SUM(rec.rec_online_15) AS rec_online_15,
                SUM(rec.sos_rec_5) AS sos_rec_5,
                SUM(rec.rec_online_20_b2) AS rec_online_20_b2,
                SUM(rec.chip_pre_rec_10) AS chip_pre_rec_10,
                SUM(rec.crip_pre_rec_20) AS crip_pre_rec_20,
                SUM(rec.rec_online_13) AS rec_online_13,
                SUM(rec.rec_online_50_b8) AS rec_online_50_b8,
                SUM(rec.rec_online_30_b4) AS rec_online_30_b4,
                SUM(rec.rec_online_40_b6) AS rec_online_40_b6,
                SUM(rec.pct_rec_1190) AS pct_rec_1190,
                SUM(rec.pct_rec_690) AS pct_rec_690,
                SUM(rec.rec_online_100_b18) AS rec_online_100_b18,
                SUM(rec.pct_rec_sos_5) AS pct_rec_sos_5,
                SUM(rec.sos_rec_3) AS sos_rec_3,
                SUM(rec.rec_online_8) AS rec_online_8
            FROM (
                SELECT DISTINCT
                    rec.id_contact, rec.date_time,
                    IF(rec.type LIKE '%RECARGA ONLINE R$10%', 1, 0) AS rec_online_10,
                    IF(rec.type LIKE '%RECARGA ONLINE R$35 + BONUS R$5%', 1, 0) AS rec_online_35_b5,
                    IF(rec.type LIKE '%RECARGA ONLINE R$15%', 1, 0) AS rec_online_15,
                    IF(rec.type LIKE '%SOS RECARGA - R$5%', 1, 0) AS sos_rec_5,
                    IF(rec.type LIKE '%RECARGA ONLINE R$20 + BONUS R$2%', 1, 0) AS rec_online_20_b2,
                    IF(rec.type LIKE '%CHIP PRE + RECARGA R$10%', 1, 0) AS chip_pre_rec_10,
                    IF(rec.type LIKE '%CHIP PRE + RECARGA R$20%', 1, 0) AS crip_pre_rec_20,
                    IF(rec.type LIKE '%RECARGA ONLINE R$13%', 1, 0) AS rec_online_13,
                    IF(rec.type LIKE '%RECARGA ONLINE R$50 + BONUS R$8%', 1, 0) AS rec_online_50_b8,
                    IF(rec.type LIKE '%RECARGA ONLINE R$30 + BONUS R$4%', 1, 0) AS rec_online_30_b4,
                    IF(rec.type LIKE '%RECARGA ONLINE R$40 + BONUS R$6%', 1, 0) AS rec_online_40_b6,
                    IF(rec.type LIKE '%PACOTE RECARGA R$11,90%', 1, 0) AS pct_rec_1190,
                    IF(rec.type LIKE '%PACOTE RECARGA R$6,90%', 1, 0) AS pct_rec_690,
                    IF(rec.type LIKE '%RECARGA ONLINE R$100 + BONUS R$18%', 1, 0) AS rec_online_100_b18,
                    IF(rec.type LIKE '%PACOTE SOS RECARGA R$5,00%', 1, 0) AS pct_rec_sos_5,
                    IF(rec.type LIKE '%SOS RECARGA - R$3%', 1, 0) AS sos_rec_3,
                    IF(rec.type LIKE '%RECARGA ONLINE R$8%', 1, 0) AS rec_online_8
                FROM (
                    SELECT DISTINCT
                        id_contact, `type`, value, date_time
                    FROM carrier.costs
                    WHERE id_contact {where_clause} {id_filter}
                ) AS rec
            ) AS rec
            GROUP BY rec.id_contact
            ) AS rec
        ) AS rec_types;
    """
    
    data = pd.read_sql_query(query, connection)
    if id_contact:
        data.drop(['id_contact'], axis=1, inplace=True)
    return data

def get_service_data(connection, id_contact=None):
    """
    Obtém dados de serviços utilizados pelos Leads.
    
    Args:
        connection: Conexão com o banco de dados.
        id_contact: ID opcional para filtrar um contato específico.
        
    Returns:
        DataFrame: Dados de serviços.
    """
    where_clause = "=" if id_contact else "IN"
    id_filter = id_contact if id_contact else """(
                    SELECT id_contact
                    FROM lead.leads
                    WHERE
                        is_chat = 1
                        AND data LIKE '%Chat%'
                        AND id_campaign = 25
                        AND created_at BETWEEN '2021-08-19 00:00:00' AND '2021-10-31 23:59:59')"""
    
    # Consulta de serviços
    query = f"""
        SELECT 
            *
        FROM (
            SELECT DISTINCT
                serv.id_contact, audio_conferencia, bina, bloqueio_originado,
                call_back, chamada_espera, chamar_direto, conteudo_adulto,
                internet, internet_extra, minha_cidade, pacotes_promo,
                prezao_diario, prezao_mensal, prezao_quinzenal, prezao_semanal,
                recarga_sos, servicos_operadora, sms_cobrar, sms_internacional,
                transf_entre_regionais, truecaller
            FROM (
                SELECT DISTINCT
                serv.id_contact,
                SUM(serv.audio_conferencia) AS audio_conferencia,
                SUM(serv.bina) AS bina,
                SUM(serv.bloqueio_originado) AS bloqueio_originado,
                SUM(serv.call_back) AS call_back,
                SUM(serv.chamada_espera) AS chamada_espera,
                SUM(serv.chamar_direto) AS chamar_direto,
                SUM(serv.conteudo_adulto) AS conteudo_adulto,
                SUM(serv.internet) AS internet,
                SUM(serv.internet_extra) AS internet_extra,
                SUM(serv.minha_cidade) AS minha_cidade,
                SUM(serv.pacotes_promo) AS pacotes_promo,
                SUM(serv.prezao_diario) AS prezao_diario,
                SUM(serv.prezao_mensal) AS prezao_mensal,
                SUM(serv.prezao_quinzenal) AS prezao_quinzenal,
                SUM(serv.prezao_semanal) AS prezao_semanal,
                SUM(serv.recarga_sos) AS recarga_sos,
                SUM(serv.servicos_operadora) AS servicos_operadora,
                SUM(serv.sms_cobrar) AS sms_cobrar,
                SUM(serv.sms_internacional) AS sms_internacional,
                SUM(serv.transf_entre_regionais) AS transf_entre_regionais,
                SUM(serv.truecaller) AS truecaller
            FROM (
                SELECT DISTINCT
                    serv.id_contact, serv.date_time,
                    IF(serv.type LIKE '%Áudio Conferência%', 1, 0) AS audio_conferencia,
                    IF(serv.type LIKE '%Bina%', 1, 0) AS bina,
                    IF(serv.type LIKE '%Bloqueio Originado%', 1, 0) AS bloqueio_originado,
                    IF(serv.type LIKE '%Call Back%', 1, 0) AS call_back,
                    IF(serv.type LIKE '%Chamada em Espera%', 1, 0) AS chamada_espera,
                    IF(serv.type LIKE '%Chamar Direto%', 1, 0) AS chamar_direto,
                    IF(serv.type LIKE '%Conteúdo Adulto%', 1, 0) AS conteudo_adulto,
                    IF(serv.type LIKE '%Internet%Móvel%', 1, 0) AS internet,
                    IF(serv.type LIKE '%Internet%Extra%', 1, 0) AS internet_extra,
                    IF(serv.type LIKE '%Minha Cidade%', 1, 0) AS minha_cidade,
                    IF(serv.type LIKE '%Pacotes Promocionais%', 1, 0) AS pacotes_promo,
                    IF(serv.type LIKE '%Prezão Diário%', 1, 0) AS prezao_diario,
                    IF(serv.type LIKE '%Prezão Mensal%', 1, 0) AS prezao_mensal,
                    IF(serv.type LIKE '%Prezão Quinzenal%', 1, 0) AS prezao_quinzenal,
                    IF(serv.type LIKE '%Prezão Semanal%', 1, 0) AS prezao_semanal,
                    IF(serv.type LIKE '%Recarga SOS%', 1, 0) AS recarga_sos,
                    IF(serv.type LIKE '%Serviços da Operadora%', 1, 0) AS servicos_operadora,
                    IF(serv.type LIKE '%SMS a Cobrar%', 1, 0) AS sms_cobrar,
                    IF(serv.type LIKE '%SMS Internacional%', 1, 0) AS sms_internacional,
                    IF(serv.type LIKE '%Transferência entre Regionais%', 1, 0) AS transf_entre_regionais,
                    IF(serv.type LIKE '%Truecaller%', 1, 0) AS truecaller
                FROM (
                    SELECT DISTINCT
                        id_contact, `type`, value, date_time
                    FROM carrier.costs
                    WHERE id_contact {where_clause} {id_filter}
                ) AS serv
            ) AS serv
            GROUP BY serv.id_contact
            ) AS serv
        ) AS serv_types;
    """
    
    data = pd.read_sql_query(query, connection)
    if id_contact:
        data.drop(['id_contact'], axis=1, inplace=True)
    return data

def collect_all_data():
    """
    Coleta todos os dados necessários e os combina em um único DataFrame.
    
    Returns:
        DataFrame: Dados completos dos Leads.
    """
    # Estabelecer conexão com o banco de dados
    connection = get_database_connection()
    
    try:
        # Obter dados de contato
        contacts_df = get_contact_data(connection)
        
        # Inicializar DataFrame final com dados de contato
        final_df = contacts_df.copy()
        
        # Para cada contato, obter dados de recarga e serviços
        for index, row in final_df.iterrows():
            contact_id = row['id_contact']
            
            # Obter dados de recarga
            recharge_df = get_recharge_data(connection, contact_id)
            recharge_types_df = get_recharge_types(connection, contact_id)
            
            # Obter dados de serviços
            service_df = get_service_data(connection, contact_id)
            
            # Adicionar colunas ao DataFrame final
            for col in recharge_df.columns:
                final_df.at[index, col] = recharge_df.iloc[0][col] if not recharge_df.empty else None
                
            for col in recharge_types_df.columns:
                final_df.at[index, col] = recharge_types_df.iloc[0][col] if not recharge_types_df.empty else None
                
            for col in service_df.columns:
                final_df.at[index, col] = service_df.iloc[0][col] if not service_df.empty else None
    
    finally:
        # Fechar conexão
        connection.close()
    
    # Salvar dados em arquivo CSV
    final_df.to_csv('data_raw.csv', sep=';', index=False)
    
    return final_df

if __name__ == "__main__":
    # Executar coleta de dados quando executado como script
    collect_all_data() 