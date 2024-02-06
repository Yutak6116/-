import numpy as np
import pandas as pd

def Preprocessing(input_df: pd.DataFrame()) -> pd.DataFrame():
    def deal_missing(input_df: pd.DataFrame()) -> pd.DataFrame():
        output_df = input_df.copy()
        for col in ['RevLineCr', 'LowDoc', 'BankState']:
            output_df[col] = input_df[col].fillna('[UNK]')
        return output_df
    def clean_money(input_df: pd.DataFrame()) -> pd.DataFrame():
        output_df = input_df.copy()
        for col in ['DisbursementGross', 'GrAppv', 'SBA_Appv']:
            output_df[col] = input_df[col].str[1:].str.replace(',', '').str.replace(' ', '').astype(float)
        return output_df
    output_df = deal_missing(input_df)
    output_df = clean_money(output_df)
    output_df['NewExist'] = np.where(input_df['NewExist'] == 1, 1, 0)
    def make_internal_features(input_df: pd.DataFrame()) -> pd.DataFrame():
        output_df = input_df.copy()
        #ここに内部データを使用した特徴量を書く

        #派生特徴量
        output_df['DisbursementGross_bin'] = pd.cut(output_df['DisbursementGross'], bins=[0, 50000, 100000, 150000, np.inf], labels=['small', 'medium', 'large', 'x-large'])
        # Employee to Loan Size Ratio
        # 財務関連の派生特徴量
        output_df['GrAppv_SBA_Appv_ratio'] = output_df['SBA_Appv'] / (output_df['GrAppv'] + 1) 
        output_df['DisbursementGross_GrAppv_ratio'] = output_df['DisbursementGross'] / (output_df['GrAppv'] + 1)
        # 時間関連の派生特徴量
        output_df['ApprovalDate'] = pd.to_datetime(output_df['ApprovalDate'])
        output_df['DisbursementDate'] = pd.to_datetime(output_df['DisbursementDate'], errors='coerce')
        output_df['Days_between_Approval_Disbursement'] = (output_df['DisbursementDate'] - output_df['ApprovalDate']).dt.days
        current_year = pd.Timestamp('now').year
        output_df['Years_since_Approval'] = current_year - output_df['ApprovalFY']
        # ビジネスの条件関連
        output_df['NewExist_NoEmp_interaction'] = output_df['NewExist'] * output_df['NoEmp']
        output_df['UrbanRural_Sector_interaction'] = output_df['UrbanRural'].astype(str) + '_' + output_df['Sector'].astype(str)
        # リスク関連の派生特徴量
        output_df['RevLineCr_LowDoc_risk_indicator'] = (output_df['RevLineCr'] == 'Y').astype(int) + (output_df['LowDoc'] == 'Y').astype(int)
        # FranchiseCodeのリスク要因
        output_df['Franchise_risk_factor'] = output_df['FranchiseCode'].apply(lambda x: 0 if x in [0, 1] else 1)

        #組み合わせ特徴量
        output_df['State_Sector'] = output_df['State'].astype(str) + '_' + output_df['Sector'].astype(str)
        # 地理的特徴の組み合わせ
        output_df['City_State'] = output_df['City'] + '_' + output_df['State']
        output_df['Lender_Borrower_SameState'] = (output_df['BankState'] == output_df['State']).astype(int)
        # 経済的特徴の組み合わせ
        output_df['Emp_to_Loan_Ratio'] = output_df['NoEmp'] / (output_df['DisbursementGross'] + 1)
        output_df['JobImpactScore'] = output_df['CreateJob'] + output_df['RetainedJob']
        output_df['Employment_creation_ratio'] = output_df['CreateJob'] / (output_df['NoEmp'] + 1)
        output_df['Disbursement_per_Term'] = output_df['DisbursementGross'] / (output_df['Term']+1)
        # 業種と金融条件の組み合わせ
        output_df['Sector_RevLineCr'] = output_df['Sector'].astype(str) + '_' + output_df['RevLineCr']
        output_df['Sector_LowDoc'] = output_df['Sector'].astype(str) + '_' + output_df['LowDoc']
        # 時間的特徴の組み合わせ
        output_df['ApprovalFY_Term'] = output_df['ApprovalFY'].astype(str) + '_' + output_df['Term'].astype(str)
        output_df['ApprovalFY_Quarter'] = output_df['ApprovalDate'].dt.quarter

        #特徴量の加工
        #DisbursementDate修正
        #データをハイフンで分割
        output_df['DisbursementDate'] = pd.to_datetime(output_df['DisbursementDate'], format='%d-%b-%y')
        output_df['DisbursementDate'] = output_df['DisbursementDate'].dt.strftime('%Y-%m-%d')
        output_df[['DisbursementYear','DisbursementMonth','DisbursementDay']] = output_df['DisbursementDate'].str.split('-',expand=True)
        #DisbursementYearをint型に補正
        output_df['DisbursementYear'] = output_df['DisbursementYear'].fillna(output_df['DisbursementYear'].mode().iloc[0])
        output_df['DisbursementYear'] = output_df['DisbursementYear'].astype(int)
        #DisbursementMonthをint型に補正
        output_df['DisbursementMonth'] = output_df['DisbursementMonth'].fillna(output_df['DisbursementMonth'].mode().iloc[0])
        output_df['DisbursementMonth'] = output_df['DisbursementMonth'].astype(int)
        #DisbursementDayをint型に補正
        output_df['DisbursementDay'] = output_df['DisbursementDay'].fillna(output_df['DisbursementDay'].mode().iloc[0])
        output_df['DisbursementDay'] = output_df['DisbursementDay'].astype(int)
              
        #ApprovalDate修正
        #データをハイフンで分割
        output_df['ApprovalDate'] = pd.to_datetime(output_df['ApprovalDate'], format='%d-%b-%y')
        output_df['ApprovalDate'] = output_df['ApprovalDate'].dt.strftime('%Y-%m-%d')
        output_df[['ApprovalYear','ApprovalMonth','ApprovalDay']] = output_df['ApprovalDate'].str.split('-',expand=True)
        #ApprovalYearをint型に補正
        output_df['ApprovalYear'] = output_df['ApprovalYear'].astype(int)
        #ApprovalMonthをint型に補正
        output_df['ApprovalMonth'] = output_df['ApprovalMonth'].astype(int)
        #ApprovalDayをint型に補正
        output_df['ApprovalDay'] = output_df['ApprovalDay'].astype(int)

        #特徴量の加工
        #lowdoc ['LowDoc_Y', 'LowDoc_S', 'LowDoc_N', 'LowDoc_A', 'LowDoc_0', 'LowDoc_UNK']
        output_df['LowDoc'] = output_df['LowDoc'].fillna('UNK')
        output_df['LowDoc_Y'] = (output_df['LowDoc'] == 'Y').astype(int)
        output_df['LowDoc_S'] = (output_df['LowDoc'] == 'S').astype(int)
        output_df['LowDoc_N'] = (output_df['LowDoc'] == 'N').astype(int)
        output_df['LowDoc_A'] = (output_df['LowDoc'] == 'A').astype(int)
        output_df['LowDoc_0'] = (output_df['LowDoc'] == '0').astype(int)
        output_df['LowDoc_UNK'] = (output_df['LowDoc'] == 'UNK').astype(int)

        #RevLineCr ['RevLineCr_Y', 'RevLineCr_T', 'RevLineCr_N', 'RevLineCr_0', 'RevLineCr_UNK']
        output_df['RevLineCr'] = output_df['RevLineCr'].fillna('UNK')
        output_df['RevLineCr_Y'] = (output_df['RevLineCr'] == 'Y').astype(int)
        output_df['RevLineCr_T'] = (output_df['RevLineCr'] == 'T').astype(int)
        output_df['RevLineCr_N'] = (output_df['RevLineCr'] == 'N').astype(int)
        output_df['RevLineCr_0'] = (output_df['RevLineCr'] == '0').astype(int)
        output_df['RevLineCr_UNK'] = (output_df['RevLineCr'] == 'UNK').astype(int)

        #DisbursementDate
        output_df['DisbursementDate'] = output_df['DisbursementDate'].fillna('UNK')
        output_df['DisbursementDate_UNK'] = (output_df['DisbursementDate'] == 'UNK').astype(int)


        return output_df
    output_df = make_internal_features(output_df)
    def make_external_features(input_df: pd.DataFrame()) -> pd.DataFrame():
        output_df = input_df.copy()
        #ここに外部データを活用した特徴量を書く
        return output_df
    output_df = make_external_features(output_df)
    return output_df
