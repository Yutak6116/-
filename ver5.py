
default_numerical_features = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob', 'DisbursementGross', 'GrAppv', 'SBA_Appv', 'ApprovalFY']
default_categorical_features = ['NewExist', 'FranchiseCode', 'RevLineCr', 'LowDoc', 'UrbanRural', 'State', 'BankState', 'City', 'Sector']
add_numerical_features = ['FranchiseCode_count_encoding', 'RevLineCr_count_encoding', 'LowDoc_count_encoding', 'UrbanRural_count_encoding', 'State_count_encoding', 'BankState_count_encoding', 'City_count_encoding', 'Sector_count_encoding',
                          'GrAppv_SBA_Appv_ratio', 'DisbursementGross_GrAppv_ratio', 'Days_between_Approval_Disbursement', 'Years_since_Approval', 'NewExist_NoEmp_interaction', 'Employment_creation_ratio', 'Disbursement_per_Term', 
                          'DisbursementYear','DisbursementMonth','DisbursementDay', 'ApprovalYear','ApprovalMonth','ApprovalDay'
                          ]
numerical_features = add_numerical_features + default_numerical_features
categorical_features = ['RevLineCr', 'LowDoc', 'UrbanRural', 'State', 'Sector', 'DisbursementGross_bin',
                        'RevLineCr_LowDoc_risk_indicator', 'Franchise_risk_factor','City_State', 'Lender_Borrower_SameState', 'ApprovalFY_Term',
                        'UrbanRural_Sector_interaction',  'Sector_RevLineCr', 'Sector_LowDoc'
                        ]
features = numerical_features + categorical_features

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
    def make_features(input_df: pd.DataFrame()) -> pd.DataFrame():
        output_df = input_df.copy()

        #派生特徴量
        output_df['DisbursementGross_bin'] = pd.cut(output_df['DisbursementGross'], bins=[0, 50000, 100000, 150000, np.inf], labels=['small', 'medium', 'large', 'x-large'])
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
        # 地理的特徴の組み合わせ
        output_df['City_State'] = output_df['City'] + '_' + output_df['State']
        output_df['Lender_Borrower_SameState'] = (output_df['BankState'] == output_df['State']).astype(int)
        # 経済的特徴の組み合わせ
        output_df['Employment_creation_ratio'] = output_df['CreateJob'] / (output_df['NoEmp'] + 1) # 分母が0にならないように1を加える
        output_df['Disbursement_per_Term'] = output_df['DisbursementGross'] / (output_df['Term']+1)
        # 業種と金融条件の組み合わせ
        output_df['Sector_RevLineCr'] = output_df['Sector'].astype(str) + '_' + output_df['RevLineCr']
        output_df['Sector_LowDoc'] = output_df['Sector'].astype(str) + '_' + output_df['LowDoc']
        # 時間的特徴の組み合わせ
        output_df['ApprovalFY_Term'] = output_df['ApprovalFY'].astype(str) + '_' + output_df['Term'].astype(str)
        
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

        return output_df
    output_df = make_features(output_df)
    return output_df

train_df = Preprocessing(train_df)
test_df = Preprocessing(test_df)

for col in categorical_features+['FranchiseCode', 'BankState', 'City']:
    count_dict = dict(train_df[col].value_counts())
    train_df[f'{col}_count_encoding'] = train_df[col].map(count_dict)
    test_df[f'{col}_count_encoding'] = test_df[col].map(count_dict)

for col in categorical_features:
    encoder = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0)
    encoder.fit(combined)
    train_df[col] = encoder.transform(train_df[col])
    test_df[col] = encoder.transform(test_df[col])