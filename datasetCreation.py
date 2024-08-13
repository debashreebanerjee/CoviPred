import numpy as np
import pandas as pd
data1 = pd.read_csv('rawdata/1.csv')

data1.rename(columns = lambda x: x.replace(' ','_'), inplace = True)
data1 = data1.replace(['positive'],1)
data1 = data1.replace(['negative'],0)
data1 = data1.replace(['not_done'],None)
data1 = data1.replace(['Não Realizado'],None)
data1 = data1.replace(['not_detected'],0)
data1 = data1.replace(['detected'],1)
data1 = data1.replace(['absent'],0)
data1 = data1.replace(['present'],1)

t = len(data1)*.1
data1.dropna(thresh = t, axis = 1, inplace = True)
data1.reset_index(inplace = True, drop = True)

for i in range(0,max(data1.index)):
  if data1.loc[i]['SARS-Cov-2_exam_result'] == 0:
    if data1.loc[i].isnull().sum(axis = 0)>len(data1.iloc[1])-10:
      data1.drop(labels = None, axis = 0, index = i, columns = None, level = None, inplace = True)

data1.rename(columns = {"SARS-Cov-2_exam_result": "CoV-2"}, inplace = True)

data1.drop(labels=None, axis=1, columns=['Patient_ID'], level=None, inplace=True)
data1.reset_index(inplace = True, drop=True)
data1.to_csv('datasets/Dataset-1.csv', index=False)

data1a = data1[['Patient_age_quantile','Patient_addmited_to_regular_ward_(1=yes,_0=no)','Patient_addmited_to_semi-intensive_unit_(1=yes,_0=no)','Patient_addmited_to_intensive_care_unit_(1=yes,_0=no)','Hematocrit','Hemoglobin','Platelets','Mean_platelet_volume_(MPV)','Red_blood_Cells','Lymphocytes','Mean_corpuscular_hemoglobin_concentration_(MCHC)','Leukocytes','Basophils','Mean_corpuscular_hemoglobin_(MCH)','Eosinophils','Mean_corpuscular_volume_(MCV)','Monocytes','Red_blood_cell_distribution_width_(RDW)','CoV-2']].copy()
data1a.dropna(subset=['Hematocrit'], inplace=True)
data1a.reset_index(inplace = True, drop=True)
data1a.to_csv('datasets/Dataset-1a.csv', index=False)

data1b = data1[['Hematocrit','Hemoglobin','Platelets','Mean_platelet_volume_(MPV)','Red_blood_Cells','Lymphocytes','Mean_corpuscular_hemoglobin_concentration_(MCHC)','Leukocytes','Basophils','Mean_corpuscular_hemoglobin_(MCH)','Eosinophils','Mean_corpuscular_volume_(MCV)','Monocytes','Red_blood_cell_distribution_width_(RDW)','CoV-2']].copy()
data1b.rename(columns = {'Mean_platelet_volume_(MPV)': 'MPV','Red_blood_Cells': 'Erythrocytes','Mean_corpuscular_hemoglobin_concentration_(MCHC)': 'MCHC','Mean_corpuscular_hemoglobin_(MCH)': 'MCH','Mean_corpuscular_volume_(MCV)': 'MCV','Red_blood_cell_distribution_width_(RDW)': 'RDW'}, inplace = True)
data1b.dropna(subset=['Hematocrit'], inplace=True)
data1b.reset_index(inplace = True, drop=True)
data1b.to_csv('datasets/Dataset-1b.csv', index=False)

data1c = data1[['Leukocytes','Platelets','Eosinophils','Monocytes','CoV-2']].copy()
data1c.dropna(subset=['Platelets'], inplace=True)
data1c.reset_index(inplace = True, drop=True)
data1c.to_csv('datasets/Dataset-1c.csv', index=False)



data2 = pd.read_csv('rawdata/2.csv')
data2.rename(columns={"CA":"Calcium", "CK":"Creatine_kinase", "CREA":"Creatinine", "ALP":"Alkaline_phosphatase", "GGT":"Gamma_glutamyltransferase", "GLU":"Glucose", "AST":"Aspartate_aminotransferase", "ALT":"Alanine_aminotransferase", "LDH":"Lactate_dehydrogenase", "UREA":"Urea", "WBC":"Leukocytes_(10^9/L)", "RBC":"Erythrocytes_(10^12/L)", "HGB":"Hemoglobin", "HCT":"Hematocrit", "PLT1":"Platelets_(10^9/L)", "NE":"Neutrophils_(%)", "LY":"Lymphocytes_(%)", "MO":"Monocytes_(%)", "EO":"Eosinophils_(%)", "BA":"Basophils_(%)", "NET":"Neutrophils_(10^9/L)", "LYT":"Lymphocytes_(10^9/L)", "MOT":"Monocytes_(10^9/L)", "EOT":"Eosinophils_(10^9/L)", "BAT":"Basophils_(10^9/L)", "Suspect":"Covid-specific_symptoms", "target":"CoV-2"}, inplace=True)

data2a = data2.copy()
data2a.drop(labels=None, axis=1, columns=['Unnamed: 0','PCR','NAT','KAL'], level=None, inplace=True)

tw = (len(data2a.columns)*2)/3
data2a.dropna(thresh=tw, axis=0, inplace=True)
data2a.reset_index(inplace = True, drop=True)

cols = data2a.columns.tolist()
cols.remove('CoV-2')
cols.remove('Sex')
cols.remove('Covid-specific_symptoms')
for c in cols:
  data2a[c]=((data2a[c])-(data2a[c].mean()))/(data2a[c].std())

data2a.to_csv('datasets/Dataset-2a.csv', index=False)

data2b = data2[['Leukocytes_(10^9/L)','Platelets_(10^9/L)','Eosinophils_(10^9/L)','Monocytes_(10^9/L)','CoV-2']].copy()
data2b.rename(columns={'Leukocytes_(10^9/L)': 'Leukocytes','Platelets_(10^9/L)': 'Platelets','Eosinophils_(10^9/L)': 'Eosinophils','Monocytes_(10^9/L)': 'Monocytes'}, inplace=True)
cols = ['Leukocytes','Platelets','Eosinophils','Monocytes']
for c in cols:
  data2b[c]=((data2b[c])-(data2b[c].mean()))/(data2b[c].std())

data2b.to_csv('datasets/Dataset-2b.csv', index=False)



d = pd.read_csv('rawdata/3.csv', dtype='object')
data3 = pd.DataFrame()

temp = pd.DataFrame()
temp['ALT1'] = (pd.to_numeric(d['TGP | U/L | <=18'].copy(), errors='coerce')-9)/9
temp['ALT2'] = (pd.to_numeric(d['TGP | U/L | <=19'].copy(), errors='coerce')-9.5)/9.5
temp['ALT3'] = (pd.to_numeric(d['TGP | U/L | <=25'].copy(), errors='coerce')-12.5)/12.5
temp['ALT4'] = (pd.to_numeric(d['TGP | U/L | 0 a 33'].copy(), errors='coerce')-16.5)/16.5
temp['ALT5'] = (pd.to_numeric(d['TGP | U/L | 0 a 41'].copy(), errors='coerce')-20.5)/20.5
temp['ALT6'] = (pd.to_numeric(d['TGP | U/L | 10 a 35'].copy(), errors='coerce')-22.5)/12.5
temp['ALT7'] = (pd.to_numeric(d['TGP | U/L | 10 a 50'].copy(), errors='coerce')-30)/20
temp = temp.bfill(axis = 'columns')
data3['ALT(U/L)']=pd.to_numeric(temp['ALT1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['AST1'] = (pd.to_numeric(d['TGO | U/L | <=155'].copy(), errors='coerce')-77.5)/77.5
temp['AST2'] = (pd.to_numeric(d['TGO | U/L | <=23'].copy(), errors='coerce')-11.5)/11.5
temp['AST3'] = (pd.to_numeric(d['TGO | U/L | <=32'].copy(), errors='coerce')-16)/16
temp['AST4'] = (pd.to_numeric(d['TGO | U/L | <=33'].copy(), errors='coerce')-16.5)/16.5
temp['AST5'] = (pd.to_numeric(d['TGO | U/L | <=40'].copy(), errors='coerce')-20)/20
temp['AST6'] = (pd.to_numeric(d['TGO | U/L | <=41'].copy(), errors='coerce')-20.5)/20.5
temp['AST7'] = (pd.to_numeric(d['TGO | U/L | <=63'].copy(), errors='coerce')-31.5)/31.5
temp['AST8'] = (pd.to_numeric(d['TGO | U/L | 0 a 32'].copy(), errors='coerce')-16)/16
temp['AST9'] = (pd.to_numeric(d['TGO | U/L | 0 a 40'].copy(), errors='coerce')-20)/20
temp = temp.bfill(axis = 'columns')
data3['AST(U/L)']=pd.to_numeric(temp['AST1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Basophils1']=(pd.to_numeric(d['Basófilos # | µL | 0 a 100'].copy(), errors='coerce')-50)/50
temp['Basophils2']=(pd.to_numeric(d['Basófilos # | µL | 0 a 200'].copy(), errors='coerce')-100)/100
temp['Basophils3']=(pd.to_numeric(d['Basófilos # | µL | 0 a 600'].copy(), errors='coerce')-300)/300
temp = temp.bfill(axis = 'columns')
data3['Basophils(counts/µL)']=pd.to_numeric(temp['Basophils1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Eosinphils01']=(pd.to_numeric(d['Eosinófilos  # | µL | 0 a 500'].copy(), errors='coerce')-250)/250
temp['Eosinphils02']=(pd.to_numeric(d['Eosinófilos  # | µL | 0 a 650'].copy(), errors='coerce')-325)/325
temp['Eosinphils03']=(pd.to_numeric(d['Eosinófilos  # | µL | 20 a 850'].copy(), errors='coerce')-435)/415
temp['Eosinphils04']=(pd.to_numeric(d['Eosinófilos  # | µL | 50 a 500'].copy(), errors='coerce')-275)/225
temp = temp.bfill(axis = 'columns')
data3['Eosinophils(counts/µL)']=pd.to_numeric(temp['Eosinphils01'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['RBC1']=(pd.to_numeric(d['Hemácias | x10^6/uL | 3.00 a 5.40'].copy(), errors='coerce')-4.2)/1.2
temp['RBC2']=(pd.to_numeric(d['Hemácias | x10^6/uL | 3.10 a 4.50'].copy(), errors='coerce')-3.8)/0.7
temp['RBC3']=(pd.to_numeric(d['Hemácias | x10^6/uL | 3.60 a 6.20'].copy(), errors='coerce')-4.9)/1.3
temp['RBC4']=(pd.to_numeric(d['Hemácias | x10^6/uL | 3.70 a 5.30'].copy(), errors='coerce')-4.5)/0.8
temp['RBC5']=(pd.to_numeric(d['Hemácias | x10^6/uL | 3.90 a 5.00'].copy(), errors='coerce')-4.45)/0.55
temp['RBC6']=(pd.to_numeric(d['Hemácias | x10^6/uL | 3.90 a 5.30'].copy(), errors='coerce')-4.6)/0.7
temp['RBC7']=(pd.to_numeric(d['Hemácias | x10^6/uL | 3.90 a 6.20'].copy(), errors='coerce')-5.05)/1.15
temp['RBC8']=(pd.to_numeric(d['Hemácias | x10^6/uL | 3.90 a 6.30'].copy(), errors='coerce')-5.1)/1.2
temp['RBC9']=(pd.to_numeric(d['Hemácias | x10^6/uL | 3.90 a 6.60'].copy(), errors='coerce')-5.25)/1.35
temp['RBC10']=(pd.to_numeric(d['Hemácias | x10^6/uL | 4.00 a 5.20'].copy(), errors='coerce')-4.6)/0.6
temp['RBC11']=(pd.to_numeric(d['Hemácias | x10^6/uL | 4.10 a 5.10'].copy(), errors='coerce')-4.6)/0.5
temp['RBC12']=(pd.to_numeric(d['Hemácias | x10^6/uL | 4.30 a 5.30'].copy(), errors='coerce')-4.8)/0.5
temp['RBC13']=(pd.to_numeric(d['Hemácias | x10^6/uL | 4.30 a 5.70'].copy(), errors='coerce')-5)/0.7
temp['RBC14']=(pd.to_numeric(d['Hemácias | x10^6/uL | 4,30 a 5,70'].copy(), errors='coerce')-5)/0.7
temp['RBC15']=(pd.to_numeric(d['Hemácias | x10^6/uL | 3,90 a 5,00'].copy(), errors='coerce')-4.45)/0.55
temp = temp.bfill(axis = 'columns')
data3['Erythrocytes(millions/µL)']=pd.to_numeric(temp['RBC1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['GGT1'] = (pd.to_numeric(d['Gama GT | U/L | 17 a 175'].copy(), errors='coerce')-96)/79
temp['GGT2'] = (pd.to_numeric(d['Gama GT | U/L | 4 a 12'].copy(), errors='coerce')-8)/4
temp['GGT3'] = (pd.to_numeric(d['Gama GT | U/L | 4 a 16'].copy(), errors='coerce')-10)/6
temp['GGT4'] = (pd.to_numeric(d['Gama GT | U/L | 5 a 101'].copy(), errors='coerce')-53)/48
temp['GGT5'] = (pd.to_numeric(d['Gama GT | U/L | 5 a 36'].copy(), errors='coerce')-20.5)/15.5
temp['GGT6'] = (pd.to_numeric(d['Gama GT | U/L | 8 a 61'].copy(), errors='coerce')-34.5)/26.5
temp = temp.bfill(axis = 'columns')
data3['GGT(U/L)']=pd.to_numeric(temp['GGT1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Hematocrit1']=(pd.to_numeric(d['Hematócrito | % | 29.0 a 41.0'].copy(), errors='coerce')-35)/6
temp['Hematocrit2']=(pd.to_numeric(d['Hematócrito | % | 31.0 a 55.0'].copy(), errors='coerce')-43)/12
temp['Hematocrit3']=(pd.to_numeric(d['Hematócrito | % | 33.0 a 39.0'].copy(), errors='coerce')-36)/3
temp['Hematocrit4']=(pd.to_numeric(d['Hematócrito | % | 33.0 a 43.0'].copy(), errors='coerce')-38)/5
temp['Hematocrit5']=(pd.to_numeric(d['Hematócrito | % | 35.0 a 45.0'].copy(), errors='coerce')-40)/5
temp['Hematocrit6']=(pd.to_numeric(d['Hematócrito | % | 36.0 a 43.0'].copy(), errors='coerce')-39.5)/3.5
temp['Hematocrit7']=(pd.to_numeric(d['Hematócrito | % | 37.0 a 47.0'].copy(), errors='coerce')-42)/5
temp['Hematocrit8']=(pd.to_numeric(d['Hematócrito | % | 39.0 a 50.0'].copy(), errors='coerce')-44.5)/5.5
temp['Hematocrit9']=(pd.to_numeric(d['Hematócrito | % | 39.0 a 63.0'].copy(), errors='coerce')-51)/12
temp['Hematocrit10']=(pd.to_numeric(d['Hematócrito | % | 42.0 a 60.0'].copy(), errors='coerce')-51)/9
temp['Hematocrit11']=(pd.to_numeric(d['Hematócrito | % | 45.0 a 60.0'].copy(), errors='coerce')-52.5)/7.5
temp['Hematocrit12']=(pd.to_numeric(d['Hematócrito | % | 35,0 a 45,0'].copy(), errors='coerce')-40)/5
temp['Hematocrit13']=(pd.to_numeric(d['Hematócrito | % | 39,0 a 50,0'].copy(), errors='coerce')-44.5)/5.5
temp = temp.bfill(axis = 'columns')
data3['Hematocrit(%)']=pd.to_numeric(temp['Hematocrit1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['LDH1'] = (pd.to_numeric(d['Dehidrogenase Láctica | U/L | <=240'].copy(), errors='coerce')-120)/120
temp['LDH2'] = (pd.to_numeric(d['Dehidrogenase Láctica | U/L | <=260'].copy(), errors='coerce')-130)/130
temp['LDH3'] = (pd.to_numeric(d['Dehidrogenase Láctica | U/L | <=270'].copy(), errors='coerce')-135)/135
temp['LDH4'] = (pd.to_numeric(d['Dehidrogenase Láctica | U/L | <=305'].copy(), errors='coerce')-152.5)/152.5
temp['LDH5'] = (pd.to_numeric(d['Dehidrogenase Láctica | U/L | <=424'].copy(), errors='coerce')-212)/212
temp['LDH6'] = (pd.to_numeric(d['Dehidrogenase Láctica | U/L | 120 a 300'].copy(), errors='coerce')-210)/90
temp['LDH7'] = (pd.to_numeric(d['Dehidrogenase Láctica | U/L | 135 a 214'].copy(), errors='coerce')-174.5)/39.5
temp['LDH8'] = (pd.to_numeric(d['Dehidrogenase Láctica | U/L | 135 a 225'].copy(), errors='coerce')-180)/45
temp['LDH9'] = (pd.to_numeric(d['Dehidrogenase Láctica | U/L | 160 a 370'].copy(), errors='coerce')-265)/105
temp['LDH10'] = (pd.to_numeric(d['Dehidrogenase Láctica | U/L | 180 a 435'].copy(), errors='coerce')-307.5)/127.5
temp = temp.bfill(axis = 'columns')
data3['LDH(U/L)']=pd.to_numeric(temp['LDH1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Leukocytes01']=(pd.to_numeric(d['Leucócitos # | µL | 3500 a 10500'].copy(), errors='coerce')-7000)/3500
temp['Leukocytes02']=(pd.to_numeric(d['Leucócitos # | µL | 4500 a 13000'].copy(), errors='coerce')-8750)/4250
temp['Leukocytes03']=(pd.to_numeric(d['Leucócitos # | µL | 5000 a 14500'].copy(), errors='coerce')-9750)/4750
temp['Leukocytes04']=(pd.to_numeric(d['Leucócitos # | µL | 5000 a 19500'].copy(), errors='coerce')-12250)/7250
temp['Leukocytes05']=(pd.to_numeric(d['Leucócitos # | µL | 5000 a 20000'].copy(), errors='coerce')-12500)/7500
temp['Leukocytes06']=(pd.to_numeric(d['Leucócitos # | µL | 6000 a 17000'].copy(), errors='coerce')-11500)/5500
temp['Leukocytes07']=(pd.to_numeric(d['Leucócitos # | µL | 6000 a 17500'].copy(), errors='coerce')-11750)/5750
temp['Leukocytes08']=(pd.to_numeric(d['Leucócitos # | µL | 9000 a 30000'].copy(), errors='coerce')-19500)/10500
temp['Leukocytes09']=(pd.to_numeric(d['Leucócitos # | µL | 9400 a 34000'].copy(), errors='coerce')-21700)/12300
temp = temp.bfill(axis = 'columns')
data3['Leukocytes(counts/µL)']=pd.to_numeric(temp['Leukocytes01'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Lymphocytes1']=(pd.to_numeric(d['Linfócitos # | µL | 1200 a 5200'].copy(), errors='coerce')-3200)/2000
temp['Lymphocytes2']=(pd.to_numeric(d['Linfócitos # | µL | 1500 a 6500'].copy(), errors='coerce')-4000)/2500
temp['Lymphocytes3']=(pd.to_numeric(d['Linfócitos # | µL | 1500 a 8500'].copy(), errors='coerce')-5000)/3500
temp['Lymphocytes4']=(pd.to_numeric(d['Linfócitos # | µL | 2000 a 11000'].copy(), errors='coerce')-6500)/4500
temp['Lymphocytes5']=(pd.to_numeric(d['Linfócitos # | µL | 2000 a 17000'].copy(), errors='coerce')-9500)/7500
temp['Lymphocytes6']=(pd.to_numeric(d['Linfócitos # | µL | 2500 a 16500'].copy(), errors='coerce')-9500)/7000
temp['Lymphocytes7']=(pd.to_numeric(d['Linfócitos # | µL | 300 a 9500'].copy(), errors='coerce')-4900)/4600
temp['Lymphocytes8']=(pd.to_numeric(d['Linfócitos # | µL | 4000 a 13500'].copy(), errors='coerce')-8750)/4750
temp['Lymphocytes9']=(pd.to_numeric(d['Linfócitos # | µL | 900 a 2900'].copy(), errors='coerce')-1900)/1000
temp = temp.bfill(axis = 'columns')
data3['Lymphocytes(counts/µL)']=pd.to_numeric(temp['Lymphocytes1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Mature Neutrophils1'] = (pd.to_numeric(d['Segmentados # | µL | 1000 a 8500'].copy(), errors='coerce')-4750)/3750
temp['Mature Neutrophils2'] = (pd.to_numeric(d['Segmentados # | µL | 1000 a 9000'].copy(), errors='coerce')-5000)/4000
temp['Mature Neutrophils3'] = (pd.to_numeric(d['Segmentados # | µL | 1000 a 9500'].copy(), errors='coerce')-5250)/4250
temp['Mature Neutrophils4'] = (pd.to_numeric(d['Segmentados # | µL | 1500 a 10000'].copy(), errors='coerce')-5750)/4250
temp['Mature Neutrophils5'] = (pd.to_numeric(d['Segmentados # | µL | 1500 a 8500'].copy(), errors='coerce')-5000)/3500
temp['Mature Neutrophils6'] = (pd.to_numeric(d['Segmentados # | µL | 1700 a 8000'].copy(), errors='coerce')-4850)/3150
temp['Mature Neutrophils7'] = (pd.to_numeric(d['Segmentados # | µL | 1800 a 8000'].copy(), errors='coerce')-4900)/3100
temp['Mature Neutrophils8'] = (pd.to_numeric(d['Segmentados # | µL | 6000 a 26000'].copy(), errors='coerce')-16000)/10000
temp = temp.bfill(axis = 'columns')
data3['MatureNeutrophils(counts/µL)']=pd.to_numeric(temp['Mature Neutrophils1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Monocytes01']=(pd.to_numeric(d['Monócitos # | µL | 0 a 800'].copy(), errors='coerce')-400)/400
temp['Monocytes02']=(pd.to_numeric(d['Monócitos # | µL | 100 a 800'].copy(), errors='coerce')-450)/350
temp['Monocytes03']=(pd.to_numeric(d['Monócitos # | µL | 300 a 900'].copy(), errors='coerce')-600)/300
temp['Monocytes04']=(pd.to_numeric(d['Monócitos # | µL | 400 a 1800'].copy(), errors='coerce')-1100)/700
temp['Monocytes05']=(pd.to_numeric(d['Monócitos # | µL | 50 a 1100'].copy(), errors='coerce')-575)/525
temp['Monocytes06']=(pd.to_numeric(d['Monócitos # | µL | 50 a 800'].copy(), errors='coerce')-425)/375
temp = temp.bfill(axis = 'columns')
data3['Monocytes(counts/µL)']=pd.to_numeric(temp['Monocytes01'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Neutrophils1'] = (pd.to_numeric(d['Neutrófilos  # | µL | 1000 a 8500'].copy(), errors='coerce')-4750)/3750
temp['Neutrophils2'] = (pd.to_numeric(d['Neutrófilos  # | µL | 1000 a 9500'].copy(), errors='coerce')-5250)/4250
temp['Neutrophils3'] = (pd.to_numeric(d['Neutrófilos  # | µL | 1500 a 10000'].copy(), errors='coerce')-5750)/4250
temp['Neutrophils4'] = (pd.to_numeric(d['Neutrófilos  # | µL | 1500 a 8500'].copy(), errors='coerce')-5000)/3500
temp['Neutrophils5'] = (pd.to_numeric(d['Neutrófilos  # | µL | 1700 a 8000'].copy(), errors='coerce')-4850)/3150
temp['Neutrophils6'] = (pd.to_numeric(d['Neutrófilos  # | µL | 1800 a 8000'].copy(), errors='coerce')-4900)/3100
temp['Neutrophils7'] = (pd.to_numeric(d['Neutrófilos  # | µL | 6000 a 26000'].copy(), errors='coerce')-16000)/10000
temp = temp.bfill(axis = 'columns')
data3['Neutrophils(counts/µL)']=pd.to_numeric(temp['Neutrophils1'].copy(), errors='coerce')

"""
temp = pd.DataFrame()
temp['pO2Art1'] = (pd.to_numeric(d['pO2 (gasometria arterial) | mm Hg | 80.0 a 90.0'].copy(), errors='coerce')-)85/5
temp['pO2Art2'] = (pd.to_numeric(d['pO2 (gasometria arterial) | NULL | 80.0 a 90.0'].copy(), errors='coerce')-)85/5
temp = temp.bfill(axis = 'columns')
data3['Arterial-pO2(mmHg)']=pd.to_numeric(temp['pO2Art1'].copy(), errors='coerce')
"""

temp = pd.DataFrame()
temp['Albumin01'] = (pd.to_numeric(d['Albumina | g/dL | 3.1 a 5.0'].copy(), errors='coerce')-4.05)/0.95
temp['Albumin02'] = (pd.to_numeric(d['Albumina | g/dL | 3.20 a 4.50'].copy(), errors='coerce')-3.85)/0.65
temp['Albumin03'] = (pd.to_numeric(d['Albumina | g/dL | 3.5 a 5.2'].copy(), errors='coerce')-4.35)/0.85
temp['Albumin04'] = (pd.to_numeric(d['Albumina | g/dL | 3.50 a 5.20'].copy(), errors='coerce')-4.35)/0.85
temp['Albumin05'] = (pd.to_numeric(d['Albumina | g/dL | 3.80 a 5.40'].copy(), errors='coerce')-4.6)/0.8
temp['Albumin06'] = (pd.to_numeric(d['Albumina | g/dL | 4.0 a 4.9'].copy(), errors='coerce')-4.45)/0.45
temp['Albumin07'] = (pd.to_numeric(d['Albumina | g/dL | 4.0 a 5.3'].copy(), errors='coerce')-4.65)/0.65
temp['Albumin08'] = (pd.to_numeric(d['Albumina | g/dL | 4.2 a 5.1'].copy(), errors='coerce')-4.65)/0.45
temp['Albumin09'] = (pd.to_numeric(d['Albumina | g/dL | 4.3 a 5.3'].copy(), errors='coerce')-4.8)/0.5
temp['Albumin010'] = (pd.to_numeric(d['Albumina | g/dL | 3,5 a 5,2'].copy(), errors='coerce')-4.35)/0.85
temp = temp.bfill(axis = 'columns')
data3['SerumAlbumin(g/dL)']=pd.to_numeric(temp['Albumin01'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Calcium1'] = (pd.to_numeric(d['Cálcio Iônico mmol/L | mmol/L | 0.95 a 1.50'].copy(), errors='coerce')-1.225)/0.275
temp['Calcium2'] = (pd.to_numeric(d['Cálcio Iônico mmol/L | mmol/L | 1.00 a 1.50'].copy(), errors='coerce')-1.25)/0.25
temp['Calcium3'] = (pd.to_numeric(d['Cálcio Iônico mmol/L | mmol/L | 1.14 a 1.31'].copy(), errors='coerce')-1.225)/0.085
temp = temp.bfill(axis = 'columns')
data3['SerumCalcium(mmol/L)']=pd.to_numeric(temp['Calcium1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Ferritin01'] = (pd.to_numeric(d['Ferritina | ng/mL | 12.00 a 266.00'].copy(), errors='coerce')-139)/127
temp['Ferritin02'] = (pd.to_numeric(d['Ferritina | ng/mL | 14.00 a 101.00'].copy(), errors='coerce')-57.5)/43.5
temp['Ferritin03'] = (pd.to_numeric(d['Ferritina | ng/mL | 150.00 a 973.00'].copy(), errors='coerce')-561.5)/411.5
temp['Ferritin04'] = (pd.to_numeric(d['Ferritina | ng/mL | 20.90 a 173.00'].copy(), errors='coerce')-96.95)/76.05
temp['Ferritin05'] = (pd.to_numeric(d['Ferritina | ng/mL | 22.00 a 491.00'].copy(), errors='coerce')-256.5)/234.5
temp['Ferritin06'] = (pd.to_numeric(d['Ferritina | ng/mL | 3.88 a 114.00'].copy(), errors='coerce')-58.94)/55.06
temp['Ferritin07'] = (pd.to_numeric(d['Ferritina | ng/mL | 8.46 a 580.00'].copy(), errors='coerce')-294.23)/285.77
temp = temp.bfill(axis = 'columns')
data3['SerumFerritin(ng/mL)']=pd.to_numeric(temp['Ferritin01'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Magnesium1'] = (pd.to_numeric(d['Magnésio | mEq/L | 1.2 a 1.8'].copy(), errors='coerce')-1.5)/0.3
temp['Magnesium2'] = (pd.to_numeric(d['Magnésio | mEq/L | 1.3 a 2.1'].copy(), errors='coerce')-1.7)/0.4
temp['Magnesium3'] = (pd.to_numeric(d['Magnésio | mEq/L | 1.4 a 1.7'].copy(), errors='coerce')-1.55)/0.15
temp['Magnesium4'] = (pd.to_numeric(d['Magnésio | mEq/L | 1.4 a 1.9'].copy(), errors='coerce')-1.65)/0.25
temp = temp.bfill(axis = 'columns')
data3['SerumMagnesium(mEq/L)']=pd.to_numeric(temp['Magnesium1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Phosphorus01'] = (pd.to_numeric(d['Fósforo | mg/dL | 2.5 a 4.5'].copy(), errors='coerce')-3.5)/1
temp['Phosphorus02'] = (pd.to_numeric(d['Fósforo | mg/dL | 2.5 a 4.8'].copy(), errors='coerce')-3.65)/1.15
temp['Phosphorus03'] = (pd.to_numeric(d['Fósforo | mg/dL | 2.7 a 4.9'].copy(), errors='coerce')-3.8)/1.1
temp['Phosphorus04'] = (pd.to_numeric(d['Fósforo | mg/dL | 2.8 a 4.8'].copy(), errors='coerce')-3.8)/1
temp['Phosphorus05'] = (pd.to_numeric(d['Fósforo | mg/dL | 2.9 a 4.8'].copy(), errors='coerce')-3.85)/0.95
temp['Phosphorus06'] = (pd.to_numeric(d['Fósforo | mg/dL | 2.9 a 5.1'].copy(), errors='coerce')-4)/1.1
temp['Phosphorus07'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.0 a 5.4'].copy(), errors='coerce')-4.2)/1.2
temp['Phosphorus08'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.1 a 5.3'].copy(), errors='coerce')-4.2)/1.1
temp['Phosphorus09'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.1 a 5.5'].copy(), errors='coerce')-4.3)/1.2
temp['Phosphorus10'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.1 a 6.0'].copy(), errors='coerce')-4.55)/1.45
temp['Phosphorus11'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.2 a 5.5'].copy(), errors='coerce')-4.35)/1.15
temp['Phosphorus12'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.2 a 5.7'].copy(), errors='coerce')-4.45)/1.25
temp['Phosphorus13'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.3 a 5.3'].copy(), errors='coerce')-4.3)/1
temp['Phosphorus14'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.3 a 5.6'].copy(), errors='coerce')-4.45)/1.15
temp['Phosphorus15'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.4 a 6.0'].copy(), errors='coerce')-4.7)/1.3
temp['Phosphorus16'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.5 a 5.8'].copy(), errors='coerce')-4.65)/1.15
temp['Phosphorus17'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.5 a 6.6'].copy(), errors='coerce')-5.05)/1.55
temp['Phosphorus18'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.7 a 6.5'].copy(), errors='coerce')-5.1)/1.4
temp['Phosphorus19'] = (pd.to_numeric(d['Fósforo | mg/dL | 3.9 a 6.9'].copy(), errors='coerce')-5.4)/1.5
temp['Phosphorus20'] = (pd.to_numeric(d['Fósforo | mg/dL | 4.0 a 5.6'].copy(), errors='coerce')-4.8)/0.8
temp['Phosphorus21'] = (pd.to_numeric(d['Fósforo | mg/dL | 4.1 a 6.4'].copy(), errors='coerce')-5.25)/1.15
temp['Phosphorus22'] = (pd.to_numeric(d['Fósforo | mg/dL | 4.6 a 7.9'].copy(), errors='coerce')-6.25)/1.65
temp['Phosphorus23'] = (pd.to_numeric(d['Fósforo | mg/dL | 5.3 a 9.8'].copy(), errors='coerce')-7.55)/2.25
temp = temp.bfill(axis = 'columns')
data3['SerumPhosphorus(mg/dL)']=pd.to_numeric(temp['Phosphorus01'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Total Bilirubin1'] = (pd.to_numeric(d['Bilirrubina Total | mg/dL | 0.0 a 10.8'].copy(), errors='coerce')-5.4)/5.4
temp['Total Bilirubin2'] = (pd.to_numeric(d['Bilirrubina Total | mg/dL | 0.0 a 13.4'].copy(), errors='coerce')-6.7)/6.7
temp['Total Bilirubin3'] = (pd.to_numeric(d['Bilirrubina Total | mg/dL | 0.0 a 6.4'].copy(), errors='coerce')-3.2)/3.2
temp['Total Bilirubin4'] = (pd.to_numeric(d['Bilirrubina Total | mg/dL | 0.1 a 1.2'].copy(), errors='coerce')-0.65)/0.55
temp = temp.bfill(axis = 'columns')
data3['TotalBilirubin(mg/dL)']=pd.to_numeric(temp['Total Bilirubin1'].copy(), errors='coerce')

data3['Platelets(thousands/µL)']=pd.to_numeric(d['Plaquetas | x10^3/uL | '].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Hemoglobin01']=(pd.to_numeric(d['Hemoglobina | g/dL | 10.0 a 18.0'].copy(), errors='coerce')-14)/4
temp['Hemoglobin02']=(pd.to_numeric(d['Hemoglobina | g/dL | 10.5 a 13.5'].copy(), errors='coerce')-12)/1.5
temp['Hemoglobin03']=(pd.to_numeric(d['Hemoglobina | g/dL | 11.0 a 14.5'].copy(), errors='coerce')-12.75)/1.75
temp['Hemoglobin04']=(pd.to_numeric(d['Hemoglobina | g/dL | 11.5 a 14.5'].copy(), errors='coerce')-13)/1.5
temp['Hemoglobin05']=(pd.to_numeric(d['Hemoglobina | g/dL | 12.0 a 14.8'].copy(), errors='coerce')-13.4)/1.4
temp['Hemoglobin06']=(pd.to_numeric(d['Hemoglobina | g/dL | 12.0 a 15.5'].copy(), errors='coerce')-13.75)/1.75
temp['Hemoglobin07']=(pd.to_numeric(d['Hemoglobina | g/dL | 12.5 a 20.5'].copy(), errors='coerce')-16.5)/4
temp['Hemoglobin08']=(pd.to_numeric(d['Hemoglobina | g/dL | 13.0 a 16.0'].copy(), errors='coerce')-14.5)/1.5
temp['Hemoglobin09']=(pd.to_numeric(d['Hemoglobina | g/dL | 13.5 a 17.5'].copy(), errors='coerce')-15.5)/2
temp['Hemoglobin010']=(pd.to_numeric(d['Hemoglobina | g/dL | 13.5 a 21.5'].copy(), errors='coerce')-17.5)/4
temp['Hemoglobin011']=(pd.to_numeric(d['Hemoglobina | g/dL | 14.5 a 22.5'].copy(), errors='coerce')-18.5)/4
temp['Hemoglobin012']=(pd.to_numeric(d['Hemoglobina | g/dL | 9.0 a 14.0'].copy(), errors='coerce')-11.5)/2.5
temp['Hemoglobin013']=(pd.to_numeric(d['Hemoglobina | g/dL | 12,0 a 15,5'].copy(), errors='coerce')-13.75)/1.75
temp['Hemoglobin014']=(pd.to_numeric(d['Hemoglobina | g/dL | 13,5 a 17,5'].copy(), errors='coerce')-15.5)/2
temp = temp.bfill(axis = 'columns')
data3['Hemoglobin(g/dL)']=pd.to_numeric(temp['Hemoglobin01'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['MPV1'] = (pd.to_numeric(d['Volume Médio Plaquetário | fL | 6,5 a 15,0'].copy(), errors='coerce')-10.75)/4.25
temp['MPV2'] = (pd.to_numeric(d['Volume Médio Plaquetário | fL | 6.5 a 15.0'].copy(), errors='coerce')-10.75)/4.25
temp = temp.bfill(axis = 'columns')
data3['MPV(fL)']=pd.to_numeric(temp['MPV1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['MCH1']=(pd.to_numeric(d['HCM | pg | 23.0 a 31.0'].copy(), errors='coerce')-27)/4
temp['MCH2']=(pd.to_numeric(d['HCM | pg | 24.0 a 32.0'].copy(), errors='coerce')-28)/4
temp['MCH3']=(pd.to_numeric(d['HCM | pg | 25.0 a 33.0'].copy(), errors='coerce')-29)/4
temp['MCH4']=(pd.to_numeric(d['HCM | pg | 25.0 a 35.0'].copy(), errors='coerce')-30)/5
temp['MCH5']=(pd.to_numeric(d['HCM | pg | 25.0 a 36.0'].copy(), errors='coerce')-30.5)/5.5
temp['MCH6']=(pd.to_numeric(d['HCM | pg | 26.0 a 34.0'].copy(), errors='coerce')-30)/4
temp['MCH7']=(pd.to_numeric(d['HCM | pg | 28.0 a 40.0'].copy(), errors='coerce')-34)/6
temp['MCH8']=(pd.to_numeric(d['HCM | pg | 29.0 a 37.0'].copy(), errors='coerce')-33)/4
temp['MCH9']=(pd.to_numeric(d['HCM | pg | 31.0 a 37.0'].copy(), errors='coerce')-34)/3
temp['MCH10']=(pd.to_numeric(d['HCM | pg | 26,0 a 34,0'].copy(), errors='coerce')-30)/4
temp = temp.bfill(axis = 'columns')
data3['MCH(pg)']=pd.to_numeric(temp['MCH1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['MCHC01']=(pd.to_numeric(d['CHCM | g/dL | 28.0 a 38.0'].copy(), errors='coerce')-33)/5
temp['MCHC02']=(pd.to_numeric(d['CHCM | g/dL | 29.0 a 36.0'].copy(), errors='coerce')-32.5)/3.5
temp['MCHC03']=(pd.to_numeric(d['CHCM | g/dL | 29.0 a 37.0'].copy(), errors='coerce')-33)/4
temp['MCHC04']=(pd.to_numeric(d['CHCM | g/dL | 30.0 a 36.0'].copy(), errors='coerce')-33)/3
temp['MCHC05']=(pd.to_numeric(d['CHCM | g/dL | 31.0 a 36.0'].copy(), errors='coerce')-33.5)/2.5
temp['MCHC06']=(pd.to_numeric(d['CHCM | g/dL | 32.0 a 36.0'].copy(), errors='coerce')-34)/2
temp['MCHC07']=(pd.to_numeric(d['CHCM | g/dL | 31,0 a 36,0'].copy(), errors='coerce')-33.5)/2.5
temp = temp.bfill(axis = 'columns')
data3['MCHC(g/dL)']=pd.to_numeric(temp['MCHC01'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['MCV01']=(pd.to_numeric(d['VCM | fL | 70.0 a 86.0'].copy(), errors='coerce')-78)/8
temp['MCV02']=(pd.to_numeric(d['VCM | fL | 74.0 a 89.0'].copy(), errors='coerce')-81.5)/7.5
temp['MCV03']=(pd.to_numeric(d['VCM | fL | 77.0 a 115.0'].copy(), errors='coerce')-96)/19
temp['MCV04']=(pd.to_numeric(d['VCM | fL | 77.0 a 91.0'].copy(), errors='coerce')-84)/7
temp['MCV05']=(pd.to_numeric(d['VCM | fL | 80.0 a 92.0'].copy(), errors='coerce')-86)/6
temp['MCV06']=(pd.to_numeric(d['VCM | fL | 81.0 a 92.0'].copy(), errors='coerce')-86.5)/5.5
temp['MCV07']=(pd.to_numeric(d['VCM | fL | 81.0 a 95.0'].copy(), errors='coerce')-88)/7
temp['MCV08']=(pd.to_numeric(d['VCM | fL | 82.0 a 98.0'].copy(), errors='coerce')-90)/8
temp['MCV09']=(pd.to_numeric(d['VCM | fL | 85.0 a 123.0'].copy(), errors='coerce')-104)/19
temp['MCV10']=(pd.to_numeric(d['VCM | fL | 86.0 a 124.0'].copy(), errors='coerce')-105)/19
temp['MCV11']=(pd.to_numeric(d['VCM | fL | 88.0 a 126.0'].copy(), errors='coerce')-107)/19
temp['MCV12']=(pd.to_numeric(d['VCM | fL | 95.0 a 115.0'].copy(), errors='coerce')-105)/10
temp['MCV13']=(pd.to_numeric(d['VCM | fL | 81,0 a 95,0'].copy(), errors='coerce')-88)/7
temp['MCV14']=(pd.to_numeric(d['VCM | fL | 82,0 a 98,0'].copy(), errors='coerce')-90)/8
temp = temp.bfill(axis = 'columns')
data3['MCV(fL)']=pd.to_numeric(temp['MCV01'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['RDW2'] = (pd.to_numeric(d['RDW | % | 11,5 a 16,5'].copy(), errors='coerce')-14)/2.5
temp['RDW3'] = (pd.to_numeric(d['RDW | % | 11.5 a 16.5'].copy(), errors='coerce')-14)/2.5
temp['RDW4'] = (pd.to_numeric(d['RDW | % | 12.0 a 14.5'].copy(), errors='coerce')-13.25)/1.25
temp = temp.bfill(axis = 'columns')
data3['RDW(%)']=pd.to_numeric(temp['RDW2'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['ALP1'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 116 a 468'].copy(), errors='coerce')-292)/176
temp['ALP2'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 122 a 469'].copy(), errors='coerce')-295.5)/173.5
temp['ALP3'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 129 a 417'].copy(), errors='coerce')-273)/144
temp['ALP4'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 142 a 335'].copy(), errors='coerce')-238.5)/96.5
temp['ALP5'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 35 a 104'].copy(), errors='coerce')-69.5)/34.5
temp['ALP6'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 40 a 129'].copy(), errors='coerce')-84.5)/44.5
temp['ALP7'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 45 a 87'].copy(), errors='coerce')-66)/21
temp['ALP8'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 50 a 117'].copy(), errors='coerce')-83.5)/33.5
temp['ALP9'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 55 a 149'].copy(), errors='coerce')-102)/47
temp['ALP10'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 57 a 254'].copy(), errors='coerce')-155.5)/98.5
temp['ALP11'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 82 a 331'].copy(), errors='coerce')-206.5)/124.5
temp['ALP12'] = (pd.to_numeric(d['Fosfatase Alcalina | U/L | 83 a 248'].copy(), errors='coerce')-165.5)/82.5
temp = temp.bfill(axis = 'columns')
data3['ALP(U/L)']=pd.to_numeric(temp['ALP1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['CoV-21'] = d['Resultado COVID-19: | NULL | '].copy()
temp['CoV-22'] = d['Resultado COVID-19: | NULL | Não detectado'].copy()
temp = temp.replace(['Não detectado'], 0)
temp = temp.replace(['Detectado'], 1)
temp = temp.bfill(axis = 'columns')
data3['CoV-2']=pd.to_numeric(temp['CoV-21'].copy(), errors='coerce')

data3.dropna(subset = ['CoV-2'], inplace=True)
t=(len(data3.columns)*2)/3
data3.dropna(thresh=t, axis=0, inplace=True)

cols = data3.columns.tolist()
cols.remove('CoV-2')
for c in cols:
  data3[c]=((data3[c])-(data3[c].mean()))/(data3[c].std())

data3.reset_index(inplace = True, drop=True)
data3.info()
data3.to_csv('datasets/Dataset-3.csv', index=False)

d = pd.read_csv('rawdata/3.csv', dtype='object')

data3a = pd.DataFrame()
data3a['ID'] = d['ID'].copy()
data3a['Date_Collected'] = d['Date Collected'].copy()

temp = pd.DataFrame()
temp['ALT_(U/L)1'] = d['TGP | U/L | '].copy()
temp['ALT_(U/L)2'] = d['TGP | U/L | <=18'].copy()
temp['ALT_(U/L)3'] = d['TGP | U/L | <=19'].copy()
temp['ALT_(U/L)4'] = d['TGP | U/L | <=25'].copy()
temp['ALT_(U/L)5'] = d['TGP | U/L | 0 a 33'].copy()
temp['ALT_(U/L)6'] = d['TGP | U/L | 0 a 41'].copy()
temp['ALT_(U/L)7'] = d['TGP | U/L | 10 a 35'].copy()
temp['ALT_(U/L)8'] = d['TGP | U/L | 10 a 50'].copy()
temp = temp.bfill(axis = 'columns')
data3a['ALT_(U/L)']=pd.to_numeric(temp['ALT_(U/L)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['AST_(U/L)1'] = d['TGO | U/L | '].copy()
temp['AST_(U/L)2'] = d['TGO | U/L | <=155'].copy()
temp['AST_(U/L)3'] = d['TGO | U/L | <=23'].copy()
temp['AST_(U/L)4'] = d['TGO | U/L | <=32'].copy()
temp['AST_(U/L)5'] = d['TGO | U/L | <=33'].copy()
temp['AST_(U/L)6'] = d['TGO | U/L | <=40'].copy()
temp['AST_(U/L)7'] = d['TGO | U/L | <=41'].copy()
temp['AST_(U/L)8'] = d['TGO | U/L | <=63'].copy()
temp['AST_(U/L)9'] = d['TGO | U/L | 0 a 32'].copy()
temp['AST_(U/L)10'] = d['TGO | U/L | 0 a 40'].copy()
temp = temp.bfill(axis = 'columns')
data3a['AST_(U/L)']=pd.to_numeric(temp['AST_(U/L)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Basophils_(counts/µL)1'] = d['Basófilos # | µL | 0 a 100'].copy()
temp['Basophils_(counts/µL)2'] = d['Basófilos # | µL | 0 a 200'].copy()
temp['Basophils_(counts/µL)3'] = d['Basófilos # | µL | 0 a 600'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Basophils_(counts/µL)']=pd.to_numeric(temp['Basophils_(counts/µL)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Eosinophils_(counts/µL)1'] = d['Eosinófilos  # | µL | '].copy()
temp['Eosinophils_(counts/µL)2'] = d['Eosinófilos  # | µL | 0 a 500'].copy()
temp['Eosinophils_(counts/µL)3'] = d['Eosinófilos  # | µL | 0 a 650'].copy()
temp['Eosinophils_(counts/µL)4'] = d['Eosinófilos  # | µL | 20 a 850'].copy()
temp['Eosinophils_(counts/µL)5'] = d['Eosinófilos  # | µL | 50 a 500'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Eosinophils_(counts/µL)']=pd.to_numeric(temp['Eosinophils_(counts/µL)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Erythrocytes_(counts(×10^6)/µL)1'] = d['Hemácias | x10^6/uL | 3,90 a 5,00'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)2'] = d['Hemácias | x10^6/uL | 3.00 a 5.40'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)3'] = d['Hemácias | x10^6/uL | 3.10 a 4.50'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)4'] = d['Hemácias | x10^6/uL | 3.60 a 6.20'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)5'] = d['Hemácias | x10^6/uL | 3.70 a 5.30'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)6'] = d['Hemácias | x10^6/uL | 3.90 a 5.00'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)7'] = d['Hemácias | x10^6/uL | 3.90 a 5.30'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)8'] = d['Hemácias | x10^6/uL | 3.90 a 6.20'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)9'] = d['Hemácias | x10^6/uL | 3.90 a 6.30'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)10'] = d['Hemácias | x10^6/uL | 3.90 a 6.60'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)11'] = d['Hemácias | x10^6/uL | 4,30 a 5,70'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)12'] = d['Hemácias | x10^6/uL | 4.00 a 5.20'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)13'] = d['Hemácias | x10^6/uL | 4.10 a 5.10'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)14'] = d['Hemácias | x10^6/uL | 4.30 a 5.30'].copy()
temp['Erythrocytes_(counts(×10^6)/µL)15'] = d['Hemácias | x10^6/uL | 4.30 a 5.70'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Erythrocytes_(counts(×10^6)/µL)']=pd.to_numeric(temp['Erythrocytes_(counts(×10^6)/µL)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Gamma-GT_(U/L)1'] = d['Gama GT | U/L | '].copy()
temp['Gamma-GT_(U/L)2'] = d['Gama GT | U/L | '].copy()
temp['Gamma-GT_(U/L)3'] = d['Gama GT | U/L | 17 a 175'].copy()
temp['Gamma-GT_(U/L)4'] = d['Gama GT | U/L | 17 a 175'].copy()
temp['Gamma-GT_(U/L)5'] = d['Gama GT | U/L | 4 a 12'].copy()
temp['Gamma-GT_(U/L)6'] = d['Gama GT | U/L | 4 a 12'].copy()
temp['Gamma-GT_(U/L)7'] = d['Gama GT | U/L | 4 a 16'].copy()
temp['Gamma-GT_(U/L)8'] = d['Gama GT | U/L | 4 a 16'].copy()
temp['Gamma-GT_(U/L)9'] = d['Gama GT | U/L | 5 a 101'].copy()
temp['Gamma-GT_(U/L)10'] = d['Gama GT | U/L | 5 a 101'].copy()
temp['Gamma-GT_(U/L)11'] = d['Gama GT | U/L | 5 a 36'].copy()
temp['Gamma-GT_(U/L)12'] = d['Gama GT | U/L | 5 a 36'].copy()
temp['Gamma-GT_(U/L)13'] = d['Gama GT | U/L | 8 a 61'].copy()
temp['Gamma-GT_(U/L)14'] = d['Gama GT | U/L | 8 a 61'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Gamma-GT_(U/L)']=pd.to_numeric(temp['Gamma-GT_(U/L)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Hematocrit1'] = d['Hematócrito | % | '].copy()
temp['Hematocrit2'] = d['Hematócrito | % | 29.0 a 41.0'].copy()
temp['Hematocrit3'] = d['Hematócrito | % | 31.0 a 55.0'].copy()
temp['Hematocrit4'] = d['Hematócrito | % | 33.0 a 39.0'].copy()
temp['Hematocrit5'] = d['Hematócrito | % | 33.0 a 43.0'].copy()
temp['Hematocrit6'] = d['Hematócrito | % | 35,0 a 45,0'].copy()
temp['Hematocrit7'] = d['Hematócrito | % | 35.0 a 45.0'].copy()
temp['Hematocrit8'] = d['Hematócrito | % | 36.0 a 43.0'].copy()
temp['Hematocrit9'] = d['Hematócrito | % | 37.0 a 47.0'].copy()
temp['Hematocrit10'] = d['Hematócrito | % | 39,0 a 50,0'].copy()
temp['Hematocrit11'] = d['Hematócrito | % | 39.0 a 50.0'].copy()
temp['Hematocrit12'] = d['Hematócrito | % | 39.0 a 63.0'].copy()
temp['Hematocrit13'] = d['Hematócrito | % | 42.0 a 60.0'].copy()
temp['Hematocrit14'] = d['Hematócrito | % | 45.0 a 60.0'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Hematocrit']=pd.to_numeric(temp['Hematocrit1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Lactate_Dehydrogenase_(U/L)1'] = d['Dehidrogenase Láctica | U/L | '].copy()
temp['Lactate_Dehydrogenase_(U/L)2'] = d['Dehidrogenase Láctica | U/L | <=240'].copy()
temp['Lactate_Dehydrogenase_(U/L)3'] = d['Dehidrogenase Láctica | U/L | <=260'].copy()
temp['Lactate_Dehydrogenase_(U/L)4'] = d['Dehidrogenase Láctica | U/L | <=270'].copy()
temp['Lactate_Dehydrogenase_(U/L)5'] = d['Dehidrogenase Láctica | U/L | <=305'].copy()
temp['Lactate_Dehydrogenase_(U/L)6'] = d['Dehidrogenase Láctica | U/L | <=424'].copy()
temp['Lactate_Dehydrogenase_(U/L)7'] = d['Dehidrogenase Láctica | U/L | 120 a 300'].copy()
temp['Lactate_Dehydrogenase_(U/L)8'] = d['Dehidrogenase Láctica | U/L | 135 a 214'].copy()
temp['Lactate_Dehydrogenase_(U/L)9'] = d['Dehidrogenase Láctica | U/L | 135 a 225'].copy()
temp['Lactate_Dehydrogenase_(U/L)10'] = d['Dehidrogenase Láctica | U/L | 160 a 370'].copy()
temp['Lactate_Dehydrogenase_(U/L)11'] = d['Dehidrogenase Láctica | U/L | 180 a 435'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Lactate_Dehydrogenase_(U/L)']=pd.to_numeric(temp['Lactate_Dehydrogenase_(U/L)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Leukocytes_(counts/µL)1'] = d['Leucócitos # | µL | '].copy()
temp['Leukocytes_(counts/µL)2'] = d['Leucócitos # | µL | 3500 a 10500'].copy()
temp['Leukocytes_(counts/µL)3'] = d['Leucócitos # | µL | 4500 a 13000'].copy()
temp['Leukocytes_(counts/µL)4'] = d['Leucócitos # | µL | 5000 a 14500'].copy()
temp['Leukocytes_(counts/µL)5'] = d['Leucócitos # | µL | 5000 a 19500'].copy()
temp['Leukocytes_(counts/µL)6'] = d['Leucócitos # | µL | 5000 a 20000'].copy()
temp['Leukocytes_(counts/µL)7'] = d['Leucócitos # | µL | 6000 a 17000'].copy()
temp['Leukocytes_(counts/µL)8'] = d['Leucócitos # | µL | 6000 a 17500'].copy()
temp['Leukocytes_(counts/µL)9'] = d['Leucócitos # | µL | 9000 a 30000'].copy()
temp['Leukocytes_(counts/µL)10'] = d['Leucócitos # | µL | 9400 a 34000'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Leukocytes_(counts/µL)']=pd.to_numeric(temp['Leukocytes_(counts/µL)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Lymphocytes_(counts/µL)1'] = d['Linfócitos # | µL | '].copy()
temp['Lymphocytes_(counts/µL)2'] = d['Linfócitos # | µL | 1200 a 5200'].copy()
temp['Lymphocytes_(counts/µL)3'] = d['Linfócitos # | µL | 1500 a 6500'].copy()
temp['Lymphocytes_(counts/µL)4'] = d['Linfócitos # | µL | 1500 a 8500'].copy()
temp['Lymphocytes_(counts/µL)5'] = d['Linfócitos # | µL | 2000 a 11000'].copy()
temp['Lymphocytes_(counts/µL)6'] = d['Linfócitos # | µL | 2000 a 17000'].copy()
temp['Lymphocytes_(counts/µL)7'] = d['Linfócitos # | µL | 2500 a 16500'].copy()
temp['Lymphocytes_(counts/µL)8'] = d['Linfócitos # | µL | 300 a 9500'].copy()
temp['Lymphocytes_(counts/µL)9'] = d['Linfócitos # | µL | 4000 a 13500'].copy()
temp['Lymphocytes_(counts/µL)10'] = d['Linfócitos # | µL | 900 a 2900'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Lymphocytes_(counts/µL)']=pd.to_numeric(temp['Lymphocytes_(counts/µL)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Mature_Neutrophils_(counts/µL)1'] = d['Segmentados # | µL | 1000 a 8500'].copy()
temp['Mature_Neutrophils_(counts/µL)2'] = d['Segmentados # | µL | 1000 a 9000'].copy()
temp['Mature_Neutrophils_(counts/µL)3'] = d['Segmentados # | µL | 1000 a 9500'].copy()
temp['Mature_Neutrophils_(counts/µL)4'] = d['Segmentados # | µL | 1500 a 10000'].copy()
temp['Mature_Neutrophils_(counts/µL)5'] = d['Segmentados # | µL | 1500 a 8500'].copy()
temp['Mature_Neutrophils_(counts/µL)6'] = d['Segmentados # | µL | 1700 a 8000'].copy()
temp['Mature_Neutrophils_(counts/µL)7'] = d['Segmentados # | µL | 1800 a 8000'].copy()
temp['Mature_Neutrophils_(counts/µL)8'] = d['Segmentados # | µL | 6000 a 26000'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Mature_Neutrophils_(counts/µL)']=pd.to_numeric(temp['Mature_Neutrophils_(counts/µL)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Monocytes_(counts/µL)1'] = d['Monócitos # | µL | 0 a 800'].copy()
temp['Monocytes_(counts/µL)2'] = d['Monócitos # | µL | 100 a 800'].copy()
temp['Monocytes_(counts/µL)3'] = d['Monócitos # | µL | 300 a 900'].copy()
temp['Monocytes_(counts/µL)4'] = d['Monócitos # | µL | 400 a 1800'].copy()
temp['Monocytes_(counts/µL)5'] = d['Monócitos # | µL | 50 a 1100'].copy()
temp['Monocytes_(counts/µL)6'] = d['Monócitos # | µL | 50 a 800'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Monocytes_(counts/µL)']=pd.to_numeric(temp['Monocytes_(counts/µL)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Neutrophils_(counts/µL)1'] = d['Neutrófilos  # | µL | '].copy()
temp['Neutrophils_(counts/µL)2'] = d['Neutrófilos  # | µL | 1000 a 8500'].copy()
temp['Neutrophils_(counts/µL)3'] = d['Neutrófilos  # | µL | 1000 a 9500'].copy()
temp['Neutrophils_(counts/µL)4'] = d['Neutrófilos  # | µL | 1500 a 10000'].copy()
temp['Neutrophils_(counts/µL)5'] = d['Neutrófilos  # | µL | 1500 a 8500'].copy()
temp['Neutrophils_(counts/µL)6'] = d['Neutrófilos  # | µL | 1700 a 8000'].copy()
temp['Neutrophils_(counts/µL)7'] = d['Neutrófilos  # | µL | 1800 a 8000'].copy()
temp['Neutrophils_(counts/µL)8'] = d['Neutrófilos  # | µL | 6000 a 26000'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Neutrophils_(counts/µL)']=pd.to_numeric(temp['Neutrophils_(counts/µL)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['pO2–Arterial_(mmHg)1'] = d['pO2 (gasometria arterial) | mm Hg | 80.0 a 90.0'].copy()
temp['pO2–Arterial_(mmHg)2'] = d['pO2 (gasometria arterial) | NULL | 80.0 a 90.0'].copy()
temp['pO2–Arterial_(mmHg)3'] = d['pO2 (gasometria venosa) | mm Hg | 25.0 a 40.0'].copy()
temp = temp.bfill(axis = 'columns')
data3a['pO2–Arterial_(mmHg)']=pd.to_numeric(temp['pO2–Arterial_(mmHg)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Serum_Albumin_(g/dL)1'] = d['Albumina | g/dL | 3,5 a 5,2'].copy()
temp['Serum_Albumin_(g/dL)2'] = d['Albumina | g/dL | 3.1 a 5.0'].copy()
temp['Serum_Albumin_(g/dL)3'] = d['Albumina | g/dL | 3.20 a 4.50'].copy()
temp['Serum_Albumin_(g/dL)4'] = d['Albumina | g/dL | 3.5 a 5.2'].copy()
temp['Serum_Albumin_(g/dL)5'] = d['Albumina | g/dL | 3.50 a 5.20'].copy()
temp['Serum_Albumin_(g/dL)6'] = d['Albumina | g/dL | 3.80 a 5.40'].copy()
temp['Serum_Albumin_(g/dL)7'] = d['Albumina | g/dL | 4.0 a 4.9'].copy()
temp['Serum_Albumin_(g/dL)8'] = d['Albumina | g/dL | 4.0 a 5.3'].copy()
temp['Serum_Albumin_(g/dL)9'] = d['Albumina | g/dL | 4.2 a 5.1'].copy()
temp['Serum_Albumin_(g/dL)10'] = d['Albumina | g/dL | 4.3 a 5.3'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Serum_Albumin_(g/dL)']=pd.to_numeric(temp['Serum_Albumin_(g/dL)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Serum_Calcium_(mmol/L)1'] = d['Cálcio Iônico mmol/L | mmol/L | 0.95 a 1.50'].copy()
temp['Serum_Calcium_(mmol/L)2'] = d['Cálcio Iônico mmol/L | mmol/L | 1.00 a 1.50'].copy()
temp['Serum_Calcium_(mmol/L)3'] = d['Cálcio Iônico mmol/L | mmol/L | 1.14 a 1.31'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Serum_Calcium_(mmol/L)']=pd.to_numeric(temp['Serum_Calcium_(mmol/L)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Serum_Ferritin_(ng/mL)1'] = d['Ferritina | ng/mL | '].copy()
temp['Serum_Ferritin_(ng/mL)2'] = d['Ferritina | ng/mL | 12.00 a 266.00'].copy()
temp['Serum_Ferritin_(ng/mL)3'] = d['Ferritina | ng/mL | 14.00 a 101.00'].copy()
temp['Serum_Ferritin_(ng/mL)4'] = d['Ferritina | ng/mL | 150.00 a 973.00'].copy()
temp['Serum_Ferritin_(ng/mL)5'] = d['Ferritina | ng/mL | 20.90 a 173.00'].copy()
temp['Serum_Ferritin_(ng/mL)6'] = d['Ferritina | ng/mL | 22.00 a 491.00'].copy()
temp['Serum_Ferritin_(ng/mL)7'] = d['Ferritina | ng/mL | 3.88 a 114.00'].copy()
temp['Serum_Ferritin_(ng/mL)8'] = d['Ferritina | ng/mL | 8.46 a 580.00'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Serum_Ferritin_(ng/mL)']=pd.to_numeric(temp['Serum_Ferritin_(ng/mL)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Serum_Magnesium_(mEq/L)1'] = d['Magnésio | mEq/L | 1.2 a 1.8'].copy()
temp['Serum_Magnesium_(mEq/L)2'] = d['Magnésio | mEq/L | 1.3 a 2.1'].copy()
temp['Serum_Magnesium_(mEq/L)3'] = d['Magnésio | mEq/L | 1.4 a 1.7'].copy()
temp['Serum_Magnesium_(mEq/L)4'] = d['Magnésio | mEq/L | 1.4 a 1.9'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Serum_Magnesium_(mEq/L)']=pd.to_numeric(temp['Serum_Magnesium_(mEq/L)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Serum_Phosphorus_(mg/dL)1'] = d['Fósforo | mg/dL | 2.5 a 4.5'].copy()
temp['Serum_Phosphorus_(mg/dL)2'] = d['Fósforo | mg/dL | 2.5 a 4.8'].copy()
temp['Serum_Phosphorus_(mg/dL)3'] = d['Fósforo | mg/dL | 2.7 a 4.9'].copy()
temp['Serum_Phosphorus_(mg/dL)4'] = d['Fósforo | mg/dL | 2.8 a 4.8'].copy()
temp['Serum_Phosphorus_(mg/dL)5'] = d['Fósforo | mg/dL | 2.9 a 4.8'].copy()
temp['Serum_Phosphorus_(mg/dL)6'] = d['Fósforo | mg/dL | 2.9 a 5.1'].copy()
temp['Serum_Phosphorus_(mg/dL)7'] = d['Fósforo | mg/dL | 3.0 a 5.4'].copy()
temp['Serum_Phosphorus_(mg/dL)8'] = d['Fósforo | mg/dL | 3.1 a 5.3'].copy()
temp['Serum_Phosphorus_(mg/dL)9'] = d['Fósforo | mg/dL | 3.1 a 5.5'].copy()
temp['Serum_Phosphorus_(mg/dL)10'] = d['Fósforo | mg/dL | 3.1 a 6.0'].copy()
temp['Serum_Phosphorus_(mg/dL)11'] = d['Fósforo | mg/dL | 3.2 a 5.5'].copy()
temp['Serum_Phosphorus_(mg/dL)12'] = d['Fósforo | mg/dL | 3.2 a 5.7'].copy()
temp['Serum_Phosphorus_(mg/dL)13'] = d['Fósforo | mg/dL | 3.3 a 5.3'].copy()
temp['Serum_Phosphorus_(mg/dL)14'] = d['Fósforo | mg/dL | 3.3 a 5.6'].copy()
temp['Serum_Phosphorus_(mg/dL)15'] = d['Fósforo | mg/dL | 3.4 a 6.0'].copy()
temp['Serum_Phosphorus_(mg/dL)16'] = d['Fósforo | mg/dL | 3.5 a 5.8'].copy()
temp['Serum_Phosphorus_(mg/dL)17'] = d['Fósforo | mg/dL | 3.5 a 6.6'].copy()
temp['Serum_Phosphorus_(mg/dL)18'] = d['Fósforo | mg/dL | 3.7 a 6.5'].copy()
temp['Serum_Phosphorus_(mg/dL)19'] = d['Fósforo | mg/dL | 3.9 a 6.9'].copy()
temp['Serum_Phosphorus_(mg/dL)20'] = d['Fósforo | mg/dL | 4.0 a 5.6'].copy()
temp['Serum_Phosphorus_(mg/dL)21'] = d['Fósforo | mg/dL | 4.1 a 6.4'].copy()
temp['Serum_Phosphorus_(mg/dL)22'] = d['Fósforo | mg/dL | 4.6 a 7.9'].copy()
temp['Serum_Phosphorus_(mg/dL)23'] = d['Fósforo | mg/dL | 5.3 a 9.8'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Serum_Phosphorus_(mg/dL)']=pd.to_numeric(temp['Serum_Phosphorus_(mg/dL)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Total_Bilirubin_(mg/dL)1'] = d['Bilirrubina Total | mg/dL | 0.0 a 10.8'].copy()
temp['Total_Bilirubin_(mg/dL)2'] = d['Bilirrubina Total | mg/dL | 0.0 a 13.4'].copy()
temp['Total_Bilirubin_(mg/dL)3'] = d['Bilirrubina Total | mg/dL | 0.0 a 6.4'].copy()
temp['Total_Bilirubin_(mg/dL)4'] = d['Bilirrubina Total | mg/dL | 0.1 a 1.2'].copy()
temp = temp.bfill(axis = 'columns')
data3a['Total_Bilirubin_(mg/dL)']=pd.to_numeric(temp['Total_Bilirubin_(mg/dL)1'].copy(), errors='coerce')

data3a['Platelets_(counts(×10^3)/µL)']=pd.to_numeric(d['Plaquetas | x10^3/uL | '].copy(), errors='coerce')

temp = pd.DataFrame()
temp['CoV-21'] = d['Resultado COVID-19: | NULL | '].copy()
temp['CoV-22'] = d['Resultado COVID-19: | NULL | Não detectado'].copy()
temp = temp.replace(['Não detectado'], 0)
temp = temp.replace(['Detectado'], 1)
temp = temp.bfill(axis = 'columns')
data3a['CoV-2']=pd.to_numeric(temp['CoV-21'].copy(), errors='coerce')

data3a.dropna(subset = ['CoV-2'], inplace=True)
t=(len(data3a.columns)*2)/3
data3a.dropna(thresh=t, axis=0, inplace=True)
data3a.drop(labels=None, axis=1, columns=['ID','Date_Collected'], level=None, inplace=True)

cols = data3a.columns.tolist()
cols.remove('CoV-2')
for c in cols:
  data3a[c]=((data3a[c])-(data3a[c].mean()))/(data3a[c].std())

data3a.reset_index(inplace = True, drop=True)
data3a.to_csv('datasets/Dataset-3a.csv', index=False)

d = pd.read_csv('rawdata/3.csv', dtype='object')
data3b = pd.DataFrame()

temp = pd.DataFrame()
temp['Hematocrit1'] = d['Hematócrito | % | '].copy()
temp['Hematocrit2'] = d['Hematócrito | % | 29.0 a 41.0'].copy()
temp['Hematocrit3'] = d['Hematócrito | % | 31.0 a 55.0'].copy()
temp['Hematocrit4'] = d['Hematócrito | % | 33.0 a 39.0'].copy()
temp['Hematocrit5'] = d['Hematócrito | % | 33.0 a 43.0'].copy()
temp['Hematocrit6'] = d['Hematócrito | % | 35,0 a 45,0'].copy()
temp['Hematocrit7'] = d['Hematócrito | % | 35.0 a 45.0'].copy()
temp['Hematocrit8'] = d['Hematócrito | % | 36.0 a 43.0'].copy()
temp['Hematocrit9'] = d['Hematócrito | % | 37.0 a 47.0'].copy()
temp['Hematocrit10'] = d['Hematócrito | % | 39,0 a 50,0'].copy()
temp['Hematocrit11'] = d['Hematócrito | % | 39.0 a 50.0'].copy()
temp['Hematocrit12'] = d['Hematócrito | % | 39.0 a 63.0'].copy()
temp['Hematocrit13'] = d['Hematócrito | % | 42.0 a 60.0'].copy()
temp['Hematocrit14'] = d['Hematócrito | % | 45.0 a 60.0'].copy()
temp = temp.bfill(axis = 'columns')
data3b['Hematocrit']=pd.to_numeric(temp['Hematocrit1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Hemoglobin1'] = d['Hemoglobina | g/dL | 10.0 a 18.0'].copy()
temp['Hemoglobin2'] = d['Hemoglobina | g/dL | 10.5 a 13.5'].copy()
temp['Hemoglobin3'] = d['Hemoglobina | g/dL | 11.0 a 14.5'].copy()
temp['Hemoglobin4'] = d['Hemoglobina | g/dL | 11.5 a 14.5'].copy()
temp['Hemoglobin5'] = d['Hemoglobina | g/dL | 12,0 a 15,5'].copy()
temp['Hemoglobin6'] = d['Hemoglobina | g/dL | 12.0 a 14.8'].copy()
temp['Hemoglobin7'] = d['Hemoglobina | g/dL | 12.0 a 15.5'].copy()
temp['Hemoglobin8'] = d['Hemoglobina | g/dL | 12.5 a 20.5'].copy()
temp['Hemoglobin9'] = d['Hemoglobina | g/dL | 13,5 a 17,5'].copy()
temp['Hemoglobin10'] = d['Hemoglobina | g/dL | 13.0 a 16.0'].copy()
temp['Hemoglobin11'] = d['Hemoglobina | g/dL | 13.5 a 17.5'].copy()
temp['Hemoglobin12'] = d['Hemoglobina | g/dL | 13.5 a 21.5'].copy()
temp['Hemoglobin13'] = d['Hemoglobina | g/dL | 14.5 a 22.5'].copy()
temp['Hemoglobin14'] = d['Hemoglobina | g/dL | 9.0 a 14.0'].copy()
temp['Hemoglobin15'] = d['Hemoglobina | NULL | '].copy()
temp = temp.bfill(axis = 'columns')
data3b['Hemoglobin']=pd.to_numeric(temp['Hemoglobin1'].copy(), errors='coerce')

data3b['Platelets']=pd.to_numeric(d['Plaquetas | x10^3/uL | '].copy(), errors='coerce')

temp = pd.DataFrame()
temp['MPV1'] = d['Volume Médio Plaquetário | fL | 6,5 a 15,0'].copy()
temp['MPV2'] = d['Volume Médio Plaquetário | fL | 6.5 a 15.0'].copy()
temp = temp.bfill(axis = 'columns')
data3b['MPV']=pd.to_numeric(temp['MPV1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Red_blood_Cells1'] = d['Hemácias | x10^6/uL | 3,90 a 5,00'].copy()
temp['Red_blood_Cells2'] = d['Hemácias | x10^6/uL | 3.00 a 5.40'].copy()
temp['Red_blood_Cells3'] = d['Hemácias | x10^6/uL | 3.10 a 4.50'].copy()
temp['Red_blood_Cells4'] = d['Hemácias | x10^6/uL | 3.60 a 6.20'].copy()
temp['Red_blood_Cells5'] = d['Hemácias | x10^6/uL | 3.70 a 5.30'].copy()
temp['Red_blood_Cells6'] = d['Hemácias | x10^6/uL | 3.90 a 5.00'].copy()
temp['Red_blood_Cells7'] = d['Hemácias | x10^6/uL | 3.90 a 5.30'].copy()
temp['Red_blood_Cells8'] = d['Hemácias | x10^6/uL | 3.90 a 6.20'].copy()
temp['Red_blood_Cells9'] = d['Hemácias | x10^6/uL | 3.90 a 6.30'].copy()
temp['Red_blood_Cells10'] = d['Hemácias | x10^6/uL | 3.90 a 6.60'].copy()
temp['Red_blood_Cells11'] = d['Hemácias | x10^6/uL | 4,30 a 5,70'].copy()
temp['Red_blood_Cells12'] = d['Hemácias | x10^6/uL | 4.00 a 5.20'].copy()
temp['Red_blood_Cells13'] = d['Hemácias | x10^6/uL | 4.10 a 5.10'].copy()
temp['Red_blood_Cells14'] = d['Hemácias | x10^6/uL | 4.30 a 5.30'].copy()
temp['Red_blood_Cells15'] = d['Hemácias | x10^6/uL | 4.30 a 5.70'].copy()
temp = temp.bfill(axis = 'columns')
data3b['Erythrocytes']=pd.to_numeric(temp['Red_blood_Cells1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Lymphocytes1'] = d['Linfócitos # | µL | '].copy()
temp['Lymphocytes2'] = d['Linfócitos # | µL | 1200 a 5200'].copy()
temp['Lymphocytes3'] = d['Linfócitos # | µL | 1500 a 6500'].copy()
temp['Lymphocytes4'] = d['Linfócitos # | µL | 1500 a 8500'].copy()
temp['Lymphocytes5'] = d['Linfócitos # | µL | 2000 a 11000'].copy()
temp['Lymphocytes6'] = d['Linfócitos # | µL | 2000 a 17000'].copy()
temp['Lymphocytes7'] = d['Linfócitos # | µL | 2500 a 16500'].copy()
temp['Lymphocytes8'] = d['Linfócitos # | µL | 300 a 9500'].copy()
temp['Lymphocytes9'] = d['Linfócitos # | µL | 4000 a 13500'].copy()
temp['Lymphocytes10'] = d['Linfócitos # | µL | 900 a 2900'].copy()
temp = temp.bfill(axis = 'columns')
data3b['Lymphocytes']=pd.to_numeric(temp['Lymphocytes1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Mean_corpuscular_hemoglobin_concentration_(MCHC)1'] = d['CHCM | g/dL | '].copy()
temp['Mean_corpuscular_hemoglobin_concentration_(MCHC)2'] = d['CHCM | g/dL | 28.0 a 38.0'].copy()
temp['Mean_corpuscular_hemoglobin_concentration_(MCHC)3'] = d['CHCM | g/dL | 29.0 a 36.0'].copy()
temp['Mean_corpuscular_hemoglobin_concentration_(MCHC)4'] = d['CHCM | g/dL | 29.0 a 37.0'].copy()
temp['Mean_corpuscular_hemoglobin_concentration_(MCHC)5'] = d['CHCM | g/dL | 30.0 a 36.0'].copy()
temp['Mean_corpuscular_hemoglobin_concentration_(MCHC)6'] = d['CHCM | g/dL | 31,0 a 36,0'].copy()
temp['Mean_corpuscular_hemoglobin_concentration_(MCHC)7'] = d['CHCM | g/dL | 31.0 a 36.0'].copy()
temp['Mean_corpuscular_hemoglobin_concentration_(MCHC)8'] = d['CHCM | g/dL | 32.0 a 36.0'].copy()
temp = temp.bfill(axis = 'columns')
data3b['MCHC']=pd.to_numeric(temp['Mean_corpuscular_hemoglobin_concentration_(MCHC)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Leukocytes1'] = d['Leucócitos # | µL | '].copy()
temp['Leukocytes2'] = d['Leucócitos # | µL | 3500 a 10500'].copy()
temp['Leukocytes3'] = d['Leucócitos # | µL | 4500 a 13000'].copy()
temp['Leukocytes4'] = d['Leucócitos # | µL | 5000 a 14500'].copy()
temp['Leukocytes5'] = d['Leucócitos # | µL | 5000 a 19500'].copy()
temp['Leukocytes6'] = d['Leucócitos # | µL | 5000 a 20000'].copy()
temp['Leukocytes7'] = d['Leucócitos # | µL | 6000 a 17000'].copy()
temp['Leukocytes8'] = d['Leucócitos # | µL | 6000 a 17500'].copy()
temp['Leukocytes9'] = d['Leucócitos # | µL | 9000 a 30000'].copy()
temp['Leukocytes10'] = d['Leucócitos # | µL | 9400 a 34000'].copy()
temp = temp.bfill(axis = 'columns')
data3b['Leukocytes']=pd.to_numeric(temp['Leukocytes1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Basophils1'] = d['Basófilos # | µL | 0 a 100'].copy()
temp['Basophils2'] = d['Basófilos # | µL | 0 a 200'].copy()
temp['Basophils3'] = d['Basófilos # | µL | 0 a 600'].copy()
temp = temp.bfill(axis = 'columns')
data3b['Basophils']=pd.to_numeric(temp['Basophils1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Mean_corpuscular_hemoglobin_(MCH)1'] = d['HCM | pg | '].copy()
temp['Mean_corpuscular_hemoglobin_(MCH)2'] = d['HCM | pg | 23.0 a 31.0'].copy()
temp['Mean_corpuscular_hemoglobin_(MCH)3'] = d['HCM | pg | 24.0 a 32.0'].copy()
temp['Mean_corpuscular_hemoglobin_(MCH)4'] = d['HCM | pg | 25.0 a 33.0'].copy()
temp['Mean_corpuscular_hemoglobin_(MCH)5'] = d['HCM | pg | 25.0 a 35.0'].copy()
temp['Mean_corpuscular_hemoglobin_(MCH)6'] = d['HCM | pg | 25.0 a 36.0'].copy()
temp['Mean_corpuscular_hemoglobin_(MCH)7'] = d['HCM | pg | 26,0 a 34,0'].copy()
temp['Mean_corpuscular_hemoglobin_(MCH)8'] = d['HCM | pg | 26.0 a 34.0'].copy()
temp['Mean_corpuscular_hemoglobin_(MCH)9'] = d['HCM | pg | 28.0 a 40.0'].copy()
temp['Mean_corpuscular_hemoglobin_(MCH)10'] = d['HCM | pg | 29.0 a 37.0'].copy()
temp['Mean_corpuscular_hemoglobin_(MCH)11'] = d['HCM | pg | 31.0 a 37.0'].copy()
temp = temp.bfill(axis = 'columns')
data3b['MCH']=pd.to_numeric(temp['Mean_corpuscular_hemoglobin_(MCH)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Eosinophils1'] = d['Eosinófilos  # | µL | '].copy()
temp['Eosinophils2'] = d['Eosinófilos  # | µL | 0 a 500'].copy()
temp['Eosinophils3'] = d['Eosinófilos  # | µL | 0 a 650'].copy()
temp['Eosinophils4'] = d['Eosinófilos  # | µL | 20 a 850'].copy()
temp['Eosinophils5'] = d['Eosinófilos  # | µL | 50 a 500'].copy()
temp = temp.bfill(axis = 'columns')
data3b['Eosinophils']=pd.to_numeric(temp['Eosinophils1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Mean_corpuscular_volume_(MCV)1'] = d['VCM | fL | 70.0 a 86.0'].copy()
temp['Mean_corpuscular_volume_(MCV)2'] = d['VCM | fL | 74.0 a 89.0'].copy()
temp['Mean_corpuscular_volume_(MCV)3'] = d['VCM | fL | 77.0 a 115.0'].copy()
temp['Mean_corpuscular_volume_(MCV)4'] = d['VCM | fL | 77.0 a 91.0'].copy()
temp['Mean_corpuscular_volume_(MCV)5'] = d['VCM | fL | 80.0 a 92.0'].copy()
temp['Mean_corpuscular_volume_(MCV)6'] = d['VCM | fL | 81,0 a 95,0'].copy()
temp['Mean_corpuscular_volume_(MCV)7'] = d['VCM | fL | 81.0 a 92.0'].copy()
temp['Mean_corpuscular_volume_(MCV)8'] = d['VCM | fL | 81.0 a 95.0'].copy()
temp['Mean_corpuscular_volume_(MCV)9'] = d['VCM | fL | 82,0 a 98,0'].copy()
temp['Mean_corpuscular_volume_(MCV)10'] = d['VCM | fL | 82.0 a 98.0'].copy()
temp['Mean_corpuscular_volume_(MCV)11'] = d['VCM | fL | 85.0 a 123.0'].copy()
temp['Mean_corpuscular_volume_(MCV)12'] = d['VCM | fL | 86.0 a 124.0'].copy()
temp['Mean_corpuscular_volume_(MCV)13'] = d['VCM | fL | 88.0 a 126.0'].copy()
temp['Mean_corpuscular_volume_(MCV)14'] = d['VCM | fL | 95.0 a 115.0'].copy()
temp['Mean_corpuscular_volume_(MCV)15'] = d['VCM | NULL | '].copy()
temp = temp.bfill(axis = 'columns')
data3b['MCV']=pd.to_numeric(temp['Mean_corpuscular_volume_(MCV)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Monocytes1'] = d['Monócitos # | µL | 0 a 800'].copy()
temp['Monocytes2'] = d['Monócitos # | µL | 100 a 800'].copy()
temp['Monocytes3'] = d['Monócitos # | µL | 300 a 900'].copy()
temp['Monocytes4'] = d['Monócitos # | µL | 400 a 1800'].copy()
temp['Monocytes5'] = d['Monócitos # | µL | 50 a 1100'].copy()
temp['Monocytes6'] = d['Monócitos # | µL | 50 a 800'].copy()
temp = temp.bfill(axis = 'columns')
data3b['Monocytes']=pd.to_numeric(temp['Monocytes1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['Red_blood_cell_distribution_width_(RDW)1'] = d['RDW | % | '].copy()
temp['Red_blood_cell_distribution_width_(RDW)2'] = d['RDW | % | 11,5 a 16,5'].copy()
temp['Red_blood_cell_distribution_width_(RDW)3'] = d['RDW | % | 11.5 a 16.5'].copy()
temp['Red_blood_cell_distribution_width_(RDW)4'] = d['RDW | % | 12.0 a 14.5'].copy()
temp['Red_blood_cell_distribution_width_(RDW)5'] = d['RDW | NULL | '].copy()
temp = temp.bfill(axis = 'columns')
data3b['RDW']=pd.to_numeric(temp['Red_blood_cell_distribution_width_(RDW)1'].copy(), errors='coerce')

temp = pd.DataFrame()
temp['CoV-21'] = d['Resultado COVID-19: | NULL | '].copy()
temp['CoV-22'] = d['Resultado COVID-19: | NULL | Não detectado'].copy()
temp = temp.replace(['Não detectado'], 0)
temp = temp.replace(['Detectado'], 1)
temp = temp.bfill(axis = 'columns')
data3b['CoV-2']=pd.to_numeric(temp['CoV-21'].copy(), errors='coerce')

data3b.dropna(subset = ['CoV-2'], inplace=True)
t=(len(data3b.columns)*2)/3
data3b.dropna(thresh=t, axis=0, inplace=True)

cols = data3b.columns.tolist()
cols.remove('CoV-2')
for c in cols:
  data3b[c]=((data3b[c])-(data3b[c].mean()))/(data3b[c].std())

data3b.reset_index(inplace = True, drop=True)
data3b.to_csv('datasets/Dataset-3b.csv', index=False)
