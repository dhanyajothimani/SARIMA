import pandas as pd
import pandas_datareader.data as web
import datetime
import os

os.chdir("/home/dhanya/Desktop/HTM_TMX/sp500")

pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)
pd.set_option('precision', 4)
# load data
#symbols = ['ARX.TO',	'ACO-X.TO',	'ATA.TO',	'ARE.TO',	'AFN.TO',	'AEM.TO',	'AC.TO',	'ASR.TO',	'AGI.TO',	'AD.TO',	'AQN.TO',	'ATD-B.TO',	'AP-UN.TO',	'ALA.TO',	'AIF.TO',	'APHA.TO',	'ATZ.TO',	'AX-UN.TO',	'ACB.TO',	'BTO.TO',	'BCE.TO',	'DOO.TO',	'BAD.TO',	'BLDP.TO',	'BMO.TO',	'BNS.TO',	'ABX.TO',	'BHC.TO',	'BTE.TO',	'BB.TO',	'BEI-UN.TO',	'BBD-B.TO',	'BLX.TO',	'BYD-UN.TO',	'BAM-A.TO',	'BBU-UN.TO',	'BIP-UN.TO',	'BPY-UN.TO',	'BEP-UN.TO',	'CAE.TO',	'CCL-B.TO',	'GIB-A.TO',	'CIX.TO',	'CCO.TO',	'GOOS.TO',	'CAR-UN.TO',	'CM.TO',	'CNR.TO',	'CNQ.TO',	'CP.TO',	'CTC-A.TO',	'CU.TO',	'CWB.TO',	'CFP.TO',	'WEED.TO',	'CPX.TO',	'CJT.TO',	'CAS.TO',	'CLS.TO',	'CVE.TO',	'CG.TO',	'CSH-UN.TO',	'CHE-UN.TO',	'CHP-UN.TO',	'CHR.TO',	'CGX.TO',	'CCA.TO',	'CIGI.TO',	'CUF-UN.TO',	'CSU.TO',	'BCB.TO',	'CPG.TO',	'CRR-UN.TO',	'CRON.TO',	'DSG.TO',	'DGC.TO',	'DOL.TO',	'DRG-UN.TO',	'DIR-UN.TO',	'D-UN.TO',	'ECN.TO',	'ELD.TO',	'EFN.TO',	'EMA.TO',	'EMP-A.TO',	'ENB.TO',	'ECA.TO',	'EDV.TO',	'EFX.TO',	'ERF.TO',	'ENGH.TO',	'EQB.TO',	'ERO.TO',	'EIF.TO',	'EXE.TO',	'FFH.TO',	'FTT.TO',	'FCR.TO',	'FR.TO',	'FM.TO',	'FSV.TO',	'FTS.TO',	'FNV.TO',	'FRU.TO',	'FEC.TO',	'MIC.TO',	'WN.TO',	'GEI.TO',	'GIL.TO',	'GTE.TO',	'GRT-UN.TO',	'GC.TO',	'GWO.TO',	'HR-UN.TO',	'HEXO.TO',	'HCG.TO',	'HBM.TO',	'HBC.TO',	'HSE.TO',	'H.TO',	'IMG.TO',	'IGM.TO',	'IMO.TO',	'INE.TO',	'IFC.TO',	'IPL.TO',	'IIP-UN.TO',	'IFP.TO',	'ITP.TO',	'IVN.TO',	'KEY.TO',	'KMP-UN.TO',	'KXS.TO',	'K.TO',	'KL.TO',	'GUD.TO',	'LIF.TO',	'LB.TO',	'LNR.TO',	'L.TO',	'LUN.TO',	'MAG.TO',	'MEG.TO',	'MTY.TO',	'MG.TO',	'MFC.TO',	'MFI.TO',	'MRE.TO',	'MX.TO',	'MRU.TO',	'MSI.TO',	'MTL.TO',	'NFI.TO',	'NA.TO',	'OSB.TO',	'NWH-UN.TO',	'NPI.TO',	'NVU-UN.TO',	'NG.TO',	'NTR.TO',	'ONEX.TO',	'OGC.TO',	'OTEX.TO',	'OR.TO',	'PAAS.TO',	'PXT.TO',	'PKI.TO',	'PSI.TO',	'PPL.TO',	'POW.TO',	'PWF.TO',	'PSK.TO',	'PBH.TO',	'PVG.TO',	'QBR-B.TO',	'QSR.TO',	'RCH.TO',	'REI-UN.TO',	'RBA.TO',	'RCI-B.TO',	'RY.TO',	'RUS.TO',	'SNC.TO',	'SSRM.TO',	'SSL.TO',	'SAP.TO',	'SEA.TO',	'SES.TO',	'SMF.TO',	'VII.TO',	'SJR-B.TO',	'SCL.TO',	'SHOP.TO',	'SIA.TO',	'SVM.TO',	'ZZZ.TO',	'SRU-UN.TO',	'TOY.TO',	'STN.TO',	'SJ.TO',	'SMU-UN.TO',	'SLF.TO',	'SU.TO',	'SPB.TO',	'TRP.TO',	'T.TO',	'TFII.TO',	'X.TO',	'TECK-B.TO',	'NWC.TO',	'TSGI.TO',	'TRI.TO',	'TXG.TO',	'TIH.TO',	'TD.TO',	'TOU.TO',	'TA.TO',	'RNW.TO',	'TCL-A.TO',	'TCN.TO',	'TRQ.TO',	'VET.TO',	'WSP.TO',	'WCN.TO',	'WDO.TO',	'WFT.TO',	'WJA.TO',	'WTE.TO',	'WPM.TO',	'WCP.TO',	'WPK.TO',	'YRI.TO',	'IAG.TO',]

symbols = ["BRK-B","BF-B"]

start = datetime.datetime(2009, 7, 1)
end = datetime.datetime(2019, 7, 31)

quotes = web.DataReader(symbols, 'yahoo', start, end)

n = len(symbols)


for i in range(0,n):
	z = quotes['Close'][symbols[i]]
	print(z.head())
	z = z.dropna()
	print(z.head())
	z.to_csv(symbols[i]+".csv")



