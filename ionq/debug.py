from ib_insync import IB, Stock

ib = IB()
ib.connect('127.0.0.1', 7496, clientId=999)
stock = Stock('IONQ', 'SMART', 'USD')
cds = ib.reqContractDetails(stock)[0]
print('Contract Details:', cds)
params = ib.reqSecDefOptParams(cds.contract.symbol, '', cds.contract.secType, cds.contract.conId)
print('SecDefOptParams:', params)
ib.disconnect()
