import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from tqdm import tqdm

class YearlyData:
	def __init__(self, train_csv, weather_csv, test_csv=None):
		super(YearlyData, self).__init__()
		self.station1 = (41.995, -87.933)
		self.station2 = (41.786, -87.752)
		self.df = pd.read_csv(train_csv)
		self.df_test = None if (test_csv == None) else pd.read_csv(test_csv)
		self.df_weather = pd.read_csv(weather_csv)


	@staticmethod
	def distance(lat1, lng1, lat2, lng2):
		R = 6373.0

		lat1 = radians(lat1)
		lng1 = radians(lng1)
		lat2 = radians(lat2)
		lng2 = radians(lng2)

		dlng = lng2 - lng1
		dlat = lat2 - lat1

		a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlng / 2)**2
		c = 2 * atan2(sqrt(a), sqrt(1 - a))

		return R * c


	def process(self, df_piece=None):
		if (df_piece is None):
			df_piece = self.df

		self.df_weather['PrecipTotal'] = self.df_weather['PrecipTotal'].replace('  T', '0.00')
		self.df_weather['PrecipTotal'] = self.df_weather['PrecipTotal'].replace('M', '0.00')
		self.df_weather['WetBulb'] = self.df_weather['WetBulb'].replace('M', '0')
		self.df_weather['StnPressure'] = self.df_weather['StnPressure'].replace('M', '0')

		tmax = []
		tmin = []
		tavg = []
		dewpoint = []
		wetbulb = []
		heat = []
		cool = []
		stnpressure = []
		sealevel = []
		resultspeed = []
		resultdir = []
		avgspeed = []
		sunrise = []
		sunset = []
		preciptotal = []
		day =[]
		month =[]
		year = []
		depart = []

		with tqdm(total=len(list(df_piece.iterrows()))) as pbar:
			for index, row in tqdm(df_piece.iterrows()):
			# for index, row in df_piece.iterrows():
				pbar.update(1)
				dis_to_s1 = self.distance(row.loc['Latitude'], row.loc['Longitude'], self.station1[0], self.station1[1])
				dis_to_s2 = self.distance(row.loc['Latitude'], row.loc['Longitude'], self.station2[0], self.station2[1])
			  
				df_weather_piece = self.df_weather.loc[(self.df_weather['Date'] == row.loc['Date'])]
				df_s_1 = df_weather_piece.loc[self.df_weather['Station'] == 1]
				df_s_2 = df_weather_piece.loc[self.df_weather['Station'] == 2]

				df_s_mandatory = df_s_1 if dis_to_s1 < dis_to_s2 else df_s_2

				date = row.loc['Date'].split('-')

				year.append(int(date[0]))
				month.append(int(date[1]))
				day.append(int(date[2]))
			 	
				tmax.append(df_s_mandatory['Tmax'].values[0])
				tmin.append(df_s_mandatory['Tmin'].values[0])
				tavg.append(float(	df_s_mandatory['Tavg'].values[0]))
				dewpoint.append(df_s_mandatory['DewPoint'].values[0])
				wetbulb.append(int(df_s_mandatory['WetBulb'].values[0]))
				heat.append(float(df_s_mandatory['Heat'].values[0]))
				cool.append(float(df_s_mandatory['Cool'].values[0]))
				stnpressure.append(float(df_s_mandatory['StnPressure'].values[0]))
				sealevel.append(float(df_s_mandatory['SeaLevel'].values[0]))
				resultspeed.append(df_s_mandatory['ResultSpeed'].values[0])
				resultdir.append(df_s_mandatory['ResultDir'].values[0])
				avgspeed.append(float(df_s_mandatory['AvgSpeed'].values[0]))
				sunrise.append(int(df_s_1['Sunrise'].values[0]))
				sunset.append(int(df_s_1['Sunset'].values[0]))
				depart.append(int(df_s_1['Depart'].values[0]))

				preciptotal_1 = float(df_s_1['PrecipTotal'].values[0])
				preciptotal_2 = float(df_s_2['PrecipTotal'].values[0])

				preciptotal.append(preciptotal_1 if preciptotal_1 > preciptotal_2 else preciptotal_2)


		species = pd.get_dummies(df_piece['Species'])
		traps = pd.get_dummies(df_piece['Trap'])
		addresses = pd.get_dummies(df_piece['Address'])

		df_piece = pd.concat([df_piece, species], axis=1, join_axes=[df_piece.index])
		df_piece = pd.concat([df_piece, traps], axis=1, join_axes=[df_piece.index])
		df_piece = pd.concat([df_piece, addresses], axis=1, join_axes=[df_piece.index])
		
		if not (self.df_test is None):
			test_species = pd.get_dummies(self.df_test['Species'])
			test_traps = pd.get_dummies(self.df_test['Trap'])
			test_addresses = pd.get_dummies(self.df_test['Address'])

			self.df_test = self.df_test.drop(['Id'], axis=1)		
			self.df_test = pd.concat([self.df_test, test_species], axis=1, join_axes=[self.df_test.index])
			self.df_test = pd.concat([self.df_test, test_traps], axis=1, join_axes=[self.df_test.index])
			self.df_test = pd.concat([self.df_test, test_addresses], axis=1, join_axes=[self.df_test.index])

			for index, column in enumerate(self.df_test.columns):
				if not (column in df_piece.columns):
					df_piece.insert(loc=index+2, column=column, value=np.zeros(len(df_piece)))
		
		df_piece = df_piece.drop(['Date', 'Street', 'AddressNumberAndStreet', 'Species', 'Trap', 'Address'], axis=1)
		
		df_piece.insert(loc=len(df_piece.columns), column='Tmax', value=tmax)
		df_piece.insert(loc=len(df_piece.columns), column='Tmin', value=tmin)
		df_piece.insert(loc=len(df_piece.columns), column='Tavg', value=tavg)
		df_piece.insert(loc=len(df_piece.columns), column='DewPoint', value=dewpoint)
		df_piece.insert(loc=len(df_piece.columns), column='WetBulb', value=wetbulb)
		df_piece.insert(loc=len(df_piece.columns), column='Heat', value=heat)
		df_piece.insert(loc=len(df_piece.columns), column='Cool', value=cool)
		df_piece.insert(loc=len(df_piece.columns), column='StnPressure', value=stnpressure)
		df_piece.insert(loc=len(df_piece.columns), column='SeaLevel', value=sealevel)
		df_piece.insert(loc=len(df_piece.columns), column='ResultSpeed', value=resultspeed)
		df_piece.insert(loc=len(df_piece.columns), column='ResultDir', value=resultdir)
		df_piece.insert(loc=len(df_piece.columns), column='AvgSpeed', value=avgspeed)
		df_piece.insert(loc=len(df_piece.columns), column='Sunrise', value=sunrise)
		df_piece.insert(loc=len(df_piece.columns), column='Sunset', value=sunset)
		df_piece.insert(loc=len(df_piece.columns), column='PrecipTotal', value=preciptotal)
		df_piece.insert(loc=len(df_piece.columns), column='Depart', value=depart)
		df_piece.insert(loc=len(df_piece.columns), column='Day', value=day)
		df_piece.insert(loc=len(df_piece.columns), column='Month', value=month)
		df_piece.insert(loc=len(df_piece.columns), column='Year', value=year)

		return df_piece

# def process_full(self):
	# 	df_piece = self.df

	# 	self.df_weather['PrecipTotal'] = self.df_weather['PrecipTotal'].replace('  T', '0.00')
	# 	self.df_weather['PrecipTotal'] = self.df_weather['PrecipTotal'].replace('M', '0.00')
	# 	self.df_weather['WetBulb'] = self.df_weather['WetBulb'].replace('M', '0')
	# 	self.df_weather['StnPressure'] = self.df_weather['StnPressure'].replace('M', '0')		

	# 	tmaxS1 = []
	# 	tmaxS2 = []
	# 	tminS1 = []
	# 	tminS2 = []
	# 	tavgS1 = []
	# 	tavgS2 = []
	# 	dewpointS1 = []
	# 	dewpointS2 = []
	# 	wetbulbS1 = []
	# 	wetbulbS2 = []
	# 	heatS1 = []
	# 	heatS2 = []
	# 	coolS1 = []
	# 	coolS2 = []
	# 	stnpressureS1 = []
	# 	stnpressureS2 = []
	# 	sealevelS1 = []
	# 	sealevelS2 = []
	# 	resultspeedS1 = []
	# 	resultspeedS2 = []
	# 	resultdirS1 = []
	# 	resultdirS2 = []
	# 	avgspeedS1 = []
	# 	avgspeedS2 = []
	# 	preciptotalS1 = []
	# 	preciptotalS2 = []
	# 	sunrise = []
	# 	sunset = []
	# 	day =[]
	# 	year = []

	# 	with tqdm(total=len(list(df_piece.iterrows()))) as pbar:
	# 		for index, row in tqdm(df_piece.iterrows()):
	# 			pbar.update(1)
	# 			dis_to_s1 = self.distance(row.loc['Latitude'], row.loc['Longitude'], self.station1[0], self.station1[1])
	# 			dis_to_s2 = self.distance(row.loc['Latitude'], row.loc['Longitude'], self.station2[0], self.station2[1])
			  
	# 			df_weather_piece = self.df_weather.loc[(self.df_weather['Date'] == row.loc['Date'])]
	# 			df_s_1 = df_weather_piece.loc[self.df_weather['Station'] == 1]
	# 			df_s_2 = df_weather_piece.loc[self.df_weather['Station'] == 2]

	# 			date = row.loc['Date'].split('-')

	# 			day.append(self.doy(int(date[2]), int(date[1]), int(date[0])))
	# 			year.append(int(date[2]))

	# 			tmaxS1.append(df_s_1['Tmax'].values[0])
	# 			tmaxS2.append(df_s_2['Tmax'].values[0])
	# 			tminS1.append(df_s_1['Tmin'].values[0])
	# 			tminS2.append(df_s_2['Tmin'].values[0])
	# 			tavgS1.append(float(df_s_1['Tavg'].values[0]))
	# 			tavgS2.append(float(df_s_2['Tavg'].values[0]))
	# 			dewpointS1.append(df_s_1['DewPoint'].values[0])
	# 			dewpointS2.append(df_s_2['DewPoint'].values[0])
	# 			wetbulbS1.append(int(df_s_1['WetBulb'].values[0]))
	# 			wetbulbS2.append(int(df_s_2['WetBulb'].values[0]))
	# 			heatS1.append(float(df_s_1['Heat'].values[0]))
	# 			heatS2.append(float(df_s_2['Heat'].values[0]))
	# 			coolS1.append(float(df_s_1['Cool'].values[0]))
	# 			coolS2.append(float(df_s_2['Cool'].values[0]))
	# 			stnpressureS1.append(float(df_s_1['StnPressure'].values[0]))
	# 			stnpressureS2.append(float(df_s_2['StnPressure'].values[0]))
	# 			sealevelS1.append(float(df_s_1['SeaLevel'].values[0]))
	# 			sealevelS2.append(float(df_s_2['SeaLevel'].values[0]))
	# 			resultspeedS1.append(df_s_1['ResultSpeed'].values[0])
	# 			resultspeedS2.append(df_s_2['ResultSpeed'].values[0])
	# 			resultdirS1.append(df_s_1['ResultDir'].values[0])
	# 			resultdirS2.append(df_s_2['ResultDir'].values[0])
	# 			avgspeedS1.append(float(df_s_1['AvgSpeed'].values[0]))
	# 			avgspeedS2.append(float(df_s_2['AvgSpeed'].values[0]))
	# 			preciptotalS1.append(float(df_s_1['PrecipTotal'].values[0]))
	# 			preciptotalS2.append(float(df_s_2['PrecipTotal'].values[0]))

	# 			sunrise.append(int(df_s_1['Sunrise'].values[0]))
	# 			sunset.append(int(df_s_1['Sunset'].values[0]))

	# 	species = pd.get_dummies(df_piece['Species'])
	# 	traps = pd.get_dummies(df_piece['Trap'])
	# 	addresses = pd.get_dummies(df_piece['Address'])

	# 	df_piece = pd.concat([df_piece, species], axis=1, join_axes=[df_piece.index])
	# 	df_piece = pd.concat([df_piece, traps], axis=1, join_axes=[df_piece.index])
	# 	df_piece = pd.concat([df_piece, addresses], axis=1, join_axes=[df_piece.index])
		
	# 	if not (self.df_test is None):
	# 		test_species = pd.get_dummies(self.df_test['Species'])
	# 		test_traps = pd.get_dummies(self.df_test['Trap'])
	# 		test_addresses = pd.get_dummies(self.df_test['Address'])

	# 		self.df_test = self.df_test.drop(['Id'], axis=1)		
	# 		self.df_test = pd.concat([self.df_test, test_species], axis=1, join_axes=[self.df_test.index])
	# 		self.df_test = pd.concat([self.df_test, test_traps], axis=1, join_axes=[self.df_test.index])
	# 		self.df_test = pd.concat([self.df_test, test_addresses], axis=1, join_axes=[self.df_test.index])

	# 		for index, column in enumerate(self.df_test.columns):
	# 			if not (column in df_piece.columns):
	# 				df_piece.insert(loc=index+2, column=column, value=np.zeros(len(df_piece)))
		
	# 	df_piece = df_piece.drop(['Date', 'Street', 'AddressNumberAndStreet', 'Species', 'Trap', 'Address'], axis=1)
		
	# 	df_piece.insert(loc=len(df_piece.columns), column='Tmax_S1', value=tmaxS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Tmax_S2', value=tmaxS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Tmin_S1', value=tminS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Tmin_S2', value=tminS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Tavg_S1', value=tavgS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Tavg_S2', value=tavgS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='DewPoint_S1', value=dewpointS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='DewPoint_S2', value=dewpointS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='WetBulb_S1', value=wetbulbS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='WetBulb_S2', value=wetbulbS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Heat_S1', value=heatS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Heat_S2', value=heatS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Cool_S1', value=coolS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Cool_S2', value=coolS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='StnPressure_S1', value=stnpressureS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='StnPressure_S2', value=stnpressureS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='SeaLevel_S1', value=sealevelS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='SeaLevel_S2', value=sealevelS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='ResultSpeed_S1', value=resultspeedS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='ResultSpeed_S2', value=resultspeedS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='ResultDir_S1', value=resultdirS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='ResultDir_S2', value=resultdirS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='AvgSpeed_S1', value=avgspeedS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='AvgSpeed_S2', value=avgspeedS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='PrecipTotal_S1', value=preciptotalS1)
	# 	df_piece.insert(loc=len(df_piece.columns), column='PrecipTotal_S2', value=preciptotalS2)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Sunrise', value=sunrise)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Sunset', value=sunset)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Day', value=day)
	# 	df_piece.insert(loc=len(df_piece.columns), column='Year', value=year)

	# 	return df_piece
