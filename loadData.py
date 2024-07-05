import importFiles as imf
import weatherDataHandling as wdh
data1=imf.handleImportOfPedestrianData()
weather=imf.handleImportOfWeatherData()
result=wdh.weatherHandling(data1, weather)
print(result)
