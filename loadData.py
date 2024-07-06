import importFiles as imf
import weatherDataHandling as wdh
data1=imf.handleImportOfPedestrianData()
weather=imf.handleImportOfWeatherData()

def load_Data():
    results=wdh.weatherHandling(data1, weather)
    return results
#print(load_Data())