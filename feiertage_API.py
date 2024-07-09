#pip install deutschland[feiertage]

from deutschland import feiertage
from pprint import pprint
from deutschland.feiertage.api import default_api
import csv
import time
from datetime import date

# Konfiguration der API
configuration = feiertage.Configuration(
    host = "https://feiertage-api.de/api"
)
# API-Aufruf in einem Kontext
with feiertage.ApiClient(configuration) as api_client:
    # Erstellen einer Instanz der API-Klasse
    api_instance = default_api.DefaultApi(api_client)
    jahr = "2024"  # Welches Jahr? (optional)
    nur_land = "BE"  # Welches Bundesland? (optional)
    nur_daten = 1  # Nur Daten oder auch Hinweise? (optional)

    try:
        # Abrufen der Feiertage
        api_response = api_instance.get_feiertage(jahr=jahr, nur_land=nur_land, nur_daten=nur_daten)
        pprint(api_response)  # Struktur der Antwort überprüfen
        
        # Umwandeln des API-Antworten in CSV-Format und speichern
        feiertage_list = []
        for name, datum in api_response.items():
            feiertage_list.append({
                "datum": datum.isoformat(),  # Konvertiere datetime.date zu String
                "name": name,
                "hinweis": ''  # Keine Hinweise in der API-Antwort
            })
        
        # CSV-Datei schreiben
        with open('feiertage.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["datum", "name", "hinweis"])
            writer.writeheader()
            for feiertag in feiertage_list:
                writer.writerow(feiertag)

        print("Feiertage wurden erfolgreich als CSV gespeichert.")

    except feiertage.ApiException as e:
        print("Exception when calling DefaultApi->get_feiertage: %s\n" % e)