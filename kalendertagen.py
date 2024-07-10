import csv
from datetime import datetime, timedelta, timezone

# Funktion, um das Datum im gewünschten Format zu erstellen
def format_date_with_timezone(dt):
    return dt.isoformat()

# Schritt 1: Alle Tage des Jahres 2024 generieren
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
delta = timedelta(days=1)

all_days = []
current_date = start_date
while current_date <= end_date:
    all_days.append(current_date)
    current_date += delta

# Feiertage aus der CSV-Datei einlesen
feiertage_set = set()
with open('feiertage.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        datum = row['datum'].split('T')[0]  # Datum kürzen auf das Format %Y-%m-%d
        feiertage_set.add(datum)

# Temporäre CSV-Datei erstellen
temp_file = 'kalendertage_temp.csv'
with open(temp_file, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = ['datum', 'feiertag', 'werktag', 'gruner_markt', 'spezialitaten_markt']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    
    for day in all_days:
        datum_str = day.strftime('%Y-%m-%d')
        feiertag = 1 if datum_str in feiertage_set else 0
        werktag = 1 if day.weekday() < 5 else 0  # Montag-Freitag sind Werktage
        
        # Grüner Markt Tage: Dienstag, Mittwoch, Freitag, Samstag
        gruner_markt = 1 if day.weekday() in [1, 2, 4, 5] else 0
        
        # Spezialitätenmarkt Tage: Dienstag, Mittwoch, Freitag, Samstag
        spezialitaten_markt = 1 if day.weekday() in [1, 2, 4, 5] else 0

        writer.writerow({
            'datum': datum_str,
            'feiertag': feiertag,
            'werktag': werktag,
            'gruner_markt': gruner_markt,
            'spezialitaten_markt': spezialitaten_markt
        })

# Schritt 2: Finale CSV-Datei mit dem gewünschten Datumsformat erstellen
final_file = 'kalendertage_2024.csv'
with open(temp_file, mode='r', encoding='utf-8') as infile, open(final_file, mode='w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ['datum', 'feiertag', 'werktag', 'gruner_markt', 'spezialitaten_markt']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in reader:
        datum_str = row['datum']
        date_obj = datetime.strptime(datum_str, '%Y-%m-%d').replace(tzinfo=timezone(timedelta(hours=1)))
        formatted_date = format_date_with_timezone(date_obj)
        writer.writerow({
            'datum': formatted_date,
            'feiertag': row['feiertag'],
            'werktag': row['werktag'],
            'gruner_markt': row['gruner_markt'],
            'spezialitaten_markt': row['spezialitaten_markt']
        })

print("Die Tabelle mit allen Kalendertagen für 2024 wurde erfolgreich erstellt.")