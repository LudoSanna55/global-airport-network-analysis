import pandas as pd

def load_data(airports_path="dataset/airports.csv",
              routes_path="dataset/routes.csv"):
    airports = pd.read_csv(airports_path, na_values=["\\N"])
    routes = pd.read_csv(routes_path, na_values=["\\N"])

    return airports, routes

def normalize_airports_columns(airports):
    rename_map = {
        "Airport ID": "airport_id",
        "Name": "name",
        "City": "city",
        "Country": "country",
        "IATA": "iata",
        "ICAO": "icao",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Altitude": "altitude",
        "Timezone": "timezone",
        "DST": "dst",
        "Tz database time zone": "tz_database_time_zone",
        "Type": "type",
        "Source": "source",
    }

    airports = airports.rename(columns=rename_map)
    return airports

def normalize_routes_columns(routes):
    rename_map = {
        "Airline": "airline",
        "Airline ID": "airline_id",
        "Source airport": "source_airport",
        "Source airport ID": "source_airport_id",
        "Destination airport": "destination_airport",
        "Destination airport ID": "destination_airport_id",
        "Codeshare": "codeshare",
        "Stops": "stops",
        "Equipment": "equipment",
    }

    routes = routes.rename(columns=rename_map)
    return routes

def clean_airports(airports):
    airports["airport_id"] = pd.to_numeric(airports["airport_id"], errors="coerce")

    airports = airports.dropna(subset=["airport_id"])

    airports["airport_id"] = airports["airport_id"].astype(int)

    cols_to_keep = [
        "airport_id",
        "name",
        "city",
        "country",
        "iata",
        "icao",
        "latitude",
        "longitude",
    ]

    cols_to_keep = [col for col in cols_to_keep if col in airports.columns]

    airports = airports[cols_to_keep].copy()

    return airports

def add_continent_column(airports):
    country_to_continent = {
        # Europa
        "Austria": "Europe",
        "Belgium": "Europe",
        "Bulgaria": "Europe",
        "Croatia": "Europe",
        "Cyprus": "Europe",
        "Czech Republic": "Europe",
        "Denmark": "Europe",
        "Estonia": "Europe",
        "Faroe Islands": "Europe",
        "Finland": "Europe",
        "France": "Europe",
        "Germany": "Europe",
        "Gibraltar": "Europe",
        "Guernsey": "Europe",
        "Hungary": "Europe",
        "Iceland": "Europe",
        "Ireland": "Europe",
        "Isle of Man": "Europe",
        "Italy": "Europe",
        "Jersey": "Europe",
        "Kosovo": "Europe",
        "Latvia": "Europe",
        "Liechtenstein": "Europe",
        "Lithuania": "Europe",
        "Luxembourg": "Europe",
        "Macedonia": "Europe",
        "Malta": "Europe",
        "Moldova": "Europe",
        "Monaco": "Europe",
        "Montenegro": "Europe",
        "Netherlands": "Europe",
        "Norway": "Europe",
        "Poland": "Europe",
        "Portugal": "Europe",
        "Romania": "Europe",
        "Russia": "Europe",
        "San Marino": "Europe",
        "Serbia": "Europe",
        "Slovakia": "Europe",
        "Slovenia": "Europe",
        "Spain": "Europe",
        "Sweden": "Europe",
        "Switzerland": "Europe",
        "Ukraine": "Europe",
        "United Kingdom": "Europe",
        "Vatican City": "Europe",

        # Nord America
        "Antigua and Barbuda": "North America",
        "Aruba": "North America",
        "Bahamas": "North America",
        "Barbados": "North America",
        "Belize": "North America",
        "Bermuda": "North America",
        "Canada": "North America",
        "Costa Rica": "North America",
        "Cuba": "North America",
        "Dominica": "North America",
        "Dominican Republic": "North America",
        "El Salvador": "North America",
        "Greenland": "North America",
        "Guatemala": "North America",
        "Haiti": "North America",
        "Honduras": "North America",
        "Jamaica": "North America",
        "Mexico": "North America",
        "Nicaragua": "North America",
        "Panama": "North America",
        "Puerto Rico": "North America",
        "Saint Kitts and Nevis": "North America",
        "Saint Lucia": "North America",
        "Saint Vincent and the Grenadines": "North America",
        "Trinidad and Tobago": "North America",
        "United States": "North America",
        "Wake Island": "North America",
        "Johnston Atoll": "North America",

        # Sud America
        "Argentina": "South America",
        "Bolivia": "South America",
        "Brazil": "South America",
        "Chile": "South America",
        "Colombia": "South America",
        "Ecuador": "South America",
        "Guyana": "South America",
        "Paraguay": "South America",
        "Peru": "South America",
        "Suriname": "South America",
        "Uruguay": "South America",
        "Venezuela": "South America",

        # Asia
        "Afghanistan": "Asia",
        "Armenia": "Asia",
        "Azerbaijan": "Asia",
        "Bangladesh": "Asia",
        "Bhutan": "Asia",
        "Brunei": "Asia",
        "Cambodia": "Asia",
        "China": "Asia",
        "East Timor": "Asia",
        "Georgia": "Asia",
        "Hong Kong": "Asia",
        "India": "Asia",
        "Indonesia": "Asia",
        "Iran": "Asia",
        "Iraq": "Asia",
        "Israel": "Asia",
        "Japan": "Asia",
        "Jordan": "Asia",
        "Kazakhstan": "Asia",
        "Kuwait": "Asia",
        "Kyrgyzstan": "Asia",
        "Laos": "Asia",
        "Lebanon": "Asia",
        "Macau": "Asia",
        "Malaysia": "Asia",
        "Maldives": "Asia",
        "Mongolia": "Asia",
        "Myanmar": "Asia",
        "Burma": "Asia",
        "Nepal": "Asia",
        "North Korea": "Asia",
        "Oman": "Asia",
        "Pakistan": "Asia",
        "Palestine": "Asia",
        "Philippines": "Asia",
        "Qatar": "Asia",
        "Saudi Arabia": "Asia",
        "Singapore": "Asia",
        "Sri Lanka": "Asia",
        "South Korea": "Asia",
        "Syria": "Asia",
        "Taiwan": "Asia",
        "Tajikistan": "Asia",
        "Thailand": "Asia",
        "Timor-Leste": "Asia",
        "Turkey": "Asia",
        "Turkmenistan": "Asia",
        "United Arab Emirates": "Asia",
        "Uzbekistan": "Asia",
        "Vietnam": "Asia",
        "Yemen": "Asia",
        "Svalbard": "Europe",

        # Africa
        "Algeria": "Africa",
        "Angola": "Africa",
        "Benin": "Africa",
        "Botswana": "Africa",
        "Burkina Faso": "Africa",
        "Burundi": "Africa",
        "Cameroon": "Africa",
        "Cape Verde": "Africa",
        "Central African Republic": "Africa",
        "Chad": "Africa",
        "Comoros": "Africa",
        "Congo (Kinshasa)": "Africa",
        "Congo (Brazzaville)": "Africa",
        "Djibouti": "Africa",
        "Egypt": "Africa",
        "Eritrea": "Africa",
        "Ethiopia": "Africa",
        "Gabon": "Africa",
        "Ghana": "Africa",
        "Guinea": "Africa",
        "Guinea-Bissau": "Africa",
        "Ivory Coast": "Africa",
        "Kenya": "Africa",
        "Lesotho": "Africa",
        "Liberia": "Africa",
        "Libya": "Africa",
        "Madagascar": "Africa",
        "Malawi": "Africa",
        "Mali": "Africa",
        "Mauritania": "Africa",
        "Mauritius": "Africa",
        "Mayotte": "Africa",
        "Morocco": "Africa",
        "Mozambique": "Africa",
        "Namibia": "Africa",
        "Niger": "Africa",
        "Nigeria": "Africa",
        "Reunion": "Africa",
        "Rwanda": "Africa",
        "Senegal": "Africa",
        "Seychelles": "Africa",
        "Sierra Leone": "Africa",
        "Somalia": "Africa",
        "South Africa": "Africa",
        "South Sudan": "Africa",
        "Sudan": "Africa",
        "Swaziland": "Africa",
        "Tanzania": "Africa",
        "Togo": "Africa",
        "Tunisia": "Africa",
        "Uganda": "Africa",
        "Western Sahara": "Africa",
        "Zambia": "Africa",
        "Zimbabwe": "Africa",

        # Oceania
        "Australia": "Oceania",
        "Cook Islands": "Oceania",
        "Fiji": "Oceania",
        "French Polynesia": "Oceania",
        "Guam": "Oceania",
        "Kiribati": "Oceania",
        "Micronesia": "Oceania",
        "Nauru": "Oceania",
        "New Caledonia": "Oceania",
        "New Zealand": "Oceania",
        "Niue": "Oceania",
        "Palau": "Oceania",
        "Papua New Guinea": "Oceania",
        "Samoa": "Oceania",
        "Solomon Islands": "Oceania",
        "Tokelau": "Oceania",
        "Tonga": "Oceania",
        "Tuvalu": "Oceania",
        "Vanuatu": "Oceania",

        # Antartide
        "Antarctica": "Antarctica",
    }

    airports["continent"] = airports["country"].map(country_to_continent)
    print("Missing continents:", airports["continent"].isna().sum())

    return airports

def clean_routes(routes):
    routes["airline_id"] = pd.to_numeric(routes["airline_id"], errors="coerce")
    routes["source_airport_id"] = pd.to_numeric(routes["source_airport_id"], errors="coerce")
    routes["destination_airport_id"] = pd.to_numeric(routes["destination_airport_id"], errors="coerce")

    routes = routes.dropna(subset=["source_airport_id", "destination_airport_id"]).copy()

    routes["source_airport_id"] = routes["source_airport_id"].astype(int)
    routes["destination_airport_id"] = routes["destination_airport_id"].astype(int)

    routes["airline_id"] = routes["airline_id"].astype("Int64")

    if "codeshare" in routes.columns:
        routes["codeshare"] = routes["codeshare"].fillna("")

    if "equipment" in routes.columns:
        routes["equipment"] = routes["equipment"].fillna("")

    cols_to_keep = [
        "airline",
        "airline_id",
        "source_airport_id",
        "destination_airport_id",
        "stops",
    ]

    cols_to_keep = [c for c in cols_to_keep if c in routes.columns]

    routes = routes[cols_to_keep].copy()

    return routes

def validate_data(airports, routes):
    print("\n=== VALIDAZIONE DATI ===")

    print("Numero aeroporti (puliti):", len(airports))
    print("Numero rotte (pulite):", len(routes))

    airport_ids = set(airports["airport_id"].unique())

    missing_source_mask = ~routes["source_airport_id"].isin(airport_ids)
    missing_destination_mask = ~routes["destination_airport_id"].isin(airport_ids)

    num_missing_source = missing_source_mask.sum()
    num_missing_destination = missing_destination_mask.sum()

    print("Rotte con source_airport_id NON presente in airports:", num_missing_source)
    print("Rotte con destination_airport_id NON presente in airports:", num_missing_destination)

    if len(routes) > 0:
        perc_missing_source = num_missing_source / len(routes) * 100
        perc_missing_destination = num_missing_destination / len(routes) * 100

        print("Percentuale rotte con source mancante rispetto ad airports: {:.2f}%".format(perc_missing_source))
        print("Percentuale rotte con destination mancante rispetto ad airports: {:.2f}%".format(perc_missing_destination))

    print("=== FINE VALIDAZIONE ===\n")

def save_clean_datasets(airports, routes,
                        airports_out_path="dataset/cleaned/airports_cleaned.csv",
                        routes_out_path="dataset/cleaned/routes_cleaned.csv"):
    airports.to_csv(airports_out_path, index=False)
    routes.to_csv(routes_out_path, index=False)

    print(f"\nDataset puliti salvati correttamente:")
    print(f" - {airports_out_path}")
    print(f" - {routes_out_path}\n")


if __name__ == "__main__":
    # STEP 1: Caricamento dataset
    airports_df, routes_df = load_data()

    print("Airports shape:", airports_df.shape)
    print("Routes shape:", routes_df.shape)

    print("\nPrime righe airports:")
    print(airports_df.head())

    print("\nPrime righe routes:")
    print(routes_df.head())

    # STEP 2: Normalizzazione colonne
    airports_df = normalize_airports_columns(airports_df)
    routes_df = normalize_routes_columns(routes_df)

    print("Airports colonne normalizzate:", airports_df.columns)
    print("Routes colonne normalizzate:", routes_df.columns)

    # STEP 3: Pulizia airports
    airports_df = clean_airports(airports_df)

    print("\nAirports puliti – shape:", airports_df.shape)
    print(airports_df.head())

    # STEP 3.1.: Aggiunta colonna continent
    airports_df = add_continent_column(airports_df)
    print("\nEsempio colonna continent:")
    print(airports_df[["airport_id", "country", "continent"]].head())

    # STEP 4: Pulizia routes
    routes_df = clean_routes(routes_df)

    print("\nRoutes pulite – shape:", routes_df.shape)
    print(routes_df.head())

    # STEP 5: Validazione
    validate_data(airports_df, routes_df)

    # STEP 6: Salvataggio dataset puliti
    save_clean_datasets(airports_df, routes_df)