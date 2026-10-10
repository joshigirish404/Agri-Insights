# Agri-Insights

Agri-Insights is an interactive Streamlit dashboard for exploring vegetation health and rainfall patterns across India. It combines MODIS NDVI imagery with actual and normal rainfall data to help users assess agricultural conditions and identify regional variations.

## Features

- **Interactive administrative filters** — Select multiple states, districts, and subdistricts, with a reset option.
- **MODIS NDVI maps** — Explore 250 m, 16-day vegetation-index imagery by year and available fortnight.
- **Rainfall deviation maps** — Compare actual rainfall with normal rainfall and view deviation categories based on IMD-style classifications.
- **Daily, weekly, and monthly analysis** — Examine rainfall over a selected date range or weekly/monthly periods.
- **Rainfall charts** — Plot actual and normal rainfall series for selected areas.
- **Interactive map display** — View spatial data over Esri World Imagery, with administrative boundaries and map legends.
- **Data downloads** — Export the selected NDVI layer as GeoTIFF and rainfall analysis data as CSV.

## Data Sources

- **Vegetation index:** NASA MODIS/Terra Vegetation Indices, MOD13Q1 Version 6.1 (250 m, 16-day).
- **Rainfall:** India Meteorological Department (IMD) daily gridded rainfall data; district and subdistrict summaries and normal values are used in the dashboard.
- **Administrative boundaries and datasets:** Survey of India (SoI).
- **Basemap:** Esri World Imagery.

