# Amazon Market Analyzer

This project contains a Streamlit dashboard for analyzing product arbitrage opportunities across different Amazon marketplaces.

## Requirements

- **Python**: 3.10 or higher is recommended.
- Install dependencies either by running `./setup.sh` or manually with `pip install -r requirements.txt`.

## Launching the App

After installing dependencies, start the dashboard with:

```bash
streamlit run app.py
```

## Configuration

Provide your Keepa API key and optional password using environment variables or a local `secrets.toml` file. When using environment variables set `API_KEY` (and `PASSWORD` if needed) before running Streamlit:

```bash
export API_KEY="your_keepa_key"
export PASSWORD="your_password"
```

Alternatively create a `.streamlit/secrets.toml` file with the same fields. This file is ignored by Git and can be passed to Streamlit automatically when placed in the project directory:

```toml
API_KEY = "your_keepa_key"
PASSWORD = "your_password"
```


## Keepa Export Files

The application expects Keepa CSV/XLSX exports for both the origin marketplace and the comparison marketplaces. Multiple comparison files can be uploaded at once and will be merged automatically. Essential headers include:

- `ASIN`
- `Title`
- `Locale`
- Price columns: `Buy Box 🚚: Current`, `Amazon: Current`, `New: Current`
- Metrics: `Sales Rank: Current`, `Sales Rank: 30 days avg.`, `Bought in past month`, `New Offer Count: Current`
- Optional data: `Brand`, `Package: Dimension (cm³)`, `Weight`, `Item Weight`, `Package: Weight (kg)` or `Package: Weight (g)`

These headers must be present (with `(base)` or `(comp)` suffixes after upload) for correct processing.

## Main Features

- **Opportunity Score** – weighted ranking combining margin, volume, sales rank, offers, trend and other factors.
- **Advanced Filters** – limit results by sales rank, offer count, price range and minimum margins.
- **Shipping & VAT Handling** – calculate net margins including shipping costs and market‑specific VAT rates.
- **Interactive Dashboard** – visualize scores, margins and volume with histograms and scatter plots; save/load parameter "recipes" for repeated analyses.
- **Cross-market Ranking** – upload multiple comparison lists and view a consolidated table showing the best marketplace and score for each ASIN.

## Result Column Order

The application displays the results using a predefined column sequence. This
order ensures that the most important information comes first. Only the
columns listed in `DISPLAY_COLS_ORDER` are shown:

1. **Locale (base)**
2. **Locale (comp)**
3. **Title (base)**
4. **ASIN**
5. **Margine_Stimato**
6. **Margine_Netto_%**
7. **Margine_Netto**
8. **Price_Base**
9. **Acquisto_Netto**
10. **Shipping_Cost**
11. **Price_Comp**
12. **Vendita_Netto**
13. **Bought_Comp**
14. **SalesRank_Comp**
15. **Trend**
16. **NewOffer_Comp**
17. **Opportunity_Score**
18. **Opportunity_Class**
19. **Volume_Score**
20. **Weight_kg**
21. **Package: Dimension (cm³) (base)**
22. **IVA_Origine**
23. **IVA_Confronto**

Any additional columns present in your dataset are ignored.


## VAT and Discount Logic

Locale strings from the CSV files are normalized to a two letter country code (e.g. `Amazon.de` or `de-DE` become `DE`). `GB` is converted to `UK` so that the correct VAT rate is applied automatically.

For purchases from foreign markets the discount is calculated on the VAT excluded price:

```
Net = Price / (1 + VAT) * (1 - Discount)
```

When buying in Italy the discount is first computed on the gross price and then subtracted from the VAT excluded price:

```
Net = Price / (1 + VAT) - Price * Discount
```

The dashboard chooses the proper VAT rate for each row and displays both origin and comparison market VAT percentages.


## License

This project is licensed under the [MIT License](LICENSE).
