"""
Pemetaan sektor/industri yfinance → internal, konstanta Graham, dan target
multiple EV/EBITDA & EV/Sales. Nilai bisnis dipertahankan apa adanya dari
skrip asli.
"""
import numpy as np


SECTOR_MAP = {
    "Technology":                "Technology",
    "Healthcare":                "Healthcare",
    "Financial Services":        "Financial Services",
    "Consumer Defensive":        "Consumer Defensive",
    "Consumer Cyclical":         "Consumer Cyclical",
    "Communication Services":    "Industrials",       # Telco masuk Industrials di IDX
    "Utilities":                 "Utilities",
    "Industrials":               "Industrials",
    "Basic Materials":           "Basic Materials",
    "Energy":                    "Energy",
    "Real Estate":               "Real Estate",
}

INDUSTRY_MAP = {
    # ── Technology ──────────────────────────────────────────────
    "Software—Application":                        "Software",
    "Software—Infrastructure":                     "Software",
    "Information Technology Services":             "Software",
    "Internet Content & Information":              "Software",
    "Computer Hardware":                           "Computer Hardware",
    "Electronic Components":                       "Computer Hardware",
    "Electronics & Computer Distribution":         "Computer Hardware",
    "Semiconductors":                              "Computer Hardware",
    "Communication Equipment":                     "Computer Hardware",
    "Consumer Electronics":                        "Computer Hardware",
    "Scientific & Technical Instruments":          "Computer Hardware",

    # ── Healthcare ───────────────────────────────────────────────
    "Drug Manufacturers—General":                  "Healthcare",
    "Drug Manufacturers—Specialty & Generic":      "Healthcare",
    "Pharmaceutical Retailers":                    "Healthcare",
    "Biotechnology":                               "Healthcare",
    "Medical Devices":                             "Healthcare",
    "Medical Distribution":                        "Healthcare",
    "Medical Instruments & Supplies":              "Healthcare",
    "Health Information Services":                 "Healthcare",
    "Healthcare Plans":                            "Healthcare",
    "Medical Care Facilities":                     "Healthcare",
    "Diagnostics & Research":                      "Healthcare",

    # ── Financial Services ───────────────────────────────────────
    "Banks—Regional":                              "Banks",
    "Banks—Diversified":                           "Banks",
    "Insurance—Life":                              "Insurance",
    "Insurance—Property & Casualty":               "Insurance",
    "Insurance—Diversified":                       "Insurance",
    "Insurance—Reinsurance":                       "Insurance",
    "Insurance—Specialty":                         "Insurance",
    "Credit Services":                             "Credit Services",
    "Financial Conglomerates":                     "Conglomerates",
    "Capital Markets":                             "Asset Management",
    "Asset Management":                            "Asset Management",
    "Investment—Banking & Investment Services":    "Asset Management",
    "Financial Data & Stock Exchanges":            "Asset Management",
    "Mortgage Finance":                            "Credit Services",

    # ── Consumer Defensive ───────────────────────────────────────
    "Tobacco":                                     "Tobacco",
    "Household & Personal Products":               "Household & Personal Products",
    "Packaged Foods":                              "Packaged Foods",
    "Confectioners":                               "Packaged Foods",
    "Beverages—Non-Alcoholic":                     "Packaged Foods",
    "Beverages—Alcoholic":                         "Packaged Foods",
    "Beverages—Brewers":                           "Packaged Foods",
    "Farm Products":                               "Packaged Foods",
    "Food Distribution":                           "Packaged Foods",
    "Grocery Stores":                              "Household & Personal Products",
    "Discount Stores":                             "Household & Personal Products",

    # ── Consumer Cyclical ────────────────────────────────────────
    "Broadcasting":                                "Entertainment",
    "Entertainment":                               "Entertainment",
    "Publishing":                                  "Entertainment",
    "Advertising Agencies":                        "Entertainment",
    "Leisure":                                     "Leisure Products",
    "Travel Services":                             "Restaurants / Travel Services",
    "Gambling":                                    "Leisure Products",
    "Specialty Retail":                            "Specialty Retail",
    "Department Stores":                           "Specialty Retail",
    "Home Improvement Retail":                     "Specialty Retail",
    "Luxury Goods":                                "Apparel Retail",
    "Apparel Retail":                              "Apparel Retail",
    "Apparel Manufacturing":                       "Apparel Retail",
    "Footwear & Accessories":                      "Apparel Retail",
    "Auto Manufacturers":                          "Auto Manufacturers",
    "Auto Parts":                                  "Auto Manufacturers",
    "Auto & Truck Dealerships":                    "Auto Manufacturers",
    "Furnishings, Fixtures & Appliances":          "Furnishings / Appliances",
    "Residential Construction":                    "Furnishings / Appliances",
    "Personal Services":                           "Restaurants / Travel Services",
    "Restaurants":                                 "Restaurants / Travel Services",
    "Hotels & Motels":                             "Restaurants / Travel Services",

    # ── Industrials (termasuk Infrastruktur IDX) ─────────────────
    "Telecom Services":                            "Telecom Services",
    "Airport & Air Services":                      "Railroads / Logistics",
    "Toll Roads":                                  "Railroads / Logistics",
    "Ports & Services":                            "Railroads / Logistics",
    "Railroads":                                   "Airlines / Railroads / Marine",
    "Airlines":                                    "Airlines / Railroads / Marine",
    "Marine Shipping":                             "Airlines / Railroads / Marine",
    "Trucking":                                    "Integrated Freight & Logistics",
    "Integrated Freight & Logistics":              "Integrated Freight & Logistics",
    "Courier & Delivery":                          "Integrated Freight & Logistics",
    "Engineering & Construction":                  "Engineering & Construction",
    "Infrastructure Operations":                   "Railroads / Logistics",
    "Conglomerates":                               "Conglomerates",
    "Specialty Industrial Machinery":              "Industrial Products",
    "Industrial Distribution":                     "Industrial Products",
    "Electrical Equipment & Parts":                "Industrial Products",
    "Metal Fabrication":                           "Industrial Products",
    "Tools & Accessories":                         "Industrial Products",
    "Staffing & Employment Services":              "Engineering & Construction",
    "Waste Management":                            "Engineering & Construction",
    "Security & Protection Services":              "Engineering & Construction",
    "Rental & Leasing Services":                   "Engineering & Construction",
    "Business Equipment & Supplies":               "Industrial Products",
    "Consulting Services":                         "Engineering & Construction",

    # ── Utilities ────────────────────────────────────────────────
    "Utilities—Regulated Electric":                "Utilities",
    "Utilities—Regulated Gas":                     "Utilities",
    "Utilities—Regulated Water":                   "Utilities",
    "Utilities—Independent Power Producers":       "Utilities",
    "Utilities—Renewable":                         "Renewable",
    "Utilities—Diversified":                       "Utilities",

    # ── Energy ───────────────────────────────────────────────────
    "Oil & Gas Integrated":                        "Oil & Gas",
    "Oil & Gas E&P":                               "Oil & Gas",
    "Oil & Gas Midstream":                         "Oil & Gas",
    "Oil & Gas Refining & Marketing":              "Oil & Gas",
    "Oil & Gas Equipment & Services":              "Oil & Gas",
    "Thermal Coal":                                "Thermal Coal",
    "Coking Coal":                                 "Thermal Coal",
    "Solar":                                       "Renewable",

    # ── Basic Materials ──────────────────────────────────────────
    "Steel":                                       "Basic Materials",
    "Aluminum":                                    "Basic Materials",
    "Copper":                                      "Basic Materials",
    "Gold":                                        "Basic Materials",
    "Silver":                                      "Basic Materials",
    "Specialty Chemicals":                         "Basic Materials",
    "Chemicals":                                   "Basic Materials",
    "Paper & Paper Products":                      "Basic Materials",
    "Lumber & Wood Production":                    "Basic Materials",
    "Agricultural Inputs":                         "Basic Materials",
    "Building Materials":                          "Basic Materials",
    "Cement":                                      "Basic Materials",
    "Other Industrial Metals & Mining":            "Basic Materials",

    # ── Real Estate ──────────────────────────────────────────────
    "Real Estate—Development":                     "Real Estate",
    "Real Estate—Diversified":                     "Real Estate",
    "Real Estate Services":                        "Real Estate",
    "REIT—Diversified":                            "Real Estate",
    "REIT—Office":                                 "Real Estate",
    "REIT—Retail":                                 "Real Estate",
    "REIT—Industrial":                             "Real Estate",
    "REIT—Residential":                            "Real Estate",
}

def normalize_sector(raw):   return SECTOR_MAP.get(raw, raw)
def normalize_industry(raw): return INDUSTRY_MAP.get(raw, raw)


# Format: (normalized_sector, normalized_industry): {K, target_pe, target_pb}

# Fallback Graham K jika sector/industry tidak ditemukan (Benjamin Graham original)
GRAHAM_K_DEFAULT = 22.5

GRAHAM_CONSTANTS = {
    # ── Consumer Defensive ───────────────────────────────────────
    ("Consumer Defensive", "Tobacco"):                              {"K": 35.6608, "target_pe": 17.92, "target_pb": 1.99},
    ("Consumer Defensive", "Household & Personal Products"):        {"K": 97.5664, "target_pe": 16.88, "target_pb": 5.78},
    ("Consumer Defensive", "Packaged Foods"):                       {"K": 33.6000, "target_pe": 12.00, "target_pb": 2.80},

    # ── Consumer Cyclical ────────────────────────────────────────
    ("Consumer Cyclical",  "Entertainment"):                        {"K": 28.2880, "target_pe": 10.40, "target_pb": 2.72},
    ("Consumer Cyclical",  "Specialty Retail"):                     {"K": 13.3986, "target_pe":  9.78, "target_pb": 1.37},
    ("Consumer Cyclical",  "Apparel Retail"):                       {"K":  8.3239, "target_pe":  3.37, "target_pb": 2.47},
    ("Consumer Cyclical",  "Auto Manufacturers"):                   {"K": 28.2880, "target_pe": 10.40, "target_pb": 2.72},
    ("Consumer Cyclical",  "Leisure Products"):                     {"K": 28.2880, "target_pe": 10.40, "target_pb": 2.72},
    ("Consumer Cyclical",  "Furnishings / Appliances"):             {"K": 28.2880, "target_pe": 10.40, "target_pb": 2.72},
    ("Consumer Cyclical",  "Restaurants / Travel Services"):        {"K": 28.2880, "target_pe": 10.40, "target_pb": 2.72},

    # ── Technology ───────────────────────────────────────────────
    ("Technology",         "Software"):                             {"K": 66.4598, "target_pe": 17.77, "target_pb": 3.74},
    ("Technology",         "Computer Hardware"):                    {"K": 33.8800, "target_pe":  8.80, "target_pb": 3.85},

    # ── Healthcare ───────────────────────────────────────────────
    ("Healthcare",         "Healthcare"):                           {"K": 15.2448, "target_pe":  7.94, "target_pb": 1.92},
    # Rumah Sakit (Jasa & Peralatan Kesehatan) — K lebih tinggi
    # yfinance biasanya masuk "Medical Care Facilities" lalu di-normalize ke "Healthcare"
    # Jika perlu split, tambahkan di sini dengan industry berbeda

    # ── Financial Services ───────────────────────────────────────
    ("Financial Services", "Banks"):                                {"K": 16.5163, "target_pe":  9.89, "target_pb": 1.67},
    ("Financial Services", "Insurance"):                            {"K": 10.3375, "target_pe":  8.27, "target_pb": 1.25},
    ("Financial Services", "Credit Services"):                      {"K": 10.3964, "target_pe": 11.06, "target_pb": 0.94},
    ("Financial Services", "Asset Management"):                     {"K": 127.344, "target_pe": 37.90, "target_pb": 3.36},
    ("Financial Services", "Conglomerates"):                        {"K": 79.9968, "target_pe": 19.23, "target_pb": 4.16},

    # ── Industrials ──────────────────────────────────────────────
    ("Industrials",        "Conglomerates"):                        {"K": 15.8841, "target_pe":  9.99, "target_pb": 1.59},
    ("Industrials",        "Industrial Products"):                  {"K":  8.9090, "target_pe":  7.55, "target_pb": 1.18},
    ("Industrials",        "Engineering & Construction"):           {"K":  7.6120, "target_pe":  6.92, "target_pb": 1.10},
    ("Industrials",        "Railroads / Logistics"):                {"K": 11.9880, "target_pe": 10.80, "target_pb": 1.11},
    ("Industrials",        "Telecom Services"):                     {"K": 32.8659, "target_pe": 15.43, "target_pb": 2.13},
    ("Industrials",        "Airlines / Railroads / Marine"):        {"K":  0.5856, "target_pe":  1.83, "target_pb": 0.32},
    ("Industrials",        "Integrated Freight & Logistics"):       {"K": 31.4424, "target_pe": 15.88, "target_pb": 1.98},

    # ── Utilities ────────────────────────────────────────────────
    ("Utilities",          "Utilities"):                            {"K": 36.6085, "target_pe": 17.35, "target_pb": 2.11},
    ("Utilities",          "Renewable"):                            {"K": 71.1657, "target_pe": 22.17, "target_pb": 3.21},

    # ── Energy ───────────────────────────────────────────────────
    ("Energy",             "Oil & Gas"):                            {"K": 71.1657, "target_pe": 22.17, "target_pb": 3.21},
    ("Energy",             "Thermal Coal"):                         {"K": 71.1657, "target_pe": 22.17, "target_pb": 3.21},
    ("Energy",             "Renewable"):                            {"K": 71.1657, "target_pe": 22.17, "target_pb": 3.21},

    # ── Basic Materials ──────────────────────────────────────────
    ("Basic Materials",    "Basic Materials"):                      {"K": 48.0150, "target_pe": 16.50, "target_pb": 2.91},

    # ── Real Estate ──────────────────────────────────────────────
    ("Real Estate",        "Real Estate"):                          {"K": 21.5775, "target_pe": 15.75, "target_pb": 1.37},
}

# EV/EBITDA target multiple — key pakai normalized sector
EV_EBITDA_TARGETS = {
    "Technology":           18.0,
    "Healthcare":           14.0,
    "Consumer Defensive":   12.0,
    "Consumer Cyclical":    10.0,
    "Financial Services":    9.0,
    "Energy":                6.0,
    "Basic Materials":       7.0,
    "Industrials":           8.0,
    "Utilities":             8.0,
    "Real Estate":           9.0,
}
EV_EBITDA_DEFAULT = 9.0

# EV/Sales (Price-to-Sales) target multiple — fallback jika FCF tidak cukup positif
# Dipakai untuk growth/loss-making companies yang tidak bisa di-DCF
EV_SALES_TARGETS = {
    "Technology":          6.0,   # SaaS/Software premium
    "Healthcare":          4.0,
    "Consumer Defensive":  2.0,
    "Consumer Cyclical":   1.5,
    "Financial Services":  3.0,
    "Energy":              1.0,
    "Basic Materials":     1.0,
    "Industrials":         1.2,
    "Utilities":           2.0,
    "Real Estate":         3.0,
}
EV_SALES_DEFAULT = 2.0

# Fallback Graham K jika sektor/industri tidak ditemukan di GRAHAM_CONSTANTS
# Menggunakan konstanta asli Benjamin Graham
GRAHAM_K_DEFAULT  = 22.5

