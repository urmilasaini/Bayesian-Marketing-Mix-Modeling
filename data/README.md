# Conjura MMM Dataset

This directory contains a Marketing Mix Modeling (MMM) dataset from eCommerce brands.

## Files

| File | Description |
|------|-------------|
| `conjura_mmm_data.csv` | Main dataset (132,760 rows) |
| `conjura_mmm_data_dictionary.tsv` | Field definitions dictionary |

## Data Structure

### Identifiers & Metadata

| Column | Type | Description |
|--------|------|-------------|
| `mmm_timeseries_id` | string | Unique identifier for each timeseries |
| `organisation_id` | string | Anonymous ID for eCommerce brand |
| `organisation_vertical` | string | Product category (follows Google eCommerce taxonomy) |
| `organisation_subvertical` | string | Product subcategory |
| `organisation_marketing_sources` | string | Ad platforms used (Google, Meta, TikTok) |
| `organisation_primary_territory_name` | string | Primary sales territory |
| `territory_name` | string | Territory ("All Territories" or country-level) |
| `date_day` | date | Observation date (minimum 449 consecutive days per timeseries) |
| `currency_code` | string | Currency code |

### Target Variables (Purchase Metrics)

| Column | Description |
|--------|-------------|
| `first_purchases` | Number of new customer purchases (acquisitions) |
| `first_purchases_units` | Units purchased by new customers |
| `first_purchases_original_price` | New customer revenue before discount |
| `first_purchases_gross_discount` | Discount amount for new customers |
| `all_purchases` | Number of all customer purchases |
| `all_purchases_units` | Units purchased by all customers |
| `all_purchases_original_price` | Total revenue before discount |
| `all_purchases_gross_discount` | Total discount amount |

**Note**: `gross_discount` is only available for completed purchases. Use caution with data leakage when using as a control variable.

### Ad Spend (Explanatory Variables)

| Platform | Channel | Column |
|----------|---------|--------|
| **Google** | Paid Search (non-branded) | `google_paid_search_spend` |
| | Shopping Ads | `google_shopping_spend` |
| | Performance Max | `google_pmax_spend` |
| | Display Ads | `google_display_spend` |
| | Video Ads | `google_video_spend` |
| **Meta** | Facebook Ads | `meta_facebook_spend` |
| | Instagram Ads | `meta_instagram_spend` |
| | Other | `meta_other_spend` |
| **TikTok** | TikTok Ads | `tiktok_spend` |

### Engagement Metrics

For each paid channel:
- `<platform>_<channel>_clicks` - Ad clicks
- `<platform>_<channel>_impressions` - Ad impressions

### Organic Traffic

| Column | Description |
|--------|-------------|
| `direct_clicks` | Direct traffic |
| `branded_search_clicks` | Traffic from branded search |
| `organic_search_clicks` | Traffic from organic search |
| `email_clicks` | Traffic from email |
| `referral_clicks` | Traffic from referrals |
| `all_other_clicks` | Other traffic sources |

## Data Characteristics

1. **Multi-brand, multi-territory**: Multiple timeseries from different `organisation_id` and `territory_name` combinations
2. **Sparse channels**: Unused channels are empty (not all brands use all channels)
3. **Timeseries length**: Each timeseries has minimum 449 consecutive days
4. **Currency**: When `territory_name = "All Territories"`, primary territory currency is used

## Use Cases

This dataset is suitable for validating Hill mixture MMM models:
- Measuring effectiveness of multiple ad channels
- Modeling saturation effects
- Estimating adstock/carryover effects
- Analyzing new customer acquisition vs total revenue

## Source & Provenance

- **Provider**: Conjura eCommerce Analytics Platform (Dublin, Ireland)
- **Author**: Anthony Anderson (Head of Data Science, Conjura)
- **DOI**: [10.6084/m9.figshare.25314841.v3](https://doi.org/10.6084/m9.figshare.25314841.v3)
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — 商用利用・派生物の公開を含むあらゆる用途で利用可。帰属表示のみ必須
- **Version**: v3 (2024-06-03)
- **Original URL**: https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841

### Attribution (required by CC BY 4.0)

> Anderson, A. (2024). Multi-Region Marketing Mix Modelling (MMM) Dataset for Several eCommerce Brands. figshare. https://doi.org/10.6084/m9.figshare.25314841.v3 — Licensed under CC BY 4.0.

### Reliability Assessment (2026-03-30)

**Provider credibility**: Conjura is a VC-funded ecommerce analytics company (EUR 15M Series A, 2022). The author holds a PhD in Applied Mathematics (Northwestern) with prior experience at Cambridge and Palantir. The data originates from real ecommerce transactions processed through Conjura's analytics platform, anonymised for public release.

**Data quality**:
- Real (not synthetic) ecommerce data from 93 brands across 15 verticals and 19 territories
- Date range: 2019-07-21 to 2024-06-02 (up to ~5 years per brand)
- Missing values are natural (channels not used by a brand), not data corruption
- No negative values in spend columns
- Data dictionary provided (`conjura_mmm_data_dictionary.tsv`)

**Known limitations**:
- No formal academic paper accompanies the dataset (no peer-reviewed methodology description)
- No academic citations as of 2026-03 (Figshare shows 5,734 views / 1,662 downloads)
- Digital channels only (no TV, radio, print) — appropriate for ecommerce scope
- No ground-truth ROAS available (inherent to real observational data)
- Anonymisation methodology not publicly documented
- Author has since left Conjura; no dataset updates since v3 (2024-06-03)
