# RAG V4 Eligible Hit@5 Root Cause Analysis

Source files:
- `eval/results/rag_v4_latest.json`
- `eval/rag_v4_report.md`

## Summary
- eligible_hit@5 misses: `54`
- root-cause distribution: `{'GOLD_BUG': 54}`
- top20 analysis: every miss has the eligible article at normalized rank 1, but raw evaluator comparison misses it

## By Category
| category | positive_cases | eligible_hit@1 | eligible_hit@5 | preferred_hit@1 | preferred_hit@5 | eligible_misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| article_only_exact | 15 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 5 |
| article_only_normalized | 10 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 3 |
| article_only_typo | 10 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 3 |
| brand_typo | 10 | 0.5000 | 0.6000 | 0.5000 | 0.6000 | 4 |
| brand_plus_article | 10 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 3 |
| article_plus_hard | 10 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 3 |
| article_natural_language | 10 | 0.7000 | 0.7000 | 0.7000 | 0.7000 | 3 |
| pn_minimum_semantics | 10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10 |
| v3_regression | 58 | 0.5690 | 0.6552 | 0.5517 | 0.6379 | 20 |

## Per-case Details
### `v4_article_only_exact_0004` | `article_only_exact` | `GOLD_BUG`
- query: `107-5450`
- resolution_mode: `article_exact`
- timing_ms: `{"deepseek": 0.009800000043469481, "resolution": 512.7492000001439}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `107-5450`
- expected.DN: `None`
- expected.PN: `None`
- expected.connection: `None`
- eligible_competitor_articles: `107-5450`
- preferred_competitor_articles: `107-5450`
- returned_top5:
  - rank 1: `1075450` | `FORTECA` | DN `15.0` | PN `40.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_only_exact_0005` | `article_only_exact` | `GOLD_BUG`
- query: `107-5451`
- resolution_mode: `article_exact`
- timing_ms: `{"deepseek": 0.0152000002344721, "resolution": 498.5114999999496}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `107-5451`
- expected.DN: `None`
- expected.PN: `None`
- expected.connection: `None`
- eligible_competitor_articles: `107-5451`
- preferred_competitor_articles: `107-5451`
- returned_top5:
  - rank 1: `1075451` | `FORTECA` | DN `20.0` | PN `40.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_only_exact_0010` | `article_only_exact` | `GOLD_BUG`
- query: `11с67п 2ЦП.00.0.016.015`
- resolution_mode: `article_exact`
- timing_ms: `{"deepseek": 0.011399999493733048, "resolution": 510.0517000000764}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.015`
- expected.DN: `None`
- expected.PN: `None`
- expected.connection: `None`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.015`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.015`
- returned_top5:
  - rank 1: `11с67п2цп000016015` | `MARSHAL` | DN `15.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_only_exact_0011` | `article_only_exact` | `GOLD_BUG`
- query: `11с67п 2ЦП.00.0.016.020`
- resolution_mode: `article_exact`
- timing_ms: `{"deepseek": 0.01469999915570952, "resolution": 500.71830000069895}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.020`
- expected.DN: `None`
- expected.PN: `None`
- expected.connection: `None`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.020`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.020`
- returned_top5:
  - rank 1: `11с67п2цп000016020` | `MARSHAL` | DN `20.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_only_exact_0012` | `article_only_exact` | `GOLD_BUG`
- query: `11с67п 2ЦП.00.0.016.025`
- resolution_mode: `article_exact`
- timing_ms: `{"deepseek": 0.015500000699830707, "resolution": 512.5631000000794}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.025`
- expected.DN: `None`
- expected.PN: `None`
- expected.connection: `None`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.025`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.025`
- returned_top5:
  - rank 1: `11с67п2цп000016025` | `MARSHAL` | DN `25.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_only_normalized_0018` | `article_only_normalized` | `GOLD_BUG`
- query: `107 5450`
- resolution_mode: `article_exact`
- timing_ms: `{"deepseek": 0.014799999917158857, "resolution": 499.80670000059035}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `107-5450`
- expected.DN: `None`
- expected.PN: `None`
- expected.connection: `None`
- eligible_competitor_articles: `107-5450`
- preferred_competitor_articles: `107-5450`
- returned_top5:
  - rank 1: `1075450` | `FORTECA` | DN `15.0` | PN `40.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_only_normalized_0022` | `article_only_normalized` | `GOLD_BUG`
- query: `11с67п 2цп.00.0.016.015`
- resolution_mode: `article_exact`
- timing_ms: `{"deepseek": 0.009599999430065509, "resolution": 479.83639999984007}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.015`
- expected.DN: `None`
- expected.PN: `None`
- expected.connection: `None`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.015`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.015`
- returned_top5:
  - rank 1: `11с67п2цп000016015` | `MARSHAL` | DN `15.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_only_normalized_0023` | `article_only_normalized` | `GOLD_BUG`
- query: `11с67п 2цп.00.0.016.020`
- resolution_mode: `article_exact`
- timing_ms: `{"deepseek": 0.0152000002344721, "resolution": 470.3693999999814}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.020`
- expected.DN: `None`
- expected.PN: `None`
- expected.connection: `None`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.020`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.020`
- returned_top5:
  - rank 1: `11с67п2цп000016020` | `MARSHAL` | DN `20.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_only_typo_0029` | `article_only_typo` | `GOLD_BUG`
- query: `0075450`
- resolution_mode: `article_fuzzy`
- timing_ms: `{"deepseek": 0.013099999705445953, "resolution": 552.9647000003024}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `107-5450`
- expected.DN: `None`
- expected.PN: `None`
- expected.connection: `None`
- eligible_competitor_articles: `107-5450`
- preferred_competitor_articles: `107-5450`
- returned_top5:
  - rank 1: `1075450` | `FORTECA` | DN `15.0` | PN `40.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_only_typo_0033` | `article_only_typo` | `GOLD_BUG`
- query: `01с67п2цп000016015`
- resolution_mode: `article_fuzzy`
- timing_ms: `{"deepseek": 0.013199999557400588, "resolution": 560.1485000006505}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.015`
- expected.DN: `None`
- expected.PN: `None`
- expected.connection: `None`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.015`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.015`
- returned_top5:
  - rank 1: `11с67п2цп000016015` | `MARSHAL` | DN `15.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_only_typo_0034` | `article_only_typo` | `GOLD_BUG`
- query: `01с67п2цп000016020`
- resolution_mode: `article_fuzzy`
- timing_ms: `{"deepseek": 0.022900000658410136, "resolution": 559.8423000001276}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.020`
- expected.DN: `None`
- expected.PN: `None`
- expected.connection: `None`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.020`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.020`
- returned_top5:
  - rank 1: `11с67п2цп000016020` | `MARSHAL` | DN `20.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_brand_typo_0038` | `brand_typo` | `GOLD_BUG`
- query: `Broem DN100 PN16 сварное`
- resolution_mode: `brand_fuzzy`
- timing_ms: `{"deepseek": 0.009800000043469481, "resolution": 0.02719999974942766, "embedding": 282.2665000003326, "qdrant": 124.9775000005684, "ranking": 0.9928999998010113, "total": 408.2739000004949}`
- expected.resolved_brand: `Broen`
- expected.resolved_article: `None`
- expected.DN: `100.0`
- expected.PN: `16.0`
- expected.connection: `сварное`
- eligible_competitor_articles: `1163065, а0029, а0216, а0468, а0609, а1545, а1695, а2519, а2541, кшг 70.102.100.а.16, кшг 71.102.100.б.16, кшг 71.112.100.а.16, кшг 73.102.100.а.16, кшг 73.112.100.а.16, кшг 78.102.100.б.16, кшг 78.106.100.б.16, кшг 79.102.100.б.16, кшг 79.112.100.б.16, кшн 20.102.100.а, кшн 21.102.100.б`
- eligible_competitor_articles_continued: `... (+11 more)`
- preferred_competitor_articles: `1163065, а0029, а0216, а0468, а0609, а1545, а1695, а2519, а2541, кшг 70.102.100.а.16, кшг 71.102.100.б.16, кшг 71.112.100.а.16, кшг 73.102.100.а.16, кшг 73.112.100.а.16, кшг 78.102.100.б.16, кшг 78.106.100.б.16, кшг 79.102.100.б.16, кшг 79.112.100.б.16, кшн 20.102.100.а, кшн 21.102.100.б`
- preferred_competitor_articles_continued: `... (+11 more)`
- returned_top5:
  - rank 1: `кшн31312100б` | `Broen` | DN `100.0` | PN `16.0` | `сварное` | score `None`
  - rank 2: `кшт60405100а16` | `Broen` | DN `100.0` | PN `16.0` | `сварное` | score `None`
  - rank 3: `кшн21312100б` | `Broen` | DN `100.0` | PN `16.0` | `сварное` | score `None`
  - rank 4: `кшг70102100а16` | `Broen` | DN `100.0` | PN `16.0` | `сварное` | score `None`
  - rank 5: `кшн31313100б` | `Broen` | DN `100.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_brand_typo_0040` | `brand_typo` | `GOLD_BUG`
- query: `Fortecaa DN15 PN40 фланцевое`
- resolution_mode: `brand_fuzzy`
- timing_ms: `{"deepseek": 0.01359999987471383, "resolution": 0.02700000004551839, "embedding": 270.0310999998692, "qdrant": 125.88840000080381, "ranking": 0.05860000055690762, "total": 396.01870000115014}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `40.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `107-5450, 107-5470`
- preferred_competitor_articles: `107-5450, 107-5470`
- returned_top5:
  - rank 1: `1075450` | `FORTECA` | DN `15.0` | PN `40.0` | `фланцевое` | score `None`
  - rank 2: `1075470` | `FORTECA` | DN `15.0` | PN `40.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_brand_typo_0041` | `brand_typo` | `GOLD_BUG`
- query: `Marsha DN15 PN16 сварное`
- resolution_mode: `brand_fuzzy`
- timing_ms: `{"deepseek": 0.01900000006571645, "resolution": 0.04059999992023222, "embedding": 223.19200000038109, "qdrant": 187.01439999949798, "ranking": 3.135400000246591, "total": 413.4014000001116}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `16.0`
- expected.connection: `сварное`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.015, 11с67п 2цп.00.0.025.015, 11с67п 2цп.00.0.040.015, 11с67п 2цп.00.1.016.015, 11с67п 2цп.00.1.025.015, 11с67п 2цп.00.1.040.015, 11с67п 2цп.00.10.016.015, 11с67п 2цп.00.10.025.015, 11с67п 2цп.00.10.040.015, 11с67п 2цп.00.6(7).016.015, 11с67п 2цп.00.6(7).025.015, 11с67п 2цп.00.6(7).040.015, 11с67п 2цп.00.6.016.015, 11с67п 2цп.00.6.025.015, 11с67п 2цп.00.6.040.015, 11с67п 2цп.00.7.016.015, 11с67п 2цп.00.7.025.015, 11с67п 2цп.00.7.040.015, 11с67п 2цп.00.9.016.015, 11с67п 2цп.00.9.025.015`
- eligible_competitor_articles_continued: `... (+128 more)`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.015, 11с67п 2цп.00.0.025.015, 11с67п 2цп.00.0.040.015, 11с67п 2цп.00.1.016.015, 11с67п 2цп.00.1.025.015, 11с67п 2цп.00.1.040.015, 11с67п 2цп.00.10.016.015, 11с67п 2цп.00.10.025.015, 11с67п 2цп.00.10.040.015, 11с67п 2цп.00.6(7).016.015, 11с67п 2цп.00.6(7).025.015, 11с67п 2цп.00.6(7).040.015, 11с67п 2цп.00.6.016.015, 11с67п 2цп.00.6.025.015, 11с67п 2цп.00.6.040.015, 11с67п 2цп.00.7.016.015, 11с67п 2цп.00.7.025.015, 11с67п 2цп.00.7.040.015, 11с67п 2цп.00.9.016.015, 11с67п 2цп.00.9.025.015`
- preferred_competitor_articles_continued: `... (+128 more)`
- returned_top5:
  - rank 1: `11с67п2цп000016015` | `MARSHAL` | DN `15.0` | PN `16.0` | `сварное` | score `None`
  - rank 2: `11с67п2цп017016015` | `MARSHAL` | DN `15.0` | PN `16.0` | `сварное` | score `None`
  - rank 3: `11с67п2цп007016015` | `MARSHAL` | DN `15.0` | PN `16.0` | `сварное` | score `None`
  - rank 4: `11с67п2цп009016015` | `MARSHAL` | DN `15.0` | PN `16.0` | `сварное` | score `None`
  - rank 5: `11с67п2цп006016015` | `MARSHAL` | DN `15.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_brand_typo_0042` | `brand_typo` | `GOLD_BUG`
- query: `Marsha DN20 PN16 сварное`
- resolution_mode: `brand_fuzzy`
- timing_ms: `{"deepseek": 0.020599999515980016, "resolution": 0.042100000428035855, "embedding": 357.14629999984027, "qdrant": 201.9774000000325, "ranking": 3.5219000001234235, "total": 562.7082999999402}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `None`
- expected.DN: `20.0`
- expected.PN: `16.0`
- expected.connection: `сварное`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.020, 11с67п 2цп.00.0.025.020, 11с67п 2цп.00.0.040.020, 11с67п 2цп.00.1.016.020, 11с67п 2цп.00.1.025.020, 11с67п 2цп.00.1.040.020, 11с67п 2цп.00.10.016.020, 11с67п 2цп.00.10.025.020, 11с67п 2цп.00.10.040.020, 11с67п 2цп.00.6(7).016.020, 11с67п 2цп.00.6(7).025.020, 11с67п 2цп.00.6(7).040.020, 11с67п 2цп.00.6.016.020, 11с67п 2цп.00.6.025.020, 11с67п 2цп.00.6.040.020, 11с67п 2цп.00.7.016.020, 11с67п 2цп.00.7.025.020, 11с67п 2цп.00.7.040.020, 11с67п 2цп.00.9.016.020, 11с67п 2цп.00.9.025.020`
- eligible_competitor_articles_continued: `... (+151 more)`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.020, 11с67п 2цп.00.0.025.020, 11с67п 2цп.00.0.040.020, 11с67п 2цп.00.1.016.020, 11с67п 2цп.00.1.025.020, 11с67п 2цп.00.1.040.020, 11с67п 2цп.00.10.016.020, 11с67п 2цп.00.10.025.020, 11с67п 2цп.00.10.040.020, 11с67п 2цп.00.6(7).016.020, 11с67п 2цп.00.6(7).025.020, 11с67п 2цп.00.6(7).040.020, 11с67п 2цп.00.6.016.020, 11с67п 2цп.00.6.025.020, 11с67п 2цп.00.6.040.020, 11с67п 2цп.00.7.016.020, 11с67п 2цп.00.7.025.020, 11с67п 2цп.00.7.040.020, 11с67п 2цп.00.9.016.020, 11с67п 2цп.00.9.025.020`
- preferred_competitor_articles_continued: `... (+151 more)`
- returned_top5:
  - rank 1: `11с67пцп006016020` | `MARSHAL` | DN `20.0` | PN `16.0` | `сварное` | score `None`
  - rank 2: `11с67п2цп000016020` | `MARSHAL` | DN `20.0` | PN `16.0` | `сварное` | score `None`
  - rank 3: `11с67п2цп006016020` | `MARSHAL` | DN `20.0` | PN `16.0` | `сварное` | score `None`
  - rank 4: `11с67п2цп007016020` | `MARSHAL` | DN `20.0` | PN `16.0` | `сварное` | score `None`
  - rank 5: `11с67п2цп011016020` | `MARSHAL` | DN `20.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_brand_plus_article_0050` | `brand_plus_article` | `GOLD_BUG`
- query: `FORTECA 107-5450 DN15 PN40 фланцевое`
- resolution_mode: `brand_and_article`
- timing_ms: `{"deepseek": 0.014999999621068127, "resolution": 531.6967000007935}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `107-5450`
- expected.DN: `15.0`
- expected.PN: `40.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `107-5450`
- preferred_competitor_articles: `107-5450`
- returned_top5:
  - rank 1: `1075450` | `FORTECA` | DN `15.0` | PN `40.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_brand_plus_article_0051` | `brand_plus_article` | `GOLD_BUG`
- query: `MARSHAL 11с67п 2ЦП.00.0.016.015 DN15 PN16 сварное`
- resolution_mode: `brand_and_article`
- timing_ms: `{"deepseek": 0.01939999947353499, "resolution": 481.0852000000523}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.015`
- expected.DN: `15.0`
- expected.PN: `16.0`
- expected.connection: `сварное`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.015`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.015`
- returned_top5:
  - rank 1: `11с67п2цп000016015` | `MARSHAL` | DN `15.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_brand_plus_article_0052` | `brand_plus_article` | `GOLD_BUG`
- query: `MARSHAL 11с67п 2ЦП.00.0.016.020 DN20 PN16 сварное`
- resolution_mode: `brand_and_article`
- timing_ms: `{"deepseek": 0.019400000383029692, "resolution": 483.78820000016276}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.020`
- expected.DN: `20.0`
- expected.PN: `16.0`
- expected.connection: `сварное`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.020`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.020`
- returned_top5:
  - rank 1: `11с67п2цп000016020` | `MARSHAL` | DN `20.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_plus_hard_0058` | `article_plus_hard` | `GOLD_BUG`
- query: `107-5450 DN15 PN40 фланцевое`
- resolution_mode: `article_exact`
- timing_ms: `{"deepseek": 0.01269999938813271, "resolution": 490.27029999979277}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `107-5450`
- expected.DN: `15.0`
- expected.PN: `40.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `107-5450`
- preferred_competitor_articles: `107-5450`
- returned_top5:
  - rank 1: `1075450` | `FORTECA` | DN `15.0` | PN `40.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_plus_hard_0062` | `article_plus_hard` | `GOLD_BUG`
- query: `11с67п 2ЦП.00.0.016.015 DN15 PN16 сварное`
- resolution_mode: `article_exact`
- timing_ms: `{"deepseek": 0.014299999747890979, "resolution": 504.6362000002773}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.015`
- expected.DN: `15.0`
- expected.PN: `16.0`
- expected.connection: `сварное`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.015`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.015`
- returned_top5:
  - rank 1: `11с67п2цп000016015` | `MARSHAL` | DN `15.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_plus_hard_0063` | `article_plus_hard` | `GOLD_BUG`
- query: `11с67п 2ЦП.00.0.016.020 DN20 PN16 сварное`
- resolution_mode: `article_exact`
- timing_ms: `{"deepseek": 0.01609999981155852, "resolution": 490.6701999998404}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.020`
- expected.DN: `20.0`
- expected.PN: `16.0`
- expected.connection: `сварное`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.020`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.020`
- returned_top5:
  - rank 1: `11с67п2цп000016020` | `MARSHAL` | DN `20.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_natural_language_0068` | `article_natural_language` | `GOLD_BUG`
- query: `Нужен аналог для 107-5450 FORTECA DN15 PN40 фланцевое`
- resolution_mode: `brand_and_article`
- timing_ms: `{"deepseek": 0.0152000002344721, "resolution": 470.43419999954494}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `107-5450`
- expected.DN: `15.0`
- expected.PN: `40.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `107-5450`
- preferred_competitor_articles: `107-5450`
- returned_top5:
  - rank 1: `1075450` | `FORTECA` | DN `15.0` | PN `40.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_natural_language_0072` | `article_natural_language` | `GOLD_BUG`
- query: `Нужен аналог для 11с67п 2ЦП.00.0.016.015 MARSHAL DN15 PN16 сварное`
- resolution_mode: `brand_and_article`
- timing_ms: `{"deepseek": 0.015199999324977398, "resolution": 480.53540000000794}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.015`
- expected.DN: `15.0`
- expected.PN: `16.0`
- expected.connection: `сварное`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.015`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.015`
- returned_top5:
  - rank 1: `11с67п2цп000016015` | `MARSHAL` | DN `15.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_article_natural_language_0073` | `article_natural_language` | `GOLD_BUG`
- query: `Нужен аналог для 11с67п 2ЦП.00.0.016.020 MARSHAL DN20 PN16 сварное`
- resolution_mode: `brand_and_article`
- timing_ms: `{"deepseek": 0.01869999960035784, "resolution": 507.88959999954386}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `11с67п 2ЦП.00.0.016.020`
- expected.DN: `20.0`
- expected.PN: `16.0`
- expected.connection: `сварное`
- eligible_competitor_articles: `11с67п 2цп.00.0.016.020`
- preferred_competitor_articles: `11с67п 2цп.00.0.016.020`
- returned_top5:
  - rank 1: `11с67п2цп000016020` | `MARSHAL` | DN `20.0` | PN `16.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_pn_minimum_semantics_0091` | `pn_minimum_semantics` | `GOLD_BUG`
- query: `ALSO DN15 PN25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.02500000027794158, "resolution": 0.02949999998236308, "embedding": 266.1117000006925, "qdrant": 161.46430000026157, "ranking": 3.642099999524362, "total": 431.27260000073875}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- eligible_competitor_articles_continued: `... (+171 more)`
- preferred_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- preferred_competitor_articles_continued: `... (+171 more)`
- returned_top5:
  - rank 1: `кшкп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 2: `кшк0152501` | `ALSO` | DN `15.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `кшп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `сварное` | score `None`
  - rank 4: `кшкп0152502` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 5: `кшкпр0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_pn_minimum_semantics_0092` | `pn_minimum_semantics` | `GOLD_BUG`
- query: `ALSO DN15 PN25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.018900000213761814, "resolution": 0.01939999947353499, "embedding": 290.07050000018353, "qdrant": 175.3236000004108, "ranking": 3.8500000000567525, "total": 469.2824000003384}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- eligible_competitor_articles_continued: `... (+171 more)`
- preferred_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- preferred_competitor_articles_continued: `... (+171 more)`
- returned_top5:
  - rank 1: `кшкп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 2: `кшк0152501` | `ALSO` | DN `15.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `кшп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `сварное` | score `None`
  - rank 4: `кшкп0152502` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 5: `кшкпр0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_pn_minimum_semantics_0093` | `pn_minimum_semantics` | `GOLD_BUG`
- query: `ALSO DN15 PN25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.021899999410379678, "resolution": 0.019499999325489625, "embedding": 281.0245999999097, "qdrant": 167.35679999965214, "ranking": 4.081299999597832, "total": 452.50409999789554}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- eligible_competitor_articles_continued: `... (+171 more)`
- preferred_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- preferred_competitor_articles_continued: `... (+171 more)`
- returned_top5:
  - rank 1: `кшкп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 2: `кшк0152501` | `ALSO` | DN `15.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `кшп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `сварное` | score `None`
  - rank 4: `кшкп0152502` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 5: `кшкпр0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_pn_minimum_semantics_0094` | `pn_minimum_semantics` | `GOLD_BUG`
- query: `ALSO DN15 PN25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.019099999917671084, "resolution": 0.021300000298651867, "embedding": 271.9959000005474, "qdrant": 187.6574999996592, "ranking": 4.118900000321446, "total": 463.81270000074437}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- eligible_competitor_articles_continued: `... (+171 more)`
- preferred_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- preferred_competitor_articles_continued: `... (+171 more)`
- returned_top5:
  - rank 1: `кшкп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 2: `кшк0152501` | `ALSO` | DN `15.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `кшп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `сварное` | score `None`
  - rank 4: `кшкп0152502` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 5: `кшкпр0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_pn_minimum_semantics_0095` | `pn_minimum_semantics` | `GOLD_BUG`
- query: `ALSO DN15 PN25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.016400000276917126, "resolution": 0.014600000213249587, "embedding": 229.49900000003254, "qdrant": 219.63070000037987, "ranking": 4.12770000002638, "total": 453.28840000092896}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- eligible_competitor_articles_continued: `... (+171 more)`
- preferred_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- preferred_competitor_articles_continued: `... (+171 more)`
- returned_top5:
  - rank 1: `кшкп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 2: `кшк0152501` | `ALSO` | DN `15.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `кшп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `сварное` | score `None`
  - rank 4: `кшкп0152502` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 5: `кшкпр0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_pn_minimum_semantics_0096` | `pn_minimum_semantics` | `GOLD_BUG`
- query: `ALSO DN15 PN25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.019299999621580355, "resolution": 0.01939999947353499, "embedding": 278.5665000001245, "qdrant": 191.1350999998831, "ranking": 4.447799999979907, "total": 474.1880999990826}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- eligible_competitor_articles_continued: `... (+171 more)`
- preferred_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- preferred_competitor_articles_continued: `... (+171 more)`
- returned_top5:
  - rank 1: `кшкп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 2: `кшк0152501` | `ALSO` | DN `15.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `кшп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `сварное` | score `None`
  - rank 4: `кшкп0152502` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 5: `кшкпр0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_pn_minimum_semantics_0097` | `pn_minimum_semantics` | `GOLD_BUG`
- query: `ALSO DN15 PN25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.02029999996011611, "resolution": 0.019600000086938962, "embedding": 281.83739999985846, "qdrant": 165.66229999989446, "ranking": 3.7505999998757034, "total": 451.2901999996757}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- eligible_competitor_articles_continued: `... (+171 more)`
- preferred_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- preferred_competitor_articles_continued: `... (+171 more)`
- returned_top5:
  - rank 1: `кшкп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 2: `кшк0152501` | `ALSO` | DN `15.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `кшп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `сварное` | score `None`
  - rank 4: `кшкп0152502` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 5: `кшкпр0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_pn_minimum_semantics_0098` | `pn_minimum_semantics` | `GOLD_BUG`
- query: `ALSO DN15 PN25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.014700000065204222, "resolution": 0.014899999769113492, "embedding": 217.03750000051514, "qdrant": 181.17550000079063, "ranking": 4.199400000288733, "total": 402.4420000014288}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- eligible_competitor_articles_continued: `... (+171 more)`
- preferred_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- preferred_competitor_articles_continued: `... (+171 more)`
- returned_top5:
  - rank 1: `кшкп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 2: `кшк0152501` | `ALSO` | DN `15.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `кшп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `сварное` | score `None`
  - rank 4: `кшкп0152502` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 5: `кшкпр0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_pn_minimum_semantics_0099` | `pn_minimum_semantics` | `GOLD_BUG`
- query: `ALSO DN15 PN25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.020200000108161476, "resolution": 0.01990000055229757, "embedding": 249.20529999963037, "qdrant": 167.11169999962294, "ranking": 3.2234000000244123, "total": 419.5804999999382}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- eligible_competitor_articles_continued: `... (+171 more)`
- preferred_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- preferred_competitor_articles_continued: `... (+171 more)`
- returned_top5:
  - rank 1: `кшкп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 2: `кшк0152501` | `ALSO` | DN `15.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `кшп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `сварное` | score `None`
  - rank 4: `кшкп0152502` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 5: `кшкпр0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_pn_minimum_semantics_0100` | `pn_minimum_semantics` | `GOLD_BUG`
- query: `ALSO DN15 PN25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.02029999996011611, "resolution": 0.019799999790848233, "embedding": 277.78180000041175, "qdrant": 172.14980000062496, "ranking": 4.288899999664864, "total": 454.26060000045254}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `15.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- eligible_competitor_articles_continued: `... (+171 more)`
- preferred_competitor_articles: `110150401mva00000, 110150401rva00000, 110150403mva00000, 110151401rva00000, 120150401mva00000, 120150401rva00000, 120150403mva00000, 120151401rva00000, 130150401mva00000, 130150403mva00000, 1468244, 1644203, 2034545, 210150401mva00000, 2116743, 2151667, 220150401mva00000, 2481115, 310150402mva00000, 310151402mva00000`
- preferred_competitor_articles_continued: `... (+171 more)`
- returned_top5:
  - rank 1: `кшкп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 2: `кшк0152501` | `ALSO` | DN `15.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `кшп0152501` | `ALSO` | DN `15.0` | PN `25.0` | `сварное` | score `None`
  - rank 4: `кшкп0152502` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
  - rank 5: `кшкпр0152501` | `ALSO` | DN `15.0` | PN `25.0` | `резьбовое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0102` | `v3_regression` | `GOLD_BUG`
- query: `ALSO DN50 PN40 штуцерное`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.010400000064691994, "resolution": 0.011799999811046291, "embedding": 249.0975000000617, "qdrant": 158.1331000006685, "ranking": 0.26509999952395447, "total": 407.5179000001299}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `50.0`
- expected.PN: `40.0`
- expected.connection: `штуцерное`
- eligible_competitor_articles: `кш.ш.050.40-01, кш.ш.050.40-02, кш.ш.gas.050.40-01, кш.ш.gas.050.40-02, кш.ш.rs.050.40-02, кш.шп.050.40-01, кш.шп.050.40-02, кш.шп.gas.050.40-01, кш.шп.gas.050.40-02, кш.шп.rs.050.40-02`
- preferred_competitor_articles: `кш.ш.050.40-01, кш.ш.050.40-02, кш.ш.gas.050.40-01, кш.ш.gas.050.40-02, кш.ш.rs.050.40-02, кш.шп.050.40-01, кш.шп.050.40-02, кш.шп.gas.050.40-01, кш.шп.gas.050.40-02, кш.шп.rs.050.40-02`
- returned_top5:
  - rank 1: `кшш0504001` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 2: `кшш0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 3: `кшшп0504001` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 4: `кшшп0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 5: `кшшпgas0504001` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0103` | `v3_regression` | `GOLD_BUG`
- query: `ALSO DN50 PN40 штуцерное КШ.ШП.RS.050.40-02`
- resolution_mode: `brand_and_article`
- timing_ms: `{"deepseek": 0.025099999220401514, "resolution": 486.9983999997203}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `КШ.ШП.RS.050.40-02`
- expected.DN: `50.0`
- expected.PN: `40.0`
- expected.connection: `штуцерное`
- eligible_competitor_articles: `кш.шп.rs.050.40-02`
- preferred_competitor_articles: `кш.шп.rs.050.40-02`
- returned_top5:
  - rank 1: `кшшпrs0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0104` | `v3_regression` | `GOLD_BUG`
- query: `ALSO DN50 PN40 штуцерное жидкость`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.02299999960087007, "resolution": 0.022500000341096893, "embedding": 218.51259999948525, "qdrant": 165.598399999908, "ranking": 0.15549999989161734, "total": 384.3119999992268}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `50.0`
- expected.PN: `40.0`
- expected.connection: `штуцерное`
- eligible_competitor_articles: `кш.ш.050.40-01, кш.ш.050.40-02, кш.ш.gas.050.40-01, кш.ш.gas.050.40-02, кш.ш.rs.050.40-02, кш.шп.050.40-01, кш.шп.050.40-02, кш.шп.gas.050.40-01, кш.шп.gas.050.40-02, кш.шп.rs.050.40-02`
- preferred_competitor_articles: `кш.ш.050.40-01, кш.ш.050.40-02, кш.ш.rs.050.40-02, кш.шп.050.40-01, кш.шп.050.40-02, кш.шп.rs.050.40-02`
- returned_top5:
  - rank 1: `кшш0504001` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 2: `кшшп0504001` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 3: `кшшп0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 4: `кшш0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 5: `кшшrs0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0105` | `v3_regression` | `GOLD_BUG`
- query: `ALSO DN50 PN40 штуцерное сталь 09г2с`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.017299999854003545, "resolution": 0.015900000107649248, "embedding": 267.9556000002776, "qdrant": 156.00129999984347, "ranking": 0.24530000064260093, "total": 424.23540000072535}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `50.0`
- expected.PN: `40.0`
- expected.connection: `штуцерное`
- eligible_competitor_articles: `кш.ш.050.40-01, кш.ш.050.40-02, кш.ш.gas.050.40-01, кш.ш.gas.050.40-02, кш.ш.rs.050.40-02, кш.шп.050.40-01, кш.шп.050.40-02, кш.шп.gas.050.40-01, кш.шп.gas.050.40-02, кш.шп.rs.050.40-02`
- preferred_competitor_articles: `кш.ш.050.40-02, кш.ш.gas.050.40-02, кш.ш.rs.050.40-02, кш.шп.050.40-02, кш.шп.gas.050.40-02, кш.шп.rs.050.40-02`
- returned_top5:
  - rank 1: `кшш0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 2: `кшшп0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 3: `кшшrs0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 4: `кшшgas0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 5: `кшшпgas0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0106` | `v3_regression` | `GOLD_BUG`
- query: `ALSO DN50 PN40 штуцерное сталь 09г2с жидкость ручное -60...200`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.020899999981338624, "resolution": 0.02009999934671214, "embedding": 224.35099999984232, "qdrant": 160.2247000000716, "ranking": 0.1793000001271139, "total": 384.7959999993691}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `50.0`
- expected.PN: `40.0`
- expected.connection: `штуцерное`
- eligible_competitor_articles: `кш.ш.050.40-01, кш.ш.050.40-02, кш.ш.gas.050.40-01, кш.ш.gas.050.40-02, кш.ш.rs.050.40-02, кш.шп.050.40-01, кш.шп.050.40-02, кш.шп.gas.050.40-01, кш.шп.gas.050.40-02, кш.шп.rs.050.40-02`
- preferred_competitor_articles: `кш.ш.050.40-02, кш.ш.rs.050.40-02, кш.шп.050.40-02, кш.шп.rs.050.40-02`
- returned_top5:
  - rank 1: `кшш0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 2: `кшшп0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 3: `кшшrs0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 4: `кшшпrs0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 5: `кшшgas0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0107` | `v3_regression` | `GOLD_BUG`
- query: `ALSO DN50 штуцерное`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.020500000573520083, "resolution": 0.023100000362319406, "embedding": 217.7551000004314, "qdrant": 158.684699999867, "ranking": 0.1627000001462875, "total": 376.64610000138055}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `50.0`
- expected.PN: `None`
- expected.connection: `штуцерное`
- eligible_competitor_articles: `кш.ш.050.40-01, кш.ш.050.40-02, кш.ш.gas.050.40-01, кш.ш.gas.050.40-02, кш.ш.rs.050.40-02, кш.шп.050.40-01, кш.шп.050.40-02, кш.шп.gas.050.40-01, кш.шп.gas.050.40-02, кш.шп.rs.050.40-02`
- preferred_competitor_articles: `кш.ш.050.40-01, кш.ш.050.40-02, кш.ш.gas.050.40-01, кш.ш.gas.050.40-02, кш.ш.rs.050.40-02, кш.шп.050.40-01, кш.шп.050.40-02, кш.шп.gas.050.40-01, кш.шп.gas.050.40-02, кш.шп.rs.050.40-02`
- returned_top5:
  - rank 1: `кшш0504001` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 2: `кшш0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 3: `кшшп0504001` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 4: `кшшп0504002` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 5: `кшшпgas0504001` | `ALSO` | DN `50.0` | PN `40.0` | `штуцерное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0108` | `v3_regression` | `GOLD_BUG`
- query: `ALSO PN40 штуцерное`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.021099999685247894, "resolution": 0.021600000764010474, "embedding": 221.80350000053295, "qdrant": 162.3239000000467, "ranking": 1.5197000002444838, "total": 385.6898000012734}`
- expected.resolved_brand: `ALSO`
- expected.resolved_article: `None`
- expected.DN: `None`
- expected.PN: `40.0`
- expected.connection: `штуцерное`
- eligible_competitor_articles: `кш.ш.015.40-01, кш.ш.015.40-02, кш.ш.020.40-01, кш.ш.020.40-02, кш.ш.025.40-01, кш.ш.025.40-02, кш.ш.032.40-01, кш.ш.032.40-02, кш.ш.040.40-01, кш.ш.040.40-02, кш.ш.050.40-01, кш.ш.050.40-02, кш.ш.gas.015.40-01, кш.ш.gas.015.40-02, кш.ш.gas.020.40-01, кш.ш.gas.020.40-02, кш.ш.gas.025.40-01, кш.ш.gas.025.40-02, кш.ш.gas.032.40-01, кш.ш.gas.032.40-02`
- eligible_competitor_articles_continued: `... (+45 more)`
- preferred_competitor_articles: `кш.ш.015.40-01, кш.ш.015.40-02, кш.ш.020.40-01, кш.ш.020.40-02, кш.ш.025.40-01, кш.ш.025.40-02, кш.ш.032.40-01, кш.ш.032.40-02, кш.ш.040.40-01, кш.ш.040.40-02, кш.ш.050.40-01, кш.ш.050.40-02, кш.ш.gas.015.40-01, кш.ш.gas.015.40-02, кш.ш.gas.020.40-01, кш.ш.gas.020.40-02, кш.ш.gas.025.40-01, кш.ш.gas.025.40-02, кш.ш.gas.032.40-01, кш.ш.gas.032.40-02`
- preferred_competitor_articles_continued: `... (+45 more)`
- returned_top5:
  - rank 1: `кшш0204002` | `ALSO` | DN `20.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 2: `кшшпgas0404001` | `ALSO` | DN `40.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 3: `кшш0254002` | `ALSO` | DN `25.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 4: `кшшgas0404001` | `ALSO` | DN `40.0` | PN `40.0` | `штуцерное` | score `None`
  - rank 5: `кшш0404001` | `ALSO` | DN `40.0` | PN `40.0` | `штуцерное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0109` | `v3_regression` | `GOLD_BUG`
- query: `Broen DN125 PN16`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.01700000029813964, "resolution": 0.014600000213249587, "embedding": 283.3079000001817, "qdrant": 124.82300000010582, "ranking": 1.5587999996569124, "total": 409.7213000004558}`
- expected.resolved_brand: `Broen`
- expected.resolved_article: `None`
- expected.DN: `125.0`
- expected.PN: `16.0`
- expected.connection: `None`
- eligible_competitor_articles: `а0030, а0069, а0080, а0217, а0234, а0300, а0469, а0476, а0491, а0495, а0498, а0501, а0580, а0620, а0628, а1026, а1546, а1696, а1707, а2520`
- eligible_competitor_articles_continued: `... (+40 more)`
- preferred_competitor_articles: `а0030, а0069, а0080, а0217, а0234, а0300, а0469, а0476, а0491, а0495, а0498, а0501, а0580, а0620, а0628, а1026, а1546, а1696, а1707, а2520`
- preferred_competitor_articles_continued: `... (+40 more)`
- returned_top5:
  - rank 1: `кшн21113125б` | `Broen` | DN `125.0` | PN `16.0` | `фланцевое` | score `None`
  - rank 2: `кшн31312125б` | `Broen` | DN `125.0` | PN `16.0` | `сварное` | score `None`
  - rank 3: `кшн51313125б` | `Broen` | DN `125.0` | PN `16.0` | `фланцевое` | score `None`
  - rank 4: `кшн21103125б` | `Broen` | DN `125.0` | PN `16.0` | `фланцевое` | score `None`
  - rank 5: `кшн21313125б` | `Broen` | DN `125.0` | PN `16.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0110` | `v3_regression` | `GOLD_BUG`
- query: `Broen DN125 PN16 фланцевое`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.020599999515980016, "resolution": 0.01880000036180718, "embedding": 225.70859999996173, "qdrant": 128.6095999994359, "ranking": 0.7126999998945394, "total": 355.07029999916995}`
- expected.resolved_brand: `Broen`
- expected.resolved_article: `None`
- expected.DN: `125.0`
- expected.PN: `16.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `а0069, а0080, а0234, а0491, а0495, а0498, а0501, а0620, а0628, а1707, а2531, а3085, г9490, кшг 71.103.125.а.16, кшг 71.113.125.а.16, кшг 71.413.125.а.16, кшг 73.103.125.а.16, кшг 73.113.125.а.16, кшн 21.103.125.б, кшн 21.113.125.б`
- eligible_competitor_articles_continued: `... (+11 more)`
- preferred_competitor_articles: `а0069, а0080, а0234, а0491, а0495, а0498, а0501, а0620, а0628, а1707, а2531, а3085, г9490, кшг 71.103.125.а.16, кшг 71.113.125.а.16, кшг 71.413.125.а.16, кшг 73.103.125.а.16, кшг 73.113.125.а.16, кшн 21.103.125.б, кшн 21.113.125.б`
- preferred_competitor_articles_continued: `... (+11 more)`
- returned_top5:
  - rank 1: `кшн51313125б` | `Broen` | DN `125.0` | PN `16.0` | `фланцевое` | score `None`
  - rank 2: `кшн21113125б` | `Broen` | DN `125.0` | PN `16.0` | `фланцевое` | score `None`
  - rank 3: `кшн21103125б` | `Broen` | DN `125.0` | PN `16.0` | `фланцевое` | score `None`
  - rank 4: `кшн21313125б` | `Broen` | DN `125.0` | PN `16.0` | `фланцевое` | score `None`
  - rank 5: `кшт61413125а16` | `Broen` | DN `125.0` | PN `16.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0117` | `v3_regression` | `GOLD_BUG`
- query: `FORTECA DN250 PN25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.011000000085914508, "resolution": 0.013100000614940654, "embedding": 270.592900000338, "qdrant": 139.65900000039255, "ranking": 0.22389999958249973, "total": 410.4999000010139}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `None`
- expected.DN: `250.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `107-5469, 107-5496, 107-5506, 107-5520, 107-5536, 107-5541, 107-6965`
- preferred_competitor_articles: `107-5469, 107-5496, 107-5506, 107-5520, 107-5536, 107-5541, 107-6965`
- returned_top5:
  - rank 1: `1075469` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 2: `1075520` | `FORTECA` | DN `250.0` | PN `25.0` | `сварное` | score `None`
  - rank 3: `1075536` | `FORTECA` | DN `250.0` | PN `25.0` | `сварное` | score `None`
  - rank 4: `1075506` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 5: `1075496` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0118` | `v3_regression` | `GOLD_BUG`
- query: `FORTECA DN250 PN25 фланцевое`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.023599999622092582, "resolution": 0.03459999970800709, "embedding": 242.33499999991182, "qdrant": 115.60649999955785, "ranking": 0.13209999997343402, "total": 358.1317999987732}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `None`
- expected.DN: `250.0`
- expected.PN: `25.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `107-5469, 107-5496, 107-5506, 107-6965`
- preferred_competitor_articles: `107-5469, 107-5496, 107-5506, 107-6965`
- returned_top5:
  - rank 1: `1075469` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 2: `1075506` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `1075496` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 4: `1076965` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0119` | `v3_regression` | `GOLD_BUG`
- query: `FORTECA DN250 PN25 фланцевое 107-6965`
- resolution_mode: `brand_and_article`
- timing_ms: `{"deepseek": 0.01920000067912042, "resolution": 552.4041000007855}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `107-6965`
- expected.DN: `250.0`
- expected.PN: `25.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `107-6965`
- preferred_competitor_articles: `107-6965`
- returned_top5:
  - rank 1: `1076965` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0120` | `v3_regression` | `GOLD_BUG`
- query: `FORTECA DN250 PN25 фланцевое жидкость`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.019999999494757503, "resolution": 0.021799999558425043, "embedding": 272.25580000049376, "qdrant": 143.03500000005442, "ranking": 0.1145999995060265, "total": 415.4471999991074}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `None`
- expected.DN: `250.0`
- expected.PN: `25.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `107-5469, 107-5496, 107-5506, 107-6965`
- preferred_competitor_articles: `107-5469, 107-5496, 107-5506, 107-6965`
- returned_top5:
  - rank 1: `1075469` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 2: `1075506` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `1075496` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 4: `1076965` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0121` | `v3_regression` | `GOLD_BUG`
- query: `FORTECA DN250 PN25 фланцевое сталь 20`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.01900000006571645, "resolution": 0.01809999957913533, "embedding": 266.1086999996769, "qdrant": 114.97259999941889, "ranking": 0.09249999948224286, "total": 381.2108999982229}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `None`
- expected.DN: `250.0`
- expected.PN: `25.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `107-5469, 107-5496, 107-5506, 107-6965`
- preferred_competitor_articles: `107-5469, 107-5496, 107-5506, 107-6965`
- returned_top5:
  - rank 1: `1075469` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 2: `1075496` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `1075506` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 4: `1076965` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0122` | `v3_regression` | `GOLD_BUG`
- query: `FORTECA DN250 PN25 фланцевое сталь 20 жидкость редуктор -40...200`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.02170000061596511, "resolution": 0.02210000002378365, "embedding": 214.74059999945894, "qdrant": 124.66349999976956, "ranking": 0.13339999986783369, "total": 339.5812999997361}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `None`
- expected.DN: `250.0`
- expected.PN: `25.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `107-5469, 107-5496, 107-5506, 107-6965`
- preferred_competitor_articles: `107-5496, 107-5506, 107-6965`
- returned_top5:
  - rank 1: `1075496` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 2: `1075506` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `1076965` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 4: `1075469` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0123` | `v3_regression` | `GOLD_BUG`
- query: `FORTECA DN250 фланцевое`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.014400000509340316, "resolution": 0.013700000636163168, "embedding": 273.9499999997861, "qdrant": 118.14529999992374, "ranking": 0.29799999992974335, "total": 392.42140000078507}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `None`
- expected.DN: `250.0`
- expected.PN: `None`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `107-5462, 107-5469, 107-5491, 107-5496, 107-5501, 107-5506, 107-6964, 107-6965`
- preferred_competitor_articles: `107-5462, 107-5469, 107-5491, 107-5496, 107-5501, 107-5506, 107-6964, 107-6965`
- returned_top5:
  - rank 1: `1075469` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 2: `1075462` | `FORTECA` | DN `250.0` | PN `16.0` | `фланцевое` | score `None`
  - rank 3: `1076965` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 4: `1075506` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 5: `1075491` | `FORTECA` | DN `250.0` | PN `16.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0124` | `v3_regression` | `GOLD_BUG`
- query: `FORTECA PN25 фланцевое`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.020300000869610813, "resolution": 0.01920000067912042, "embedding": 273.90899999954854, "qdrant": 124.09920000027341, "ranking": 1.064099999894097, "total": 399.1118000012648}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `None`
- expected.DN: `None`
- expected.PN: `25.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `107-5450, 107-5451, 107-5452, 107-5453, 107-5454, 107-5455, 107-5463, 107-5464, 107-5465, 107-5466, 107-5467, 107-5468, 107-5469, 107-5470, 107-5471, 107-5472, 107-5473, 107-5474, 107-5475, 107-5482`
- eligible_competitor_articles_continued: `... (+17 more)`
- preferred_competitor_articles: `107-5450, 107-5451, 107-5452, 107-5453, 107-5454, 107-5455, 107-5463, 107-5464, 107-5465, 107-5466, 107-5467, 107-5468, 107-5469, 107-5470, 107-5471, 107-5472, 107-5473, 107-5474, 107-5475, 107-5482`
- preferred_competitor_articles_continued: `... (+17 more)`
- returned_top5:
  - rank 1: `1075482` | `FORTECA` | DN `65.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 2: `1075464` | `FORTECA` | DN `80.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 3: `1075463` | `FORTECA` | DN `65.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 4: `1075472` | `FORTECA` | DN `25.0` | PN `40.0` | `фланцевое` | score `None`
  - rank 5: `1075467` | `FORTECA` | DN `150.0` | PN `25.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0127` | `v3_regression` | `GOLD_BUG`
- query: `MARSHAL DN40 PN40 фланцевое Цф.00.1.040.040`
- resolution_mode: `brand_and_article`
- timing_ms: `{"deepseek": 0.009700000191514846, "resolution": 515.909400000055}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `Цф.00.1.040.040`
- expected.DN: `40.0`
- expected.PN: `40.0`
- expected.connection: `фланцевое`
- eligible_competitor_articles: `цф.00.1.040.040`
- preferred_competitor_articles: `цф.00.1.040.040`
- returned_top5:
  - rank 1: `цф001040040` | `MARSHAL` | DN `40.0` | PN `40.0` | `фланцевое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0144` | `v3_regression` | `GOLD_BUG`
- query: `forteca Ду250 Ру25`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.01559999964229064, "resolution": 0.017700000171316788, "embedding": 266.78870000068855, "qdrant": 131.82839999990392, "ranking": 0.21949999973003287, "total": 398.8699000001361}`
- expected.resolved_brand: `FORTECA`
- expected.resolved_article: `None`
- expected.DN: `250.0`
- expected.PN: `25.0`
- expected.connection: `None`
- eligible_competitor_articles: `107-5469, 107-5496, 107-5506, 107-5520, 107-5536, 107-5541, 107-6965`
- preferred_competitor_articles: `107-5469, 107-5496, 107-5506, 107-5520, 107-5536, 107-5541, 107-6965`
- returned_top5:
  - rank 1: `1075469` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 2: `1075520` | `FORTECA` | DN `250.0` | PN `25.0` | `сварное` | score `None`
  - rank 3: `1075506` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 4: `1075496` | `FORTECA` | DN `250.0` | PN `25.0` | `фланцевое` | score `None`
  - rank 5: `1075536` | `FORTECA` | DN `250.0` | PN `25.0` | `сварное` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`

### `v4_v3_regression_0145` | `v3_regression` | `GOLD_BUG`
- query: `marshal Ду40 Ру40`
- resolution_mode: `brand_exact`
- timing_ms: `{"deepseek": 0.02029999996011611, "resolution": 0.019600000086938962, "embedding": 237.15619999984483, "qdrant": 179.0010999993683, "ranking": 4.359399999884772, "total": 420.55659999914496}`
- expected.resolved_brand: `MARSHAL`
- expected.resolved_article: `None`
- expected.DN: `40.0`
- expected.PN: `40.0`
- expected.connection: `None`
- eligible_competitor_articles: `11с67п 2цп.00.0.040.040, 11с67п 2цп.00.1.040.040, 11с67п 2цп.00.10.040.040, 11с67п 2цп.00.3.040.040, 11с67п 2цп.00.6(7).040.040, 11с67п 2цп.00.6.040.040, 11с67п 2цп.00.7.040.040, 11с67п 2цп.00.9.040.040, 11с67п 2цп.01.0.040.040, 11с67п 2цп.01.1.040.040, 11с67п 2цп.01.10.040.040, 11с67п 2цп.01.3.040.040, 11с67п 2цп.01.6(7).040.040, 11с67п 2цп.01.6.040.040, 11с67п 2цп.01.7.040.040, 11с67п 2цп.01.9.040.040, 11с67п 2цпф.00.0.040.040, 11с67п 2цпф.00.1.040.040, 11с67п 2цпф.00.10.040.040, 11с67п 2цпф.00.3.040.040`
- eligible_competitor_articles_continued: `... (+197 more)`
- preferred_competitor_articles: `11с67п 2цп.00.0.040.040, 11с67п 2цп.00.1.040.040, 11с67п 2цп.00.10.040.040, 11с67п 2цп.00.3.040.040, 11с67п 2цп.00.6(7).040.040, 11с67п 2цп.00.6.040.040, 11с67п 2цп.00.7.040.040, 11с67п 2цп.00.9.040.040, 11с67п 2цп.01.0.040.040, 11с67п 2цп.01.1.040.040, 11с67п 2цп.01.10.040.040, 11с67п 2цп.01.3.040.040, 11с67п 2цп.01.6(7).040.040, 11с67п 2цп.01.6.040.040, 11с67п 2цп.01.7.040.040, 11с67п 2цп.01.9.040.040, 11с67п 2цпф.00.0.040.040, 11с67п 2цпф.00.1.040.040, 11с67п 2цпф.00.10.040.040, 11с67п 2цпф.00.3.040.040`
- preferred_competitor_articles_continued: `... (+197 more)`
- returned_top5:
  - rank 1: `цф001040040` | `MARSHAL` | DN `40.0` | PN `40.0` | `фланцевое` | score `None`
  - rank 2: `2цф001040040032` | `MARSHAL` | DN `40.0` | PN `40.0` | `фланцевое` | score `None`
  - rank 3: `цп001040040` | `MARSHAL` | DN `40.0` | PN `40.0` | `сварное` | score `None`
  - rank 4: `2цп001040040032` | `MARSHAL` | DN `40.0` | PN `40.0` | `сварное` | score `None`
  - rank 5: `11с67пцр000040040` | `MARSHAL` | DN `40.0` | PN `40.0` | `резьбовое` | score `None`
- raw_hit@20: `False`
- norm_hit@20: `True`
- normalized_rank_within_20: `1`
