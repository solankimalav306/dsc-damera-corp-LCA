**CSE558 Data Science • Midterm Project Report • Monsoon 2026**

# How do employers' filing choices relate to the Department of Labor's decision on their Labor Condition Application (LCA)?

---

| | |
|---|---|
| **Group name** | Damera Corp. |
| **Dataset** | <https://www.dol.gov/media/LCA_Disclosure_Data_FY2026_Q3.xlsx> |
| **Code** | github.com/team-kurtosis/cse558-midterm (notebook and README) |

| **#** | **Name** | **Roll no.** | **Institute email** | **Contribution** |
|---|---|---|---|---|
| 1 | Malav Mehulkumar Solanki | 2023306 | malav23306@iiitd.ac.in | Dataset and Data Preparation |
| 2 | Rishith Rohan Anamalla | 2023081 | anamalla23081@iiitd.ac.in | Problem Statement,EDA and Hypotheses Testing |
| 3 | Vyom Rameshbhai Sitapara | 2023604 | vyom23604@iiitd.ac.in | Hypotheses Testing and problem statement |
| 4 | Abhinith Woona | 2023025 | abhinith23025@iiitd.ac.in | Problem Statement and EDA |

---

## 1. Dataset

**Source:** U.S. Department of Labor, Office of Foreign Labor Certification (OFLC), "LCA Disclosure Data, FY2026 Q3": public disclosure file of Labor Condition Applications, downloaded on 20 October 2026. One row is one LCA filing.

**What is raw about it:** `WAGE_RATE_OF_PAY_FROM`, `WAGE_RATE_OF_PAY_TO` and `PREVAILING_WAGE` are stored in different units `PW_UNIT_OF_PAY` and `WAGE_UNIT_OF_PAY` units (Hour, Week, Bi-Weekly, Year, Month) ;`PW_OES_YEAR` looks like a date but is a text range ("7/1/2024 – 6/30/2025"); 90.1% of `PW_OTHER_SOURCE`, 10.3% of `PW_WAGE_LEVEL`, 48.7% of `PREPARER_LAST_NAME` and 69.1% of `WAGE_RATE_OF_PAY_TO` are missing or `UNKNOWN` but not all are to be considered as a defect; `JOB_TITLE` has many variants referring to the same occupational title (for example "Database Administrator" and "Database Admin/WebLogic Admin-Developer").

| | **Raw file** | **After encoding** |
|---|---|---|
| Instances (rows) | 437,497 | 427,599 |
| Attributes (columns) | 98 | 430 |
| Instances × attributes | 42,874,608 | 183,867,570 |

**Size check:** 427,599 × 430 = 183,867,570 ≥ 3,000,000, so the requirement is met. The raw file already exceeded 3M, and encoding adds more attributes: 150 TF-IDF features derived from `JOB_TITLE`, and 190 one-hot encoded features from categorical columns such as `EMPLOYER_STATE` and `WORKSITE_STATE`.

**Key features**

| **Feature** | **Type** | **Real-world meaning** |
|---|---|---|
| `PREVAILING_WAGE` | continuous | minimum wage for the occupation in the area |
| `NAICS_CODE` | nominal | industry classification of the employer |
| `VISA_CLASS` | nominal | visa class of the worker: H-1B, H-1B1 (Chile or Singapore), E-3 (Australian) |
| `H-1B_DEPENDENT` | nominal | employer is classed as H-1B dependent (large share of its workforce on H-1B visas) |
| `TOTAL_WORKER_POSITIONS` | discrete | number of foreign workers requested in the filing |
| `PW_WAGE_LEVEL` | ordinal | prevailing wage skill level (I, II, III, IV) |
| `WILLFUL_VIOLATOR` | nominal | employer has been found a willful violator of the rules (yes/no) |
| `SOC_CODE` | nominal | standard occupational classification code for the job |
| `PW_UNIT_OF_PAY` | nominal | unit of the prevailing wage (Hour, Week, Bi-Weekly, Month, Year) |
| `JOB_TITLE` | nominal | title of the position, as provided by the employer |
| `FULL_TIME_POSITION` | nominal | whether the position is full-time (yes/no) |
| `WORKSITE_STATE` | nominal | state where the job will be performed |
| `RECEIVED_DATE` | Interval | date the LCA was received by the DOL |
| `BEGIN_DATE` | Interval | intended start date of employment |
| `END_DATE` | Interval | intended end date of employment |

## 2. Data preparation

| **Issue found** | **What we did** | **Why, and not the alternative** |
|---|---|---|
| Rows with no `CASE_STATUS` | dropped them | a label is needed to learn from the data (the ground truth). Treating the empty ones as `UNKNOWN` would feed factually incorrect labels to the model |
| `WILLFUL_VIOLATOR` and `H-1B_DEPENDENT` are coded as YES/NO and Y/N, and 2.4% of fields are empty | converted to three classes: `YES`, `NO` and `UNKNOWN` | this standardises the variables so they can be one-hot encoded. Dropping the empty rows is the alternative, but it would lose data |
| `EMPLOYER_POC_COUNTRY` and `EMPLOYER_COUNTRY` are 99.9% United States of America | replaced them with boolean flags `POC_IS_US` and `EMPLOYER_IS_US` | this avoids many near-empty columns after one-hot encoding, while keeping the fact that the majority is the U.S. |
| `WORKSITE_CITY` has 10,393, `EMPLOYER_CITY` has 6,837, `WORKSITE_POSTAL_CODE` has 13,491 and `EMPLOYER_POSTAL_CODE` has 10,355 unique values | created a `MISMATCH_LOCATION` flag that compares the worksite and employer cities and postal codes | this sharply cuts the number of unique values, and hence the columns after one-hot encoding, while keeping the signal of whether the job is at the employer's own location |
| Detailed information columns: `PW_OES_YEAR`, the `EMPLOYER_POC_*` contact details, the `AGENT_ATTORNEY_POC_*` contact details, the `PREPARER_*` details, `LAWFIRM_*` details and `*_COURT` details, 55 attributes in total | dropped all 55 attributes | these are background and contact details that support the filing process but do not describe the employer's filing choices directly. Keeping them would add many sparse columns |
| `JOB_TITLE` is plain text with conflicting Entities that cannnot be matched directly. | Encodeded it using `TF-IDF` from natural language | Use of TF-IDF makes matching easier to find similar Job titles, alternatively it could be one hot encodeded or could be encoded using Sentence Transformers creating heavy embeddings. |
| `NAICS_CODE` has 1,827 different codes (strings) | split into `NAICS_SECTOR`, `NAICS_SUBSECTOR`, `NAICS_INDUSTRY_GROUP_FREQ`, `NAICS_INDUSTRY_FREQ` and `NAICS_NATIONAL_INDUSTRY_FREQ` | the column could be dropped or one-hot encoded, but the leading digits of a NAICS code encode a hierarchy (sector, subsector, industry group, industry, national industry). Extracting each level as its own attribute keeps that structure and keeps the dimensionality low |
| `SOC_CODE` has 845 different codes (strings) | split into `SOC_SPECIALTY`, `SOC_MAJOR`, `SOC_MINOR`, `SOC_BROAD_FREQ` and `SOC_DETAILED_FREQ` | the column could be dropped or one-hot encoded, but a SOC code encodes a hierarchy (major, minor, broad and detailed groups). Extracting each level as its own attribute keeps that structure and avoids hundreds of one-hot columns |
| `WAGE_RATE_OF_PAY_FROM`, `WAGE_RATE_OF_PAY_TO` and `PREVAILING_WAGE` are recorded in 5 different units (Hour, Week, Bi-Weekly, Month, Year), given by `WAGE_UNIT_OF_PAY` and `PW_UNIT_OF_PAY` | converted every wage to dollars per hour, using each row's own unit column: ÷40 for weekly, ÷80 for bi-weekly, ÷(2,080/12) for monthly and ÷2,080 for yearly | raw amounts are not comparable (a yearly 90,000 would look larger than an hourly 45). We follow the DOL's own convention of 2,080 hours a year (40 hours × 52 weeks), stated in its Prevailing Wage Policy Guidance (Nov 2009). Converting to yearly is the alternative, but hourly is how part-time wages are entered on the LCA. Keeping the units and one-hot encoding them would leave the amounts incomparable |
| 9,423 exact duplicate rows: identical in all 43 selected columns plus the label (8,668 Certified, 625 Certified–Withdrawn, 110 Withdrawn, 20 Denied) | removed all of them, keeping the first copy | identical copies add no information and can land in both train and test, inflating scores. Keeping them would overweight repeated filings. Only 0.21% were `Denied`, so removal did not skew the classes |
| 474 rows with impossible hourly wages: above \$1,000/hr or ≤\$0 (the maximum was \$534,188 per hour; 346 Certified, 54 Denied, 50 Certified–Withdrawn, 24 Withdrawn) | removed the rows with a fixed domain threshold | these are data-entry errors, such as an annual wage entered with the unit "Hour". The IQR rule would also remove about 14k genuine high salaries (physicians, executives), so we used it only to measure outliers. Capping or fixing (÷2,080) was the alternative, but it is a guess if the unit was typed wrong. These rows were denied 20 times more often than normal, so removing them loses 54 of 2,557 `Denied` rows |

## 3. What changed after preparation

| **Measure** | **Before** | **After** |
|---|---|---|
| Rows | 437,497 | 427,599 |
| Attributes | 98 | 430 |
| Missing or unusable cells | 1,699,320 | 0 (none dropped; 405,503 kept as an `UNKNOWN` level, the rest turned into 0/1 flags) |
| Repeated rows (43 columns + label) | 9,423 | 0 |
| Implausible wages (above \$1,000/hr or ≤ 0) | 474 | 0 (removed: 346 Certified, 54 Denied, 50 Certified–Withdrawn, 24 Withdrawn) |
| Wage rows beyond the IQR fence (prevailing / offered) | 9,823 (2.3%) / 13,241 (3.1%) | same, kept (genuine high salaries) |
| Missing in `PW_OTHER_SOURCE` | 90.1% of rows | 0: replaced by the flag `HAS_NON_STANDARD_WAGE_SOURCE` |
| Missing in `STATUTORY_BASIS` | 77.1% of rows | 0: replaced by the flag `HAS_CLAIMED_EXEMPTION` |
| Missing in `WAGE_RATE_OF_PAY_TO` | 69.1% of rows | 0: replaced by the flag `HAS_WAGE_RANGE` |
| Missing in `PREPARER_LAST_NAME` | 48.7% of rows | 0: replaced by the flag `HAS_PREPARER` |
| Missing in `H-1B_DEPENDENT` | 2.4% of rows | 0: filled with `UNKNOWN` |
| Missing in `WILLFUL_VIOLATOR` | 2.4% of rows | 0: filled with `UNKNOWN` |
| Text identifier columns (non-categorical) | 12 | 0: converted to boolean, TF-IDF or one-hot encoded |
| Wage units of pay used | 5 (Hour, Week, Bi-Weekly, Month, Year) | 1: all wages standardised per hour |

In short, every text column was turned into 0/1 flags, one-hot columns or TF-IDF scores, and the `NAICS_CODE` and `SOC_CODE` hierarchies were split into levels (employer industry and worker occupation). This is what makes the group comparisons in Section 5 possible.

## 4. Basic questions and problem statement

![Figure 1](fig_outcome.png)

**Figure 1:** 91.5% of applications are certified and 6.2% are certified then withdrawn. Only 2.3% end withdrawn (1.7%) or denied (0.6%), so always guessing "certified" would be right 97.7% of the time.

![Figure 2](fig_rates.png)

**Figure 2:** Applications with no earlier certified application are not certified 4.38% of the time, against 1.69% with one. Applications filed without an agent are not certified 3.16% of the time, against 2.06% with one; the two wage comparisons differ by under 0.2 points. Intervals ignore employer clustering.

![Figure 3](fig_naics.png)

**Figure 3:** Professional, scientific and technical services file 49.5% of applications. Not-certified rates run from 1.4% (information) to 3.5% (health care) around the 2.3% average. Retail is shown without its largest employer (16.7% of retail applications, 0.44% not certified); the rest of retail is 2.52%. Intervals ignore employer clustering.

![Figure 4](fig_rate_soc.png)

**Figure 4:** Not-certified rates run from 1.8% (computer and math) to 3.4% (management), so occupation may confound the pooled tests; T1 is re-checked within computer and math occupations in Section 5.

![Figure 5](fig_premium.png)

**Figure 5:** About a third of applications (35.1%) offer exactly the prevailing wage; the median offer is 5.1% above it, and none is below. The 79 applications with a ratio above 10 are not plotted.

![Figure 6](fig_agent.png)

**Figure 6:** 78.0% of applications are filed through an agent.

![Figure 7](fig_wage_level.png)

**Figure 7:** Level II is the most common prevailing-wage level (35.8%); levels I, III and IV each hold 17–19%. 9.1% have no level, and 93.8% of those use a non-standard wage source.

![Figure 8](fig_occupation.png)

**Figure 8:** Computer and mathematical occupations make up 58.7% of applications. Business & finance and architecture & engineering add about 10% each, so the top three cover about 79%.

![Figure 9](fig_title_sim.png)

**Figure 9:** Mean similarity to an employer's earlier titles measures how varied its job titles are. Titles outside the 150-term vocabulary score 0, so the spikes at 0 and 1 partly reflect that limit; an exact-match check gives 1.57% not certified for repeated titles vs 1.95% for new ones.

![Figure 10](fig_state.png)

**Figure 10:** Four states (CA, TX, NY, IL) hold about 49% of worksites, and the top ten hold about 72%.

**Basic questions**

1. What share of applications end certified, withdrawn or denied?
2. Which occupation groups file the most?
3. Which industry sectors file the most?
4. Which worksite states are most common?
5. How are applications spread across wage levels?
6. How does the offered wage compare with the prevailing wage?
7. What share file through an agent?

**Our problem statements**

> **P1 (primary, employer experience).** Is having an earlier certified application behind the employer, or filing through an agent, linked to a different chance of the application ending withdrawn or denied?
>
> **P2 (pay).** Is offering exactly the prevailing wage rather than more, or using a non-standard wage source, linked to a different chance of ending withdrawn or denied?

**How they are novel:** the basic questions describe one column at a time (shares, spreads, sizes). Ours relate the outcome to features we derived: whether the employer has an earlier *certified* application before this one was received, and whether the offered hourly wage equals the prevailing hourly wage. Neither is a column in the file.

**Causal caveat:** these are questions about association. "Withdrawn" is the employer's own action and the file gives no reason, so we cannot say that any filing choice *causes* a different outcome.

## 5. Statistical inferences

**Split and rows tested:** before any exploration we split the 427,599 applications by employer (`FEIN_ID`), stratified by employer volume, seed 558, so that no employer is in both halves. All plots above use the exploration half (213,732 rows). Every test below was run once on the confirmation half (213,732 rows; 4,643 withdrawn or denied, 2.17%). Outcome: withdrawn or denied = 1; certified or certified–withdrawn = 0. Group 1 and group 0 are defined in the table below; derived features are fixed rules, with nothing fitted on outcomes.

**a) Hypotheses, set before the tests**

| **Test** | **Group 1 vs group 0** | **Hypotheses** |
|---|---|---|
| **P1: Employer experience and use of an agent** | | |
| T1 | Earlier certified application (`has_history`) vs none | H₀: p₁ = p₀; H₁: p₁ ≠ p₀ |
| T2 | Filed through an agent vs not | H₀: p₁ = p₀; H₁: p₁ ≠ p₀ |
| **P2: Wage choices** | | |
| T3 | Wage ratio ≤ 1.001 vs > 1.001 (`at_min`) | H₀: p₁ = p₀; H₁: p₁ ≠ p₀ |
| T4 | Non-standard wage source vs standard | H₀: p₁ = p₀; H₁: p₁ ≠ p₀ |

**Test setup.** p₁ and p₀ are the shares of applications ending withdrawn or denied in groups 1 and 0. For each of T1–T4 we use a two-sided pooled two-proportion z-test at α = 0.05, with a χ² test of independence (d.f. = 1) as a check. Before running the tests, we set 1 percentage point as the smallest difference that would matter in each test.. The overall not-certified rate is about 2.3%; this threshold is substantial relative to the base rate. Across four tests, we also compare each p-value with 0.05/4 = 0.0125; adjusted p-values are not reported.

**b) Other tests we considered**

All four comparisons (T1–T4) involve two groups and a binary outcome, so for each of them the pooled two-proportion z-test directly compares the rates, and the alternatives below were set aside for the same reasons. For each comparison, we also report the equivalent χ² test of independence as a cross-check: z² = χ² with one degree of freedom.

| **Also considered** | **Why we did not use it** |
|---|---|
| Two-sample t-test | A t-test on a 0/1 outcome also compares group means, which equal the rates. The two-proportion test states our rate comparison directly. |
| F-test | It tests equality of variances, while our question concerns outcome rates. |
| χ² goodness-of-fit test | It compares one distribution with specified proportions; our question compares rates between two groups. |
| Correlation test | Coding the group and outcome as 0/1 gives a related test of association. The two-proportion test gives the rate difference directly. |
| Fisher exact test | It is suitable for sparse counts. Expected counts here range from about 495 to 1,684, so the z-test and chi-square approximations are adequate. |
| Paired test | The applications are not matched pairs; each application belongs to one group only. |

**c) Results (confirmation half)**

| **Test** | **Group 1 (n; count; rate)** | **Group 0 (n; count; rate)** |
|---|---|---|
| T1 | 164,953; 2,521; 1.53% | 48,779; 2,122; 4.35% |
| T2 | 154,079; 3,222; 2.09% | 59,653; 1,421; 2.38% |
| T3 | 77,515; 1,541; 1.99% | 136,217; 3,102; 2.28% |
| T4 | 22,777; 430; 1.89% | 190,955; 4,213; 2.21% |

| **Test** | **p₁ − p₀ (points)** | **95% CI** | **z** | **χ² (d.f. 1)** | **p-value** |
|---|---|---|---|---|---|
| T1 | −2.82 | −3.01 to −2.63 | −37.56 | 1,410.66 | < 0.001 |
| T2 | −0.29 | −0.43 to −0.15 | −4.14 | 17.13 | 3.5 × 10⁻⁵ |
| T3 | −0.29 | −0.42 to −0.16 | −4.41 | 19.45 | 1.0 × 10⁻⁵ |
| T4 | −0.32 | −0.51 to −0.13 | −3.12 | 9.71 | 0.0018 |

For each test, z² matches the χ² statistic. The confidence intervals use the unpooled standard error: p₁ − p₀ ± 1.96 SE. All four p-values are below 0.0125,so the decision for each of T1–T4 is to reject H₀. Only T1 reaches the pre-set 1-point threshold for a meaningful difference.

Decision: we reject H₀ in all four tests.

**What the results show**

- **T1: largest difference:** Applications from employers with an earlier certified filing ended withdrawn or denied at 1.53%, compared with 4.35% for employers without one. The difference is −2.82 percentage points (95% CI −3.01 to −2.63), exceeding our 1-point threshold. The exploration half showed the same pattern: 1.69% with prior history versus 4.38% without. Within computer and mathematical occupations (SOC 15; 58.8% of the confirmation half), the difference was −2.10 points (95% CI −2.35 to −1.84).
- **T2–T4: statistically significant, but small:** The rates were lower by 0.29 points for agent-filed applications, 0.29 points for applications at or very slightly above the prevailing wage, and 0.32 points for applications using a non-standard wage source. Each 95% confidence interval lies entirely between −1 and +1 percentage point. These differences are statistically significant but do not reach our threshold for practical size.
- **Interpretation:** In T2–T4, the group named first has a slightly lower not-certified rate. These are associations and do not show that using an agent or choosing a wage source changes the outcome.
- **T3 sensitivity check:** Removing the 78 confirmation-half applications with an offered-to-prevailing wage ratio above 10 gives a difference of −0.28 points (95% CI −0.40 to −0.15), with the same conclusion.

## 6. Limitations

- **These results are associations:** A withdrawal is the employer's action, and the file does not give the reason. Employers with prior certified filings or agents may also differ in size, experience, or industry. The tests cannot separate these factors or establish cause.
- **Applications from one employer may be related:** The confirmation half contains 213,732 applications, including employers with multiple filings. Our proportion tests treat applications as independent, so the confidence intervals may be too narrow.
- **Occupation differences:** Not-certified rates vary across occupation groups (about 1.8% to 3.4%). The T1 check within the largest group shows the same direction, but we did not repeat all four tests within occupations.
- **Overlapping wage measures:** Among applications with no wage level, 93.8% use a non-standard wage source. T3 and T4 therefore do not represent independent pieces of evidence. No application in the prepared data offers less than the prevailing wage.
- **Limited filing history:** The file covers decisions issued from 1 October 2025 to 30 June 2026, so the "no earlier certified application" group includes both new filers and employers whose earlier certifications fall outside the file. Some applications were received much earlier (about 10,000 in 2023–2024). We also kept applications decided before July 2025 that are marked "Certified - Withdrawn."
- **Possible repeat filings:** We treat each application as a separate record. In the raw data, 5,998 rows match on all fields except case number; this match alone does not confirm that they are resubmissions.
- **Features fitted before the split:** The 150-term title vocabulary and frequency columns were fitted using all rows. The four tests do not use these features, but this was not a fully isolated train/test workflow.
- **Cleaning changed after results were seen:** We removed 9,423 exact duplicates and 474 applications with implausible hourly wages. These rules were changed after the confirmation results had been viewed. The earlier run gave the same overall conclusions, but this change limits how strongly we can describe the results as confirmatory.
