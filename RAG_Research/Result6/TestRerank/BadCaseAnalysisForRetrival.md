# Bad-Case Analysis: Retrieval Failures -- `qwen3_rrf_n10_top5`

Best TestRerank method (qwen3-embedding:0.6b, RRF fusion, BM25/cosine shortlist N=10, top-5) -- F1=0.457, the best arm across TestAblation + TestRerank combined.

Every false negative (missed gold answer) is attributed to exactly one of two stages, using the retrieved chunk text already saved in `results/qwen3_rrf_n10_top5/gpt-5.4.json` (`by_contract.<contract>.<category>.context`) -- no re-running of retrieval or the LLM:

- **Retrieval miss**: the gold answer text does not appear anywhere in the chunks that were retrieved for that category -- the retriever picked the WRONG chunks; the LLM was never given a chance.
- **Extraction miss**: the gold answer text IS present in the retrieved chunks, but the LLM still didn't return it -- a model/prompt problem, not retrieval.

## Summary

- Total false negatives (missed gold answers): **87**
- Caused by retrieval (wrong chunks retrieved): **48** (55.2%)
- Caused by extraction (right chunks, LLM still missed it): **39** (44.8%)
- Categories with at least one retrieval-caused miss: **19** of 24 categories that had any FN at all

## Every category with a retrieval-caused miss (sorted by count, most first)

| Category | Retrieval misses | Extraction misses (same category) |
|---|---|---|
| Parties | 8 | 1 |
| License Grant | 6 | 0 |
| Volume Restriction | 5 | 3 |
| Cap On Liability | 4 | 6 |
| Agreement Date | 3 | 3 |
| Minimum Commitment | 3 | 3 |
| Document Name | 2 | 2 |
| Audit Rights | 2 | 2 |
| Non-Compete | 2 | 0 |
| Exclusivity | 2 | 0 |
| Revenue/Profit Sharing | 2 | 1 |
| Ip Ownership Assignment | 2 | 1 |
| Effective Date | 1 | 3 |
| Expiration Date | 1 | 3 |
| Post-Termination Services | 1 | 1 |
| Warranty Duration | 1 | 3 |
| Covenant Not To Sue | 1 | 1 |
| No-Solicit Of Customers | 1 | 0 |
| Renewal Term | 1 | 0 |

## Full detail: every retrieval-missed gold answer, by category

### Parties (8 retrieval misses)

- **Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO ...**: 'Fleet Laboratories Limited'
- **Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO ...**: 'Columbia Laboratories, (Bermuda) Ltd.'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Airspan Networks Incorporated'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'Diplomat Ambassador Eyewear Group'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'The Sterling/Winters Co.'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'SW'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'Biopure Corporation'
- **DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGRE...**: 'Company and Consultant shall sometimes be referred to herein singularly as a "Party" or collectively as the "Parties" to this Agreement.'

### License Grant (6 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Distributor's appointment as a distributor of the Airspan Products grants to Distributor only a license to resell the"
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Airspan Products to Distributor's customers in the Territory, and does not transfer any right, title, or interest in any of the Airspan Software to Distributor."
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: '(2) optical cases, optical eye chains, eye pins, and lens cleaning kits sold only in optical retailers; and\n\n                        (3) such other optical acce...'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'Upon the terms and conditions set forth in this Agreement, KI, Inc. hereby grants to Diplomat and Diplomat hereby accepts the right, license and privilege of ut...'
- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: 'Subject to the terms and conditions set forth herein, Theismann hereby grants to Bizzingo and its affiliates the unlimited  right and privilege during the Term ...'
- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: 'It being understood and agreed that Bizzingo shall have the right to exhibit commercials, infomercials, advertisements and  otherwise make use of all Property o...'

### Volume Restriction (5 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Airspan shall provide a single technical course in the English language for up to two (2) qualified technicians of Distributor during the first year of this Agr...'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'The photo sessions shall be up to two (2) consecutive days in duration, each day to consist of no more than eight (8) working hours.'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'in duration, each day to consist of no more than eight (8) working hours.'
- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: "Make himself available for four (4) sessions for production of photographs, or radio, television, video or other multi-media  programming for use in Bizzingo's ..."
- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: 'Make four (4) public appearance for the purpose of promoting the Network, which may include autograph sessions, dinner  appearances, and/or other appearances no...'

### Cap On Liability (4 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'In any event, Airspan shall not be liable      for any direct, indirect, consequential, or special losses or damages      (including, but not limited to, loss o...'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Airspan shall not be liable to Distributor on account of termination or expiration of this Agreement for reimbursement or damages for loss of goodwill, prospect...'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Airspan shall not be liable to Distributor for damages of any kind, including incidental or consequential damages, on account of the termination of this agreeme...'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Airspan's obligation and Distributor's sole remedy under this warranty are limited to the replacement or repair, at Airspan's option, of the defective Equipment..."

### Agreement Date (3 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: '31st day of March, 2000'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'August 24, 1995'
- **DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGRE...**: 'May 1, 2019'

### Minimum Commitment (3 retrieval misses)

- **Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO ...**: 'The amounts set forth for the [***] in each Production Schedule shall constitute a firm purchase order and shall be binding upon Columbia (each a "Purchase Orde...'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: '(i) make all field sales representatives of the Agent available to       work with field sales representatives of the Company at least two (2) full       busine...'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: '(h) make all field sales personnel of the Agent available for at       least four (4) hours, and telesales personnel available for at least one       (1) hour, ...'

### Document Name (2 retrieval misses)

- **Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO ...**: 'AMENDMENT NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'AGENCY AGREEMENT'

### Audit Rights (2 retrieval misses)

- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'Except as otherwise expressly provided in the Business Plan, the Agent will at its sole expense'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: '(f) meet with the Company at least once each quarter (starting with       the quarter in which the Agent Launch Date occurs), at a mutually       agreeable time...'

### Non-Compete (2 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Distributor will give Airspan thirty (30) days' prior, written notice of each new potential representation role being considered by Distributor, and Distributor..."
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Except as\n\n\n\n\n\nprovided above, in no event will Airspan consent to Distributor's consultation for or representation of a manufacturer or supplier, which is dire..."

### Exclusivity (2 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Subject to the provisions of this Agreement, Airspan hereby appoints Distributor as an independent, exclusive distributor to assist Airspan in marketing the Air...'
- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: 'Notwithstanding the foregoing, during the term and for a period of one (1) year thereafter, Theismann shall not use,  permit the use of, or license to others th...'

### Revenue/Profit Sharing (2 retrieval misses)

- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'Diplomat agrees to pay KI, Inc. as royalty a sum equal to  % of the net wholesale volume of the products covered by this Agreement by Diplomat and its affiliate...'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'The       Company will compensate the Agent an additional two (2) percent through a       discount off of the current price or promotional price of the Product ...'

### Ip Ownership Assignment (2 retrieval misses)

- **DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGRE...**: "Consultant agrees to assist Company, or its designee, at the Company's expense, in every proper way to secure the Company's rights in Inventions in any and all ..."
- **DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGRE...**: "Consultant agrees that, if the Company is unable because of Consultant's unavailability, dissolution, mental or physical incapacity, or for any other reason, to..."

### Effective Date (1 retrieval miss)

- **DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGRE...**: 'May 1, 2019'

### Expiration Date (1 retrieval miss)

- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'This Agreement will become effective as of the date first written above and will continue in effect thereafter until terminated pursuant to Paragraph 4.2 below.'

### Post-Termination Services (1 retrieval miss)

- **DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGRE...**: 'Consultant agrees to keep and maintain adequate, current, accurate, and authentic written records of all Inventions made by Consultant (solely or jointly with o...'

### Warranty Duration (1 retrieval miss)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Subject to the provisions of this warranty clause, defective parts or components must be returned by Distributor to Airspan's designated facility located within..."

### Covenant Not To Sue (1 retrieval miss)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Distributor admits Airspan\'s exclusive ownership of the name "Airspan Networks Incorporated", "Airspan Communications Ltd.", "ANI", "ACL", and any abbreviations...'

### No-Solicit Of Customers (1 retrieval miss)

- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'Except as otherwise expressly provided in the Business Plan, the Agent will at its sole expense'

### Renewal Term (1 retrieval miss)

- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: 'Unless sooner terminated under the provisions hereof, this Agreement shall commence on the Effective Date and continue for a period  of one (1) year ("Term"). p...'


---
*Generated by `bad_case_retrieval_analysis.py`. Re-run any time after re-running the underlying arm -- this script makes no API calls.*
