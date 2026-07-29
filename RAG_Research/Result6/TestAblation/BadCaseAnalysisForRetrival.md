# Bad-Case Analysis: Retrieval Failures -- `bge3_cosine_top5`

Best TestAblation method (bge-m3, cosine-only, top-5) -- F1=0.449, the single best arm across all 14 TestAblation arms.

Every false negative (missed gold answer) is attributed to exactly one of two stages, using the retrieved chunk text already saved in `results/bge3_cosine_top5/gpt-5.4.json` (`by_contract.<contract>.<category>.context`) -- no re-running of retrieval or the LLM:

- **Retrieval miss**: the gold answer text does not appear anywhere in the chunks that were retrieved for that category -- the retriever picked the WRONG chunks; the LLM was never given a chance.
- **Extraction miss**: the gold answer text IS present in the retrieved chunks, but the LLM still didn't return it -- a model/prompt problem, not retrieval.

## Summary

- Total false negatives (missed gold answers): **98**
- Caused by retrieval (wrong chunks retrieved): **54** (55.1%)
- Caused by extraction (right chunks, LLM still missed it): **44** (44.9%)
- Categories with at least one retrieval-caused miss: **18** of 24 categories that had any FN at all

## Every category with a retrieval-caused miss (sorted by count, most first)

| Category | Retrieval misses | Extraction misses (same category) |
|---|---|---|
| Parties | 7 | 2 |
| Cap On Liability | 6 | 5 |
| Volume Restriction | 5 | 3 |
| Non-Compete | 4 | 0 |
| Covenant Not To Sue | 4 | 0 |
| Document Name | 3 | 1 |
| Agreement Date | 3 | 1 |
| Change Of Control | 3 | 0 |
| Minimum Commitment | 3 | 5 |
| Revenue/Profit Sharing | 3 | 1 |
| Post-Termination Services | 2 | 1 |
| Exclusivity | 2 | 0 |
| License Grant | 2 | 2 |
| No-Solicit Of Customers | 2 | 0 |
| Ip Ownership Assignment | 2 | 1 |
| Effective Date | 1 | 3 |
| Audit Rights | 1 | 4 |
| Warranty Duration | 1 | 3 |

## Full detail: every retrieval-missed gold answer, by category

### Parties (7 retrieval misses)

- **Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO ...**: 'Fleet Laboratories Limited'
- **Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO ...**: 'Columbia Laboratories, (Bermuda) Ltd.'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Airspan Networks Incorporated'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'Diplomat Ambassador Eyewear Group'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'The Sterling/Winters Co.'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'Biopure Corporation'
- **DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGRE...**: 'Company and Consultant shall sometimes be referred to herein singularly as a "Party" or collectively as the "Parties" to this Agreement.'

### Cap On Liability (6 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'In any event, Airspan shall not be liable      for any direct, indirect, consequential, or special losses or damages      (including, but not limited to, loss o...'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Airspan shall not be liable to Distributor on account of termination or expiration of this Agreement for reimbursement or damages for loss of goodwill, prospect...'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'WITHOUT PREJUDICE TO SECTION 16.4, NEITHER Airspan, NOR ANY OF ITS OFFICERS, DIRECTORS, EMPLOYEES, AGENTS, REPRESENTATIVES, SHAREHOLDERS, OR AFFILIATES (Airspan...'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Airspan shall not be liable to Distributor for damages of any kind, including incidental or consequential damages, on account of the termination of this agreeme...'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Airspan's obligation and Distributor's sole remedy under this warranty are limited to the replacement or repair, at Airspan's option, of the defective Equipment..."
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: "The Customer's exclusive remedy for a breach of any of the foregoing warranties will be the replacement, at the delivery point thereof, freight prepaid, of any ..."

### Volume Restriction (5 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Airspan shall provide a single technical course in the English language for up to two (2) qualified technicians of Distributor during the first year of this Agr...'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'The photo sessions shall be up to two (2) consecutive days in duration, each day to consist of no more than eight (8) working hours.'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'in duration, each day to consist of no more than eight (8) working hours.'
- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: "Make himself available for four (4) sessions for production of photographs, or radio, television, video or other multi-media  programming for use in Bizzingo's ..."
- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: 'Make four (4) public appearance for the purpose of promoting the Network, which may include autograph sessions, dinner  appearances, and/or other appearances no...'

### Non-Compete (4 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'During the term of this Agreement Distributor agrees that neither it      nor any organization or entity controlled or directed by it will, without      Airspan...'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Distributor will give Airspan thirty (30) days' prior, written notice of each new potential representation role being considered by Distributor, and Distributor..."
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'During the term of this Agreement, and for a period of three (3) months following the expiration or termination of this Agreement, Distributor agrees that neith...'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Except as\n\n\n\n\n\nprovided above, in no event will Airspan consent to Distributor's consultation for or representation of a manufacturer or supplier, which is dire..."

### Covenant Not To Sue (4 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Distributor admits Airspan\'s exclusive ownership of the name "Airspan Networks Incorporated", "Airspan Communications Ltd.", "ANI", "ACL", and any abbreviations...'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Distributor acknowledges Airspan's exclusive right, title, and interest in and to any trademarks, trade names, logos and designations which Airspan may at any t..."
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'In connection with any reference to the Trademarks, Distributor shall not in any manner represent that it has an ownership interest in the Trademarks or registr...'
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Distributor recognizes the validity of Airspan's copyright in any written material to which Airspan shall have made a claim to copyright protection, and Distrib..."

### Document Name (3 retrieval misses)

- **Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO ...**: 'AMENDMENT NO. 2 TO MANUFACTURING AND SUPPLY AGREEMENT'
- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'Endorsement Agreement'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'AGENCY AGREEMENT'

### Agreement Date (3 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: '31st day of March, 2000'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'March 29, 1999'
- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: 'March 14, 2012'

### Change Of Control (3 retrieval misses)

- **Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO ...**: '(ii) a Change of Control Event with respect to Fleet occurs;'
- **Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO ...**: 'Columbia shall have the right to terminate this Agreement upon [***] notice to Fleet in the event:'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'In the event of any material change in the organization,       ownership, management or control of the business of the Agent, the Company       may, at its opti...'

### Minimum Commitment (3 retrieval misses)

- **Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO ...**: 'The amounts set forth for the [***] in each Production Schedule shall constitute a firm purchase order and shall be binding upon Columbia (each a "Purchase Orde...'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: '(i) make all field sales representatives of the Agent available to       work with field sales representatives of the Company at least two (2) full       busine...'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: '(h) make all field sales personnel of the Agent available for at       least four (4) hours, and telesales personnel available for at least one       (1) hour, ...'

### Revenue/Profit Sharing (3 retrieval misses)

- **AMBASSADOREYEWEARGROUPINC_11_17_1997-EX-10.28-ENDORSEME...**: 'Diplomat agrees to pay KI, Inc. as royalty a sum equal to  % of the net wholesale volume of the products covered by this Agreement by Diplomat and its affiliate...'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'The       Company will compensate the Agent an additional two (2) percent through a       discount off of the current price or promotional price of the Product ...'
- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: 'The Royalty payable under the Agreement shall be in the form of one (1) common stock purchase warrant of Bizzingo (as further  described herein) for each Activa...'

### Post-Termination Services (2 retrieval misses)

- **Columbia Laboratories, (Bermuda) Ltd. - AMEND NO. 2 TO ...**: 'Upon termination of this Agreement, Fleet agrees to perform its obligations under this Agreement until the earlier of [***].'
- **DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGRE...**: 'Consultant agrees to keep and maintain adequate, current, accurate, and authentic written records of all Inventions made by Consultant (solely or jointly with o...'

### Exclusivity (2 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: 'Subject to the provisions of this Agreement, Airspan hereby appoints Distributor as an independent, exclusive distributor to assist Airspan in marketing the Air...'
- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: 'Notwithstanding the foregoing, during the term and for a period of one (1) year thereafter, Theismann shall not use,  permit the use of, or license to others th...'

### License Grant (2 retrieval misses)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Distributor's appointment as a distributor of the Airspan Products grants to Distributor only a license to resell the"
- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Airspan Products to Distributor's customers in the Territory, and does not transfer any right, title, or interest in any of the Airspan Software to Distributor."

### No-Solicit Of Customers (2 retrieval misses)

- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: 'Except as otherwise expressly provided in the Business Plan, the Agent will at its sole expense'
- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: '(e) not solicit or accept orders for the Products other than from       Customers within the Territory after the Agent Launch Date; and not       knowingly, or ...'

### Ip Ownership Assignment (2 retrieval misses)

- **DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGRE...**: "Consultant agrees to assist Company, or its designee, at the Company's expense, in every proper way to secure the Company's rights in Inventions in any and all ..."
- **DRIVENDELIVERIES,INC_05_22_2020-EX-10.4-CONSULTING AGRE...**: "Consultant agrees that, if the Company is unable because of Consultant's unavailability, dissolution, mental or physical incapacity, or for any other reason, to..."

### Effective Date (1 retrieval miss)

- **BizzingoInc_20120322_8-K_EX-10.17_7504499_EX-10.17_Endo...**: 'March 1, 2012'

### Audit Rights (1 retrieval miss)

- **BIOPURECORP_06_30_1999-EX-10.13-AGENCY AGREEMENT**: '(f) meet with the Company at least once each quarter (starting with       the quarter in which the Agent Launch Date occurs), at a mutually       agreeable time...'

### Warranty Duration (1 retrieval miss)

- **AIRSPANNETWORKSINC_04_11_2000-EX-10.5-Distributor Agree...**: "Subject to the provisions of this warranty clause, defective parts or components must be returned by Distributor to Airspan's designated facility located within..."


---
*Generated by `bad_case_retrieval_analysis.py`. Re-run any time after re-running the underlying arm -- this script makes no API calls.*
