# Licensing and third-party notices

This repository mixes original work with material from other parties. This file says which terms apply to what. It is a plain statement of what the sources say, not legal advice.

## Covered by the MIT License ([LICENSE](LICENSE))

Code written for this project:

- `ingest_esco.py`, `score_skills.py`, `score_skills_typesafe.py`, `compare_skill_scores.py`, `aggregate_scores.py`, `generate_narratives.py`, `merge_narrative_shards.py`, `build_portfolio_data.py`
- `site/explorer.html`, `site/portfolio.html`, `site/insights.html`, `site/scorer.js`
- `site/index.html`, except the treemap layout functions noted under [karpathy/jobs](#karpathyjobs)
- `README.md`, `pyproject.toml`, `.github/workflows/`

## Covered by CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA))

What this project generated: the automation-risk, amplification-potential and evolution-potential scores, the quadrant assignments, the per-skill rationales and the occupation narratives, as found in `data/occupation_narratives.json`, `site/data.json` and `site/portfolio_data.json`. They were generated with Google Gemini Flash via OpenRouter. Google does not claim ownership of generated content ([Gemini API Additional Terms](https://ai.google.dev/gemini-api/terms)), and OpenRouter defers output ownership to the model terms ([OpenRouter Terms](https://openrouter.ai/terms)).

The ESCO and ISCO-08 titles and descriptions embedded in those files are not part of this grant.

## Not covered by either licence

### ESCO classification (European Commission)

`data/esco/*.csv` is the ESCO v1.2.1 download, and ESCO titles and descriptions appear in the derived JSON under `site/`.

This publication uses the ESCO classification of the European Commission.

The data in `site/` and `data/occupation_narratives.json` is a modified and adapted version of ESCO v1.2.1: the scores, quadrants, rationales and narratives are AI-generated additions made by this project and are not part of ESCO.

ESCO is reusable under its own terms, not under CC BY 4.0 or the EUPL: "the ESCO classification can be downloaded, used, reproduced and reused for any purpose and by any interested party free of charge", provided its use is acknowledged and "any modified or adapted version of ESCO must be clearly indicated as such" ([ESCO download conditions](https://esco.ec.europa.eu/en/use-esco/download/privacy-statement), under Commission Decision 2011/833/EU). The European Commission cannot guarantee that the information in ESCO is accurate, up to date or complete, and is not liable for any consequence of its use, reuse or deployment.

### ISCO-08 (International Labour Organization)

The occupation group structure, titles and definitions in `data/esco/ISCOGroups_en.csv`, and the group titles in `site/data.json`, come from ISCO-08 by way of ESCO. ESCO's own acknowledgement reads:

> Information and data in ESCO is based on an original work published by the ILO under the title International Standard Classification of Occupations, ISCO-08. Structure, Group Definitions and Correspondence Tables. Copyright © 2012 International Labour Organization. Adapted and reproduced with permission.

That permission was given to the European Commission. ISCO-08 is a 2012 ILO publication (ISBN 978-92-2-125952-7); the ILO's CC BY 4.0 open-access policy applies only to publications from 3 May 2023 onwards ([ILO rights and permissions](https://www.ilo.org/rights-and-permissions)). ISCO-08 material is therefore excluded from this repository's licence grants. The responsibility for opinions expressed here rests solely with this project, and publication does not constitute an endorsement by the International Labour Office.

### U.S. Bureau of Labor Statistics

`html/`, `occupational_outlook_handbook.html`, `occupations.json` and `occupations.csv` are taken from the Occupational Outlook Handbook, and `data/crosswalks/isco_soc_crosswalk.xls` is the BLS crosswalk between ISCO-08 and the 2010 SOC ([BLS crosswalks](https://www.bls.gov/soc/soccrosswalks.htm)).

Source: Bureau of Labor Statistics, U.S. Department of Labor, *Occupational Outlook Handbook*, at https://www.bls.gov/ooh/. BLS material is in the public domain ([BLS copyright information](https://www.bls.gov/opub/copyright-information.htm)); the photographs the pages link to are not, and none are included here. BLS does not endorse this project.

### karpathy/jobs

This project started from [karpathy/jobs](https://github.com/karpathy/jobs). The following files are byte-identical copies from that repository (checked 2026-09-18):

- `score.py`, `make_prompt.py`, `build_site_data.py`, `make_csv.py`, `process.py`, `scrape.py`, `parse_detail.py`, `parse_occupations.py`
- `prompt.md`, `scores.json`, `occupations.json`, `occupations.csv`, `occupational_outlook_handbook.html` and the 342 pages under `html/`

The squarified treemap layout in `site/index.html` (the layout function and `worstAspect()`) is also taken from it.

karpathy/jobs publishes no licence, so no licence is granted for these files here either. They are not covered by this repository's MIT or CC BY 4.0 grants, and the rights in them remain with their author. The ESCO pipeline and the published site do not depend on them.

### TypeSafe

`score_skills_typesafe.py` calls the TypeSafe System One API through `typesafe-sdk` (MIT). This project is not affiliated with or endorsed by TypeSafe AI, Inc. No TypeSafe output is included in this repository: their customer agreement prohibits publishing benchmarks or performance information about the service.

## Names and emblems

Nothing here grants any right to use the names, logos or emblems of the European Union, the International Labour Organization, the U.S. Bureau of Labor Statistics, Google, OpenRouter or TypeSafe.
