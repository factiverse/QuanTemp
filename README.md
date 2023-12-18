# QuanTemp
A repository for baselines to verify Numerical Claims.

An example from dataset
```
{
        "country_of_origin": "usa",
        "label": "Conflicting",
        "url": "https://www.politifact.com/factchecks/2010/aug/08/donald-carcieri/carcieri-says-tax-repeal-spawned-new-business/",
        "lang": "en",
        "claim": "Repealing the sales tax on boats in Rhode Island has spawned 2,000 companies, 7,000 jobs and close to $2 billion a year in sales activity.",
        "doc": "The furor over U.S. Sen. John Kerry\u2019s yacht being docked on the tax-free shores of Rhode Island -- not in Massachusetts where he lives -- has subsided now that the senator\u2019s team has promised he will pay the $500,000 in owed taxes.But Rhode Island Governor Donald Carcieri did his part to make sure the underlying issue of Rhode Island\u2019s tax exemption for boats is not forgotten.In one of the governor\u2019s frequent appearances on Fox News\u2019 \"The Neil Cavuto Show,\" Carcieri touted the positive effect of the sales tax repeal on boat sales, the marine industry and the state\u2019s economy as a whole.\"We abolished the tax on boats back in 1993 and in the intervening period we have over 2,000 little companies that have been spawned, employing almost 7,000 people, generating close to $2 billion a year in sales activity,\" an enthusiastic Carcieri told Cavuto, a conservative pundit from Barrington, R.I., who seems tickled by the Ocean State connection.It\u2019s no wonder Carcieri was excited. With the nation's fourth-highest unemployment rate, Rhode Island rarely gets to brag about any positive job news.And it's clear the sales tax repeal -- enacted under then-Governor Bruce Sundlun -- helped rescue a sector that was taking on water by the early 1990s, having struggled after a boom in previous decades. Following the law's passage, boat registrations in Rhode Island increased every year for two decades before declining slightly during the recent recession.But those are big numbers for a fairly narrow industry and they warrant a careful check.The Carcieri administration told us that prior to the interview, they put out a request to several state agencies for data demonstrating the impact of the tax repeal.The state Economic Development Corporation directed them to two studies by the Rhode Island Marine Trades Association: one from 2008 that focused on the skills gap in the industry and another from 2007 that looked at the effect of the repeal.We got a hold of those reports and it appears the governor is close to correct. The studies collectively find that the Rhode Island marine industry includes 6,600 jobs across 2,300 companies, which generates $1.6 billion a year in sales activity. The numbers don't exactly line up, but they're fairly close to those that Carcieri cited.The sales tax study does not make clear how it arrived at the final tallies and the Marine Trades Association refused to explain its methodology, saying it was confidential.The second study tries harder to explain its approach. It relied on a combination of 136 survey results, as well as U.S. Economic Census numbers and economic modeling, to determine the totals. The researchers started by using the ratio of payroll to sales obtained from the 2002 U.S. Economic Census to determine the number of jobs, companies and total sales in what they call \"core marine trades industries.\" It then appears that they added those numbers to a secondary group of similarly calculated figures that included companies considered to be \"indirect marine trades businesses.\" That math makes sense to us. But then the study offers this caveat: \"In 2006, Rhode Island had approximately 1,700 businesses at least some of whose sales could be attributed to the state's 'direct' marine trades industries. Exactly how much was attributable to marine trades is impossible to determine from employment data reported by the Rhode Island Department of Labor and Training.\"Put another way, there are a lot of companies in this state that may devote some, but not all, of their business to the boating industry. Paul Harden, manager of business and workforce development for the state Economic Development Corporation, offered a hypothetical example of a canvas textile company. It could manufacture boat covers, as well as awnings for shops. One product is clearly connected to the industry, the other is not. That unknown makes it difficult to track the number of jobs created and the business generated within the company specifically for marine work.Based on experience, Harden, who helped oversee the Marine Trades study, said researchers assigned estimated percentages that each subcategory of business devotes to marine sales. He acknowledged that it is an imperfect science.That's where Carcieri's claim gets a little shaky. He suggests the tallies are official, despite the study's stipulation that it is \"impossible\" to pinpoint an exact number.But the bigger issue is that Carcieri says that all of these jobs, businesses and sales activity were \"spawned\" by the 1993 tax repeal on boats, which the study never concludes. It seems clear that the repeal had a positive effect on the industry, but some of the businesses and jobs predated the repeal.\"I don't think you can attribute all the job growth and all the sales activity to the repeal of the sales tax,\" said Harden, of the EDC.Given the study's limits, we tried to verify the numbers independently.We started with the state Division of Taxation. But because boats are not taxed, their sales are not reported to the division, said Tax Administrator David Sullivan. And the department doesn't collect data on marine-related sales.Next we spoke with the state Department of Labor and Training, which collects data on jobs and companies in different industries. Similar to the Marine Trades study, its data shows that in 2009, approximately 4,560 people were employed in \"primary marine industries.\" A second class of \"related industries\" adds about another 1,400 jobs for a total of close to 6,000. But here again, a DLT spokeswoman said it's impossible to know just how many of those related jobs are definitively connected to the marine sector. Realistically, the number probably falls somewhere between 4,560 and 6,000, both of which are lower than the number Carcieri cites.And again, that's total jobs, not those \"spawned\" solely by the sales tax repeal.Carcieri got it right by doing his research before spouting statistics on national television, and the studies he cites offer some measurable data. However, it is impossible to know how precise those numbers are and the governor should have offered that caveat. More troublesome, he should never have said the tax repeal \"spawned\" what amounts to an entire industry. We find this Half True.",
        "taxonomy_label": "statistical",
        "label_original": "half-true"
    }
```

The pipeline consists of claim decomposition, evidence retrieval and stance detection steps.

The following is the project structure

###  :file_folder: File Structure


```
.
├── data
│   ├── decomposed_questions
│   │   ├── test_claimdecomp.csv
│   │   └── test_programfc.json
│   ├── raw_data
│   │   ├── test_claims_quantemp.json
│   │   ├── train_claims_quantemp.json
│   │   └── val_claims_quantemp.json
│   └── bm25_scored_evidence
│       ├── bm25_top_100_claimdecomp.json
│       
├── code
│    ├── data_processing
│        ├──
│    │   ├── test_claimdecomp.csv
│    │   └── test_programfc.json
│    ├── utils
│    │   ├── load_veracity_predictor.py
│    ├── nli_inference
│    │   ├── veracity_prediction.py
├── requirements.txt
└── README.md