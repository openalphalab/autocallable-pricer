# Data Download Instructions

This folder stores the input market data used by the project.

## Rates data

1. Open the rates repository:
   `https://www.dolthub.com/repositories/post-no-preference/rates/data/master`
2. Go to the `Query` section on the website.
3. Run this SQL:

```sql
SELECT *
FROM `us_treasury`
ORDER BY `date` ASC
```

4. Start a job so the full result set can be exported without row limits.
5. Download the job output as CSV.
6. Save the file in this folder as `rates.csv`.

## SPY option data

1. Open the options repository:
   `https://www.dolthub.com/repositories/post-no-preference/options/data/master`
2. Go to the `Query` section on the website.
3. Run this SQL:

```sql
SELECT *
FROM `option_chain`
WHERE `act_symbol` = 'SPY'
ORDER BY `date` ASC, `act_symbol` ASC, `expiration` ASC, `strike` ASC, `call_put` ASC
```

4. Start a job so the full result set can be exported without row limits.
5. Download the job output as CSV.
6. Save the file in this folder as `SPY.csv`.

## Notes

- These exports should be downloaded from the DoltHub website, not from the limited in-browser query preview.
- If the UI offers both direct query results and job execution, use job execution for unlimited CSV export.
