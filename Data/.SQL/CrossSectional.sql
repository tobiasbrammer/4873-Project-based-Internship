;WITH Sagsopgaver AS (SELECT
	 jt.[Global Dimension 1 Code] 'Afdeling'
	,jt.[Job No_] Sagsnummer
    ,jt.[Job Task No_] Sagsopgave
    ,j.[Description] Beskrivelse
     FROM [NRGIDW_Extract].[elcon].[Job Task] jt
	 LEFT JOIN [NRGIDW_Extract].[elcon].[Job] j
	 ON jt.[Job No_] = j.No_
     WHERE 1=1
	 AND jt.[Job Posting Group] IN ('FASTPRIS', 'PROJEKT')
	 AND jt.[Global Dimension 1 Code] IN ('421','505','515')
	 AND LEN(jt.[Job Task No_])>1
	 AND jt.[Job Task No_] <> '00'
     AND jt.[Job Task No_] <> '9999'
     AND jt.[Description] NOT LIKE '%Sum%'
	 AND jt.[Job No_] LIKE 'S%'
     ),

Dato AS (SELECT [Job No_],
			 [WIP Date] 'MaxDato'
	  FROM (SELECT [Job No_],
	  [WIP Date],
	  Row_number() OVER(Partition by [Job No_] Order by [WIP Date] desc) as rn
	  FROM [NRGIDW_Extract].[elcon].[JobWIP Details]) t
	  WHERE rn = 1),

Budget AS (SELECT
wd.[Job No_] Sagsnummer,
wd.[Job Task No_] Sagsopgave,
Dato.MaxDato,
SUM(wd.[Sales Estimate Sales]) Tilbudskalk_oms,
SUM(wd.[Sales Estimate Cost]) Tilbudskalk_omk,
SUM(wd.[Estimate Sales]) Produktkalk_oms,
SUM(wd.[Estimate Cost]) Produktkalk_omk,
SUM(wd.[Final Estimate Sales]) Slutvurdering_oms,
SUM(wd.[Final Estimate Cost]) Slutvurdering_omk,
SUM(wd.[Acc_ Deviation Sales] - wd.[Acc_ Deviation Cost]) Ændring
FROM [NRGIDW_Extract].[elcon].[JobWIP Details] wd
JOIN Dato On wd.[Job No_] = Dato.[Job No_]
WHERE 1=1
AND wd.[WIP Date] = Dato.[MaxDato]
GROUP BY wd.[Job No_], wd.[Job Task No_],Dato.[MaxDato]),

Realiseret AS (SELECT
		jle.[Job No_],
    	jle.[Job Task No_] As Sagsopgave,
    	SUM(CASE WHEN jle.[Entry Type] = 1 THEN (-jle.[Line Amount (LCY)])
    			 WHEN jle.[Entry Type] = 0 THEN -jle.[Total Cost (LCY)]
    			 ELSE 0
    			 END) AS 'Realiseret',
	   CASE WHEN jle.[Job Task No_] = '9000' THEN
		    	SUM(CASE WHEN jle.[Entry Type] = 1 THEN (-jle.[Line Amount (LCY)])
    			 WHEN jle.[Entry Type] = 0 THEN -jle.[Total Cost (LCY)]
    			 ELSE 0
    			 END)
			ELSE 0
			END AS 'Regningsarbejde'
    	FROM [NRGIDW_Extract].[elcon].[Job Ledger Entry] jle
    	WHERE 1=1
    	GROUP BY jle.[Job No_], jle.[Job Task No_])

SELECT
Sagsopgaver.Afdeling,
Sagsopgaver.Sagsnummer,
Budget.[MaxDato] 'Senest opdateret',
SUM(Tilbudskalk_oms) 'Tilbudskalk_oms',
SUM(Tilbudskalk_omk) 'Tilbudskalk_omk',
SUM(Produktkalk_oms) 'Produktkalk_oms',
SUM(Produktkalk_omk) 'Produktkalk_omk',
SUM(Ændring) 'Ændring_col',
SUM(Slutvurdering_oms) 'Slutvurdering_oms',
SUM(Slutvurdering_omk) 'Slutvurdering_omk', 
SUM(Realiseret) 'Realiseret_col',
SUM(Regningsarbejde) 'Regningsarbejde_col'
FROM Sagsopgaver
LEFT JOIN Budget
ON CONCAT(Budget.Sagsnummer, Budget.[Sagsopgave]) = CONCAT(Sagsopgaver.Sagsnummer, Sagsopgaver.[Sagsopgave])
LEFT JOIN Realiseret
ON CONCAT(Sagsopgaver.Sagsnummer, Sagsopgaver.[Sagsopgave]) = CONCAT(Realiseret.[Job No_],Realiseret.[Sagsopgave])
WHERE 1=1
AND NOT Budget.[MaxDato] IS NULL
GROUP BY Sagsopgaver.Afdeling, Sagsopgaver.Sagsnummer,Sagsopgaver.Beskrivelse, Budget.[MaxDato] 
ORDER BY Budget.[MaxDato], Sagsopgaver.Afdeling, Sagsopgaver.Sagsnummer
