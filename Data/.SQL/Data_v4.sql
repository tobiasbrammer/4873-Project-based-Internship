set nocount on
        SELECT 
			[No_], [Status], [Global Dimension 1 Code], [Job Posting Group], [Ending Date], [Description], [Bill-to Customer No_], [Person Responsible],
            [Ship-to Address], [Ship-to Post Code], [Ship-to City]
		INTO #Sager
        FROM [NRGIDW_Extract].[elcon].[Job]
        WHERE [Status] IN (2,3)
        AND [Global Dimension 1 Code] IN ('515','505','421')
        AND [Job Posting Group] IN ('FASTPRIS','PROJEKT')

        SELECT
			Sagsopgaver.[Global Dimension 1 Code], Sagsopgaver.[Job No_], Sagsopgaver.[Job Task No_], Sagsopgaver.[Description]
		INTO #Sagsopgaver
        FROM [NRGIDW_Extract].[elcon].[Job Task] AS Sagsopgaver
        INNER JOIN #Sager ON #Sager.[No_] = Sagsopgaver.[Job No_]


		SELECT 
			Arbejdssedler.[Source No_]
			,MAX(Arbejdssedler.[Completion Date]) AS 'Dato'
		INTO #Arbejdssedler
		FROM [NRGIDW_Extract].[elcon].[Work Order Header] AS Arbejdssedler
		INNER JOIN #Sager
		ON #Sager.[No_] = Arbejdssedler.[Source No_]
		GROUP BY Arbejdssedler.[Source No_]



		SELECT 
			Sagsopgaver.[Job No_]
			,SUM(CASE Sagsbudget.[Line Type] WHEN 1 THEN Sagsbudget.[Line Amount (LCY)] WHEN 2 THEN [Line Amount (LCY)] ELSE 0 END) AS 'Indtaegtsbudget'
			,SUM(CASE Sagsbudget.[Line Type] WHEN 0 THEN Sagsbudget.[Total Cost (LCY)] WHEN 2 THEN [Total Cost (LCY)] ELSE 0 END) AS 'Omkostningsbudget'
			,Sager.[Ending Date] AS 'Slutdato'
		INTO #Sagsbudget
		FROM [NRGIDW_Extract].[elcon].[Job Planning Line Entry] AS Sagsbudget
		Inner JOIN #Sagsopgaver Sagsopgaver
		ON CONCAT(Sagsbudget.[Job No_],Sagsbudget.[Job Task No_]) = CONCAT(Sagsopgaver.[Job No_],Sagsopgaver.[Job Task No_])
		INNER JOIN #Sager Sager
		ON Sager.[No_]=Sagsbudget.[Job No_]
		GROUP BY 
		Sagsopgaver.[Job No_], Sager.[Ending Date]


		SELECT
  			FORMAT(Sagsposter.[Posting Date], 'MM') AS 'month'
			,FORMAT(Sagsposter.[Posting Date], '01-MM-yyyy') AS 'date'
  			,FORMAT(Sagsposter.[Posting Date], 'MM-yyyy') AS 'month-year'
  			,FORMAT(Sagsposter.[Posting Date], 'yyyy') AS 'year'
  			,Sagsposter.[Global Dimension 1 Code] AS 'department'
			,Sagsposter.[Job No_]
			,-SUM(CASE Sagsposter.[Entry Type] WHEN 1 THEN Sagsposter.[Line Amount (LCY)] ELSE 0 END) AS 'revenue'
			,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END) AS 'costs'
			,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 0 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS 'costs_of_labor'
			,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 1 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS 'costs_of_materials'
			,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 2 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS  'other_costs'
		INTO #Sagsposter
		FROM [NRGIDW_Extract].[elcon].[Job Ledger Entry] AS Sagsposter
		INNER JOIN #Sagsopgaver Sagsopgaver
		ON CONCAT(Sagsposter.[Job No_],Sagsposter.[Job Task No_]) = CONCAT(Sagsopgaver.[Job No_],Sagsopgaver.[Job Task No_])
		INNER JOIN #Sager Sager
		ON Sager.[No_]=Sagsposter.[Job No_]
		WHERE 1=1
		AND Year(Sagsposter.[Posting Date]) >= 2018
		GROUP BY 
		Sagsposter.[Global Dimension 1 Code],
		Sagsposter.[Job No_],
		Sagsposter.[Posting Date],
		Sagsposter.[Entry Type]
	
		SELECT 
			[date],
			[month],
			[year],
			[month-year],
			[department],
			[Job No_],
			SUM(revenue) 'revenue',
			SUM(costs) 'costs',
			SUM(costs_of_labor) 'costs_of_labor',
			SUM(costs_of_materials) 'costs_of_materials',
			SUM(other_costs) 'other_costs'
		INTO #Regnskab
		FROM #Sagsposter Sagsposter
		GROUP BY
		[date],
		[month],
		[year],
		[month-year],
		[department],
		[Job No_]

		SELECT 
			FORMAT([WIP Date],'yyyy-MM-01') AS 'date',
			[Job No_] AS 'job_no',
			MAX([Archive No_]) AS max_archive_no
		INTO #MaxArchive
		FROM [NRGIDW_Extract].[elcon].[JobWIP Details]
		INNER JOIN #Sager Sager ON Sager.No_ = [JobWIP Details].[Job No_]
		WHERE [Job Task Type] = 0
		GROUP BY FORMAT([WIP Date],'yyyy-MM-01'), [Job No_]

		SELECT DISTINCT
		[Month]
		INTO #Cal
		FROM [ElconDW].[dim].[Calendar]
		WHERE [Year] >= '2018'

		SELECT
			FORMAT(j.[WIP Date],'yyyy-MM-01') AS 'date',
			j.[Job No_] AS 'job_no',
			j.[Sales Estimate Cost] AS 'sales_estimate_cost',
			j.[Sales Estimate Sales] AS 'sales_estimate_sales',
			j.[Estimate Cost] AS 'estimate_cost',
			j.[Estimate Sales] AS 'estimate_sales',
			j.[Final Estimate Cost] AS 'final_estimate_cost',
			j.[Final Estimate Sales] AS 'final_estimate_sales'
		INTO #Budget
		FROM [NRGIDW_Extract].[elcon].[JobWIP Details] j
		INNER JOIN #Sager Sager ON Sager.No_ = j.[Job No_]
		JOIN #MaxArchive m ON j.[Job No_] = m.job_no AND FORMAT(j.[WIP Date], 'yyyy-MM-01') = m.date AND j.[Archive No_] = m.max_archive_no
		WHERE [Job Task Type] = 0

		SELECT 
			date,
			job_no,
			SUM(sales_estimate_cost) AS 'sales_estimate_cost',
			SUM(sales_estimate_sales) AS 'sales_estimate_sales',
			SUM(estimate_cost) AS 'estimate_cost',
			SUM(estimate_sales) AS 'estimate_sales',
			SUM(final_estimate_cost) AS 'final_estimate_cost',
			SUM(final_estimate_sales) AS 'final_estimate_sales'
		INTO #budget_v2
		FROM #budget budget
		GROUP BY date, job_no

		SELECT 
			job_no,
			[Month] AS 'date'
		INTO #AllCombinations
		FROM #budget_v2
		CROSS JOIN #Cal

	SELECT 
    FORMAT(ac.date,'01-MM-yyyy') 'date',
    ac.job_no,
    COALESCE(b.sales_estimate_cost, la.sales_estimate_cost) AS 'sales_estimate_cost',
    COALESCE(b.sales_estimate_sales, la.sales_estimate_sales) AS 'sales_estimate_sales',
    COALESCE(b.estimate_cost, la.estimate_cost) AS 'estimate_cost',
    COALESCE(b.estimate_sales, la.estimate_sales) AS 'estimate_sales',
    COALESCE(b.final_estimate_cost, la.final_estimate_cost) AS 'final_estimate_cost',
    COALESCE(b.final_estimate_sales, la.final_estimate_sales) AS 'final_estimate_sales'
	INTO #Budget_final
	FROM #AllCombinations ac
	LEFT JOIN #budget_v2 b ON ac.date = b.date AND ac.job_no = b.job_no
	OUTER APPLY (
		SELECT 
			sales_estimate_cost,
			sales_estimate_sales,
			estimate_cost,
			estimate_sales,
			final_estimate_cost,
			final_estimate_sales
		FROM #budget_v2 b3
		WHERE b3.job_no = ac.job_no AND b3.date = (
			SELECT MAX(b4.date)
			FROM #budget_v2 b4
			WHERE b4.job_no = ac.job_no AND b4.date < ac.date
		)
	) la
	WHERE YEAR(ac.date) >= 2018


	SELECT  
	FORMAT(rle.[Posting Date], '01-MM-yyyy') 'date'
	,rle.[Job No_] 'job_no'
	,CASE
		WHEN ad.title LIKE '%Lærling%' THEN 'Kun Lærlinge'
		ELSE 'Ekskl. lærlinge'
	END AS 'MedarbejderType'
	,CASE tt.Tidstype WHEN 'Fakturerbar tid' THEN rle.Quantity ELSE 0 END 'billable_hours_qty'
	,CASE tt.Tidstype WHEN 'Fakturerbar tid' THEN rle.[Total Cost] ELSE 0 END 'billable_hours'
	,CASE tt.Tidstype WHEN 'Optjent afspadsering' THEN rle.Quantity ELSE 0 END 'earned_time_off_qty'
	,CASE tt.Tidstype WHEN 'Optjent afspadsering' THEN rle.[Total Cost] ELSE 0 END 'earned_time_off'
	,CASE tt.Tidstype WHEN 'Overtid' THEN rle.Quantity ELSE 0 END 'over_time_qty'
	,CASE tt.Tidstype WHEN 'Overtid' THEN rle.[Total Cost] ELSE 0 END 'over_time'
	,CASE tt.Tidstype WHEN 'Tillæg' THEN rle.Quantity ELSE 0 END 'allowance_qty'
	,CASE tt.Tidstype WHEN 'Tillæg' THEN rle.[Total Cost] ELSE 0 END 'allowance'
  INTO #RLE
  FROM [NRGIDW_Extract].[elcon].[Res_ Ledger Entry] rle
  LEFT JOIN [NRGIDW_Extract].[elcon].[TidsTyper] tt ON tt.[Kode] = rle.[Work Type Code]
  LEFT JOIN [NRGIDW_Extract].[elcon].[AD] ad ON rle.[Resource No_] = ad.employeeNumber
  INNER JOIN #Sager ON #Sager.No_ = rle.[Job No_]
  WHERE [Entry Type] = 0
  AND tt.Tidstype IN ('Fakturerbar tid','Optjent afspadsering','Overtid','Tillæg')

  SELECT 
  date
  ,job_no
  ,SUM(billable_hours_qty) 'billable_hours_qty'
  ,SUM(earned_time_off_qty) 'earned_time_off_qty'
  ,SUM(over_time_qty) 'over_time_qty'
  ,SUM(allowance_qty) 'allowance_qty'
  ,SUM(billable_hours) 'billable_hours'
  ,SUM(earned_time_off) 'earned_time_off'
  ,SUM(over_time) 'over_time'
  ,SUM(allowance) 'allowance'
  INTO #rle_2
  FROM #RLE
  GROUP BY date
  ,job_no



-- DEPARTMENT BILLABLE RATE
	SELECT 
	rle.[Resource No_] Arbejder
	,dv.[Consolidation Code] AS 'Division'
	,dv.[Code] AS 'Afdeling'
	,rle.[Posting Date] AS 'Dato'
	,CASE
	  WHEN tt.Tidstype = 'Fakturerbar tid' THEN 'Fakturerbar tid'
	  WHEN tt.Tidstype = 'Sygdom' THEN 'Sygdom'
	  WHEN tt.Tidstype = 'Kursus' THEN 'Kursus'
	  WHEN tt.Tidstype = 'Intern tid' THEN 'Intern tid'
	  WHEN tt.Tidstype = 'Skoleophold' THEN 'Skoleophold'
 	  WHEN rle.[Work Type Code] IN ('2060', '3010', '3015','4050','4055','4071','4075','4080','4120','4122','4920') THEN 'Anden fravær'
 	  WHEN rle.[Work Type Code] IN ('4030','4035','4040') THEN 'Barsel og orlov'
 	  ELSE 'Bruges ikke'
	END AS 'Tidstype'
	,(rle.Quantity) Timer
	INTO #ArbejderTid
	FROM [NRGIDW_Extract].[elcon].[Res_ Ledger Entry] rle
	LEFT JOIN [NRGIDW_Extract].[elcon].[TidsTyper] tt ON rle.[Work Type Code] = tt.[Kode]
	LEFT JOIN [NRGIDW_Extract].[elcon].[Dimension Value] dv ON rle.[Global Dimension 1 Code] = dv.Code
	WHERE rle.[Entry Type] = 0
	AND dv.[Dimension Code] = 'AFDELING'
	AND dv.[Code] IN ('421','505','515')
	AND Tidstype <> 'Bruges ikke'
	 --GROUP BY rle.[Resource No_], dv.[Consolidation Code],[Code], rle.[Posting Date], Tidstype
	 --HAVING (Quantity) <> 0
	 
	
	SELECT DISTINCT
	  Arbejder,
	  Division,
	  Afdeling,
	  DATEADD(month, DATEDIFF(month, 0, EOMONTH([Dato])), 0) Dato,
	  (CASE WHEN Tidstype = 'Fakturerbar tid' OR
					Tidstype = 'Sygdom' OR
					Tidstype = 'Kursus' OR
					Tidstype = 'Intern tid' THEN SUM(Timer) ELSE 0 END) 'Jobtimer',
	  (CASE WHEN Tidstype = 'Fakturerbar tid' THEN SUM(Timer) ELSE 0 END) 'Fakturerbar tid',
	  (CASE WHEN Tidstype = 'Sygdom' THEN SUM(Timer) ELSE 0 END) 'Sygdom',
	  CASE WHEN (CASE WHEN Tidstype = 'Sygdom' THEN SUM(Timer) ELSE 0 END) >= 74 THEN (CASE WHEN Tidstype = 'Sygdom' THEN SUM(Timer) ELSE 0 END) ELSE 0 END AS 'Langtidssygdom',
	  (CASE WHEN Tidstype = 'Kursus' THEN SUM(Timer) ELSE 0 END) 'Kursus',
	  (CASE WHEN Tidstype = 'Intern tid' THEN SUM(Timer) ELSE 0 END) 'Intern tid',
	  (CASE WHEN Tidstype = 'Skoleophold' THEN SUM(Timer) ELSE 0 END) 'Skoleophold',
	  (CASE WHEN Tidstype = 'Anden fravær' THEN SUM(Timer) ELSE 0 END) 'Anden fravær',
	  (CASE WHEN Tidstype = 'Barsel og orlov' THEN SUM(Timer) ELSE 0 END) 'Barsel og orlov'
	INTO #ArbejderTimer
	FROM #ArbejderTid 
	GROUP BY
	  Arbejder,
	  Division,
	  Afdeling,
	  DATEADD(month, DATEDIFF(month, 0, EOMONTH([Dato])), 0),
	  Tidstype

	SELECT 
	Afdeling, Dato,
	SUM(Jobtimer) 'Jobtimer',
	SUM([Fakturerbar tid]) 'Fakturerbar tid',
	SUM(Sygdom) 'Sygdom',
	SUM(Langtidssygdom) 'Langtidssygdom',
	SUM(Kursus) 'Kursus',
	SUM([Intern tid]) 'Intern tid',
	SUM(Skoleophold) 'Skoleophold',
	SUM([Anden fravær]) 'Anden fravær',
	SUM([Barsel og orlov]) 'Barsel og orlov'
	INTO #Faktureringsgrad
	FROM #ArbejderTimer
	GROUP BY Afdeling, Dato




 SELECT DISTINCT
	 Sagsposter.[date]
	,Sagsposter.[month]
	,Sagsposter.[year]
    ,Sager.[Job Posting Group] AS 'job_posting_group'
	,CASE
		WHEN Sagsposter.[department]  = '421' THEN '505'
		ELSE Sagsposter.[department]
	END AS 'department'
    ,Sager.[No_] AS 'job_no'
	,CASE Sager.[Status] WHEN 2 THEN 'wip' ELSE 'finished' END AS 'status'
    ,Sager.[Description] AS 'description'
	,Kunder.[No_] AS 'customer'
	,Kunder.[VAT Registration No_] AS 'cvr'
	,Kunder.[Post Code] AS 'customer_zip'
    ,CONCAT(Sager.[Ship-to Address],' ',Sager.[Ship-to Post Code],' ',Sager.[Ship-to City]) AS 'address'
	,Sager.[Ship-to Post Code] AS 'zip'
    ,Medarbejdere.[No_] AS 'responsible'
	,CASE 
		WHEN Sager.[Status] = 3 THEN
			CASE 
				WHEN ISNULL(Sagsbudget.Slutdato, 0) = 0 THEN  FORMAT(CAST(Arbejdssedler.Dato AS date), '01-MM-yyyy')
				ELSE FORMAT(CAST(Sagsbudget.Slutdato AS date), '01-MM-yyyy')
			END
		ELSE FORMAT(CAST(Sagsbudget.Slutdato AS date), '01-MM-yyyy')
	END AS 'end_date'
	,ISNULL(Budget_final.sales_estimate_sales - Budget_final.sales_estimate_cost,0) 'sales_estimate_contribution'
	,ISNULL(Budget_final.estimate_sales - Budget_final.estimate_cost,0) 'production_estimate_contribution'
	,ISNULL(Budget_final.final_estimate_sales - Budget_final.final_estimate_cost,0) 'final_estimate_contribution'
    ,(ISNULL(Sagsbudget.[Indtaegtsbudget],0)) AS 'budget_revenue'
    ,(ISNULL(Sagsbudget.[Omkostningsbudget],0)) AS 'budget_costs'
    ,((ISNULL(Sagsbudget.[Indtaegtsbudget],0)) - (ISNULL(Sagsbudget.[Omkostningsbudget],0))) AS 'budget_contribution'
    ,(ISNULL(Sagsposter.[revenue],0)) AS 'revenue'
	,(ISNULL(Sagsposter.[revenue],0))/(NULLIF(Sagsbudget.[Indtaegtsbudget],0)) AS 'revenue_budget_share'
    ,(ISNULL(Sagsposter.costs,0)) 'costs'
	,(ISNULL(Sagsposter.costs,0))/(NULLIF(Sagsbudget.[Omkostningsbudget],0)) AS 'costs_budget_share'
    ,(ISNULL(Sagsposter.costs_of_labor,0))/(NULLIF(Sagsposter.costs,0)) AS 'costs_of_labor_share'
    ,(ISNULL(Sagsposter.costs_of_materials,0))/(NULLIF(Sagsposter.costs,0)) AS 'costs_of_materials_share'
    ,(ISNULL(Sagsposter.other_costs,0))/(NULLIF(Sagsposter.costs,0)) AS  'other_costs_share'
	,(ISNULL(Sagsposter.revenue,0) - ISNULL(Sagsposter.costs,0))/(NULLIF(Sagsposter.revenue,0)) AS 'contribution_margin'
	,(ISNULL(Sagsposter.revenue,0) - ISNULL(Sagsposter.costs,0)) AS 'contribution'
	,(CASE 
        WHEN 
            ((ISNULL(Sagsposter.costs,0)) = 0 AND (ISNULL(Sagsbudget.[Omkostningsbudget],0)) = 0) 
            OR ISNULL((ISNULL(Sagsbudget.[Indtaegtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF((Sagsbudget.[Indtaegtsbudget]),0),0) >= 1 
        THEN (ISNULL(Sagsbudget.[Indtaegtsbudget],0)) 
        ELSE 
            CASE 
                WHEN ((ISNULL(Sagsposter.costs,0))/NULLIF((1-ISNULL((ISNULL(Sagsbudget.[Indtaegtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF((Sagsbudget.[Indtaegtsbudget]),0),0)), 0)) > (ISNULL(Sagsbudget.[Indtaegtsbudget],0)) 
                THEN (ISNULL(Sagsbudget.[Indtaegtsbudget],0)) 
                ELSE ((ISNULL(Sagsposter.costs,0))/NULLIF((1-ISNULL((ISNULL(Sagsbudget.[Indtaegtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF((Sagsbudget.[Indtaegtsbudget]),0),0)), 0))
            END 
    END) AS 'estimated_revenue'
	,(CASE 
        WHEN 
            (Sagsposter.costs = 0 AND Sagsbudget.[Omkostningsbudget] = 0) 
            OR ISNULL((ISNULL(Sagsbudget.[Indtaegtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtaegtsbudget],0),0) >= 1 
        THEN (ISNULL(Sagsbudget.[Indtaegtsbudget],0)) 
        ELSE 
            CASE 
                WHEN ((ISNULL(Sagsposter.costs,0))/NULLIF((1-ISNULL((ISNULL(Sagsbudget.[Indtaegtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF((Sagsbudget.[Indtaegtsbudget]),0),0)), 0)) > (ISNULL(Sagsbudget.[Indtaegtsbudget],0)) 
                THEN (ISNULL(Sagsbudget.[Indtaegtsbudget],0)) 
                ELSE ((ISNULL(Sagsposter.costs,0))/NULLIF((1-ISNULL((ISNULL(Sagsbudget.[Indtaegtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF((Sagsbudget.[Indtaegtsbudget]),0),0)), 0))
            END 
    END) - (ISNULL(Sagsposter.costs,0)) AS 'estimated_contribution'
	,ISNULL(#rle_2.billable_hours_qty,0) 'billable_hours_qty'
	,ISNULL(#rle_2.earned_time_off_qty,0)/NULLIF(ISNULL(#rle_2.billable_hours_qty,0),0) 'earned_time_off_rate'
	,ISNULL(#rle_2.over_time_qty,0)/NULLIF(ISNULL(#rle_2.billable_hours_qty,0),0) 'over_time_rate'
	,ISNULL(#rle_2.allowance_qty,0)/NULLIF(ISNULL(#rle_2.billable_hours_qty,0),0) 'allowance_rate'
	--,ISNULL(#Faktureringsgrad.[Jobtimer],0) 'work_hours_dep'
	--,ISNULL(#Faktureringsgrad.[Fakturerbar tid],0) 'billable_dep'
	,(ISNULL(#Faktureringsgrad.[Fakturerbar tid],0)/NULLIF(ISNULL(#Faktureringsgrad.[Jobtimer],0),0)) 'billable_rate_dep'
	,(ISNULL(#Faktureringsgrad.[Sygdom],0)/NULLIF((ISNULL(#Faktureringsgrad.[Jobtimer],0)+ISNULL(#Faktureringsgrad.[Skoleophold],0)),0)) 'illness_rate_dep'
	,(ISNULL(#Faktureringsgrad.[Langtidssygdom],0)/NULLIF((ISNULL(#Faktureringsgrad.[Jobtimer],0)+ISNULL(#Faktureringsgrad.[Skoleophold],0)),0)) 'long_term_illness_rate_dep'
	,(ISNULL(#Faktureringsgrad.[Intern tid],0)/NULLIF(ISNULL(#Faktureringsgrad.[Jobtimer],0),0)) 'internal_rate_dep'
	,(ISNULL(#Faktureringsgrad.[Langtidssygdom],0)/NULLIF(ISNULL(#Faktureringsgrad.[Sygdom],0),0)) 'long_term_illness_share_dep'
--	,ISNULL(#rle_2.billable_hours,0) 'billable_hours'
--	,ISNULL(#rle_2.earned_time_off,0) 'earned_time_off'
--	,ISNULL(#rle_2.over_time,0) 'over_time'
--	,ISNULL(#rle_2.allowance,0) 'allowance'
    FROM #Regnskab as Sagsposter
    Left JOIN #Sagsopgaver Sagsopgaver
    ON  Sagsopgaver.[Job No_] = Sagsposter.[Job No_]
    INNER JOIN #Sager Sager
    ON Sager.[No_]=Sagsopgaver.[Job No_]
    LEFT JOIN [NRGIDW_Extract].[elcon].[Customer] AS Kunder
    ON Kunder.[No_]=Sager.[Bill-to Customer No_]
    LEFT JOIN [NRGIDW_Extract].[elcon].[Resource] AS Medarbejdere
    ON Medarbejdere.[No_]=Sager.[Person Responsible]
    LEFT JOIN #Sagsbudget Sagsbudget
    ON Sagsopgaver.[Job No_] = Sagsbudget.[Job No_]
    LEFT JOIN #Arbejdssedler Arbejdssedler
    ON Arbejdssedler.[Source No_] = Sagsopgaver.[Job No_]
	LEFT JOIN #Budget_final Budget_final
	ON CONCAT(Budget_final.[date],Budget_final.[job_no]) = CONCAT(Sagsposter.[date],Sagsposter.[Job No_])
	LEFT JOIN #rle_2
	ON CONCAT(#rle_2.[date],#rle_2.[job_no]) = CONCAT(Sagsposter.[date],Sagsposter.[Job No_])
	LEFT JOIN #Faktureringsgrad
	ON CONCAT(Sagsposter.[date], CASE WHEN Sagsposter.[department]  = '421' THEN '505' ELSE Sagsposter.[department] END) = CONCAT(FORMAT(#Faktureringsgrad.Dato,'dd-MM-yyyy'),#Faktureringsgrad.Afdeling)
    WHERE 1=1
	AND (CASE WHEN ISNULL(Sagsbudget.[Indtaegtsbudget],0) = 0 THEN 0 ELSE 1 END +
		 CASE WHEN ISNULL(Sagsbudget.[Omkostningsbudget],0) = 0 THEN 0 ELSE 1 END +
		 CASE WHEN ISNULL(Sagsposter.costs,0) = 0 THEN 0 ELSE 1 END +
		 CASE WHEN ISNULL(Sagsposter.revenue,0) = 0 THEN 0 ELSE 1 END) <> 0
	AND Sagsposter.[department] IN ('421','515','505')
	ORDER BY Sager.[No_], [year], [month] ASC



	  
DROP TABLE #Sager, #Sagsbudget, #AllCombinations, #Arbejdssedler, #Budget, #Budget_final, #budget_v2, #Cal, #MaxArchive, #Regnskab, #Sagsopgaver, #Sagsposter, #RLE, #rle_2, #ArbejderTid, #ArbejderTimer, #Faktureringsgrad
