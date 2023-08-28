    ;WITH Sager AS (
        SELECT [No_], [Status], [Global Dimension 1 Code], [Job Posting Group], [Ending Date], [Description], [Bill-to Customer No_], [Person Responsible],
           [Ship-to Address], [Ship-to Post Code], [Ship-to City]
        FROM [NRGIDW_Extract].[elcon].[Job]
        WHERE [Status] IN (2,3)
        AND [Global Dimension 1 Code] IN ('515','505','421')
        AND [Job Posting Group] IN ('FASTPRIS','PROJEKT')
    ),

    Sagsopgaver AS(
        SELECT Sagsopgaver.[Global Dimension 1 Code], Sagsopgaver.[Job No_], Sagsopgaver.[Job Task No_], Sagsopgaver.[Description]
        FROM [NRGIDW_Extract].[elcon].[Job Task] AS Sagsopgaver
        INNER JOIN Sager ON Sager.[No_] = Sagsopgaver.[Job No_]
    ),

    Sagsbudget AS(
    SELECT 
    Sagsopgaver.[Job No_]
    ,SUM(CASE Sagsbudget.[Line Type] WHEN 1 THEN Sagsbudget.[Line Amount (LCY)] WHEN 2 THEN [Line Amount (LCY)] ELSE 0 END) AS 'Indtaegtsbudget'
    ,SUM(CASE Sagsbudget.[Line Type] WHEN 0 THEN Sagsbudget.[Total Cost (LCY)] WHEN 2 THEN [Total Cost (LCY)] ELSE 0 END) AS 'Omkostningsbudget'
	,Sager.[Ending Date] AS 'Slutdato'
    FROM [NRGIDW_Extract].[elcon].[Job Planning Line Entry] AS Sagsbudget
    Inner JOIN Sagsopgaver
    ON CONCAT(Sagsbudget.[Job No_],Sagsbudget.[Job Task No_]) = CONCAT(Sagsopgaver.[Job No_],Sagsopgaver.[Job Task No_])
    INNER JOIN Sager
    ON Sager.[No_]=Sagsbudget.[Job No_]
    GROUP BY 
    Sagsopgaver.[Job No_], Sager.[Ending Date]),

    Arbejdssedler AS(
    SELECT 
    Arbejdssedler.[Source No_]
    ,Count(Arbejdssedler.[Source No_]) AS Antal
	,MAX(Arbejdssedler.[Completion Date]) AS 'Dato'
    FROM [NRGIDW_Extract].[elcon].[Work Order Header] AS Arbejdssedler
    INNER JOIN Sager
    ON Sager.[No_] = Arbejdssedler.[Source No_]
    WHERE 1=1
	  AND (NOT(Arbejdssedler.[Status] = 3) OR Arbejdssedler.[Status] IS NULL)
    GROUP BY Arbejdssedler.[Source No_]),

    Sagsposter AS(
    SELECT
  	FORMAT(Sagsposter.[Posting Date], 'MM') AS 'month'
  	,FORMAT(Sagsposter.[Posting Date], 'MM-yyyy') AS 'month-year'
  	,FORMAT(Sagsposter.[Posting Date], 'yyyy') AS 'year'
  	,Sagsposter.[Global Dimension 1 Code] AS 'department'
    ,Sagsposter.[Job No_]
    ,-SUM(CASE Sagsposter.[Entry Type] WHEN 1 THEN Sagsposter.[Line Amount (LCY)] ELSE 0 END) AS 'revenue'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END) AS 'costs'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 0 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS 'costs_of_labor'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 1 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS 'costs_of_materials'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 2 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS  'other_costs'												
    FROM [NRGIDW_Extract].[elcon].[Job Ledger Entry] AS Sagsposter
    INNER JOIN Sagsopgaver
    ON CONCAT(Sagsposter.[Job No_],Sagsposter.[Job Task No_]) = CONCAT(Sagsopgaver.[Job No_],Sagsopgaver.[Job Task No_])
    INNER JOIN Sager
    ON Sager.[No_]=Sagsposter.[Job No_]
    WHERE 1=1
	AND Year(Sagsposter.[Posting Date]) >= 2018
    GROUP BY 
	Sagsposter.[Global Dimension 1 Code],
    Sagsposter.[Job No_],
	Sagsposter.[Posting Date],
	Sagsposter.[Entry Type]
	),


	Regnskab AS (
	SELECT 
	[month],
	[year],
	[month-year],
	[department],
	[Job No_],
	SUM(revenue) 'revenue',
	SUM(costs) 'costs',
	SUM(costs_of_labor) 'costs_of_labor',
	SUM(costs_of_materials) 'costs_of_materials',
	SUM(other_costs) 'other_costs',
	SUM(SUM(costs_of_labor)) OVER(PARTITION BY [department] ORDER BY [year], [month]) / 
        NULLIF(SUM(SUM(costs)) OVER(PARTITION BY [department] ORDER BY [year], [month]), 0) AS 'labor_cost_share',
	SUM(SUM(costs_of_materials)) OVER(PARTITION BY [department] ORDER BY [year], [month]) / 
       NULLIF(SUM(SUM(costs)) OVER(PARTITION BY [department] ORDER BY [year], [month]), 0) AS 'material_cost_share'
	FROM Sagsposter
	GROUP BY
	[month],
	[year],
	[month-year],
	[department],
	[Job No_])

 SELECT DISTINCT
 CAST(CONCAT('01-',Sagsposter.[month],'-',Sagsposter.[year]) as date) 'date'
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
				WHEN ISNULL(Sagsbudget.Slutdato, 0) = 0 THEN Arbejdssedler.Dato
				ELSE CAST(Sagsbudget.Slutdato AS date)
			END
		ELSE CAST(Sagsbudget.Slutdato AS date) 
	END AS 'end_date'
    ,(ISNULL(Sagsbudget.[Indtaegtsbudget],0)) AS 'budget_revenue'
    ,(ISNULL(Sagsbudget.[Omkostningsbudget],0)) AS 'budget_costs'
    ,((ISNULL(Sagsbudget.[Indtaegtsbudget],0)) - (ISNULL(Sagsbudget.[Omkostningsbudget],0))) AS 'budget_contribution'
    ,(ISNULL(Sagsposter.[revenue],0)) AS 'revenue'
	,(ISNULL(Sagsposter.[revenue],0))/(NULLIF(Sagsbudget.[Indtaegtsbudget],0)) AS 'revenue_budget_share'
    ,(ISNULL(Sagsposter.costs,0)) 'costs'
	,(ISNULL(Sagsposter.costs,0))/(NULLIF(Sagsbudget.[Omkostningsbudget],0)) AS 'costs_budget_share'
    ,(ISNULL(Sagsposter.costs_of_labor,0)) AS 'costs_of_labor'
    ,(ISNULL(Sagsposter.costs_of_materials,0)) AS 'costs_of_materials'
    ,(ISNULL(Sagsposter.other_costs,0)) AS  'other_costs'
	,(ISNULL(Sagsposter.labor_cost_share,0)) AS 'labor_cost_share'
	,(ISNULL(Sagsposter.material_cost_share,0)) AS 'material_cost_share'
	,(1-((ISNULL(Sagsposter.labor_cost_share,0))+(ISNULL(Sagsposter.material_cost_share,0)))) AS 'other_cost_share'
	,(ISNULL(Sagsposter.labor_cost_share,0))*(ISNULL(Sagsbudget.[Omkostningsbudget],0)) AS 'budget_labor_cost'
	,(ISNULL(Sagsposter.material_cost_share,0))*(ISNULL(Sagsbudget.[Omkostningsbudget],0)) AS 'budget_labor_cost'
	,(1-((ISNULL(Sagsposter.labor_cost_share,0))+(ISNULL(Sagsposter.material_cost_share,0))))*(ISNULL(Sagsbudget.[Omkostningsbudget],0)) AS 'budget_other_cost'
	,(ISNULL(Sagsposter.revenue,0)) - (ISNULL(Sagsposter.costs,0)) AS 'contribution'
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
    FROM Regnskab as Sagsposter
    Left JOIN Sagsopgaver
    ON  Sagsopgaver.[Job No_] = Sagsposter.[Job No_]
    INNER JOIN Sager
    ON Sager.[No_]=Sagsopgaver.[Job No_]
    LEFT JOIN [NRGIDW_Extract].[elcon].[Customer] AS Kunder
    ON Kunder.[No_]=Sager.[Bill-to Customer No_]
    LEFT JOIN [NRGIDW_Extract].[elcon].[Resource] AS Medarbejdere
    ON Medarbejdere.[No_]=Sager.[Person Responsible]
    LEFT JOIN Sagsbudget
    ON Sagsopgaver.[Job No_] = Sagsbudget.[Job No_]
    Left JOIN Arbejdssedler
    ON Arbejdssedler.[Source No_] = Sagsopgaver.[Job No_]
    WHERE 1=1
	AND (CASE WHEN ISNULL(Sagsbudget.[Indtaegtsbudget],0) = 0 THEN 0 ELSE 1 END +
		 CASE WHEN ISNULL(Sagsbudget.[Omkostningsbudget],0) = 0 THEN 0 ELSE 1 END +
		 CASE WHEN ISNULL(Sagsposter.costs,0) = 0 THEN 0 ELSE 1 END +
		 CASE WHEN ISNULL(Sagsposter.revenue,0) = 0 THEN 0 ELSE 1 END) <> 0
	AND Sagsposter.[department] IN ('421','515','505')
	ORDER BY [year], [month] ASC