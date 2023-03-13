    DECLARE @startdate DATETIME 
    DECLARE @date DATETIME 
    DECLARE @count INT 
	DECLARE @afd INT

    SET @startdate='2022-10-31' 
    SET @date= DATEADD(dd,1,@startdate)
    SET @count=0 
	SET @afd =''

    WHILE @count < 2 
    BEGIN 
      IF Datename(dw, @date) NOT IN ( 'Saturday', 'Sunday' ) 
        BEGIN 
            SET @date=Dateadd(dd, 1, @date) 
            SET @count=@count + 1 
        END 
      ELSE 
        BEGIN 
            SET @date=Dateadd(dd, 1, @date) 
        END 
    END


    ;WITH Sager AS (
    SELECT *
    From [NRGIDW_Extract].[elcon].[Job] AS Sager
    WHERE 1=1
	AND Sager.[Status] = 2
    AND Sager.[Job Posting Group] IN ('FASTPRIS','PROJEKT','REGNING','TBL')),

	DimDato AS(
    SELECT
	   Cal.[CalendarDate]
      ,Cal.[DayName]
      ,Cal.[DayNameShort]
      ,Cal.[YearInt]
      ,Cal.[MonthEnd]
      ,Cal.[MonthName]
      ,Cal.[MonthNameShort]
      ,Cal.[DayOfYear]
      ,Cal.[DayOfMonth]
      ,Cal.[DayOfWeek_EU]
      ,Cal.[WeekDayNameShort]
      ,Cal.[WeekOfYear_EU]
      ,Cal.[MonthOfYear]
      ,Cal.[HolyDay]
      ,Cal.[WorkDay]
      ,Cal.[WorkDayOfMonth]
    FROM [ElconDW].[dim].[Calendar] AS Cal),


    Sagsopgaver AS(
    SELECT
    Sagsopgaver.[Job No_]
    ,Sagsopgaver.[Job Task No_]
    ,Sagsopgaver.[Description]
    FROM [NRGIDW_Extract].[elcon].[Job Task] AS Sagsopgaver
    INNER JOIN Sager
    On Sager.[No_] = Sagsopgaver.[Job No_]
    WHERE 1=1
	--AND Sagsopgaver.[Global Dimension 1 Code] = @afd
	),

    Sagsbudget AS(
    SELECT 
    Sagsopgaver.[Job No_]
    ,SUM(CASE Sagsbudget.[Line Type] WHEN 1 THEN Sagsbudget.[Line Amount (LCY)] WHEN 2 THEN [Line Amount (LCY)] ELSE 0 END) AS 'Indtægtsbudget'
    ,SUM(CASE Sagsbudget.[Line Type] WHEN 0 THEN Sagsbudget.[Total Cost (LCY)] WHEN 2 THEN [Total Cost (LCY)] ELSE 0 END) AS 'Omkostningsbudget'
    FROM [NRGIDW_Extract].[elcon].[Job Planning Line Entry] AS Sagsbudget
    Inner JOIN Sagsopgaver
    ON CONCAT(Sagsbudget.[Job No_],Sagsbudget.[Job Task No_]) = CONCAT(Sagsopgaver.[Job No_],Sagsopgaver.[Job Task No_])
    INNER JOIN Sager
    ON Sager.[No_]=Sagsbudget.[Job No_]
    GROUP BY 
    Sagsopgaver.[Job No_]),

    Arbejdssedler AS(
    SELECT 
    Arbejdssedler.[Source No_]
    ,Count(Arbejdssedler.[Source No_]) AS Antal
    FROM [NRGIDW_Extract].[elcon].[Work Order Header] AS Arbejdssedler
    INNER JOIN Sager
    ON Sager.[No_] = Arbejdssedler.[Source No_]
    WHERE 1=1
	AND (NOT(Arbejdssedler.[Status] = 3) OR Arbejdssedler.[Status] IS NULL)
    GROUP BY Arbejdssedler.[Source No_]),

    Sagsposter AS(
    SELECT
	DimDato.MonthName
	,Sagsposter.[Global Dimension 1 Code] AS Afdeling
    ,Sagsposter.[Job No_]
    ,-SUM(CASE Sagsposter.[Entry Type] WHEN 1 THEN Sagsposter.[Line Amount (LCY)] ELSE 0 END) AS 'Faktureret indtægt'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END) AS 'Bogført omkostning'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 0 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS 'Ressource omkostning'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 1 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS 'Vare omkostning'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 2 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS  'Andre omkostning'
    ,MAX(Sagsposter.[Posting Date]) AS 'Seneste bogføringsdato'
    ,-SUM(CASE Sagsposter.[Entry Type] WHEN 1 THEN CASE Format(Sagsposter.[Posting Date],'yyyy-MM') WHEN FORMAT(@startdate,'yyyy-MM')THEN
    Sagsposter.[Line Amount (LCY)] ELSE 0 END ELSE 0 END) AS 'Faktureret i måneden'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Format(Sagsposter.[Posting Date],'yyyy-MM') WHEN FORMAT(@startdate,'yyyy-MM')THEN
    Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS 'Forbrug i måneden'
    FROM [NRGIDW_Extract].[elcon].[Job Ledger Entry] AS Sagsposter
    INNER JOIN Sagsopgaver
    ON CONCAT(Sagsposter.[Job No_],Sagsposter.[Job Task No_]) = CONCAT(Sagsopgaver.[Job No_],Sagsopgaver.[Job Task No_])
    INNER JOIN Sager
    ON Sager.[No_]=Sagsposter.[Job No_]
	LEFT JOIN DimDato
	ON Sagsposter.[Posting Date] = DimDato.CalendarDate
    WHERE 1=1
	AND Sagsposter.[Posting Date]<= @date
    GROUP BY 
    Sagsposter.[Global Dimension 1 Code], Sagsposter.[Job No_], DimDato.MonthName)
    
    SELECT DISTINCT
	Sagsposter.MonthName
	,Sagsposter.Afdeling
    ,Sager.[Job Posting Group] AS 'Sagsbogføringsgruppe'
    ,Sager.[No_] AS 'Sagsnr.'
    ,Sager.[Description] AS 'Beskrivelse'
    ,Kunder.[Name] AS 'Kundenavn'
    ,CONCAT(Sager.[Ship-to Address],' ',Sager.[Ship-to Post Code],' ',Sager.[Ship-to City]) AS Leveringsadresse
    ,Medarbejdere.[Name] AS Ansvarlig
	,Sager.[Starting Date] AS Startdato
	,Sager.[Ending Date] AS Slutdato
    ,ISNULL(Sagsbudget.[Indtægtsbudget],0) AS 'Slut vurdering indtægt'
    ,ISNULL(Sagsbudget.[Omkostningsbudget],0) AS 'Slut vurdering omkostning'
    ,ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0) AS 'Slut vurdering DB'
	,(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0) AS 'Slut vurdering DG'
	,CASE WHEN (ISNULL(Sagsbudget.[Omkostningsbudget],0) IS NULL AND ISNULL(Sagsbudget.[Omkostningsbudget],0) IS NULL)
			    THEN Sagsbudget.[Indtægtsbudget] 
		  WHEN (ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0) >= 1
			   THEN Sagsbudget.[Indtægtsbudget]
		  WHEN ISNULL(Sagsposter.[Bogført omkostning],0)/(1-(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0)) > Sagsbudget.[Indtægtsbudget]
			   THEN Sagsbudget.[Indtægtsbudget]
		  ELSE ISNULL(Sagsposter.[Bogført omkostning],0)/(1-(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0))
	 END AS 'Beregnet indtægt'
	 ,(CASE WHEN (ISNULL(Sagsbudget.[Omkostningsbudget],0) IS NULL AND ISNULL(Sagsbudget.[Omkostningsbudget],0) IS NULL)
			    THEN Sagsbudget.[Indtægtsbudget] 
		  WHEN (ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0) >= 1
			   THEN Sagsbudget.[Indtægtsbudget]
		  WHEN ISNULL(Sagsposter.[Bogført omkostning],0)/(1-(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0)) > Sagsbudget.[Indtægtsbudget]
			   THEN Sagsbudget.[Indtægtsbudget]
		  ELSE ISNULL(Sagsposter.[Bogført omkostning],0)/(1-(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0))
	 END - Sagsposter.[Bogført omkostning]) AS 'Beregnet DB'
	,(CASE WHEN (ISNULL(Sagsbudget.[Omkostningsbudget],0) IS NULL AND ISNULL(Sagsbudget.[Omkostningsbudget],0) IS NULL)
			    THEN Sagsbudget.[Indtægtsbudget] 
		  WHEN (ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0) >= 1
			   THEN Sagsbudget.[Indtægtsbudget]
		  WHEN ISNULL(Sagsposter.[Bogført omkostning],0)/(1-(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0)) > Sagsbudget.[Indtægtsbudget]
			   THEN Sagsbudget.[Indtægtsbudget]
		  ELSE ISNULL(Sagsposter.[Bogført omkostning],0)/(1-(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0))
	 END - Sagsposter.[Bogført omkostning])/NULLIF((CASE WHEN (ISNULL(Sagsbudget.[Omkostningsbudget],0) IS NULL AND ISNULL(Sagsbudget.[Omkostningsbudget],0) IS NULL)
			    THEN Sagsbudget.[Indtægtsbudget] 
		  WHEN (ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0) >= 1
			   THEN Sagsbudget.[Indtægtsbudget]
		  WHEN ISNULL(Sagsposter.[Bogført omkostning],0)/(1-(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0)) > Sagsbudget.[Indtægtsbudget]
			   THEN Sagsbudget.[Indtægtsbudget]
		  ELSE ISNULL(Sagsposter.[Bogført omkostning],0)/(1-(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0))
	 END),0) AS 'Beregnet DG'
	,((CASE WHEN (ISNULL(Sagsbudget.[Omkostningsbudget],0) IS NULL AND ISNULL(Sagsbudget.[Omkostningsbudget],0) IS NULL)
			    THEN Sagsbudget.[Indtægtsbudget] 
		  WHEN (ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0) >= 1
			   THEN Sagsbudget.[Indtægtsbudget]
		  WHEN ISNULL(Sagsposter.[Bogført omkostning],0)/(1-(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0)) > Sagsbudget.[Indtægtsbudget]
			   THEN Sagsbudget.[Indtægtsbudget]
		  ELSE ISNULL(Sagsposter.[Bogført omkostning],0)/(1-(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0))
	 END) - Sagsposter.[Faktureret indtægt]) AS 'Igangværende arbejder'
    ,ISNULL(Sagsposter.[Faktureret indtægt],0) AS 'Faktureret indtægt'
    ,ISNULL(Sagsposter.[Bogført omkostning],0) 'Bogført omkostning'
    ,ISNULL(Sagsposter.[Ressource omkostning],0) AS 'Ressource omkostning'
    ,ISNULL(Sagsposter.[Vare omkostning],0) AS 'Vare omkostning'
    ,ISNULL(Sagsposter.[Andre omkostning],0) AS  'Andre omkostning'
    ,ISNULL(Arbejdssedler.[Antal],0) AS 'Antal ikke lukkede arbejdssedler på sagen'
	,ISNULL(Sagsposter.[Bogført omkostning],0)/NULLIF(Sagsbudget.[Omkostningsbudget],0) AS 'Færdiggørelsesgrad'
	,ISNULL(Sagsposter.[Ressource omkostning],0)/NULLIF(Sagsposter.[Bogført omkostning],0) AS 'Ressourcer i %'
	,ISNULL(Sagsposter.[Vare omkostning],0)/NULLIF(Sagsposter.[Bogført omkostning],0) AS 'Varer i %'
	,ISNULL(Sagsposter.[Andre omkostning],0)/NULLIF(Sagsposter.[Bogført omkostning],0) AS 'Andet i %'
    FROM Sagsopgaver
    Left JOIN Sagsposter
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
	AND (CASE WHEN ISNULL(Sagsbudget.[Indtægtsbudget],0) = 0 THEN 0 ELSE 1 END +
		 CASE WHEN ISNULL(Sagsbudget.[Omkostningsbudget],0) = 0 THEN 0 ELSE 1 END +
		 CASE WHEN ISNULL(Sagsposter.[Bogført omkostning],0) = 0 THEN 0 ELSE 1 END +
		 CASE WHEN ISNULL(Sagsposter.[Faktureret indtægt],0) = 0 THEN 0 ELSE 1 END) <> 0