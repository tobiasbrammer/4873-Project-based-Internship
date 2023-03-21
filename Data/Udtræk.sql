    DECLARE @startdate DATETIME
	DECLARE @enddate DATETIME
    DECLARE @date DATETIME 
    DECLARE @count INT 
	DECLARE @jobno CHAR(20)

	DECLARE @afd TABLE (Value INT)
	INSERT INTO @afd VALUES (421)
	INSERT INTO @afd VALUES (505)
	INSERT INTO @afd VALUES (515)


    SET @startdate = '2015-01-01' 
	SET @enddate = '2023-02-28' 

	SET @jobno = ''

    ;WITH Sager AS (
    SELECT *
    From [NRGIDW_Extract].[elcon].[Job] AS Sager
    WHERE 1=1
	AND Sager.No_ = @jobno OR Coalesce(@jobno,'') = ''
	AND Sager.[Status] = 2
    AND (Sager.[Job Posting Group] = 'FASTPRIS' OR Sager.[Job Posting Group] = 'PROJEKT')),

    Sagsopgaver AS(
    SELECT
	Sagsopgaver.[Global Dimension 1 Code]
    ,Sagsopgaver.[Job No_]
    ,Sagsopgaver.[Job Task No_]
    ,Sagsopgaver.[Description]
    FROM [NRGIDW_Extract].[elcon].[Job Task] AS Sagsopgaver
    INNER JOIN Sager
    On Sager.[No_] = Sagsopgaver.[Job No_]
    WHERE 1=1
	AND Sagsopgaver.[Global Dimension 1 Code] IN (SELECT Value FROM @afd)),

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
	FORMAT(Sagsposter.[Posting Date],'dd-MM-yyyy') AS 'Dato'
    ,Datepart(iso_week,Sagsposter.[Posting Date]) AS 'Uge'
	,CONCAT(Datepart(iso_week,Sagsposter.[Posting Date]),'-',FORMAT(Sagsposter.[Posting Date], 'yyyy')) AS 'Uge-år'
	,FORMAT(Sagsposter.[Posting Date], 'MM-yyyy') AS 'Måned-år'
	,FORMAT(Sagsposter.[Posting Date], 'yyyy') AS 'År'
	,Sagsposter.[Global Dimension 1 Code] AS 'Afd'
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
    WHERE 1=1
	AND Sagsposter.[Posting Date] BETWEEN @startdate AND @enddate
    GROUP BY 
	Sagsposter.[Global Dimension 1 Code],
    Sagsposter.[Job No_],
	Sagsposter.[Posting Date]
	)
    
    SELECT DISTINCT
	Sagsposter.[Uge]
	,Sagsposter.[Uge-år]
	,Sagsposter.[Måned-år]
	,Sagsposter.[År]
    ,Sager.[Job Posting Group] AS 'Sagsbogføringsgruppe'
	,Sagsposter.[Afd]
    ,Sager.[No_] AS 'Sagsnr.'
    ,Sager.[Description] AS 'Beskrivelse'
    ,Kunder.[Name] AS 'Kundenavn'
    ,CONCAT(Sager.[Ship-to Address],' ',Sager.[Ship-to Post Code],' ',Sager.[Ship-to City]) AS 'Leveringsadresse'
	,Sager.[Ship-to Post Code] AS 'Postnummer'
    ,Medarbejdere.[Name] AS 'Ansvarlig'
    ,SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) AS 'Slut-vurdering indtægt'
    ,SUM(ISNULL(Sagsbudget.[Omkostningsbudget],0)) AS 'Slut-vurdering omkostninger'
    ,SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0)) AS 'Slut-vurdering DB'
	,ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0) AS 'Slut-vurdering DG'
    ,SUM(ISNULL(Sagsposter.[Faktureret indtægt],0)) AS 'Faktureret indtægt'
    ,SUM(ISNULL(Sagsposter.[Bogført omkostning],0)) 'Bogførte omkostninger'
    ,SUM(ISNULL(Sagsposter.[Ressource omkostning],0)) AS 'Ressourceomkostninger'
    ,SUM(ISNULL(Sagsposter.[Vare omkostning],0)) AS 'Vareomkostninger'
    ,SUM(ISNULL(Sagsposter.[Andre omkostning],0)) AS  'Andre omkostninger'
	,SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsposter.[Faktureret indtægt],0)) AS 'Igangværende arbejder'
	,(CASE 
        WHEN 
            (SUM(ISNULL(Sagsposter.[Bogført omkostning],0)) = 0 AND SUM(ISNULL(Sagsbudget.[Omkostningsbudget],0)) = 0) 
            OR ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0) >= 1 
        THEN SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
        ELSE 
            CASE 
                WHEN (SUM(ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0)), 0)) > SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                THEN SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                ELSE (SUM(ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0)), 0))
            END 
    END) AS 'Beregnet indtægt'
	,(CASE 
        WHEN 
            (SUM(ISNULL(Sagsposter.[Bogført omkostning],0)) = 0 AND SUM(ISNULL(Sagsbudget.[Omkostningsbudget],0)) = 0) 
            OR ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0) >= 1 
        THEN SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
        ELSE 
            CASE 
                WHEN (SUM(ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0)), 0)) > SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                THEN SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                ELSE (SUM(ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0)), 0))
            END 
    END) - SUM(ISNULL(Sagsposter.[Bogført omkostning],0)) AS 'Beregnet DB'
	,ISNULL(((CASE 
        WHEN 
            (SUM(ISNULL(Sagsposter.[Bogført omkostning],0)) = 0 AND SUM(ISNULL(Sagsbudget.[Omkostningsbudget],0)) = 0) 
            OR ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0) >= 1 
        THEN SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
        ELSE 
            CASE 
                WHEN (SUM(ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0)), 0)) > SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                THEN SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                ELSE (SUM(ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0)), 0))
            END 
    END) - SUM(ISNULL(Sagsposter.[Bogført omkostning],0)))/NULLIF((CASE 
        WHEN 
            (SUM(ISNULL(Sagsposter.[Bogført omkostning],0)) = 0 AND SUM(ISNULL(Sagsbudget.[Omkostningsbudget],0)) = 0) 
            OR ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0) >= 1 
        THEN SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
        ELSE 
            CASE 
                WHEN (SUM(ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0)), 0)) > SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                THEN SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                ELSE (SUM(ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(SUM(ISNULL(Sagsbudget.[Indtægtsbudget],0)),0),0)), 0))
            END 
    END),0),0) AS 'Beregnet DG'
    ,ROUND(MAX(ISNULL(Arbejdssedler.[Antal],0)),0) AS 'Antal ikke lukkede arbejdssedler på sagen'
    FROM Sagsopgaver
    LEFT JOIN [NRGIDW_Extract].[elcon].[DynamicsNavHyperlink] AS Link
    ON Link.[Sagsnummer]=Sagsopgaver.[Job No_]
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
	GROUP BY 
	Sagsposter.[Uge]
	,Sagsposter.[Uge-år]
	,Sagsposter.[Måned-år]
	,Sagsposter.[År]
    ,Sager.[Job Posting Group]
	,Sagsposter.[Afd]
    ,Sager.[No_]
    ,Sager.[Description]
    ,Kunder.[Name]
    ,CONCAT(Sager.[Ship-to Address],' ',Sager.[Ship-to Post Code],' ',Sager.[Ship-to City])
	,Sager.[Ship-to Post Code]
    ,Medarbejdere.[Name]
	ORDER BY [År],[Uge],[Måned-år],[Afd],[Sagsnr.]