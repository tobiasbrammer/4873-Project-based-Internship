library(DBI)
library(RMySQL)
library(odbc)
library(dplyr)
library(tidyr)
library(lubridate)
library(ggplot2)
library(ggthemes)
library(ggpubr)
library(feather)
#install.packages(c("DBI", "RMySQL", "odbc", "dplyr", "tidyr", "lubridate", "ggplot2", "ggthemes", "ggpubr", "feather"))

# Connect to the MySQL database: con

rm(list = ls())


# Get directory of file
dir <- "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"

setwd(dir)

con <- dbConnect(odbc::odbc(),
                 Driver = "SQL Server",
                 Database = 'NRGIDW_Extract',
                 Server = 'SARDUSQLBI01',
                 user = 'NRGI\tobr',
                 Trusted_Connection = "True")  
sQuery <- "
    ;WITH Sager AS (
    SELECT *
    From [NRGIDW_Extract].[elcon].[Job] AS Sager
    WHERE 1=1
 	  AND Sager.[Status] IN (2,3)
	  AND Sager.[Global Dimension 1 Code] IN ('515','505')
    AND Sager.[Job Posting Group] IN ('FASTPRIS','PROJEKT')),

    Sagsopgaver AS(
    SELECT
	  Sagsopgaver.[Global Dimension 1 Code]
    ,Sagsopgaver.[Job No_]
    ,Sagsopgaver.[Job Task No_]
    ,Sagsopgaver.[Description]
    FROM [NRGIDW_Extract].[elcon].[Job Task] AS Sagsopgaver
    INNER JOIN Sager
    On Sager.[No_] = Sagsopgaver.[Job No_]
    WHERE 1=1),

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
  	Sagsposter.[Global Dimension 1 Code] AS 'Afd'
    ,Sagsposter.[Job No_]
    ,-SUM(CASE Sagsposter.[Entry Type] WHEN 1 THEN Sagsposter.[Line Amount (LCY)] ELSE 0 END) AS 'Faktureret indtægt'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END) AS 'Bogført omkostning'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 0 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS 'Ressource omkostning'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 1 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS 'Vare omkostning'
    ,SUM(CASE Sagsposter.[Entry Type] WHEN 0 THEN CASE Sagsposter.[Type] WHEN 2 THEN Sagsposter.[Total Cost (LCY)] ELSE 0 END ELSE 0 END) AS  'Andre omkostning'
    ,MAX(Sagsposter.[Posting Date]) AS 'Seneste bogføringsdato'														
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
	Sagsposter.[Entry Type]
	)

    SELECT DISTINCT
    Sager.[Job Posting Group] AS 'Sagsbogføringsgruppe'
	,Sagsposter.[Afd]
    ,Sager.[No_] AS 'Sagsnr.'
	,CASE Sager.[Status] WHEN 2 THEN 'Igangværende' ELSE 'Afsluttet' END AS 'Status'
    ,Sager.[Description] AS 'Beskrivelse'
	,Kunder.[No_] AS 'Kundenummer'
    ,Kunder.[Name] AS 'Kundenavn'
    ,CONCAT(Sager.[Ship-to Address],' ',Sager.[Ship-to Post Code],' ',Sager.[Ship-to City]) AS 'Leveringsadresse'
	,Sager.[Ship-to Post Code] AS 'Postnummer'
    ,Medarbejdere.[Name] AS 'Ansvarlig'
    ,(ISNULL(Sagsbudget.[Indtægtsbudget],0)) AS 'Slut_vurdering_indtægt'
    ,(ISNULL(Sagsbudget.[Omkostningsbudget],0)) AS 'Slut_vurdering_omkostninger'
    ,((ISNULL(Sagsbudget.[Indtægtsbudget],0)) - (ISNULL(Sagsbudget.[Omkostningsbudget],0))) AS 'Slut_vurdering_DB'
    ,(ISNULL(Sagsposter.[Faktureret indtægt],0)) AS 'Faktureret_indtægt'
    ,(ISNULL(Sagsposter.[Bogført omkostning],0)) 'Bogførte_omkostninger'
    ,(ISNULL(Sagsposter.[Ressource omkostning],0)) AS 'Ressourceomkostninger'
    ,(ISNULL(Sagsposter.[Vare omkostning],0)) AS 'Vareomkostninger'
    ,(ISNULL(Sagsposter.[Andre omkostning],0)) AS  'Andre_omkostninger'
	,(ISNULL(Sagsposter.[Faktureret indtægt],0)) - (ISNULL(Sagsposter.[Bogført omkostning],0)) AS 'Dækningsbidrag'
	,(ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsposter.[Faktureret indtægt],0)) AS 'Budget_-_faktureret'
	,(CASE 
        WHEN 
            ((ISNULL(Sagsposter.[Bogført omkostning],0)) = 0 AND (ISNULL(Sagsbudget.[Omkostningsbudget],0)) = 0) 
            OR ISNULL((ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF((Sagsbudget.[Indtægtsbudget]),0),0) >= 1 
        THEN (ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
        ELSE 
            CASE 
                WHEN ((ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL((ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF((Sagsbudget.[Indtægtsbudget]),0),0)), 0)) > (ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                THEN (ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                ELSE ((ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL((ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF((Sagsbudget.[Indtægtsbudget]),0),0)), 0))
            END 
    END) AS 'Beregnet_indtægt'
	,(CASE 
        WHEN 
            (Sagsposter.[Bogført omkostning] = 0 AND Sagsbudget.[Omkostningsbudget] = 0) 
            OR ISNULL((ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF(Sagsbudget.[Indtægtsbudget],0),0) >= 1 
        THEN (ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
        ELSE 
            CASE 
                WHEN ((ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL((ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF((Sagsbudget.[Indtægtsbudget]),0),0)), 0)) > (ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                THEN (ISNULL(Sagsbudget.[Indtægtsbudget],0)) 
                ELSE ((ISNULL(Sagsposter.[Bogført omkostning],0))/NULLIF((1-ISNULL((ISNULL(Sagsbudget.[Indtægtsbudget],0) - ISNULL(Sagsbudget.[Omkostningsbudget],0))/NULLIF((Sagsbudget.[Indtægtsbudget]),0),0)), 0))
            END 
    END) - (ISNULL(Sagsposter.[Bogført omkostning],0)) AS 'Beregnet_DB'
    ,(ISNULL(Arbejdssedler.[Antal],0)) AS 'Antal_ikke_lukkede_arbejdssedler_på_sagen'
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
"

query <- dbSendQuery(con,sQuery)

dfData <- dbFetch(query)

feather::write_feather(dfData, "dfDataX.feather")
dbClearResult(query)
