    DECLARE @startdate DATETIME 
    DECLARE @date DATETIME 
    DECLARE @count INT 
	DECLARE @afd INT

    SET @startdate='2022-10-31' 
    SET @date= DATEADD(dd,1,@startdate)
    SET @count=0 
	SET @afd ='270'

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


    ;WITH Tilbud AS (
    SELECT [No_]
	,sh.[Bill-to Customer No_]
	,sh.[Bill-to Name]
	,sh.[Bill-to Address]
	,sh.[Bill-to City]
	,sh.[Ship-to Name]
	,sh.[Ship-to Address]
	,sh.[Ship-to City]
	,sh.[Order Date]
	,sh.[Shipment Date]
	,sh.[Salesperson Code]
	,sh.[Ship-to Post Code]
	,sh.[Document Date]
	,sh.[Campaign No_]
	,sh.[Responsibility Center]
	,sh.[Requested Delivery Date]
	,sh.[Changed date]
	FROM [NRGIDW_Extract].[elcon].[Sales Header] sh
	WHERE 1=1
	AND ([Document Type] = 0)
	AND ([Responsibility Center] = @afd))