# 4873-Project-based-Internship
Project-based internship at ELCON in autumn 2023.
Part of my Masters degree in Economics at Aarhus University. The internship is worth 20 ECTS-points, and will be my final course before writing the Masters Thesis in the spring of 2024.

## Problem Statement
The construction sector, including ELCON's Contracting Division, faces numerous challenges such as constrained profit margins, frequent cost overruns, and fluctuating productivity metrics. These challenges are exacerbated within the division due to its responsibility for managing a substantial portfolio of 300 distinct projects. The conventional methodologies currently in use have proven inadequate in addressing the complex and dynamic nature of construction projects. This necessitates the exploration of more advanced and integrated approaches to better manage and forecast project outcomes.

The core research question driving this investigation is:
**Can Machine Learning Models Accurately Forecast Margins and Risks of Construction Projects?**
This question arises from the limitations observed in the current subjective risk assessment practices within ELCON's Contracting Division, which lack a formal statistical foundation. The division's significant contribution to the company's financials underscores the urgent need for a more robust and adaptable data model. Such a model is envisioned to accurately evaluate risks, forecast margins, and support more informed decision-making, thereby enhancing the overall management of construction projects.

## Data Model
ELCON's construction project dataset is collected monthly from the Enterprise Resource Planning (ERP) system, detailing revenue, expenses, labor, and materials costs. Projects undergo three budget estimations: sales estimates at tendering, production estimates during procurement, and final estimates in the construction phase. For older projects, only final estimates were recorded, requiring a synthetic 'budget' category aligned with current protocols.

A financial metric used during construction calculates estimated revenue by adjusting incurred costs against the budgeted margin. This aids in assessing project viability and calculating work in progress and profit margins. Project management data includes logged hours in categories like billable, illness-related, and internal hours. These categories help calculate performance indices (billable, illness, and internal rates) to gauge departmental productivity. However, ELCON cannot analyze productivity below the department level. Project completion dates are modeled using s-curves, linking time to fiscal progress. Expenditure rates are forecasted by comparing actual costs to s-curve projections. Risk is assessed using a linear regression model that normalizes residuals by production cost estimates, reflecting project-specific risks. Qualitative descriptions are textually vectorized to identify key terminologies. The dataset is further enriched with macroeconomic indicators from Statistics Denmark, incorporating external factors like confidence indicators and sector-specific constraints.

## Model Selection

### Statistical
- OLS
- Elastic Net

### Bagging
- Random Forest
- Extremely Randomized Trees

### Boosting
- AdaBoost
- Gradient Boost
- XGBoost

### Deep Learning
- RNN
- CNN
- LSTM
- MES-LSTM
