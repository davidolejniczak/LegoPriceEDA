
# LegoPricePredictor

A project that scrapes Lego pricing data from various websites and past market sales to predict expected value growth using statistical analysis. 

This project showcases my ability to integrate data collection, cloud services, and statistical analysis into a cohesive and impactful tool for investment analysis.

---

### Project Overview
The LegoPricePredictor project analyzes the potential investment value of Lego sets by analyzing historical prices for retired Leto sets. 

1. **Lego official website**: Product pricing and set details.
2. **Lego price data sets off Kaggle**: Historical pricing data. 

The project employs predictive modeling to estimate ROI for Lego sets post-retirement. It leverages statistical methods and visualizations to offer insights for Lego investors like myself.

---

### Technologies Used
- **Python Libraries**:
  - `Selenium`: Web scraping with automated browser interaction.
  - `BeautifulSoup`: HTML parsing for extracting data.
  - `pandas`, `numpy`: Data processing and analysis.
  - `matplotlib`: Data visualization.
  - `sklearn`: Building predictive models.
  - `fake_useragent` for mimicking legitimate browser behavior and avoiding blocking.
- **Google Cloud Services**:
  - Google Sheets API for data storage.
  - Google Cloud Console for project management.
- **Authentication**:
  - Auth0 for secure API access and authentication.

---

### Data Analysis
Predictive analysis focused on estimating the ROI of Lego sets post-retirement using the following methods:

#### 1. Least Squares Regression
- **Results**:
  - Training R2 Score: 0.5328
  - Training MAE Score: 0.0935
  - Test R2 Score: 0.3724
  - Test MAE Score: 0.0969

#### 2. Cross-Validation
- **Results**:
  - Cross-Validation R2 Score: 0.1374
  - Cross-Validation MAE Score: 0.0986

#### 3. Lasso Regression
- **Results**:
  - Training R2 Score: 0.3219
  - Training MAE Score: 0.1112
  - Test R2 Score: 0.2369
  - Test MAE Score: 0.1086

#### Feature Importance from Lasso Regression:
| Feature             | Coefficient   |
|---------------------|---------------|
| pop_price          | 0.001091      |
| num_parts          | 0.000003      |
| num_unique_figs    | -0.000000     |
| num_figs           | -0.000000     |
| set_rating         | 0.000000      |
| num_reviews        | -0.000000     |
| set_id             | -0.000001     |
| retail_price       | -0.000847     |

---

### Results and Insights
#### ROI Chart
The average ROI chart by category revealed:
- **Top Categories**:
  - Modular Buildings
  - City > Airport
  - Speed Champions

#### Take Aways
Based on the results of the linear regression models, the Least Squares Regression method is the most suitable for modeling the ROI of Lego sets. It achieved the highest R2 score of 0.5328 during training and 0.3724 during testing, indicating better predictive performance compared to Cross-Validation and Lasso Regression. However, the relatively low R2 scores across all models highlight the need for additional features and a larger dataset to improve prediction accuracy. Despite these limitations, the project successfully demonstrated the feasibility of using statistical models to estimate Lego set returns, paving the way for future enhancements in data collection and modeling.

---

### Future Improvements
1. **Expand Dataset**:
   - Scraping more data points to improve model accuracy.
2. **Predictive Models**:
   - Enhance the predictive algorithm for higher R2 scores or include AI into the analysis. 
3. **Frontend Development**:
   - Build a web interface using React to make the tool accessible for users.
---

### Legal Disclaimer
This project was developed solely for educational purposes. The web scraping conducted complies with the terms and conditions of the data sources as it only accesses publicly available information. Care has been taken to minimize server load. 

---

### How to Run the Project
#### Prerequisites
1. Python 3.8 or higher.
2. Google Cloud Console account with Sheets API enabled.
3. Necessary Python libraries installed:
   ```bash
   pip install -r requirements.txt
   ```

#### Steps
**Scrape lego data**:
   - Launch the Selenium-based scraper to collect data and upload the links in the text files
   - The SheetUpdater is for pricing data from the offical Lego website
   - The UpdatesManager is for data from an third party Lego site
   ```bash
   python SheetUpdater.py
   ```
   Or
   ```
   python UpdatesManager.py
   ```
**Run the Analysis**:
   - Run the analysis.py script to view the ROI charts and predictive insights will be outputted to the console and saved as visual files.
   ```bash
   python analysis.py
   ```
---
