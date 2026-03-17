# 🛒 Instacart Purchase Behavior — Hierarchical Bayesian Modeling



![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Stan](https://img.shields.io/badge/Stan-B2171D?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAAA&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)



> A Bayesian computational statistics project applying **Hierarchical Bayesian Logistic Regression** and **Bayesian Poisson Regression** to model customer purchase behavior on **Instacart** — predicting product reorder probability and cart size with full uncertainty quantification using **Stan** and **Python**.



---



## 📌 Table of Contents



- [Overview](#-overview)
- [About the Dataset](#-about-the-dataset)
- [Objectives](#-objectives)
- [Key Factors Analyzed](#-key-factors-analyzed)
  - [Reorder Prediction](#-reorder-prediction)
  - [Cart Size Forecasting](#-cart-size-forecasting)
- [Model Architecture](#-model-architecture)
  - [Bayesian Hierarchical Logistic Regression](#-bayesian-hierarchical-logistic-regression)
  - [Bayesian Poisson Regression](#-bayesian-poisson-regression)
- [Data Sources](#-data-sources)
- [Tech Stack](#-tech-stack)
- [Methodology](#-methodology)
- [Results & Insights](#-results--insights)
- [Model Comparison](#-model-comparison)
- [Advantages & Limitations](#-advantages--limitations)
- [Conclusion](#-conclusion)



---



## 🔍 Overview



Online grocery shopping data are notoriously **sparse and skewed**. Most customers place only a handful of orders and buy a tiny fraction of the available catalog, while occasional power users or promotional events produce very large basket sizes. Traditional ML classifiers and regressors may achieve good average accuracy on dense segments, but they **fail to adapt to cold-start users or niche products** and offer no principled way to quantify risk.

This project addresses these challenges by applying:

1. **Hierarchical Bayesian Logistic Regression** — for reorder probability prediction with user- and product-level random intercepts
2. **Bayesian Poisson Regression** — for cart-size forecasting with full predictive distributions

This combination delivers **robust, well-calibrated predictions** across the long-tail of e-commerce behavior and provides **credible intervals and tail-risk estimates** essential for inventory planning, delivery staffing, and personalized recommendations.



---



## 🗂️ About the Dataset



**Instacart's** open-source dataset contains a sample of over **3 million grocery orders** from more than **200,000 users**. For each user, between 4 and 100 of their orders are provided, along with the sequence of products purchased, the week and hour of day the order was placed, and a relative measure of time between orders.



The dataset comprises **six relational tables**:



| Table | Description |
|-------|-------------|
| `orders` | Each row represents a grocery order (prior, train, test splits) |
| `order_products_prior` | Prior orders with reorder indicator per product |
| `order_products_train` | Training orders with reorder indicator per product |
| `products` | All products and their related information |
| `aisles` | All aisles and their related information |
| `departments` | All departments and their related information |



**Key Dataset Statistics:**

| Metric | Value |
|--------|-------|
| Total Orders | ~3.4 million |
| Unique Users | ~200,000+ |
| Unique Products | ~50,000 |
| Reorder Rate | **59.01%** |
| First Purchase Rate | **40.99%** |
| Training Set Users | ~131,000 |
| Test Set Users | ~75,000 |



---



## 🎯 Objectives



- **Predict** product reorder probability for any user–product pair with uncertainty quantification
- **Forecast** cart size (number of items per order) with full predictive distributions
- **Quantify uncertainty** via Bayesian credible intervals rather than single-point estimates
- **Compare** Bayesian models against classical ML baselines (Logistic Regression, Poisson Regression)
- **Analyze** ordering patterns, reorder behaviors, and temporal trends through EDA
- **Provide** actionable probabilistic forecasts for inventory, staffing, and recommendation decisions



---



## 🔑 Key Factors Analyzed



### 🔄 Reorder Prediction



**Target Variable:** Binary — whether user *u* reorders product *p* on a given order

| Feature | Description |
|---------|-------------|
| `order_number` | Sequential order count for the user |
| `order_dow` | Day of the week the order was placed |
| `order_hour_of_day` | Hour of the day the order was placed |
| `user_idx` | User-level random intercept (hierarchical) |
| `product_idx` | Product-level random intercept (hierarchical) |



### 📦 Cart Size Forecasting



**Target Variable:** Count — number of items in the order (*y* ∈ {0, 1, 2, ...})

| Feature | Description |
|---------|-------------|
| `order_number` | Sequential order count for the user |
| `order_dow` | Day of the week |
| `order_hour_of_day` | Hour of the day |
| `avg_cart_size` | User's historical average basket size |



> All predictors were **standardized** (mean = 0, sd = 1) before modeling.



---



## 🏗️ Model Architecture



### 📊 Bayesian Hierarchical Logistic Regression



A hierarchical model for **binary reorder prediction**, capturing both global trends and individual-level variability


> **Key Insight:** The non-centered parameterization (`u_raw × σ_u`) improves MCMC sampling efficiency and allows sparse users/products to **shrink toward the population mean**.



### 📈 Bayesian Poisson Regression



A count regression model for **cart-size forecasting** with log-link



> **Key Insight:** The Bayesian framework yields **full predictive distributions** — enabling credible intervals like *"we're 90% sure this basket will contain between 5 and 20 items"*.



---



## 📊 Data Sources



| Source | Data Collected |
|--------|----------------|
| [Instacart Dataset (Kaggle)](https://www.kaggle.com/c/instacart-market-basket-analysis) | Orders, products, aisles, departments, reorder flags |
| Instacart Open Dataset | 3M+ orders from 200K+ users with product-level detail |



---



## 🛠️ Tech Stack



### Languages & Frameworks

- **Python** — Data wrangling, EDA, modeling, evaluation
- **Stan (CmdStanPy)** — Bayesian probabilistic programming & MCMC sampling



### Key Libraries & Tools



| Purpose | Libraries |
|---------|-----------|
| Bayesian Modeling | `CmdStanPy`, `Stan` |
| Data Wrangling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| ML Baselines | `scikit-learn` (`LogisticRegression`, `PoissonRegressor`) |
| Metrics | `accuracy_score`, `roc_auc_score`, `log_loss`, `RMSE`, `MAE` |



### MCMC Configuration

| Parameter | Value |
|-----------|-------|
| Chains | 4 |
| Warmup Iterations | 1,000 |
| Sampling Iterations | 1,000 |
| `adapt_delta` | 0.99 |
| `max_treedepth` | 15 |
| Seed | 42 |



---



## 📐 Methodology



<table>
<tr>
<td align="center" width="140">
<h3>📥</h3>
<b>Step 1</b><br>
Data Collection<br>
<sub>Instacart Dataset<br>6 Relational Tables<br>3M+ Orders<br>200K+ Users</sub>
</td>
<td align="center" width="30">
<h3>➡️</h3>
</td>
<td align="center" width="140">
<h3>🔍</h3>
<b>Step 2</b><br>
EDA<br>
<sub>Order Patterns<br>Reorder Rates<br>Temporal Trends<br>Department Analysis</sub>
</td>
<td align="center" width="30">
<h3>➡️</h3>
</td>
<td align="center" width="140">
<h3>⚙️</h3>
<b>Step 3</b><br>
Feature Engineering<br>
<sub>User/Product Indexing<br>Standardization<br>Cart Size Aggregation<br>Avg Basket Size</sub>
</td>
<td align="center" width="30">
<h3>➡️</h3>
</td>
<td align="center" width="140">
<h3>🧮</h3>
<b>Step 4</b><br>
Stan Modeling<br>
<sub>Hierarchical Logistic<br>Poisson Regression<br>MCMC Sampling<br>Prior Specification</sub>
</td>
<td align="center" width="30">
<h3>➡️</h3>
</td>
<td align="center" width="140">
<h3>📊</h3>
<b>Step 5</b><br>
Posterior Analysis<br>
<sub>Posterior Distributions<br>Credible Intervals<br>Convergence Checks<br>Predictive Checks</sub>
</td>
<td align="center" width="30">
<h3>➡️</h3>
</td>
<td align="center" width="140">
<h3>⚖️</h3>
<b>Step 6</b><br>
Comparison<br>
<sub>Bayesian vs ML<br>AUC & LogLoss<br>RMSE & MAE<br>Uncertainty Analysis</sub>
</td>
</tr>
</table>



---



## 📈 Results & Insights



### 1. 🔍 Exploratory Data Analysis — Key Findings



| Insight | Detail |
|---------|--------|
| 📅 Order Peaks | Saturdays and Sundays; dips mid-week |
| ⏰ Hour Peaks | 10 AM (morning planning) & 6–8 PM (evening restocking) |
| 📦 Product Distribution | Top 10% of products account for ~50% of all order-product lines |
| 🔄 Highest Reorder Rates | Dairy & eggs, produce, household essentials |
| 📆 Order Frequency | Clear spikes at 7 days (weekly shoppers), with modes at 14 and 30 days |
| 🛒 Basket Size | Ranges widely — from single-item top-ups to heavy weekly stock-ups |



---



### 2. 🔄 Reorder Prediction — Posterior Insights



**Hyperparameter Posteriors:**

| Parameter | Posterior Mean | Interpretation |
|-----------|---------------|----------------|
| **α** (Global Intercept) | **0.44** | Baseline log-odds → logistic(0.44) ≈ **61% reorder probability** for an average user–product pair |
| **σ_u** (User SD) | **0.62** | 95% of users have reorder odds between **0.30× to 3.39×** baseline |
| **σ_p** (Product SD) | **0.76** | Products vary more than users; 95% odds ratios in **[0.22, 4.5]** |
| **β_num** (Order Number) | **0.75** | exp(0.75) = 2.12 → Higher order numbers are **~2× more likely** to be reorders |
| **β_dow** (Day of Week) | **-0.05** | exp(-0.05) = 0.95 → ~5% drop in odds; **minimal effect** |
| **β_hour** (Hour of Day) | **~0.00** | Straddles zero → **no strong effect** on reorder probability |

> **Interpretation:** The model is highly confident this user will reorder this product, with a narrow credible interval reflecting strong historical signal.

### 3. 📦 Cart Size Prediction — Posterior Insights



**Coefficient Posteriors:**

| Coefficient | Posterior Mean | Interpretation |
|-------------|---------------|----------------|
| **β₀** (Intercept) | **2.68** | exp(2.68) ≈ 14.6 → Baseline expected cart size |
| **β₁** (Order Number) | **0.001** | Negligible effect on cart size |
| **β₂** (Day of Week) | **-0.006** | Slight decrease later in the week |
| **β₃** (Hour of Day) | **-0.001** | Negligible effect |
| **β₄** (Avg Cart Size) | **0.455** | **Strongest predictor** — users with larger historical baskets continue to buy more

> **Interpretation:** For a user placing their 5th order on Wednesday at 3 PM with an average basket of 12.3 items, the model predicts ~12 items with a wide but informative predictive interval.



### 4. 🔬 Posterior Distribution Analysis



#### User Intercepts (γ_u)

- **Centered near zero** — most users behave close to the global average
- **Long tails** — a handful of users have significantly higher or lower reorder propensity
- **Positive γ_u** → user more likely to reorder (habitual buyer)
- **Negative γ_u** → user less likely to reorder (explorer/one-time buyer)



#### Product Intercepts (δ_p)

- **Wider spread** than user intercepts — products vary more in reorderability
- **Positive δ_p** → staple products (milk, eggs, bananas) frequently reordered
- **Negative δ_p** → niche or one-time-purchase products
