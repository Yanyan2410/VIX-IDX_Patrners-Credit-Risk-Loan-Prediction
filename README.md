# VIX-IDX_Patrners-Credit-Risk-Loan-Prediction

Tools: Python, Google Colab, Numpy, Pandas, Scipy, Matplotlib, Seaborn, Scikit-Learn

Data Dictionary: [Lending Club Data Dictionary](https://docs.google.com/spreadsheets/d/1Og_fBlwLbltnhaWC8TTjwUUr3iagMFLQ/edit?usp=sharing&ouid=117420950293102487407&rtpof=true&sd=true)

# Project Background

Credit risk refers to the potential loss that arises when a borrower fails to repay a loan or fulfill their contractual obligations. When evaluating loan applications, credit companies must determine whether to accept or decline based on the applicant's profile. Currently, Lending Club has a 12.29% default rate from 2007-2016. To address this, we aim to increase the amount financed by Lending Club and decrease the number of accepted loans with bad repayment histories. Our approach involves analyzing historical credit application data to gain insights into borrower behavior and developing a predictive model to effectively mitigate credit default risks.

# Exploratory Data Analysis

The dataset contains 466,285 rows, each representing a unique loan application, and comprises 74 features. These features provide valuable information about the loan applicants, including their personal details, financial information, employment history, and more. The dataset is carefully curated to capture a comprehensive view of the borrowers and their creditworthiness.

![Borrower Status Rate](https://github.com/Yanyan2410/VIX-IDX_Patrners-Credit-Risk-Loan-Prediction/blob/main/Images/Borrower's%20Status%20Rate.png)

Key Takeaways:

The data analysis from 2007 to 2016 reveals a significant data imbalance, with 87.71% classified as "good" borrowers and only 12.29% as "bad.

![Loan Status](https://github.com/Yanyan2410/VIX-IDX_Patrners-Credit-Risk-Loan-Prediction/blob/main/Images/Loan%20Status%20Countplot.png)

Key Takeaways:

- The loan status distribution reveals that the majority of loans are either "Current" or "Fully Paid," indicating that most borrowers are meeting their repayment obligations. However, there is a significant number of loans in the "Charged Off" category, suggesting a higher credit risk and potential financial loss for the lender.

- The presence of loans in the "Late (31-120 days)" category signifies delayed payments and a potential warning sign of financial difficulties for borrowers. Additionally, loans categorized as "Charged Off" represent a relatively small percentage but should not be overlooked due to the associated financial loss for the lender

![Loan Purpose](https://github.com/Yanyan2410/VIX-IDX_Patrners-Credit-Risk-Loan-Prediction/blob/main/Images/Number%20of%20Credit%20Purpose.png)

Key Takeaways:

- Debt consolidation is the most common credit purpose. This is likely due to the fact that many people have high-interest credit card debt, and they are looking to consolidate this debt into a single loan with a lower interest rate.

- Credit card debt is the second most common credit purpose. This is likely due to the fact that many people use credit cards for everyday purchases, and they can quickly accumulate debt if they are not careful.
 
![Grade](https://github.com/Yanyan2410/VIX-IDX_Patrners-Credit-Risk-Loan-Prediction/blob/main/Images/Number%20of%20grade.png)

Key Takeaways:

- The distribution of borrower grades follows a typical bell-shaped curve, with a higher concentration of borrowers in the middle grades (B, C, and D) and fewer borrowers in the extreme grades (A, E, F, and G). This suggests that the majority of borrowers are considered to have moderate credit risk.

- Grades A and B represent a relatively lower credit risk than the other grades. These borrowers typically have higher creditworthiness and are more likely to have a good repayment history.
 
![Borrower Status Grade](https://github.com/Yanyan2410/VIX-IDX_Patrners-Credit-Risk-Loan-Prediction/blob/main/Images/Borrower's%20status%20rate%20by%20grade.png)

Key Takeaways:

- Borrowers with a grade of A are the least likely to default on their loans. Only 4.3% of borrowers with a grade of A defaulted on their loans, while 95.7% were good borrowers.

- Borrowers with a grade of G are the most likely to default on their loans. 33.5% of borrowers with a grade of G defaulted on their loans, while only 66.5% were good borrowers.

- There is a clear trend of increasing default rates as the grade decreases. This suggests that borrowers with lower grades are more likely to default on their loans.

- The difference in default rates between grades is significant. For example, the difference in default rates between borrowers with a grade of A and borrowers with a grade of G is 29.2%.

![Borrower Status by Last Credit Pull](https://github.com/Yanyan2410/VIX-IDX_Patrners-Credit-Risk-Loan-Prediction/blob/main/Images/Borrower's%20status%20rate%20by%20last%20credit%20pull%20the%20year..png)

Key Takeaways:

- The percentage of good borrowers has generally remained higher than the percentage of bad borrowers across the years from 2007 to 2016. This indicates that, on average, a higher proportion of borrowers have been classified as good borrowers.

- There is a noticeable improvement in borrower quality from 2007 to 2009, as the percentage of bad borrowers decreases significantly. However, from 2010 onwards, the percentage of bad borrowers shows a slight increase, suggesting a potential deterioration in borrower quality during that period.
  
![Borrower Status by Last Payment](https://github.com/Yanyan2410/VIX-IDX_Patrners-Credit-Risk-Loan-Prediction/blob/main/Images/Borrower's%20status%20rate%20by%20last%20payment%20year.png)

key Takeaways:

- There is a decreasing trend in the percentage of borrowers experiencing issues (bad borrowers) from 2007 to 2016. This indicates an overall improvement in the quality of borrowers over time.

- The percentage of borrowers who successfully make payments (good borrowers) tends to increase each year. This suggests that the number of borrowers effectively managing their repayment obligations is growing over time.

# Data Preprocessing
- Feature Engineering
= Handling Duplicate Value
= Feature Encoding
= Data Transformation
= Outlier Handling

# Modeling

XGBClassifier Model is tested with AUC-ROC and Kolmogorov-Smirnov as Evaluation Targets to find the optimal value. Turns out the optimal value goes to 90% AUC-ROC Score and 64.65% KS Score. In the credit risk modeling practice, AUC above 0.7 and KS above 0.3 are considered good performance.

![Borrower Status by Last Payment](https://github.com/Yanyan2410/VIX-IDX_Patrners-Credit-Risk-Loan-Prediction/blob/main/Images/ROC%20AUC%20Curve.png)
![Borrower Status by Last Payment](https://github.com/Yanyan2410/VIX-IDX_Patrners-Credit-Risk-Loan-Prediction/blob/main/Images/KS.png)

# Business Insights & Recommendation

- Utilize a predictive model to make informed credit decisions, enabling better risk assessment and borrower selection.
- Adjust interest rates dynamically based on the probability of default, ensuring fair and accurate pricing for borrowers.
- Implement strategies to address late payments near the end of loan terms, such as reminders and incentives, to improve timely repayment.
- Employ segmented marketing approaches to offer tailored financing solutions that meet the unique needs of different customer segments.
- Provide comprehensive financial education and support programs to borrowers, empowering them to improve their financial literacy and repayment capabilities.
- Implement proactive collections strategies to minimize default risks and enhance loan recovery outcomes, ensuring a robust and efficient collections process.




