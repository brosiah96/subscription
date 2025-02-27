# Data Science Techniques for Subscription Services (Using Scikit-Learn)

Subscription-based businesses can leverage data science to **retain customers, personalize experiences, and forecast growth**. In this beginner-friendly guide, we’ll explore four key use cases and show how to implement them with Python’s Scikit-Learn library. Each section provides step-by-step explanations and example code snippets to help you get started:

- **Customer Churn Prediction** – Identify customers likely to unsubscribe using classification models (logistic regression, random forests).
- **Recommendation Systems** – Suggest content or products using collaborative filtering and content-based techniques.
- **Revenue Forecasting** – Use regression models to predict future revenue trends from user behavior data.
- **Customer Segmentation** – Apply clustering (K-Means) to group customers by behavior and preferences.

Let’s dive into each topic and see how data science can improve a subscription service.

## Customer Churn Prediction

**Customer churn** refers to the loss of customers (unsubscribe or cancel) over a given period ([Customer churn prediction in telecom using Python in Deepnote](https://deepnote.com/guides/tutorials/customer-churn-prediction-in-telecom-using-python#:~:text=Customer%20churn%20refers%20to%20the,churn%2C%20enabling%20targeted%20retention%20strategies)). Predicting churn is crucial for subscription services because retaining existing subscribers is often more cost-effective than acquiring new ones ([Customer churn prediction in telecom using Python in Deepnote](https://deepnote.com/guides/tutorials/customer-churn-prediction-in-telecom-using-python#:~:text=Customer%20churn%20refers%20to%20the,churn%2C%20enabling%20targeted%20retention%20strategies)). By analyzing historical user data (e.g. engagement frequency, subscription tenure, support calls), we can train machine learning models to flag which customers are at risk of leaving, allowing the business to take proactive retention measures.

**How to build a churn prediction model:** The task is a binary classification problem (churn or not churn). Typical steps include:

1. **Gather and prepare data** – Assemble historical customer data with features (usage metrics, account age, etc.) and a label indicating whether each customer churned or remained. Clean the data and convert categorical fields to numeric form as needed.
2. **Train/test split** – Divide the data into a training set (to build the model) and a test set (to evaluate performance). This helps ensure the model generalizes to unseen customers.
3. **Choose a model** – Start with simple classifiers like logistic regression, then try more complex ones like decision trees or random forests. Logistic regression outputs a probability of churn (useful for ranking risk), while random forests can capture non-linear patterns for higher accuracy ([Top ML Models for Predicting Customer Churn: A Comparative Analysis | Pecan AI](https://www.pecan.ai/blog/best-ml-models-for-predicting-customer-churn/#:~:text=Known%20for%20its%20simplicity%20and,classification%20problems%20like%20churn%20prediction)) ([Top ML Models for Predicting Customer Churn: A Comparative Analysis | Pecan AI](https://www.pecan.ai/blog/best-ml-models-for-predicting-customer-churn/#:~:text=Random%20forest%20models%20are%20revered,typically%20yields%20high%20prediction%20accuracy)).
4. **Train and evaluate** – Fit the model on the training data and evaluate on the test set using metrics such as accuracy, precision/recall, or AUC. If performance is poor, iterate by adding features, trying different models, or tuning hyperparameters.

**Logistic Regression vs. Random Forest:** Logistic regression is a simple and effective baseline for churn prediction – it predicts the probability of churn and is easy to interpret, showing how each feature contributes to churn risk ([Top ML Models for Predicting Customer Churn: A Comparative Analysis | Pecan AI](https://www.pecan.ai/blog/best-ml-models-for-predicting-customer-churn/#:~:text=Known%20for%20its%20simplicity%20and,classification%20problems%20like%20churn%20prediction)). Random forest is an ensemble of decision trees that often yields higher prediction accuracy by handling non-linear relationships and interactions between features ([Top ML Models for Predicting Customer Churn: A Comparative Analysis | Pecan AI](https://www.pecan.ai/blog/best-ml-models-for-predicting-customer-churn/#:~:text=Random%20forest%20models%20are%20revered,typically%20yields%20high%20prediction%20accuracy)). The trade-off is that random forests are more complex (harder to interpret and computationally heavier) compared to logistic regression ([Top ML Models for Predicting Customer Churn: A Comparative Analysis | Pecan AI](https://www.pecan.ai/blog/best-ml-models-for-predicting-customer-churn/#:~:text=Random%20forest%20models%20are%20revered,typically%20yields%20high%20prediction%20accuracy)). It's common to start with logistic regression for insights, and then use a random forest to capture additional patterns once you understand the data.

Below is a Python example using Scikit-Learn to train a logistic regression and a random forest on a churn dataset. (In practice, you would replace the dummy data with your actual subscription customer data.) We’ll split the data, train both models, and then predict the churn likelihood for a new customer.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Example data: features might be [monthly_visits, support_calls, account_age_months]
X = np.array([
    [10, 1, 12],   # e.g. 10 visits, 1 support call, 12 months tenure
    [5, 3, 8],
    [15, 0, 24],
    [3, 5, 6],
    # ... (more rows for each customer)
])
y = np.array([0, 1, 0, 1])  # churn labels (0 = stayed, 1 = churned) for each customer

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predict churn probability for a new customer using logistic regression
new_customer = [[7, 2, 10]]  # new customer's features
churn_prob = lr_model.predict_proba(new_customer)[0][1]
print(f"Predicted churn probability (Logistic Regression): {churn_prob:.2f}")

# Predict churn class for the same customer using random forest
churn_pred = rf_model.predict(new_customer)[0]
print(f"Predicted churn class (Random Forest): {churn_pred}")
```

In this snippet, the logistic model’s `predict_proba` gives an estimated churn probability (e.g., 0.65 or 65% chance the customer will churn). The random forest’s `predict` yields a binary outcome (1 for churn, 0 for no churn) based on its internal vote. In a real scenario, you would evaluate the models (e.g., compare their accuracy on `X_test, y_test`) and possibly tune them. A high churn probability from the logistic model or a positive churn prediction from the forest would indicate the customer is likely to unsubscribe, prompting a retention action (such as sending a special offer or reaching out to provide support).

## Recommendation Systems

**Recommendation systems** help personalize the user experience by suggesting relevant items (products, movies, articles, etc.) to each user based on data. In subscription services, a good recommender can increase engagement and reduce churn by helping users discover value in the service (for example, a video streaming platform recommending shows a user might like).

There are two common approaches to building recommendations:

- **Content-Based Filtering:** Recommend items similar to those a user already likes, based on item attributes or description. This approach uses information about the items (content) themselves. For example, if a user has watched many action movies, a content-based system will suggest other action movies with similar genre or keywords ([Content Based Filtering and Collaborative Filtering: Difference | Aman Kharwal](https://thecleverprogrammer.com/2023/04/20/content-based-filtering-and-collaborative-filtering-difference/#:~:text=Content,with%20similar%20preferences%20have%20liked)).
- **Collaborative Filtering:** Recommend items that users with similar tastes and behaviors have liked. This approach uses **user behavior data** (e.g. ratings, watch history) rather than item properties ([Content Based Filtering and Collaborative Filtering: Difference | Aman Kharwal](https://thecleverprogrammer.com/2023/04/20/content-based-filtering-and-collaborative-filtering-difference/#:~:text=Aspect%20Content,user%20data%20to%20be%20effective)). For instance, it might recommend a movie that a cluster of users with viewing patterns similar to yours enjoyed, even if the movie is of a genre you haven’t watched before ([Content Based Filtering and Collaborative Filtering: Difference | Aman Kharwal](https://thecleverprogrammer.com/2023/04/20/content-based-filtering-and-collaborative-filtering-difference/#:~:text=On%20the%20other%20hand%2C%20a,interacted%20with%20these%20products%20before)). Collaborative filtering can be **user-based** (find users similar to you and recommend what they liked) or **item-based** (find items often liked together and recommend related items).

Both methods can also be combined in a hybrid system. **Scikit-Learn** does not have a dedicated recommendation module, but we can implement basic recommenders using its tools. For content-based filtering, we can represent items by feature vectors and compute similarities. For collaborative filtering, we can use similarity between user interaction profiles.

*Content-based recommendation with Scikit-Learn:* One simple way is to encode item features and use a nearest-neighbors model to find items with the most similar features. For example, imagine we have movies with genres as features; we can use cosine similarity to find movies closest in genre to a user’s favorites. Below is a small example:

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Toy item feature matrix (e.g., 4 movies with one-hot encoded genres: [Action, Comedy, Drama])
item_features = np.array([
    [1, 0, 1],  # Movie 0: Action & Drama
    [1, 1, 0],  # Movie 1: Action & Comedy
    [0, 1, 1],  # Movie 2: Comedy & Drama
    [1, 0, 0],  # Movie 3: Action only
])

# Fit a NearestNeighbors model using cosine similarity on item feature vectors
recommender = NearestNeighbors(metric='cosine')
recommender.fit(item_features)

# Suppose a user enjoyed Movie 0. Find the 2 most similar movies to Movie 0.
movie_index = 0
distances, indices = recommender.kneighbors([item_features[movie_index]], n_neighbors=3)
print(f"Movies most similar to Movie {movie_index}: {indices[0][1:]}")
```

In this code, each movie is represented by a vector of genres. We use `NearestNeighbors` with cosine distance (cosine similarity) as the metric ([Building a Recommendation System with Scikit-Learn:](https://abhaysinghr.hashnode.dev/building-a-simple-recommendation-system-with-scikit-learn#:~:text=recommendation%20system,the%20similarity%20between%20the%20movies)). For the given `movie_index` (Movie 0), the model finds the closest movies in terms of genre. The output indices (excluding the first one, which is the movie itself) are the recommended movie IDs. For instance, Movie 1 might be recommended for Movie 0 if they share similar genres. In a real system, item features could come from text descriptions (using TF-IDF vectors), tags, or other content attributes. We would also maintain a profile of each user’s liked items to decide which item’s “neighbors” to fetch.

*Collaborative filtering approach:* Instead of item features, use the user-item interaction matrix. For example, create a matrix of users vs. items with ratings or binary preferences. You can compute similarity between users (rows) to find users with similar taste, then recommend items those similar users liked. This can also be done with `NearestNeighbors` on user vectors or by computing a cosine similarity matrix. Another technique is matrix factorization (e.g., using SVD) to discover latent features. Libraries like **Surprise** or **implicit** provide specialized tools for collaborative filtering, but understanding the concept with basic sklearn tools is a good starting point.

**Summary:** Recommendation systems use past data to predict what a user will enjoy next. Content-based filtering relies on item properties, making it good for new users (when we know what content they like) but it may miss trends outside of that profile ([Content Based Filtering and Collaborative Filtering: Difference | Aman Kharwal](https://thecleverprogrammer.com/2023/04/20/content-based-filtering-and-collaborative-filtering-difference/#:~:text=Content,with%20similar%20preferences%20have%20liked)). Collaborative filtering taps into community wisdom (finding look-alike users or items), which can uncover novel recommendations but needs enough user data to work well ([Content Based Filtering and Collaborative Filtering: Difference | Aman Kharwal](https://thecleverprogrammer.com/2023/04/20/content-based-filtering-and-collaborative-filtering-difference/#:~:text=Data%20required%20Information%20about%20the,user%20data%20to%20be%20effective)). By combining these methods, subscription services can keep users engaged with personalized suggestions.

## Revenue Forecasting

Subscription businesses thrive on recurring revenue, so being able to forecast future revenue is highly valuable. **Revenue forecasting** uses historical data to predict future trends in metrics like monthly recurring revenue (MRR), subscriber count, or other key performance indicators. In essence, it’s a form of time series prediction: using previously observed values to estimate future outcomes ([
Time Series Forecasting for Key Subscription Metrics | Recurly
](https://recurly.com/blog/time-series-forecasting-for-key-subscription-metrics/#:~:text=Time%20series%20forecasting%20is%20a,the%20past%2024%20months%E2%80%99%20data)). For example, one might analyze the past 24 months of subscription data to predict the monthly revenue 12 months from now ([
Time Series Forecasting for Key Subscription Metrics | Recurly
](https://recurly.com/blog/time-series-forecasting-for-key-subscription-metrics/#:~:text=Time%20series%20forecasting%20is%20a,the%20past%2024%20months%E2%80%99%20data)). Accurate forecasts help in budgeting, staffing, and strategic decisions by highlighting anticipated growth or downturns ([
Time Series Forecasting for Key Subscription Metrics | Recurly
](https://recurly.com/blog/time-series-forecasting-for-key-subscription-metrics/#:~:text=Time%20series%20forecasting%20can%20be,better%20fit%20for%20your%20business)).

There are many approaches to forecasting. Traditional time-series models (ARIMA, exponential smoothing, etc.) are specialized for sequential data. Here, we’ll demonstrate a simpler **regression-based approach** using Scikit-Learn, which treats forecasting as a supervised learning problem. The idea is to use **historical subscription metrics as features** and **future revenue as the target**. For instance, features for each time period could include the number of active users, number of new sign-ups, cancellations, or even lagged revenue from previous periods.

**Steps for revenue forecasting with regression:**

- **Data Preparation:** Aggregate your data by time period (e.g., monthly). Each record will represent one period (month) with features and the actual revenue in that period as the label. For example, you might have a table where each row is a month and columns include “active_subscribers”, “new_signups”, “cancellations”, and “revenue”.
- **Feature Engineering:** Create features that are predictive of future revenue. Common features are the current subscriber count, growth rates, average revenue per user, seasonal indicators (month of year), or previous period values. In a pure time-series approach, you often use lagged values (e.g., last month’s revenue) as features to predict next month’s revenue.
- **Train/Test Split:** Since time is involved, ensure your training set is the older data and the test set is the more recent data (to simulate forecasting future from past). For example, train on 2020–2023 data, and test on the first months of 2024.
- **Choose and Train Model:** Start with a simple model like **Linear Regression** to capture the overall trend. If the relationship between features and revenue is non-linear, you can try more complex models like decision tree regressors or RandomForestRegressor. Fit the model on the training data.
- **Evaluate and Iterate:** Check the model’s predictions against the actual revenue in the test set. Use metrics like Mean Absolute Error (MAE) or Mean Squared Error (MSE) to quantify accuracy. If the error is large, consider adding more features (or using a more sophisticated time-series model). Once satisfied, use the model to forecast future periods by plugging in expected feature values for those periods.

Below is an example using a simple linear regression on simulated subscription data. We will use features such as active users, new sign-ups, and cancellations to predict monthly revenue:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Simulated historical data for 12 months
months = np.arange(1, 13)
active_users = np.array([100, 111, 135, 171, 196, 218, 233, 269, 302, 318, 354, 386])  # e.g., active subscribers each month
new_signups = np.array([26, 39, 48, 34, 30, 27, 48, 40, 26, 45, 38, 42])              # new subscribers each month
cancellations = np.array([15, 15, 12, 9, 8, 12, 12, 7, 10, 9, 6, 12])                 # cancellations each month
revenue = active_users * 10  + np.random.normal(0, 5, size=12)  # revenue (e.g., $10 per user, with some noise)

# Prepare feature matrix X and target vector y. 
# We'll use active_users, new_signups, cancellations of a month to predict that month's revenue.
X = np.column_stack([active_users, new_signups, cancellations])
y = revenue

# Train a Linear Regression model on first 10 months, reserve last 2 for testing
X_train, X_test = X[:10], X[10:]
y_train, y_test = y[:10], y[10:]
model = LinearRegression()
model.fit(X_train, y_train)

# Forecast revenue for month 11 and 12 using the model
y_pred = model.predict(X_test)
print("Predicted revenues:", y_pred)
print("Actual revenues:   ", y_test)
```

In this toy example, we created a simple growth scenario where revenue is roughly proportional to the number of active users. The linear regression learns this relationship and makes a prediction for months 11 and 12. We print the predicted revenues and compare them to actual values. In a real scenario, after training the model you would use it to predict future months (e.g., month 13 onwards) by supplying the projected features (like expected active users, signups, etc.). 

Keep in mind that time series data often has trends and seasonality. If your subscription metrics have clear patterns (e.g., upticks during holidays or weekends), you may want to include time indicators or use a specialized forecasting approach. Nonetheless, a regression model can serve as a quick baseline to get a sense of future trends. The goal is to anticipate changes in revenue early – for instance, a forecasted plateau or decline in MRR would alert the team to investigate causes (maybe rising churn or market saturation) and take action.

## Customer Segmentation

Not all customers are alike – they have different behaviors and preferences. **Customer segmentation** is the practice of dividing a customer base into groups (segments) that share similar characteristics ([  Customer Segmentation Using K Means Clustering - KDnuggets](https://www.kdnuggets.com/2019/11/customer-segmentation-using-k-means-clustering.html#:~:text=Customer%20Segmentation%20is%20the%20subdivision,uniquely%20appealing%20products%20and%20services)). In a subscription context, segments could be based on how customers use the service, what content they prefer, their spending level, etc. By identifying these segments, companies can tailor marketing strategies and product offerings to each group’s needs, potentially **discovering unmet needs and opportunities** ([  Customer Segmentation Using K Means Clustering - KDnuggets](https://www.kdnuggets.com/2019/11/customer-segmentation-using-k-means-clustering.html#:~:text=Customer%20Segmentation%20is%20the%20subdivision,uniquely%20appealing%20products%20and%20services)). For example, one segment might be “power users” who use the service daily and want premium features, while another segment might be “casual users” who are price-sensitive and need engagement incentives.

One of the most popular algorithms for customer segmentation is **K-Means clustering** ([k means - best algorithms for clustering customers, customer segmentation - Data Science Stack Exchange](https://datascience.stackexchange.com/questions/118812/best-algorithms-for-clustering-customers-customer-segmentation#:~:text=K,can%20be%20good%20at%20times)). K-Means is an unsupervised learning method that groups data points into *K* clusters based on feature similarity. You have to decide on the number of clusters `K` in advance, then the algorithm will iteratively assign each customer to the nearest cluster centroid (the mean of the cluster), and recompute cluster centroids until things stabilize ([k means - best algorithms for clustering customers, customer segmentation - Data Science Stack Exchange](https://datascience.stackexchange.com/questions/118812/best-algorithms-for-clustering-customers-customer-segmentation#:~:text=K,can%20be%20good%20at%20times)) ([K-Means Clustering Explained](https://neptune.ai/blog/k-means-clustering#:~:text=%3E%20%E2%80%9CK,%E2%80%9D%20%E2%80%93%20Source)). The result is that each customer belongs to one of K clusters such that they are more similar to customers in their own cluster than to those in other clusters.

In practice, to use K-Means for segmentation you should: 
1. **Choose segmentation features** – These could be behavioral metrics like average session duration, number of logins per month, total amount spent, preferred content category, etc. It’s important to scale the features (using standardization) especially if they have different units, so that no one feature dominates due to scale.
2. **Determine K (number of segments)** – You might start with an educated guess or use methods like the elbow method (plotting explained variance vs. K) to find a reasonable number of clusters.
3. **Run K-Means** – Use Scikit-Learn’s `KMeans` to fit the model on your customer feature data. This will assign a cluster label to each customer.
4. **Interpret segments** – Analyze the cluster centers (the mean feature values of each cluster) to characterize each segment. For instance, you might find one cluster has high average spending and high usage (your loyal heavy users), and another has low usage and shorter tenure (at-risk users). Give each segment a meaningful name and consider targeted actions for each (e.g., reward the loyal users, re-engage the at-risk group).

 ([K-Means Clustering Explained](https://neptune.ai/blog/k-means-clustering)) *Example of clustering customers into three segments based on two features (Age Group on the vertical axis and Total Spends on the horizontal axis). Each ellipse encloses customers that belong to the same cluster. In this illustrative plot, the clusters might represent distinct groups such as young-low spenders (pink cluster on left), high spenders (blue cluster on right), and low-budget older customers (yellow cluster at bottom). Such segmentation helps businesses tailor marketing and service strategies to each group’s characteristics.*

Using Scikit-Learn, performing K-Means clustering is straightforward. Below is an example where we cluster a small set of customers based on two features: monthly visits and average monthly spend. We’ll use `KMeans` to find 3 clusters and then output the cluster labels and centers:

```python
import numpy as np
from sklearn.cluster import KMeans

# Sample customer data: each row [monthly_visits, avg_monthly_spend]
X = np.array([
    [5,   50],   # Customer 0: 5 visits, $50 spend
    [6,   45],   # Customer 1: 6 visits, $45 spend
    [25, 200],   # Customer 2: 25 visits, $200 spend
    [30, 220],   # Customer 3: 30 visits, $220 spend
    [2,   20],   # Customer 4: 2 visits, $20 spend
    [3,   30],   # Customer 5: 3 visits, $30 spend
])

# Apply K-Means clustering to group customers into 3 segments
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

print("Cluster labels for each customer:", labels)
print("Cluster centers (visits, spend):", centers)
```

In this toy dataset, we have 6 customers. The K-Means algorithm (with K=3) will likely group customers with similar visit and spend patterns. For example, customers 2 and 3 (who visit ~25-30 times and spend ~$200+) might form a “high usage, high spend” cluster. Customers 4 and 5 (few visits, low spend) could form a “low usage” cluster, and customers 0 and 1 might form a moderate cluster. The `labels` array shows the segment index (0, 1, or 2) assigned to each customer. The `cluster_centers_` output gives the average [visits, spend] for each cluster, which you can examine to interpret the segments.

After clustering, you would attach these segment labels back to your customer records. This enables actions like: sending different marketing messages to each segment, customizing product features for a particular segment, or prioritizing feature development for the most valuable segment. Segmentation is an exploratory tool – there’s no “right” answer for how many segments or what they represent, so it often involves some iteration and domain knowledge to decide which groupings are most useful for the business.

## Conclusion

In this guide, we explored how data science techniques can be applied to subscription services using Scikit-Learn. We covered predicting customer churn with classification models to improve retention, building recommendation systems to personalize content, forecasting revenue trends with regression models for better planning, and segmenting customers with clustering to inform strategy. These examples only scratch the surface, but they provide a starting point for beginners. By applying these methods and continuously refining them with real data, subscription-based businesses can enhance customer satisfaction and optimize their growth strategies. Each technique – churn prediction, recommendations, forecasting, and segmentation – offers actionable insights that can help a company stay competitive in the subscription economy.
